"""
rl_core.py -- shared reinforcement-learning library used by all ten papers.

Contents
  * Domain            : base class for a (simulated) medical decision process
  * rollout           : log trajectories under a behaviour policy (the "dataset")
  * build_mdp         : estimate an empirical MDP (counts -> P, R, discount-aware PG)
  * value_iteration / policy_iteration / robust_value_iteration   (dynamic programming)
  * policy_value      : exact policy evaluation by solving (I - P_pi) V = R_pi
  * mc_control        : first-visit Monte Carlo control, epsilon-soft
  * q_learning        : tabular off-policy Q-learning
  * eval_policy       : evaluate a policy in the true simulator

Conventions
  States 0..nS-1, plus an absorbing terminal state with index nS.
  PG[s,a,s'] = P[s,a,s'] * E[gamma^duration]  (duration=1 for ordinary MDPs,
  >1 for the semi-Markov screening problem of paper 5).
"""
import numpy as np


def band(x, edges):
    return int(np.searchsorted(edges, x, side="right"))


# ----------------------------------------------------------------------------
class Domain:
    name = "domain"
    nS = 0
    nA = 0
    gamma = 0.97
    max_steps = 30
    info_keys = []          # per-step floats returned in info, summed per episode
    chan = []               # subset of info_keys used as reward channels
    behavior_eps = 0.25     # epsilon of the epsilon-noisy "clinician" behaviour policy
    default_action = 0
    state_labels = None
    action_labels = None

    def reset(self, rng):
        raise NotImplementedError

    def step(self, a, rng):
        """returns (s_next, reward, done, info_dict, duration)"""
        raise NotImplementedError

    def rule(self, s):
        """deterministic guideline / clinician heuristic"""
        raise NotImplementedError

    def behavior(self, s, rng):
        if rng.random() < self.behavior_eps:
            return int(rng.integers(self.nA))
        return self.rule(s)

    def mask(self):
        """boolean (nS+1, nA) of allowed actions (default: all allowed)"""
        return np.ones((self.nS + 1, self.nA), dtype=bool)


# ----------------------------------------------------------------------------
def rollout(dom, policy, rng, n_eps):
    """policy: callable (s, rng) -> a.  Returns dict of arrays (the logged dataset)."""
    rows = []
    nk = len(dom.info_keys)
    for e in range(n_eps):
        s = dom.reset(rng)
        for t in range(dom.max_steps):
            a = policy(s, rng)
            s2, r, done, info, dur = dom.step(a, rng)
            last = done or t == dom.max_steps - 1
            rows.append((s, a, r, s2, 1.0 if last else 0.0, dur, e, t) +
                        tuple(info.get(k, 0.0) for k in dom.info_keys))
            s = s2
            if last:
                break
    arr = np.array(rows, dtype=float)
    d = dict(s=arr[:, 0].astype(int), a=arr[:, 1].astype(int), r=arr[:, 2],
             s2=arr[:, 3].astype(int), done=arr[:, 4], dur=arr[:, 5],
             ep=arr[:, 6].astype(int), t=arr[:, 7].astype(int))
    for i, k in enumerate(dom.info_keys):
        d[k] = arr[:, 8 + i]
    return d


def concat_data(*ds):
    out = {}
    off = 0
    for d in ds:
        for k, v in d.items():
            if k == "ep":
                v = v + off
            out.setdefault(k, []).append(v)
        off += d["ep"].max() + 1
    return {k: np.concatenate(v) for k, v in out.items()}


# ----------------------------------------------------------------------------
class MDP:
    pass


def build_mdp(data, dom, weights=None, chan=None, min_n=10, term_prior=1.0):
    """Empirical (maximum-likelihood) MDP from logged transitions.
    (s,a) pairs observed fewer than min_n times are completed pessimistically: they go
    to the absorbing terminal state with the worst reward seen in the data."""
    nS, nA = dom.nS, dom.nA
    T = nS
    w = np.ones(len(data["s"])) if weights is None else np.asarray(weights, float)
    s, a = data["s"], data["a"]
    s2 = np.where(data["done"] > 0, T, data["s2"])
    N = np.zeros((nS + 1, nA, nS + 1))
    np.add.at(N, (s, a, s2), w)
    n = N.sum(2)
    Rsum = np.zeros((nS + 1, nA))
    np.add.at(Rsum, (s, a), w * data["r"])
    Gsum = np.zeros((nS + 1, nA, nS + 1))
    np.add.at(Gsum, (s, a, s2), w * dom.gamma ** data["dur"])
    m = MDP()
    m.nS, m.nA, m.gamma = nS, nA, dom.gamma
    m.n = n
    m.N = N
    seen = n >= min_n          # support constraint: thinly observed (s,a) are completed pessimistically
    n_eff = n + term_prior       # one pseudo-observation of absorption guards against 'immortal' loops
    P = np.zeros_like(N)
    P[seen] = N[seen] / n_eff[seen][:, None]
    P[seen, T] += term_prior / n_eff[seen]
    G = np.full_like(N, dom.gamma)
    nz = N > 0
    G[nz] = Gsum[nz] / N[nz]
    R = np.zeros((nS + 1, nA))
    R[seen] = Rsum[seen] / n_eff[seen]
    rmin = float(data["r"].min())
    m.rmin, m.rmax = rmin, float(data["r"].max())
    P[~seen, T] = 1.0
    R[~seen] = rmin
    P[T, :, :] = 0.0
    P[T, :, T] = 1.0
    R[T, :] = 0.0
    m.P, m.G, m.R = P, G, R
    m.PG = P * G
    # reward channels
    m.Rc = {}
    for c in (chan if chan is not None else dom.chan):
        cs = np.zeros((nS + 1, nA))
        np.add.at(cs, (s, a), w * data[c])
        cr = np.zeros((nS + 1, nA))
        cr[seen] = cs[seen] / n_eff[seen]
        m.Rc[c] = cr
    # start distribution = first state of every episode
    first = data["t"] == 0
    mu = np.bincount(data["s"][first], minlength=nS + 1).astype(float)
    m.mu = mu / mu.sum()
    m.visits = (n * seen).sum(1)[:nS]
    m.seen = seen
    m.Rsum = Rsum
    return m


# ----------------------------------------------------------------------------
# dynamic programming
def _q(m, V, R):
    return R + np.einsum("sat,t->sa", m.PG, V)


def value_iteration(m, R=None, mask=None, tol=1e-9, max_iter=20000, penalty=None):
    R = m.R if R is None else R
    if penalty is not None:
        R = R - penalty
    V = np.zeros(m.nS + 1)
    for _ in range(max_iter):
        Q = _q(m, V, R)
        if mask is not None:
            Q = np.where(mask, Q, -1e18)
        Vn = Q.max(1)
        Vn[m.nS] = 0.0
        if np.abs(Vn - V).max() < tol:
            V = Vn
            break
        V = Vn
    Q = _q(m, V, R)
    if mask is not None:
        Q = np.where(mask, Q, -1e18)
    pi = Q.argmax(1)[:m.nS]
    return pi, V, Q


def policy_value(m, pi, R=None):
    """exact V^pi on MDP m"""
    R = m.R if R is None else R
    pi = np.asarray(pi)
    if len(pi) == m.nS:
        pi = np.append(pi, 0)
    idx = np.arange(m.nS + 1)
    Ppi = m.PG[idx, pi]
    Rpi = R[idx, pi]
    V = np.linalg.solve(np.eye(m.nS + 1) - Ppi, Rpi)
    return V


def J(m, pi, R=None):
    V = policy_value(m, pi, R)
    return float(m.mu @ V)


def policy_iteration(m, R=None, mask=None, max_iter=200):
    R = m.R if R is None else R
    pi = np.zeros(m.nS + 1, dtype=int)
    for _ in range(max_iter):
        V = policy_value(m, pi, R)
        Q = _q(m, V, R)
        if mask is not None:
            Q = np.where(mask, Q, -1e18)
        new = Q.argmax(1)
        # keep current action on ties for stability
        tie = np.isclose(Q[np.arange(m.nS + 1), new], Q[np.arange(m.nS + 1), pi])
        new = np.where(tie, pi, new)
        if np.array_equal(new, pi):
            break
        pi = new
    return pi[:m.nS], V


def robust_value_iteration(m, c=1.0, mask=None, tol=1e-8, max_iter=5000):
    """(s,a)-rectangular L1-ball robust DP.  radius rho_sa = min(2, c/sqrt(n_sa)).
    Valid for unit-duration MDPs (PG = gamma * P)."""
    nS1, nA = m.nS + 1, m.nA
    rho = np.minimum(2.0, c / np.sqrt(np.maximum(m.n, 1.0)))
    rho[m.n == 0] = 0.0
    P2 = m.P.reshape(nS1 * nA, nS1)
    t = (rho.reshape(-1) / 2.0)
    V = np.zeros(nS1)
    for _ in range(max_iter):
        order = np.argsort(-V)          # descending value
        low = int(np.argmin(V))
        ps = P2[:, order]
        add = np.minimum(t, 1.0 - P2[:, low])
        cum_prev = np.cumsum(ps, 1) - ps
        rem = np.clip(add[:, None] - cum_prev, 0, None)
        removed = np.minimum(ps, rem)
        ps2 = ps - removed
        Pn = np.empty_like(ps2)
        Pn[:, order] = ps2
        Pn[:, low] += removed.sum(1)
        EV = (Pn @ V).reshape(nS1, nA)
        Q = m.R + m.gamma * EV
        if mask is not None:
            Q = np.where(mask, Q, -1e18)
        Vn = Q.max(1)
        Vn[m.nS] = 0.0
        if np.abs(Vn - V).max() < tol:
            V = Vn
            break
        V = Vn
    return Q.argmax(1)[:m.nS], V, Q


# ----------------------------------------------------------------------------
# model-free learners (sample from the empirical MDP)
class Sampler:
    def __init__(self, m, rng):
        self.m, self.rng = m, rng
        self.cum = np.cumsum(m.P, axis=2)
        self.cmu = np.cumsum(m.mu)
        self.T = m.nS

    def start(self):
        return min(int(np.searchsorted(self.cmu, self.rng.random())), self.m.nS - 1)

    def step(self, s, a):
        s2 = int(np.searchsorted(self.cum[s, a], self.rng.random()))
        if s2 > self.T:
            s2 = self.T
        return s2, self.m.R[s, a], self.m.G[s, a, s2]


def eps_exp(k, total, e0=1.0, e1=0.05):
    return e1 + (e0 - e1) * np.exp(-5.0 * k / total)


def _greedy(Q, s, mask):
    q = Q[s]
    if mask is not None:
        q = np.where(mask[s], q, -1e18)
    return int(q.argmax())


def _explore(rng, s, nA, mask):
    if mask is None:
        return int(rng.integers(nA))
    ok = np.flatnonzero(mask[s])
    return int(ok[rng.integers(len(ok))])


def q_learning(m, rng, steps, eps_fn=eps_exp, alpha0=1.0, alpha_min=0.02,
               q0=0.0, mask=None, n_ckpt=25, max_len=200, ckpt_fn=None):
    nS, nA = m.nS, m.nA
    ckpt_fn = ckpt_fn or (lambda p: J(m, p))
    Q = np.full((nS + 1, nA), q0)
    Q[nS] = 0.0
    cnt = np.zeros((nS + 1, nA))
    smp = Sampler(m, rng)
    ck = np.unique(np.linspace(0, steps, n_ckpt + 1).astype(int))
    curve = []
    s, ep_t = smp.start(), 0
    ci = 0
    for k in range(steps + 1):
        if ci < len(ck) and k == ck[ci]:
            pi = np.array([_greedy(Q, x, mask) for x in range(nS)])
            curve.append((k, ckpt_fn(pi)))
            ci += 1
        if k == steps:
            break
        if rng.random() < eps_fn(k, steps):
            a = _explore(rng, s, nA, mask)
        else:
            a = _greedy(Q, s, mask)
        s2, r, g = smp.step(s, a)
        tgt = r + (0.0 if s2 == nS else g * Q[s2].max() if mask is None
                   else g * np.where(mask[s2], Q[s2], -1e18).max())
        cnt[s, a] += 1
        al = max(alpha_min, alpha0 / (1.0 + cnt[s, a] ** 0.7))
        Q[s, a] += al * (tgt - Q[s, a])
        ep_t += 1
        if s2 == nS or ep_t >= max_len:
            s, ep_t = smp.start(), 0
        else:
            s = s2
    pi = np.array([_greedy(Q, x, mask) for x in range(nS)])
    return pi, Q, np.array(curve)


def mc_control(m, rng, steps, eps_fn=eps_exp, q0=0.0, mask=None, n_ckpt=25,
               max_len=200, ckpt_fn=None):
    """first-visit Monte Carlo control with epsilon-soft behaviour policy and
    incremental-average updates."""
    nS, nA = m.nS, m.nA
    ckpt_fn = ckpt_fn or (lambda p: J(m, p))
    Q = np.full((nS + 1, nA), q0)
    Q[nS] = 0.0
    N = np.zeros((nS + 1, nA))
    smp = Sampler(m, rng)
    ck = np.unique(np.linspace(0, steps, n_ckpt + 1).astype(int))
    curve = []
    used, ci = 0, 0
    while True:
        while ci < len(ck) and used >= ck[ci]:
            pi = np.array([_greedy(Q, x, mask) for x in range(nS)])
            curve.append((ck[ci], ckpt_fn(pi)))
            ci += 1
        if used >= steps:
            break
        s = smp.start()
        traj = []
        for _ in range(max_len):
            if rng.random() < eps_fn(used + len(traj), steps):
                a = _explore(rng, s, nA, mask)
            else:
                a = _greedy(Q, s, mask)
            s2, r, g = smp.step(s, a)
            traj.append((s, a, r, g))
            s = s2
            if s2 == nS:
                break
        used += len(traj)
        # returns, backwards; first-visit
        Gret = 0.0
        rets = []
        for (s_, a_, r_, g_) in reversed(traj):
            Gret = r_ + g_ * Gret
            rets.append(Gret)
        rets.reverse()
        seen = set()
        for (s_, a_, _, _), G_ in zip(traj, rets):
            if (s_, a_) in seen:
                continue
            seen.add((s_, a_))
            N[s_, a_] += 1
            Q[s_, a_] += (G_ - Q[s_, a_]) / N[s_, a_]
    pi = np.array([_greedy(Q, x, mask) for x in range(nS)])
    return pi, Q, np.array(curve)


# ----------------------------------------------------------------------------
def fill_policy(pi, m, fallback):
    """for states never seen in the data use the fallback (behaviour mode) action"""
    pi = np.array(pi, copy=True)
    unseen = m.visits == 0
    pi[unseen] = np.asarray(fallback)[unseen]
    return pi


def behavior_mode(data, dom):
    """most frequent logged action per state (what clinicians actually did)"""
    cnt = np.zeros((dom.nS, dom.nA))
    np.add.at(cnt, (data["s"], data["a"]), 1)
    pi = cnt.argmax(1)
    pi[cnt.sum(1) == 0] = dom.default_action
    return pi


def eval_policy(dom, pi, seed=12345, n_eps=300):
    """discounted return and summed info keys per episode in the TRUE simulator.
    Episode e always starts from rng stream [seed, e] (common random numbers)."""
    rets = np.zeros(n_eps)
    infos = {k: np.zeros(n_eps) for k in dom.info_keys}
    for c in dom.chan:
        infos["d_" + c] = np.zeros(n_eps)      # discounted channel returns
    for e in range(n_eps):
        rng = np.random.default_rng([seed, e])
        s = dom.reset(rng)
        disc = 1.0
        for t in range(dom.max_steps):
            a = int(pi[s])
            s, r, done, info, dur = dom.step(a, rng)
            rets[e] += disc * r
            for c in dom.chan:
                infos["d_" + c][e] += disc * info.get(c, 0.0)
            disc *= dom.gamma ** dur
            for k in dom.info_keys:
                infos[k][e] += info.get(k, 0.0)
            if done:
                break
    return rets, infos


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))
