"""planner.py -- our method: learn only how patients behave, plan everything else with DP.

Sir's learners estimate every transition probability of every (state, action) pair from the logs. With 150
logged days most pairs are seen fewer than 10 times, so the support rule sends them back to the overbooking
rule. Our planner splits the problem in two:
  * learned from data : how patients behave (show-up chance, how the no-show level moves, waiting-list arrivals)
  * known             : the clinic's own rules (10 slots, overtime and idle costs, the waiting list is drained
                        by extra bookings, the reward formula)
The two are combined into an MDP that is solved with sir's value_iteration (rl_core). Every logged session
informs the same few patient numbers, so they are estimated precisely from little data, and every action's
value can be computed even if the logs never tried it.

  clinic_model(data, dom)          comparison A: model for sir's simulator (domains.Clinic), from logs only
  real_model(data, dom, p_extra)   comparison B: model for the real-data clinic; show-up chances come from
                                   the no-show classifier (noshow_data.py)
  lookahead(dom, V, ...)           comparison B: at decision time use each booked patient's own predicted
                                   risk for today's reward and the DP values for the future
  one_session_optimiser(...)       baseline from published work: best booking for today only
  threshold_policy(theta)          baseline from published work: double-book patients above a risk threshold
"""
import numpy as np
from scipy import stats
from rl_core import MDP, band

WAIT_E = [1, 3, 6, 9]


# ----------------------------------------------------------------------------- shared pieces
def session_value(q_reg, k, q_ext, cap=10):
    """expected  shows - 3 x overtime - 0.5 x idle  for one session.
    q_reg: show-up probabilities of the booked patients; k extra patients each showing with prob q_ext.
    Exact (Poisson-binomial) distribution of the number of patients who show."""
    dist = np.array([1.0])
    for q in np.concatenate([np.asarray(q_reg, float), np.full(k, q_ext)]):
        dist = np.convolve(dist, [1 - q, q])
    n = np.arange(len(dist))
    return float((dist * (n - 3 * np.maximum(0, n - cap) - 0.5 * np.maximum(0, cap - n))).sum())


def waiting_list_model(data, wmax=12):
    """from the logged waiting-list lengths: arrival rate per session and which exact lengths sit in each band"""
    ep, a, wait = data["ep"], data["a"], data["wait"].astype(int)
    prev = np.full(len(a), -1)
    same = np.r_[False, ep[1:] == ep[:-1]]
    prev[same] = wait[:-1][same[1:]]
    ok = (prev >= 0) & (wait < wmax)                       # skip the first session and capped lists
    lam = float(np.mean(wait[ok] - (prev[ok] - np.minimum(prev[ok], a[ok]))))
    Pw = np.zeros((5, wmax + 1))
    for w in prev[prev >= 0]:
        Pw[band(w, WAIT_E), w] += 1
    for b in range(5):                                     # bands never seen: spread evenly over their lengths
        if Pw[b].sum() == 0:
            Pw[b, [w for w in range(wmax + 1) if band(w, WAIT_E) == b]] = 1
    return lam, Pw / Pw.sum(1, keepdims=True)


def waiting_step(Pw_row, k, lam, wait_cost, wmax=12):
    """expected waiting-list part of the reward and the distribution of the next waiting band,
    given the distribution of the current exact length (Pw_row) and k extra bookings"""
    pois = stats.poisson.pmf(np.arange(30), lam); pois[-1] += 1 - pois.sum()
    r, nxt = 0.0, np.zeros(5)
    for w, pw in enumerate(Pw_row):
        if pw == 0:
            continue
        dr = min(w, k)
        r += pw * (-wait_cost * w + 0.3 * dr)
        for A, pa in enumerate(pois):
            nxt[band(min(wmax, w - dr + A), WAIT_E)] += pw * pa
    return r, nxt


def _finish(P, R, nS, nA, gamma, mu):
    T = nS
    P[T] = 0.0; P[T, :, T] = 1.0; R[T] = 0.0
    m = MDP()
    m.nS, m.nA, m.gamma = nS, nA, gamma
    m.P, m.R = P, R
    m.G = np.full_like(P, gamma)
    m.PG = P * gamma
    m.mu = mu
    return m


def _start_dist(data, nS):
    mu = np.bincount(data["s"][data["t"] == 0], minlength=nS + 1).astype(float)
    return mu / mu.sum()


# ----------------------------------------------------------------------------- comparison A
def clinic_model(data, dom, wait_cost=0.4):
    """structured model of sir's Clinic (state = weekday x no-show band x waiting band), from the logs only"""
    nS, nA = dom.nS, dom.nA
    d_, rest = np.divmod(data["s"], 20)
    pb_ = rest // 5
    pb2 = (data["s2"] % 20) // 5
    booked = 10 + data["a"]
    q = np.array([data["shows"][pb_ == b].sum() / max(booked[pb_ == b].sum(), 1) for b in range(4)])
    Tp = np.full((5, 4, 4), 0.25)                          # weak prior: a quarter pseudo-count per cell
    np.add.at(Tp, (d_, pb_, pb2), 1.0)
    Tp /= Tp.sum(2, keepdims=True)
    lam, Pw = waiting_list_model(data)
    sess = {(b, k): session_value(np.full(10, q[b]), k, q[b]) for b in range(4) for k in range(nA)}
    R = np.zeros((nS + 1, nA)); P = np.zeros((nS + 1, nA, nS + 1))
    for s in range(nS):
        d, rest = divmod(s, 20)
        pb, wb = divmod(rest, 5)
        nd = (d + 1) % 5
        for k in range(nA):
            rw, wnext = waiting_step(Pw[wb], k, lam, wait_cost)
            R[s, k] = sess[(pb, k)] + rw
            P[s, k, nd * 20: nd * 20 + 20] = np.outer(Tp[d, pb], wnext).ravel()
    m = _finish(P, R, nS, nA, dom.gamma, _start_dist(data, nS))
    m.est = dict(show_up=q.tolist(), arrivals=lam)
    return m


# ----------------------------------------------------------------------------- comparison B
def real_model(data, dom, q_extra):
    """structured model of the real-data clinic (state = slots filled x risk band x waiting band).
    Session part: from the classifier's predicted risk of the patients logged in each (slots, risk) cell.
    Next session: today's session does not change tomorrow's patients, so the next (slots, risk) cell is
    drawn from the logged frequencies; the waiting band follows the waiting-list model."""
    nS, nA = dom.nS, dom.nA
    sb_, rest = np.divmod(data["s"], 20)
    rb_ = rest // 5
    cell = sb_ * 4 + rb_
    n_reg, p_mean = data["booked"] - data["a"], data["p_reg_mean"]
    freq = np.bincount(cell, minlength=12).astype(float) + 0.5
    freq /= freq.sum()
    sess = np.zeros((12, nA))
    for c in range(12):
        idx = np.flatnonzero(cell == c)
        if len(idx) == 0:                                  # unseen cell: use every logged session
            idx = np.arange(len(cell))
        idx = idx[:400]
        for k in range(nA):
            sess[c, k] = np.mean([session_value(np.full(int(n_reg[i]), 1 - p_mean[i]), k, q_extra) for i in idx])
    lam, Pw = waiting_list_model(data)
    R = np.zeros((nS + 1, nA)); P = np.zeros((nS + 1, nA, nS + 1))
    for s in range(nS):
        c, wb = divmod(s, 5)
        for k in range(nA):
            rw, wnext = waiting_step(Pw[wb], k, lam, dom.wait_cost)
            R[s, k] = sess[c, k] + rw
            P[s, k, :nS] = np.outer(freq, wnext).ravel()
    m = _finish(P, R, nS, nA, dom.gamma, _start_dist(data, nS))
    m.est = dict(arrivals=lam, q_extra=q_extra, cell_freq=freq.tolist())
    m.lam, m.Pw, m.freq = lam, Pw, freq
    return m


def lookahead(m, V, q_extra, wait_cost):
    """decide with today's actual patients: their own predicted show-up chances for today's reward,
    the DP value V of the next state for everything after today"""
    pois = stats.poisson.pmf(np.arange(30), m.lam); pois[-1] += 1 - pois.sum()

    def act(dom, s):
        q_reg = 1 - dom.p_reg
        best, bk = -1e18, 0
        for k in range(dom.nA):
            dr = min(dom.w, k)
            nxt = np.zeros(5)
            for A, pa in enumerate(pois):
                nxt[band(min(12, dom.w - dr + A), WAIT_E)] += pa
            future = dom.gamma * (np.outer(m.freq, nxt).ravel() @ V[:dom.nS])
            val = session_value(q_reg, k, q_extra) + 0.3 * dr + future
            if val > best:
                best, bk = val, k
        return bk
    return act


# ----------------------------------------------------------------------------- baselines from published work
def one_session_optimiser(q_extra):
    """best number of extra bookings for today's session alone, from each patient's predicted risk
    (marginal analysis as in Topuz et al. 2024; static Poisson-binomial optimiser as in public code on this dataset)"""
    def act(dom, s):
        q_reg = 1 - dom.p_reg
        return int(np.argmax([session_value(q_reg, k, q_extra) + 0.3 * min(dom.w, k) for k in range(dom.nA)]))
    return act


def threshold_policy(theta):
    """fill empty regular slots, then double-book every booked patient whose predicted no-show risk >= theta
    (the fixed-threshold baselines of Amalina and An 2026)"""
    def act(dom, s):
        return int(min(dom.nA - 1, (dom.CAP - dom.n_reg) + int((dom.p_reg >= theta).sum())))
    return act
