"""Probe: can simple, standard changes to the model-building step beat the Paper 7 (clinic) learners?
Same data seeds as the paper's benchmark (1000-1004), 150 logged episodes, scored in the true simulator."""
import sys, time, copy
import _paths  # adds topic7/code (our copy of sir's code) to sys.path
import numpy as np
from rl_core import *
from domains import Clinic

dom = Clinic()
N_EVAL = 300


class Spec:
    def __init__(self, nS, nA, gamma):
        self.nS, self.nA, self.gamma, self.chan = nS, nA, gamma, []


def abstract_model(data, dom, min_n=10):
    """drop the weekday: 20 coarse states (no-show band x waiting band)"""
    d = dict(data); d["s"] = data["s"] % 20; d["s2"] = data["s2"] % 20
    return build_mdp(d, Spec(20, dom.nA, dom.gamma), min_n=min_n)


def smooth_to_coarse(data, dom, lam, min_n=10):
    """Empirical-Bayes smoothing of the 100-state model toward the weekday-free model:
    P(.|s,a) = (N(s,a,.) + lam * P_abs(.|c(s),a) mapped to next weekday) / (n + lam)."""
    m = build_mdp(data, dom, min_n=1)
    ma = abstract_model(data, dom, min_n=min_n)
    nS, nA, T = dom.nS, dom.nA, dom.nS
    N, Rsum = m.N, m.Rsum
    n = N.sum(2)
    P = np.zeros_like(N); R = np.zeros_like(Rsum); seen = np.zeros((nS + 1, nA), bool)
    for s in range(nS):
        day, c = divmod(s, 20)
        nd = (day + 1) % 5
        for a in range(nA):
            lam_eff = lam if ma.seen[c, a] else 0.0
            ntot = n[s, a] + lam_eff
            if ntot < min_n:
                continue
            prior = np.zeros(nS + 1)
            prior[nd * 20: nd * 20 + 20] = ma.P[c, a, :20]
            prior[T] = ma.P[c, a, 20]
            P[s, a] = (N[s, a] + lam_eff * prior) / (ntot + 1.0)
            P[s, a, T] += 1.0 / (ntot + 1.0)
            R[s, a] = (Rsum[s, a] + lam_eff * ma.R[c, a]) / (ntot + 1.0)
            seen[s, a] = True
    rmin = float(data["r"].min())
    P[~seen] = 0; P[~seen, T] = 1.0; R[~seen] = rmin
    P[T] = 0; P[T, :, T] = 1.0; R[T] = 0
    mm = copy.copy(m)
    mm.P, mm.R, mm.PG = P, R, P * dom.gamma
    mm.G = np.full_like(P, dom.gamma)
    mm.seen = seen
    mm.visits = (n * seen).sum(1)[:nS]
    return mm


def bootstrap_ensemble(data, dom, K=25, rng=None, min_n=10):
    ids = np.unique(data["ep"])
    Qs = []
    for k in range(K):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([np.flatnonzero(data["ep"] == i) for i in pick])
        d = {kk: v[idx] for kk, v in data.items()}
        # keep 't' so mu is right; episodes re-labelled implicitly
        m = build_mdp(d, dom, min_n=min_n)
        _, _, Q = value_iteration(m)
        Qs.append(Q)
    Q = np.mean(Qs, 0)
    return Q.argmax(1)[:dom.nS]



rows = {}
t0 = time.time()
rule = np.array([dom.rule(s) for s in range(dom.nS)])
r, _ = eval_policy(dom, rule, n_eps=N_EVAL); rows["Rule"] = [r.mean()] * 5
d_or = rollout(dom, dom.behavior, np.random.default_rng(9999), 1500)
m_or = build_mdp(d_or, dom); p_or, _, _ = value_iteration(m_or); p_or = fill_policy(p_or, m_or, behavior_mode(d_or, dom))
r, _ = eval_policy(dom, p_or, n_eps=N_EVAL); rows["Oracle DP (1500 eps)"] = [r.mean()] * 5

for sd in range(5):
    data = rollout(dom, dom.behavior, np.random.default_rng(1000 + sd), 150)
    bm = behavior_mode(data, dom)

    def score(name, pi, m):
        pi = fill_policy(pi, m, bm)
        r, _ = eval_policy(dom, pi, n_eps=N_EVAL)
        rows.setdefault(name, []).append(r.mean())

    for mn in (10, 5, 3, 1):
        m = build_mdp(data, dom, min_n=mn); p, _, _ = value_iteration(m); score(f"DP min_n={mn}", p, m)
    ma = abstract_model(data, dom); pa, _, _ = value_iteration(ma)
    m10 = build_mdp(data, dom)
    score("DP on weekday-free abstraction", np.array([pa[s % 20] for s in range(dom.nS)]), m10)
    for lam in (5, 20, 50):
        m = smooth_to_coarse(data, dom, lam); p, _, _ = value_iteration(m); score(f"DP smoothed->no-weekday lam={lam}", p, m)
    m10 = build_mdp(data, dom)
    p = bootstrap_ensemble(data, dom, K=25, rng=np.random.default_rng(50 + sd)); score("Bootstrap-ensemble DP (K=25)", p, m10)
    print("seed", sd, f"{time.time()-t0:.0f}s", flush=True)

base = np.array(rows["DP min_n=10"])
print(f"\n{'method':40s} {'mean':>8s} {'ci':>6s}  {'vs DP(paper)':>14s}")
for k, v in rows.items():
    v = np.array(v)
    d = v - base
    print(f"{k:40s} {v.mean():8.2f} {ci95(v):6.2f}  {d.mean():+8.2f} ± {ci95(d):.2f}")
