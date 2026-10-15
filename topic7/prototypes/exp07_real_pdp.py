"""Prediction + DP lookahead (PDP): the classifier supplies the show distribution of each session;
backward induction over (session index, waiting list) supplies the future value of draining the list.
Compared against the prior-work baselines and the Paper 7 learners in the real-trace clinic."""
import sys, time
import _paths  # adds topic7/code (our copy of sir's code) to sys.path
from real_env import *

GAM, H, WMAX = 0.97, 40, 12
POIS = stats.poisson.pmf(np.arange(0, 15), 1.5); POIS[-1] += 1 - POIS.sum()


def imm(p10, k):
    pe = p10.mean()
    ps = np.concatenate([1 - p10, np.full(k, 1 - pe)])
    dist = np.array([1.0])
    for q in ps: dist = np.convolve(dist, [1 - q, q])
    n = np.arange(len(dist))
    return (dist * (n - 3 * np.maximum(0, n - 10) - 0.5 * np.maximum(0, 10 - n))).sum()


def next_w_dist(w, k):
    base = w - min(w, k); out = np.zeros(WMAX + 1)
    for a, pa in enumerate(POIS): out[min(WMAX, base + a)] += pa
    return out


NW = np.array([[next_w_dist(w, k) for k in range(5)] for w in range(WMAX + 1)])   # (w, k, w')


def build_pdp(n_panels=300, seed=7):
    rng = np.random.default_rng(seed)
    # sample panels per weekday and precompute immediate expected rewards (panel part)
    R = {}
    for d in range(5):
        rows = []
        for _ in range(n_panels):
            p, _ = sess[d][rng.integers(len(sess[d]))]
            p10 = p[rng.permutation(len(p))[:10]]
            rows.append([imm(p10, k) for k in range(5)])
        R[d] = np.array(rows)                     # panels x k
    V = np.zeros((H + 1, WMAX + 1))
    for t in range(H - 1, -1, -1):
        d = t % 5
        cont = GAM * NW @ V[t + 1]                # (w, k)
        for w in range(WMAX + 1):
            q = R[d] + 0.3 * np.minimum(w, np.arange(5)) - 0.4 * w + cont[w]   # panels x k
            V[t, w] = q.max(1).mean()
    return V


def pdp_policy(V):
    def act(dom, s):
        t, w = dom.t, dom.w
        p10 = dom.pp[:10]
        q = [imm(p10, k) + 0.3 * min(w, k) + GAM * NW[w, k] @ V[min(t + 1, H)] for k in range(5)]
        return int(np.argmax(q))
    return act


t0 = time.time()
V = build_pdp()
print(f"PDP value table built {time.time()-t0:.1f}s; V[0, w=0..5] =", V[0, :6].round(1))
dom = RealClinic("pred")
res = {"Rule (paper 7)": evaluate(dom, lambda d_, s: d_.rule(s)),
       "Fixed +2 (best static level)": evaluate(dom, lambda d_, s: 2),
       "Myopic marginal analysis (Topuz-style)": evaluate(dom, myopic),
       "Threshold double-booking p>=0.3 (Amalina-style)": evaluate(dom, threshold(0.3)),
       "OURS: prediction + DP lookahead (PDP)": evaluate(dom, pdp_policy(V))}
for th in (0.2, 0.25, 0.35):
    res[f"Threshold double-booking p>={th}"] = evaluate(dom, threshold(th))
for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
    print(f"{k:52s} {v:8.2f}")
print(f"done {time.time()-t0:.0f}s")
