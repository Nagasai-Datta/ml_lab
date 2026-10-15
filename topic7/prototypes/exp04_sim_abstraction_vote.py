"""Probe 7d: ensemble over abstraction levels (no simulator needed to pick one).
(a) majority vote of the 7 abstraction-DP policies; (b) average of their Q tables; (c) vote of the 2-feature ones."""
import sys, itertools
import _paths  # adds topic7/code (our copy of sir's code) to sys.path
import numpy as np
from rl_core import *
from domains import Clinic
from scipy import stats

class Spec:
    def __init__(self, nS, nA, gamma):
        self.nS, self.nA, self.gamma, self.chan = nS, nA, gamma, []

dom = Clinic()
SIZES = (5, 4, 5)
def feats(s): d, rest = divmod(s, 20); p, w = divmod(rest, 5); return (d, p, w)
def code(f, keep):
    c = 0
    for i in keep: c = c * SIZES[i] + f[i]
    return c
combos = [k for r in (3, 2, 1) for k in itertools.combinations(range(3), r)]
out = {}
for sd in range(5):
    data = rollout(dom, dom.behavior, np.random.default_rng(1000 + sd), 150)
    bm = behavior_mode(data, dom); m = build_mdp(data, dom)
    P, Q = {}, {}
    for keep in combos:
        enc = np.array([code(feats(s), keep) for s in range(dom.nS)])
        nS = int(np.prod([SIZES[i] for i in keep]))
        d = dict(data); d["s"] = enc[data["s"]]; d["s2"] = enc[np.minimum(data["s2"], dom.nS - 1)]
        ma = build_mdp(d, Spec(nS, dom.nA, dom.gamma))
        pa, _, Qa = value_iteration(ma)
        P[keep] = np.array([pa[enc[s]] for s in range(dom.nS)])
        Q[keep] = np.array([Qa[enc[s]] for s in range(dom.nS)])
    def vote(keys):
        V = np.zeros((dom.nS, dom.nA))
        for k in keys: V[np.arange(dom.nS), P[k]] += 1
        return V.argmax(1)
    cands = {"full model (paper DP)": P[(0, 1, 2)],
             "vote, all 7 abstractions": vote(combos),
             "vote, 3 two-feature abstractions": vote([k for k in combos if len(k) == 2]),
             "Q-average, all 7": np.mean([Q[k] for k in combos], 0).argmax(1),
             "Q-average, two-feature": np.mean([Q[k] for k in combos if len(k) == 2], 0).argmax(1)}
    for nm, p in cands.items():
        out.setdefault(nm, []).append(eval_policy(dom, fill_policy(p, m, bm), n_eps=300)[0].mean())
base = np.array(out["full model (paper DP)"])
for nm, v in out.items():
    dd = np.array(v) - base
    pv = stats.ttest_1samp(dd, 0).pvalue if dd.std() > 0 else 1
    print(f"{nm:36s} J={np.mean(v):7.2f} ± {ci95(v):.2f}   vs paper DP {dd.mean():+6.2f} (p={pv:.4f})")
