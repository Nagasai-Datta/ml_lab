"""Probe 7c: which features can be pooled? DP on every abstraction that drops one or two of the
three state features (weekday 5, no-show band 4, waiting band 5)."""
import sys, itertools
import _paths  # adds topic7/code (our copy of sir's code) to sys.path
import numpy as np
from rl_core import *
from domains import Clinic

class Spec:
    def __init__(self, nS, nA, gamma):
        self.nS, self.nA, self.gamma, self.chan = nS, nA, gamma, []

dom = Clinic()
SIZES = (5, 4, 5)                      # weekday, no-show band, waiting band  (s = (d*4 + p)*5 + w)
def feats(s): d, rest = divmod(s, 20); p, w = divmod(rest, 5); return (d, p, w)
def code(f, keep):
    c = 0
    for i in keep: c = c * SIZES[i] + f[i]
    return c
combos = [k for r in (3, 2, 1) for k in itertools.combinations(range(3), r)]
names = {0: "weekday", 1: "no-show", 2: "waiting"}
out = {}
for sd in range(5):
    data = rollout(dom, dom.behavior, np.random.default_rng(1000 + sd), 150)
    bm = behavior_mode(data, dom); m = build_mdp(data, dom)
    for keep in combos:
        enc = np.array([code(feats(s), keep) for s in range(dom.nS + 1)]); enc[dom.nS] = 0
        nS = int(np.prod([SIZES[i] for i in keep]))
        d = dict(data); d["s"] = enc[data["s"]]; d["s2"] = enc[np.minimum(data["s2"], dom.nS - 1)]
        ma = build_mdp(d, Spec(nS, dom.nA, dom.gamma))
        pa, _, _ = value_iteration(ma)
        pl = np.array([pa[enc[s]] for s in range(dom.nS)])
        out.setdefault(keep, []).append(eval_policy(dom, fill_policy(pl, m, bm), n_eps=300)[0].mean())
for keep, v in out.items():
    print(f"keep {'+'.join(names[i] for i in keep):28s} states={int(np.prod([SIZES[i] for i in keep])):4d}  J={np.mean(v):7.2f} ± {ci95(v):.2f}")
