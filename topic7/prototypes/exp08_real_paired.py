"""Paired test with more episodes: PDP vs the strongest baselines (common random numbers)."""
import sys
import _paths  # adds topic7/code (our copy of sir's code) to sys.path
import exp07_real_pdp as P
from real_env import *

def ep_returns(dom, act, n=1500, seed=777):
    out = np.zeros(n)
    for e in range(n):
        rng = np.random.default_rng([seed, e]); s = dom.reset(rng); disc = 1.0
        for _ in range(dom.max_steps):
            s, r, done, _, _ = dom.step(int(act(dom, s)), rng); out[e] += disc * r; disc *= dom.gamma
            if done: break
    return out

dom = RealClinic("pred")
pdp = ep_returns(dom, P.pdp_policy(P.V))
for nm, act in (("Myopic marginal analysis", myopic), ("Fixed +2", lambda d_, s: 2), ("Rule (paper 7)", lambda d_, s: d_.rule(s))):
    b = ep_returns(dom, act); d = pdp - b
    print(f"PDP minus {nm:26s} {d.mean():+6.2f} ± {ci95(d):.2f}  p={stats.ttest_1samp(d, 0).pvalue:.2g}   (PDP {pdp.mean():.2f}, base {b.mean():.2f})")
