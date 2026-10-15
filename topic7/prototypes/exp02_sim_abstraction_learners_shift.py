"""Probe 7b: does the weekday-free abstraction also lift Q-learning and MC, and does it hold under the
paper's no-show shifts (+0.08, -0.05)?  Same seeds as the paper benchmark."""
import sys, time
import _paths  # adds topic7/code (our copy of sir's code) to sys.path
import numpy as np
from rl_core import *
from domains import Clinic


class Spec:
    def __init__(self, nS, nA, gamma):
        self.nS, self.nA, self.gamma, self.chan = nS, nA, gamma, []


dom = Clinic()
sims = {"nominal": Clinic(0.0), "+0.08": Clinic(0.08), "-0.05": Clinic(-0.05)}
rule = np.array([dom.rule(s) for s in range(dom.nS)])
res = {}
t0 = time.time()
for k, sim in sims.items():
    res.setdefault(("Rule", k), []).extend([eval_policy(sim, rule, n_eps=300)[0].mean()] * 5)
for sd in range(5):
    data = rollout(dom, dom.behavior, np.random.default_rng(1000 + sd), 150)
    bm = behavior_mode(data, dom)
    m = build_mdp(data, dom)
    d = dict(data); d["s"] = data["s"] % 20; d["s2"] = data["s2"] % 20
    ma = build_mdp(d, Spec(20, dom.nA, dom.gamma))
    lift = lambda p20: np.array([p20[s % 20] for s in range(dom.nS)])
    pols = {}
    p, _, _ = value_iteration(m); pols["DP (paper)"] = fill_policy(p, m, bm)
    p, _, _ = q_learning(m, np.random.default_rng(3000 + sd), 150000); pols["Q-learning (paper)"] = fill_policy(p, m, bm)
    p, _, _ = mc_control(m, np.random.default_rng(2000 + sd), 150000); pols["MC (paper)"] = fill_policy(p, m, bm)
    p, _, _ = value_iteration(ma); pols["DP + abstraction"] = fill_policy(lift(p), m, bm)
    p, _, _ = q_learning(ma, np.random.default_rng(3000 + sd), 150000); pols["Q-learning + abstraction"] = fill_policy(lift(p), m, bm)
    p, _, _ = mc_control(ma, np.random.default_rng(2000 + sd), 150000); pols["MC + abstraction"] = fill_policy(lift(p), m, bm)
    for nm, p in pols.items():
        for k, sim in sims.items():
            res.setdefault((nm, k), []).append(eval_policy(sim, p, n_eps=300)[0].mean())
    print("seed", sd, f"{time.time()-t0:.0f}s", flush=True)

names = ["Rule", "DP (paper)", "Q-learning (paper)", "MC (paper)", "DP + abstraction", "Q-learning + abstraction", "MC + abstraction"]
print(f"\n{'method':28s}" + "".join(f"{k:>18s}" for k in sims))
for nm in names:
    print(f"{nm:28s}" + "".join(f"{np.mean(res[(nm,k)]):10.2f} ± {ci95(res[(nm,k)]):4.2f}" for k in sims))
from scipy import stats
for a, b in (("DP + abstraction", "MC (paper)"), ("Q-learning + abstraction", "Q-learning (paper)"), ("MC + abstraction", "MC (paper)"), ("DP + abstraction", "DP (paper)")):
    dd = np.array(res[(a, "nominal")]) - np.array(res[(b, "nominal")])
    print(f"{a} minus {b}: {dd.mean():+.2f} ± {ci95(dd):.2f}  p={stats.ttest_1samp(dd, 0).pvalue:.4f}")
