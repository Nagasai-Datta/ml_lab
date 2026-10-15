"""Probe 9: same abstraction trick on the rehab domain (drop the week index), dense reward, H = 12 and 24."""
import sys, time
import _paths  # adds topic7/code (our copy of sir's code) to sys.path
import numpy as np
from rl_core import *
from domains import Rehab


class Spec:
    def __init__(self, nS, nA, gamma):
        self.nS, self.nA, self.gamma, self.chan = nS, nA, gamma, []


for H in (12, 24):
    dom = Rehab(sparse=False, H=H)
    rule = np.array([dom.rule(s) for s in range(dom.nS)])
    out = {"Rule": [eval_policy(dom, rule, n_eps=250)[0].mean()] * 5}
    d_or = rollout(dom, dom.behavior, np.random.default_rng(9999), 8000)
    m_or = build_mdp(d_or, dom); p, _, _ = value_iteration(m_or); p = fill_policy(p, m_or, behavior_mode(d_or, dom))
    out["Oracle DP"] = [eval_policy(dom, p, n_eps=250)[0].mean()] * 5
    for sd in range(5):
        data = rollout(dom, dom.behavior, np.random.default_rng(1000 + sd), 800)
        bm = behavior_mode(data, dom); m = build_mdp(data, dom)
        p, _, _ = value_iteration(m); out.setdefault("DP (paper)", []).append(eval_policy(dom, fill_policy(p, m, bm), n_eps=250)[0].mean())
        d = dict(data); d["s"] = data["s"] // H; d["s2"] = data["s2"] // H
        ma = build_mdp(d, Spec(18, dom.nA, dom.gamma))
        pa, _, _ = value_iteration(ma)
        pl = np.array([pa[s // H] for s in range(dom.nS)])
        out.setdefault("DP + drop week index", []).append(eval_policy(dom, fill_policy(pl, m, bm), n_eps=250)[0].mean())
    print(f"H={H}", {k: f"{np.mean(v):.2f}±{ci95(v):.2f}" for k, v in out.items()}, flush=True)
