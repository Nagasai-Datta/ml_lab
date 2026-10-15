"""common.py -- standard benchmark (DP vs Monte Carlo vs Q-learning) + plotting helpers."""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rl_core import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
COL = {"Behavior": "#7f7f7f", "Rule": "#555555", "DP": "#0072B2", "Monte Carlo": "#D55E00",
       "Q-learning": "#009E73", "Oracle DP": "#000000"}
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8c564b", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": False})


def odir(idea):
    d = f"{OUT}/idea{idea:02d}"
    os.makedirs(d, exist_ok=True)
    return d


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def save_json(idea, name, obj):
    with open(f"{odir(idea)}/{name}.json", "w") as f:
        json.dump(jsonable(obj), f, indent=1)


def fixed_policy(dom, a):
    return np.full(dom.nS, a, dtype=int)


def standard_benchmark(dom, n_eps, n_oracle, steps, seeds=5, n_eval=300, mask=None,
                       dur_one=False, q_kw=None, mc_kw=None, verbose=True):
    """Compare Behaviour / Rule / DP / MC / Q-learning / Oracle-DP.
    Everything is judged in the TRUE simulator (discounted return)."""
    q_kw, mc_kw = q_kw or {}, mc_kw or {}
    t0 = time.time()
    d_or = rollout(dom, dom.behavior, np.random.default_rng(9999), n_oracle)
    if dur_one: d_or["dur"] = np.ones_like(d_or["dur"])
    m_or = build_mdp(d_or, dom)
    pi_or, _, _ = value_iteration(m_or, mask=mask)
    pi_or = fill_policy(pi_or, m_or, behavior_mode(d_or, dom))
    rule = np.array([dom.rule(s) for s in range(dom.nS)])
    res = {"methods": {}, "curves": {"Monte Carlo": [], "Q-learning": []}, "policies": {},
           "agree": {"Monte Carlo": [], "Q-learning": []}, "optimism_gap": [], "pi_equals_vi": []}

    def record(name, pi, seed_idx):
        r, inf = eval_policy(dom, pi, n_eps=n_eval)
        e = res["methods"].setdefault(name, {"J": [], "Jor": [], "info": {}})
        e["J"].append(float(r.mean())); e["Jor"].append(J(m_or, pi))
        for k, v in inf.items():
            e["info"].setdefault(k, []).append(float(v.mean()))
        e.setdefault("J_ep", []).append(r.tolist())
        if seed_idx == 0:
            res["policies"][name] = np.asarray(pi).tolist()

    record("Rule", rule, 0)
    record("Oracle DP", pi_or, 0)
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        d = rollout(dom, dom.behavior, rng, n_eps)
        if dur_one: d["dur"] = np.ones_like(d["dur"])
        m = build_mdp(d, dom)
        bm = behavior_mode(d, dom)
        pi_dp, V, Q = value_iteration(m, mask=mask)
        pi_pi, _ = policy_iteration(m, mask=mask)
        res["pi_equals_vi"].append(bool(abs(J(m, pi_dp) - J(m, pi_pi)) < 1e-6))
        pi_dp = fill_policy(pi_dp, m, bm)
        cf = (lambda p, m=m, bm=bm: J(m_or, fill_policy(p, m, bm)))
        pi_mc, _, c_mc = mc_control(m, np.random.default_rng(2000 + sd), steps, mask=mask, ckpt_fn=cf, **mc_kw)
        pi_q, _, c_q = q_learning(m, np.random.default_rng(3000 + sd), steps, mask=mask, ckpt_fn=cf, **q_kw)
        res["optimism_gap"].append(float(J(m, pi_dp) - J(m_or, pi_dp)))
        # learning curves are measured on the oracle MDP (cheap, exact)
        res["curves"]["Monte Carlo"].append(c_mc.tolist())
        res["curves"]["Q-learning"].append(c_q.tolist())
        pi_mc, pi_q = fill_policy(pi_mc, m, bm), fill_policy(pi_q, m, bm)
        vis = m.visits >= 5
        for nm, p in (("Monte Carlo", pi_mc), ("Q-learning", pi_q)):
            res["agree"][nm].append(float((p[vis] == pi_dp[vis]).mean()) if vis.any() else 1.0)
        record("Behavior", bm, sd); record("DP", pi_dp, sd)
        record("Monte Carlo", pi_mc, sd); record("Q-learning", pi_q, sd)
    res["J_oracle_model"] = {"Oracle DP": J(m_or, pi_or), "Rule": J(m_or, rule)}
    res["n_states"], res["n_actions"] = dom.nS, dom.nA
    res["runtime_s"] = time.time() - t0
    res["config"] = dict(n_eps=n_eps, n_oracle=n_oracle, steps=steps, seeds=seeds, n_eval=n_eval)
    res["m_or"] = m_or
    if verbose:
        print(dom.name, {k: round(float(np.mean(v["J"])), 2) for k, v in res["methods"].items()},
              f"{res['runtime_s']:.0f}s", flush=True)
    return res


def summarize(res):
    """mean and 95% CI over seeds for each method"""
    out = {}
    for k, v in res["methods"].items():
        out[k] = dict(J=float(np.mean(v["J"])), ci=ci95(v["J"]),
                      info={i: float(np.mean(x)) for i, x in v["info"].items()},
                      info_ci={i: ci95(x) for i, x in v["info"].items()})
    return out


# ------------------------------------------------------------------ plots
def curve_mean(curves):
    arr = np.array(curves)            # seeds x ckpt x 2
    return arr[0, :, 0], arr[:, :, 1].mean(0), arr[:, :, 1].std(0, ddof=1) if len(arr) > 1 else 0 * arr[0, :, 1]


def fig_learning(res, path, title="", ylabel="Policy value on oracle MDP"):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for nm in ("Monte Carlo", "Q-learning"):
        x, mu, sd = curve_mean(res["curves"][nm])
        ax.plot(x, mu, color=COL[nm], label=nm, lw=1.8)
        ax.fill_between(x, mu - sd, mu + sd, color=COL[nm], alpha=0.18)
    ax.axhline(res["J_oracle_model"]["Oracle DP"], color="k", ls="--", lw=1, label="Oracle DP optimum")
    ax.axhline(res["J_oracle_model"]["Rule"], color=COL["Rule"], ls=":", lw=1.2, label="Rule-based policy")
    ax.set_xlabel("Environment steps"); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(path); plt.close(fig)


def fig_bars(res, path, title="", order=("Behavior", "Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP"),
             ylabel="Discounted return in true simulator"):
    """dot plot with 95% CI (avoids truncated-bar misreading)"""
    s = summarize(res)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = np.arange(len(order))
    for x, k in zip(xs, order):
        ax.errorbar(x, s[k]["J"], yerr=s[k]["ci"], fmt="o", color=COL[k], capsize=4, ms=7, lw=1.8)
    ax.set_xticks(xs); ax.set_xticklabels([k.replace(" ", "\n") for k in order])
    ax.set_ylabel(ylabel); ax.set_xlim(-0.6, len(order) - 0.4)
    if title: ax.set_title(title)
    fig.savefig(path); plt.close(fig)


def fig_policy_strip(res, dom, path, methods=("Behavior", "DP", "Monte Carlo", "Q-learning", "Oracle DP"),
                     title=""):
    """actions chosen by each method for every state (visited states only)"""
    vis = res["m_or"].visits > 0
    idx = np.flatnonzero(vis)
    M = np.array([np.asarray(res["policies"][k])[idx] for k in methods])
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(methods) + 1.0))
    im = ax.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0, vmax=dom.nA - 1)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    ax.set_xlabel("State index (states observed in the data)"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, ticks=range(dom.nA), fraction=0.03)
    if dom.action_labels: cb.ax.set_yticklabels(dom.action_labels)
    if title: ax.set_title(title)
    fig.savefig(path); plt.close(fig)


def fig_agreement(res, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    names = ["Monte Carlo", "Q-learning"]
    vals = [np.mean(res["agree"][n]) for n in names]
    errs = [ci95(res["agree"][n]) for n in names]
    ax.bar(names, vals, yerr=errs, color=[COL[n] for n in names], capsize=3, width=0.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Agreement with DP policy\n(states with >=5 visits)")
    fig.savefig(path); plt.close(fig)


def standard_figs(res, dom, idea, tag="", title=""):
    d = odir(idea)
    fig_learning(res, f"{d}/fig_learning{tag}.png", title)
    fig_bars(res, f"{d}/fig_bars{tag}.png", title)
    fig_policy_strip(res, dom, f"{d}/fig_policies{tag}.png", title=title)
    fig_agreement(res, f"{d}/fig_agreement{tag}.png")


def steps_to_fraction(curve_list, target):
    """first checkpoint at which the curve reaches `target` (per seed); nan if never"""
    out = []
    for c in curve_list:
        c = np.array(c)
        hit = np.flatnonzero(c[:, 1] >= target)
        out.append(c[hit[0], 0] if len(hit) else np.nan)
    return np.array(out, float)


def drop_models(res):
    r = dict(res)
    r.pop("m_or", None)
    r["methods"] = {k: {kk: vv for kk, vv in v.items() if kk != "J_ep"} for k, v in res["methods"].items()}
    return r
