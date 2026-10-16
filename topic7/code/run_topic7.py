"""run_topic7.py -- every experiment for our Topic 7 results (Paper 7 extension).

    python topic7/code/run_topic7.py            full run (writes topic7/code/out/topic7/)
    python topic7/code/run_topic7.py --quick    smoke test, a few seeds (writes topic7/code/out/topic7_quick/)

Comparison A  sir's own simulator (domains.Clinic), his data seeds (1000+), his scoring (eval_policy, 300 episodes)
    A1  sir's rule, DP, Monte Carlo, Q-learning, oracle  vs  ours (structured model + DP), and Monte Carlo /
        Q-learning run on our model; utilization, overtime, idle, waiting list; paired tests over seeds
    A2  convergence: true return of the greedy policy during Monte Carlo and Q-learning, on sir's model and ours
    A3  data size: sir's DP vs ours from 25 to 1,500 logged clinic days
    A4  no-show shift: every policy re-scored with no-show rates +0.08 and -0.05 (sir's shift test)
Comparison B  real Kaggle sessions (real_clinic.RealClinic), state = slots filled x predicted risk band x waiting band
    B1  rule, published-style baselines (fixed overbooking, threshold double-booking, one-session optimiser),
        sir's DP / Monte Carlo / Q-learning, oracle  vs  ours; logs from 16-25 May, scored on 30 May - 8 Jun
    B2  convergence of Monte Carlo and Q-learning, as A2
    B3  busier clinic: waiting cost 0.4 / 1.0 and 1.5 / 2.5 new patients per session
Sir's code is reused unchanged: rl_core (rollout, build_mdp, value_iteration, mc_control, q_learning,
eval_policy, fill_policy, behavior_mode), domains.Clinic, ideas.paired, common (plot style, save_json).
"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

from rl_core import (rollout, build_mdp, value_iteration, q_learning, mc_control, eval_policy, fill_policy,
                     behavior_mode, ci95)
from domains import Clinic
from common import OUT, COL, save_json, jsonable
from ideas import paired
import noshow_data as nd
from real_clinic import RealClinic, risk_edges
import planner as pl

QUICK = "--quick" in sys.argv
CFG = dict(seeds=3 if QUICK else 20, n_logged=150, n_oracle=1500, steps=30000 if QUICK else 150000,
           n_eval_A=300, n_eval_B=200 if QUICK else 1000, ckpt_eps=30 if QUICK else 60,
           sizes=[50, 150] if QUICK else [25, 50, 150, 500, 1500], size_seeds=2 if QUICK else 10,
           demand=[(0.4, 1.5), (1.0, 1.5), (0.4, 2.5), (1.0, 2.5)], demand_seeds=2 if QUICK else 10,
           n_eval_demand=100 if QUICK else 500, split="2016-05-30")
D = os.path.join(OUT, "topic7_quick" if QUICK else "topic7")
os.makedirs(D, exist_ok=True)
OURS = "#CC79A7"          # our method; sir's colours (COL) are kept for his methods
BASE = "#E69F00"          # baselines from published work
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def evaluate(dom, act, n, seed=12345):
    """discounted return per episode and per-episode totals of every info key, for a policy given as a function
    act(dom, s) -> k (needed for policies that look at today's patients); common random numbers like eval_policy"""
    rets = np.zeros(n)
    infos = {k: np.zeros(n) for k in dom.info_keys}
    for e in range(n):
        rng = np.random.default_rng([seed, e]); s = dom.reset(rng); disc = 1.0
        for _ in range(dom.max_steps):
            s, r, done, info, _ = dom.step(int(act(dom, s)), rng)
            rets[e] += disc * r; disc *= dom.gamma
            for k in dom.info_keys:
                infos[k][e] += info.get(k, 0.0)
            if done:
                break
    return rets, infos


def table(pi):
    return lambda dom, s: pi[s]


def metrics(rets, infos, sessions):
    """the problem statement's measures, per session: utilization of the 10 regular slots, overtime, idle slots,
    waiting list; plus patients seen per episode"""
    return dict(J=float(rets.mean()), utilization=float(1 - infos["idle"].mean() / (10 * sessions)),
                overtime=float(infos["overtime"].mean() / sessions), idle=float(infos["idle"].mean() / sessions),
                waiting=float(infos["wait"].mean() / sessions), shows=float(infos["shows"].mean()))


class Rec:
    """collects per-seed results for each method"""
    def __init__(self):
        self.m = {}

    def add(self, name, rets, infos, sessions, ep=None):
        d = self.m.setdefault(name, {"J": [], "utilization": [], "overtime": [], "idle": [], "waiting": [], "shows": [], "ep": []})
        for k, v in metrics(rets, infos, sessions).items():
            d[k].append(v)
        d["ep"].append(rets)

    def summary(self):
        out = {}
        for k, d in self.m.items():
            out[k] = {m: dict(mean=float(np.mean(d[m])), ci=ci95(d[m]), per_seed=d[m])
                      for m in ("J", "utilization", "overtime", "idle", "waiting", "shows")}
        return out

    def J(self, name):
        return np.array(self.m[name]["J"])

    def ep_mean(self, name):
        return np.mean(self.m[name]["ep"], 0)


# ============================================================================= comparison A
def comparison_A():
    dom = Clinic()
    n, A = CFG["n_eval_A"], {}
    rec = Rec()
    shifts = {"nominal": dom, "no-show +0.08": Clinic(0.08), "no-show -0.05": Clinic(-0.05)}
    shift = {k: {} for k in shifts}
    curves = {k: [] for k in ("Monte Carlo|sir", "Q-learning|sir", "Monte Carlo|ours", "Q-learning|ours")}
    est = []

    def score(name, pi):
        r, i = eval_policy(dom, pi, n_eps=n)
        rec.add(name, r, i, dom.max_steps)
        for k, sim in shifts.items():
            if k != "nominal":
                shift[k].setdefault(name, []).append(float(eval_policy(sim, pi, n_eps=n)[0].mean()))
        shift["nominal"].setdefault(name, []).append(float(r.mean()))

    true_J = lambda pi: float(eval_policy(dom, pi, n_eps=CFG["ckpt_eps"], seed=777)[0].mean())
    rule = np.array([dom.rule(s) for s in range(dom.nS)])
    d_or = rollout(dom, dom.behavior, np.random.default_rng(9999), CFG["n_oracle"])
    m_or = build_mdp(d_or, dom)
    p_or = fill_policy(value_iteration(m_or)[0], m_or, behavior_mode(d_or, dom))
    for sd in range(CFG["seeds"]):
        data = rollout(dom, dom.behavior, np.random.default_rng(1000 + sd), CFG["n_logged"])
        m = build_mdp(data, dom); bm = behavior_mode(data, dom)
        cf = lambda p, m=m, bm=bm: true_J(fill_policy(p, m, bm))
        score("Rule", rule); score("Oracle DP (10x data)", p_or)
        score("Sir: DP", fill_policy(value_iteration(m)[0], m, bm))
        p, _, c = mc_control(m, np.random.default_rng(2000 + sd), CFG["steps"], ckpt_fn=cf)
        score("Sir: Monte Carlo", fill_policy(p, m, bm)); curves["Monte Carlo|sir"].append(c.tolist())
        p, _, c = q_learning(m, np.random.default_rng(3000 + sd), CFG["steps"], ckpt_fn=cf)
        score("Sir: Q-learning", fill_policy(p, m, bm)); curves["Q-learning|sir"].append(c.tolist())
        ms = pl.clinic_model(data, dom); est.append(ms.est)
        score("Ours: DP", value_iteration(ms)[0])
        p, _, c = mc_control(ms, np.random.default_rng(2000 + sd), CFG["steps"], ckpt_fn=true_J)
        score("Ours: Monte Carlo", p); curves["Monte Carlo|ours"].append(c.tolist())
        p, _, c = q_learning(ms, np.random.default_rng(3000 + sd), CFG["steps"], ckpt_fn=true_J)
        score("Ours: Q-learning", p); curves["Q-learning|ours"].append(c.tolist())
        log(f"A seed {sd}: sir DP {rec.J('Sir: DP')[-1]:.2f}  ours {rec.J('Ours: DP')[-1]:.2f}")
    A["methods"] = rec.summary()
    A["paired"] = {f"Ours: DP vs {b}": paired(rec.J("Ours: DP"), rec.J(b))
                   for b in ("Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning", "Rule", "Oracle DP (10x data)")}
    A["paired"]["Ours: Q-learning vs Sir: Q-learning"] = paired(rec.J("Ours: Q-learning"), rec.J("Sir: Q-learning"))
    A["paired"]["Ours: Monte Carlo vs Sir: Monte Carlo"] = paired(rec.J("Ours: Monte Carlo"), rec.J("Sir: Monte Carlo"))
    A["shift"] = {k: {nm: dict(mean=float(np.mean(v)), ci=ci95(v)) for nm, v in d.items()} for k, d in shift.items()}
    A["curves"] = curves
    A["estimates"] = est

    # A3: data size
    sizes = {}
    for nlog in CFG["sizes"]:
        sj, oj = [], []
        for sd in range(CFG["size_seeds"]):
            data = rollout(dom, dom.behavior, np.random.default_rng(1000 + sd), nlog)
            m = build_mdp(data, dom); bm = behavior_mode(data, dom)
            sj.append(float(eval_policy(dom, fill_policy(value_iteration(m)[0], m, bm), n_eps=n)[0].mean()))
            oj.append(float(eval_policy(dom, value_iteration(pl.clinic_model(data, dom))[0], n_eps=n)[0].mean()))
        sizes[str(nlog)] = {"Sir: DP": dict(mean=float(np.mean(sj)), ci=ci95(sj)),
                            "Ours: DP": dict(mean=float(np.mean(oj)), ci=ci95(oj)), "diff": paired(oj, sj)}
        log(f"A size {nlog}: sir DP {np.mean(sj):.2f}  ours {np.mean(oj):.2f}")
    A["data_size"] = sizes
    return A


# ============================================================================= comparison B
def build_real(lam=1.5, wait_cost=0.4, cache={}):
    if "test" not in cache:
        df = nd.load()
        test, info = nd.fit_predictor(df)
        past, fut = test[test.AD < CFG["split"]], test[test.AD >= CFG["split"]]
        sp, pp = nd.build_sessions(past); sf, pf = nd.build_sessions(fut)
        cache.update(df=df, test=test, info=info, past=past, fut=fut, sp=sp, pp=pp, sf=sf, pf=pf, RE=risk_edges(sp),
                     q_extra=float(1 - past.p.mean()))
    c = cache
    return (RealClinic(c["sp"], c["pp"], c["RE"], lam, wait_cost), RealClinic(c["sf"], c["pf"], c["RE"], lam, wait_cost), c)


def comparison_B():
    Dp, Df, c = build_real()
    n, B, rec = CFG["n_eval_B"], {}, Rec()
    q_extra = c["q_extra"]
    sizes_p = [len(p) for d in c["sp"] for p, _ in c["sp"][d]]
    B["data"] = dict(summary=nd.dataset_summary(c["df"]), classifier=c["info"], risk_edges=c["RE"].tolist(), q_extra=q_extra,
                     sessions_past=int(sum(len(v) for v in c["sp"].values())), sessions_future=int(sum(len(v) for v in c["sf"].values())),
                     share_not_full=float(np.mean(np.array(sizes_p) < 10)),
                     noshow_past=float(c["past"].y.mean()), noshow_future=float(c["fut"].y.mean()))
    log(f"B data ready: AUC {c['info']['auc_test']:.3f}")

    def score(name, act, seeds=1):
        r, i = evaluate(Df, act, n)
        for _ in range(seeds):
            rec.add(name, r, i, Df.max_steps)

    S = CFG["seeds"]
    score("Rule", table([Df.rule(s) for s in range(Df.nS)]), S)
    score("One-session optimiser", pl.one_session_optimiser(q_extra), S)
    thr = {}
    for th in (0.2, 0.25, 0.3, 0.35):
        thr[th] = float(evaluate(Df, pl.threshold_policy(th), n)[0].mean())
    best_th = max(thr, key=thr.get)
    score(f"Threshold double-booking (p >= {best_th})", pl.threshold_policy(best_th), S)
    fixed = {k: float(evaluate(Df, lambda dom, s, k=k: k, n)[0].mean()) for k in range(5)}
    best_k = max(fixed, key=fixed.get)
    score(f"Fixed overbooking (+{best_k})", lambda dom, s: best_k, S)
    d_or = rollout(Dp, Dp.behavior, np.random.default_rng(9999), CFG["n_oracle"])
    m_or = build_mdp(d_or, Dp)
    score("Oracle DP (10x data)", table(fill_policy(value_iteration(m_or)[0], m_or, behavior_mode(d_or, Dp))), S)
    B["baseline_grid"] = dict(threshold={str(k): v for k, v in thr.items()}, fixed={str(k): v for k, v in fixed.items()})
    log("B baselines scored")
    true_J = lambda pi: float(evaluate(Df, table(pi), CFG["ckpt_eps"], seed=777)[0].mean())
    curves = {k: [] for k in ("Monte Carlo|sir", "Q-learning|sir", "Monte Carlo|ours", "Q-learning|ours")}
    est = []
    for sd in range(S):
        data = rollout(Dp, Dp.behavior, np.random.default_rng(1000 + sd), CFG["n_logged"])
        m = build_mdp(data, Dp); bm = behavior_mode(data, Dp)
        cf = lambda p, m=m, bm=bm: true_J(fill_policy(p, m, bm))
        score("Sir: DP", table(fill_policy(value_iteration(m)[0], m, bm)))
        p, _, cv = mc_control(m, np.random.default_rng(2000 + sd), CFG["steps"], ckpt_fn=cf)
        score("Sir: Monte Carlo", table(fill_policy(p, m, bm))); curves["Monte Carlo|sir"].append(cv.tolist())
        p, _, cv = q_learning(m, np.random.default_rng(3000 + sd), CFG["steps"], ckpt_fn=cf)
        score("Sir: Q-learning", table(fill_policy(p, m, bm))); curves["Q-learning|sir"].append(cv.tolist())
        ms = pl.real_model(data, Dp, q_extra); est.append(ms.est["arrivals"])
        score("Ours: DP", table(value_iteration(ms)[0]))
        p, _, cv = mc_control(ms, np.random.default_rng(2000 + sd), CFG["steps"], ckpt_fn=true_J)
        score("Ours: Monte Carlo", table(p)); curves["Monte Carlo|ours"].append(cv.tolist())
        p, _, cv = q_learning(ms, np.random.default_rng(3000 + sd), CFG["steps"], ckpt_fn=true_J)
        score("Ours: Q-learning", table(p)); curves["Q-learning|ours"].append(cv.tolist())
        log(f"B seed {sd}: sir DP {rec.J('Sir: DP')[-1]:.2f}  ours {rec.J('Ours: DP')[-1]:.2f}")
    B["methods"] = rec.summary()
    th_name, fx_name = f"Threshold double-booking (p >= {best_th})", f"Fixed overbooking (+{best_k})"
    B["paired_seeds"] = {f"Ours: DP vs {b}": paired(rec.J("Ours: DP"), rec.J(b))
                         for b in ("Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning", "Rule", "Oracle DP (10x data)",
                                   "One-session optimiser", th_name, fx_name)}
    B["paired_episodes"] = {f"Ours: DP vs {b}": paired(rec.ep_mean("Ours: DP"), rec.ep_mean(b))
                            for b in ("One-session optimiser", th_name, fx_name, "Rule", "Sir: DP")}
    B["curves"] = curves
    B["arrival_estimates"] = est

    # B3: busier clinic
    dem = {}
    for wc, lam in CFG["demand"]:
        Dp2, Df2, _ = build_real(lam, wc)
        nn = CFG["n_eval_demand"]
        one = float(evaluate(Df2, pl.one_session_optimiser(q_extra), nn)[0].mean())
        fx = {k: float(evaluate(Df2, lambda dom, s, k=k: k, nn)[0].mean()) for k in range(5)}
        sj, oj = [], []
        for sd in range(CFG["demand_seeds"]):
            data = rollout(Dp2, Dp2.behavior, np.random.default_rng(1000 + sd), CFG["n_logged"])
            m = build_mdp(data, Dp2); bm = behavior_mode(data, Dp2)
            sj.append(float(evaluate(Df2, table(fill_policy(value_iteration(m)[0], m, bm)), nn)[0].mean()))
            oj.append(float(evaluate(Df2, table(value_iteration(pl.real_model(data, Dp2, q_extra))[0]), nn)[0].mean()))
        key = f"wait cost {wc}, arrivals {lam}"
        dem[key] = {"Ours: DP": dict(mean=float(np.mean(oj)), ci=ci95(oj)), "Sir: DP": dict(mean=float(np.mean(sj)), ci=ci95(sj)),
                    "One-session optimiser": one, "Best fixed overbooking": max(fx.values()),
                    "gain_vs_one_session_pct": float(100 * (np.mean(oj) - one) / abs(one))}
        log(f"B demand {key}: ours {np.mean(oj):.2f}  one-session {one:.2f}  sir DP {np.mean(sj):.2f}")
    B["demand"] = dem
    return B


# ============================================================================= figures
def dot_plot(summary, order, colors, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(order) + 1.0))
    for i, nm in enumerate(order):
        s = summary[nm]["J"]
        ax.errorbar(s["mean"], i, xerr=s["ci"], fmt="D" if nm.startswith("Ours") else "o", color=colors[nm],
                    capsize=3, ms=7, lw=1.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
    ax.set_xlabel("Discounted return (higher is better; 95% CI over seeds)")
    ax.set_title(title)
    fig.savefig(path); plt.close(fig)


def colors_for(names):
    c = {}
    for nm in names:
        if nm.startswith("Ours"): c[nm] = OURS
        elif nm == "Rule": c[nm] = COL["Rule"]
        elif nm.startswith("Oracle"): c[nm] = COL["Oracle DP"]
        elif nm.startswith("Sir: "): c[nm] = COL[nm[5:]]
        else: c[nm] = BASE
    return c


def curve_plot(curves, ref, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, ls in (("Monte Carlo|sir", "-"), ("Q-learning|sir", "-"), ("Monte Carlo|ours", "--"), ("Q-learning|ours", "--")):
        alg, who = key.split("|")
        arr = np.array(curves[key]); x = arr[0, :, 0]; mu = arr[:, :, 1].mean(0)
        sd = arr[:, :, 1].std(0, ddof=1) if len(arr) > 1 else 0 * mu
        ax.plot(x, mu, ls=ls, color=COL[alg], lw=1.8, label=f"{alg}, {'our model' if who == 'ours' else 'sir’s model'}")
        ax.fill_between(x, mu - sd, mu + sd, color=COL[alg], alpha=0.12)
    for nm, (v, ls) in ref.items():
        ax.axhline(v, color=OURS if nm.startswith("Ours") else COL["Rule"], ls=ls, lw=1.1, label=nm)
    ax.set_xlabel("Learning steps"); ax.set_ylabel("True return of the greedy policy")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    fig.savefig(path); plt.close(fig)


def metric_panels(summary, order, path, title):
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 0.36 * len(order) + 1.2), sharey=True)
    colors = colors_for(order)
    for ax, (key, lab, scale) in zip(axs, (("utilization", "Slot utilization (%)", 100), ("overtime", "Overtime patients per session", 1),
                                           ("waiting", "Waiting list per session", 1))):
        for i, nm in enumerate(order):
            s = summary[nm][key]
            ax.errorbar(s["mean"] * scale, i, xerr=s["ci"] * scale, fmt="D" if nm.startswith("Ours") else "o",
                        color=colors[nm], capsize=3, ms=6, lw=1.4)
        ax.set_xlabel(lab)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
    fig.suptitle(title, fontsize=9); fig.savefig(path); plt.close(fig)


def figures(A, B):
    oA = ["Rule", "Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning", "Oracle DP (10x data)", "Ours: Monte Carlo", "Ours: Q-learning", "Ours: DP"]
    dot_plot(A["methods"], oA, colors_for(oA), f"{D}/fig_A_returns.png", "A. Sir’s simulator: return by method")
    metric_panels(A["methods"], oA, f"{D}/fig_A_metrics.png", "A. Sir’s simulator: utilization, overtime, waiting")
    curve_plot(A["curves"], {"Ours: DP": (A["methods"]["Ours: DP"]["J"]["mean"], "--"), "Rule": (A["methods"]["Rule"]["J"]["mean"], ":")},
               f"{D}/fig_A_convergence.png", "A. Monte Carlo vs Q-learning convergence")
    # data size
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    xs = [int(k) for k in A["data_size"]]
    for nm, col, mk in (("Sir: DP", COL["DP"], "o"), ("Ours: DP", OURS, "D")):
        ax.errorbar(xs, [A["data_size"][str(x)][nm]["mean"] for x in xs], yerr=[A["data_size"][str(x)][nm]["ci"] for x in xs],
                    marker=mk, color=col, capsize=3, lw=1.6, label=nm)
    ax.axhline(A["methods"]["Rule"]["J"]["mean"], color=COL["Rule"], ls=":", lw=1.1, label="Rule")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(xs)
    ax.set_xlabel("Logged clinic days used for learning"); ax.set_ylabel("True return"); ax.legend(fontsize=7)
    ax.set_title("A. How much data each method needs")
    fig.savefig(f"{D}/fig_A_datasize.png"); plt.close(fig)
    # shift
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    conds = list(A["shift"]); names = ["Rule", "Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning", "Ours: DP"]
    cs = colors_for(names); w = 0.16
    for i, nm in enumerate(names):
        ax.bar(np.arange(len(conds)) + (i - 2) * w, [A["shift"][c][nm]["mean"] for c in conds], w * 0.92,
               yerr=[A["shift"][c][nm]["ci"] for c in conds], color=cs[nm], capsize=2, label=nm)
    lo = min(A["shift"][c][nm]["mean"] for c in conds for nm in names)
    ax.set_ylim(lo * 0.92, None); ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
    ax.set_ylabel("True return"); ax.set_title("A. When the no-show rate changes")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(f"{D}/fig_A_shift.png"); plt.close(fig)

    oB = ["Rule", [k for k in B["methods"] if k.startswith("Threshold")][0], [k for k in B["methods"] if k.startswith("Fixed")][0],
          "One-session optimiser", "Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning", "Oracle DP (10x data)",
          "Ours: Monte Carlo", "Ours: Q-learning", "Ours: DP"]
    dot_plot(B["methods"], oB, colors_for(oB), f"{D}/fig_B_returns.png", "B. Real Kaggle sessions: return by method")
    metric_panels(B["methods"], oB, f"{D}/fig_B_metrics.png", "B. Real Kaggle sessions: utilization, overtime, waiting")
    curve_plot(B["curves"], {"Ours: DP": (B["methods"]["Ours: DP"]["J"]["mean"], "--"), "Rule": (B["methods"]["Rule"]["J"]["mean"], ":")},
               f"{D}/fig_B_convergence.png", "B. Monte Carlo vs Q-learning convergence")
    # demand
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    keys = list(B["demand"]); names = ["Sir: DP", "Best fixed overbooking", "One-session optimiser", "Ours: DP"]
    cs = {"Sir: DP": COL["DP"], "Best fixed overbooking": "#999999", "One-session optimiser": BASE, "Ours: DP": OURS}; w = 0.2
    for i, nm in enumerate(names):
        vals = [B["demand"][k][nm]["mean"] if isinstance(B["demand"][k][nm], dict) else B["demand"][k][nm] for k in keys]
        ax.bar(np.arange(len(keys)) + (i - 1.5) * w, vals, w * 0.92, color=cs[nm], label=nm)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace(", ", "\n") for k in keys], fontsize=7)
    ax.set_ylabel("True return"); ax.set_title("B. Busier clinic: more arrivals, costlier waiting")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.savefig(f"{D}/fig_B_demand.png"); plt.close(fig)


# ============================================================================= summary print
def show(title, summary, paired_d=None):
    print(f"\n{title}\n{'method':44s} {'return':>15s} {'util %':>7s} {'overtime':>8s} {'waiting':>8s}")
    for nm, s in sorted(summary.items(), key=lambda kv: -kv[1]["J"]["mean"]):
        print(f"{nm:44s} {s['J']['mean']:8.2f} ± {s['J']['ci']:4.2f} {100 * s['utilization']['mean']:7.1f} "
              f"{s['overtime']['mean']:8.2f} {s['waiting']['mean']:8.2f}")
    for k, v in (paired_d or {}).items():
        print(f"  {k:60s} {v['mean']:+7.2f} ± {v['ci']:.2f}  p={v['p']:.2g}")


if __name__ == "__main__":
    log(f"config {CFG}")
    A = comparison_A()
    B = comparison_B()
    figures(A, B)
    res = dict(config=CFG, A=A, B=B, runtime_s=time.time() - T0)
    with open(f"{D}/results.json", "w") as f:
        import json
        json.dump(jsonable(res), f, indent=1)
    show(f"A. Sir's simulator ({CFG['seeds']} seeds)", A["methods"], A["paired"])
    show("B. Real Kaggle sessions", B["methods"], B["paired_episodes"])
    print("\nA data size:", {k: (round(v["Sir: DP"]["mean"], 2), round(v["Ours: DP"]["mean"], 2)) for k, v in A["data_size"].items()})
    print("B busier clinic:", {k: (round(v["Ours: DP"]["mean"], 2), round(v["One-session optimiser"], 2), f"{v['gain_vs_one_session_pct']:+.1f}%")
                               for k, v in B["demand"].items()})
    log(f"done; results and figures in {D}")
