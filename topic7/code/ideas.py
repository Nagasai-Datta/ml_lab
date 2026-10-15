"""ideas.py -- the ten experiments (one per paper).  Run:  python ideas.py <k>"""
import copy, sys, time
import numpy as np
from scipy import stats
from common import *
from domains import *


def blend(mi, mp, lam):
    """empirical-Bayes 'digital twin': patient counts + lam pseudo-counts from the population model"""
    m = copy.copy(mi)
    nS, nA = mi.nS, mi.nA
    lam_eff = np.where(mp.seen, lam, 0.0)
    n_tot = mi.n + lam_eff
    ok = n_tot > 0
    P = np.zeros_like(mi.P)
    P[ok] = (mi.N[ok] + lam_eff[ok][:, None] * mp.P[ok]) / n_tot[ok][:, None]
    R = np.zeros_like(mi.R)
    R[ok] = (mi.Rsum[ok] + lam_eff[ok] * mp.R[ok]) / n_tot[ok]
    P[~ok, nS] = 1.0
    R[~ok] = mi.rmin
    P[nS] = 0.0; P[nS, :, nS] = 1.0; R[nS] = 0.0
    m.P, m.R, m.n = P, R, n_tot
    m.PG = P * mi.gamma
    m.G = np.full_like(P, mi.gamma)
    m.visits = n_tot.sum(1)[:nS]
    return m


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return dict(mean=float(d.mean()), ci=ci95(d), n=len(d),
                p=float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 2 and d.std() > 0 else 1.0)


def norm_score(J_, lo, hi):
    return (np.asarray(J_) - lo) / (hi - lo + 1e-12)


# =============================================================================
def idea1():
    I = 1; d = odir(I)
    dom = Diabetes()
    res = standard_benchmark(dom, n_eps=150, n_oracle=1500, steps=120000, seeds=5)
    standard_figs(res, dom, I, title="Population MDP (Diabetes CGM simulator)")
    rule = np.array([dom.rule(s) for s in range(dom.nS)])
    SIG = [0.15, 0.3, 0.5, 0.8]            # heterogeneity of insulin sensitivity (log-sd)
    NDS = [3, 7, 14, 28]
    LAMS = [10, 30, 100]
    names = ["Rule", "Pooled", "Individual"] + [f"Twin (lam={l})" for l in LAMS] + ["Oracle twin"]
    NP = 20
    R = {sg: {n: {nd: {"J": [], "in_range": [], "hypo": []} for nd in NDS} for n in names} for sg in SIG}
    SENS = {}
    for gi, sg in enumerate(SIG):
        rng = np.random.default_rng(77 + gi)
        reg_sens = np.clip(rng.lognormal(0, sg, 40), 0.35, 2.5)
        parts = []
        for s_ in reg_sens:
            di = Diabetes(sens=float(s_))
            parts.append(rollout(di, di.behavior, rng, 20))
        m_pop = build_mdp(concat_data(*parts), dom)
        pi_pop, _, _ = value_iteration(m_pop)
        pi_pop = fill_policy(pi_pop, m_pop, rule)
        test_sens = np.clip(np.random.default_rng(5 + gi).lognormal(0, sg, NP), 0.35, 2.5)
        SENS[sg] = test_sens

        def ev(n, nd, di, pi):
            r, inf = eval_policy(di, pi, n_eps=100)
            R[sg][n][nd]["J"].append(float(r.mean()))
            R[sg][n][nd]["in_range"].append(float(inf["in_range"].mean()))
            R[sg][n][nd]["hypo"].append(float(inf["hypo"].mean()))

        for pix, s_ in enumerate(test_sens):
            di = Diabetes(sens=float(s_))
            mo = build_mdp(rollout(di, di.behavior, np.random.default_rng([8, gi, pix]), 2000), dom)
            po, _, _ = value_iteration(mo); po = fill_policy(po, mo, rule)
            for nd in NDS:
                dat = rollout(di, di.behavior, np.random.default_rng([9, gi, pix, nd]), nd)
                mi = build_mdp(dat, dom)
                ev("Rule", nd, di, rule); ev("Pooled", nd, di, pi_pop); ev("Oracle twin", nd, di, po)
                p, _, _ = value_iteration(mi); ev("Individual", nd, di, fill_policy(p, mi, rule))
                for l in LAMS:
                    mb = blend(mi, m_pop, l)
                    p, _, _ = value_iteration(mb); ev(f"Twin (lam={l})", nd, di, fill_policy(p, mb, rule))
        print("idea1 sigma", sg, {n: round(float(np.mean(R[sg][n][7]["J"])), 2) for n in names}, flush=True)
    cols = {"Rule": "#555555", "Pooled": "#CC79A7", "Individual": "#E69F00", "Twin (lam=10)": "#56B4E9",
            "Twin (lam=30)": "#0072B2", "Twin (lam=100)": "#009E73", "Oracle twin": "k"}
    # fig A: J vs nd for each sigma
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=False)
    for ax, sg in zip(axs, SIG):
        for n in names:
            mu = [np.mean(R[sg][n][nd]["J"]) for nd in NDS]; ci = [ci95(R[sg][n][nd]["J"]) for nd in NDS]
            ax.errorbar(range(4), mu, yerr=ci, marker="o", ms=3, lw=1.3, capsize=1.5, color=cols[n], label=n, ls="--" if n == "Oracle twin" else "-")
        ax.set_xticks(range(4)); ax.set_xticklabels(NDS); ax.set_title(f"sigma = {sg}", fontsize=9); ax.set_xlabel("Patient's own days")
    axs[0].set_ylabel("True discounted return"); axs[0].legend(fontsize=6)
    fig.savefig(f"{d}/fig_twin_curve.png"); plt.close(fig)
    # fig B: ceiling & realised value of personalisation
    ceil = [paired(R[sg]["Oracle twin"][7]["J"], R[sg]["Pooled"][7]["J"]) for sg in SIG]
    tw = {l: [paired(R[sg][f"Twin (lam={l})"][7]["J"], R[sg]["Pooled"][7]["J"]) for sg in SIG] for l in LAMS}
    ind = [paired(R[sg]["Individual"][7]["J"], R[sg]["Pooled"][7]["J"]) for sg in SIG]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.errorbar(range(4), [c["mean"] for c in ceil], yerr=[c["ci"] for c in ceil], marker="s", color="k", ls="--", capsize=2, label="Oracle twin (2000 days) - Pooled")
    for l, c_ in zip(LAMS, ("#56B4E9", "#0072B2", "#009E73")):
        ax.errorbar(range(4), [c["mean"] for c in tw[l]], yerr=[c["ci"] for c in tw[l]], marker="o", color=c_, capsize=2, label=f"Twin (lam={l}) - Pooled")
    ax.errorbar(range(4), [c["mean"] for c in ind], yerr=[c["ci"] for c in ind], marker="o", color="#E69F00", capsize=2, label="Individual - Pooled")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(SIG)
    ax.set_xlabel("Heterogeneity of insulin sensitivity (log-sd)"); ax.set_ylabel("Return difference vs pooled model (7 days of data)")
    ax.legend(fontsize=6.5); fig.savefig(f"{d}/fig_twin_gain.png"); plt.close(fig)
    # fig C: scatter at sigma=0.8, 7 days
    sg = 0.8; nd = 7; sens = SENS[sg]
    gp = np.array(R[sg]["Twin (lam=30)"][nd]["J"]) - np.array(R[sg]["Pooled"][nd]["J"])
    go = np.array(R[sg]["Oracle twin"][nd]["J"]) - np.array(R[sg]["Pooled"][nd]["J"])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    x = np.log(sens)
    axs[0].scatter(x, gp, color="#0072B2"); axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_xlabel("log insulin sensitivity of the patient"); axs[0].set_ylabel("Twin(lam=30) - Pooled return"); axs[0].set_title("sigma=0.8, 7 days", fontsize=8)
    axs[1].scatter(x, go, color="#D55E00"); axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_xlabel("log insulin sensitivity of the patient"); axs[1].set_ylabel("Oracle twin - Pooled return"); axs[1].set_title("ceiling of personalisation", fontsize=8)
    rho = stats.spearmanr(np.abs(x), go)
    fig.savefig(f"{d}/fig_twin_scatter.png"); plt.close(fig)
    # fig D: clinical outcomes at sigma 0.8, nd 7
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    sel = ["Rule", "Pooled", "Individual", "Twin (lam=30)", "Oracle twin"]
    for ax, key, yl in ((axs[0], "in_range", "In-range hours / day (of 24)"), (axs[1], "hypo", "Hypoglycaemic hours / day")):
        ax.bar(range(len(sel)), [np.mean(R[sg][n][nd][key]) for n in sel], yerr=[ci95(R[sg][n][nd][key]) for n in sel], color=[cols[n] for n in sel], capsize=3)
        ax.set_xticks(range(len(sel))); ax.set_xticklabels([s_.replace(" ", "\n") for s_ in sel], fontsize=7); ax.set_ylabel(yl)
    fig.savefig(f"{d}/fig_twin_clinical.png"); plt.close(fig)
    summ = {str(sg_): {n: {nd_: dict(J=float(np.mean(R[sg_][n][nd_]["J"])), ci=ci95(R[sg_][n][nd_]["J"]),
                                    in_range=float(np.mean(R[sg_][n][nd_]["in_range"])), hypo=float(np.mean(R[sg_][n][nd_]["hypo"])))
                          for nd_ in NDS} for n in names} for sg_ in SIG}
    out = dict(standard=summarize(res), twin=summ, nds=NDS, sigmas=SIG, lams=LAMS,
               ceiling_vs_pooled={str(sg_): ceil[i] for i, sg_ in enumerate(SIG)},
               twin_vs_pooled={str(l): {str(sg_): tw[l][i] for i, sg_ in enumerate(SIG)} for l in LAMS},
               indiv_vs_pooled={str(sg_): ind[i] for i, sg_ in enumerate(SIG)},
               twin_vs_indiv={str(l): {str(sg_): paired(R[sg_][f"Twin (lam={l})"][7]["J"], R[sg_]["Individual"][7]["J"]) for sg_ in SIG} for l in LAMS},
               twin_vs_pooled_by_nd={str(l): {str(sg_): {str(nd_): paired(R[sg_][f"Twin (lam={l})"][nd_]["J"], R[sg_]["Pooled"][nd_]["J"]) for nd_ in NDS} for sg_ in SIG} for l in LAMS},
               spearman_atyp_ceiling=dict(rho=float(rho.correlation), p=float(rho.pvalue)),
               frac_oracle_beats_pooled={str(sg_): float((np.array(R[sg_]["Oracle twin"][7]["J"]) > np.array(R[sg_]["Pooled"][7]["J"])).mean()) for sg_ in SIG},
               test_sens={str(k): v for k, v in SENS.items()})
    save_json(I, "results", dict(out, extra=drop_models(res)))
    return out


# =============================================================================
def idea2():
    I = 2; d = odir(I)
    res = standard_benchmark(Sepsis(3), n_eps=500, n_oracle=5000, steps=150000, seeds=5)
    standard_figs(res, Sepsis(3), I, title="Sepsis simulator, 27-state discretisation")
    GR = [2, 3, 5, 8]; KS = [0, 0.05, 0.1, 0.25, 0.5, 1.0]
    SEEDS = 5
    Jt = np.zeros((len(GR), len(KS), SEEDS)); gap = np.zeros_like(Jt); mort = np.zeros_like(Jt)
    agree = np.zeros_like(Jt); rule_J = []; cover = []
    for gi, g in enumerate(GR):
        dom = Sepsis(g)
        rule = np.array([dom.rule(s) for s in range(dom.nS)])
        r, inf = eval_policy(dom, rule, n_eps=300)
        rule_J.append((float(r.mean()), float(inf["died"].mean())))
        for sd in range(SEEDS):
            data = rollout(dom, dom.behavior, np.random.default_rng(100 + sd), 500)
            m = build_mdp(data, dom)
            cover.append((g, float((m.visits > 0).mean()), float(np.median(m.n[m.n > 0]))))
            rng_ = m.rmax - m.rmin
            pen = np.where(m.n > 0, rng_ / np.sqrt(np.maximum(m.n, 1)), 0.0)
            for ki, k in enumerate(KS):
                pi, _, _ = value_iteration(m, penalty=k * pen)
                pi = fill_policy(pi, m, rule)
                r, inf = eval_policy(dom, pi, n_eps=300)
                Jt[gi, ki, sd] = r.mean(); mort[gi, ki, sd] = inf["died"].mean()
                gap[gi, ki, sd] = J(m, pi) - r.mean()
                vis = m.visits >= 5
                agree[gi, ki, sd] = (pi[vis] == rule[vis]).mean() if vis.any() else np.nan
        print("idea2 gran", g, np.round(Jt[gi].mean(1), 1), flush=True)
    # gain over the clinician rule (rule is evaluated on the same discretisation)
    rJ = np.array([x[0] for x in rule_J])
    Gain = Jt - rJ[:, None, None]
    cov_g = {g: np.mean([c for gg, c, _ in cover if gg == g]) for g in GR}
    med_n = {g: np.mean([mm for gg, _, mm in cover if gg == g]) for g in GR}
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    M = Gain.mean(2)
    im = axs[0].imshow(M, aspect="auto", cmap="RdYlGn", vmin=-15, vmax=15); axs[0].grid(False)
    axs[0].set_xticks(range(len(KS))); axs[0].set_xticklabels(KS); axs[0].set_yticks(range(len(GR)))
    axs[0].set_yticklabels([f"{g} bins ({g**3} states)" for g in GR]); axs[0].set_xlabel("Pessimism kappa")
    axs[0].set_title("Gain over guideline rule (true return)", fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axs[0].text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.02)
    xs = np.arange(len(GR))
    axs[1].plot(xs, [100 * cov_g[g] for g in GR], marker="o", color="#0072B2", label="States with supported data (%)")
    axs[1].plot(xs, [100 * np.nanmean(agree[i, 0]) for i in range(len(GR))], marker="s", color="#D55E00", label="Agreement with rule, kappa=0 (%)")
    axs[1].plot(xs, [Gain[i, 0].mean() * 5 for i in range(len(GR))], marker="^", color="#009E73", label="Gain over rule x5, kappa=0")
    axs[1].set_xticks(xs); axs[1].set_xticklabels([f"{g}" for g in GR]); axs[1].set_xlabel("Bins per variable"); axs[1].legend(fontsize=6.5)
    fig.savefig(f"{d}/fig_heatmaps.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for gi, g in enumerate(GR):
        axs[0].errorbar(KS, Jt[gi].mean(1), yerr=[ci95(Jt[gi, k]) for k in range(len(KS))], marker="o", ms=3, capsize=2, label=f"{g} bins", color=OKABE[gi])
        axs[1].plot(KS, 100 * mort[gi].mean(1), marker="o", ms=3, label=f"{g} bins", color=OKABE[gi])
        axs[0].axhline(rule_J[gi][0], color=OKABE[gi], ls=":", lw=0.9)
    axs[0].set_xlabel("kappa"); axs[0].set_ylabel("True discounted return (dotted = guideline rule)")
    axs[1].set_xlabel("kappa"); axs[1].set_ylabel("Simulated in-ICU mortality (%)")
    axs[0].legend(fontsize=7)
    fig.savefig(f"{d}/fig_kappa.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for gi, g in enumerate(GR):
        ax.plot(KS, np.nanmean(agree[gi], 1), marker="o", ms=3, label=f"{g} bins", color=OKABE[gi])
    ax.set_xlabel("kappa"); ax.set_ylabel("Agreement with guideline rule\n(states with >=5 visits)"); ax.legend(fontsize=7)
    fig.savefig(f"{d}/fig_agreement_rule.png"); plt.close(fig)
    best = np.unravel_index(np.argmax(Jt.mean(2)), Jt.shape[:2])
    out = dict(GR=GR, KS=KS, gain=Gain.mean(2), gain_ci=np.array([[ci95(Gain[i, j]) for j in range(len(KS))] for i in range(len(GR))]), cov=cov_g, med_n=med_n, J=Jt.mean(2), J_ci=np.array([[ci95(Jt[i, j]) for j in range(len(KS))] for i in range(len(GR))]),
               gap=gap.mean(2), mort=mort.mean(2), agree=np.nanmean(agree, 2), rule=rule_J, cover=cover,
               best=dict(gran=GR[best[0]], kappa=KS[best[1]], J=float(Jt.mean(2)[best])),
               pess_vs_naive={str(g): paired(Jt[gi, 3], Jt[gi, 0]) for gi, g in enumerate(GR)},
               best_pess_vs_naive_gap={str(g): [float(gap[gi, 0].mean()), float(gap[gi, 3].mean())] for gi, g in enumerate(GR)})
    save_json(I, "results", dict(out, standard=summarize(res), extra=drop_models(res)))
    return out


# =============================================================================
def idea3():
    I = 3; d = odir(I)
    base = Hospital(1.0); surge = Hospital(1.3)
    res = standard_benchmark(base, n_eps=800, n_oracle=6000, steps=150000, seeds=5)
    standard_figs(res, base, I, title="Hospital capacity MDP (baseline demand)")
    NEW = [0, 5, 10, 20, 40, 80]; SEEDS = 6
    variants = ["Static (baseline only)", "Pooled, equal weight", "Pooled, recency x10", "Surge data only"]
    Js = {v: np.zeros((len(NEW), SEEDS)) for v in variants}
    Jb = {v: np.zeros((len(NEW), SEEDS)) for v in variants}
    ov = []
    rule = np.array([base.rule(s) for s in range(base.nS)])
    # oracle for surge
    ds = rollout(surge, surge.behavior, np.random.default_rng(555), 6000)
    mo = build_mdp(ds, surge); po, _, _ = value_iteration(mo); po = fill_policy(po, mo, behavior_mode(ds, surge))
    r_or, _ = eval_policy(surge, po, n_eps=300); r_rule, _ = eval_policy(surge, rule, n_eps=300)
    for sd in range(SEEDS):
        d0 = rollout(base, base.behavior, np.random.default_rng(10 + sd), 800)
        dn_all = rollout(surge, surge.behavior, np.random.default_rng(20 + sd), max(NEW))
        for ni, n in enumerate(NEW):
            if n == 0:
                dd = d0; wlist = {v: None for v in variants}
            for v in variants:
                if n == 0:
                    comb, w = d0, None
                else:
                    sub = {k: val[dn_all["ep"] < n] for k, val in dn_all.items()}
                    if v == "Static (baseline only)": comb, w = d0, None
                    elif v == "Surge data only": comb, w = sub, None
                    else:
                        comb = concat_data(d0, sub)
                        wnew = 1.0 if v == "Pooled, equal weight" else 10.0
                        w = np.concatenate([np.ones(len(d0["s"])), np.full(len(sub["s"]), wnew)])
                if v == "Surge data only" and n == 0:
                    comb = d0
                m = build_mdp(comb, base, weights=w)
                p, _, _ = value_iteration(m); p = fill_policy(p, m, rule)
                r, _ = eval_policy(surge, p, n_eps=200); Js[v][ni, sd] = r.mean()
                r2, _ = eval_policy(base, p, n_eps=200); Jb[v][ni, sd] = r2.mean()
        print("idea3 seed", sd, flush=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, v in enumerate(variants):
        ax.errorbar(NEW, Js[v].mean(1), yerr=[ci95(Js[v][k]) for k in range(len(NEW))], marker="o", ms=3.5, capsize=2, color=OKABE[i], label=v)
    ax.axhline(r_or.mean(), color="k", ls="--", lw=1, label="Oracle (surge-trained)")
    ax.axhline(r_rule.mean(), color="#555555", ls=":", lw=1.2, label="Occupancy rule")
    ax.set_xlabel("Surge-regime episodes (28 days each) available"); ax.set_ylabel("Return under +30% demand surge")
    ax.set_xticks(NEW); ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2)
    fig.savefig(f"{d}/fig_adaptation.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for i, v in enumerate(variants):
        ax.plot(NEW, Jb[v].mean(1), marker="o", ms=3.5, color=OKABE[i], label=v)
    ax.set_xlabel("Surge-regime episodes available"); ax.set_ylabel("Return under original demand (regret check)")
    ax.legend(fontsize=7)
    fig.savefig(f"{d}/fig_regret.png"); plt.close(fig)
    # cost-structure plot for the two heads
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    labs = ["Static", "Pooled x10 (20 eps)", "Surge-only (20 eps)", "Oracle"]
    vals = [Js[variants[0]][0].mean(), Js[variants[2]][3].mean(), Js[variants[3]][3].mean(), r_or.mean()]
    errs = [ci95(Js[variants[0]][0]), ci95(Js[variants[2]][3]), ci95(Js[variants[3]][3]), ci95(r_or)]
    ax.bar(labs, vals, yerr=errs, color=[OKABE[0], OKABE[2], OKABE[3], "k"], capsize=3)
    ax.set_ylabel("Return under surge"); ax.set_xticklabels(labs, fontsize=7, rotation=12)
    lo = min(vals) - 15; ax.set_ylim(lo, max(vals) + 10)
    fig.savefig(f"{d}/fig_surge_bars.png"); plt.close(fig)
    # ---- recency-weight sweep: the adaptation / regret frontier
    OM = [1, 3, 10, 30, 100]; NN = [10, 20, 40]
    Jo = {(n, w): ([], []) for n in NN for w in OM}
    for sd in range(SEEDS):
        d0 = rollout(base, base.behavior, np.random.default_rng(10 + sd), 800)
        dn_all = rollout(surge, surge.behavior, np.random.default_rng(20 + sd), max(NN))
        for n in NN:
            sub = {k: val[dn_all["ep"] < n] for k, val in dn_all.items()}
            comb = concat_data(d0, sub)
            for w_ in OM:
                wt = np.concatenate([np.ones(len(d0["s"])), np.full(len(sub["s"]), float(w_))])
                m = build_mdp(comb, base, weights=wt)
                p, _, _ = value_iteration(m); p = fill_policy(p, m, rule)
                Jo[(n, w_)][0].append(float(eval_policy(surge, p, n_eps=200)[0].mean()))
                Jo[(n, w_)][1].append(float(eval_policy(base, p, n_eps=200)[0].mean()))
        print("idea3 omega seed", sd, flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for i, n in enumerate(NN):
        axs[0].errorbar(range(len(OM)), [np.mean(Jo[(n, w_)][0]) for w_ in OM], yerr=[ci95(Jo[(n, w_)][0]) for w_ in OM], marker="o", capsize=2, color=OKABE[i], label=f"{n} surge episodes")
        axs[1].plot([np.mean(Jo[(n, w_)][1]) for w_ in OM], [np.mean(Jo[(n, w_)][0]) for w_ in OM], marker="o", color=OKABE[i], label=f"{n} surge episodes")
        for w_ in OM: axs[1].annotate(f"{w_}", (np.mean(Jo[(n, w_)][1]), np.mean(Jo[(n, w_)][0])), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axs[0].set_xticks(range(len(OM))); axs[0].set_xticklabels(OM); axs[0].set_xlabel("Recency weight omega on surge data"); axs[0].set_ylabel("Return under surge"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel("Return under original demand"); axs[1].set_ylabel("Return under surge")
    fig.savefig(f"{d}/fig_omega.png"); plt.close(fig)
    omega = {str(n): {str(w_): dict(surge=float(np.mean(Jo[(n, w_)][0])), surge_ci=ci95(Jo[(n, w_)][0]), base=float(np.mean(Jo[(n, w_)][1]))) for w_ in OM} for n in NN}
    out = dict(omega=omega, OM=OM, NN=NN, NEW=NEW, variants=variants,
               J_surge={v: dict(mean=Js[v].mean(1), ci=[ci95(Js[v][k]) for k in range(len(NEW))]) for v in variants},
               J_base={v: dict(mean=Jb[v].mean(1)) for v in variants},
               oracle_surge=float(r_or.mean()), rule_surge=float(r_rule.mean()),
               static_vs_pooled10_20=paired(Js[variants[2]][3], Js[variants[0]][0]),
               surgeonly_vs_static_20=paired(Js[variants[3]][3], Js[variants[0]][0]),
               pooled10_vs_surgeonly_5=paired(Js[variants[2]][1], Js[variants[3]][1]))
    save_json(I, "results", dict(out, standard=summarize(res), extra=drop_models(res)))
    return out


# =============================================================================
def idea4():
    I = 4; d = odir(I)
    dom = Hypertension(); mk = dom.mask()
    ru = standard_benchmark(dom, n_eps=300, n_oracle=3000, steps=120000, seeds=10, mask=None, n_eval=500)
    rm = standard_benchmark(dom, n_eps=300, n_oracle=3000, steps=120000, seeds=10, mask=mk, n_eval=500)
    standard_figs(rm, dom, I, tag="_guard", title="Guardrailed learners")
    standard_figs(ru, dom, I, tag="_free", title="Unconstrained learners")
    # learning curves 4-way
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for nm, ls_ in (("Monte Carlo", "-"), ("Q-learning", "-")):
        for tag, res_, c in (("free", ru, 0.45), ("guarded", rm, 1.0)):
            x, mu, sd = curve_mean(res_["curves"][nm])
            ax.plot(x, mu, ls=ls_, color=COL[nm], alpha=c, lw=2 if tag == "guarded" else 1.2,
                    label=f"{nm} ({tag})")
    ax.axhline(rm["J_oracle_model"]["Oracle DP"], color="k", ls="--", lw=1, label="Guardrailed DP optimum")
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Policy value (guarded oracle MDP)"); ax.legend(fontsize=7)
    fig.savefig(f"{d}/fig_learning_compare.png"); plt.close(fig)
    meth = ["Rule", "DP", "Monte Carlo", "Q-learning"]
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.0))
    for ax, (key, yl) in zip(axs, (("hypo", "Hypotensive months / episode"), ("unsafe_up", "Unsafe up-titrations / episode"), ("J", "True return"))):
        w = 0.38
        for i, (res_, lab, c) in enumerate(((ru, "free", "#999999"), (rm, "guardrailed", "#0072B2"))):
            vals, errs = [], []
            for mth in meth:
                v = res_["methods"][mth]["J"] if key == "J" else res_["methods"][mth]["info"][key]
                vals.append(np.mean(v)); errs.append(ci95(v))
            ax.bar(np.arange(len(meth)) + (i - 0.5) * w, vals, w, yerr=errs, capsize=2, color=c, label=lab)
        ax.set_xticks(range(len(meth))); ax.set_xticklabels([m.replace(" ", "\n") for m in meth], fontsize=7); ax.set_ylabel(yl)
    axs[0].legend(fontsize=7)
    fig.savefig(f"{d}/fig_safety.png"); plt.close(fig)
    s_u, s_m = summarize(ru), summarize(rm)
    lo = float(np.mean(rm["methods"]["Behavior"]["Jor"])); hi = rm["J_oracle_model"]["Oracle DP"]
    t90 = {}
    for nm in ("Monte Carlo", "Q-learning"):
        for tag, rr in (("free", ru), ("guard", rm)):
            lo_ = float(np.mean(rr["methods"]["Behavior"]["Jor"])); hi_ = rr["J_oracle_model"]["Oracle DP"]
            t90[f"{nm}|{tag}"] = np.nanmean(steps_to_fraction(rr["curves"][nm], lo_ + 0.9 * (hi_ - lo_)))
    out = dict(free=s_u, guard=s_m, steps_to_90=t90,
               pair_unsafe={m: paired(ru["methods"][m]["info"]["unsafe_up"], rm["methods"][m]["info"]["unsafe_up"]) for m in meth[1:]},
               pair_hypo={m: paired(ru["methods"][m]["info"]["hypo"], rm["methods"][m]["info"]["hypo"]) for m in meth[1:]},
               pair_J={m: paired(rm["methods"][m]["J"], ru["methods"][m]["J"]) for m in meth[1:]},
               J_oracle=dict(free=ru["J_oracle_model"]["Oracle DP"], guard=rm["J_oracle_model"]["Oracle DP"]))
    save_json(I, "results", dict(out, free_full=drop_models(ru), guard_full=drop_models(rm)))
    return out


# =============================================================================
def idea5():
    I = 5; d = odir(I)
    dom = Screening()
    rs = standard_benchmark(dom, n_eps=3000, n_oracle=30000, steps=150000, seeds=5, n_eval=15000)
    rn = standard_benchmark(dom, n_eps=3000, n_oracle=30000, steps=150000, seeds=5, dur_one=True, n_eval=15000)
    standard_figs(rs, dom, I, tag="_smdp", title="Semi-Markov screening MDP")
    fixed = {"Annual": fixed_policy(dom, 0), "Biennial": fixed_policy(dom, 1), "Triennial": fixed_policy(dom, 2)}
    fx = {}
    for k, p in fixed.items():
        r, inf = eval_policy(dom, p, n_eps=15000)
        fx[k] = dict(J=float(r.mean()), ci=ci95(r), screens=float(inf["screens"].mean()), early=float(inf["early"].mean()), late=float(inf["late"].mean()))
    # re-evaluate rule/dp/mc/q with larger n for the pareto
    pts = {}
    for k in ("Rule", "DP", "Monte Carlo", "Q-learning"):
        pts[f"{k} (SMDP)"] = dict(screens=float(np.mean(rs["methods"][k]["info"]["screens"])), early=float(np.mean(rs["methods"][k]["info"]["early"])),
                                  late=float(np.mean(rs["methods"][k]["info"]["late"])), J=float(np.mean(rs["methods"][k]["J"])), ci=ci95(rs["methods"][k]["J"]))
    for k in ("DP", "Monte Carlo", "Q-learning"):
        pts[f"{k} (naive MDP)"] = dict(screens=float(np.mean(rn["methods"][k]["info"]["screens"])), early=float(np.mean(rn["methods"][k]["info"]["early"])),
                                       late=float(np.mean(rn["methods"][k]["info"]["late"])), J=float(np.mean(rn["methods"][k]["J"])), ci=ci95(rn["methods"][k]["J"]))
    for k, v in fx.items(): pts[k] = v
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    mk = {"Rule (SMDP)": ("s", "#555555"), "DP (SMDP)": ("o", "#0072B2"), "Monte Carlo (SMDP)": ("o", "#D55E00"), "Q-learning (SMDP)": ("o", "#009E73"),
          "DP (naive MDP)": ("^", "#0072B2"), "Monte Carlo (naive MDP)": ("^", "#D55E00"), "Q-learning (naive MDP)": ("^", "#009E73"),
          "Annual": ("X", "#CC79A7"), "Biennial": ("X", "#E69F00"), "Triennial": ("X", "#56B4E9")}
    for k, v in pts.items():
        ax.scatter(v["screens"], v["early"], marker=mk[k][0], color=mk[k][1], s=45, label=k, edgecolor="k", lw=0.4)
    ax.set_xlabel("Screens per patient-episode"); ax.set_ylabel("Early (stage-2) detections per episode")
    ax.legend(fontsize=6, ncol=2); fig.savefig(f"{d}/fig_pareto.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    names = list(pts.keys()); vals = [pts[k]["J"] for k in names]; errs = [pts[k]["ci"] for k in names]
    ax.barh(range(len(names)), vals, xerr=errs, color=[mk[k][1] for k in names], capsize=2)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7); ax.set_xlabel("Discounted return (true simulator)")
    fig.savefig(f"{d}/fig_returns.png"); plt.close(fig)
    # interval heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, rr, tl in ((axs[0], rs, "SMDP-aware DP"), (axs[1], rn, "Naive MDP DP")):
        pol = np.array(rr["policies"]["Oracle DP"]); M = np.zeros((3, 6))
        for t in range(3):
            for ab in range(6):
                M[t, ab] = pol[((t * 6 + ab) * 2 + 0) * 3 + 1] + 1
        im = ax.imshow(M, cmap="YlGnBu", vmin=1, vmax=3); ax.grid(False)
        for t in range(3):
            for ab in range(6): ax.text(ab, t, f"{int(M[t, ab])}y", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(6)); ax.set_xticklabels(["<50", "50-54", "55-59", "60-64", "65-69", "70+"], fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(["low", "mid", "high"]); ax.set_xlabel("Age band"); ax.set_ylabel("Risk tier"); ax.set_title(tl, fontsize=9)
    fig.savefig(f"{d}/fig_intervals.png"); plt.close(fig)
    # ---- ablation: how much does the semi-Markov correction matter as the discount steepens?
    GMS = [0.6, 0.75, 0.9, 0.97]; SW = {}
    for gm in GMS:
        dg = Screening(gamma=gm); diffs, differ, Js_, Jn_ = [], [], [], []
        for sd in range(5):
            dat = rollout(dg, dg.behavior, np.random.default_rng(4000 + sd), 3000)
            dn = dict(dat); dn["dur"] = np.ones_like(dat["dur"])
            ms, mn = build_mdp(dat, dg), build_mdp(dn, dg)
            bm = behavior_mode(dat, dg)
            ps, _, _ = value_iteration(ms); pn, _, _ = value_iteration(mn)
            ps, pn = fill_policy(ps, ms, bm), fill_policy(pn, mn, bm)
            rs_, _ = eval_policy(dg, ps, n_eps=10000); rn_, _ = eval_policy(dg, pn, n_eps=10000)
            Js_.append(float(rs_.mean())); Jn_.append(float(rn_.mean())); diffs.append(float((rs_ - rn_).mean()))
            vis = ms.visits >= 5; differ.append(float((ps[vis] != pn[vis]).mean()))
        SW[gm] = dict(smdp=float(np.mean(Js_)), naive=float(np.mean(Jn_)), diff=paired(Js_, Jn_), differ=float(np.mean(differ)))
        print("idea5 gamma", gm, SW[gm], flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    axs[0].errorbar(range(len(GMS)), [SW[g]["diff"]["mean"] for g in GMS], yerr=[SW[g]["diff"]["ci"] for g in GMS], marker="o", color="#0072B2", capsize=3)
    axs[0].axhline(0, color="k", lw=0.8); axs[0].set_xticks(range(len(GMS))); axs[0].set_xticklabels(GMS)
    axs[0].set_xlabel("Yearly discount factor"); axs[0].set_ylabel("Return: SMDP-aware minus naive DP")
    axs[1].bar(range(len(GMS)), [100 * SW[g]["differ"] for g in GMS], color="#D55E00")
    axs[1].set_xticks(range(len(GMS))); axs[1].set_xticklabels(GMS); axs[1].set_xlabel("Yearly discount factor"); axs[1].set_ylabel("States where the two policies differ (%)")
    fig.savefig(f"{d}/fig_gamma_sweep.png"); plt.close(fig)
    out = dict(smdp=summarize(rs), naive=summarize(rn), fixed=fx, points=pts,
               smdp_vs_naive_dp=paired(rs["methods"]["DP"]["J"], rn["methods"]["DP"]["J"]),
               smdp_vs_naive_q=paired(rs["methods"]["Q-learning"]["J"], rn["methods"]["Q-learning"]["J"]),
               smdp_vs_naive_mc=paired(rs["methods"]["Monte Carlo"]["J"], rn["methods"]["Monte Carlo"]["J"]),
               gamma_sweep={str(k): v for k, v in SW.items()}, oracle_policy_smdp=rs["policies"]["Oracle DP"], oracle_policy_naive=rn["policies"]["Oracle DP"])
    save_json(I, "results", dict(out, smdp_full=drop_models(rs), naive_full=drop_models(rn)))
    return out


# =============================================================================
def idea6():
    I = 6; d = odir(I)
    dom = CKD()
    res = standard_benchmark(dom, n_eps=800, n_oracle=6000, steps=200000, seeds=5, n_eval=400)
    standard_figs(res, dom, I, title="CKD care MDP (lambda = 1)")
    rule = np.array([dom.rule(s) for s in range(dom.nS)])
    LAMS = [0, 0.25, 0.5, 1, 2, 4, 8, 16]; SEEDS = 10
    BUD = [2, 4, 6]; MARG = [0.0, 0.25, 0.5]
    B = np.zeros((len(LAMS), SEEDS)); C = np.zeros_like(B); F = np.zeros_like(B)
    shp = (len(MARG), len(BUD), SEEDS)
    Bb, Cb, Pred, Fb, Lm = (np.zeros(shp) for _ in range(5))
    for sd in range(SEEDS):
        data = rollout(dom, dom.behavior, np.random.default_rng(300 + sd), 800)
        m = build_mdp(data, dom)
        R1, R2 = m.Rc["c1"], m.Rc["c2"]
        for li, lam in enumerate(LAMS):
            p, _, _ = value_iteration(m, R=R1 - lam * R2); p = fill_policy(p, m, rule)
            r, inf = eval_policy(dom, p, n_eps=300)
            B[li, sd] = inf["d_c1"].mean(); C[li, sd] = inf["d_c2"].mean(); F[li, sd] = inf["failure"].mean()

        def burden(lam):
            p, _, _ = value_iteration(m, R=R1 - lam * R2)
            return float(m.mu @ policy_value(m, p, R2)), p

        for mi_, marg in enumerate(MARG):
            for bi, bud in enumerate(BUD):
                target = bud / (1.0 + marg)
                lo_, hi_ = 0.0, 64.0
                if burden(0.0)[0] <= target: lam = 0.0
                else:
                    for _ in range(25):
                        mid = 0.5 * (lo_ + hi_)
                        if burden(mid)[0] > target: lo_ = mid
                        else: hi_ = mid
                    lam = hi_
                pred, p = burden(lam); p = fill_policy(p, m, rule)
                r, inf = eval_policy(dom, p, n_eps=300)
                Bb[mi_, bi, sd] = inf["d_c1"].mean(); Cb[mi_, bi, sd] = inf["d_c2"].mean()
                Pred[mi_, bi, sd] = pred; Fb[mi_, bi, sd] = inf["failure"].mean(); Lm[mi_, bi, sd] = lam
        print("idea6 seed", sd, flush=True)
    fx = {}
    for nm, p in (("Always routine", fixed_policy(dom, 0)), ("Always enhanced", fixed_policy(dom, 1)), ("Always intensive", fixed_policy(dom, 2)), ("Nephrologist rule", rule)):
        r, inf = eval_policy(dom, p, n_eps=600)
        fx[nm] = dict(benefit=float(inf["d_c1"].mean()), burden=float(inf["d_c2"].mean()), failure=float(inf["failure"].mean()))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.errorbar(C.mean(1), B.mean(1), xerr=[ci95(C[i]) for i in range(len(LAMS))], yerr=[ci95(B[i]) for i in range(len(LAMS))], marker="o", color="#0072B2", lw=1.6, capsize=2, label="Lagrangian DP frontier")
    for i, l in enumerate(LAMS): ax.annotate(f"{l:g}", (C.mean(1)[i], B.mean(1)[i]), textcoords="offset points", xytext=(4, -9), fontsize=7)
    for (nm, v), c in zip(fx.items(), ("#999999", "#E69F00", "#CC79A7", "#555555")):
        ax.scatter(v["burden"], v["benefit"], color=c, s=50, marker="X", edgecolor="k", lw=0.4, label=nm)
    ax.set_xlabel("Discounted treatment burden"); ax.set_ylabel("Discounted kidney-function benefit (incl. failure penalty)")
    ax.legend(fontsize=7); fig.savefig(f"{d}/fig_pareto.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(LAMS, 100 * F.mean(1), marker="o", color="#D55E00"); ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlabel("Lagrange multiplier (price of burden)"); ax.set_ylabel("Kidney failure within the horizon (%)")
    fig.savefig(f"{d}/fig_failure.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
    x = np.arange(len(BUD)); w = 0.26
    for mi_, (marg, c) in enumerate(zip(MARG, ("#999999", "#56B4E9", "#0072B2"))):
        axs[0].bar(x + (mi_ - 1) * w, Cb[mi_].mean(1), w, yerr=[ci95(Cb[mi_, i]) for i in range(len(BUD))], color=c, capsize=2, label=f"margin {int(100 * marg)}%")
        axs[1].bar(x + (mi_ - 1) * w, Bb[mi_].mean(1), w, yerr=[ci95(Bb[mi_, i]) for i in range(len(BUD))], color=c, capsize=2)
    axs[0].plot(x, BUD, "k_", ms=26, mew=2, label="Budget")
    for ax in axs: ax.set_xticks(x); ax.set_xticklabels(BUD); ax.set_xlabel("Burden budget")
    axs[0].set_ylabel("Realised discounted burden (true simulator)"); axs[1].set_ylabel("Realised kidney-function benefit"); axs[0].legend(fontsize=7)
    fig.savefig(f"{d}/fig_budget.png"); plt.close(fig)
    viol = np.array([[float((Cb[mi_, bi] > BUD[bi] + 1e-9).mean()) for bi in range(len(BUD))] for mi_ in range(len(MARG))])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for mi_, (marg, c) in enumerate(zip(MARG, ("#999999", "#56B4E9", "#0072B2"))):
        ax.plot(BUD, 100 * viol[mi_], marker="o", color=c, label=f"margin {int(100 * marg)}%")
    ax.set_xlabel("Burden budget"); ax.set_ylabel("Replicates exceeding the budget (%)"); ax.legend(fontsize=7)
    fig.savefig(f"{d}/fig_violation.png"); plt.close(fig)
    out = dict(LAMS=LAMS, benefit=B.mean(1), burden=C.mean(1), failure=F.mean(1),
               benefit_ci=[ci95(B[i]) for i in range(len(LAMS))], burden_ci=[ci95(C[i]) for i in range(len(LAMS))],
               fixed=fx, BUD=BUD, MARG=MARG,
               realised=Cb.mean(2), realised_ci=[[ci95(Cb[m_, b_]) for b_ in range(len(BUD))] for m_ in range(len(MARG))],
               predicted=Pred.mean(2), benefit_b=Bb.mean(2), failure_b=Fb.mean(2), lam_b=Lm.mean(2), violation=viol,
               standard=summarize(res))
    save_json(I, "results", dict(out, extra=drop_models(res)))
    return out


# =============================================================================
def idea7():
    I = 7; d = odir(I)
    dom = Clinic()
    res = standard_benchmark(dom, n_eps=150, n_oracle=1500, steps=150000, seeds=5)
    standard_figs(res, dom, I, title="Clinic overbooking MDP")
    T = 150000
    qopt = 14.0 * (1 - dom.gamma ** 40) / (1 - dom.gamma)
    SCH = {"eps=0.1 const": (lambda k, t: 0.1, 0.0), "eps exp. decay": (eps_exp, 0.0),
           "GLIE 1/k": (lambda k, t: max(0.02, (t / 50.0) / (t / 50.0 + k)), 0.0),
           "Optimistic init": (lambda k, t: 0.02, 0.5 * qopt), "Near-greedy (0.02)": (lambda k, t: 0.02, 0.0)}
    SEEDS = 6
    d_or = rollout(dom, dom.behavior, np.random.default_rng(9999), 1500); m_or = build_mdp(d_or, dom)
    pi_or, _, _ = value_iteration(m_or); J_opt = J(m_or, pi_or)
    J_beh = J(m_or, behavior_mode(d_or, dom))
    sh = {"nominal": Clinic(0.0), "no-show +0.08": Clinic(0.08), "no-show -0.05": Clinic(-0.05)}
    CUR = {a: {s: [] for s in SCH} for a in ("Q-learning", "Monte Carlo")}
    FIN = {a: {s: {k: [] for k in sh} for s in SCH} for a in CUR}
    for sd in range(SEEDS):
        data = rollout(dom, dom.behavior, np.random.default_rng(700 + sd), 150)
        m = build_mdp(data, dom); bm = behavior_mode(data, dom)
        cf = (lambda p, m=m, bm=bm: J(m_or, fill_policy(p, m, bm)))
        for sn, (ef, q0) in SCH.items():
            for an, fn in (("Q-learning", q_learning), ("Monte Carlo", mc_control)):
                p, _, c = fn(m, np.random.default_rng([sd, list(SCH).index(sn), 1]), T, eps_fn=ef, q0=q0, ckpt_fn=cf)
                CUR[an][sn].append(c.tolist())
                p = fill_policy(p, m, bm)
                for k, sim in sh.items():
                    r, _ = eval_policy(sim, p, n_eps=120)
                    FIN[an][sn][k].append(float(r.mean()))
        print("idea7 seed", sd, flush=True)
    t90 = {a: {} for a in CUR}; auc = {a: {} for a in CUR}; fin = {a: {} for a in CUR}
    for a in CUR:
        for sn in SCH:
            cs = CUR[a][sn]
            tgt = J_beh + 0.9 * (J_opt - J_beh)
            t90[a][sn] = steps_to_fraction(cs, tgt)
            auc[a][sn] = [float(np.mean([(v[1] - J_beh) / (J_opt - J_beh) for v in c])) for c in cs]
            fin[a][sn] = [float((c[-1][1] - J_beh) / (J_opt - J_beh)) for c in cs]
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, a in zip(axs, CUR):
        for i, sn in enumerate(SCH):
            arr = np.array(CUR[a][sn]); x = arr[0, :, 0]
            y = (arr[:, :, 1] - J_beh) / (J_opt - J_beh)
            ax.plot(x, y.mean(0), color=OKABE[i], label=sn, lw=1.6)
            ax.fill_between(x, y.mean(0) - y.std(0, ddof=1), y.mean(0) + y.std(0, ddof=1), color=OKABE[i], alpha=0.12)
        ax.axhline(1, color="k", ls="--", lw=0.8); ax.axhline(0, color="#777777", ls=":", lw=0.8)
        ax.set_title(a, fontsize=9); ax.set_xlabel("Environment steps")
    axs[0].set_ylabel("Normalised value (0 = clinic behaviour, 1 = DP optimum)"); axs[0].legend(fontsize=7)
    fig.savefig(f"{d}/fig_schedules.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for ai, a in enumerate(CUR):
        v = [np.nanmean(t90[a][s]) / 1000 for s in SCH]
        axs[0].bar(np.arange(len(SCH)) + (ai - 0.5) * 0.38, v, 0.38, color=COL[a], label=a)
        axs[1].bar(np.arange(len(SCH)) + (ai - 0.5) * 0.38, [np.mean(auc[a][s]) for s in SCH], 0.38, yerr=[ci95(auc[a][s]) for s in SCH], color=COL[a], capsize=2)
    for ax in axs: ax.set_xticks(range(len(SCH))); ax.set_xticklabels(list(SCH), rotation=25, ha="right", fontsize=7)
    axs[0].set_ylabel("Steps (thousands) to reach 90% of optimum\n(NaN-safe mean over seeds that reach it)"); axs[1].set_ylabel("Area under learning curve (normalised)")
    axs[0].legend(fontsize=7); fig.savefig(f"{d}/fig_efficiency.png"); plt.close(fig)
    # shift robustness (best schedule per algorithm = eps exp)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    dp_pol = fill_policy(value_iteration(build_mdp(rollout(dom, dom.behavior, np.random.default_rng(700), 150), dom))[0],
                         build_mdp(rollout(dom, dom.behavior, np.random.default_rng(700), 150), dom), np.array([dom.rule(s) for s in range(dom.nS)]))
    rule = np.array([dom.rule(s) for s in range(dom.nS)])
    shift_res = {}
    for k, sim in sh.items():
        shift_res[k] = {}
        for nm, p in (("Rule", rule), ("DP", dp_pol)):
            r, _ = eval_policy(sim, p, n_eps=300); shift_res[k][nm] = float(r.mean())
        for a in CUR:
            shift_res[k][a + " (eps exp)"] = float(np.mean(FIN[a]["eps exp. decay"][k]))
    ks = list(sh); w = 0.2
    for i, nm in enumerate(["Rule", "DP", "Q-learning (eps exp)", "Monte Carlo (eps exp)"]):
        ax.bar(np.arange(len(ks)) + (i - 1.5) * w, [shift_res[k][nm] for k in ks], w, label=nm, color=[COL["Rule"], COL["DP"], COL["Q-learning"], COL["Monte Carlo"]][i])
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks); ax.set_ylabel("True return"); ax.legend(fontsize=7)
    lo = min(v for k in ks for v in shift_res[k].values()); ax.set_ylim(lo * 0.9, None)
    fig.savefig(f"{d}/fig_shift.png"); plt.close(fig)
    out = dict(schedules=list(SCH), J_opt=J_opt, J_beh=J_beh,
               t90={a: {s: [None if np.isnan(x) else float(x) for x in t90[a][s]] for s in SCH} for a in CUR},
               auc={a: {s: dict(mean=float(np.mean(auc[a][s])), ci=ci95(auc[a][s])) for s in SCH} for a in CUR},
               final={a: {s: dict(mean=float(np.mean(fin[a][s])), ci=ci95(fin[a][s])) for s in SCH} for a in CUR},
               shift={a: {s: {k: float(np.mean(FIN[a][s][k])) for k in sh} for s in SCH} for a in CUR}, shift_ref=shift_res,
               reach_rate={a: {s: float(np.mean(~np.isnan(t90[a][s]))) for s in SCH} for a in CUR}, standard=summarize(res))
    save_json(I, "results", dict(out, extra=drop_models(res)))
    return out


# =============================================================================
def resample(data, rng):
    ids = np.unique(data["ep"]); pick = rng.choice(ids, len(ids), replace=True)
    order = np.argsort(data["ep"], kind="stable"); ep_sorted = data["ep"][order]
    starts = np.searchsorted(ep_sorted, ids); ends = np.searchsorted(ep_sorted, ids, side="right")
    idx = np.concatenate([order[starts[i]:ends[i]] for i in np.searchsorted(ids, pick)])
    return {k: v[idx] for k, v in data.items()}


def idea8():
    I = 8; d = odir(I)
    dom = Discharge()
    res = standard_benchmark(dom, n_eps=1000, n_oracle=8000, steps=150000, seeds=8, n_eval=1500)
    standard_figs(res, dom, I, title="Discharge MDP (training distribution)")
    SC = [0.8, 1.0, 1.25, 1.5, 2.0]; CS = [0.5, 1, 2, 4]; SEEDS = 8
    names = ["Rule", "DP (nominal)"] + [f"Robust DP (c={c})" for c in CS] + ["Monte Carlo", "Q-learning"]
    Jm = {n: np.zeros((len(SC), SEEDS)) for n in names}
    rd = {n: np.zeros(SEEDS) for n in names}; fu = {n: np.zeros(SEEDS) for n in names}
    rule = np.array([dom.rule(s) for s in range(dom.nS)])
    sims = [Discharge(s) for s in SC]
    for sd in range(SEEDS):
        data = rollout(dom, dom.behavior, np.random.default_rng(900 + sd), 1000)
        m = build_mdp(data, dom); bm = behavior_mode(data, dom)
        pols = {"Rule": rule}
        p, _, _ = value_iteration(m); pols["DP (nominal)"] = fill_policy(p, m, bm)
        for c in CS:
            p, _, _ = robust_value_iteration(m, c=c); pols[f"Robust DP (c={c})"] = fill_policy(p, m, bm)
        cf = None
        p, _, _ = mc_control(m, np.random.default_rng(1 + sd), 120000); pols["Monte Carlo"] = fill_policy(p, m, bm)
        p, _, _ = q_learning(m, np.random.default_rng(2 + sd), 120000); pols["Q-learning"] = fill_policy(p, m, bm)
        for n in names:
            for si, sim in enumerate(sims):
                r, inf = eval_policy(sim, pols[n], n_eps=800)
                Jm[n][si, sd] = r.mean()
                if SC[si] == 1.0:
                    rd[n][sd] = inf["readmit"].mean(); fu[n][sd] = inf["followup"].mean()
        print("idea8 seed", sd, flush=True)
    # bootstrap stability on seed-0 data
    data = rollout(dom, dom.behavior, np.random.default_rng(900), 1000)
    m0 = build_mdp(data, dom); vis = m0.visits >= 5
    p_nom, _, _ = value_iteration(m0); p_rob, _, _ = robust_value_iteration(m0, c=2)
    ag_n, ag_r = [], []
    for b in range(60):
        mb = build_mdp(resample(data, np.random.default_rng(b)), dom)
        pn, _, _ = value_iteration(mb); pr, _, _ = robust_value_iteration(mb, c=2)
        ag_n.append(float((pn[vis] == p_nom[vis]).mean())); ag_r.append(float((pr[vis] == p_rob[vis]).mean()))
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    show = ["Rule", "DP (nominal)", "Robust DP (c=1)", "Robust DP (c=2)", "Robust DP (c=4)", "Monte Carlo", "Q-learning"]
    cc = {"Rule": "#555555", "DP (nominal)": "#0072B2", "Robust DP (c=1)": "#CC79A7", "Robust DP (c=2)": "#E69F00", "Robust DP (c=4)": "#8c564b", "Monte Carlo": "#D55E00", "Q-learning": "#009E73"}
    for n in show:
        ax.errorbar(SC, Jm[n].mean(1), yerr=[ci95(Jm[n][i]) for i in range(len(SC))], marker="o", ms=3, capsize=2, color=cc[n], label=n)
    ax.set_xlabel("Readmission-risk multiplier in deployment (1.0 = training)"); ax.set_ylabel("True discounted return"); ax.legend(fontsize=7)
    fig.savefig(f"{d}/fig_shift.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    worst = {n: Jm[n].min(0) for n in names}
    axs[0].bar(range(len(show)), [worst[n].mean() for n in show], yerr=[ci95(worst[n]) for n in show], color=[cc[n] for n in show], capsize=2)
    axs[0].set_xticks(range(len(show))); axs[0].set_xticklabels([s.replace(" (", "\n(").replace(" DP", "\nDP") for s in show], fontsize=6)
    axs[0].set_ylabel("Worst-case return over deployment shifts")
    axs[1].boxplot([ag_n, ag_r]); axs[1].set_xticklabels(["Nominal DP", "Robust DP (c=2)"]); axs[1].set_ylabel("Bootstrap policy agreement\nwith full-data policy")
    fig.savefig(f"{d}/fig_stability.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    xs = np.arange(len(show)); w = 0.38
    ax.bar(xs - w / 2, [100 * rd[n].mean() for n in show], w, color="#D55E00", label="Readmission rate (%)")
    ax.bar(xs + w / 2, [100 * fu[n].mean() for n in show], w, color="#0072B2", label="Follow-up used (%)")
    ax.set_xticks(xs); ax.set_xticklabels([s.replace(" (", "\n(").replace(" DP", "\nDP") for s in show], fontsize=6); ax.legend(fontsize=7)
    fig.savefig(f"{d}/fig_followup.png"); plt.close(fig)
    # ---- when (if ever) does robustification pay?  sweep the amount of logged data
    NS = [200, 400, 1000, 3000]; CSW = [1, 2]; sw = {}
    for n_ in NS:
        acc = {k: {sc: [] for sc in (1.0, 1.5)} for k in ["nominal"] + [f"robust c={c}" for c in CSW]}
        for sd in range(8):
            dat = rollout(dom, dom.behavior, np.random.default_rng(5000 + sd), n_)
            m_ = build_mdp(dat, dom); bm_ = behavior_mode(dat, dom)
            pp = {"nominal": fill_policy(value_iteration(m_)[0], m_, bm_)}
            for c in CSW: pp[f"robust c={c}"] = fill_policy(robust_value_iteration(m_, c=c)[0], m_, bm_)
            for k, p in pp.items():
                for sc in (1.0, 1.5):
                    acc[k][sc].append(float(eval_policy(Discharge(sc), p, n_eps=800)[0].mean()))
        sw[n_] = {k: {str(sc): float(np.mean(v)) for sc, v in d_.items()} for k, d_ in acc.items()}
        sw[n_]["diff_c2_1.5"] = paired(acc["robust c=2"][1.5], acc["nominal"][1.5])
        sw[n_]["diff_c2_1.0"] = paired(acc["robust c=2"][1.0], acc["nominal"][1.0])
        print("idea8 sweep", n_, sw[n_]["diff_c2_1.5"], flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, key, tl in ((axs[0], "diff_c2_1.0", "No shift (training risk)"), (axs[1], "diff_c2_1.5", "Deployment risk x1.5")):
        ax.errorbar(range(len(NS)), [sw[n_][key]["mean"] for n_ in NS], yerr=[sw[n_][key]["ci"] for n_ in NS], marker="o", color="#E69F00", capsize=3)
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(NS))); ax.set_xticklabels(NS); ax.set_xlabel("Logged episodes"); ax.set_title(tl, fontsize=9)
    axs[0].set_ylabel("Return: robust (c=2) minus nominal DP")
    fig.savefig(f"{d}/fig_datasweep.png"); plt.close(fig)
    out = dict(SC=SC, names=names, J={n: dict(mean=Jm[n].mean(1), ci=[ci95(Jm[n][i]) for i in range(len(SC))]) for n in names},
               worst={n: dict(mean=float(worst[n].mean()), ci=ci95(worst[n])) for n in names},
               readmit={n: float(rd[n].mean()) for n in names}, followup={n: float(fu[n].mean()) for n in names},
               sweep={str(k): v for k, v in sw.items()}, boot=dict(nominal=float(np.mean(ag_n)), robust=float(np.mean(ag_r)), nominal_sd=float(np.std(ag_n)), robust_sd=float(np.std(ag_r))),
               rob2_vs_nom_worst=paired(worst["Robust DP (c=2)"], worst["DP (nominal)"]),
               rob2_vs_nom_nominal=paired(Jm["Robust DP (c=2)"][1], Jm["DP (nominal)"][1]),
               rob2_vs_nom_at2=paired(Jm["Robust DP (c=2)"][4], Jm["DP (nominal)"][4]),
               standard=summarize(res))
    save_json(I, "results", dict(out, extra=drop_models(res)))
    return out


# =============================================================================
def idea9():
    I = 9; d = odir(I)
    HS = [6, 12, 24]; R = {}
    for H in HS:
        for sp in (False, True):
            dom = Rehab(sparse=sp, H=H)
            r = standard_benchmark(dom, n_eps=800, n_oracle=8000, steps=250000, seeds=5, n_eval=250)
            R[(H, sp)] = (r, dom)
            if H == 12:
                standard_figs(r, dom, I, tag="_sparse" if sp else "_dense", title=f"Rehabilitation, H={H}, {'sparse' if sp else 'dense'} reward")
    summ = {}
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, sp in zip(axs, (False, True)):
        r, _ = R[(12, sp)]
        beh = float(np.mean(r["methods"]["Behavior"]["Jor"])); opt = r["J_oracle_model"]["Oracle DP"]
        for nm in ("Monte Carlo", "Q-learning"):
            x, mu, sd = curve_mean(r["curves"][nm]); y = (mu - beh) / (opt - beh); s_ = sd / (opt - beh)
            ax.plot(x, y, color=COL[nm], label=nm, lw=1.8); ax.fill_between(x, y - s_, y + s_, color=COL[nm], alpha=0.15)
        ax.axhline(1, color="k", ls="--", lw=0.8); ax.set_title("Sparse (terminal) reward" if sp else "Dense (weekly) reward", fontsize=9)
        ax.set_xlabel("Environment steps")
    axs[0].set_ylabel("Normalised value (0 = therapist behaviour, 1 = DP optimum)"); axs[0].legend(fontsize=7)
    fig.savefig(f"{d}/fig_curves_h12.png"); plt.close(fig)
    t90 = {}; fin = {}
    for (H, sp), (r, dom) in R.items():
        beh = float(np.mean(r["methods"]["Behavior"]["Jor"])); opt = r["J_oracle_model"]["Oracle DP"]
        for nm in ("Monte Carlo", "Q-learning"):
            t90[(H, sp, nm)] = steps_to_fraction(r["curves"][nm], beh + 0.9 * (opt - beh))
            fin[(H, sp, nm)] = [float((c[-1][1] - beh) / (opt - beh)) for c in r["curves"][nm]]
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, sp in zip(axs, (False, True)):
        for nm in ("Monte Carlo", "Q-learning"):
            ax.errorbar(HS, [np.mean(fin[(H, sp, nm)]) for H in HS], yerr=[ci95(fin[(H, sp, nm)]) for H in HS], marker="o", color=COL[nm], capsize=2, label=nm)
        ax.set_xticks(HS); ax.set_xlabel("Treatment horizon H (weeks)"); ax.set_title("Sparse reward" if sp else "Dense reward", fontsize=9)
    axs[0].set_ylabel("Normalised value after 250k steps"); axs[0].legend(fontsize=7)
    fig.savefig(f"{d}/fig_horizon.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    lab = [f"H={H}\n{'sparse' if sp else 'dense'}" for H in HS for sp in (False, True)]
    for i, nm in enumerate(("Monte Carlo", "Q-learning")):
        ax.bar(np.arange(len(lab)) + (i - 0.5) * 0.38, [np.nanmean(t90[(H, sp, nm)]) / 1000 if np.isfinite(np.nanmean(t90[(H, sp, nm)])) else 0 for H in HS for sp in (False, True)], 0.38, color=COL[nm], label=nm)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel("Steps (thousands) to 90% of optimum\n(0 = never reached)"); ax.legend(fontsize=7)
    fig.savefig(f"{d}/fig_steps90.png"); plt.close(fig)
    for (H, sp), (r, dom) in R.items():
        beh = float(np.mean(r["methods"]["Behavior"]["Jor"])); opt = r["J_oracle_model"]["Oracle DP"]
        summ[f"H{H}_{'sparse' if sp else 'dense'}"] = dict(
            summary=summarize(r), t90={nm: [None if np.isnan(x) else float(x) for x in t90[(H, sp, nm)]] for nm in ("Monte Carlo", "Q-learning")},
            reach={nm: float(np.mean(~np.isnan(t90[(H, sp, nm)]))) for nm in ("Monte Carlo", "Q-learning")},
            final={nm: dict(mean=float(np.mean(fin[(H, sp, nm)])), ci=ci95(fin[(H, sp, nm)])) for nm in ("Monte Carlo", "Q-learning")},
            Jopt=opt, Jbeh=beh, mc_minus_q=paired(fin[(H, sp, "Monte Carlo")], fin[(H, sp, "Q-learning")]))
    save_json(I, "results", dict(runs=summ, HS=HS))
    return summ


# =============================================================================
def traj_means(dom, pi, key, n=200):
    out = np.zeros(dom.max_steps); cnt = np.zeros(dom.max_steps)
    for e in range(n):
        rng = np.random.default_rng([4242, e]); s = dom.reset(rng)
        for t in range(dom.max_steps):
            s, r, done, info, dur = dom.step(int(pi[s]), rng)
            out[t] += info[key]; cnt[t] += 1
            if done: break
    return out / np.maximum(cnt, 1)


def idea10():
    I = 10; d = odir(I)
    dm, d0 = Nudge(True), Nudge(False)
    rm = standard_benchmark(dm, n_eps=300, n_oracle=3000, steps=150000, seeds=6)
    r0 = standard_benchmark(d0, n_eps=300, n_oracle=3000, steps=150000, seeds=6)
    standard_figs(rm, dm, I, tag="_memory", title="Fatigue-aware (memory) state")
    standard_figs(r0, d0, I, tag="_memoryless", title="Memoryless state")
    meth = ["Rule", "DP", "Monte Carlo", "Q-learning"]
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for ax, (key, yl) in zip(axs, (("J", "True return"), ("health", "Health score summed over 28 days"), ("nudges", "Nudge-days / 28 days"))):
        for i, (rr, lab, c) in enumerate(((r0, "memoryless state", "#999999"), (rm, "fatigue-aware state", "#0072B2"))):
            vals = [np.mean(rr["methods"][m]["J"] if key == "J" else rr["methods"][m]["info"][key]) for m in meth]
            errs = [ci95(rr["methods"][m]["J"] if key == "J" else rr["methods"][m]["info"][key]) for m in meth]
            ax.errorbar(np.arange(len(meth)) + (i - 0.5) * 0.18, vals, yerr=errs, fmt="o", color=c, capsize=3, ms=6, label=lab)
        ax.set_xticks(range(len(meth))); ax.set_xticklabels([m.replace(" ", "\n") for m in meth], fontsize=7); ax.set_ylabel(yl)
        ax.set_xlim(-0.5, len(meth) - 0.5)
    axs[0].legend(fontsize=7, loc="lower left"); fig.savefig(f"{d}/fig_memory_compare.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for key, ax, yl in (("nudges", axs[0], "Nudge probability on day t"), ("health", axs[1], "Mean health score on day t")):
        for dom_, rr, lab, c, ls_ in ((d0, r0, "DP memoryless", "#999999", "-"), (dm, rm, "DP fatigue-aware", "#0072B2", "-")):
            ax.plot(range(1, 29), traj_means(dom_, rr["policies"]["DP"], key), color=c, ls=ls_, label=lab)
        ax.plot(range(1, 29), traj_means(dm, [dm.rule(s) for s in range(dm.nS)], key), color="#555555", ls=":", label="Rule")
        ax.set_xlabel("Day"); ax.set_ylabel(yl)
    axs[0].legend(fontsize=7); fig.savefig(f"{d}/fig_trajectories.png"); plt.close(fig)
    pol = np.array(rm["policies"]["Oracle DP"]); vis = rm["m_or"].visits > 0
    hb = np.arange(dm.nS) % 4
    rate = [float((pol[(hb == h) & vis] > 0).mean()) if ((hb == h) & vis).any() else np.nan for h in range(4)]
    mix = np.zeros((4, 4))
    for h in range(4):
        sel = (hb == h) & vis
        for a in range(4): mix[h, a] = float((pol[sel] == a).mean()) if sel.any() else np.nan
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    im = ax.imshow(mix, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    ax.set_xticks(range(4)); ax.set_xticklabels(dm.action_labels); ax.set_yticks(range(4)); ax.set_yticklabels(["h0 (fresh)", "h1", "h2", "h3 (habituated)"])
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{mix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_ylabel("Habituation band"); fig.colorbar(im, fraction=0.046)
    fig.savefig(f"{d}/fig_habituation_policy.png"); plt.close(fig)
    out = dict(memory=summarize(rm), memoryless=summarize(r0), nudge_rate_by_habituation=rate, action_mix_by_habituation=mix,
               mem_vs_less_dp=paired(rm["methods"]["DP"]["J"], r0["methods"]["DP"]["J"]),
               mem_vs_less_q=paired(rm["methods"]["Q-learning"]["J"], r0["methods"]["Q-learning"]["J"]),
               mem_vs_less_mc=paired(rm["methods"]["Monte Carlo"]["J"], r0["methods"]["Monte Carlo"]["J"]),
               nudges_dp=paired(r0["methods"]["DP"]["info"]["nudges"], rm["methods"]["DP"]["info"]["nudges"]),
               states=dict(memory=dm.nS, memoryless=d0.nS))
    save_json(I, "results", dict(out, mem_full=drop_models(rm), less_full=drop_models(r0)))
    return out


if __name__ == "__main__":
    k = int(sys.argv[1])
    t = time.time()
    globals()[f"idea{k}"]()
    print(f"idea{k} done in {time.time() - t:.0f}s", flush=True)
