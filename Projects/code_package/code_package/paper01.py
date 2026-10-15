import numpy as np
from scipy import stats
from docx_helper import Paper
from paper_common import *

I = 1


def build():
    r = load(I); S = r["standard"]; T = r["twin"]; E = r["extra"]
    SIG, NDS, LAMS = r["sigmas"], r["nds"], r["lams"]
    sk = lambda sg: str(sg)
    Jt = lambda sg, n, nd: T[sk(sg)][n][str(nd)]["J"]
    Ct = lambda sg, n, nd: T[sk(sg)][n][str(nd)]["ci"]
    Xt = lambda sg, n, nd, k: T[sk(sg)][n][str(nd)][k]
    ceil = r["ceiling_vs_pooled"]; ind = r["indiv_vs_pooled"]; tvp = r["twin_vs_pooled"]; tvi = r["twin_vs_indiv"]
    by_nd = r["twin_vs_pooled_by_nd"]
    pd_ = lambda d: f"{d['mean']:+.2f} ± {d['ci']:.2f} ({pstr(d['p'])})"
    ND7 = 7

    # ---- derived quantities (all from results.json)
    meth = E["methods"]
    rule_J = S["Rule"]["J"]
    def paired_vs_rule(k):
        d = np.array(meth[k]["J"]) - meth["Rule"]["J"][0]
        return float(d.mean()), 1.96 * float(d.std(ddof=1) / np.sqrt(len(d))), float(stats.ttest_1samp(d, 0).pvalue)
    pv = {k: paired_vs_rule(k) for k in ("DP", "Monte Carlo", "Q-learning")}
    dp_J = np.array(meth["DP"]["J"])
    n_dp_worse = int((dp_J < rule_J - 1e-6).sum()); n_dp_equal = int((np.abs(dp_J - rule_J) <= 1e-6).sum()); n_dp_better = int((dp_J > rule_J + 1e-6).sum())
    seeds = len(dp_J)
    gap = S["Oracle DP"]["J"] - rule_J
    gap_model = E["J_oracle_model"]["Oracle DP"] - E["J_oracle_model"]["Rule"]
    pol = E["policies"]
    n_states = E["n_states"]
    n_or_diff = int(np.sum(np.array(pol["Oracle DP"]) != np.array(pol["Rule"])))
    n_dp_diff = int(np.sum(np.array(pol["DP"]) != np.array(pol["Rule"])))
    n_mc_diff = int(np.sum(np.array(pol["Monte Carlo"]) != np.array(pol["Rule"])))
    n_q_diff = int(np.sum(np.array(pol["Q-learning"]) != np.array(pol["Rule"])))
    n_or_3u = int(np.sum(np.array(pol["Oracle DP"]) == 3)); n_rule_3u = int(np.sum(np.array(pol["Rule"]) == 3))
    agQ, agM = 100 * np.mean(E["agree"]["Q-learning"]), 100 * np.mean(E["agree"]["Monte Carlo"])
    og = np.array(E["optimism_gap"]); n_og_pos = int((og > 0).sum())
    cq = np.array(E["curves"]["Q-learning"]); cm = np.array(E["curves"]["Monte Carlo"])
    q_first = float(cq[:, 1, 1].mean()); m_first = float(cm[:, 1, 1].mean()); step_first = int(cq[0, 1, 0])
    q_last = float(cq[:, -10:, 1].mean()); q_last_sd = float(cq[:, -10:, 1].std())
    m_last = float(cm[:, -1, 1].mean())
    J_or_m, J_rule_m = E["J_oracle_model"]["Oracle DP"], E["J_oracle_model"]["Rule"]
    cfg = E["config"]

    pooled_minus_rule = {sg: Jt(sg, "Pooled", ND7) - Jt(sg, "Rule", ND7) for sg in SIG}
    ceil_range = (min(c["mean"] for c in ceil.values()), max(c["mean"] for c in ceil.values()))
    ind_range = (min(c["mean"] for c in ind.values()), max(c["mean"] for c in ind.values()))
    cells = [(l, sg) for l in LAMS for sg in SIG]
    n_pos_sig = sum(1 for l, sg in cells if tvp[str(l)][sk(sg)]["p"] < 0.05 and tvp[str(l)][sk(sg)]["mean"] > 0)
    n_neg_sig = sum(1 for l, sg in cells if tvp[str(l)][sk(sg)]["p"] < 0.05 and tvp[str(l)][sk(sg)]["mean"] < 0)
    n_neg_mean = sum(1 for l, sg in cells if tvp[str(l)][sk(sg)]["mean"] < 0)
    allc = [(l, sg, nd) for l in LAMS for sg in SIG for nd in NDS]
    a_pos = sum(1 for l, sg, nd in allc if by_nd[str(l)][sk(sg)][str(nd)]["p"] < 0.05 and by_nd[str(l)][sk(sg)][str(nd)]["mean"] > 0)
    a_neg = sum(1 for l, sg, nd in allc if by_nd[str(l)][sk(sg)][str(nd)]["p"] < 0.05 and by_nd[str(l)][sk(sg)][str(nd)]["mean"] < 0)
    a_negmean = sum(1 for l, sg, nd in allc if by_nd[str(l)][sk(sg)][str(nd)]["mean"] < 0)
    n_all = len(allc)
    exp_false = 0.05 * n_all
    sig_cells = [(l, sg) for l, sg in cells if tvp[str(l)][sk(sg)]["p"] < 0.05 and tvp[str(l)][sk(sg)]["mean"] > 0]
    best_gain = max(cells, key=lambda c: tvp[str(c[0])][sk(c[1])]["mean"])
    bg = tvp[str(best_gain[0])][sk(best_gain[1])]
    frac_best = bg["mean"] / ceil[sk(best_gain[1])]["mean"]
    best_lam = {sg: max(LAMS, key=lambda l: Jt(sg, f"Twin (lam={l})", ND7)) for sg in SIG}
    tw_ind_min = min(tvi[str(l)][sk(sg)]["mean"] for l in LAMS for sg in SIG)
    tw_ind_max = max(tvi[str(l)][sk(sg)]["mean"] for l in LAMS for sg in SIG)
    tw_ind_pmax = max(tvi[str(l)][sk(sg)]["p"] for l in LAMS for sg in SIG)
    frac_oracle = r["frac_oracle_beats_pooled"]
    spr = r["spearman_atyp_ceiling"]
    ts = np.array(r["test_sens"]["0.8"])
    n_lo_clip = int((ts <= 0.35 + 1e-9).sum()); n_hi_clip = int((ts >= 2.5 - 1e-9).sum())
    # individual vs rule coincidence
    ind_eq_rule = {sg: [nd for nd in NDS if abs(Jt(sg, "Individual", nd) - Jt(sg, "Rule", nd)) < 1e-9] for sg in SIG}
    n_coinc = sum(len(v) for v in ind_eq_rule.values())
    # hypoglycaemia at 28 days
    hyp = lambda sg, n, nd: Xt(sg, n, nd, "hypo")
    ratios = [ceil[sk(sg)]["mean"] / Jt(sg, "Pooled", ND7) for sg in SIG]
    n_not_better = {sg: int(round(ceil[sk(sg)]["n"] * (1 - frac_oracle[sk(sg)]))) for sg in SIG}
    hyp_rise = all(Xt(sg, f"Twin (lam={l})", nd, "hypo") > Xt(sg, "Pooled", nd, "hypo") for sg in SIG for l in LAMS for nd in (14, 28))
    assert hyp_rise
    lam10_lowest = sum(1 for sg in SIG for nd in NDS if Jt(sg, "Twin (lam=10)", nd) < min(Jt(sg, "Twin (lam=30)", nd), Jt(sg, "Twin (lam=100)", nd)))
    n_lam_cells = len(SIG) * len(NDS)
    tw_vs_rule7 = {sg: Jt(sg, "Twin (lam=30)", ND7) - Jt(sg, "Rule", ND7) for sg in SIG}
    SG = SIG[1]; SG_hi = SIG[-1]; SG_lo = SIG[0]; SG_mid = SIG[2]

    P = Paper("Digital-Twin Insulin Dosing: Shrinking a Patient-Specific Markov Model Toward a Population Model, and How Much Personalisation Can Be Learned from a Few Days of Glucose Data",
              "type 1 diabetes; continuous glucose monitoring; digital twin; insulin dosing; Markov decision process; empirical Bayes; dynamic programming; Q-learning",
              running="Digital-twin insulin dosing", refdict=REFS)

    P.abstract(
        "Insulin sensitivity differs between people, so a dosing policy learned from a population may be wrong for an individual, while a model learned from one "
        "person's first few days of continuous glucose monitoring (CGM) data is too thin to trust. We study this trade-off in a Markov decision process (MDP) "
        "framework on an hourly glucose simulator whose columns mirror Kaggle CGM tables (the Kaggle site was unreachable, HTTP 403, so no real file was used). "
        "We define a lightweight digital twin as a patient-specific MDP whose transition and reward counts are shrunk toward a population model by λ pseudo-counts, and we compare "
        "it with a pooled population policy, an individual-only policy, the guideline-like rule and an oracle twin fitted to 2,000 days of the patient's own data. "
        f"On a single simulated patient, dynamic programming (DP), Monte Carlo control and Q-learning from {cfg['n_eps']} days of logs do not beat the logged rule "
        f"(DP {S['DP']['J']:.2f}, Monte Carlo {S['Monte Carlo']['J']:.2f}, Q-learning {S['Q-learning']['J']:.2f}, rule {S['Rule']['J']:.2f}), although an oracle fitted to {cfg['n_oracle']:,} days reaches {S['Oracle DP']['J']:.2f}. "
        f"Across {len(SIG)} levels of between-patient heterogeneity the oracle twin beats the pooled policy by {ceil_range[0]:.2f} to {ceil_range[1]:.2f} return units (all {pstr(max(c['p'] for c in ceil.values()))}), "
        f"so a personalisation ceiling exists. Individual-only models fitted to 7 days are worse than pooled by {-ind_range[1]:.2f} to {-ind_range[0]:.2f}. "
        f"The shrunk twin removes that loss (it beats individual-only at 7 days by {tw_ind_min:.2f} to {tw_ind_max:.2f}) but realises little of the ceiling: at 7 days it differs significantly from the pooled policy in "
        f"{n_pos_sig} of {len(cells)} (λ, heterogeneity) cells (best gain {bg['mean']:+.2f}, {frac_best * 100:.0f}% of that cell's ceiling) and is not significantly better elsewhere. "
        f"Hypoglycaemic time is higher than for the pooled policy in every twin fitted to 14 or 28 days. These are simulation results and make no clinical claim.")

    P.h1("Introduction")
    P.p("People with type 1 diabetes must dose insulin several times a day, trading the risk of hyperglycaemia against the more immediate danger of hypoglycaemia. "
        "Closed-loop controllers have long used model-based control, from physiological glucose-insulin models [@bergman] to nonlinear model predictive control [@hovorka], "
        "and in silico simulators now serve as preclinical test beds [@kovatchev]. More recently, reinforcement learning (RL) has been proposed for closed-loop glucose control [@fox]. "
        "A recurrent obstacle is inter-individual variability: the same dose lowers glucose by very different amounts in different people.")
    P.p("The natural remedy is personalisation, often described as a digital twin: fit a model to the individual and plan on it. The difficulty is statistical. "
        "A new patient has a few days of data, whereas a tabular MDP over 36 states and four actions has 144 state-action cells. "
        "Learning from the patient alone is noisy, while borrowing strength from a population is biased if the patient is atypical. The classical statistical answer is partial pooling, or "
        "empirical-Bayes shrinkage [@efron,gelman]; the offline RL answer is conservatism where data are thin [@levine,jin,kumar]. This paper asks how far a tabular, interpretable version of "
        "this idea can go, and measures the answer against exact simulator ground truth.")
    P.p("**Contributions.** (i) A transparent hourly glucose simulator with a single patient-specific parameter, insulin sensitivity, whose heterogeneity across patients is controlled by a log-standard-deviation. "
        "(ii) A pseudo-count shrinkage estimator for a patient-specific MDP, with a short bias-variance argument for how strongly it should shrink (Eq. 3 to 5). "
        "(iii) A personalisation-ceiling analysis, an oracle twin fitted to 2,000 days compared with the pooled policy, which separates what could be gained from what is realised. "
        f"(iv) A sweep over heterogeneity ({SIG[0]} to {SIG[-1]}), amount of patient data ({NDS[0]} to {NDS[-1]} days) and shrinkage strength (λ in {LAMS}) with paired tests over {ceil[sk(SIG[0])]['n']} patients per condition. "
        "(v) A report of the null and negative results that the data contain: learning from 150 days does not beat the rule, individual-only twins are worse than the population policy, and shrunk twins capture little of the ceiling and can raise hypoglycaemic time.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached from the build environment (all requests returned HTTP 403), so no real CGM file was analysed. The simulator mirrors the "
        "variables of CGM tables (time, glucose, trend, meal, insulin dose) with made-up dynamics. Findings concern algorithms, not patients, and nothing here is dosing advice.")

    P.h1("Related work")
    P.p("**Model-based glucose control.** Minimal models of glucose-insulin kinetics estimate an insulin-sensitivity parameter per individual [@bergman], and model predictive control built on such models "
        "has been tested clinically [@hovorka]. The UVA/Padova-style simulators allow in silico trials with virtual cohorts [@kovatchev]. Our simulator is far simpler than these, and we do not claim physiological fidelity; "
        "we borrow only the notion that one sensitivity parameter per person drives much of the variability.")
    P.p("**Reinforcement learning for glucose control.** Deep RL agents have been trained on simulated glucose dynamics for closed-loop control [@fox]. Surveys stress that evaluation needs care and that learning from retrospective data is fragile [@yu,gottesman]. We use a tabular MDP so every step is inspectable.")
    P.p("**Shrinkage and hierarchical models.** When groups (patients) share structure, partial pooling estimates each group's parameters as a precision-weighted blend of its own mean and the "
        "population mean [@efron,gelman]. Our twin is the transition-count analogue: a Dirichlet-style prior centred on the population kernel with concentration λ.")
    P.p("**Offline RL and pessimism.** Policies learned from logs are exploited by estimation error in thinly observed cells; constraining to the data's support and penalising uncertainty are standard "
        "remedies [@levine,fujimoto,kumar,jin]. We use a count threshold, which interacts directly with personalisation: a few days of one patient's data support very few cells.")

    background(P)

    P.h1("Problem formulation: an hourly insulin-dosing simulator")
    P.p("Each episode is one simulated day of 24 hourly decisions. At each hour the agent chooses an insulin dose of 0, 1, 2 or 3 units. Glucose relaxes toward a target "
        "that falls linearly with the dose and with the patient's insulin sensitivity (sens). Specifically g_{t+1} = g_t + 0.35 (180 − 40 · a · sens − g_t) + meal_t + ε_t, where the meal term is "
        "55 mg/dL times a uniform factor in [0.7, 1.3] at hours 7, 13 and 19 and zero otherwise, ε_t is Gaussian with standard deviation 10 mg/dL, and glucose is clipped to [30, 400] mg/dL. "
        "A patient with sens = 1 has a dose-3 target of 60 mg/dL; one with sens = 2 has a dose-3 target of −60 mg/dL, below the clipping floor, so the same dose is far more dangerous. Initial glucose is drawn from "
        "a normal distribution with mean 120 and standard deviation 20 mg/dL, clipped to [50, 300]. The agent sees only glucose band, short-term trend and a meal flag.")
    P.table([["Element", "Definition"],
             ["State (36)", "glucose band (6): <70, 70-99, 100-139, 140-179, 180-249, ≥250 mg/dL × trend (3): falling (change < −15), flat, rising (> +15) × meal-hour flag (2)"],
             ["Actions (4)", "insulin dose of 0, 1, 2 or 3 units"],
             ["Reward per hour", "glucose <54: −10; 54-69: −4; 70-180: +1; 181-250: −1; >250: −3; minus 0.05 per unit of insulin"],
             ["Discount, horizon", "γ = 0.97; 24 hourly steps"],
             ["Patient parameter", "insulin sensitivity sens (1.0 for the single-patient benchmark); heterogeneity study: sens ~ lognormal(0, σ) clipped to [0.35, 2.5]"],
             ["Behaviour policy", "guideline-like rule (0 u below 100 mg/dL; 1 u at meal hours (else 0 u) for 100-139; 1-2 u for 140-179; 2 u for 180-249; 3 u above 250), ε = 0.25 uniform random doses"],
             ["Dataset mirrored", "Diabetes CGM tables (time, glucose, trend, meal, insulin)"]],
            "The glucose-control MDP.", widths=[3.2, 13.2], font=8.5)
    P.p("The reward encodes the clinical asymmetry: an hour below 54 mg/dL costs ten times what an in-range hour earns, while moderate hyperglycaemia costs one. Because the rule never doses "
        "below 100 mg/dL and escalates with glucose, its own hypoglycaemia is rare, so any headroom for learning must come from doses the rule does not use, which only the ε-exploration reveals. "
        "The ε = 0.25 exploration in the logs is the only reason that non-rule doses are ever observed, and it governs how much any learner can discover from a given number of days.")

    P.h1("Method: shrinkage digital twins")
    P.p("Let N_i(s, a, s') and n_i(s, a) be the transition counts and visit counts from patient i's own logs, and P̂_pop(s' | s, a), R̂_pop(s, a) the population model estimated from other patients' logs (here 40 "
        "patients × 20 days). The shrunk twin with strength λ is")
    P.eq("P̂_i(s' | s, a) = [ N_i(s, a, s') + λ P̂_pop(s' | s, a) ] / [ n_i(s, a) + λ ]", "3")
    P.p("and R̂_i(s, a) = [ Σ r_i + λ R̂_pop(s, a) ] / [ n_i(s, a) + λ ], for every cell the population model supports (λ is set to zero for cells it has not seen). Equation (3) is the posterior mean of a "
        "Dirichlet prior with concentration λ centred on the population kernel. The weight placed on the patient's own data in a cell is w = n_i / (n_i + λ). Individual-only (λ = 0, with the 10-observation support rule) and pooled (λ → ∞) are the limits.")
    P.h2("How strongly should the twin shrink?")
    P.p("Consider one probability with population value p_pop, patient value p_i = p_pop + Δ, and Bernoulli noise variance σ_b² per observation. The estimator p̂ = w p̂_i + (1 − w) p_pop has mean squared error "
        "(ignoring the population estimate's own sampling error)")
    P.eq("MSE(w) = w² σ_b² / n_i + (1 − w)² Δ²,", "4")
    P.p("minimised at w* = Δ² / (Δ² + σ_b² / n_i). Writing w = n_i / (n_i + λ) gives the optimal strength")
    P.eq("λ* = σ_b² / Δ².", "5")
    P.p("The optimal prior strength does not depend on the amount of patient data: it depends on how far patients differ from the population (Δ², which grows with the heterogeneity σ) relative to observation noise, so more heterogeneous populations call for smaller λ. "
        f"Because Δ varies by cell and is unknown, λ is treated here as a single constant swept over {LAMS}; Eq. (5) explains a trend and is not a tuned rule, and the sweep is too coarse to test it.")
    P.p("A second ingredient is how thin cells are treated, and it differs between the estimators. The individual-only model uses the standard 10-observation support rule, so it plans only on cells with at least 10 of the patient's own observations. "
        "In the shrunk twin the 10-observation rule is applied to the population model (which sets λ to zero where the population is unsupported) but is not applied again to the blend: any cell with positive effective count n_i + λ is used, and only cells with no observation at all are completed pessimistically. "
        "With λ ≥ 10 and a supported population cell every cell is therefore usable, which is why shrinkage widens the set of cells on which the twin can plan. The flip side is that cells the population does not support but the patient has "
        "visited once or twice enter the model with raw counts; this is a possible source of overfitting that we return to in Section 8.")
    P.code("""Algorithm 1  Shrinkage digital twin for one patient
 input : population logs (many patients), patient's own logs (nd days), shrinkage lambda, rule pi_rule
 1. build population MDP (P_pop, R_pop) from pooled logs; mark cells seen_pop(s,a) with >= 10 observations
 2. build patient counts N_i(s,a,s'), n_i(s,a), reward sums from the patient's own logs
 3. for each (s,a): lam_sa = lambda if seen_pop(s,a) else 0
        P_i(.|s,a) = (N_i(s,a,.) + lam_sa * P_pop(.|s,a)) / (n_i(s,a) + lam_sa)
        R_i(s,a)   = (sum r_i + lam_sa * R_pop(s,a))      / (n_i(s,a) + lam_sa)
 4. cells with n_i + lam_sa = 0 (no data, no population support) go to an absorbing state with the worst observed reward
 5. pi_i <- value_iteration(P_i, R_i, gamma)
 6. in states never observed (patient or population) use the rule's action
 7. deploy pi_i (advisory); recompute the twin as new days accumulate""")
    P.p(f"Reference policies are: **Rule**, the logged guideline-like rule; **Pooled**, DP on the population model; **Individual**, DP on the patient's own data only (λ = 0); and **Oracle twin**, DP on "
        "2,000 days of the patient's own data, which approximates the best any tabular twin could do and defines the personalisation ceiling.")

    protocol(P, [["Item", "Setting"],
                 ["Simulator", "hourly glucose, 24-step days; patient insulin sensitivity sens"],
                 ["States / actions", f"{n_states} / {E['n_actions']}"],
                 ["Single-patient benchmark", f"sens = 1; {cfg['n_eps']} logged days per replicate; oracle MDP from {cfg['n_oracle']:,} days; {cfg['seeds']} replicates; {cfg['n_eval']} evaluation days"],
                 ["Heterogeneity levels", f"log-sd σ in {SIG}; 40 population patients × 20 days to build the pooled model"],
                 ["Test patients", f"{ceil[sk(SIG[0])]['n']} new patients per σ; 100 evaluation days per policy; common random numbers"],
                 ["Patient data", f"{NDS} days of own logs; oracle twin 2,000 days"],
                 ["Shrinkage λ", f"{LAMS} pseudo-counts"],
                 ["Learners (benchmark)", f"DP (value iteration), first-visit Monte Carlo, Q-learning; {cfg['steps']:,} steps each"],
                 ["Statistics", "paired differences across the 20 patients; 95% CI = 1.96 standard errors; one-sample t-test on differences"]],
             cfg["steps"], cfg["seeds"], cfg["n_eval"])

    P.h1("Results")
    P.h2("Single patient: three solvers versus the logged rule")
    rows = std_rows(S, ("in_range", "hypo", "insulin"), ("In-range h/day", "Hypo h/day", "Insulin u/day"))
    t1 = P.table(rows, f"Single simulated patient (sens = 1). True-simulator return and clinical channels per 24-hour day ({cfg['seeds']} replicates, 95% CI). The rule and the oracle are single deterministic policies, so their CI is zero by construction.",
                 widths=[5.2, 3.4, 2.5, 2.4, 2.5], font=8.5)
    P.p(f"Table {t1} is the baseline. DP from {cfg['n_eps']} days scores {S['DP']['J']:.2f} ± {S['DP']['ci']:.2f}, Monte Carlo {S['Monte Carlo']['J']:.2f} ± {S['Monte Carlo']['ci']:.2f} and Q-learning "
        f"{S['Q-learning']['J']:.2f} ± {S['Q-learning']['ci']:.2f}, against {rule_J:.2f} for the rule. Paired over replicates the differences from the rule are {pv['DP'][0]:+.2f} ± {pv['DP'][1]:.2f} "
        f"({pstr(pv['DP'][2])}) for DP, {pv['Monte Carlo'][0]:+.2f} ± {pv['Monte Carlo'][1]:.2f} ({pstr(pv['Monte Carlo'][2])}) for Monte Carlo and {pv['Q-learning'][0]:+.2f} ± {pv['Q-learning'][1]:.2f} "
        f"({pstr(pv['Q-learning'][2])}) for Q-learning, so none of the three is distinguishable from the rule. Replicate by replicate, DP was worse than the rule in {n_dp_worse} of {seeds} runs, equal in {n_dp_equal} and better in {n_dp_better}; "
        f"when it equalled the rule the learned policy simply reproduced it. Yet the oracle DP fitted to {cfg['n_oracle']:,} days scores {S['Oracle DP']['J']:.2f}, which is {gap:.2f} above the rule (on the oracle model the same gap is {gap_model:.2f}, "
        f"{J_or_m:.2f} versus {J_rule_m:.2f}; absolute model values are lower than simulator values because the model includes a terminal pseudo-count). Headroom therefore exists, and it comes with "
        f"more insulin ({S['Oracle DP']['info']['insulin']:.1f} units per day against {S['Rule']['info']['insulin']:.1f}) and more in-range time ({S['Oracle DP']['info']['in_range']:.1f} h against {S['Rule']['info']['in_range']:.1f}) "
        f"with no hypoglycaemic hours recorded for any method in this table. The headroom is out of reach of 150 days of logs, and the learners' wide intervals ({S['DP']['ci']:.1f} to {S['Q-learning']['ci']:.1f}) exceed any difference among DP, Monte Carlo and Q-learning.")
    f2 = P.fig(fig(I, "fig_learning"), "Learning curves on the oracle MDP (mean ± s.d. over 5 replicates). Dashed: oracle DP optimum; dotted: rule.", 6.5)
    P.p(f"Figure {f2} shows learning dynamics. Both solvers jump from near zero to about {q_first:.1f} (Q-learning) and {m_first:.1f} (Monte Carlo) within the first checkpoint at {step_first:,} steps and then stop improving: "
        f"averaged over the last ten checkpoints Q-learning sits at {q_last:.1f} (s.d. across checkpoints and replicates {q_last_sd:.1f}) and Monte Carlo ends at {m_last:.1f}, against {J_rule_m:.2f} for the rule and {J_or_m:.2f} for the oracle on the same model. "
        f"The plateau is set by data, not by compute. The additional steps only replay a model whose supported cells do not contain a better policy. Q-learning also oscillates, plausibly because its greedy policy flips between "
        f"near-equal actions in cells with few samples (we did not test this).")
    f3 = P.fig(fig(I, "fig_policies"), "Dose chosen by each method in each state observed in the oracle data (colour: 0 to 3 units). The learned policies resemble the behaviour policy; the oracle DP uses 3 u in many more states.", 7.5)
    P.p(f"Figure {f3} explains why. Of the {n_states} states, the learned policies differ from the rule in only {n_dp_diff} (DP), {n_mc_diff} (Monte Carlo) and {n_q_diff} (Q-learning) while the oracle differs in {n_or_diff}; the oracle uses the "
        f"3-unit dose in {n_or_3u} states against {n_rule_3u} for the rule. With ε = 0.25 each non-rule dose is tried in about 6% of visits, so with {cfg['n_eps'] * 24:,} logged transitions (150 days × 24 hours) only the busiest states accumulate the 10 observations "
        f"needed to support a non-rule dose. Elsewhere, the support threshold forces the solver back to the rule. This is intended conservatism, and it caps what 150 days can achieve. Learner agreement with DP in well-visited states was "
        f"{agQ:.0f}% for Q-learning and {agM:.0f}% for Monte Carlo (the agreement figure is fig_agreement in the code package), which says that the model-free methods recover the DP policy where there is data to do so. "
        f"The optimism gap (value of the DP policy on its own estimated model minus its value on the oracle model) was positive in {n_og_pos} of {seeds} replicates, so estimated models are not reliably optimistic or pessimistic here.")

    P.h2("The personalisation ceiling and the cost of using only the patient's own data")
    pds = lambda d: f"{d['mean']:+.2f} ± {d['ci']:.2f} ({pstr(d['p']).replace(' ', '').replace('=', '=')})"
    rows = [["σ", "Rule", "Pooled", "Indiv. (7 d)", "Oracle twin", "Oracle − Pooled", "Indiv. − Pooled"]]
    for sg in SIG:
        rows.append([str(sg), pm(Jt(sg, "Rule", ND7), Ct(sg, "Rule", ND7)), pm(Jt(sg, "Pooled", ND7), Ct(sg, "Pooled", ND7)), pm(Jt(sg, "Individual", ND7), Ct(sg, "Individual", ND7)),
                     pm(Jt(sg, "Oracle twin", ND7), Ct(sg, "Oracle twin", ND7)), pds(ceil[sk(sg)]), pds(ind[sk(sg)])])
    t2 = P.table(rows, "Reference policies on 20 new patients per heterogeneity level (mean ± 95% CI over patients; paired differences with p-values). Individual uses 7 days of the patient's own data.",
                 widths=[1.0, 2.2, 2.2, 2.2, 2.2, 3.3, 3.3], font=7.5)
    P.p(f"Table {t2} contains the two facts on which the rest of the paper rests. First, a population policy built from {40 * 20} days of other patients' logs beats the logged rule at every heterogeneity level: by "
        + ", ".join(f"{pooled_minus_rule[sg]:+.2f} (σ = {sg})" for sg in SIG) +
        f". The pooled policy trained on 800 days across patients does what the single-patient benchmark could not do with 150 days. (This comparison confounds data quantity with population diversity, since diverse patients also "
        f"visit more states and so act as exploration; the experiments cannot separate the two.) Second, an oracle twin fitted to 2,000 days of the patient's own data beats the pooled policy by {ceil_range[0]:.2f} to {ceil_range[1]:.2f}, with every "
        f"paired test at {pstr(max(c['p'] for c in ceil.values()))}, and the oracle twin is the better policy for {', '.join(f'{100 * frac_oracle[sk(sg)]:.0f}%' for sg in SIG)} of patients (σ = {', '.join(str(s_) for s_ in SIG)}, respectively). The ceiling is real but modest, "
        f"roughly {100 * min(ratios):.1f}% to {100 * max(ratios):.1f}% of the pooled return, and it does not grow monotonically with heterogeneity: it is {ceil[sk(SIG[0])]['mean']:.2f}, "
        f"{ceil[sk(SIG[1])]['mean']:.2f}, {ceil[sk(SIG[2])]['mean']:.2f} and {ceil[sk(SIG[3])]['mean']:.2f}, with overlapping intervals. We expected the opposite and did not investigate why. One candidate is that a wider population "
        f"also gives the pooled model broader coverage of sensitivities.")
    P.p(f"The third column group of Table {t2} is the cautionary result. A twin built only from the patient's own 7 days is worse than the pooled policy by {-ind_range[1]:.2f} to {-ind_range[0]:.2f} return units at every σ "
        f"(all {pstr(max(c['p'] for c in ind.values()))}), that is, worse than the pooled policy by more than the oracle twin gains over it. The likely cause is the support constraint: "
        f"in {n_coinc} of the {len(SIG) * len(NDS)} (σ, days) cells the individual-only policy has exactly the same return as the rule, which happens when in nearly every state only the rule's own dose has 10 observations, so the planner has nothing else it may choose. Where it differs "
        f"(for example at {NDS[-1]} days) it is still no better than the rule: at σ = {SIG[0]} it scores {Jt(SIG[0], 'Individual', NDS[-1]):.2f} against {Jt(SIG[0], 'Rule', NDS[-1]):.2f}. Learning a patient model from days of data, with no prior, throws away the large amount of information that the population provides.")
    f4 = P.fig(fig(I, "fig_twin_gain"), "Return gain relative to the pooled policy at 7 days of patient data, by heterogeneity (mean ± 95% CI over 20 patients). Black dashed: oracle twin (ceiling). Coloured: shrunk twins. Orange: individual-only.", 7.5)
    P.p(f"Figure {f4} makes the contrast visible. The ceiling is the dashed black line above zero; individual-only is the orange line far below; the three shrunk twins hover around zero, "
        f"with intervals that cross zero in most places. Shrinkage thus moves the estimator from the harmful orange region to approximately the pooled policy, and only occasionally above it.")

    P.h2("Shrunk twins versus the pooled policy")
    rows = [["λ"] + [f"σ = {sg}" for sg in SIG]]
    for l in LAMS:
        rows.append([str(l)] + [pd_(tvp[str(l)][sk(sg)]) for sg in SIG])
    t3 = P.table(rows, "Shrunk twin minus pooled policy at 7 days of patient data (mean ± 95% CI over 20 patients; paired t-test).", widths=[1.2, 3.8, 3.8, 3.8, 3.8], font=7.5)
    P.p(f"Table {t3} is the primary test of the twin. The shrunk twin is significantly better than the pooled policy in {n_pos_sig} of {len(cells)} cells ("
        + "; ".join(f"λ = {l}, σ = {sg}: {tvp[str(l)][sk(sg)]['mean']:+.2f}, {pstr(tvp[str(l)][sk(sg)]['p'])}" for l, sg in sig_cells) +
        f") and significantly worse in {n_neg_sig}. The mean difference is negative in {n_neg_mean} of {len(cells)} cells. The largest gain is {bg['mean']:+.2f} ± {bg['ci']:.2f} at λ = {best_gain[0]} and σ = {best_gain[1]}, which is "
        f"{frac_best * 100:.0f}% of the ceiling for that σ ({ceil[sk(best_gain[1])]['mean']:.2f}); elsewhere the twin recovers essentially none of it. "
        f"Against the individual-only model, by contrast, at 7 days every shrunk twin is better by {tw_ind_min:.2f} to {tw_ind_max:.2f} (largest p-value {pstr(tw_ind_pmax)}), so the shrinkage works as a safeguard "
        f"even though it is not a source of consistent gains. The best λ at 7 days is {', '.join(f'{best_lam[sg]} (σ = {sg})' for sg in SIG)}; differences between λ values are well inside the intervals (λ = 10 had the lowest mean return of the three in {lam10_lowest} of {n_lam_cells} (σ, days) cells), so no pattern in λ can be claimed and Eq. (5)'s prediction of smaller λ for larger σ is neither supported nor refuted.")
    f5 = P.fig(fig(I, "fig_twin_curve"), "True return as a function of the patient's own days, at each heterogeneity level (mean ± 95% CI over 20 patients). The rule, pooled and oracle lines are flat by construction.", 13.5)
    P.p(f"Figure {f5} shows the data-quantity dimension. If more of the patient's own data made the twin better, the coloured lines would rise toward the dashed black ceiling. They do not. The twins show no consistent rise toward the oracle with more days; at {NDS[-1]} days several twins fall below the pooled policy (for σ = {SIG[0]}: λ = 10 scores {Jt(SIG[0], 'Twin (lam=10)', NDS[-1]):.2f}, "
        f"pooled {Jt(SIG[0], 'Pooled', NDS[-1]):.2f}) and their intervals widen sharply. Individual-only (orange) stays on the rule. A sensible reading is that, in this tabular representation, adding days increases the number of cells where the patient's counts are "
        f"just enough to dominate a prior of strength λ but not enough to be accurate, so more data may move the estimate away from a good prior toward a noisy one (a hypothesis we did not test).")
    rows = [["σ"] + [f"{nd} d" for nd in NDS]]
    for sg in SIG:
        rows.append([str(sg)] + [f"{by_nd['30'][sk(sg)][str(nd)]['mean']:+.2f} ({pstr(by_nd['30'][sk(sg)][str(nd)]['p'])})" for nd in NDS])
    t4 = P.table(rows, "Shrunk twin (λ = 30) minus pooled, by amount of patient data (mean difference over 20 patients; p-value in parentheses).", widths=[1.4, 3.7, 3.7, 3.7, 3.7], font=8)
    P.p(f"Table {t4} gives the numbers behind Figure {f5} for λ = 30. Across all {n_all} (λ, σ, days) combinations in the study the twin is significantly better than pooled in {a_pos} and significantly worse in {a_neg}, "
        f"and its mean difference is negative in {a_negmean}. With {n_all} tests one expects about {exp_false:.1f} nominally significant results at the 5% level even if nothing were happening, so the {a_pos} positive ones should be treated as exploratory and not as "
        "evidence of benefit. There is no monotone benefit from more patient data.")

    P.h2("Which patients benefit, and what happens to safety")
    f6 = P.fig(fig(I, "fig_twin_scatter"), "At σ = 0.8 and 7 days: per-patient gain over the pooled policy against the patient's log insulin sensitivity. Left: shrunk twin (λ = 30). Right: oracle twin (ceiling).", 7.5)
    P.p(f"Figure {f6} looks at individual patients. In the right-hand panel the ceiling is largest for the least insulin-sensitive patients (log sensitivity near {np.log(0.35):.2f}, the clipping bound; {n_lo_clip} of 20 test patients sit at that bound) and moderate elsewhere, "
        f"with {n_not_better[SG_hi]} of {ceil[sk(SG_hi)]['n']} patients for whom the oracle is not the better policy. In the left-hand panel the shrunk twin's gain is positive for most of the most insulin-sensitive patients and negative for a cluster of near-average patients, "
        f"so it tends to help or harm in different places from the oracle. The Spearman correlation between the ceiling and the absolute log sensitivity (atypicality) is {spr['rho']:.2f} ({pstr(spr['p'])}), which is not distinguishable from zero with 20 patients. "
        f"We therefore cannot say that atypical patients benefit most, which is the intuitive hypothesis behind a digital twin; {n_hi_clip} patients sit at the upper bound {2.5}, and clipping may blunt the relationship.")
    rows = [["σ", "Rule", "Pooled", "Individual", "Twin (λ=30), 7 d", "Twin (λ=30), 28 d", "Oracle twin"]]
    for sg in SIG:
        c = lambda n, nd: f"{Xt(sg, n, nd, 'in_range'):.1f} / {Xt(sg, n, nd, 'hypo'):.3f}"
        rows.append([str(sg), c("Rule", ND7), c("Pooled", ND7), c("Individual", ND7), c("Twin (lam=30)", ND7), c("Twin (lam=30)", NDS[-1]), c("Oracle twin", ND7)])
    t5 = P.table(rows, "Clinical channels per 24-hour day: in-range hours / hypoglycaemic hours (mean over 20 patients). Individual and the 7-day twin use 7 days of patient data.", widths=[1.2, 2.3, 2.3, 2.3, 2.9, 2.9, 2.3], font=7.5)
    P.p(f"Table {t5} reports what the return hides. In-range time follows the return: the oracle twin has the most at every σ, and the rule and individual-only (which reverts to it) the least. Hypoglycaemia is rare everywhere, consistent with a heavy reward penalty and a cautious rule, "
        f"but it is not zero for the personalised policies. At σ = {SG} the twin fitted to {NDS[-1]} days spends {hyp(SG, 'Twin (lam=30)', NDS[-1]):.3f} hours per day below 70 mg/dL against {hyp(SG, 'Pooled', ND7):.3f} for the pooled policy and "
        f"{hyp(SG, 'Rule', ND7):.3f} for the rule; at σ = {SG_hi} the figures are {hyp(SG_hi, 'Twin (lam=30)', NDS[-1]):.3f}, {hyp(SG_hi, 'Pooled', ND7):.3f} and {hyp(SG_hi, 'Rule', ND7):.3f}. These are per-day means of small quantities with "
        f"no CIs stored in the results file, so they are reported as observed, not tested. They are several times the pooled value in these cells (and every twin fitted to 14 or 28 days has a higher mean than the pooled policy at every σ), and the pattern reappears at 14 days (σ = {SG}: {hyp(SG, 'Twin (lam=30)', 14):.3f} h per day). "
        f"A twin that doses more aggressively for a patient it believes to be insensitive exposes that patient to hypoglycaemia whenever the belief is wrong, so a real system would need an explicit hypoglycaemia constraint, not only a reward penalty.")
    P.h1("Discussion")
    P.p("**What the evidence supports.** Three conclusions are well supported. (1) Population learning beats the rule in this simulator, from pooled logs of many patients, though not from 150 days of one patient. "
        "(2) There is a personalisation ceiling, an oracle twin from 2,000 days beats the pooled policy in a large majority of patients, so personalisation has value in principle. (3) Individual-only twins from 3 to 28 days stay at or within noise of the rule, and the 7-day version is worse than pooled, plausibly because "
        f"the support constraint leaves little to plan with. One conclusion is not supported: that a shrunk twin from days of data captures the ceiling. At 7 days it differs significantly from the pooled policy in {n_pos_sig} of {len(cells)} cells, all in the positive direction when it differs "
        f"({n_neg_sig} significantly worse); across all {n_all} combinations it was significantly better in {a_pos} and significantly worse in {a_neg}.")
    P.p("**Why the twin does not do better.** We offer hypotheses, not findings. (a) The tabular state is coarse (6 glucose bands). The patient's sensitivity is expressed in how glucose changes after a dose, which a coarse trend flag only "
        "partly captures, so even a perfect estimate of the patient's own kernel supports a modest gain, as the ceiling shows. (b) Seven days is 168 transitions spread over up to 144 cells; the cells where "
        "the patient's data are informative are the busy ones, where the population is also well estimated. (c) Because data come from the ε-noisy rule, doses outside the rule are explored in only 6% of visits, "
        "so the twin rarely sees that a dose is more potent; active, safe exploration was not tested.")
    P.p("**Design guidance.** If a team builds a twin from sparse data it should (i) start from a population model and shrink, because the alternative is demonstrably worse here; (ii) not assume that more days improve the twin without checking, "
        "since the 28-day twins show no consistent gain over the 7-day ones in this study; (iii) track hypoglycaemic time as a first-class outcome, because gains in return can be paid for in safety; and (iv) report personalisation "
        f"relative to a pooled baseline and to an oracle ceiling, not only to a rule, which here would have flattered every method by {min(pooled_minus_rule.values()):.2f} to {max(pooled_minus_rule.values()):.2f} return units.")
    limitations(P, [
        "**One-parameter heterogeneity.** Patients differ only in insulin sensitivity. Real patients also differ in meal size and timing, insulin absorption delay, stress and exercise effects; a twin would need several parameters.",
        "**No insulin-on-board.** Insulin acts instantly on the glucose target in the simulator. Real insulin acts over hours, which makes the problem non-Markov in the coarse state and favours richer histories.",
        "**Matched population.** The population model is built from patients drawn from the same lognormal as the test patients. A mismatched population (for example a paediatric cohort used for adults) would hurt the pooled policy and the twin.",
        "**Twenty patients per cell.** Paired tests over 20 patients and 48 (λ, σ, days) combinations leave low power for small effects and a high chance of spurious significance; hypoglycaemia is reported without intervals.",
        "**Data from one behaviour rule.** Patient and population logs come from the same ε-noisy rule, which limits the doses ever observed; real logs have different, patient-specific behaviour.",
    ])
    ethics_future(P, "(1) Replace the one-parameter simulator with a physiological model and several latent parameters; (2) add a hypoglycaemia constraint through a constrained or shielded policy "
                  "and test whether it removes the extra hypoglycaemic time seen in twins fitted to 14 or more days; (3) estimate cell-specific shrinkage by empirical Bayes instead of a global λ, which would test Eq. (5); (4) add safe active exploration to the patient's logs; "
                  "(5) run the adapter on a real CGM table and compare the model-predicted advantage of the twin with a prospective holdout.")
    P.h1("Conclusion")
    P.p(f"A patient-specific MDP fitted to a few days of CGM data is a poor basis for dosing: at 7 days it is worse than a pooled population policy by {-ind_range[1]:.2f} to {-ind_range[0]:.2f} return units. Shrinking it toward the population with λ pseudo-counts removes that "
        f"loss, and at 7 days the shrunk twin is better than individual-only in every condition tested (by {tw_ind_min:.2f} to {tw_ind_max:.2f}). But there is a real personalisation ceiling ({ceil_range[0]:.2f} to {ceil_range[1]:.2f} over pooled), and the shrunk twin realised significant gains in only {n_pos_sig} of {len(cells)} cells at 7 days, "
        f"with no benefit from additional days and a rise in hypoglycaemic time at longer data windows. On a single patient, learning from {cfg['n_eps']} days failed to beat the rule although an oracle did ({S['Oracle DP']['J']:.2f} versus {rule_J:.2f}). "
        "These are simulator results. The contribution is a tested recipe, a ceiling measure and an honest account of what did not work.")
    defense(P, [
        ("Why should the shrunk twin not be considered a failure, given that it mostly equals the pooled policy?",
         f"Because the relevant comparison for a new patient with few days is individual-only, and the twin is better than that at 7 days by {tw_ind_min:.2f} to {tw_ind_max:.2f} in every (λ, σ) condition (Section 7.3 and Figure {f4}). It is a safeguard, not a source of gains, and we say so; the claim of "
         f"large gains from personalisation is the part the data do not support."),
        ("Is a ceiling of about half a return unit meaningful?",
         f"It is statistically clear (every paired test at {pstr(max(c['p'] for c in ceil.values()))}) and the oracle is better in {min(frac_oracle.values()) * 100:.0f}% to {max(frac_oracle.values()) * 100:.0f}% of patients, but its clinical meaning is unknown: the return is a weighted sum of in-range hours and hypoglycaemia penalties, "
         f"and in-range hours at σ = {SG_hi} rise from {Xt(SG_hi, 'Pooled', ND7, 'in_range'):.1f} to {Xt(SG_hi, 'Oracle twin', ND7, 'in_range'):.1f} out of 24."),
        ("Do more days of patient data help?",
         f"Not in this study. Table {t4} shows the λ = 30 twin has no consistent benefit as days increase, and Figure {f5} shows several twins below pooled at {NDS[-1]} days. With only 20 patients per cell we cannot rule out a small benefit, but we find no evidence of one."),
        ("Why did the single-patient learners not beat the rule when the pooled policy did?",
         f"The pooled model uses 800 days across diverse patients, the single-patient benchmark 150 days. The support threshold of 10 observations leaves most non-rule doses unsupported with 150 days (Figure {f3}). The experiment cannot separate data volume from population diversity, and we flag that."),
        ("Does the twin make patients safer?",
         f"No evidence for that. In-range time rises but hypoglycaemic hours for twins fitted to {NDS[-1]} days are higher than for the pooled policy (for example {hyp(SG, 'Twin (lam=30)', NDS[-1]):.3f} versus {hyp(SG, 'Pooled', ND7):.3f} h per day at σ = {SG}). These are small absolute numbers without intervals, but the direction is a warning."),
    ])
    repro(P, 1, ["fig_bars", "fig_learning", "fig_policies", "fig_agreement", "fig_twin_curve", "fig_twin_gain", "fig_twin_scatter", "fig_twin_clinical"])
    import domains
    code_appendix(P, 1, [domains.Diabetes])
    rl_core_appendix(P)
    P.refs()
    path = f"{PAP}/Paper01_DigitalTwin_Insulin.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
