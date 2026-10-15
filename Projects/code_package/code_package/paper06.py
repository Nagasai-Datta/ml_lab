import numpy as np
from scipy import stats
from docx_helper import Paper
from paper_common import *
import domains

I = 6


def build():
    r = load(I); S = r["standard"]; E = r["extra"]
    LAMS, BUD, MARG = r["LAMS"], r["BUD"], r["MARG"]
    Bn, Cn, Fl = np.array(r["benefit"]), np.array(r["burden"]), np.array(r["failure"])
    Bc, Cc = np.array(r["benefit_ci"]), np.array(r["burden_ci"])
    fx = r["fixed"]
    REAL, REALC = np.array(r["realised"]), np.array(r["realised_ci"])
    PRED, BENB, FAILB, LAMB, VIOL = (np.array(r[k]) for k in ("predicted", "benefit_b", "failure_b", "lam_b", "violation"))
    cfg = E["config"]; meth = E["methods"]
    nl = len(LAMS)
    NSEED = 10                                          # replicates in the lambda / budget sweeps (SEEDS in ideas.idea6)
    tf10 = float(stats.t.ppf(0.975, NSEED - 1) / 1.96)  # t-interval factor for 10 replicates
    nS = E["n_states"]; dom = domains.CKD()
    rule_J = S["Rule"]["J"]

    # ---- benchmark statistics
    def pvr(k, ref="Rule"):
        d = np.array(meth[k]["J"]) - np.array(meth[ref]["J"])
        return float(d.mean()), 1.96 * float(d.std(ddof=1) / np.sqrt(len(d))), float(stats.ttest_1samp(d, 0).pvalue)
    pv = {k: pvr(k) for k in ("DP", "Monte Carlo", "Q-learning")}
    dp_q, dp_mc = pvr("DP", "Q-learning"), pvr("DP", "Monte Carlo")
    JF = {k: v["benefit"] - v["burden"] for k, v in fx.items()}      # lambda = 1 return of fixed policies (600 episodes)
    rule_noise = JF["Nephrologist rule"] - rule_J                      # same policy, 600 vs 400 evaluation episodes
    gap_or_model = E["J_oracle_model"]["Oracle DP"] - S["Oracle DP"]["J"]
    pol = E["policies"]
    n_diff = {k: int(np.sum(np.array(pol[k]) != np.array(pol["Rule"]))) for k in ("DP", "Monte Carlo", "Q-learning", "Oracle DP")}
    n_act = {k: np.bincount(pol[k], minlength=3).tolist() for k in pol}
    agQ, agM = 100 * np.mean(E["agree"]["Q-learning"]), 100 * np.mean(E["agree"]["Monte Carlo"])
    cq = np.array(E["curves"]["Q-learning"]); cm = np.array(E["curves"]["Monte Carlo"])
    q0, q_last, m_last = float(cq[:, 0, 1].mean()), float(cq[:, -10:, 1].mean()), float(cm[:, -1, 1].mean())
    J_or_m, J_rule_m = E["J_oracle_model"]["Oracle DP"], E["J_oracle_model"]["Rule"]
    og = np.array(E["optimism_gap"]); n_og_pos = int((og > 0).sum())

    # ---- frontier statistics
    mono = bool(np.all(np.diff(Cn) < 0) and np.all(np.diff(Bn) < 0) and np.all(np.diff(Fl) > 0))
    slope = np.diff(Bn) / np.diff(Cn)       # realised benefit lost per unit burden between consecutive lambdas (increasing lambda -> reading from low to high)
    slope_ok = [bool(LAMS[i] <= slope[i] <= LAMS[i + 1]) for i in range(nl - 1)]
    n_slope_ok = int(sum(slope_ok))
    SC = np.array([[Bn[j] - l * Cn[j] for j in range(nl)] for l in LAMS])    # scalarised true value at price l of the policy trained at price LAMS[j]
    best_col = SC.argmax(1)
    n_diag = int(sum(1 for i in range(nl) if best_col[i] == i))
    diag_loss = [float(SC[i].max() - SC[i, i]) for i in range(nl)]
    order = np.argsort(Cn)
    fr_at = lambda c: float(np.interp(c, Cn[order], Bn[order]))
    li = {l: i for i, l in enumerate(LAMS)}

    # ---- budget statistics
    ratio = REAL / PRED
    over = REAL - np.array(BUD)[None, :]
    over_sig = np.abs(over) > REALC * tf10
    viol_n = np.rint(VIOL * NSEED).astype(int)
    mi = {m: i for i, m in enumerate(MARG)}; bi = {b: i for i, b in enumerate(BUD)}
    ben_cost = BENB[mi[0.0]] - BENB[mi[0.5]]
    fail_cost = 100 * (FAILB[mi[0.5]] - FAILB[mi[0.0]])
    req_margin = REAL[mi[0.0]] / PRED[mi[0.0]] - 1.0
    first = lambda a, b: a[0] / b[0]
    unconstrained = bool(LAMB[mi[0.0], bi[6]] < 5e-4)

    P = Paper("Lagrangian Dynamic Programming for Chronic Kidney Disease Care: Benefit-Burden Frontiers and Safety Margins for Burden Budgets",
              "chronic kidney disease; constrained Markov decision process; Lagrangian relaxation; Pareto frontier; treatment burden; safety margin; dynamic programming",
              running="Lagrangian DP for CKD care", refdict=REFS)

    P.abstract(
        "Managing chronic kidney disease (CKD) means choosing, at each visit, how intensively to treat: more intensive care slows the loss of kidney function but adds burden for the patient and the health system. "
        "We study this trade-off as a constrained Markov decision process on a simulated CKD cohort whose variables mirror public CKD tables (the Kaggle site was unreachable, HTTP 403, so no real file was used). "
        f"We learn from {cfg['n_eps']} logged patient trajectories with dynamic programming (DP), Monte Carlo (MC) control and Q-learning, add a Lagrange multiplier that prices burden to trace a benefit-burden frontier, "
        "and propose a safety margin that tightens the budget used in planning to protect against under-predicted burden. "
        f"At unit price, DP, Q-learning and MC scored {S['DP']['J']:.2f}, {S['Q-learning']['J']:.2f} and {S['Monte Carlo']['J']:.2f} against {rule_J:.2f} for a nephrologist-style rule, "
        f"but a constant 'always enhanced' policy scored {JF['Always enhanced']:.2f}, so the improvement is largely available without learning. "
        f"Across eight prices the frontier is monotone (burden from {Cn[0]:.2f} to {Cn[-1]:.2f}, benefit from {Bn[0]:.1f} to {Bn[-1]:.1f}, simulated kidney-failure rate from {100 * Fl[0]:.1f}% to {100 * Fl[-1]:.1f}%), "
        f"but the 'always enhanced' policy lies above it ({fx['Always enhanced']['benefit']:.2f} versus {fr_at(fx['Always enhanced']['burden']):.2f} at the same burden), and for prices of 2 to 8 the policy trained at a given price was not the best at that price in the true simulator. "
        f"Planning to a burden budget with no margin exceeded the budget in {int(viol_n[mi[0.0]].sum())} of {viol_n.shape[1] * NSEED} replicate-budget runs (realised burden was {ratio.min():.2f} to {ratio.max():.2f} times the model's prediction); "
        f"a 50% margin reduced exceedances to {int(viol_n[mi[0.5]].sum())} at a cost of {ben_cost.min():.2f} to {ben_cost.max():.2f} benefit units. All results are simulated and carry no clinical meaning.")

    P.h1("Introduction")
    P.p("CKD is a progressive loss of kidney function, usually tracked by the estimated glomerular filtration rate (eGFR) and by markers such as proteinuria; clinical guidelines tie monitoring and treatment intensity to the stage of disease [@kdigo]. "
        "Intensifying care, for example with tighter blood-pressure control [@whelton], more frequent review, or additional drugs, can slow decline, but each step adds burden: clinic time, side effects, cost. "
        "A treatment policy is therefore a sequence of decisions that trade a health benefit against a cost, and a policymaker may reasonably say not 'maximise a weighted sum' but 'maximise benefit within a burden budget'.")
    P.p("Sequential decision problems with resource constraints are the subject of constrained Markov decision processes (CMDPs) [@altman]; safe reinforcement learning has built methods around them [@garcia]. A classical route is Lagrangian relaxation: "
        "price the constrained resource by a multiplier and solve an ordinary MDP for the combined reward. This is attractive for tabular health problems because the inner solver is plain dynamic programming and the outcome, "
        "a frontier of efficient policies, is easy to inspect. The difficulty in healthcare is that the model is estimated from logs, so the burden of a candidate policy is itself an estimate, and a budget that is met in the model may be broken in practice.")
    P.p("**Contributions.** (i) A transparent CKD simulator with competing risks, a benefit channel (kidney function, with a large failure penalty) and a burden channel (treatment intensity). "
        f"(ii) A Lagrangian DP pipeline that sweeps the burden price over {nl} values and, by bisection, finds the price that meets a model-predicted burden budget (Eq. 3 to 5). "
        "(iii) A safety-margin rule that plans to a tightened budget, with an empirical calibration ratio that says how large the margin must be (Eq. 6). "
        "(iv) Benchmarks of DP, MC and Q-learning on the scalarised problem and comparison with three constant policies and a clinical-style rule. "
        "(v) Negative results: a constant policy matches the learned policy at unit price; the learned frontier is partly dominated by that constant policy; and the policy trained at a high price is not the best at that price.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached from the build environment (all requests returned HTTP 403), so no real CKD table was analysed. The simulator mirrors the variables of CKD tables "
        "(eGFR, proteinuria, treatment intensity, outcome) with made-up dynamics. Findings concern algorithms, not patients, and no clinical claim is made.")

    P.h1("Related work")
    P.p("**CKD care and guidelines.** Guidelines classify CKD by eGFR and albuminuria and recommend monitoring frequency and treatment according to risk [@kdigo]; blood-pressure targets are set by hypertension guidelines [@whelton]. "
        "Our rule, intensive care below an eGFR threshold, enhanced care at intermediate eGFR and routine care above, is a stylised version of this stratification and is not a guideline.")
    P.p("**Constrained MDPs and safe RL.** Altman's monograph develops CMDPs, including the Lagrangian dual and the fact that optimal constrained policies may need randomisation [@altman]. Surveys of safe RL describe the Lagrangian and other approaches [@garcia], "
        "and shielding enforces safety by blocking unsafe actions [@shield]. We use the plain dual with deterministic policies.")
    P.p("**Offline RL.** Learning from logs requires caution because models are wrong where data are thin [@levine,jin,kumar]. In this paper the same fact appears as a calibration problem: the model under-predicts the burden of the policy it chooses.")
    background(P)

    P.h1("Problem formulation: a CKD care simulator")
    P.p(f"A patient has a continuous eGFR (mL/min/1.73 m²) and a binary proteinuria flag. At entry eGFR is uniform on [25, 85] and 40% of patients have proteinuria. At each of up to {dom.max_steps} visits the clinician chooses one of three intensities: routine, enhanced or intensive. "
        "eGFR then falls by a base amount (0.45 without proteinuria, 1.1 with it) multiplied by (1 − r), where r = 0, 0.25 or 0.5 is the reduction due to the chosen intensity, multiplied by 1.4 if eGFR is already below 30, plus Gaussian noise with standard deviation 0.6. "
        "Proteinuria, if present, resolves with probability 0.04 + 0.1 × (intensity index) per visit; if absent it appears with probability 0.05 − 0.015 × (intensity index). The episode ends in kidney failure when eGFR falls below 15, on competing mortality "
        "(1.5% per visit, 2.5% when eGFR is below 30), or at the horizon.")
    P.table([["Element", "Definition"],
             ["State (120)", f"eGFR band (20 bands with cut points every 4 units from {dom.GE[0]} to {dom.GE[-1]}) × proteinuria (2) × previous intensity (3)"],
             ["Actions (3)", "routine, enhanced, intensive (eGFR-decline reduction 0, 25%, 50%)"],
             ["Benefit channel c1", "eGFR / 60 per visit, minus 20 in the visit when kidney failure occurs"],
             ["Burden channel c2", f"{dom.BUR[0]}, {dom.BUR[1]} and {dom.BUR[2]} per visit for routine, enhanced and intensive"],
             ["Scalar reward", "c1 − λ c2; λ = 1 for the benchmark; λ varied for the frontier"],
             ["Discount, horizon", f"γ = {dom.gamma}; up to {dom.max_steps} visits"],
             ["Nephrologist-style rule", "intensive if the eGFR band midpoint is below 45, enhanced if below 60, otherwise routine"],
             ["Behaviour policy", "the rule with ε = 0.25 random actions"],
             ["Dataset mirrored", "Chronic Kidney Disease table (eGFR, proteinuria, treatment, outcome)"]],
            "The CKD care MDP.", widths=[3.6, 12.8], font=8.5)
    P.p("Two features of this design deserve attention. First, intensive care carries a burden of 0.5 per visit against 0.15 for enhanced care, so over a long episode intensive care is expensive, and the price of burden determines which "
        "intensity is worth giving. Second, because the rule is intensive for low eGFR and routine for high eGFR, the logged data contain few intensive visits at high eGFR and few routine visits at low eGFR, "
        "which limits what a learner can discover about those combinations; this will matter in the results.")

    P.h1("Method: Lagrangian DP and safety margins")
    P.p("Let J_b(π) and J_c(π) be the expected discounted benefit and burden of a policy π. The constrained problem for a budget B is")
    P.eq("max_π J_b(π)   subject to   J_c(π) ≤ B.", "3")
    P.p("Relaxing the constraint with a multiplier λ ≥ 0 gives the scalarised problem max_π [J_b(π) − λ J_c(π)], an ordinary MDP with reward c1 − λ c2, which we solve by value iteration on the estimated model. "
        "Let π_λ be its solution and C(λ) = J_c(π_λ). A standard exchange argument shows that C(λ) is non-increasing in λ and that, if π_λ and π_λ' are both optimal at their own prices, the exchange rate between them satisfies")
    P.eq("λ ≤ [J_b(π_λ) − J_b(π_λ')] / [J_c(π_λ) − J_c(π_λ')] ≤ λ'   for λ < λ'.", "4")
    P.p("Equation (4) is a testable property: the realised benefit lost per unit of burden saved between neighbouring frontier points must lie between their prices. Because C(λ) is monotone, the price that meets a budget can be found by bisection. "
        "Deterministic policies reach only the vertices of the achievable set, so the budget is met approximately (a randomised mixture of two adjacent policies would be needed to meet it exactly).")
    P.p("The burden is predicted from the estimated model, and the model can be wrong about it. In this pipeline pairs with fewer than 10 observations are treated as unsupported and absorbing; the model assigns them no burden channel reward, while the deployed policy "
        "falls back to the rule in states it has not seen, and the rule incurs real burden. A policy chosen to meet a predicted budget can therefore exceed it. The **safety margin** m plans to a tightened budget,")
    P.eq("find the smallest λ such that   Ĵ_c(π_λ) ≤ B / (1 + m).", "5")
    P.p("If the realised burden is a roughly constant multiple ρ of the predicted burden, the realised burden is ρ B / (1 + m), which meets the budget if")
    P.eq("1 + m ≥ ρ,      ρ = J_c(π_λ) / Ĵ_c(π_λ).", "6")
    P.p("The calibration ratio ρ can be estimated by comparing predicted and realised burden at m = 0. We report it together with the exceedance counts.")
    P.code("""Algorithm 1  Lagrangian DP with a burden budget and safety margin
 input : logged episodes, budget B, margin m, price grid, bisection tolerance
 1. build the empirical MDP (supported pairs n >= 10; unsupported pairs absorbing) with channels c1, c2
 2. burden(lam): pi <- value_iteration(reward = c1 - lam * c2);  return  mu . V_pi[c2]  (model-predicted discounted burden)
 3. target <- B / (1 + m)
 4. if burden(0) <= target: lam <- 0
    else: lo <- 0, hi <- 64;  repeat 25 times: mid <- (lo + hi) / 2;  if burden(mid) > target: lo <- mid else hi <- mid;  lam <- hi
 5. pi <- value_iteration(reward = c1 - lam * c2);  in unsupported states use the rule's action
 6. evaluate pi in the true simulator: realised benefit, burden, kidney-failure rate""")

    protocol(P, [["Item", "Setting"],
                 ["Simulator", "CKD care MDP; 120 states; 3 actions; up to 60 visits"],
                 ["Benchmark (lambda = 1)", f"{cfg['n_eps']} logged episodes; oracle MDP from {cfg['n_oracle']:,} episodes; {cfg['seeds']} replicates; {cfg['n_eval']} evaluation episodes"],
                 ["Learners (benchmark)", f"DP (value and policy iteration agreed in all replicates: {'yes' if all(E['pi_equals_vi']) else 'no'}), first-visit MC, Q-learning; {cfg['steps']:,} steps"],
                 ["Frontier", f"{nl} prices {LAMS}; {NSEED} replicates (data seeds 300-309); 300 evaluation episodes"],
                 ["Budgets and margins", f"burden budgets {BUD} (discounted units) x margins {[int(100 * m) for m in MARG]}%; {NSEED} replicates; bisection on [0, 64]"],
                 ["Constant policies", "always routine, always enhanced, always intensive and the rule; 600 evaluation episodes each"],
                 ["Statistics", "95% CI = 1.96 standard errors over replicates; where a significance statement is made the interval is widened to a t-interval (factor 1.42 for 5 replicates, 1.15 for 10)"]],
             cfg["steps"], cfg["seeds"], cfg["n_eval"],
             extra="In the frontier and budget sweeps unsupported states fall back to the nephrologist-style rule.")

    P.h1("Results")
    P.h2("Benchmark at unit price")
    rows = std_rows(S, ("d_c1", "d_c2", "failure"), ("Benefit", "Burden", "Failure rate"), nd=2)
    t1 = P.table(rows, f"Benchmark at lambda = 1 ({nS} states). Return is discounted benefit minus discounted burden; benefit, burden and the kidney-failure fraction per episode are also shown ({cfg['seeds']} replicates, 95% CI; rule and oracle are single policies).",
                 widths=[5.0, 3.5, 2.4, 2.4, 3.1], font=8.5)
    P.p(f"Table {t1} shows that all three learners beat the rule at unit price: DP {S['DP']['J']:.2f} ± {S['DP']['ci']:.2f}, Q-learning {S['Q-learning']['J']:.2f} ± {S['Q-learning']['ci']:.2f} and MC {S['Monte Carlo']['J']:.2f} ± {S['Monte Carlo']['ci']:.2f} against {rule_J:.2f}. "
        f"Paired over replicates the differences from the rule are {pv['DP'][0]:+.2f} ± {pv['DP'][1]:.2f} ({pstr(pv['DP'][2])}) for DP, {pv['Q-learning'][0]:+.2f} ± {pv['Q-learning'][1]:.2f} ({pstr(pv['Q-learning'][2])}) for Q-learning and {pv['Monte Carlo'][0]:+.2f} ± {pv['Monte Carlo'][1]:.2f} ({pstr(pv['Monte Carlo'][2])}) for MC. "
        f"The solvers are ranked DP above Q-learning (paired difference {dp_q[0]:+.2f} ± {dp_q[1]:.2f}, {pstr(dp_q[2])}) above MC (DP minus MC {dp_mc[0]:+.2f} ± {dp_mc[1]:.2f}, {pstr(dp_mc[2])}). "
        f"The gain is bought by trading burden for failures: the learners cut burden from {S['Rule']['info']['d_c2']:.2f} to {S['DP']['info']['d_c2']:.2f} (DP) but raise the simulated kidney-failure rate from {100 * S['Rule']['info']['failure']:.1f}% to {100 * S['DP']['info']['failure']:.1f}%. "
        f"Whether that is a good trade depends entirely on the price λ = 1, which is a modelling choice, so the improvement is real only relative to that price. The constant 'always enhanced' policy, evaluated on 600 episodes, scores {JF['Always enhanced']:.2f} at this price (always routine {JF['Always routine']:.2f}, always intensive {JF['Always intensive']:.2f}), so a learned policy that is statistically ahead of the rule is not clearly ahead of the best constant policy. "
        f"The oracle DP, fitted to {cfg['n_oracle']:,} episodes, scored {S['Oracle DP']['J']:.2f}, which is within {abs(S['Oracle DP']['J'] - S['DP']['J']):.2f} of DP fitted to {cfg['n_eps']} episodes; the tabular representation, not data volume, limits performance here "
        f"(the oracle model's own value for its policy is {J_or_m:.2f}, {gap_or_model:.1f} above its true value; we did not diagnose the cause, which may be state aggregation or the stationary treatment of the horizon).")
    f1 = P.fig(fig(I, "fig_learning"), "Learning curves on the oracle MDP (mean ± s.d. over 5 replicates). Dashed: oracle DP optimum; dotted: rule.", 7.0)
    P.p(f"Figure {f1} shows that both learners start near {q0:.1f} on the oracle model, already above the rule's {J_rule_m:.1f}, and finish at {q_last:.1f} (Q-learning, last ten checkpoints) and {m_last:.1f} (MC), far below the oracle optimum ({J_or_m:.1f}). "
        f"Little learning is visible in the curves: the start is above the rule because the initial greedy policy (we did not inspect it, but the curve at step 0 is its value) already scores well on this model, and the improvement afterwards is at most about {max(q_last, m_last) - q0:.1f} units. "
        f"The learned policies differ from the rule in {n_diff['DP']} (DP), {n_diff['Monte Carlo']} (MC) and {n_diff['Q-learning']} (Q-learning) of {nS} states, and the oracle in {n_diff['Oracle DP']}; Q-learning agreed with DP in {agQ:.0f}% of well-visited states and MC in {agM:.0f}% (Figure {f1 + 1}). "
        f"The DP policy's model value minus its true value (optimism gap) was positive in {n_og_pos} of {len(og)} replicates, so on average the model is not systematically optimistic about the policy it picks, in contrast to its optimism about the oracle's.")
    f2 = P.fig(fig(I, "fig_policies"), "Action in each state observed in the data (routine, enhanced, intensive). The learned policies follow the structure of the behaviour policy and change many individual states; the oracle differs most.", 9.0)
    P.p(f"Figure {f2} shows that the learners keep the behaviour policy's coarse structure (intensive at low eGFR, routine at high eGFR) and change many individual cells. Because the cells are sorted by eGFR band, the striping reveals that the changes "
        f"depend on proteinuria and previous action, as the state encodes, but also that many are isolated single cells, which look like noise from thin data. Over all {nS} states the oracle policy uses routine care in {n_act['Oracle DP'][0]}, enhanced in {n_act['Oracle DP'][1]} and intensive in {n_act['Oracle DP'][2]}, against {n_act['Rule'][0]}, {n_act['Rule'][1]} and {n_act['Rule'][2]} for the rule: it gives much less intensive care than the rule, which is what a unit price on burden rewards, and it departs from the rule far more than the learners do ({n_act['DP'][2]} intensive states for DP).")

    P.h2("The benefit-burden frontier")
    rows = [["Price λ", "Benefit", "Burden", "Kidney failure", "Benefit lost per burden saved vs previous λ"]]
    for i, l in enumerate(LAMS):
        rows.append([f"{l:g}", pm(Bn[i], Bc[i]), pm(Cn[i], Cc[i]), f"{100 * Fl[i]:.1f}%", "-" if i == 0 else f"{slope[i - 1]:.2f}"])
    t2 = P.table(rows, f"Lagrangian DP frontier: true-simulator discounted benefit and burden and the kidney-failure fraction by burden price ({NSEED} replicates, 95% CI). The last column is the realised exchange rate between consecutive points.", widths=[1.8, 3.2, 3.0, 2.8, 5.6], font=8.5)
    P.p(f"Table {t2} is the frontier. As the price rises from {LAMS[0]} to {LAMS[-1]}, burden falls from {Cn[0]:.2f} to {Cn[-1]:.2f}, benefit falls from {Bn[0]:.2f} to {Bn[-1]:.2f}, and the kidney-failure fraction rises from {100 * Fl[0]:.1f}% to {100 * Fl[-1]:.1f}%; "
        f"all three are {'strictly monotone in the price' if mono else 'not strictly monotone in the price'}, as theory predicts. At the top price the policy is nearly the always-routine policy (benefit {fx['Always routine']['benefit']:.2f}, burden {fx['Always routine']['burden']:.2f}, failure {100 * fx['Always routine']['failure']:.1f}%). "
        f"The exchange rate in the last column increases overall with the price (from {slope[0]:.2f} to {slope[-1]:.2f}), which is the signature of a concave frontier: the first units of burden saved are cheap in benefit and the last are expensive. "
        f"Equation (4) requires each rate to lie between the two prices that bound it; it does for {n_slope_ok} of the {nl - 1} pairs, and fails for "
        + ", ".join(f"{LAMS[i]:g} to {LAMS[i + 1]:g} ({slope[i]:.2f}, {'below the lower' if slope[i] < LAMS[i] else 'above the upper'} price)" for i in range(nl - 1) if not slope_ok[i]) + ". At the high-price end, where the realised rate falls short of the price, the learned policy at the higher price is not optimal for the lower price in the true simulator.")
    f3 = P.fig(fig(I, "fig_pareto"), "Benefit against burden: the Lagrangian DP frontier (points labelled by price; bars: 95% CI over 10 replicates) and the four reference policies.", 9.0)
    P.p(f"Figure {f3} puts the frontier next to the fixed policies. The 'always enhanced' policy sits above the frontier: it achieves benefit {fx['Always enhanced']['benefit']:.2f} at burden {fx['Always enhanced']['burden']:.2f}, whereas linear interpolation between learned points gives {fr_at(fx['Always enhanced']['burden']):.2f} at that burden, "
        f"a shortfall of {fx['Always enhanced']['benefit'] - fr_at(fx['Always enhanced']['burden']):.2f}, larger than the frontier's own interval widths ({Bc[li[0.5]]:.2f} to {Bc[li[1]]:.2f} at the neighbouring prices). The nephrologist-style rule lies beyond the right end of the frontier: it has burden {fx['Nephrologist rule']['burden']:.2f} "
        f"against {Cn[0]:.2f} for the most benefit-seeking learned policy (λ = 0), and higher benefit ({fx['Nephrologist rule']['benefit']:.2f} against {Bn[0]:.2f}) and a lower failure rate ({100 * fx['Nephrologist rule']['failure']:.1f}% against {100 * Fl[0]:.1f}%). "
        f"'Always intensive' reaches benefit {fx['Always intensive']['benefit']:.2f} at burden {fx['Always intensive']['burden']:.2f}, far above anything the learner explores. In other words, the learned frontier is limited by the support of the logs: "
        f"with λ = 0 a benefit-maximising planner would give intensive care everywhere, but it can do so only in states where the logs contain enough intensive visits, and those are the low-eGFR states where the rule already does it. "
        f"That reading is a hypothesis about the mechanism, supported by the behaviour policy's structure (Figure {f2}) but not tested separately.")
    f4 = P.fig(fig(I, "fig_failure"), "Simulated kidney-failure fraction against the burden price (symmetric-log axis).", 6.0)
    P.p(f"Figure {f4} shows the safety cost of pricing burden. The failure fraction rises slowly from {100 * Fl[0]:.1f}% at λ = 0 to {100 * Fl[li[0.5]]:.1f}% at λ = 0.5, then steeply to {100 * Fl[li[4]]:.1f}% at λ = 4 and flattens near {100 * Fl[-1]:.1f}%, close to the always-routine rate ({100 * fx['Always routine']['failure']:.1f}%). "
        f"A failure costs 20 benefit units, comparable to the whole discounted benefit of a surviving patient (Table {t2}), so this curve is the part of the frontier that a clinician would worry about: a price of 1 already multiplies the failure rate by {Fl[li[1]] / Fl[0]:.1f} relative to λ = 0 ({100 * Fl[li[1]]:.1f}% versus {100 * Fl[0]:.1f}%), and the rule's {100 * fx['Nephrologist rule']['failure']:.1f}% is below every learned point (the lowest is {100 * Fl.min():.1f}%).")
    rows = [["Price λ", "Best-trained price at this evaluation price", "Value of the policy trained at λ", "Value of the best policy", "Shortfall"]]
    for i, l in enumerate(LAMS):
        rows.append([f"{l:g}", f"{LAMS[best_col[i]]:g}", f"{SC[i, i]:.2f}", f"{SC[i].max():.2f}", f"{diag_loss[i]:.2f}"])
    t3 = P.table(rows, "Price-consistency check from the sweep means: at each price λ, the true scalarised value (benefit − λ × burden) of the policy trained at λ, compared with the best policy among the eight trained.", widths=[1.8, 3.9, 4.0, 3.8, 2.4], font=8.5)
    P.p(f"Table {t3} tests whether the policy trained at a price is the best at that price in the true simulator. It is for {n_diag} of {nl} prices (λ = {', '.join(f'{LAMS[i]:g}' for i in range(nl) if best_col[i] == i)}), and not for the others. For λ = {', '.join(f'{LAMS[i]:g}' for i in range(nl) if best_col[i] != i and LAMS[i] >= 2)} "
        f"the best policy is one trained at a higher price (λ = {', '.join(f'{LAMS[best_col[i]]:g}' for i in range(nl) if best_col[i] != i and LAMS[i] >= 2)}), by margins of up to {max(diag_loss):.2f}. This is the same finding as the failure of Eq. (4) at the high end, seen from the other side: at high prices the learned policy "
        "underestimates how valuable it is to save burden or overestimates how much benefit it gives up, so it saves too little burden for its price. A different reading is that those policies are noisy (the estimated model is least reliable far from the logged behaviour). The data cannot separate the two.")

    P.h2("Burden budgets and safety margins")
    rows = [["Budget", "Margin", "Mean price λ", "Predicted burden", "Realised burden", "Realised / predicted", "Exceed budget?", "Benefit", "Failure"]]
    for b in BUD:
        for m in MARG:
            i, j = mi[m], bi[b]
            rows.append([str(b), f"{int(100 * m)}%", f"{LAMB[i, j]:.2f}", f"{PRED[i, j]:.2f}", f"{REAL[i, j]:.2f} ± {REALC[i, j]:.2f}", f"{ratio[i, j]:.2f}",
                         f"{viol_n[i, j]}/{NSEED}" + (" *" if (over[i, j] > 0 and over_sig[i, j]) else ""), f"{BENB[i, j]:.2f}", f"{100 * FAILB[i, j]:.1f}%"])
    t4 = P.table(rows, "Burden-budget planning: price chosen by bisection, model-predicted and realised burden, number of replicates (of 10) whose realised burden exceeded the budget, benefit and failure fraction. "
                 "An asterisk marks a mean realised burden above the budget with a t-based 95% interval that excludes it.", widths=[1.3, 1.3, 1.7, 2.0, 2.6, 2.1, 2.0, 1.6, 1.6], font=7.5)
    P.p(f"Table {t4} is the safety-margin result. With no margin the model-predicted burden is on target by construction (it is at or below the budget) but the realised burden is higher in every cell: the ratio of realised to predicted is {ratio[mi[0.0]].min():.2f} to {ratio[mi[0.0]].max():.2f} at m = 0 "
        f"and {ratio.min():.2f} to {ratio.max():.2f} over all cells. Consequently the budget was exceeded in {viol_n[mi[0.0], bi[2]]}, {viol_n[mi[0.0], bi[4]]} and {viol_n[mi[0.0], bi[6]]} of {NSEED} replicates for budgets of 2, 4 and 6. "
        + (f"(For the budget of 6 and no margin the model's unconstrained policy already satisfies the budget, so the price is 0, and the realised burden still exceeds it in {viol_n[mi[0.0], bi[6]]} of {NSEED} replicates.) " if unconstrained else "")
        + f"A 25% margin brought exceedances to {viol_n[mi[0.25], bi[2]]}, {viol_n[mi[0.25], bi[4]]} and {viol_n[mi[0.25], bi[6]]}, and a 50% margin to {viol_n[mi[0.5], bi[2]]}, {viol_n[mi[0.5], bi[4]]} and {viol_n[mi[0.5], bi[6]]}. "
        f"The calibration rule of Eq. (6) predicts that a margin of at least ρ − 1, that is {', '.join(f'{100 * x:.0f}%' for x in req_margin)} for budgets 2, 4 and 6 (from the m = 0 row), is needed on average; "
        f"this fits the counts, since 25% was insufficient and 50% was sufficient or nearly so for every budget.")
    f5 = P.fig(fig(I, "fig_budget"), "Realised burden (left; black bars: budget) and benefit (right) by budget and margin (10 replicates; 95% CI).", 11.0)
    P.p(f"Figure {f5} shows the same in the true-simulator units. The mean realised burden shrinks with the margin: at a 25% margin it is {REAL[mi[0.25], bi[2]]:.2f}, {REAL[mi[0.25], bi[4]]:.2f} and {REAL[mi[0.25], bi[6]]:.2f} for budgets 2, 4 and 6, and at 50% it is {REAL[mi[0.5], bi[2]]:.2f}, {REAL[mi[0.5], bi[4]]:.2f} and {REAL[mi[0.5], bi[6]]:.2f}. A mean at or just under the budget still leaves many replicates above it, because replicates scatter around the mean (the error bars), which is why exceedances persist at 25%. "
        f"The price of safety is visible on the right: a 50% margin instead of none costs {ben_cost[bi[2]]:.2f}, {ben_cost[bi[4]]:.2f} and {ben_cost[bi[6]]:.2f} units of benefit and raises the kidney-failure fraction by {fail_cost[bi[2]]:.1f}, {fail_cost[bi[4]]:.1f} and {fail_cost[bi[6]]:.1f} percentage points at budgets 2, 4 and 6. "
        "Looser budgets are cheaper to protect than tight ones, consistent with a concave frontier: the same proportional tightening removes a smaller slice of benefit where the frontier is flatter.")
    f6 = P.fig(fig(I, "fig_violation"), "Share of replicates whose realised burden exceeded the budget, by budget and margin (10 replicates).", 6.0)
    P.p(f"Figure {f6} summarises the exceedance rates as percentages. With 10 replicates each point has a granularity of 10 percentage points and a wide binomial interval, so the figure should be read as monotone evidence (more margin, fewer violations) and not as a precise rate. "
        f"Even with a 50% margin the budget of 2 was exceeded in {viol_n[mi[0.5], bi[2]]} of {NSEED} replicates, which a user who needs a hard guarantee would not accept; a distribution-free guarantee would require a conformal-style or hold-out calibration, which we did not do.")

    P.h1("Discussion")
    P.p(f"**What the study shows.** Three findings are well supported. (1) A burden price traces a monotone, concave benefit-burden frontier whose failure cost rises steeply beyond a price of about 0.5 to 1 (Table {t2}, Figure {f4}). "
        f"(2) Planning to a burden budget in the estimated model under-delivers: realised burden was {ratio.min():.2f} to {ratio.max():.2f} times the prediction in every cell, so a budget needs a margin. "
        f"(3) The margin can be calibrated from the predicted-versus-realised ratio, and the required size ({', '.join(f'{100 * x:.0f}%' for x in req_margin)}) matches the observed exceedance counts.")
    P.p(f"**What it does not show.** The learned policy is not clearly better than a trivial one. At unit price 'always enhanced' scored {JF['Always enhanced']:.2f} against DP's {S['DP']['J']:.2f} (benchmark) and {Bn[li[1]] - Cn[li[1]]:.2f} (sweep, price 1), with a rule-evaluation noise of {abs(rule_noise):.2f} "
        f"between 400 and 600 evaluation episodes; differences of that size cannot be resolved. Along the frontier, 'always enhanced' beats the learned points at its burden. The learned Lagrangian is therefore not an improvement on simple constant policies in this simulator, although "
        f"it is the only one of these that can meet an arbitrary budget. The reason is likely that the logged data are too narrow to discover that intensive care is valuable outside the region where the rule already uses it (Figure {f3}); we did not test it by changing the behaviour policy.")
    P.p("**Practical use.** A deployment would report the frontier, the calibration ratio and the exceedance rates together, choose the margin from the ratio, and compare with the best constant policy before claiming a learned benefit. It would also check the price-consistency property (Table "
        f"{t3}) as a diagnostic on the model; a failure at high prices is a sign that the model is unreliable there and that policies at those prices should not be trusted.")
    limitations(P, [
        "**Linear scalarisation of failure.** Kidney failure enters the benefit channel as a −20 term, so the 'benefit' conflates slowing decline and avoiding failure. A hard constraint on failure rate would be a different problem.",
        "**Deterministic policies and approximate budgets.** Randomised policies are needed to meet a budget exactly; we use bisection on deterministic policies and accept the discretisation gap.",
        "**Margin by calibration on the same data.** The ratio ρ is estimated in the same replicates in which exceedances are counted, so the agreement between the required margin and the counts is partly circular; no held-out calibration was done.",
        f"**Few replicates.** Ten replicates give exceedance rates in steps of 10 percentage points; the benchmark uses 5 replicates and 1.96-standard-error intervals that are narrower than t-intervals by a factor of {stats.t.ppf(0.975, cfg['seeds'] - 1) / 1.96:.2f}.",
        "**Evaluation noise.** The same rule is scored differently with 400 and 600 evaluation episodes; small differences between policies are not resolved.",
    ])
    ethics_future(P, "(1) Compare the learned frontier with the best constant and randomised mixtures of constant policies at each burden; (2) widen the logged behaviour (more exploration at high eGFR) and test whether the frontier then overtakes 'always enhanced'; "
                  "(3) calibrate the margin on held-out episodes and report a distribution-free guarantee; (4) solve the CMDP with randomised policies by linear programming and compare; (5) apply the adapter to a real CKD table, where burden would have to be defined from visit counts and costs.")
    P.h1("Conclusion")
    P.p(f"On a simulated CKD cohort, Lagrangian dynamic programming traces a monotone benefit-burden frontier (burden {Cn[0]:.2f} to {Cn[-1]:.2f}; kidney-failure fraction {100 * Fl[0]:.1f}% to {100 * Fl[-1]:.1f}%) and, at unit price, DP, Q-learning and MC all beat a nephrologist-style rule ({S['DP']['J']:.2f}, {S['Q-learning']['J']:.2f}, {S['Monte Carlo']['J']:.2f} versus {rule_J:.2f}). "
        f"The gain is mostly available from a constant 'always enhanced' policy ({JF['Always enhanced']:.2f}), which also lies above the learned frontier at its burden. Planning to a budget in the estimated model under-delivers by a factor of {ratio.min():.2f} to {ratio.max():.2f}, "
        f"and a safety margin sized from the predicted-to-realised ratio (about {100 * req_margin.min():.0f}% to {100 * req_margin.max():.0f}%) is needed to meet it, at a cost of {ben_cost.min():.2f} to {ben_cost.max():.2f} units of benefit for a 50% margin. These are simulator results, not clinical evidence.")
    defense(P, [
        ("Is a learned policy that merely matches 'always enhanced' worth having?",
         f"Not on this evidence. At unit price the benchmark DP scored {S['DP']['J']:.2f} and 'always enhanced' {JF['Always enhanced']:.2f}, within the evaluation noise of {abs(rule_noise):.2f} that we measured on the rule, and at its burden the constant policy is above the learned frontier by {fx['Always enhanced']['benefit'] - fr_at(fx['Always enhanced']['burden']):.2f}. "
         "What the Lagrangian approach adds is the ability to meet a stated budget, which constant policies cannot do."),
        ("Why does the realised burden exceed the model's prediction?",
         "The code shows a plausible mechanism: unsupported state-action pairs carry zero burden in the model and are absorbing, while in deployment the rule is used in unseen states and incurs burden. The measured ratio (Table "
         f"{t4}) is evidence that the bias exists and is of order {100 * (ratio.min() - 1):.0f}% to {100 * (ratio.max() - 1):.0f}%; we did not isolate the mechanism by experiment."),
        ("Is a 50% margin the right choice?",
         f"It is the largest margin we tried. It removed exceedances at budgets 4 and 6 ({viol_n[mi[0.5], bi[4]]} and {viol_n[mi[0.5], bi[6]]} of {NSEED}) but not at 2 ({viol_n[mi[0.5], bi[2]]} of {NSEED}), where the calibration ratio ({ratio[mi[0.0], bi[2]]:.2f}) is largest. The margin should be sized from ρ, per budget."),
        ("Why does the policy trained at a high price fail the consistency check?",
         f"Table {t3} shows the policy trained at λ = 4 is beaten, at price 4, by the policy trained at 16 (shortfall {diag_loss[li[4]]:.2f}). Either the model mis-prices burden saving at high prices or those policies are noisy; the results do not separate the explanations."),
        ("Can the results be used to choose a treatment intensity?",
         "No. The simulator's dynamics, burden values and failure penalty are invented; the study is about how a Lagrangian planner behaves and how it should be protected against its own model error."),
    ])
    repro(P, 6, ["fig_bars", "fig_learning", "fig_policies", "fig_agreement", "fig_pareto", "fig_failure", "fig_budget", "fig_violation"])
    code_appendix(P, 6, [domains.CKD])
    rl_core_appendix(P)
    P.refs()
    path = f"{PAP}/Paper06_CKD_Lagrangian_Pareto.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
