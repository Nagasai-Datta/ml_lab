import math
import numpy as np
from scipy import stats
from docx_helper import Paper
from paper_common import *

I = 8


def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def build():
    import domains
    r = load(I)
    SC, J, W = r["SC"], r["J"], r["worst"]
    std, ex, SW = r["standard"], r["extra"], r["sweep"]
    NS = list(SW.keys())                                   # "200", "400", "1000", "3000"
    nrep = ex["config"]["seeds"]
    tfac = stats.t.ppf(0.975, nrep - 1) / 1.96
    i1, i15, i2 = SC.index(1.0), SC.index(1.5), SC.index(2.0)
    RC = ["Robust DP (c=0.5)", "Robust DP (c=1)", "Robust DP (c=2)", "Robust DP (c=4)"]
    nsweep = 2 * len(NS)
    bonf = 0.05 / nsweep
    rn, rw = r["rob2_vs_nom_nominal"], r["rob2_vs_nom_worst"]
    boot = r["boot"]
    D = domains.Discharge
    prob = np.outer(D.BASE, D.STAB)
    c_theory = math.sqrt(2 * ((ex["n_states"] + 1) * math.log(2) + math.log(20)))

    # ---- learned vs rule at training distribution (standard benchmark; rule is a single evaluation)
    rule_std = std["Rule"]["J"]
    VR = {}
    for m in ("DP", "Monte Carlo", "Q-learning"):
        j = np.asarray(ex["methods"][m]["J"], float)
        VR[m] = dict(mean=float(j.mean() - rule_std), ci=_ci(j), p=float(stats.ttest_1samp(j, rule_std).pvalue))
    # accounting identity (undiscounted) for DP versus rule: stay days, readmissions, follow-ups
    a_los = std["Rule"]["info"]["los"] - std["DP"]["info"]["los"]
    a_re = (std["DP"]["info"]["readmit"] - std["Rule"]["info"]["readmit"])
    a_fu = (std["DP"]["info"]["followup"] - std["Rule"]["info"]["followup"])
    gain_days, cost_re, cost_fu = a_los, 25 * a_re, 2 * a_fu
    breakeven = (gain_days - cost_fu) / a_re

    # ---- model-predicted versus true advantage of the oracle policy
    mo = ex["J_oracle_model"]
    pred, true = mo["Oracle DP"] - mo["Rule"], std["Oracle DP"]["J"] - std["Rule"]["J"]
    og = np.asarray(ex["optimism_gap"], float)
    og_p = float(stats.ttest_1samp(og, 0).pvalue)

    # ---- shift behaviour
    drop_rule = J["Rule"]["mean"][i1] - J["Rule"]["mean"][i2]
    drop_dp = J["DP (nominal)"]["mean"][i1] - J["DP (nominal)"]["mean"][i2]
    sens_rule = 25 * r["readmit"]["Rule"]
    sens_dp = 25 * r["readmit"]["DP (nominal)"]
    vs_rule2 = {m: dict(d=J[m]["mean"][i2] - J["Rule"]["mean"][i2], ci=J[m]["ci"][i2]) for m in J}
    below2 = [m for m in J if m != "Rule" and vs_rule2[m]["d"] < 0 and abs(vs_rule2[m]["d"]) > vs_rule2[m]["ci"]]
    above = {m: [SC[i] for i in range(len(SC)) if J[m]["mean"][i] > J["Rule"]["mean"][i]] for m in J if m != "Rule"}

    P = Paper("Does Robust Dynamic Programming Help Readmission-Aware Discharge Planning? A Data-Size Sweep Under Readmission-Risk Shift for DP, Monte Carlo and Q-learning",
              "hospital discharge; readmission; robust dynamic programming; offline reinforcement learning; distribution shift; pessimism; Monte Carlo control; Q-learning",
              running="Robust discharge planning", refdict=REFS)

    P.abstract(
        "Deciding when to discharge a patient, and whether to add a follow-up visit, trades days in hospital against the risk of readmission. A policy learned from logs "
        "is only as good as the readmission risks estimated from them, and those risks can differ at deployment. We ask whether robust dynamic programming (DP), which plans against the worst case "
        "in an L1 ball around the estimated transitions, protects a learned discharge policy. On a simulator mirroring the Diabetes 130-US Hospitals table (the Kaggle site was unreachable, HTTP 403, so no real file was used) "
        f"we compare nominal DP, four robust DP radii, Monte Carlo control and Q-learning over {nrep} replicates, scoring every policy under readmission-risk multipliers from {SC[0]:g} to {SC[-1]:g} "
        f"and over logged-data sizes from {NS[0]} to {NS[-1]} episodes. At the training risk the learned policies beat a guideline rule (DP {std['DP']['J']:.2f} versus {rule_std:.2f}) by discharging earlier "
        f"({std['DP']['info']['los']:.1f} versus {std['Rule']['info']['los']:.1f} days) at the price of more readmissions ({100 * std['DP']['info']['readmit']:.1f}% versus {100 * std['Rule']['info']['readmit']:.1f}%). "
        f"Robust DP did not protect against risk shift: its worst-case return was {rw['mean']:+.2f} ± {rw['ci']:.2f} relative to nominal DP ({pstr(rw['p'])}), and its readmission rate rose with the radius. "
        f"It did help with scarce data when risk was unchanged (+{SW['200']['diff_c2_1.0']['mean']:.2f} at {NS[0]} episodes, {pstr(SW['200']['diff_c2_1.0']['p'])}) and the benefit vanished by {NS[-1]} episodes "
        f"({SW['3000']['diff_c2_1.0']['mean']:+.2f}, {pstr(SW['3000']['diff_c2_1.0']['p'])}); under a 1.5-fold risk shift it never differed from nominal DP. All results are simulated and carry no clinical meaning.")

    # ------------------------------------------------------------------ 1
    P.h1("Introduction")
    P.p("Every inpatient stay ends with a decision: keep the patient another day, discharge, or discharge with a follow-up arrangement such as a call, a clinic slot or home visit. "
        "Staying longer costs beds and exposes the patient to hospital risks; leaving early raises the chance of readmission. Readmission risk is routinely modelled from records [@kansagara], "
        "and the sequence of keep-or-discharge decisions is a natural Markov decision process [@puterman,schaefer]. Dynamic programming (DP), Monte Carlo (MC) control and Q-learning "
        "[@sb,bellman,watkins] can in principle learn such a policy from logged stays.")
    P.p("The difficulty is that the readmission probabilities in the model are estimates, and the population to which the policy is applied may carry different risks from the one that generated the logs: "
        "a different case mix, a different community support system, a different season. Robust MDPs, in which the planner maximises the worst-case value over an uncertainty set around the estimated "
        "transitions [@iyengar,nilim], are the textbook response, and the offline-RL literature has found related pessimism useful when data are limited [@jin,kumar,levine]. "
        "This paper asks in a controlled simulator how much a robust planner actually protects, against what, and how that depends on the amount of logged data.")
    P.p("**Contributions.** (i) A 108-state, 3-action discharge simulator in which the readmission risk can be rescaled at deployment. (ii) A comparison of nominal DP, four robust-DP radii, MC control and Q-learning, "
        f"{nrep} replicates, scored exactly under five risk multipliers. (iii) A data-size sweep (200 to 3,000 logged episodes) of robust versus nominal DP with and without risk shift, with paired tests. "
        "(iv) A bootstrap stability analysis and a model-predicted-versus-true diagnostic. (v) Candid findings: robust DP does not help against risk shift in this simulator, helps only with scarce data at unchanged risk, "
        "and the learned policies' advantage over a rule comes from earlier discharge and more readmissions, a trade-off dictated by hand-set reward weights.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached from the build environment (all requests returned HTTP 403), so no real dataset was analysed. The simulator mirrors the variables of the "
        "Diabetes 130-US Hospitals table (length of stay, readmission) with hand-specified dynamics. Every number below is a simulated result about algorithms, and no clinical claim is made.")

    # ------------------------------------------------------------------ 2
    P.h1("Related work")
    P.p("**Robust MDPs.** Iyengar and Nilim and El Ghaoui introduced robust dynamic programming for MDPs with uncertain transition matrices and showed that, for rectangular uncertainty sets, "
        "the Bellman operator remains a contraction and the inner minimisation is tractable [@iyengar,nilim]. Uncertainty sets can be built from concentration inequalities for the empirical distribution; "
        "the L1 deviation bound of Weissman and colleagues gives a radius that shrinks as one over the square root of the sample size [@weissman]. We use that scaling with a tunable constant.")
    P.p("**Pessimism and offline RL.** When data are logged by another policy, value estimates for poorly covered actions are exploited by the optimiser. Remedies include restricting policies to the data's support "
        "[@fujimoto], conservative value estimates [@kumar] and provable pessimism [@jin]; surveys of RL in healthcare stress careful evaluation and warn that improvements over clinician practice are not guaranteed [@levine,yu,gottesman]. "
        "Our robust DP is a pessimism device of this family, so its behaviour in a data-size sweep is a direct test of the offline-RL claim that pessimism matters most when data are scarce.")
    background(P)

    # ------------------------------------------------------------------ 4
    P.h1("Problem formulation: a discharge simulator with risk shift")
    P.p("A patient arrives with a risk tier (0 to 3; probabilities 0.35, 0.30, 0.20, 0.15) and a clinical-stability level (unstable, borderline or stable; probabilities 0.45, 0.40, 0.15). Each day the clinician "
        "chooses to extend the stay, discharge, or discharge with follow-up. Extending costs -1 and may improve stability (probability 0.40 - 0.05 risk, if not already stable) or worsen it (probability 0.05 + 0.03 risk); a stay is "
        "forced to end at day 9. Discharge ends the episode; a readmission occurs with probability equal to the baseline for the risk tier times a stability multiplier times 0.6 if follow-up was arranged, "
        "times the deployment risk multiplier, capped at 0.95, and costs -25. Follow-up itself costs -2. The state is the risk tier, the days so far (capped at 8) and the stability level.")
    rows = [["Risk tier", "Unstable (×1.9)", "Borderline (×1.0)", "Stable (×0.55)"]]
    for k in range(4):
        rows.append([f"Tier {k} (baseline {D.BASE[k]:.2f})"] + [f"{100 * prob[k, j]:.1f}%" for j in range(3)])
    t_p = P.table(rows, "Readmission probability after plain discharge by risk tier and stability at the training risk (multiplier 1.0); follow-up multiplies these by 0.6.", widths=[5.4, 3.7, 3.7, 3.7], font=8.5)
    P.table([["Element", "Definition"],
             ["State (108)", "risk tier (4) × days in hospital 0-8 (9) × stability (3)"],
             ["Actions (3)", "extend the stay; discharge; discharge with follow-up"],
             ["Reward", "-1 per extension day; -25 per readmission; -2 per follow-up"],
             ["Discount, horizon", "γ = 0.99; at most 10 decisions (forced discharge at day 9)"],
             ["Deployment shift", f"readmission probabilities multiplied by {', '.join(f'{x:g}' for x in SC)} (capped at 0.95); training risk is 1.0"],
             ["Behaviour policy / rule", "rule: if stable after day 2 discharge (with follow-up for tiers 2-3); from day 6 discharge with follow-up; otherwise extend. Behaviour = rule with epsilon = 0.25 random actions"],
             ["Dataset mirrored", "Diabetes 130-US Hospitals (time in hospital, readmission label)"]],
            "The discharge MDP.", widths=[3.4, 13.0], font=8.5)
    P.p(f"Table {t_p} shows that readmission risk spans a factor of {prob.max() / prob.min():.0f} across states, from {100 * prob.min():.1f}% to {100 * prob.max():.0f}%. "
        "The reward weights are an arbitrary price list: one readmission is equivalent to 25 days of stay, and a follow-up to 2 days. All conclusions about which policy is better are conditional on these weights, "
        "and we return to this in the Discussion. Because readmission is an outcome of the discharge action, it enters the model as part of the reward, not as a transition.")

    # ------------------------------------------------------------------ 5 method
    P.h1("Method: L1-ball robust DP and the data-size sweep")
    P.p("Let P̂(·|s,a) be the empirical transition distribution estimated from n_sa logged observations of (s,a). Robust DP replaces the expectation in the Bellman equation by a worst case over an L1 ball:")
    P.eq("V(s) = max_a [ R̂(s,a) + γ min_{p: ||p − P̂(·|s,a)||₁ ≤ ρ_sa} Σ_{s'} p(s') V(s') ]", "3")
    P.p("with radius ρ_sa = min(2, c / √n_sa), so that radii shrink as an (s,a) pair is observed more often. ")
    P.p("The inner minimisation has a closed form: the adversary removes up to ρ_sa / 2 of probability mass from the highest-value next states and places it on the lowest-value one, which is how rl_core.robust_value_iteration implements it "
        f"[@nilim]. The constant c sets the conservatism: c = 0 is nominal DP, and we use c in {{0.5, 1, 2, 4}}. A concentration bound for the full simplex over {ex['n_states'] + 1} next states would require c of about {c_theory:.1f} "
        f"at 95% confidence [@weissman], so the radii used here are far smaller than a guarantee would demand; this is a mild, tunable regulariser, not a certified robust policy. For example, with n_sa = 25 observations "
        f"and c = 2 the radius is {2 / math.sqrt(25):.2f}, and with 400 observations it is {2 / math.sqrt(400):.2f}.")
    P.p("**What the uncertainty set covers.** The set perturbs transition probabilities only; the reward R̂(s,a) is left at its point estimate. In this simulator the readmission risk, the quantity that shifts at deployment, "
        "sits in the reward of the discharge actions, whereas the transitions that carry uncertainty are those of the extend action (stability changes and the day counter). The robust planner can therefore be pessimistic "
        "about extending but cannot be pessimistic about discharging. This is a property of the implementation, and it frames our expectation: robust DP, as built, is not designed to hedge a shift in readmission risk.")
    P.code("""Algorithm 1  Nominal versus robust DP from discharge logs
 input : logged stays (state, action, reward, next state, done), discount gamma, radius constant c, support threshold n_min = 10
 1. counts N(s,a,s'), n_sa = sum_s' N;  R(s,a) = mean reward;  unsupported pairs (n_sa < n_min) absorb with the worst observed reward
 2. nominal:  V <- max_a [ R + gamma * sum_s' P_hat * V ]   until convergence
 3. robust:   rho_sa = min(2, c / sqrt(n_sa));  inner step: move up to rho_sa/2 mass from the highest-V next states to the lowest-V state
              V <- max_a [ R + gamma * min_p p.V ]   until convergence
 4. in states never observed, fall back to the modal logged action
 5. deploy pi; score it under every readmission multiplier m in {0.8, 1, 1.25, 1.5, 2} in the TRUE simulator, common random numbers""")
    P.p("Two hypotheses frame the experiments: (H1) a larger radius improves the worst-case return over deployment shifts; (H2) robustness helps most when logged data are scarce, because radii shrink as n_sa grows. "
        "A first-order argument also gives a baseline expectation for the shift itself. If a policy's readmission rate at training risk is r, multiplying the risk by m changes its return by about -25 γ̄ r (m − 1), "
        "where γ̄ is a discount factor near one; a policy that discharges earlier has a higher r and therefore loses more when risk rises.")

    protocol(P, [["Item", "Setting"],
                 ["Simulator", "discharge MDP; 108 states, 3 actions; γ = 0.99; readmission multiplier 0.8 to 2.0"],
                 ["Logged data", f"{ex['config']['n_eps']:,} episodes per replicate from the epsilon-noisy rule (main comparison)"],
                 ["Oracle model", f"{ex['config']['n_oracle']:,} episodes, for the oracle DP and exact learning-curve scoring"],
                 ["Replicates", f"{nrep} independent data draws and learner seeds per condition"],
                 ["Learners", f"nominal DP, robust DP (c = 0.5, 1, 2, 4), first-visit MC, Q-learning; {ex['config']['steps']:,} steps (training-risk benchmark) or 120,000 steps (shift experiment)"],
                 ["Scoring", f"{ex['config']['n_eval']:,} episodes per policy at training risk (benchmark); 800 episodes per policy and multiplier (shift experiment)"],
                 ["Data-size sweep", f"{', '.join(NS)} logged episodes; nominal versus robust (c = 1, 2); risk multiplier 1.0 and 1.5; {nrep} replicates (different seeds); 800 evaluation episodes"],
                 ["Bootstrap stability", "60 episode-level resamples of one replicate's data (1,000 episodes); agreement with the full-data policy in states with at least 5 visits"]],
             ex["config"]["steps"], nrep, ex["config"]["n_eval"],
             extra=f"**Statistical conventions.** Confidence half-widths are 1.96 standard errors over the {nrep} replicates (a t-based interval would be {tfac:.2f} times wider; this paper uses the normal form, not the 't-style' intervals of the generic description above). "
                   f"The rule is a single deterministic evaluation and has no replicate-based interval. Only the c = 2 comparison was paired in the code: robust versus nominal at multiplier 1.0 and at the worst shift (the same comparison here, since multiplier 2.0 is the worst for every replicate), "
                   f"and the {nsweep} sweep comparisons; the Bonferroni threshold for the sweep family is {bonf:.4f}. Comparisons of other radii are read from means and intervals and are not tested.")

    # ------------------------------------------------------------------ results
    P.h1("Results")
    P.h2("At the training risk: learned policies beat the rule by discharging earlier")
    rows = [["Method", "Return (mean ± 95% CI)", "Readmission rate", "Days in hospital", "Follow-up rate"]]
    for k in ("Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP"):
        v = std[k]
        rows.append([NICE.get(k, k), pm(v["J"], v["ci"]) if v["ci"] > 0 else f"{v['J']:.2f}", f"{100 * v['info']['readmit']:.1f}%", f"{v['info']['los']:.2f}", f"{100 * v['info']['followup']:.0f}%"])
    t1 = P.table(rows, f"Training-risk benchmark ({nrep} replicates, {ex['config']['n_eval']:,} evaluation episodes). The rule and the oracle are single evaluations; the guideline rule is the same as the logged behaviour policy apart from the epsilon noise.",
                 widths=[4.8, 3.6, 2.6, 2.6, 2.6], font=8.5)
    P.p(f"Table {t1} shows that all three learners beat the rule: DP by {VR['DP']['mean']:+.2f} ± {VR['DP']['ci']:.2f} ({pstr(VR['DP']['p'])}), Monte Carlo by {VR['Monte Carlo']['mean']:+.2f} ± {VR['Monte Carlo']['ci']:.2f} ({pstr(VR['Monte Carlo']['p'])}) "
        f"and Q-learning by {VR['Q-learning']['mean']:+.2f} ± {VR['Q-learning']['ci']:.2f} ({pstr(VR['Q-learning']['p'])}); the oracle DP reaches {std['Oracle DP']['J']:.2f}. The learners are statistically indistinguishable from each other. "
        f"The mechanism is visible in the other columns: DP shortens the stay from {std['Rule']['info']['los']:.2f} to {std['DP']['info']['los']:.2f} days and readmissions rise from {100 * std['Rule']['info']['readmit']:.1f}% to {100 * std['DP']['info']['readmit']:.1f}%, "
        f"with follow-up used in {100 * std['DP']['info']['followup']:.0f}% of stays rather than {100 * std['Rule']['info']['followup']:.0f}%. Accounting for the weights, {gain_days:.2f} days saved is worth {gain_days:.2f}, the extra readmissions cost {cost_re:.2f} "
        f"and the extra follow-ups {cost_fu:.2f}, a net of about {gain_days - cost_re - cost_fu:+.2f} (undiscounted; the measured difference of {VR['DP']['mean']:+.2f} is discounted at 0.99). "
        f"The gain is entirely a consequence of the price list: the same DP policy would stop beating the rule if one readmission were worth more than about {breakeven:.0f} days of stay rather than 25. "
        f"A health system that regards a {100 * (std['DP']['info']['readmit'] / std['Rule']['info']['readmit'] - 1):.0f}% relative rise in readmissions as unacceptable would not adopt this policy whatever its return.")
    cm, cq = np.array(ex["curves"]["Monte Carlo"]), np.array(ex["curves"]["Q-learning"])
    def first_above(c):
        h = np.flatnonzero(c[:, :, 1].mean(0) > mo["Rule"])
        return c[0, h[0], 0]
    f_lc = P.fig(fig(I, "fig_learning"), f"Learning curves on the oracle MDP (mean ± s.d. over replicates). The mean MC curve passes the rule's model value (dotted) by {first_above(cm):,.0f} steps and the Q-learning curve by {first_above(cq):,.0f}; both plateau below the oracle DP optimum (dashed).", 7.5)
    P.p(f"Figure {f_lc} shows quick convergence: MC reaches {cm[:, 1, 1].mean():.2f} after {cm[0, 1, 0]:,.0f} steps and Q-learning {cq[:, 1, 1].mean():.2f}, ending at {cm[:, -1, 1].mean():.2f} and {cq[:, -1, 1].mean():.2f}. "
        f"Q-learning agrees with the DP policy in {100 * np.mean(ex['agree']['Q-learning']):.0f}% of well-visited states and MC in {100 * np.mean(ex['agree']['Monte Carlo']):.0f}%. ")

    P.h2("Readmission-risk shift: does robustness protect?")
    rows = [["Method"] + [f"×{s:g}" for s in SC]]
    for k in J:
        rows.append([k] + [f"{J[k]['mean'][i]:.2f}" if J[k]["ci"][i] == 0 else pm(J[k]["mean"][i], J[k]["ci"][i]) for i in range(len(SC))])
    t2 = P.table(rows, f"True return under deployment readmission-risk multipliers (mean ± 95% CI over {nrep} replicates; ×1 is the training risk). The rule has no replicate interval.",
                 widths=[3.9, 2.5, 2.5, 2.5, 2.5, 2.5], font=7.5)
    f_sh = P.fig(fig(I, "fig_shift"), "Return versus deployment readmission-risk multiplier for the rule and the learned policies (error bars: 95% intervals over replicates).", 9.5)
    P.p(f"Table {t2} and Figure {f_sh} contain the main negative result. Every policy loses return roughly linearly as risk rises, and the learned policies lose faster: the rule falls by {drop_rule:.2f} between ×1 and ×2, nominal DP by {drop_dp:.2f}. "
        f"This is the first-order effect predicted in Section 5: 25 times the training readmission rate is {sens_rule:.2f} for the rule and {sens_dp:.2f} for nominal DP, a difference of {sens_dp - sens_rule:.2f} against an observed difference of {drop_dp - drop_rule:.2f}. "
        f"The learned policies stay above the rule up to ×{max(above['DP (nominal)']):g} but lose the advantage at ×{SC[i2]:g}, where nominal DP scores {J['DP (nominal)']['mean'][i2]:.2f} ± {J['DP (nominal)']['ci'][i2]:.2f} against {J['Rule']['mean'][i2]:.2f} for the rule "
        f"({vs_rule2['DP (nominal)']['d']:+.2f}, within the interval). At ×{SC[i2]:g} the methods whose interval excludes the rule's value are: {', '.join(below2) if below2 else 'none'}, all below it. "
        "Robust variants are not above nominal DP at any multiplier by more than their intervals; if anything the larger radii are lower at the highest multiplier. This disposes of H1 as stated.")
    rows = [["Method", "Worst-case return (mean ± 95% CI)", "Readmission rate at ×1", "Follow-up rate at ×1"]]
    for k in J:
        rows.append([k, pm(W[k]["mean"], W[k]["ci"]) if W[k]["ci"] > 0 else f"{W[k]['mean']:.2f}", f"{100 * r['readmit'][k]:.1f}%", f"{100 * r['followup'][k]:.0f}%"])
    t3 = P.table(rows, "Worst return over the five multipliers (multiplier ×2 in every replicate), and readmission and follow-up rates at the training risk, shift experiment.", widths=[4.4, 5.2, 3.5, 3.3], font=8.5)
    P.p(f"Table {t3} puts numbers on this. The paired comparison of robust DP (c = 2) with nominal DP gives {rw['mean']:+.2f} ± {rw['ci']:.2f} in the worst case ({pstr(rw['p'])}) and {rn['mean']:+.2f} ± {rn['ci']:.2f} at the training risk ({pstr(rn['p'])}): "
        f"neither is significant, and the worst-case point estimate is unfavourable to robust DP. Over the radii the worst-case return runs from {W['DP (nominal)']['mean']:.2f} (nominal) through {W['Robust DP (c=0.5)']['mean']:.2f}, {W['Robust DP (c=1)']['mean']:.2f}, {W['Robust DP (c=2)']['mean']:.2f} to {W['Robust DP (c=4)']['mean']:.2f} (c = 4), "
        f"a monotone decline whose end points differ by {W['DP (nominal)']['mean'] - W['Robust DP (c=4)']['mean']:.2f}; the intervals are about ±{W['Robust DP (c=4)']['ci']:.2f}, so only the largest differences are distinguishable. "
        f"The readmission rate at the training risk also rises monotonically with the radius, from {100 * r['readmit']['DP (nominal)']:.1f}% (nominal) to {100 * r['readmit']['Robust DP (c=4)']:.1f}% (c = 4). "
        "This is consistent with the design property noted in Section 5: pessimism applied to the extend action alone makes staying look worse and pushes the policy toward earlier discharge, which exposes more patients to readmission. We did not record stay lengths for the robust variants, so this explanation is untested.")
    f_st = P.fig(fig(I, "fig_stability"), "Left: worst-case return over deployment shifts by method. Right: bootstrap agreement of nominal and robust (c = 2) DP policies with the full-data policy (60 resamples).", 11.0)
    P.p(f"Figure {f_st} adds a stability check, which also goes against the usual argument for robustness. Under episode-level bootstrap resampling of one replicate's data, nominal DP reproduces its full-data policy in {100 * boot['nominal']:.0f}% ± {100 * boot['nominal_sd']:.0f}% of well-visited states (mean ± s.d.), "
        f"whereas robust DP (c = 2) does so in {100 * boot['robust']:.0f}% ± {100 * boot['robust_sd']:.0f}%. Robust DP is less, not more, stable by this measure. A single data draw was used, so the comparison is descriptive.")

    P.h2("When does robustness help? The data-size sweep")
    rows = [["Logged episodes", "Nominal ×1", "Robust c=1 ×1", "Robust c=2 ×1", "c=2 − nominal, ×1 (p)", "Nominal ×1.5", "Robust c=2 ×1.5", "c=2 − nominal, ×1.5 (p)"]]
    for n in NS:
        s = SW[n]
        rows.append([n, f"{s['nominal']['1.0']:.2f}", f"{s['robust c=1']['1.0']:.2f}", f"{s['robust c=2']['1.0']:.2f}",
                     f"{s['diff_c2_1.0']['mean']:+.2f} ± {s['diff_c2_1.0']['ci']:.2f} ({'<0.001' if s['diff_c2_1.0']['p'] < 0.001 else format(s['diff_c2_1.0']['p'], '.3f')})",
                     f"{s['nominal']['1.5']:.2f}", f"{s['robust c=2']['1.5']:.2f}",
                     f"{s['diff_c2_1.5']['mean']:+.2f} ± {s['diff_c2_1.5']['ci']:.2f} ({'<0.001' if s['diff_c2_1.5']['p'] < 0.001 else format(s['diff_c2_1.5']['p'], '.3f')})"])
    t4 = P.table(rows, f"Data-size sweep (mean true return over {nrep} replicates; paired differences ± 95% CI with p-values). Robust c = 1 was not paired-tested.", widths=[1.5, 1.4, 1.7, 1.7, 3.3, 1.5, 1.7, 3.3], font=7)
    f_ds = P.fig(fig(I, "fig_datasweep"), "Robust (c = 2) minus nominal DP return versus logged episodes, with no shift (left) and a 1.5-fold readmission-risk shift (right). Error bars: 95% intervals.", 11.5)
    sig = [n for n in NS if SW[n]["diff_c2_1.0"]["p"] < bonf]
    sig05 = [n for n in NS if SW[n]["diff_c2_1.0"]["p"] < 0.05]
    P.p(f"Table {t4} and Figure {f_ds} are where robustness earns its keep, within limits. With unchanged risk, robust DP (c = 2) beats nominal DP by {SW['200']['diff_c2_1.0']['mean']:+.2f} ± {SW['200']['diff_c2_1.0']['ci']:.2f} at {NS[0]} episodes ({pstr(SW['200']['diff_c2_1.0']['p'])}) "
        f"and {SW['400']['diff_c2_1.0']['mean']:+.2f} ± {SW['400']['diff_c2_1.0']['ci']:.2f} at {NS[1]} ({pstr(SW['400']['diff_c2_1.0']['p'])}), the benefit shrinks to {SW['1000']['diff_c2_1.0']['mean']:+.2f} ± {SW['1000']['diff_c2_1.0']['ci']:.2f} at {NS[2]} ({pstr(SW['1000']['diff_c2_1.0']['p'])}) "
        f"and is {SW['3000']['diff_c2_1.0']['mean']:+.2f} ± {SW['3000']['diff_c2_1.0']['ci']:.2f} at {NS[3]} ({pstr(SW['3000']['diff_c2_1.0']['p'])}). At the Bonferroni threshold of {bonf:.4f} over the {nsweep} sweep comparisons, the sizes with a significant gain are {', '.join(sig)} episodes; "
        f"{', '.join(sig05)} are nominally significant. Robust c = 1 is in between, for example {SW['200']['robust c=1']['1.0']:.2f} against {SW['200']['nominal']['1.0']:.2f} nominal at {NS[0]} episodes (not tested). "
        f"The gain at 1,000 episodes in this sweep ({SW['1000']['diff_c2_1.0']['mean']:+.2f}) is consistent in sign with the main comparison at the same data size ({rn['mean']:+.2f} ± {rn['ci']:.2f}, {pstr(rn['p'])}), a separate set of replicates in which it was not significant. "
        f"H2 is therefore supported, but only for unchanged risk. Under a 1.5-fold shift the difference is {SW['200']['diff_c2_1.5']['mean']:+.2f}, {SW['400']['diff_c2_1.5']['mean']:+.2f}, {SW['1000']['diff_c2_1.5']['mean']:+.2f} and {SW['3000']['diff_c2_1.5']['mean']:+.2f} for the four sizes, "
        f"with every interval containing zero (smallest p-value {min(SW[n]['diff_c2_1.5']['p'] for n in NS):.2f}). Robustness bought with scarce data does not carry over to a shift in readmission risk.")
    P.p("Why does pessimism help at small samples and not at large? A plausible, untested account is that nominal DP built from few stays is distorted by the support constraint: pairs seen fewer than ten times are completed with the worst observed reward, "
        "so rarely logged discharge actions look terrible and the planner over-extends stays; transition-side pessimism on extending counteracts this and moves the policy toward the earlier discharges that the benchmark rewards. "
        "As data grow every pair is supported, radii shrink as one over the square root of n_sa and the two planners coincide. If this account is right, the benefit is a correction of a small-sample artefact of the pipeline, not robustness to the world.")

    P.h2("Model-based diagnosis")
    rows = [["Quantity", "Value"],
            ["Model value of the oracle DP policy / of the rule (oracle MDP)", f"{mo['Oracle DP']:.2f} / {mo['Rule']:.2f}"],
            ["Model-predicted advantage of oracle DP over the rule", f"{pred:+.2f}"],
            ["True return of the oracle DP policy / of the rule", f"{std['Oracle DP']['J']:.2f} / {std['Rule']['J']:.2f}"],
            ["True advantage", f"{true:+.2f}"],
            ["Sign of predicted and true advantage agree?", "yes" if pred * true > 0 else "no"],
            [f"Optimism of DP trained on {ex['config']['n_eps']:,} episodes (estimated minus oracle-model value; mean ± 95% CI)", pm(og.mean(), _ci(og))]]
    t5 = P.table(rows, "Model-predicted versus true policy advantage. Absolute model values are not comparable with simulator values (termination hazard approximation), but sign and size of the advantage are informative.", widths=[11.4, 5.0], font=8.5)
    P.p(f"Table {t5} shows a healthier picture than a badly misspecified model would: the oracle model ranks its own policy above the rule, and the simulator agrees ({pred:+.2f} predicted, {true:+.2f} true), though the model overstates the gain {pred / true:.1f}-fold. "
        f"Estimated models of {ex['config']['n_eps']:,} episodes are on average slightly pessimistic ({og.mean():+.2f} ± {_ci(og):.2f}, {pstr(og_p)}), as expected from the support constraint. "
        f"These checks concern the training risk only; they say nothing about how well the model extrapolates when risk is rescaled, which Table {t2} answers directly.")

    # ------------------------------------------------------------------ discussion
    P.h1("Discussion")
    P.p(f"**What robustness did and did not do.** Robust DP is usually motivated as protection against a model that is wrong at deployment. In this simulator it did not do that: its worst-case return under a doubling of readmission risk was no better than nominal DP's ({rw['mean']:+.2f} ± {rw['ci']:.2f}) "
        "its readmission rate rose with the radius and its policies were less stable under bootstrap. It did what pessimism is expected to do in offline RL [@jin,kumar]: it improved decisions made from scarce data at unchanged risk, and this faded as data grew. "
        "A planner built to withstand risk shift would have to place the uncertainty where the shift acts, in the readmission probability, and a rectangular L1 ball around transitions is the wrong shape for that.")
    P.p(f"**Learned policies versus the rule.** The learned policies beat the rule under training and mildly shifted risk and lost the advantage near a doubling. The advantage was bought with earlier discharge and a {100 * (std['DP']['info']['readmit'] / std['Rule']['info']['readmit'] - 1):.0f}% higher readmission rate, "
        f"and it disappears if a readmission is priced above about {breakeven:.0f} stay days. The right conclusion for a hospital would be to examine the price list with clinicians before examining the algorithm: a learned policy optimises exactly the weights it is given.")
    P.p("**DP, MC and Q-learning.** At the training risk the three solvers were within noise of each other. Under shift, MC fell furthest at the highest multiplier "
        f"({J['Monte Carlo']['mean'][i2]:.2f} ± {J['Monte Carlo']['ci'][i2]:.2f}, against {J['Q-learning']['mean'][i2]:.2f} ± {J['Q-learning']['ci'][i2]:.2f} for Q-learning); with intervals of this width and no paired test we regard this as suggestive only.")
    limitations(P, [
        "**Uncertainty set does not cover the shift.** The L1 ball perturbs transitions, whereas the shifted quantity lives in the reward. A set built on the readmission probability itself, or a distributionally robust formulation over rewards, might behave differently, and our negative result applies to the tested construction.",
        "**One shift family.** Risk was rescaled uniformly. Real shifts change case mix and the response to follow-up, not only the overall level.",
        f"**Few replicates and multiple comparisons.** {nrep} replicates with 1.96-standard-error intervals ({tfac:.2f} times narrower than t-intervals); only c = 2 was paired-tested; the bootstrap used one data draw.",
        "**Hand-set reward weights.** Conclusions about which policy is better depend on the price of a readmission and of a stay day.",
        "**Dataset mirrored, not used.** The Diabetes 130-US Hospitals table is, to our knowledge, one row per encounter, so it carries no within-stay trajectories; day-by-day transitions would need hospital operational data.",
    ])
    ethics_future(P, "(1) Place the uncertainty set on the readmission probability, for example an interval around each estimated risk, and repeat the shift experiment; "
                  "(2) vary the price of a readmission and report the policy frontier instead of a single weighted return; (3) record stay lengths for every robust variant to test the earlier-discharge explanation; "
                  "(4) repeat the data-size sweep with more replicates and a shift family with case-mix changes; (5) apply the adapter to a hospital table with within-stay records and compare model-predicted with held-out outcomes.")
    P.h1("Conclusion")
    P.p(f"On a simulated discharge problem, learned policies beat a guideline rule at the training risk ({std['DP']['J']:.2f} versus {rule_std:.2f} for DP) by shortening stays and accepting more readmissions, and lost that advantage when readmission risk doubled. "
        f"Robust DP did not protect against the shift (worst case {rw['mean']:+.2f} ± {rw['ci']:.2f} versus nominal, {pstr(rw['p'])}); it helped with scarce data at unchanged risk ({SW['200']['diff_c2_1.0']['mean']:+.2f} at {NS[0]} episodes, {pstr(SW['200']['diff_c2_1.0']['p'])}) and the benefit disappeared by {NS[-1]} episodes. "
        "The practical lessons are to place uncertainty where the shift acts and to treat reward weights as part of the clinical question. These are simulation findings about algorithms, to be re-examined on real discharge data, and they support no clinical conclusion.")

    defense(P, [
        ("Why should I trust a simulator result about discharge?",
         "You should not read it as evidence about patients. The Kaggle site returned HTTP 403, so no real table was analysed, and the simulator is hand-specified and uncalibrated. "
         "Its value is that true policy return is measurable under any risk multiplier, which lets us test what robustness actually buys."),
        ("Did robust DP improve worst-case performance?",
         f"No. The paired worst-case difference for c = 2 is {rw['mean']:+.2f} ± {rw['ci']:.2f} ({pstr(rw['p'])}), and the worst-case return falls from {W['DP (nominal)']['mean']:.2f} (nominal) to {W['Robust DP (c=4)']['mean']:.2f} (c = 4) as the radius grows. "
         "The uncertainty set perturbs transitions while the shift acts on the reward, so the tested construction was never designed to hedge it."),
        ("Where did robustness help, and is that result reliable?",
         f"With scarce data at unchanged risk: {SW['200']['diff_c2_1.0']['mean']:+.2f} ± {SW['200']['diff_c2_1.0']['ci']:.2f} at {NS[0]} episodes ({pstr(SW['200']['diff_c2_1.0']['p'])}) and {SW['400']['diff_c2_1.0']['mean']:+.2f} at {NS[1]} ({pstr(SW['400']['diff_c2_1.0']['p'])}), both below the Bonferroni threshold of {bonf:.4f}. "
         f"At {NS[3]} episodes the difference is {SW['3000']['diff_c2_1.0']['mean']:+.2f} ({pstr(SW['3000']['diff_c2_1.0']['p'])}). The mechanism proposed in the text was not tested."),
        ("Why do the learned policies beat the rule at the training risk, and is that clinically meaningful?",
         f"Because they discharge sooner ({std['DP']['info']['los']:.2f} versus {std['Rule']['info']['los']:.2f} days) and the reward charges 1 per day against 25 per readmission. Readmissions rise from {100 * std['Rule']['info']['readmit']:.1f}% to {100 * std['DP']['info']['readmit']:.1f}%, "
         f"and the advantage vanishes if a readmission is priced above about {breakeven:.0f} days. It is a statement about the price list, not about quality of care."),
        ("Are eight replicates enough?",
         f"They resolve the large effects (the small-data gain, the loss of the advantage near ×2) but not small ones such as the c = 2 worst-case difference ({pstr(rw['p'])}). Our intervals are 1.96 standard errors; t-intervals would be {tfac:.2f} times wider."),
        ("What would be needed to run this on the real Kaggle file?",
         "Within-stay data. The Diabetes 130-US Hospitals table is, to our knowledge, one row per encounter with time in hospital and a readmission label, so it could calibrate risk by group but not day-by-day transitions; "
         "kaggle_adapter.transitions_from_dataframe would need a day-level table, and evaluation would have to be model-based and labelled as an estimate."),
    ])
    repro(P, 8, ["fig_learning", "fig_bars", "fig_agreement", "fig_policies", "fig_shift", "fig_stability", "fig_followup", "fig_datasweep"])
    code_appendix(P, 8, [domains.Discharge])
    rl_core_appendix(P)
    P.refs()
    path = f"{PAP}/Paper08_Discharge_Robust_DataSweep.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
