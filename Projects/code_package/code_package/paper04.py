import numpy as np
from scipy import stats
from rl_core import ci95
from docx_helper import Paper
from paper_common import *

I = 4


def pair(a, b):
    """paired difference a - b over replicates: mean, 95% CI, p (one-sample t-test)"""
    d = np.asarray(a, float) - np.asarray(b, float)
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if d.std(ddof=1) > 0 else 1.0
    return float(d.mean()), float(ci), p


def build():
    import domains
    r = load(I); F, G = r["free"], r["guard"]
    ff, gf = r["free_full"], r["guard_full"]
    dom = domains.Hypertension(); mk = dom.mask()
    n_pairs = dom.nS * dom.nA
    n_blocked = int((~mk[:dom.nS]).sum())
    n_up_blocked = int((~mk[:dom.nS, 2]).sum()); n_down_blocked = int((~mk[:dom.nS, 0]).sum())
    seeds = gf["config"]["seeds"]
    rule_J = F["Rule"]["J"]
    meth = ["DP", "Monte Carlo", "Q-learning"]
    pu, ph, pj = r["pair_unsafe"], r["pair_hypo"], r["pair_J"]
    jf = lambda full, m: full["methods"][m]["J"]
    # paired comparisons against the rule (rule is deterministic: single value)
    vs_rule = {m: pair(jf(gf, m), [rule_J] * seeds) for m in meth}
    vs_rule_free = {m: pair(jf(ff, m), [rule_J] * seeds) for m in meth}
    q_vs_dp = pair(jf(gf, "Q-learning"), jf(gf, "DP"))
    mc_vs_dp = pair(jf(gf, "Monte Carlo"), jf(gf, "DP"))
    gap_or = G["Oracle DP"]["J"] - rule_J
    gap_dp = G["Oracle DP"]["J"] - G["DP"]["J"]
    # oracle-model values
    jo_or, jo_rule = gf["J_oracle_model"]["Oracle DP"], gf["J_oracle_model"]["Rule"]
    jor_beh = float(np.mean(gf["methods"]["Behavior"]["Jor"]))
    thr90 = jor_beh + 0.9 * (jo_or - jor_beh)

    def curve_final(full, nm):
        c = np.array(full["curves"][nm])
        return c[:, 0, 0], c[:, 0, 1].mean(), c[:, 1, 0], c[:, 1, 1].mean(), c[:, -1, 1].mean(), c[:, :, 1].max(1).mean()
    cq_g, cq_f = curve_final(gf, "Q-learning"), curve_final(ff, "Q-learning")
    cm_g, cm_f = curve_final(gf, "Monte Carlo"), curve_final(ff, "Monte Carlo")
    first_ck = float(cq_g[2][0])
    agree = {k: (100 * np.mean(ff["agree"][k]), 100 * np.mean(gf["agree"][k])) for k in ("Monte Carlo", "Q-learning")}
    og_f, og_g = np.mean(ff["optimism_gap"]), np.mean(gf["optimism_gap"])
    og_fc, og_gc = ci95(ff["optimism_gap"]), ci95(gf["optimism_gap"])
    n_rep_pi = sum(gf["pi_equals_vi"]) + sum(ff["pi_equals_vi"])
    # per-replicate win counts of guarded over free
    wins = {m: int(np.sum(np.array(jf(gf, m)) > np.array(jf(ff, m)))) for m in meth}
    # how many replicates have a guarded learner above the rule
    above = {m: int(np.sum(np.array(jf(gf, m)) > rule_J)) for m in meth}
    mc_wide = F["Monte Carlo"]["ci"] / max(G["Monte Carlo"]["ci"], 1e-9)
    # replicates in which unconstrained learner made any unsafe up-titration
    nviol = {m: int(np.sum(np.array(ff["methods"][m]["info"]["unsafe_up"]) > 0)) for m in meth}
    # months per episode at risk of unsafe action: fraction unsafe among 24 months
    H = dom.max_steps

    P = Paper("Guardrails for Hypertension Titration: Action Masks Remove Unsafe Dose Increases from Dynamic-Programming, Monte Carlo and Q-learning Policies at No Measurable Cost",
              "hypertension; medication titration; action masking; safe reinforcement learning; dynamic programming; Monte Carlo control; Q-learning",
              running="Guardrails for hypertension titration", refdict=REFS)
    P.abstract(
        "Choosing whether to raise, hold or lower an antihypertensive dose at each monthly visit is a sequential decision, and a policy learned from logged visits "
        "can rediscover actions that a clinician would never take. We ask whether a simple, clinically legible guardrail - an action mask that forbids up-titration when "
        "systolic pressure is already below 120 mmHg and down-titration when it is at or above 140 mmHg - removes such actions from policies learned by dynamic programming (DP), "
        "first-visit Monte Carlo control and Q-learning, and what it costs. The study uses a hand-written simulator whose variables mirror heart-disease and cardiovascular tables "
        "(the Kaggle site was unreachable, HTTP 403, so no real file was used) and "
        f"{seeds} independent replicates. Without the mask, the learned policies made {F['DP']['info']['unsafe_up']:.2f} (DP), {F['Q-learning']['info']['unsafe_up']:.2f} (Q-learning) and "
        f"{F['Monte Carlo']['info']['unsafe_up']:.2f} (Monte Carlo) unsafe up-titrations per 24-month episode; with the mask these fell to {G['DP']['info']['unsafe_up']:.3f}, "
        f"{G['Q-learning']['info']['unsafe_up']:.3f} and {G['Monte Carlo']['info']['unsafe_up']:.3f}. The reduction is statistically clear only for Monte Carlo "
        f"({pu['Monte Carlo']['mean']:.2f} ± {pu['Monte Carlo']['ci']:.2f}, {pstr(pu['Monte Carlo']['p'])}); for DP ({pstr(pu['DP']['p'])}) and Q-learning ({pstr(pu['Q-learning']['p'])}) "
        f"the ten-replicate test does not reach p < 0.05. The mask did not reduce return: the paired change was {pj['DP']['mean']:+.2f} for DP ({pstr(pj['DP']['p'])}), "
        f"{pj['Q-learning']['mean']:+.2f} for Q-learning and {pj['Monte Carlo']['mean']:+.2f} for Monte Carlo. No learned policy beat the simple guideline rule "
        f"({rule_J:.2f}); guardrailed DP scored {G['DP']['J']:.2f} ± {G['DP']['ci']:.2f}, and Monte Carlo was far worse ({G['Monte Carlo']['J']:.2f}). All results are simulated and carry no clinical meaning.")

    P.h1("Introduction")
    P.p("Hypertension is among the most common chronic conditions, and its drug treatment is a textbook example of sequential decision making: at each visit the "
        "clinician sees a blood pressure reading and the current regimen and decides to intensify, maintain or reduce treatment, with the consequences visible only at the "
        "next visit [@whelton]. Guidelines give target ranges and general rules but leave much to judgement, and the idea of learning a titration policy from the records of "
        "past visits is natural, in the spirit of dynamic treatment regimes [@murphy03] and reinforcement learning in health care [@yu,gottesman].")
    P.p("The central risk of learning from logs is that the optimiser will exploit gaps in the data. A logged policy that is mostly sensible still contains occasional "
        "unusual actions, and an estimated model of those actions is noisy; a maximiser tends to pick the actions whose estimated value is most optimistic, which are often "
        "the least-observed ones [@levine,jin]. In hypertension, the harm of such an error is concrete: raising a dose in a patient who is already at 110 mmHg risks "
        "hypotension. The safe-RL literature offers several remedies, including constrained optimisation [@altman], shielding the policy with a verified safety "
        "layer [@shield] and a broader family of methods surveyed by Garcia and Fernandez [@garcia]. The simplest of these is an **action mask**: a table of forbidden "
        "(state, action) pairs that the learner never selects.")
    P.p("**Contributions.** (i) A transparent hypertension titration simulator with a heterogeneous dose-response, a drug-effect lag and an asymmetric reward in which "
        "hypotension is penalised eight times more than the largest benefit of a month in range. (ii) A two-rule guardrail encoded as a mask on a 72-state, 3-action MDP, "
        "enforced identically inside DP (maximisation over allowed actions), Monte Carlo control and Q-learning (restricted exploration and greedy choice). "
        f"(iii) A controlled comparison of mask on versus mask off over {seeds} replicates for all three solvers, scored in the true simulator, with paired tests. "
        "(iv) Safety-specific outcome measures - unsafe up-titrations, hypotensive months and dose changes - in addition to return. "
        "(v) Candid negative results: the mask's return effect is not distinguishable from zero, its safety benefit is significant for only one of three solvers at this sample size, "
        "and no learned policy outperforms a simple guideline rule.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached from the build environment (all requests returned HTTP 403), so no real dataset was analysed. The simulator "
        "mirrors the variables of heart-disease and cardiovascular-disease tables (blood pressure, risk strata, treatment intensity) with made-up dynamics. Nothing in this paper "
        "is a clinical claim about any drug, dose or person.")

    P.h1("Related work")
    P.p("**Hypertension management.** The ACC/AHA guideline defines blood-pressure categories and recommends a stepwise approach to titration, intensifying treatment "
        "when readings are above target and reassessing at regular intervals [@whelton]. Our simulator's target range and the guardrail thresholds are stylised readings of that "
        "kind of advice, not guideline values.")
    P.p("**Safe reinforcement learning.** Garcia and Fernandez classify safe-RL approaches into those that modify the optimisation criterion and those that modify "
        "exploration, including constraints and external knowledge [@garcia]. Constrained MDPs [@altman] treat safety as an expected-cost constraint; shielding "
        "[@shield] instead filters actions by a specification so that violations cannot occur by construction. A hard mask is the tabular special case of a shield whose specification "
        "depends only on the observed state. It gives a guarantee that an expected-cost penalty cannot give: a forbidden action is never taken when its precondition is observed.")
    P.p("**Offline RL in medicine.** Reviews warn that policies learned from observational records may extrapolate to poorly supported actions and that evaluation is difficult "
        "without a simulator [@gottesman,yu,levine]. Conservative methods penalise or avoid unsupported actions [@kumar,fujimoto]. A mask is orthogonal: it removes actions that "
        "are unsafe for reasons external to the data, even if the data happen to contain a few optimistic samples of them.")
    P.p("**Treatment optimisation by MDPs.** Markov decision processes have long been used to schedule or time treatment [@schaefer,puterman], and learning clinical policies "
        "from logged data has been demonstrated for sepsis [@komorowski]. Our contribution is not a new clinical model but a controlled test of one design decision - hard guardrails "
        "- on a problem where ground truth is available.")
    background(P)

    P.h1("Problem formulation: a monthly titration simulator")
    P.p("Each month the clinician chooses to lower, hold or raise the dose, which takes one of four levels (0-3). A patient has a discrete risk stratum (low, medium or high) "
        f"that sets the untreated systolic baseline (140, 152 or 166 mmHg), and a hidden drug sensitivity drawn from a normal distribution with mean 1.0 and standard deviation 0.3, clipped to [0.5, 1.8]. "
        "The systolic pressure moves halfway each month toward a dose-dependent target, baseline − 11 · dose · sensitivity, plus noise with standard deviation 4 mmHg; this is a one-month lag "
        "in the drug effect. A mean-sensitivity patient therefore falls 33 mmHg at the highest dose, but a patient with sensitivity 1.8 falls almost 60 mmHg, which is how a "
        "sound-looking dose increase can overshoot.")
    P.table([["Element", "Definition"],
             ["State (72)", "systolic band (6: <110, 110-119, 120-129, 130-139, 140-159, ≥160 mmHg) × risk stratum (3) × dose level (4)"],
             ["Actions (3)", "down (lower dose), hold, up (raise dose); dose clipped to 0-3"],
             ["Reward per month", "SBP < 105: −8.0;  105-109: −0.5;  110-130: +1.0;  131-139: +0.4;  140-159: −0.3;  ≥160: −1.5;  then − 0.15 · dose − 0.1 · 1[dose changed]"],
             ["Discount, horizon", f"γ = {dom.gamma}, {H} monthly visits"],
             ["Guardrail rule 1", f"never choose 'up' when SBP < 120 mmHg (bands 0-1): {n_up_blocked} state-action pairs blocked"],
             ["Guardrail rule 2", f"never choose 'down' when SBP ≥ 140 mmHg (bands 4-5): {n_down_blocked} state-action pairs blocked"],
             ["Safety counters", "unsafe up-titration = 'up' chosen when SBP < 120;  hypotensive month = SBP < 105 after the step"],
             ["Behaviour policy", f"guideline rule (up if SBP ≥ 140, down if SBP < 110, otherwise hold) with ε = {dom.behavior_eps} random actions"],
             ["Dataset mirrored", "Heart Disease UCI; Cardiovascular Disease (blood-pressure and risk columns)"]],
            "Hypertension titration MDP and the two guardrail rules.", widths=[3.6, 12.8], font=8.5)
    P.p(f"The mask blocks {n_blocked} of {n_pairs} state-action pairs ({100 * n_blocked / n_pairs:.0f}%). Rule 1 is strictly narrower than the reward's own notion of danger: the reward penalises "
        "a month below 105 mmHg, whereas the mask acts a full 15 mmHg earlier, at 120. The mask is thus a conservative clinical rule of thumb rather than a restatement of the reward. "
        "Because the observed band edges are at 110 and 120, the rule cannot be made finer than a 10 mmHg step without changing the state definition.")
    P.p("The guideline rule used as a baseline is deterministic and never up-titrates below 140 or down-titrates above 110, so by construction it violates neither guardrail. "
        "Consequently the rule is itself a strong baseline on safety, and the question for any learned policy is whether it can match the rule's safety while improving return.")

    P.h1("Method: guardrailed planning and learning")
    P.p("Let A(s) ⊆ A be the set of allowed actions in state s, given by the mask. We replace the unconstrained operators with their masked counterparts. The masked Bellman "
        "optimality equation is")
    P.eq("V*_m(s) = max_{a ∈ A(s)} [ R(s,a) + γ Σ_{s'} P(s'|s,a) V*_m(s') ]", "3")
    P.p("and the masked Q-learning update and exploration use the same restriction,")
    P.eq("Q(s,a) ← Q(s,a) + α [ r + γ max_{a' ∈ A(s')} Q(s',a') − Q(s,a) ],      a ~ ε-greedy over A(s)", "4")
    P.p("Monte Carlo control uses the same restriction in both its behaviour and its greedy improvement step. Two properties follow directly. First, any policy that is greedy with respect to a masked "
        "value function satisfies the guardrail in every state where the mask is defined, regardless of the quality of the model. Second, the optimum of the masked problem cannot be "
        "better than that of the unmasked problem on the model, V*_m ≤ V*, because it maximises over a subset. The interesting empirical question is therefore how much of the unmasked optimum "
        "is lost - and, in finite data, whether the restriction in fact helps by removing optimistic errors. The two effects work in opposite directions and we measure their net result.")
    P.p("A practical subtlety deserves mention. Planning uses a support threshold, so some state-action pairs have too little data to be trusted and are completed pessimistically. "
        "A state that was never observed in the logs has no model, and the deployed policy defers there to the modal logged action. The mask is enforced inside the learners but the fallback "
        "is not itself masked, so the guarantee holds in every state with a model and may fail in a rarely visited unmodelled state. We report the residual violations that this leaves.")
    P.code("""Algorithm 1  Guardrailed titration from logs
 input : logged visits (SBP band, risk, dose, action, reward, next state); mask M[s,a] in {allowed, forbidden}
 1. build the empirical MDP (counts N(s,a,s'), mean rewards); pairs with < 10 visits are unsupported (pessimistic)
 2. DP     : value iteration with max over {a : M[s,a] = allowed}                           -> pi_DP
 3. MC     : first-visit control on samples from the empirical MDP; eps-greedy over allowed actions
 4. Q      : tabular Q-learning with the masked max in the target and eps-greedy over allowed actions
 5. fill   : in states with no model, use the modal logged action
 6. deploy : each visit, read SBP band, risk, dose; output pi(s); the clinician may always override
 7. report : unsafe up-titrations, hypotensive months, dose changes, in-range months, return""")
    P.p("A worked example shows how the mask changes a decision. Take a mean-sensitivity, medium-risk patient (baseline 152 mmHg) on dose 2, whose last reading was 118 mmHg. The unmasked "
        "learner may rank 'up' above 'hold' because its logs contain a few visits in which a similar patient was raised and happened to stay in range (a noisy, favourable sample). Raising the dose to 3 "
        "moves the target to 152 − 33 = 119 mmHg, a small overshoot risk for a patient with sensitivity above 1, but for a sensitivity of 1.8 the target is 152 − 59 = 93 mmHg, in the "
        "region where the reward is −8. Rule 1 removes 'up' from this state, so the learner chooses between 'hold' and 'down' using only well-supported evidence.")
    protocol(P, [["Item", "Setting"],
                 ["Simulator", "monthly hypertension titration; SBP, risk stratum, dose; hidden drug sensitivity"],
                 ["States / actions", f"{dom.nS} / {dom.nA}; mask blocks {n_blocked} of {n_pairs} pairs"],
                 ["Logged data", f"{gf['config']['n_eps']} episodes × {H} months = {gf['config']['n_eps'] * H:,} transitions per replicate"],
                 ["Oracle model", f"{gf['config']['n_oracle']:,} episodes (for exact learning-curve scoring)"],
                 ["Conditions", "mask off (unconstrained) and mask on (guardrailed), identical logged data per replicate"],
                 ["Replicates", f"{seeds} independent data draws and learner seeds per condition"],
                 ["Learners", f"DP (value iteration; policy iteration cross-check), first-visit Monte Carlo, Q-learning; {gf['config']['steps']:,} steps each"],
                 ["Scoring", f"{gf['config']['n_eval']} episodes per policy in the true simulator (common random numbers)"]],
             gf["config"]["steps"], seeds, gf["config"]["n_eval"],
             extra=f"Both conditions share the same logged datasets and seeds, so differences between them are paired. Value and policy iteration returned policies of identical model value in {n_rep_pi} of {2 * seeds} runs, as required.")

    P.h1("Results")
    P.h2("Headline comparison")
    rows = [["Method", "Return, mask off", "Return, mask on", "Paired change (on − off)"]]
    for k in ("Rule", "Behavior", "DP", "Monte Carlo", "Q-learning", "Oracle DP"):
        chg = f"{pj[k]['mean']:+.2f} ± {pj[k]['ci']:.2f} ({pstr(pj[k]['p'])})" if k in pj else "-"
        rows.append([NICE.get(k, k), pm(F[k]["J"], F[k]["ci"]), pm(G[k]["J"], G[k]["ci"]), chg])
    t1 = P.table(rows, f"True-simulator discounted return (24 months, mean ± 95% CI over {seeds} replicates) with the guardrail off and on. The rule and the oracle are single fixed policies, so their intervals are zero.",
                 widths=[5.0, 3.4, 3.4, 5.0], font=8.5)
    P.p(f"Table {t1} gives the headline numbers. The guideline rule scores {rule_J:.2f}, and the clinician-behaviour policy (the modal logged action) {F['Behavior']['J']:.2f}. "
        f"Learned policies do not exceed them: unmasked DP scores {F['DP']['J']:.2f} ± {F['DP']['ci']:.2f}, Q-learning {F['Q-learning']['J']:.2f} ± {F['Q-learning']['ci']:.2f} and Monte Carlo "
        f"{F['Monte Carlo']['J']:.2f} ± {F['Monte Carlo']['ci']:.2f}. With the mask, the means move to {G['DP']['J']:.2f}, {G['Q-learning']['J']:.2f} and {G['Monte Carlo']['J']:.2f}. "
        f"The mask raised the mean return for every solver, but none of the paired changes is statistically distinguishable from zero at the 5% level ({pstr(pj['DP']['p'])}, {pstr(pj['Q-learning']['p'])} and "
        f"{pstr(pj['Monte Carlo']['p'])} for DP, Q-learning and Monte Carlo); the guarded return exceeded the unguarded return in {wins['DP']}, {wins['Q-learning']} and {wins['Monte Carlo']} of {seeds} replicates. "
        f"The defensible reading is that the mask does not cost return here, and may help slightly, not that it improves return.")
    P.p(f"The oracle DP, trained on ten times more data, scores {G['Oracle DP']['J']:.2f}, {gap_or:.2f} above the rule; hence there is real headroom in this simulator, and the "
        f"learned policies at {gf['config']['n_eps']} episodes capture none of it. Guardrailed DP is {G['Oracle DP']['J'] - G['DP']['J']:.2f} below the oracle. This gap is a data-quantity effect (estimation error "
        f"in {dom.nS * dom.nA} state-action pairs from {gf['config']['n_eps'] * H:,} transitions), and the mask cannot close it because the oracle's own policy already satisfies the guardrail "
        f"(its unsafe up-titration rate is {G['Oracle DP']['info']['unsafe_up']:.3f}; the oracle's model value is identical with and without the mask, {r['J_oracle']['free']:.2f} versus {r['J_oracle']['guard']:.2f}).")
    f1 = P.fig(fig(I, "fig_bars_guard"), "True return of each method with the guardrail on (means with 95% CIs over replicates). Monte Carlo is far below the others; the mask-off figure has the same ordering.", 8.5)
    P.p(f"Figure {f1} (and Table {t1} for the mask-off condition) shows the same ordering in both conditions: rule and behaviour policy at the top, DP just below them, Q-learning lower and Monte Carlo lowest. Against the rule, the paired "
        f"differences with the mask on are {vs_rule['DP'][0]:+.2f} ± {vs_rule['DP'][1]:.2f} for DP ({pstr(vs_rule['DP'][2])}), {vs_rule['Q-learning'][0]:+.2f} ± {vs_rule['Q-learning'][1]:.2f} for Q-learning "
        f"({pstr(vs_rule['Q-learning'][2])}) and {vs_rule['Monte Carlo'][0]:+.2f} ± {vs_rule['Monte Carlo'][1]:.2f} for Monte Carlo ({pstr(vs_rule['Monte Carlo'][2])}). In other words, guardrailed DP is statistically "
        f"indistinguishable from the rule at this sample size (and above it in {above['DP']} of {seeds} replicates), Q-learning is reliably below it and Monte Carlo is far below it. "
        f"Among learners, Q-learning trails DP by {-q_vs_dp[0]:.2f} ± {q_vs_dp[1]:.2f} ({pstr(q_vs_dp[2])}) and Monte Carlo trails DP by {-mc_vs_dp[0]:.2f} ± {mc_vs_dp[1]:.2f} ({pstr(mc_vs_dp[2])}).")

    P.h2("Safety outcomes")
    rows = [["Method", "Unsafe up-titrations, off", "Unsafe up-titrations, on", "Paired reduction (p)", "Hypotensive months, off", "Hypotensive months, on", "Paired reduction (p)"]]
    for k in ("Rule", "DP", "Monte Carlo", "Q-learning"):
        if k in pu:
            ru_, rh_ = f"{pu[k]['mean']:.2f} ± {pu[k]['ci']:.2f} ({pstr(pu[k]['p'])})", f"{ph[k]['mean']:.3f} ± {ph[k]['ci']:.3f} ({pstr(ph[k]['p'])})"
        else:
            ru_, rh_ = "-", "-"
        rows.append([NICE.get(k, k), f"{F[k]['info']['unsafe_up']:.3f}", f"{G[k]['info']['unsafe_up']:.3f}", ru_,
                     f"{F[k]['info']['hypo']:.3f}", f"{G[k]['info']['hypo']:.3f}", rh_])
    t2 = P.table(rows, f"Safety counters per 24-month episode (means over {seeds} replicates). 'Paired reduction' is mask off minus mask on with a 95% CI and a one-sample t-test p-value.",
                 widths=[3.0, 2.2, 2.2, 3.2, 2.2, 2.2, 3.2], font=8)
    P.p(f"Table {t2} is the main evidence for the guardrail. Without it, all three learners take actions that rule 1 forbids: {F['DP']['info']['unsafe_up']:.2f} per episode for DP, "
        f"{F['Q-learning']['info']['unsafe_up']:.2f} for Q-learning and {F['Monte Carlo']['info']['unsafe_up']:.2f} for Monte Carlo, against {F['Rule']['info']['unsafe_up']:.2f} for the rule. "
        f"Monte Carlo makes about one such error per two-year course, which is {100 * F['Monte Carlo']['info']['unsafe_up'] / H:.0f}% of its visits. In the replicates, an unconstrained DP policy made at least one unsafe up-titration "
        f"in {nviol['DP']} of {seeds} replicates, Q-learning in {nviol['Q-learning']} and Monte Carlo in {nviol['Monte Carlo']}. With the mask the counts are {G['DP']['info']['unsafe_up']:.3f}, "
        f"{G['Q-learning']['info']['unsafe_up']:.3f} and {G['Monte Carlo']['info']['unsafe_up']:.3f}. The residual for Monte Carlo is not exactly zero. Since the mask is enforced in every state with a model, "
        "this residual is most plausibly due to the fallback to the modal logged action in states without a model; we did not trace individual violations, so this is a hypothesis.")
    P.p(f"Statistical strength differs by solver. For Monte Carlo the paired reduction is {pu['Monte Carlo']['mean']:.2f} ± {pu['Monte Carlo']['ci']:.2f} ({pstr(pu['Monte Carlo']['p'])}). "
        f"For DP it is {pu['DP']['mean']:.2f} ± {pu['DP']['ci']:.2f} ({pstr(pu['DP']['p'])}) and for Q-learning {pu['Q-learning']['mean']:.2f} ± {pu['Q-learning']['ci']:.2f} ({pstr(pu['Q-learning']['p'])}): the point estimates are "
        "of the same order as Monte Carlo's per replicate but the replicate-to-replicate variation is large, because unsafe actions occur in only some data draws. The direction is consistent, the "
        "interval for DP and Q-learning includes zero, and we therefore describe the effect for these two solvers as suggestive rather than established. We did not adjust for multiple comparisons; "
        "adjusting would weaken even the Monte Carlo result slightly, which should be borne in mind.")
    P.p(f"Hypotensive months, the outcome that the guardrail is intended to prevent, are rare. Unmasked DP has {F['DP']['info']['hypo']:.3f} per episode and the guarded version {G['DP']['info']['hypo']:.3f}; "
        f"for Q-learning the figures are {F['Q-learning']['info']['hypo']:.3f} and {G['Q-learning']['info']['hypo']:.3f}; for Monte Carlo {F['Monte Carlo']['info']['hypo']:.3f} and {G['Monte Carlo']['info']['hypo']:.3f}. "
        f"None of the paired reductions is significant ({pstr(ph['DP']['p'])}, {pstr(ph['Q-learning']['p'])} and {pstr(ph['Monte Carlo']['p'])}). So the mask removes the unsafe "
        "action reliably, but at this sample size we cannot show that it removes the downstream harm: the rare outcome needs more evaluation episodes or more replicates to detect.")
    f2 = P.fig(fig(I, "fig_safety"), "Left: hypotensive months per episode. Middle: unsafe up-titrations per episode. Right: true return. Grey: mask off; blue: mask on (means with 95% CIs).", 14.5)
    P.p(f"Figure {f2} summarises the three outcomes. The middle panel carries the effect: grey bars for the learners are well above zero and the blue bars are at zero. The left panel shows "
        f"the much smaller hypotension counts, in which overlapping intervals are visible for every solver, and the right panel shows that return does not move beyond noise. Monte Carlo has "
        f"the widest intervals in all panels (its return interval with the mask off is {mc_wide:.1f} times that with the mask on), consistent with the high variance of full-episode returns.")

    P.h2("Treatment-pattern outcomes")
    rows = [["Method", "In-range months (of 24), off", "In-range months, on", "Dose changes, off", "Dose changes, on", "Cumulative dose, off", "Cumulative dose, on"]]
    for k in ("Rule", "Behavior", "DP", "Monte Carlo", "Q-learning", "Oracle DP"):
        rows.append([NICE.get(k, k)] + [f"{x[k]['info'][c]:.1f}" for c in ("in_target", "changes", "dose") for x in (F, G)])
    t3 = P.table(rows, "Secondary outcomes: months with 110 ≤ SBP ≤ 130 mmHg, number of dose changes, and cumulative dose-months (sum of dose level over 24 months; maximum 72).",
                 widths=[4.2, 2.4, 2.0, 2.0, 2.0, 2.2, 2.0], font=8)
    P.p(f"Table {t3} shows that the learned policies are more active than the rule. The rule makes {G['Rule']['info']['changes']:.1f} dose changes per course, DP {G['DP']['info']['changes']:.1f}, Q-learning "
        f"{G['Q-learning']['info']['changes']:.1f} and Monte Carlo {G['Monte Carlo']['info']['changes']:.1f}; the oracle makes {G['Oracle DP']['info']['changes']:.1f} and obtains the most in-range months ({G['Oracle DP']['info']['in_target']:.1f} against "
        f"{G['Rule']['info']['in_target']:.1f} for the rule). One possible reading, which we have not tested, is that the optimum makes fine adjustments near the 130 mmHg edge of the target range, which the reward's switching penalty (0.1 per change) "
        f"only partly discourages. The learned policies do not achieve that benefit: in-range months for guarded DP are {G['DP']['info']['in_target']:.1f}, below the rule's {G['Rule']['info']['in_target']:.1f}, "
        f"while making {G['DP']['info']['changes'] - G['Rule']['info']['changes']:.1f} more dose changes. The mask changes these patterns only slightly (for DP, {F['DP']['info']['changes']:.1f} changes without and {G['DP']['info']['changes']:.1f} with the mask). "
        f"Clinically, a policy that changes the dose more often than the rule while keeping fewer months in range would be unattractive even if its return were equal, an argument for reporting these secondary measures alongside return.")

    P.h2("Learning dynamics")
    f3 = P.fig(fig(I, "fig_learning_compare"), "Policy value on the oracle MDP during learning (mean over replicates), with the mask off (pale lines) and on (bold lines). The dashed line is the guardrailed DP optimum on the oracle model.", 10.5)
    P.p(f"Figure {f3} compares learning curves. Q-learning reaches its plateau by the first checkpoint ({first_ck:,.0f} steps), at {cq_g[4]:.2f} with the mask and {cq_f[4]:.2f} without; "
        f"it never approaches the oracle optimum of {jo_or:.2f}, because it is solving the estimated model, whose optimum differs from the oracle's. Monte Carlo is slower and plateaus much lower, at {cm_g[4]:.2f} with the mask "
        f"and {cm_f[4]:.2f} without. The plateau is higher with the mask for both learners (by {cq_g[4] - cq_f[4]:.2f} for Q-learning and {cm_g[4] - cm_f[4]:.2f} for Monte Carlo) and in the starting value: "
        f"the value at step 0 is {cq_g[1]:.1f} for guarded Q-learning against {cq_f[1]:.1f} for unguarded, presumably because an exploring policy that cannot raise the dose at low pressure is less harmful (we did not isolate this). "
        f"A conventional convergence statistic would be steps to 90% of the gap between the behaviour policy and the oracle ({thr90:.2f} on this scale); " + ("no learner reached it in any condition, so we do not report one." if max(cq_g[5], cq_f[5], cm_g[5], cm_f[5]) < thr90 else "see the curves.") + f"")
    f4 = P.fig(fig(I, "fig_learning_guard"), "Learning curves with the mask on, with ±1 standard deviation bands over replicates; dotted line is the rule's value on the oracle model.", 8.5)
    P.p(f"Figure {f4} shows the guarded curves separately, with the rule's model value ({jo_rule:.2f}) as a dotted line. "
        f"Both Q-learning curves sit above the rule on the oracle model, "
        f"which is consistent with the model prediction that a learned policy can beat the rule. The simulator tells a different story: Table {t1} shows no learned policy beating the rule in the real dynamics. "
        f"A related quantity is the optimism gap, the DP policy's value on its own estimated MDP minus its value on the oracle MDP. It averaged {og_f:+.2f} ± {og_fc:.2f} without the mask and "
        f"{og_g:+.2f} ± {og_gc:.2f} with it; " + ("both intervals include zero, so we do not claim a systematic optimism here." if abs(og_f) < og_fc and abs(og_g) < og_gc else "at least one interval excludes zero.") + f" Absolute model values are also not comparable to simulator values, because the oracle MDP treats the 24-month horizon as a constant termination hazard (see limitations).")

    P.h2("Agreement between learners and policy structure")
    rows = [["Learner", "Agreement with DP, mask off", "Agreement with DP, mask on"]]
    for k in ("Monte Carlo", "Q-learning"):
        rows.append([k, f"{agree[k][0]:.0f}%", f"{agree[k][1]:.0f}%"])
    t4 = P.table(rows, "Fraction of well-visited states (at least 5 visits) in which the model-free learner chooses the same action as DP on the same estimated MDP.", widths=[4.4, 5.8, 5.8], font=8.5)
    f6 = P.fig(fig(I, "fig_policies_guard"), "Policies by state with the mask on (states observed in the data): behaviour, DP, Monte Carlo, Q-learning and oracle. Colours: down, hold, up.", 12.5)
    f7 = P.fig(fig(I, "fig_policies_free"), "Policies by state with the mask off.", 12.5)
    P.p(f"Table {t4} shows that Q-learning reproduces the DP action in about {agree['Q-learning'][1]:.0f}% of well-visited states and Monte Carlo in about {agree['Monte Carlo'][1]:.0f}%, with "
        f"similar values without the mask ({agree['Q-learning'][0]:.0f}% and {agree['Monte Carlo'][0]:.0f}%). Monte Carlo's lower agreement matches its lower return. Figures {f6} and {f7} show the policies. In both conditions the policies "
        "agree on the extremes (lowering the dose at the lowest pressures, raising it at the highest) and differ in the middle of the state space, where the behaviour policy simply holds. "
        "The oracle's raising of the dose in several mid-range states is the main structural difference from the rule and is consistent with its extra in-range months in Table "
        f"{t3}. In the unmasked panel (Figure {f7}) some cells in the left half of the state axis show 'up' for the learned policies where the masked panel shows 'hold', "
        "which may be the visual counterpart of the unsafe up-titrations in Table " + str(t2) + "; we did not map the plotted state indices back to bands, so this is an impression, not a count.")

    P.h1("Discussion")
    P.p("**What the guardrail does and does not do.** The data support a narrow claim: a hard action mask removes the forbidden action from learned policies, at no detectable cost in "
        f"simulator return, with the clearest effect for Monte Carlo ({pstr(pu['Monte Carlo']['p'])}) and suggestive effects for DP and Q-learning. The mask does not turn a poor learner into a good one: "
        f"Monte Carlo guarded ({G['Monte Carlo']['J']:.2f}) remains well below the rule. The guardrail changes what a policy is allowed to do, not how well it estimates value.")
    P.p("**Why safety may come free.** The guardrail is cheap here because the optimal policy already respects it: the masked and unmasked oracle optima have equal model value. Where the mask removes only actions "
        "the optimum would not use, it removes estimation-error exploits without removing value. A mask that conflicted with the true optimum would impose a cost proportional to the reward foregone, "
        "which is bounded above by the gap V* − V*_m of Eq. (3). The free-lunch result is therefore a property of this simulator, whose reward punishes low pressure much more heavily than it rewards intensive treatment, "
        "and may not transfer to a setting in which a rule of thumb conflicts with individual optima.")
    P.p("**Why the rule is hard to beat.** The rule is almost the structure of the reward: raise when pressure is high, lower when it is low, otherwise hold. The simulator's reward has a "
        "wide 110-130 mmHg band and a switching cost, which together favour a stable policy; the learners have 7,200 transitions to estimate a 72-state, 3-action model, with sparse coverage of "
        "non-rule actions because at least 75% of the behaviour policy's actions are the rule's. With such data, estimation noise is of the same size as the headroom. The oracle shows that headroom exists "
        f"({gap_or:+.2f} over the rule), which is a reminder that the negative result is about the sample size, not about the problem.")
    P.p("**Practical guidance.** (1) Encode any rule that a clinician would state without hesitation as a mask; it is the cheapest safety layer, auditable by inspection of a table. (2) Enforce it in "
        "the planner, the learner and the deployed fallback; leaving the fallback unmasked can leak violations. (3) Report safety counters with confidence intervals and the number of "
        "replicates in which any violation occurred, because averages hide the fact that violations arise in only some data draws. (4) Do not use a guardrail as a substitute for a "
        "validation study: it bounds the actions, not the outcomes.")
    limitations(P, [
        "**Hand-written masks.** The thresholds 120 and 140 mmHg are stylised choices tied to the state's band edges, not guideline values, and the study does not test other thresholds or a data-driven mask.",
        f"**Few replicates for rare events.** Unsafe actions occurred in only {nviol['DP']} of {seeds} (DP), {nviol['Q-learning']} of {seeds} (Q-learning) and {nviol['Monte Carlo']} of {seeds} (Monte Carlo) unmasked replicates, and hypotension is rarer still; the tests are underpowered for DP and Q-learning and we ran no multiplicity correction.",
        "**Unmasked fallback.** The mask is enforced inside the learners but not in the modal-action fallback for unmodelled states, which is the most likely source of the small residual violation rate for Monte Carlo.",
        "**Single-drug, single-dose-axis simulator.** Real titration involves several drug classes, adherence, side effects and measurement error (white-coat effect, home readings); none is modelled. The hidden sensitivity is the only patient heterogeneity.",
        "**Reward design.** The relative size of the penalty for hypotension (−8) and the benefit of range (+1) was fixed by us and determines how costly the guardrail is; different weights could change the cost of the mask.",
    ])
    ethics_future(P, "(1) Learn the mask from data or from guideline text and quantify how often it conflicts with the optimum, so that its cost is measured rather than assumed; (2) add individual "
                  "adherence and white-coat noise, and test whether the guardrail remains cost-free under observation noise; (3) compare the hard mask with a soft penalty and a shielded policy in the same "
                  "simulator; (4) run the adapter on a real cardiovascular table and evaluate the guarded policy off-policy with explicit uncertainty, as a pre-requisite for any clinical discussion.")
    P.h1("Conclusion")
    P.p(f"On a hypertension titration simulator, DP, Monte Carlo control and Q-learning learned from {gf['config']['n_eps']} logged episodes make unsafe up-titrations when unconstrained "
        f"({F['DP']['info']['unsafe_up']:.2f}, {F['Monte Carlo']['info']['unsafe_up']:.2f} and {F['Q-learning']['info']['unsafe_up']:.2f} per course for DP, Monte Carlo and Q-learning); a two-rule action mask removes them "
        f"({G['DP']['info']['unsafe_up']:.3f}, {G['Monte Carlo']['info']['unsafe_up']:.3f} and {G['Q-learning']['info']['unsafe_up']:.3f}), and the paired change in return is not distinguishable from zero for any solver. "
        f"The statistical evidence for the safety gain is clear for Monte Carlo ({pstr(pu['Monte Carlo']['p'])}) and suggestive for DP and Q-learning. No learned policy beat the simple rule ({rule_J:.2f}), and Monte Carlo remained far below it. "
        "The practical message is modest and useful: where a hard clinical rule can be stated, enforce it as a mask in the planner, the learner and the fallback. The scientific caveat is equally important: "
        "this is a simulator study and says nothing about the safety or benefit of any treatment.")
    defense(P, [
        ("Does the guardrail actually make the policies safer, or is it just removing a counter that you defined yourself?",
         f"It removes exactly the counted action by construction, so the counter alone proves little. The informative results are the secondary ones: return did not fall ({pj['DP']['mean']:+.2f} for DP, {pstr(pj['DP']['p'])}), "
         f"and hypotensive months fell in point estimate but not significantly ({F['DP']['info']['hypo']:.3f} to {G['DP']['info']['hypo']:.3f} for DP, {pstr(ph['DP']['p'])}). We therefore claim that the rule is enforced at no detectable return cost, not that downstream harm is shown to fall."),
        ("Why are only the Monte Carlo results statistically significant?",
         f"Monte Carlo has the largest rate of unsafe actions ({F['Monte Carlo']['info']['unsafe_up']:.2f} per course) so the effect is large relative to noise ({pstr(pu['Monte Carlo']['p'])}). For DP and Q-learning the rate is about {F['DP']['info']['unsafe_up']:.2f} and the violations occur in only {nviol['DP']} and {nviol['Q-learning']} of {seeds} replicates, so a {seeds}-replicate test has little power ({pstr(pu['DP']['p'])} and {pstr(pu['Q-learning']['p'])}). More replicates would settle it; we did not run them."),
        ("If the learned policies are worse than the rule, why use them at all?",
         f"At {gf['config']['n_eps']} episodes they are not better, and we do not claim they are: guarded DP is {vs_rule['DP'][0]:+.2f} from the rule ({pstr(vs_rule['DP'][2])}). The oracle, with ten times more data, is {gap_or:.2f} above the rule, so the argument for learning is "
         "conditional on data quantity, and the argument for the guardrail is that it makes learning from small logs safer to explore."),
        ("Is the guardrail free in general?",
         f"No. Here it is free because the oracle optimum already satisfies it (model value {r['J_oracle']['free']:.2f} with and without the mask, Eq. (3) bound is tight). A rule that conflicts with the true optimum would cost the reward it forbids. The cost should be measured, as proposed in the future work."),
        ("Why does Monte Carlo do so badly?",
         f"Its return is {G['Monte Carlo']['J']:.2f} ± {G['Monte Carlo']['ci']:.2f} against {G['DP']['J']:.2f} for DP and its agreement with DP is only {agree['Monte Carlo'][1]:.0f}%. Full-episode returns of 24 noisy months give high-variance estimates, and its plateau on the oracle model is {cm_g[4]:.2f} against {cq_g[4]:.2f} for Q-learning. The mask lessens, but does not cure, this."),
        ("Are the unsafe actions a real feature of the data or an artefact of your behaviour policy?",
         f"We cannot separate the two here. The behaviour policy's ε = {dom.behavior_eps} random actions include 'up' at low pressure, which gives the learner sparse samples of it; plausibly some of these samples are optimistic by chance (we did not test this). Whether real logs contain analogous samples is an empirical question that the adapter can answer."),
    ])
    repro(P, 4, ["fig_bars_guard", "fig_bars_free", "fig_safety", "fig_learning_compare", "fig_learning_guard", "fig_learning_free", "fig_policies_guard", "fig_policies_free", "fig_agreement_guard", "fig_agreement_free"])
    code_appendix(P, 4, [domains.Hypertension])
    rl_core_appendix(P)
    P.refs()
    path = f"{PAP}/Paper04_Guardrailed_Hypertension.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
