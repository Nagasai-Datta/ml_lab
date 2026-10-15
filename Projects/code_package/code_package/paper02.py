import numpy as np
from scipy import stats
from docx_helper import Paper
from paper_common import *
import domains

I = 2


def build():
    r = load(I); S = r["standard"]; E = r["extra"]
    GR, KS = r["GR"], r["KS"]
    G = np.array(r["gain"]); GC = np.array(r["gain_ci"]); JJ = np.array(r["J"]); JC = np.array(r["J_ci"])
    GAP = np.array(r["gap"]); MORT = np.array(r["mort"]); AGR = np.array(r["agree"])
    RULE = r["rule"]; COV = r["cov"]; MEDN = r["med_n"]
    PV = r["pess_vs_naive"]; BG = r["best_pess_vs_naive_gap"]
    cfg = E["config"]; meth = E["methods"]
    gi = {g: i for i, g in enumerate(GR)}; ki = {k: i for i, k in enumerate(KS)}
    nrep = PV[str(GR[0])]["n"]
    d3 = domains.Sepsis(3)
    cells = {g: g ** 3 * 9 for g in GR}
    kap_hoeff = float(np.sqrt(np.log(2 / 0.05) / 2))
    rule_J = S["Rule"]["J"]

    # --- standard benchmark paired tests vs the rule
    def pvr(k):
        d = np.array(meth[k]["J"]) - meth["Rule"]["J"][0]
        return float(d.mean()), 1.96 * float(d.std(ddof=1) / np.sqrt(len(d))), float(stats.ttest_1samp(d, 0).pvalue)
    pv = {k: pvr(k) for k in ("DP", "Monte Carlo", "Q-learning")}
    dpJ = np.array(meth["DP"]["J"]); n_dp_better = int((dpJ > rule_J + 1e-6).sum()); worst_dp = float(dpJ.min()); worst_idx = int(dpJ.argmin()) + 1
    seeds = len(dpJ)
    gap_or = S["Oracle DP"]["J"] - rule_J
    pol = E["policies"]; nS = E["n_states"]
    n_or_diff = int(np.sum(np.array(pol["Oracle DP"]) != np.array(pol["Rule"])))
    n_dp_diff = int(np.sum(np.array(pol["DP"]) != np.array(pol["Rule"])))
    n_mc_diff = int(np.sum(np.array(pol["Monte Carlo"]) != np.array(pol["Rule"])))
    n_q_diff = int(np.sum(np.array(pol["Q-learning"]) != np.array(pol["Rule"])))
    agQ, agM = 100 * np.mean(E["agree"]["Q-learning"]), 100 * np.mean(E["agree"]["Monte Carlo"])
    og = np.array(E["optimism_gap"]); n_og_neg = int((og < 0).sum())
    cq = np.array(E["curves"]["Q-learning"]); cm = np.array(E["curves"]["Monte Carlo"])
    q_first, m_first, step_first = float(cq[:, 1, 1].mean()), float(cm[:, 1, 1].mean()), int(cq[0, 1, 0])
    q_last, m_last = float(cq[:, -10:, 1].mean()), float(cm[:, -1, 1].mean())
    J_or_m, J_rule_m = E["J_oracle_model"]["Oracle DP"], E["J_oracle_model"]["Rule"]
    tfac = float(stats.t.ppf(0.975, nrep - 1) / 1.96)
    sig = lambda g, k: abs(G[gi[g], ki[k]]) > GC[gi[g], ki[k]] * tfac
    lows = {k: S[k]['J'] - S[k]['ci'] for k in ('DP', 'Monte Carlo', 'Q-learning')}
    n_clear = sum(1 for k in lows if lows[k] > rule_J)
    learner_gain = [S[k]['J'] - rule_J for k in ('DP', 'Monte Carlo', 'Q-learning')]

    # --- sweep derived quantities
    k0 = ki[0]; k25 = ki[0.25]
    n_improve = 0; max_improve = 0.0; n_total = 0
    for g in GR:
        for k in KS:
            if k == 0: continue
            n_total += 1
            dlt = JJ[gi[g], ki[k]] - JJ[gi[g], k0]
            if dlt > 1e-9:
                n_improve += 1; max_improve = max(max_improve, float(dlt))
    gain0 = {g: G[gi[g], k0] for g in GR}
    best_abs = r["best"]
    best_gain_cell = np.unravel_index(np.argmax(G), G.shape)
    worst_gain_cell = np.unravel_index(np.argmin(G), G.shape)
    n_below = int((G < 0).sum()); n_below_sig = int(sum(1 for g in GR for k in KS if G[gi[g], ki[k]] < 0 and sig(g, k)))
    n_above_sig = int(sum(1 for g in GR for k in KS if G[gi[g], ki[k]] > 0 and sig(g, k)))
    pess_p_sig = [g for g in GR if PV[str(g)]["p"] < 0.05]
    gap_all_neg = bool((GAP < 0).all())
    abs_range = (float(JJ[:, k0].min()), float(JJ[:, k0].max()))
    rule_range = (min(x[0] for x in RULE), max(x[0] for x in RULE))
    mort_rule = {g: 100 * RULE[gi[g]][1] for g in GR}
    g2, g3, g5, g8 = GR

    P = Paper("Granularity and Pessimism in Offline Sepsis Policy Learning: Where the Support Constraint Does the Work",
              "sepsis; intensive care; fluids and vasopressors; offline reinforcement learning; state discretisation; pessimism; Markov decision process",
              running="Granularity and pessimism in sepsis policies", refdict=REFS)

    P.abstract(
        "Reinforcement-learning policies for sepsis treatment are usually learned offline from retrospective intensive-care records, after the continuous patient state has been discretised, and "
        "they are often regularised to avoid actions the data barely support. Two design choices therefore shape the result: how finely to discretise, and how pessimistic to be. We study both on a "
        "simulated sepsis simulator with a three-dimensional state (blood pressure, lactate, organ-failure score) and nine fluid-and-vasopressor actions, whose variables mirror public ICU "
        "sepsis tables (the Kaggle site was unreachable, HTTP 403, so no real file was used). We sweep bins per variable "
        f"({', '.join(str(g) for g in GR)}) against the strength kappa of a count-based pessimism penalty ({', '.join(str(k) for k in KS)}) with {nrep} replicates and score every policy in the true simulator against the guideline-style rule "
        f"defined on the same discretisation. With the model-based dynamic programming (DP) policy and no penalty, the gain over the rule shrinks as bins get finer: {gain0[g2]:+.1f}, {gain0[g3]:+.1f}, {gain0[g5]:+.1f} and {gain0[g8]:+.2f} "
        f"return units for {g2}, {g3}, {g5} and {g8} bins, as the share of states with supported data falls from {100 * COV[str(g2)]:.0f}% to {100 * COV[str(g8)]:.0f}%; at {g8} bins the learned policy is the rule. "
        f"Pessimism did not help: the penalty at kappa = 0.25 was no better than none for any granularity ({', '.join(f'{PV[str(g)]['mean']:+.2f} at {g} bins' for g in GR)}, paired over replicates), and larger kappa pushed returns "
        f"below the rule at the coarsest and finest settings. This is consistent with the built-in support threshold already providing the pessimism, so that a second penalty double-counts it. "
        f"On the {nS}-state benchmark, DP, Monte Carlo and Q-learning were within noise of one another ({S['DP']['J']:.1f}, {S['Monte Carlo']['J']:.1f}, {S['Q-learning']['J']:.1f}) and above the rule ({rule_J:.1f}), "
        f"but far below an oracle fitted to {cfg['n_oracle']:,} episodes ({S['Oracle DP']['J']:.1f}). All results are simulated and carry no clinical meaning.")

    P.h1("Introduction")
    P.p("Sepsis, a life-threatening response to infection, is managed in intensive care with intravenous fluids and vasopressors, and its definition and severity scoring are well codified [@singer]. The choice of dose at each "
        "step is uncertain and varies across clinicians, which makes sepsis a favourite test case for reinforcement learning (RL) from retrospective records. The AI Clinician learned a policy from a large ICU database by discretising "
        "patient state into clusters and actions into dose bins [@komorowski], and subsequent work moved to continuous state spaces with deep networks [@raghu]. Public data challenges make such records widely available [@reyna]. "
        "Guidelines for the field emphasise that conclusions depend on design choices and on how the learned policy is evaluated [@gottesman,yu].")
    P.p("Two of those choices are the subject of this paper. The first is **granularity**: a tabular policy needs a finite state set, and the analyst chooses how finely to cut each vital sign. Finer cuts represent patients more "
        "faithfully but spread a fixed dataset over more states, so most (state, action) pairs are seen rarely or never. The second is **pessimism**: in offline RL, value estimates for rarely observed actions are unreliable "
        "and optimisation exploits that unreliability, so many methods penalise or forbid actions that are poorly supported [@levine,fujimoto,kumar,jin]. It is natural to assume that the two choices should be tuned together, "
        "and that more pessimism is safer when granularity is fine. We test that assumption directly.")
    P.p("**Contributions.** (i) A transparent ICU simulator whose state can be discretised at four granularities, with a guideline-style rule defined on each, so that gain over the rule is measurable at every granularity. "
        "(ii) A pessimism family, a count-based reward penalty with strength kappa, added to a support threshold that is already present in the pipeline, with a short argument for its functional form (Eq. 3 to 5). "
        f"(iii) A full granularity-by-pessimism sweep ({len(GR)} x {len(KS)} settings, {nrep} replicates each) scored exactly in the simulator, with coverage, agreement-with-rule, mortality and model-value diagnostics. "
        "(iv) A benchmark of DP, Monte Carlo (MC) control and Q-learning at the middle granularity. (v) Candid negative results: pessimism beyond the support threshold never improved on none, and the finest granularity "
        "was safe only because it collapsed to the rule.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached from the build environment (HTTP 403), so no real ICU table was analysed. The simulator mirrors the variables of sepsis tables (blood pressure, lactate, "
        "organ-failure score, fluid and vasopressor doses, outcome) with made-up dynamics. Findings concern algorithms, not patients, and no clinical claim is made.")

    P.h1("Related work")
    P.p("**RL for sepsis.** The AI Clinician is the best-known tabular example [@komorowski]; it clusters physiology into a fixed number of discrete states and actions into dose bins and evaluates by off-policy methods. Deep RL variants use continuous states "
        "[@raghu]. Our sweep asks, in a controlled setting with a known ground truth, how the number of states changes what a tabular learner can deliver.")
    P.p("**Offline RL and pessimism.** Offline RL has to avoid actions absent from the data. Policy constraints keep the learner close to the behaviour policy [@fujimoto], value-penalty methods lower the value of uncertain actions [@kumar], "
        "and theory shows that pessimism is what makes offline learning provably efficient under limited coverage [@jin]; reviews summarise the field [@levine]. Our penalty follows the confidence-bound idea and uses the Hoeffding-type form [@hoeffding].")
    P.p("**Evaluation in healthcare.** Surveys stress that offline learned policies are hard to evaluate and that recommendations should be reported with their uncertainty [@gottesman,yu]. A simulator is not a substitute for clinical evidence, "
        "but it supplies exact policy values, which lets us measure what offline estimators cannot.")
    background(P)

    P.h1("Problem formulation: a sepsis simulator with adjustable discretisation")
    me, le, se = d3.me, d3.le, d3.se
    P.p("A simulated patient has three continuous variables: mean arterial pressure (MAP, mmHg), lactate (mmol/L) and an organ-failure score (SOFA-like, 0 to 20; higher is worse [@singer]). At admission MAP is normal with mean 62 and "
        "standard deviation 8 (clipped to [40, 100]), lactate is lognormal with median 2.8 (log-sd 0.4), and the score is uniform on [3, 9]. At each of up to 24 decision steps the agent chooses one of three fluid levels (0 to 2) "
        "and one of three vasopressor levels (0 to 2), giving nine actions. MAP relaxes halfway toward a target that falls with the score and rises with vasopressor (+7 per level) and fluid (+2.5 per level) and carries Gaussian noise (sd 3). "
        "Lactate relaxes 40% of the way toward a source term that grows with the score and with low MAP, and is further raised by the highest vasopressor level. The score rises when lactate is above 3.5 or MAP below 60, "
        "falls when MAP is at least 68 with lactate below 2.5, and is raised by the highest fluid level in patients who are already sick (score at least 8). The episode ends with a reward of −100 if the score reaches 14 or MAP falls below 42 (death), "
        "+100 if the score is at most 2 with lactate below 2.0 (recovery), and otherwise after 24 steps with reward 40 − 5 × score. Every other step costs 0.1.")
    P.table([["Element", "Definition"],
             ["State", f"MAP band × lactate band × score band, g bins each (g in {GR}; {', '.join(str(g ** 3) for g in GR)} states). For g = 3 the MAP cut points are {me[0]:.1f}, {me[1]:.1f} mmHg, lactate {le[0]:.2f}, {le[1]:.2f} mmol/L, score {se[0]:.1f}, {se[1]:.1f}"],
             ["Actions (9)", "fluid level (0, 1, 2) × vasopressor level (0, 1, 2); labelled f0v0 ... f2v2"],
             ["Reward", "−0.1 per step; −100 on death (score ≥ 14 or MAP < 42); +100 on recovery (score ≤ 2 and lactate < 2.0); at step 24: 40 − 5 × score"],
             ["Discount, horizon", f"γ = {d3.gamma}; up to {d3.max_steps} decision steps"],
             ["Guideline-style rule", "vasopressor level from the MAP band (highest level in the lowest band); fluid level from the lactate band (highest level in the top band); defined for every g by the band's fractional position"],
             ["Behaviour policy", f"the rule with ε = {d3.behavior_eps}: with probability 0.25 a uniform random one of the nine actions"],
             ["Dataset mirrored", "ICU sepsis tables (PhysioNet/CinC 2019, eICU demo): MAP, lactate, SOFA, fluids, vasopressors, outcome"]],
            "The sepsis MDP. Only the discretisation g changes between the settings of the sweep; the simulator is identical.", widths=[3.2, 13.2], font=8.5)
    P.p("Two properties of this design matter for the analysis. First, the rule is a function of the banded state, so it changes with g: a two-bin rule is crude (a patient is either low or high on each variable) and an eight-bin rule is nearly continuous. "
        f"Its true return is therefore {', '.join(f'{RULE[gi[g]][0]:.1f}' for g in GR)} for {', '.join(str(g) for g in GR)} bins (simulated mortality {', '.join(f'{mort_rule[g]:.1f}%' for g in GR)}). "
        "The gain of a learned policy over the rule on the same discretisation is the quantity we report, because absolute return confounds the quality of the state representation with the quality of the learner. "
        f"Second, the number of state-action cells grows as 9 g³: {', '.join(f'{cells[g]:,}' for g in GR)} for {', '.join(str(g) for g in GR)} bins, while the dataset is fixed at {cfg['n_eps']} logged episodes, so "
        "data per cell collapses as g increases.")

    P.h1("Method: support-aware pessimism and a granularity sweep")
    P.p("The learner builds an empirical MDP from logged transitions with two safeguards already present in the pipeline: a pair (s, a) observed fewer than 10 times is treated as unsupported and sent to an absorbing state with the worst observed reward, "
        "and one pseudo-observation of absorption is added to supported pairs. On top of this we add a graded, count-based pessimism: the reward in the Bellman backup is reduced by a penalty that shrinks as the pair is observed more often. With "
        "n(s, a) the number of observations and Δr the range of rewards in the data,")
    P.eq("Q_κ(s, a) = R̂(s, a) − κ Δr / √n(s, a) + γ Σ_{s'} P̂(s' | s, a) V_κ(s')", "3")
    P.p("and the policy is greedy with respect to Q_κ, with V_κ(s) = max_a Q_κ(s, a); κ = 0 recovers the unpenalised planner.")
    P.p("The form follows from a Hoeffding-type bound [@hoeffding]: for rewards in an interval of width Δr, the estimate R̂ built from n independent observations is within Δr √(ln(2/δ) / (2n)) of the truth with probability at least 1 − δ. "
        "Equating the penalty to this half-width gives")
    P.eq("κ(δ) = √( ln(2/δ) / 2 ),", "4")
    P.p(f"which is {kap_hoeff:.2f} for δ = 0.05, so even the largest strength tested ({KS[-1]}) is weaker than a 95% lower confidence bound on each reward would require. The bound covers reward error only; error in P̂ also contributes and is not penalised. "
        "Because the penalty is subtracted at every step, it behaves as an extra negative step reward. In an episodic task with absorbing outcomes this has a side effect worth stating: when the penalty at the best-supported action exceeds the "
        "expected improvement from continuing, a planner prefers actions that end the episode sooner. We return to this in Section 7.3.")
    P.p("**Granularity and coverage.** Let c(g) be the fraction of states with at least one supported action. As g grows, c(g) falls, and in every unsupported state the deployed policy falls back to the rule. The learned policy therefore interpolates between "
        "an unconstrained plug-in planner when c is close to one and the rule itself when c is close to zero. We define the gain over the rule,")
    P.eq("Gain(g, κ) = J_true(π_{g,κ}) − J_true(rule_g),", "5")
    P.p("with both terms measured in the true simulator on the same discretisation, and report it with a 95% interval over replicates (the rule is a single deterministic policy, so its return has no replicate variation). "
        "The sweep uses DP on the penalised model only; MC and Q-learning are compared at g = 3 with no penalty.")
    P.code("""Algorithm 1  Granularity x pessimism sweep
 for g in {2,3,5,8}:                                   # bins per variable
     dom <- Sepsis(g);  rule_g <- guideline-style rule on the g-bin state;  J_rule <- evaluate(rule_g) in the true simulator
     for replicate in 1..5:
         D <- 500 episodes from the epsilon-noisy rule;   M <- empirical MDP (support threshold n >= 10, absorption pseudo-count)
         for kappa in {0, 0.05, 0.1, 0.25, 0.5, 1.0}:
             pen(s,a) <- kappa * (r_max - r_min) / sqrt(n(s,a))   if n(s,a) > 0 else 0
             pi <- value_iteration(M, reward = R - pen);   pi[unseen states] <- rule_g
             record  J_true(pi) (300 episodes), mortality, J_model(pi) - J_true(pi), agreement(pi, rule_g) on states with >= 5 visits
     coverage c(g) <- share of states with at least one supported action
 report  Gain(g,kappa) = J_true(pi) - J_rule  with 95% CI over replicates""")

    protocol(P, [["Item", "Setting"],
                 ["Simulator", "sepsis MDP; MAP, lactate, score; 9 fluid-vasopressor actions; up to 24 steps"],
                 ["Benchmark (g = 3)", f"{nS} states; {cfg['n_eps']} logged episodes per replicate; oracle MDP from {cfg['n_oracle']:,} episodes; {cfg['seeds']} replicates; {cfg['n_eval']} evaluation episodes"],
                 ["Learners (benchmark)", f"DP (value iteration and policy iteration, which agreed in all replicates: {'yes' if all(E['pi_equals_vi']) else 'no'}), first-visit MC, Q-learning; {cfg['steps']:,} steps"],
                 ["Sweep", f"g in {GR} x kappa in {KS}; {cfg['n_eps']} logged episodes; {nrep} replicates; 300 evaluation episodes per policy"],
                 ["Data seeds", "sweep data use seeds 100-104; the benchmark uses seeds 1000-1004, so the g = 3, kappa = 0 cell of the sweep is an independent draw"],
                 ["Statistics", "95% CI = 1.96 standard errors over replicates; paired one-sample t-tests on per-replicate differences"]],
             cfg["steps"], cfg["seeds"], cfg["n_eval"],
             extra="In the sweep the fallback in unsupported states is the guideline-style rule of the same granularity, not the modal logged action.")

    P.h1("Results")
    P.h2("The three solvers at the middle granularity")
    rows = std_rows(S, ("died", "fluid", "vaso"), ("Death rate", "Fluid levels", "Vasopressor levels"), nd=2)
    t1 = P.table(rows, f"Benchmark at g = 3 ({nS} states). True-simulator return and per-episode outcomes ({cfg['seeds']} replicates, 95% CI; rule and oracle are single deterministic policies).",
                 widths=[5.0, 3.3, 2.4, 2.4, 3.3], font=8.5)
    P.p(f"Table {t1} shows that all three learners improve on the rule on average, with wide intervals: DP {S['DP']['J']:.1f} ± {S['DP']['ci']:.1f}, MC {S['Monte Carlo']['J']:.1f} ± {S['Monte Carlo']['ci']:.1f} and Q-learning "
        f"{S['Q-learning']['J']:.1f} ± {S['Q-learning']['ci']:.1f} against {rule_J:.1f}. Paired over replicates the differences from the rule are {pv['DP'][0]:+.1f} ± {pv['DP'][1]:.1f} ({pstr(pv['DP'][2])}) for DP, "
        f"{pv['Monte Carlo'][0]:+.1f} ± {pv['Monte Carlo'][1]:.1f} ({pstr(pv['Monte Carlo'][2])}) for MC and {pv['Q-learning'][0]:+.1f} ± {pv['Q-learning'][1]:.1f} ({pstr(pv['Q-learning'][2])}) for Q-learning. "
        f"The improvement is therefore not conclusively established for any solver with {seeds} replicates, and the replicate-level picture explains why: DP beat the rule in {n_dp_better} of {seeds} replicates, but replicate {worst_idx} scored {worst_dp:.1f}, below the rule. "
        f"The ranking among the solvers is not resolved (their intervals overlap almost entirely). The oracle DP, fitted to {cfg['n_oracle']:,} episodes, reaches {S['Oracle DP']['J']:.1f}, {gap_or:.1f} above the rule, with a death rate of {S['Oracle DP']['info']['died']:.3f} "
        f"against {S['Rule']['info']['died']:.3f}; it achieves this with much more vasopressor ({S['Oracle DP']['info']['vaso']:.1f} dose-levels per episode against {S['Rule']['info']['vaso']:.1f}) and slightly less fluid ({S['Oracle DP']['info']['fluid']:.1f} against {S['Rule']['info']['fluid']:.1f}). "
        f"Headroom is large, and {cfg['n_eps']} episodes capture only a fraction of it.")
    f1 = P.fig(fig(I, "fig_bars"), "True-simulator return per method at g = 3 (points: mean; bars: 95% CI over 5 replicates).", 6.5)
    P.p(f"Figure {f1} displays the same comparison. The behaviour policy and the rule coincide, the three learners sit {min(learner_gain):.1f} to {max(learner_gain):.1f} points above them, and the oracle sits far above. "
        f"The lower ends of the plotted intervals are above the rule's value for {n_clear} of the three learners, but these intervals use 1.96 standard errors, which with {nrep} replicates are narrower than a t-interval by a factor of {tfac:.2f}; with t-intervals all three would include the rule's value, which is what the paired tests above say.  The pattern is that of an estimator whose variance is large relative to any systematic gain: more episodes would shrink the intervals, while a better state representation would raise the ceiling.")
    f2 = P.fig(fig(I, "fig_learning"), "Learning curves on the oracle MDP (mean ± s.d. over 5 replicates). Dashed: oracle DP optimum; dotted: rule.", 6.5)
    P.p(f"Figure {f2} shows Q-learning and MC rising from near {cq[:, 0, 1].mean():.0f} to about {q_first:.0f} (Q-learning) and {m_first:.0f} (MC) by the first checkpoint at {step_first:,} steps and then creeping up; averaged over the last ten checkpoints Q-learning sits at {q_last:.1f} and MC ends at {m_last:.1f}, "
        f"a few points above the rule ({J_rule_m:.1f} on the oracle model) and far below the oracle optimum ({J_or_m:.1f}). Training longer does not close the gap, because the limit is the information in {cfg['n_eps']} logged episodes. "
        f"Of the {nS} states, the learned policies differ from the rule in {n_dp_diff} (DP), {n_mc_diff} (MC) and {n_q_diff} (Q-learning), whereas the oracle differs in {n_or_diff} ({100 * n_or_diff / nS:.0f}% of states; Figure {f2 + 1}).")
    f3 = P.fig(fig(I, "fig_policies"), "Action chosen in each state observed in the oracle data (colours: fluid and vasopressor levels). The learned policies stay close to the behaviour policy; the oracle departs from it in many states.", 8.5)
    P.p(f"Figure {f3} makes the same point state by state. The learners change the rule in only a handful of states, and different learners change different ones (MC and Q-learning disagree with DP and with each other in the middle columns), "
        f"while the oracle rewrites the policy in a much larger part of the table. Q-learning agreed with DP in {agQ:.0f}% of well-visited states and MC in {agM:.0f}%; MC's lower agreement is consistent with its higher variance. "
        f"The model value of the DP policy minus its true value (optimism gap) was negative in {n_og_neg} of {seeds} replicates, so the estimated model was usually pessimistic, not optimistic. We return to this in Section 7.3.")

    P.h2("Granularity: finer states mean less to learn and a policy that collapses to the rule")
    rows = [["Bins g", "States", "State-action cells", "States with support", "Median n per observed cell", "Rule return (mortality)", "DP, kappa = 0", "Gain over rule", "Agreement with rule"]]
    for g in GR:
        rows.append([str(g), str(g ** 3), f"{cells[g]:,}", f"{100 * COV[str(g)]:.0f}%", f"{MEDN[str(g)]:.1f}", f"{RULE[gi[g]][0]:.1f} ({mort_rule[g]:.1f}%)",
                     pm(JJ[gi[g], k0], JC[gi[g], k0], 1), f"{G[gi[g], k0]:+.2f} ± {GC[gi[g], k0]:.2f}", f"{100 * AGR[gi[g], k0]:.0f}%"])
    t2 = P.table(rows, "Effect of granularity with no penalty (kappa = 0): data coverage, and the learned policy's return, gain over the same-granularity rule and agreement with it (5 replicates).",
                 widths=[1.1, 1.3, 1.9, 1.9, 2.0, 2.4, 2.3, 2.1, 1.7], font=7.5)
    P.p(f"Table {t2} is the central result. As the number of bins per variable rises from {g2} to {g8}, the state-action space grows from {cells[g2]:,} to {cells[g8]:,} cells while the data stay fixed; the share of states with any supported action falls from "
        f"{100 * COV[str(g2)]:.0f}% to {100 * COV[str(g8)]:.0f}%, and the median count among observed cells falls from {MEDN[str(g2)]:.1f} to {MEDN[str(g8)]:.1f}, which is below the support threshold of 10. "
        f"The gain over the rule falls in step: {gain0[g2]:+.2f} ± {GC[gi[g2], k0]:.2f}, {gain0[g3]:+.2f} ± {GC[gi[g3], k0]:.2f}, {gain0[g5]:+.2f} ± {GC[gi[g5], k0]:.2f} and {gain0[g8]:+.2f} ± {GC[gi[g8], k0]:.2f}. "
        f"With t-based intervals the first three {'all exclude' if all(sig(g, 0) for g in (g2, g3, g5)) else 'do not all exclude'} zero, though the g = {g2} and g = {g3} intervals are wide; the last is positive by a hair and means nothing: at {g8} bins the learned policy agrees with the rule in {100 * AGR[gi[g8], k0]:.0f}% of well-visited states, and in the remaining states it is the rule by construction. "
        f"Agreement with the rule rises from {100 * AGR[gi[g2], k0]:.0f}% to {100 * AGR[gi[g8], k0]:.0f}% over the same range, which is the signature of a planner that has little to say. Mortality moves with the gain, from {100 * MORT[gi[g2], k0]:.1f}% (learned) against {mort_rule[g2]:.1f}% (rule) at {g2} bins to "
        f"{100 * MORT[gi[g8], k0]:.1f}% against {mort_rule[g8]:.1f}% at {g8} bins.")
    f4 = P.fig(fig(I, "fig_heatmaps"), "Left: gain over the same-granularity rule for each (bins, kappa) setting (green: better than the rule). Right: coverage, agreement with the rule at kappa = 0, and gain at kappa = 0 (multiplied by 5 for visibility).", 13.0)
    P.p(f"Figure {f4} (left) shows the whole sweep, and the right panel shows the mechanism. The blue line (coverage) and the green line (gain) fall together while the orange line (agreement with the rule) rises. In other words, "
        f"granularity does not make learning harder in an abstract sense; it turns the learned policy into the rule, state by state, because the data no longer support any deviation. The finest granularity has the highest absolute return of the sweep "
        f"({best_abs['J']:.1f} at g = {best_abs['gran']}, kappa = {best_abs['kappa']}), but that value is the rule's own (the rule at {g8} bins scores {RULE[gi[g8]][0]:.1f}); no learning is involved. A reader comparing absolute returns "
        f"across granularities, which range only from {abs_range[0]:.1f} to {abs_range[1]:.1f} at kappa = 0 even though the rule alone ranges from {rule_range[0]:.1f} to {rule_range[1]:.1f}, would wrongly conclude that fine bins are best for learning. "
        f"A second caution is that the coarsest setting offers the largest gain ({gain0[g2]:+.1f}) because the two-bin rule is poor, not because two bins is a good representation: its mortality ({mort_rule[g2]:.1f}% for the rule) is the highest of all rules.")

    P.h2("Pessimism: a graded penalty on top of a hard support threshold does not help")
    rows = [["Bins"] + [f"kappa = {k}" for k in KS]]
    for g in GR:
        rows.append([str(g)] + [f"{G[gi[g], ki[k]]:+.2f}{'*' if sig(g, k) else ''} ± {GC[gi[g], ki[k]]:.2f}" for k in KS])
    t3 = P.table(rows, f"Gain over the same-granularity rule, by granularity and pessimism strength (mean ± 95% CI over 5 replicates). An asterisk marks cells whose t-based 95% interval (the ± value multiplied by {tfac:.2f}) excludes zero.",
                 widths=[1.2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5], font=7.5)
    P.p(f"Table {t3} is the pessimism result. Reading across any row, the gain never rises materially as kappa increases. Of the {n_total} (granularity, kappa > 0) settings, the penalised policy had a higher mean return than the unpenalised one in only {n_improve}, and by at most {max_improve:.2f} return units, "
        f"which is far inside the replicate intervals. For {g3} bins the gain collapses between kappa = {KS[ki[0.1]]} ({G[gi[g3], ki[0.1]]:+.2f}) and kappa = {KS[k25]} ({G[gi[g3], k25]:+.2f}), after which the learned policy is the rule; for {g5} bins the gain reaches zero at kappa = {KS[ki[0.1]]}; "
        f"for {g2} bins it persists until kappa = {KS[k25]} ({G[gi[g2], k25]:+.1f}) and then vanishes. At the largest strengths the policies are worse than the rule: {G[gi[g2], ki[1.0]]:+.1f} at {g2} bins and {G[gi[g8], ki[1.0]]:+.1f} at {g8} bins for kappa = {KS[-1]} "
        f"({n_below_sig} of the {n_below} negative cells have t-based intervals that exclude zero). The worst cell in the table is {GR[worst_gain_cell[0]]} bins at kappa = {KS[worst_gain_cell[1]]}.")
    rows = [["Bins", "Return, kappa = 0", "Return, kappa = 0.25", "Paired difference (kappa 0.25 minus 0)", "Model value minus true value, kappa = 0", "Same, kappa = 0.25"]]
    for g in GR:
        rows.append([str(g), pm(JJ[gi[g], k0], JC[gi[g], k0], 1), pm(JJ[gi[g], k25], JC[gi[g], k25], 1),
                     f"{PV[str(g)]['mean']:+.2f} ± {PV[str(g)]['ci']:.2f} ({pstr(PV[str(g)]['p'])})", f"{BG[str(g)][0]:+.1f}", f"{BG[str(g)][1]:+.1f}"])
    t4 = P.table(rows, "Paired test of moderate pessimism (kappa = 0.25) against none, and the gap between the DP model's value of the chosen policy and its true value.", widths=[1.1, 2.7, 2.7, 4.0, 3.2, 2.7], font=7.5)
    P.p(f"Table {t4} gives the paired comparison. Moderate pessimism (kappa = {KS[k25]}) is not better than none at any granularity, and is significantly worse in {len(pess_p_sig)} of {len(GR)} ({', '.join(str(g) + ' bins' for g in pess_p_sig) or 'none'}): "
        f"{', '.join(f'{PV[str(g)]['mean']:+.2f} ({pstr(PV[str(g)]['p'])}) at {g} bins' for g in GR)}. The right-hand columns explain why a penalty is redundant. The model value of the chosen policy is "
        f"{'below' if gap_all_neg else 'not always below'} its true value in every one of the {GAP.size} sweep settings (the table shows the replicate mean at kappa = 0 and 0.25), and by a large margin as granularity increases ({BG[str(g5)][0]:+.1f} at {g5} bins and {BG[str(g8)][0]:+.1f} at {g8}). "
        "The reason is that unsupported pairs are completed with the worst observed reward and an absorbing state, which is already a strong form of pessimism. When most pairs are unsupported the model undervalues almost every policy, and a second penalty only moves the planner further toward the best-observed action. "
        "The right measure of danger in this setting is therefore not over-optimism (the model is already pessimistic) but pessimism so severe that the planner has nothing to choose among.")
    f5 = P.fig(fig(I, "fig_kappa"), "Left: true return against kappa for each granularity (dotted lines: the rule at that granularity; bars: 95% CI). Right: simulated in-ICU mortality against kappa.", 11.0)
    P.p(f"Figure {f5} shows the same sweep on the return scale and adds mortality. For {g3} bins the return drops from about {JJ[gi[g3], k0]:.0f} at kappa = 0 to about {JJ[gi[g3], k25]:.0f} at kappa = {KS[k25]} and then stays flat at the rule's level; mortality moves from {100 * MORT[gi[g3], k0]:.1f}% to {100 * MORT[gi[g3], k25]:.1f}%. "
        f"For {g2} bins the line is flat until kappa = {KS[ki[0.1]]}, falls to {JJ[gi[g2], ki[0.5]]:.1f} at kappa = {KS[ki[0.5]]} and to {JJ[gi[g2], ki[1.0]]:.1f} at kappa = {KS[-1]}, below the rule ({RULE[gi[g2]][0]:.1f}), and its mortality climbs from {100 * MORT[gi[g2], k0]:.1f}% to {100 * MORT[gi[g2], ki[1.0]]:.1f}%, above the rule's {mort_rule[g2]:.1f}%. "
        f"A penalty that is large relative to the +100 and −100 terminal rewards (the penalty at n = 10 is about 0.32 × kappa × Δr, which is of order 60 × kappa when Δr = 200) can make ending the episode quickly attractive, and the rising mortality at {g2} bins is consistent with that mechanism; "
        "we did not test it directly, so it remains a hypothesis. The tight intervals at 5 and 8 bins show how little variation a mostly unsupported model produces.")
    f6 = P.fig(fig(I, "fig_agreement_rule"), "Agreement of the learned policy with the rule (states with at least 5 visits) as kappa grows, by granularity.", 6.5)
    P.p(f"Figure {f6} traces how pessimism changes the learned policy's resemblance to the rule. For {g2} bins agreement rises from {100 * AGR[gi[g2], k0]:.0f}% (kappa = 0) to {100 * AGR[gi[g2], ki[0.5]]:.0f}% (kappa = {KS[ki[0.5]]}), as expected of a penalty that favours the most observed action, which under an ε-noisy rule is the rule's. "
        f"At {g8} bins agreement is already {100 * AGR[gi[g8], k0]:.0f}% at kappa = 0 and then falls to {100 * AGR[gi[g8], ki[1.0]]:.0f}% at kappa = {KS[-1]}, and at {g5} bins from {100 * AGR[gi[g5], ki[0.1]]:.0f}% at kappa = {KS[ki[0.1]]} to {100 * AGR[gi[g5], ki[1.0]]:.0f}%. "
        "A pessimism penalty is supposed to move a policy toward the behaviour policy, and it does so only when the penalty is moderate and coverage is reasonable; at fine granularity and large kappa it moves the policy away from the rule, possibly toward whichever action in the few supported cells has the largest count, which need not be the rule's. "
        "We did not investigate this further.")

    P.h1("Discussion")
    P.p("**What the sweep says about granularity.** The tabular analyst's choice of bins is a choice of how much of the data to waste. At two bins nearly every state is supported and the learner can find improvements over a crude rule, "
        f"but the representation is so coarse that the policy cannot be good (the rule's mortality is {mort_rule[g2]:.1f}%, while the oracle at three bins reaches {100 * S['Oracle DP']['info']['died']:.1f}%). At eight bins the representation is richer and the rule is better, but the dataset cannot support a single departure from it. "
        f"In between, the gain is distinguishable from zero at {g2}, {g3} and {g5} bins but shrinks steadily with granularity ({gain0[g2]:+.1f}, {gain0[g3]:+.1f}, {gain0[g5]:+.1f}), and the largest gains are where the rule is weakest. Practical advice: report the share of supported states and the agreement with the behaviour or rule policy next to any learned sepsis policy, "
        "since agreement near 100% is a warning that the policy is the clinician's and not the learner's.")
    P.p("**What the sweep says about pessimism.** The hard support threshold already implements the conservative principle that offline RL aims at, so a further penalty adds nothing here and, with a per-step form, risks "
        f"distortions. This is a statement about this pipeline: algorithms without a hard threshold (for example neural Q-learning with a continuous state) have no such safeguard and would need a penalty. The practical lesson is to count the pessimism that is already present before adding more. "
        f"The model-value gap (negative at every setting) is a cheap diagnostic: an estimated value that is far below the true value of the same policy means the model is too pessimistic to rank policies reliably, in which case estimated values should not be used to choose kappa. We could choose kappa here only because the simulator supplies true values.")
    P.p(f"**The solvers.** At the middle granularity DP, MC and Q-learning are indistinguishable. The replicate variability of MC (± {S['Monte Carlo']['ci']:.1f}) is similar to that of DP (± {S['DP']['ci']:.1f}) and Q-learning (± {S['Q-learning']['ci']:.1f}); we did not test why it is not larger on this 24-step task. "
        f"Agreement of MC with DP is lower ({agM:.0f}%) than that of Q-learning ({agQ:.0f}%), in line with MC's higher variance, but it does not translate into a return difference.")
    limitations(P, [
        f"**One discretisation scheme.** Bins are equal-width in MAP and score and geometric in lactate. Data-driven clustering, as in clinical work [@komorowski], may support different conclusions at the same number of states.",
        "**DP only in the sweep.** The pessimism sweep uses DP on a penalised model; MC control and Q-learning with a penalty were not tested and might react differently.",
        "**Penalty form.** Only one functional form (1/√n on the reward) was tested. Penalties on transition uncertainty or on the value of the behaviour policy are different and were not tested.",
        f"**Five replicates.** Several intervals are wide (for example {GC[gi[g2], k0]:.1f} at {g2} bins) and the 1.96-standard-error intervals are narrower than t-intervals by a factor of {tfac:.2f}; differences between nearby kappa values are not resolved, and only the kappa = 0.25 comparison has a paired test.",
        "**Rule on each granularity.** The rule is constructed from the band's fractional position and is not a validated clinical guideline; the size of the gain depends on how good it is.",
    ])
    ethics_future(P, "(1) Replace equal-width bins by clusters learned from the data and repeat the sweep at matched numbers of states; (2) test penalties on transition uncertainty and a per-episode (not per-step) penalty that cannot reward early termination; "
                  "(3) run MC control and Q-learning in the sweep; (4) report the support share and agreement with the clinician alongside any learned sepsis policy and use them to choose granularity; (5) apply the adapter to a real ICU table "
                  "and compare model-based value estimates with importance-sampling estimates on a held-out split.")
    P.h1("Conclusion")
    P.p(f"In a simulated sepsis task, a tabular learner's gain over a same-granularity rule falls from {gain0[g2]:+.1f} at {g2} bins per variable to {gain0[g8]:+.2f} at {g8}, as the share of supported states falls from {100 * COV[str(g2)]:.0f}% to {100 * COV[str(g8)]:.0f}% and the learned policy converges to the rule. "
        f"A count-based pessimism penalty improved on no setting by more than {max_improve:.2f} return units, was significantly harmful at moderate strength in {len(pess_p_sig)} of {len(GR)} granularities, and made the policy worse than the rule at the extremes; this is consistent with the hard support threshold already supplying the pessimism, as the negative model-value gaps indicate. "
        f"At the middle granularity the three solvers were within noise of each other ({S['DP']['J']:.1f}, {S['Monte Carlo']['J']:.1f}, {S['Q-learning']['J']:.1f}) and of the rule ({rule_J:.1f}) and well below the oracle ({S['Oracle DP']['J']:.1f}). "
        "All findings are from a simulator and are method evidence, not clinical evidence.")
    defense(P, [
        ("Is it meaningful to call a policy better than the rule when the intervals are so wide?",
         f"Only where the (t-based) interval excludes zero, and Table {t3} marks those cells. At g = {g3} the gain at kappa = 0 is {G[gi[g3], k0]:+.2f} ± {GC[gi[g3], k0]:.2f}, which {'excludes' if sig(g3, 0) else 'does not exclude'} zero on that basis, but the benchmark comparison at the same granularity is {pv['DP'][0]:+.1f} ± {pv['DP'][1]:.1f} ({pstr(pv['DP'][2])}) for DP. We draw conclusions about trends across settings, not about any single cell."),
        ("Why is the finest granularity not the best, when it has the highest absolute return?",
         f"Because its return is the rule's. At {g8} bins the learned policy agrees with the rule in {100 * AGR[gi[g8], k0]:.0f}% of well-visited states and the gain over the rule is {gain0[g8]:+.3f} (Table {t2}). The rule is better at finer granularity, and learning has nothing to add."),
        ("Does your result mean pessimism is useless in offline RL?",
         f"No. It means that in this pipeline a hard support threshold is already in place, and the model values are already below the true values in every setting (Table {t4}). Methods without such a threshold would need a penalty. We also tested only one penalty form and only DP."),
        ("Why does a larger penalty make the policy worse than the rule?",
         f"The data show that it does (Table {t3}); the mechanism is a hypothesis. A per-step penalty is an extra negative reward and can make early termination attractive; at {g2} bins mortality rises to {100 * MORT[gi[g2], ki[1.0]]:.1f}% at kappa = {KS[-1]}, above the rule's {mort_rule[g2]:.1f}%, which is consistent with that mechanism but does not prove it."),
        ("What would you need to run this on real data?",
         "Map the table to (MAP, lactate, score, dose, reward) rows, discretise with the same bins, and call the adapter in the code package. A real table has no simulator, so the headline measure, true gain over the rule, would have to be replaced by held-out importance-sampling estimates, which have their own variance problems; "
         "the support share and agreement with the clinician can be computed without a simulator and are the first things to report."),
    ])
    repro(P, 2, ["fig_bars", "fig_learning", "fig_policies", "fig_agreement", "fig_heatmaps", "fig_kappa", "fig_agreement_rule"])
    code_appendix(P, 2, [domains.Sepsis])
    rl_core_appendix(P)
    P.refs()
    path = f"{PAP}/Paper02_Sepsis_Granularity_Pessimism.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
