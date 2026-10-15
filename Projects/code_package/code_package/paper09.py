import re
import inspect
import numpy as np
from scipy import stats
from docx_helper import Paper
from paper_common import *
import domains
import ideas

I = 9
ALG = ("Monte Carlo", "Q-learning")


def build():
    r = load(I); RUN = r["runs"]; HS = r["HS"]
    # experiment configuration: results.json stores summaries only, so the call parameters are read from the source of ideas.idea9
    m_cfg = re.search(r"n_eps=(\d+), n_oracle=(\d+), steps=(\d+), seeds=(\d+), n_eval=(\d+)", inspect.getsource(ideas.idea9))
    N_EPS, N_OR, STEPS, NSEED, N_EVAL = (int(x) for x in m_cfg.groups())
    tf = float(stats.t.ppf(0.975, NSEED - 1) / 1.96)             # t-interval factor for the replicate count
    tcrit = float(stats.t.ppf(0.975, NSEED - 1))
    dom = domains.Rehab(sparse=False, H=12)
    gam = dom.gamma
    key = lambda H, sp: f"H{H}_{'sparse' if sp else 'dense'}"
    Rn = {(H, sp): RUN[key(H, sp)] for H in HS for sp in (False, True)}
    J = lambda H, sp, m: Rn[(H, sp)]["summary"][m]["J"]
    C = lambda H, sp, m: Rn[(H, sp)]["summary"][m]["ci"]
    INF = lambda H, sp, m, k: Rn[(H, sp)]["summary"][m]["info"][k]
    INFC = lambda H, sp, m, k: Rn[(H, sp)]["summary"][m]["info_ci"][k]
    FIN = lambda H, sp, m: Rn[(H, sp)]["final"][m]
    nstates = {H: 6 * 3 * H for H in HS}
    LEARN = ("DP", "Monte Carlo", "Q-learning")

    def vs_rule(H, sp, m):                   # difference from the rule; significance from the t-widened interval of the learner
        d = J(H, sp, m) - J(H, sp, "Rule"); hw = C(H, sp, m) * tf
        return d, hw, ("up" if d - hw > 0 else "down" if d + hw < 0 else "ns")

    def cmp(H, sp, a, b):                    # unpaired approximation from stored intervals (no per-replicate values are stored)
        se = np.sqrt((C(H, sp, a) / 1.96) ** 2 + (C(H, sp, b) / 1.96) ** 2)
        d = J(H, sp, a) - J(H, sp, b)
        return d, tcrit * float(se), abs(d) > tcrit * float(se)

    def cap(H, sp, m):                       # headroom captured
        rl, orc = J(H, sp, "Rule"), J(H, sp, "Oracle DP")
        return (J(H, sp, m) - rl) / (orc - rl)

    mark = {"up": "*", "down": "†", "ns": ""}
    VR = {(H, sp, m): vs_rule(H, sp, m) for H in HS for sp in (False, True) for m in LEARN}
    n_up_d = sum(1 for (H, sp, m), v in VR.items() if not sp and v[2] == "up")
    n_up_s = sum(1 for (H, sp, m), v in VR.items() if sp and v[2] == "up")
    n_dn_s = sum(1 for (H, sp, m), v in VR.items() if sp and v[2] == "down")
    n_ns_s = sum(1 for (H, sp, m), v in VR.items() if sp and v[2] == "ns")
    n_cfg = len(LEARN) * len(HS)
    gain_range_d = (min(VR[(H, False, m)][0] for H in HS for m in LEARN), max(VR[(H, False, m)][0] for H in HS for m in LEARN))
    orc_d = {H: J(H, False, "Oracle DP") - J(H, False, "Rule") for H in HS}
    orc_s = {H: J(H, True, "Oracle DP") - J(H, True, "Rule") for H in HS}
    sp_dn = [H for H in HS if orc_s[H] < 0]; sp_up = [H for H in HS if orc_s[H] > 0]
    capd = {(H, m): cap(H, False, m) for H in HS for m in LEARN}
    caps24 = {m: cap(24, True, m) for m in LEARN}
    # identity check: sparse return = gamma^(H-1) x total gain, for the rule
    ident = {H: (J(H, True, "Rule") / INF(H, True, "Rule", "gain"), gam ** (H - 1)) for H in HS}
    ident_err = max(abs(a - b) for a, b in ident.values())
    ratio_sd = {H: J(H, True, "Rule") / J(H, False, "Rule") for H in HS}
    sd_s0 = (40 - 12) / np.sqrt(12.0)
    per_state = N_EPS * 1.0 / 18.0                             # logged transitions per state if visits were spread evenly (independent of H)
    p_off = dom.behavior_eps / dom.nA
    p_rule = 1 - dom.behavior_eps + p_off
    pair_off, pair_off_or = per_state * p_off, N_OR / 18.0 * p_off
    reach_n = int(round(sum(Rn[(H, sp)]["reach"][a] * NSEED for H in HS for sp in (False, True) for a in ALG)))
    n_runs = len(HS) * 2 * len(ALG) * NSEED
    max_final = max(FIN(H, sp, a)["mean"] for H in HS for sp in (False, True) for a in ALG)
    max_final_k = max(((H, sp, a) for H in HS for sp in (False, True) for a in ALG), key=lambda k: FIN(*k)["mean"])
    mcq = {(H, sp): Rn[(H, sp)]["mc_minus_q"] for H in HS for sp in (False, True)}
    rank_d = {H: sorted(ALG + ("DP",), key=lambda m: -J(H, False, m)) for H in HS}
    inj = lambda H, sp, m: INF(H, sp, m, "injury")
    inj_up = lambda H, sp, m: inj(H, sp, m) - inj(H, sp, "Rule")
    def inj_sig(H, sp, m):
        d = inj_up(H, sp, m); hw = INFC(H, sp, m, "injury") * tf
        return "up" if d - hw > 0 else "down" if d + hw < 0 else "ns"
    n_inj_dn_d = sum(1 for H in HS for m in LEARN if inj_sig(H, False, m) == "down")
    n_inj_up_s = sum(1 for H in HS for m in LEARN if inj_sig(H, True, m) == "up")
    n_inj_dn_s = sum(1 for H in HS for m in LEARN if inj_sig(H, True, m) == "down")
    mc_inj_frac = {H: inj(H, True, "Monte Carlo") / H for H in HS}
    q_fin_d = [FIN(H, False, "Q-learning")["mean"] for H in HS]; q_fin_s = [FIN(H, True, "Q-learning")["mean"] for H in HS]
    mc_fin_d = [FIN(H, False, "Monte Carlo")["mean"] for H in HS]; mc_fin_s = [FIN(H, True, "Monte Carlo")["mean"] for H in HS]
    n_neg_fin_s = sum(1 for H in HS for a in ALG if FIN(H, True, a)["mean"] < 0)
    n_sig_neg_fin = sum(1 for H in HS for sp in (False, True) for a in ALG if FIN(H, sp, a)["mean"] + FIN(H, sp, a)["ci"] * tf < 0)
    mc_sparse_loss = {H: J(H, True, "Monte Carlo") - J(H, True, "Rule") for H in HS}

    # guards for the qualitative wording below: the build fails loudly if the data no longer support it
    assert n_up_d == n_cfg and n_up_s == 0 and n_dn_s > 0 and sp_dn == [6, 12] and sp_up == [24]
    assert all(rank_d[24][0] == "Monte Carlo" for _ in [0]) and rank_d[6][-1] == "Monte Carlo"
    assert ident_err < 0.01 and all(mcq[(H, True)]["p"] < 0.05 for H in HS)
    assert reach_n == 0 and n_inj_dn_d == n_cfg
    assert q_fin_d[0] > q_fin_d[1] > q_fin_d[2] and q_fin_s[0] > q_fin_s[1] > q_fin_s[2]
    assert all(capd[(H, "DP")] > capd[(H + k, "DP")] for H, k in ((6, 6), (12, 12)))
    assert mcq[(12, False)]["p"] > 0.05 and mcq[(6, False)]["p"] < 0.05 and mcq[(24, False)]["p"] < 0.05

    P = Paper("Horizon and Reward Sparsity in Reinforcement Learning for Stroke Rehabilitation Intensity: When Terminal-Only Rewards Break Learning",
              "stroke rehabilitation; treatment horizon; sparse reward; dense reward; Monte Carlo control; Q-learning; reward design; fatigue; injury",
              running="Rehabilitation horizon and sparse reward", refdict=REFS)

    P.abstract(
        "Choosing how hard a stroke patient should train each week balances faster recovery against fatigue and injury, and the reward that a learning algorithm sees can be given weekly (dense) or only as the total improvement at the end of the programme (sparse). "
        f"We study how the treatment horizon (6, 12 and 24 weeks) and reward sparsity affect dynamic programming (DP), Monte Carlo (MC) control and Q-learning in a simulator that mirrors the variables of public stroke and rehabilitation tables (the Kaggle site was unreachable, HTTP 403, so no real file was used and all results are simulated). "
        f"Each configuration uses {N_EPS} logged patient courses and {NSEED} replicates. "
        f"With dense rewards all {n_cfg} learner-horizon combinations beat a fatigue-band rule in the true simulator by {gain_range_d[0]:.1f} to {gain_range_d[1]:.1f} return units, but they capture a shrinking part of the headroom that a DP policy fitted to ten times as much data reveals: for DP {100 * capd[(6, 'DP')]:.0f}%, {100 * capd[(12, 'DP')]:.0f}% and {100 * capd[(24, 'DP')]:.0f}% at 6, 12 and 24 weeks. "
        f"With sparse rewards no learner was significantly above the rule in any of {n_cfg} combinations ({n_dn_s} were significantly below, {n_ns_s} not distinguishable), MC with terminal-only reward was far below it (returns {J(6, True, 'Monte Carlo'):.1f}, {J(12, True, 'Monte Carlo'):.1f} and {J(24, True, 'Monte Carlo'):.1f}, with up to {inj(24, True, 'Monte Carlo'):.1f} injury weeks per course), "
        f"and even the oracle DP was below the rule at {sp_dn[0]} and {sp_dn[1]} weeks. The two reward types are almost the same objective in the true process (the sparse return equals the total gain times gamma^(H-1)), so the sparse-reward failure comes from what the learners can estimate and not from a different goal. "
        f"No run reached 90% of the optimal improvement on the fitted model ({reach_n} of {n_runs}). All results are simulated and carry no clinical meaning.")

    P.h1("Introduction")
    P.p("After a stroke, recovery of motor function depends strongly on the amount and intensity of rehabilitation, and guidelines recommend intensive, task-oriented therapy while warning about overload [@winstein]. A therapist therefore decides each week how hard to push, with more intensive sessions giving more recovery but also more fatigue and, past a point, a risk of injury. "
        "This is a sequential decision problem whose horizon is the length of the programme, from a few weeks of inpatient care to a half-year of outpatient therapy.")
    P.p("Two design choices shape what reinforcement learning (RL) can do with such a problem. The first is the horizon: longer programmes need more decisions, and errors in an estimated model compound over more steps. "
        "The second is the reward. A natural reward for a rehabilitation programme is the final improvement in a functional score, which is observed only at the end; a dense alternative credits the weekly change. "
        "The textbook view is that sparse rewards make credit assignment harder because a single number must be attributed to many earlier decisions [@sb], and that reward shaping can help without changing the optimal policy under the right conditions [@ng99].")
    P.p("We test these views in a controlled simulator. A fact about the simulator makes the comparison clean: dense and sparse rewards are two ways of paying out the same total improvement, so the true objective is nearly identical and any difference in outcome comes from what the learners can estimate from data. "
        "We report what happens, including the cases in which the learned policies are worse than a simple rule.")
    P.p("**Contributions.** (i) A rehabilitation MDP with a score, a fatigue state, an injury risk and a variable horizon (Section 4). "
        "(ii) A matched dense and sparse formulation, with a check that the two define almost the same true return (Eq. 3). "
        "(iii) A factorial study of three horizons by two reward types for DP, MC and Q-learning, evaluated in the true simulator and on a fitted model (Eq. 4, 5). "
        "(iv) A hypothesis about the hidden-history term in a sparse reward (Eq. 6), which we state but do not test. "
        f"(v) Negative results: no learner reliably beats the rule under sparse rewards, MC under sparse rewards is far worse than the rule, and even the oracle DP is below the rule at {sp_dn[0]} and {sp_dn[1]} weeks.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached from the build environment (all requests returned HTTP 403), so no real stroke or rehabilitation table was analysed. The simulator mirrors the kind of variables found in such tables (functional score, fatigue, therapy intensity, adverse event) with made-up dynamics. "
        "Findings concern algorithms and reward design, not patients, and no clinical claim is made.")

    P.h1("Related work")
    P.p("**Stroke rehabilitation.** Clinical guidelines emphasise early, intensive, task-specific practice and monitoring for fatigue and adverse events [@winstein]. Our rule, which picks a lighter therapy as fatigue rises, is a stylised version of that advice and not a guideline.")
    P.p("**Reward design.** The reward is the specification of the task; with a sparse terminal reward, credit for the outcome has to propagate backwards through every decision [@sb]. Potential-based shaping adds a term of the form gamma times a potential of the next state minus the potential of the current state, and leaves the optimal policy unchanged [@ng99]. "
        "In our simulator the weekly gain is exactly the change in the score, so with gamma equal to one the dense reward is a potential-based rewrite of the terminal one; with gamma below one it differs by discounting only.")
    P.p("**RL in healthcare.** Surveys describe the opportunities and the pitfalls of RL for treatment, including evaluation without a simulator and the sensitivity to reward choice [@yu,gottesman]. Dynamic treatment regimes provide the statistical framing for multi-stage decisions [@murphy03,murphy05]. "
        "Offline learning from logs is limited by coverage, which motivates support constraints and pessimism [@levine,jin,kumar].")
    background(P)

    P.h1("Problem formulation: a stroke rehabilitation simulator")
    P.p(f"A patient has a functional score between 0 and 66 (higher is better), initially uniform on [12, 40], and a fatigue level between 0 and 1, initially uniform on [0, 0.4]. "
        f"Each week the therapist chooses one of four intensities: rest, light, moderate or high. The weekly gain is the intensity's base gain ({', '.join(f'{x:g}' for x in dom.GAIN)} for rest, light, moderate and high) times (1 − score / 66) times (1.1 − fatigue), plus Gaussian noise with standard deviation 0.4. "
        f"Fatigue changes by {', '.join(f'{x:+.2f}' for x in dom.FAT)} for the four intensities plus Gaussian noise with standard deviation 0.07, and is kept between 0 and 1. "
        "An injury occurs when the new fatigue exceeds 0.85 and the intensity is moderate or high; it subtracts 2.5 from that week's gain. The score is kept between 0 and 66 after the gain is applied, and the recorded gain is the actual change in score. "
        f"The programme lasts H weeks (H = {', '.join(str(h) for h in HS)}), with discount gamma = {gam}.")
    P.table([["Element", "Definition"],
             [f"State ({', '.join(str(nstates[h]) for h in HS)})", f"score band (6; cut points {', '.join(str(x) for x in dom.SE)}) × fatigue band (3; cut points {', '.join(str(x) for x in dom.FE)}) × week index (H); 18 H states for H = {', '.join(str(h) for h in HS)}"],
             ["Actions (4)", "rest, light, moderate, high (base gains " + ", ".join(f"{x:g}" for x in dom.GAIN) + "; fatigue change " + ", ".join(f"{x:+.2f}" for x in dom.FAT) + ")"],
             ["Dense reward", "weekly gain in score (actual change after clipping and injury)"],
             ["Sparse reward", "0 in every week but the last; in the last week, final score minus initial score"],
             ["Injury", "fatigue above 0.85 with a moderate or high intensity: gain reduced by 2.5"],
             ["Fatigue-band rule", "high intensity at low fatigue, moderate at medium fatigue, light at high fatigue"],
             ["Behaviour policy", f"the rule with ε = {dom.behavior_eps} uniformly random actions"],
             ["Dataset mirrored", "Stroke Prediction table and rehabilitation telemonitoring data (score, therapy intensity, adverse event)"]],
            "The stroke rehabilitation MDP.", widths=[3.4, 13.0], font=8.5)
    P.p(f"Three simulator facts matter. First, the state contains the week index, so the number of states is 18 H, but a log of {N_EPS} episodes provides {N_EPS} H transitions, so the average number of logged transitions per state is {per_state:.0f} whatever the horizon (visits are not spread evenly, so this is an average). "
        f"Second, the behaviour policy takes the rule's action with probability {p_rule:.4f} and each other action with probability {p_off:.4f}, so a given off-rule state-action pair is expected to be seen about {pair_off:.1f} times with {N_EPS} episodes and about {pair_off_or:.0f} times with {N_OR:,}; pairs seen fewer than 10 times are treated as unsupported. "
        f"Third, the sparse reward depends on the initial score, which is uniform on [12, 40] (standard deviation {sd_s0:.1f}) and is not part of the state. The effect of one week's intensity on the score is a few points at most (base gains up to {max(dom.GAIN):g}), smaller than the spread of initial scores, which is the reason a terminal-only reward is a noisy signal for any single week.")

    P.h1("Method: matched dense and sparse returns, headroom and learning value")
    P.p("Let S_t be the score at week t, g_t = S_{t+1} − S_t the recorded gain, and H the horizon. The dense and sparse returns of a course are")
    P.eq("G_dense = Σ_{t=0}^{H−1} γ^t g_t,      G_sparse = γ^{H−1} (S_H − S_0) = γ^{H−1} Σ_{t=0}^{H−1} g_t.", "3")
    P.p("Because g_t is the actual change of the clipped score, the gains telescope exactly, so the two returns differ only in the weights that discounting gives to individual weeks. "
        "Equation (3) gives a numerical check that we apply to the rule: the sparse return divided by the total (undiscounted) gain should equal gamma to the power H − 1.")
    P.p("To compare learners across configurations whose returns have different scales, we normalise the value of a policy on the large oracle model (an MDP fitted to ten times as many episodes) against the behaviour policy and the oracle optimum, and we define the share of the achievable improvement over the rule that a learned policy captures in the true simulator:")
    P.eq("v(π) = [J_model(π) − J_beh] / [J_opt − J_beh],      c(π) = [J(π) − J(rule)] / [J(oracle) − J(rule)].", "4")
    P.p("The first measures learning on the model scale, where v = 0 is the behaviour policy and v = 1 the model optimum; we also record the first checkpoint at which v reaches 0.9. The second is computed from true-simulator returns; it is defined only when the oracle beats the rule. "
        "The paired difference of final normalised values between MC and Q-learning is tested with a one-sample t-test over replicates.")
    P.eq("t90 = min { t_j : v(π_{t_j}) ≥ 0.9 },      reach = fraction of replicates for which t90 exists.", "5")
    P.p("Finally, we state a hypothesis about the sparse reward as the tabular model sees it. In the last week, the logged reward is S_H − S_0, and the model averages it over all logged visits to a state-action pair (s, a):")
    P.eq("R̂_sparse(s, a) = E[S_H − S_{H−1} | s, a] + E[S_{H−1} − S_0 | s].", "6")
    P.p("The first term is the effect of the last action and is what matters for the decision. The second term, the improvement accumulated before the last week, is hidden by the state; the model assigns it as a constant to each last-week state, as an average over the logging population. "
        "A policy that arrives in a state by a different route than the logged patients is credited with the logged patients' average improvement and not its own. This hidden-history term does not exist in the dense reward. We did not test whether it explains the results below.")
    P.code("""Algorithm 1  Horizon x reward study
 input : horizons H in {6, 12, 24}, reward type in {dense, sparse}, 800 logged episodes, 5 replicates
 1. for each (H, reward): simulate logs with the rule + 25% random actions
 2. fit the empirical MDP (pairs seen < 10 times unsupported); fit an oracle MDP from 8000 episodes
 3. DP: value iteration on the fitted MDP;  MC: first-visit control;  Q-learning with visit-count steps (250000 steps each)
 4. fill unobserved states with the logged modal action
 5. record greedy-policy value on the oracle MDP at 25 checkpoints; normalise with Eq. (4); t90 from Eq. (5)
 6. evaluate each final policy in the true simulator (250 episodes); report J, total gain and injury weeks
 7. compare the rule, DP, MC, Q-learning and the oracle DP in each configuration""")

    protocol(P, [["Item", "Setting"],
                 ["Simulator", f"rehabilitation MDP; {', '.join(str(nstates[h]) for h in HS)} states for H = {', '.join(str(h) for h in HS)}; 4 actions; gamma = {gam}"],
                 ["Design", f"3 horizons × 2 reward types = {len(HS) * 2} configurations; {N_EPS} logged episodes; oracle MDP from {N_OR:,} episodes; {NSEED} replicates; {N_EVAL} evaluation episodes"],
                 ["Learners", f"DP (value iteration), first-visit MC, Q-learning; {STEPS:,} steps each"],
                 ["Metrics", "true-simulator return, total gain and injury weeks per course; normalised model value v(π) and t90; headroom captured c(π)"],
                 ["Statistics", f"95% CI = 1.96 standard errors over replicates; where a significance statement is made the interval is widened to a t-interval (factor {tf:.2f} for {NSEED} replicates)"]],
             STEPS, NSEED, N_EVAL,
             extra="Because results.json for this study stores summary statistics only, comparisons between two learners use the stored 95% intervals (an unpaired approximation with a t critical value) and comparisons with the rule use the learner's t-widened interval; the MC versus Q-learning comparisons on the normalised scale are the stored paired tests.")

    P.h1("Results")
    P.h2("Dense weekly reward")
    rows = [["Method"] + [f"H = {H}" for H in HS]]
    for m in ("Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP"):
        row = [NICE.get(m, m) if m != "Rule" else "Fatigue-band rule"]
        for H in HS:
            if m in LEARN:
                row.append(pm(J(H, False, m), C(H, False, m)) + mark[VR[(H, False, m)][2]])
            else:
                row.append(f"{J(H, False, m):.2f}")
        rows.append(row)
    t1 = P.table(rows, f"True-simulator return with the dense (weekly) reward ({N_EPS} logged episodes, {NSEED} replicates, 95% CI; rule and oracle are single policies). "
                 "An asterisk marks a learner significantly above the rule and a dagger one significantly below it, using the t-widened interval.",
                 widths=[5.2, 3.7, 3.7, 3.7], font=8.5)
    P.p(f"Table {t1} shows that with a dense reward every learner beats the rule at every horizon: all {n_up_d} of {n_cfg} learner-horizon combinations are significantly above it, by {gain_range_d[0]:.2f} to {gain_range_d[1]:.2f} units. "
        f"The oracle DP, fitted to {N_OR:,} episodes, is far higher still ({J(6, False, 'Oracle DP'):.2f}, {J(12, False, 'Oracle DP'):.2f} and {J(24, False, 'Oracle DP'):.2f} against the rule's {J(6, False, 'Rule'):.2f}, {J(12, False, 'Rule'):.2f} and {J(24, False, 'Rule'):.2f}), so the achievable headroom grows with the horizon ({orc_d[6]:.1f}, {orc_d[12]:.1f} and {orc_d[24]:.1f} units). "
        f"At six weeks DP and Q-learning are close ({J(6, False, 'DP'):.2f} and {J(6, False, 'Q-learning'):.2f}) and MC is lower ({J(6, False, 'Monte Carlo'):.2f} ± {C(6, False, 'Monte Carlo'):.2f}); the DP-minus-MC difference of {cmp(6, False, 'DP', 'Monte Carlo')[0]:.2f} has an interval half-width of {cmp(6, False, 'DP', 'Monte Carlo')[1]:.2f} (unpaired approximation). "
        f"At twelve weeks the three are indistinguishable (DP {J(12, False, 'DP'):.2f}, MC {J(12, False, 'Monte Carlo'):.2f}, Q-learning {J(12, False, 'Q-learning'):.2f}). At 24 weeks the order has reversed: MC scores {J(24, False, 'Monte Carlo'):.2f} against {J(24, False, 'DP'):.2f} for DP and {J(24, False, 'Q-learning'):.2f} for Q-learning, "
        f"{cmp(24, False, 'Monte Carlo', 'DP')[0]:+.2f} ± {cmp(24, False, 'Monte Carlo', 'DP')[1]:.2f} relative to DP and {cmp(24, False, 'Monte Carlo', 'Q-learning')[0]:+.2f} ± {cmp(24, False, 'Monte Carlo', 'Q-learning')[1]:.2f} relative to Q-learning. "
        "Model-free MC is therefore the weakest learner at the shortest horizon and the strongest at the longest. A plausible reading is that the model-based solvers (DP and Q-learning on samples from the fitted model) inherit model error that compounds over more weeks, while MC with a dense reward uses the sampled returns directly; the data here do not test that.")
    rows = [["Learner", "H = 6 (dense)", "H = 12 (dense)", "H = 24 (dense)", "H = 24 (sparse)"]]
    for m in LEARN:
        rows.append([NICE.get(m, m)] + [f"{100 * capd[(H, m)]:.0f}%" for H in HS] + [f"{100 * caps24[m]:.0f}%"])
    t2 = P.table(rows, f"Share c(π) of the oracle's improvement over the rule captured by each learner in the true simulator (Eq. 4). For sparse rewards the oracle is below the rule at H = {sp_dn[0]} and {sp_dn[1]}, so c is undefined there. Negative values mean the learner is below the rule.",
                 widths=[4.6, 2.8, 2.8, 2.8, 3.0], font=8.5)
    P.p(f"Table {t2} turns the same numbers into a measure that is comparable across horizons. DP captures {100 * capd[(6, 'DP')]:.0f}%, {100 * capd[(12, 'DP')]:.0f}% and {100 * capd[(24, 'DP')]:.0f}% of the headroom at 6, 12 and 24 weeks and Q-learning {100 * capd[(6, 'Q-learning')]:.0f}%, {100 * capd[(12, 'Q-learning')]:.0f}% and {100 * capd[(24, 'Q-learning')]:.0f}%: both capture a smaller share as the horizon grows, "
        f"even though the average number of logged transitions per state is the same ({per_state:.0f}). MC does not follow this pattern ({100 * capd[(6, 'Monte Carlo')]:.0f}%, {100 * capd[(12, 'Monte Carlo')]:.0f}%, {100 * capd[(24, 'Monte Carlo')]:.0f}%). "
        f"The unexplained part of the headroom is largest at 24 weeks, where the oracle is {orc_d[24]:.1f} units above the rule and the best learner (MC) is {J(24, False, 'Oracle DP') - J(24, False, 'Monte Carlo'):.1f} units short of it. "
        f"The logs support few off-rule pairs (about {pair_off:.1f} visits per pair on average), and the oracle's advantage comes in part from actions the small data cannot support; this explanation is based on the logging arithmetic and not tested by varying the data.")

    P.h2("Sparse terminal reward")
    rows = [["Method"] + [f"H = {H}" for H in HS]]
    for m in ("Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP"):
        row = [NICE.get(m, m) if m != "Rule" else "Fatigue-band rule"]
        for H in HS:
            if m in LEARN:
                row.append(pm(J(H, True, m), C(H, True, m)) + mark[VR[(H, True, m)][2]])
            else:
                row.append(f"{J(H, True, m):.2f}")
        rows.append(row)
    t3 = P.table(rows, f"True-simulator return with the sparse (terminal) reward ({N_EPS} logged episodes, {NSEED} replicates, 95% CI). The asterisk and dagger have the same meaning as in Table {t1}.", widths=[5.2, 3.7, 3.7, 3.7], font=8.5)
    P.p(f"Table {t3} shows a very different picture. No learner is significantly above the rule in any of the {n_cfg} combinations; {n_dn_s} are significantly below it and {n_ns_s} cannot be distinguished from it. "
        f"DP stays close to the rule at six and twelve weeks ({J(6, True, 'DP'):.2f} ± {C(6, True, 'DP'):.2f} and {J(12, True, 'DP'):.2f} ± {C(12, True, 'DP'):.2f} against {J(6, True, 'Rule'):.2f} and {J(12, True, 'Rule'):.2f}) but with a much wider interval than in the dense case, and Q-learning is {VR[(6, True, 'Q-learning')][0]:+.2f} ± {VR[(6, True, 'Q-learning')][1]:.2f} from the rule at six weeks (not distinguishable) and significantly below at twelve ({VR[(12, True, 'Q-learning')][0]:+.2f}) and 24 weeks ({VR[(24, True, 'Q-learning')][0]:+.2f}). "
        f"MC is below the rule at every horizon ({mc_sparse_loss[6]:+.2f}, {mc_sparse_loss[12]:+.2f} and {mc_sparse_loss[24]:+.2f} units) and at 24 weeks is also well below DP ({J(24, True, 'Monte Carlo') - J(24, True, 'DP'):+.1f}). "
        f"At 24 weeks DP ({J(24, True, 'DP'):.2f} ± {C(24, True, 'DP'):.2f}) is {J(24, True, 'DP') - J(24, True, 'Q-learning'):.1f} units above Q-learning ({J(24, True, 'Q-learning'):.2f}), so Q-learning on the fitted model does not converge to the DP solution of that model within {STEPS:,} steps when the reward is terminal and the horizon is long.")
    P.p(f"The oracle DP is the sharpest result. Even with {N_OR:,} episodes its sparse-reward policy is {abs(orc_s[sp_dn[0]]):.2f} below the rule at {sp_dn[0]} weeks ({J(sp_dn[0], True, 'Oracle DP'):.2f} against {J(sp_dn[0], True, 'Rule'):.2f}) and {abs(orc_s[sp_dn[1]]):.2f} below at {sp_dn[1]} weeks ({J(sp_dn[1], True, 'Oracle DP'):.2f} against {J(sp_dn[1], True, 'Rule'):.2f}), and only at {sp_up[0]} weeks is it above ({orc_s[sp_up[0]]:+.2f}). "
        "With ten times the data and an exact solver, the dense formulation reaches far above the rule in every configuration (Table "
        f"{t1}), so the failure under the sparse reward is not only a matter of sample size or of how the learner assigns credit. Something in the sparse formulation misleads the planner itself. "
        "The code suggests the hidden-history term of Eq. (6) as a candidate (the reward depends on the initial score, which the state does not contain); we did not test it, and other explanations, such as the way the terminal reward interacts with the unsupported-pair rule, cannot be excluded.")
    f1 = P.fig(fig(I, "fig_bars_sparse"), f"True-simulator return at H = 12 with the sparse reward (single replicate set; 95% CI over {NSEED} replicates; rule and oracle are single policies).", 7.0)
    P.p(f"Figure {f1} shows the twelve-week case. MC's return is {J(12, True, 'Monte Carlo'):.2f} ± {C(12, True, 'Monte Carlo'):.2f}, negative on a scale where the rule scores {J(12, True, 'Rule'):.2f}. A negative return means the average course ends with a lower score than it started with, "
        f"and the total gain column of the summary confirms it ({INF(12, True, 'Monte Carlo', 'gain'):.2f} points per course against {INF(12, True, 'Rule', 'gain'):.2f} for the rule). The interval of MC is narrow (±{C(12, True, 'Monte Carlo'):.2f}), so this is a stable property of MC's policies under the sparse reward in all {NSEED} replicates and not a single bad run. "
        f"The consistency across horizons ({J(6, True, 'Monte Carlo'):.2f}, {J(12, True, 'Monte Carlo'):.2f}, {J(24, True, 'Monte Carlo'):.2f}) is also in line with a variance explanation: first-visit MC assigns the same terminal number, discounted, to every state-action pair in an episode, and that number contains the initial-score spread (standard deviation {sd_s0:.1f}) that is larger than the effect of any one week; "
        "this is a standard property of Monte Carlo returns [@sb] but we did not isolate it by experiment.")

    P.h2("Horizon, reward type and learning on the fitted model")
    rows = [["Configuration", "Q-learning final v", "MC final v", "MC − Q-learning (paired)", "Reached 90% (Q / MC)"]]
    for sp in (False, True):
        for H in HS:
            q, mc = FIN(H, sp, "Q-learning"), FIN(H, sp, "Monte Carlo"); d = mcq[(H, sp)]
            rows.append([f"H = {H}, {'sparse' if sp else 'dense'}", pm(q["mean"], q["ci"]), pm(mc["mean"], mc["ci"]), f"{d['mean']:+.2f} ± {d['ci']:.2f} ({pstr(d['p'])})",
                         f"{int(round(Rn[(H, sp)]['reach']['Q-learning'] * NSEED))}/{NSEED} / {int(round(Rn[(H, sp)]['reach']['Monte Carlo'] * NSEED))}/{NSEED}"])
    t4 = P.table(rows, f"Final normalised value v(π) on the oracle model after {STEPS:,} steps (0 is the logged behaviour, 1 the model optimum; {NSEED} replicates, 95% CI), the paired MC-minus-Q-learning difference and the number of replicates reaching 90% of the optimum.",
                 widths=[3.3, 3.2, 3.2, 4.5, 2.6], font=8.5)
    P.p(f"Table {t4} reports learning on the fitted model. No run reached the 90% target: {reach_n} of {n_runs} runs did, so t90 is undefined everywhere (the steps-to-90% figure in the code output is empty for that reason). The best final normalised value is {max_final:+.2f} ({max_final_k[2]}, H = {max_final_k[0]}, {'sparse' if max_final_k[1] else 'dense'}). "
        f"Q-learning's final value falls with the horizon under both reward types ({q_fin_d[0]:.2f}, {q_fin_d[1]:.2f}, {q_fin_d[2]:.2f} for dense; {q_fin_s[0]:.2f}, {q_fin_s[1]:.2f}, {q_fin_s[2]:.2f} for sparse), and MC's does not follow a monotone pattern under dense rewards ({mc_fin_d[0]:.2f}, {mc_fin_d[1]:.2f}, {mc_fin_d[2]:.2f}). "
        f"Under sparse rewards {n_neg_fin_s} of {2 * len(HS)} learner-horizon values are negative: the greedy policy is worse than the logged behaviour on the very model it was trained on, and {n_sig_neg_fin} of the {2 * len(HS) * 2} values (both reward types) are significantly below zero. "
        f"These policies are worse than the logged behaviour on the very model they were trained against, which is a failure of the learners on the model scale, although given the oracle result above the model scale itself is not a reliable guide under sparse rewards. MC minus Q-learning is significantly negative in all three sparse configurations (p = {', '.join(f'{mcq[(H, True)]['p']:.3f}' for H in HS)}), "
        f"and under the dense reward it is significantly negative at H = 6 ({mcq[(6, False)]['mean']:+.2f}), not significant at H = 12 ({pstr(mcq[(12, False)]['p'])}) and significantly positive at H = 24 ({mcq[(24, False)]['mean']:+.2f}), the same reversal as in the true-simulator returns.")
    f2 = P.fig(fig(I, "fig_curves_h12"), "Normalised learning curves at H = 12 (mean ± s.d. over 5 replicates). 0 is the logged behaviour and 1 the oracle-model optimum.", 11.0)
    P.p(f"Figure {f2} shows the contrast at twelve weeks. With the dense reward both learners rise to about the same plateau within a few tens of thousands of steps and stay below the optimum (final values {FIN(12, False, 'Q-learning')['mean']:.2f} for Q-learning and {FIN(12, False, 'Monte Carlo')['mean']:.2f} for MC). "
        f"With the sparse reward Q-learning rises more slowly and ends near the behaviour policy ({FIN(12, True, 'Q-learning')['mean']:.2f} ± {FIN(12, True, 'Q-learning')['ci']:.2f}), while MC settles below it ({FIN(12, True, 'Monte Carlo')['mean']:.2f} ± {FIN(12, True, 'Monte Carlo')['ci']:.2f}). Both start far below zero because the first greedy policy, from an all-zero value table, is poor. "
        "The plateaus are reached early and do not move, so more environment steps would not change the picture within the fitted model, which suggests that the limit is what the model and the reward formulation can support and not training time.")
    f3 = P.fig(fig(I, "fig_horizon"), "Final normalised value against the treatment horizon, dense (left) and sparse (right) reward (mean ± 95% CI over 5 replicates).", 11.0)
    P.p(f"Figure {f3} summarises the horizon effect. Under dense rewards the normalised value of Q-learning falls steadily from {q_fin_d[0]:.2f} at six weeks to {q_fin_d[2]:.2f} at 24 weeks, with non-overlapping intervals at the ends, while MC dips at the short horizon and recovers. "
        f"Under sparse rewards the values are lower throughout and fall much faster (Q-learning from {q_fin_s[0]:.2f} to {q_fin_s[2]:.2f}), so the horizon penalty is much larger when the reward is terminal: the 24-week sparse values ({q_fin_s[2]:.2f} for Q-learning and {mc_fin_s[2]:.2f} for MC) are more than one oracle-model improvement unit below the behaviour policy. "
        "The interaction of horizon and sparsity is the main quantitative result of the study: the loss from a longer horizon is modest under a dense reward and severe under a sparse one.")

    P.h2("Injuries and policies")
    rows = [["Method"] + [f"dense H = {H}" for H in HS] + [f"sparse H = {H}" for H in HS]]
    for m in ("Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP"):
        row = [NICE.get(m, m) if m != "Rule" else "Fatigue-band rule"]
        for sp in (False, True):
            for H in HS:
                row.append(f"{inj(H, sp, m):.2f}" if m in ("Rule", "Oracle DP") else f"{inj(H, sp, m):.2f} ± {INFC(H, sp, m, 'injury'):.2f}")
        rows.append(row)
    t5 = P.table(rows, f"Injury weeks per course in the true simulator (a week counts as an injury week when fatigue exceeds 0.85 and the intensity is moderate or high; a course has H weeks; {NSEED} replicates, 95% CI).",
                 widths=[3.6, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], font=7.5)
    P.p(f"Table {t5} shows the safety side. With dense rewards the learned policies have fewer injury weeks than the rule in all {n_inj_dn_d} of {n_cfg} combinations (significantly so using the t-widened interval); for example at twelve weeks the rule has {inj(12, False, 'Rule'):.2f} and DP {inj(12, False, 'DP'):.2f}, and the oracle only {inj(12, False, 'Oracle DP'):.2f}. "
        f"With sparse rewards the picture reverses: {n_inj_up_s} of {n_cfg} combinations have significantly more injury weeks than the rule and {n_inj_dn_s} fewer. MC's policies are the extreme: {inj(6, True, 'Monte Carlo'):.2f}, {inj(12, True, 'Monte Carlo'):.2f} and {inj(24, True, 'Monte Carlo'):.2f} injury weeks per course at 6, 12 and 24 weeks, "
        f"that is {100 * mc_inj_frac[6]:.0f}%, {100 * mc_inj_frac[12]:.0f}% and {100 * mc_inj_frac[24]:.0f}% of all weeks, against {inj(6, True, 'Rule'):.2f}, {inj(12, True, 'Rule'):.2f} and {inj(24, True, 'Rule'):.2f} for the rule. "
        f"Even the sparse-reward oracle has {inj(12, True, 'Oracle DP'):.2f} injury weeks at H = 12 against {inj(12, True, 'Rule'):.2f} for the rule. The injury penalty is a one-off 2.5 points in the week it occurs, which is smaller than the gain that high intensity can provide, so a policy that ignores injury risk is not obviously suboptimal under either reward in the true objective; a plausible reading is that the dense learners reduce injuries as a by-product of maximising the total gain, "
        "because injured weeks lose gain, and that the sparse learners do not because they receive no signal about which week an injury happened (not tested).")
    f4 = P.fig(fig(I, "fig_policies_dense"), "Intensity chosen in each state observed in the data (H = 12, dense reward, replicate 0). States are sorted by score band, fatigue band and week.", 9.0)
    f5 = P.fig(fig(I, "fig_policies_sparse"), "The same for the sparse reward.", 9.0)
    P.p(f"Figures {f4} and {f5} show the policies for the first replicate at twelve weeks. In both, the behaviour policy (top row) follows the rule's stripes by fatigue band. With the dense reward the learned policies depart from it in many cells and the oracle chooses rest in broad blocks of states, which fits its very low injury rate ({inj(12, False, 'Oracle DP'):.2f}). "
        f"With the sparse reward the learners choose high intensity (yellow) in cells where the behaviour policy picks lighter therapy, most visibly for MC, in line with its {inj(12, True, 'Monte Carlo'):.2f} injury weeks per course; the sparse oracle departs from the rule in fewer places and keeps more high-intensity cells than the dense oracle. "
        "These are visual readings of the figures for one replicate; the policies are not stored numerically in the results file.")

    P.h1("Discussion")
    P.p(f"**What the study shows.** (1) With a weekly reward, standard learners improve on a fatigue-band rule at every horizon ({n_up_d} of {n_cfg} combinations significantly above it), but capture a decreasing share of the achievable improvement as the horizon grows (DP {100 * capd[(6, 'DP')]:.0f}% to {100 * capd[(24, 'DP')]:.0f}%). "
        f"(2) With a terminal reward, none is significantly above the rule, {n_dn_s} of {n_cfg} are significantly below it, and MC's policies are severely worse and unsafe. (3) The oracle with ten times the data is also below the rule at {sp_dn[0]} and {sp_dn[1]} weeks under the sparse reward, so the failure is not only a sample-size effect. "
        f"(4) On the fitted model no run reached 90% of the optimum ({reach_n} of {n_runs}), and Q-learning's final value declines with the horizon.")
    P.p(f"**Why sparse rewards fail here.** The dense and sparse returns are the same quantity up to discounting. For the rule the sparse return divided by the total gain is {', '.join(f'{ident[H][0]:.4f}' for H in HS)} at 6, 12 and 24 weeks against {', '.join(f'{ident[H][1]:.4f}' for H in HS)} for gamma^(H−1), so the true objectives agree to floating-point precision (largest difference {ident_err:.0e}). "
        "The difference is therefore informational: a dense reward tells the learner which week contributed the gain, while a terminal reward gives one noisy number per course. Three mechanisms are plausible and not mutually exclusive: variance of the terminal return for MC; slow value propagation for Q-learning over long horizons; and the hidden-history term of Eq. (6) in the model, which would affect DP as well. "
        "The pattern of results is consistent with all three (MC worst, Q-learning intermediate and DP the least affected of the learners, yet the oracle also failing at shorter horizons), but the experiment does not separate them, and the oracle failure is the observation that most needs an explanation.")
    P.p(f"**What it does not show.** It does not show that horizon alone is harmful: the share captured by MC at 24 weeks ({100 * capd[(24, 'Monte Carlo')]:.0f}%) is higher than at six ({100 * capd[(6, 'Monte Carlo')]:.0f}%). It does not isolate why the sparse oracle fails. It does not show that a particular reward is right clinically: the weekly gain is a score change and the injury penalty is invented. "
        "And, because the summaries stored in results.json have no per-replicate values, it relies on interval-based comparisons between learners rather than paired tests (except for the stored MC-minus-Q-learning test).")
    P.p("**Practical use.** A programme designer who can observe interim progress should use it: when a terminal outcome is the only signal, the experiments suggest checking the learned policy against the existing rule before anything else, and preferring a model-based solver to Monte Carlo returns. "
        "A sparse-reward pipeline should also report an oracle-style check (the same method with much more data), because a failure of that check points to the formulation and not to the learner.")
    limitations(P, [
        f"**Few replicates and stored summaries.** {NSEED} replicates (t-interval factor {tf:.2f}); results.json stores means and intervals but not per-replicate values, so paired tests against the rule are not available and the comparisons use stored intervals. Learning curves, policies and agreement are available only as figures.",
        "**One replicate in the policy figures.** The policy maps show the first replicate at H = 12 and cannot be generalised to the averages.",
        "**Hidden-history hypothesis untested.** Eq. (6) is a reading of the code. We did not construct a variant whose state contains the initial score or the progress so far, which would be the direct test.",
        "**Rule is a single policy.** The rule's return has no replicate variation, so differences from the rule are judged by the learner's interval only.",
        "**Invented dynamics and penalties.** The gain, fatigue and injury equations, and the 2.5-point injury penalty, are chosen by hand; different values could change which reward type is easier.",
    ])
    ethics_future(P, "(1) Add the initial score or the accumulated improvement to the state and test whether the sparse-reward oracle then beats the rule; (2) add a potential-based shaped reward with a partial signal and measure how much of the gap closes; "
                  "(3) vary the amount of logged data at each horizon to separate coverage from compounding error; (4) store per-replicate returns so that paired tests against the rule are possible; "
                  "(5) apply the adapter to a real rehabilitation table, where interim functional scores would define the dense reward.")
    P.h1("Conclusion")
    P.p(f"In a simulated stroke-rehabilitation programme, DP, MC and Q-learning trained on {N_EPS} logged courses were significantly above a fatigue-band rule at 6, 12 and 24 weeks when rewarded weekly (by {gain_range_d[0]:.1f} to {gain_range_d[1]:.1f} units), but captured a share of the oracle's headroom that fell with the horizon for DP and Q-learning ({100 * capd[(6, 'DP')]:.0f}% to {100 * capd[(24, 'DP')]:.0f}% for DP). "
        f"With a terminal reward none was significantly above the rule, MC was far below it with up to {inj(24, True, 'Monte Carlo'):.1f} injury weeks per course, and the oracle DP was below the rule at {sp_dn[0]} and {sp_dn[1]} weeks, even though the two rewards define almost the same true objective. "
        "These are simulator results, not clinical evidence, and the reasons for the sparse-reward failure remain hypotheses.")
    defense(P, [
        ("If dense and sparse rewards are the same objective, why do the results differ so much?",
         f"They are the same in the true process (the rule's sparse return over its total gain is {ident[12][0]:.4f} against gamma^11 = {ident[12][1]:.4f} at H = 12), but not in what can be estimated: a terminal reward provides one noisy number per course, and the model's last-week reward includes a hidden-history term (Eq. 6). The results show the difference but do not isolate which mechanism produces it."),
        ("Why is the oracle DP below the rule under the sparse reward?",
         f"It scores {J(6, True, 'Oracle DP'):.2f} and {J(12, True, 'Oracle DP'):.2f} against {J(6, True, 'Rule'):.2f} and {J(12, True, 'Rule'):.2f} at 6 and 12 weeks, with {inj(12, True, 'Oracle DP'):.2f} injury weeks at 12. With ten times the data and an exact solver, a sample-size explanation does not apply, which points to the formulation. We offered the hidden-history term as a candidate but did not test it."),
        ("Does a longer horizon hurt learning?",
         f"For DP and Q-learning under a dense reward the share of headroom captured falls ({100 * capd[(6, 'DP')]:.0f}%, {100 * capd[(12, 'DP')]:.0f}%, {100 * capd[(24, 'DP')]:.0f}% for DP) and Q-learning's normalised value falls ({q_fin_d[0]:.2f} to {q_fin_d[2]:.2f}); for MC it does not. Under a sparse reward the decline is much steeper ({q_fin_s[0]:.2f} to {q_fin_s[2]:.2f} for Q-learning)."),
        ("Is MC just a worse algorithm?",
         f"Not uniformly. It is the weakest learner at six weeks under a dense reward ({J(6, False, 'Monte Carlo'):.2f} against {J(6, False, 'DP'):.2f} for DP) and the strongest at 24 weeks ({J(24, False, 'Monte Carlo'):.2f} against {J(24, False, 'DP'):.2f}), and it is far worse than the others under a sparse reward."),
        ("Can these results guide a real rehabilitation programme?",
         "No. The dynamics, the injury rule and the penalty are invented; the study asks how reward design and horizon affect learning algorithms, and finds that a terminal-only reward is a hazard even when the true objective is unchanged."),
        ("Why were no run able to reach 90% of the optimal improvement?",
         f"Under the dense reward the best final normalised value is {max(FIN(H, False, a)['mean'] for H in HS for a in ALG):.2f}; the fitted model has few supported off-rule pairs (about {pair_off:.1f} visits per pair with {N_EPS} episodes), so the learned policy plausibly cannot adopt many actions the optimum uses (not tested). The plateaus in Figure {f2} are reached early, so more steps would not change this."),
    ])
    repro(P, 9, ["fig_agreement_dense", "fig_agreement_sparse", "fig_bars_dense", "fig_bars_sparse", "fig_curves_h12", "fig_horizon", "fig_learning_dense", "fig_learning_sparse", "fig_policies_dense", "fig_policies_sparse", "fig_steps90"])
    code_appendix(P, 9, [domains.Rehab])
    rl_core_appendix(P)
    P.refs()
    path = f"{PAP}/Paper09_Stroke_Rehab_Horizon_Reward.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
