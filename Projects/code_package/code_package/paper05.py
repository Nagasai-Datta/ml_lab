import numpy as np
from scipy import stats
from docx_helper import Paper
from paper_common import *

I = 5


def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def build():
    r = load(I)
    S, N, F, GS = r["smdp"], r["naive"], r["fixed"], r["gamma_sweep"]
    fs, fn = r["smdp_full"], r["naive_full"]
    pts = r["points"]
    GK = list(GS.keys())                      # "0.6", "0.75", "0.9", "0.97"
    rule = S["Rule"]["J"]
    nrep = fs["config"]["seeds"]
    tfac = stats.t.ppf(0.975, nrep - 1) / 1.96
    dpd, qd, mcd = r["smdp_vs_naive_dp"], r["smdp_vs_naive_q"], r["smdp_vs_naive_mc"]
    ntests = 3 + len(GK)
    bonf = 0.05 / ntests

    # ---- learned policy versus the rule (one-sample test of per-replicate return against the single rule value)
    def vs_rule(full, m):
        j = np.asarray(full["methods"][m]["J"], float)
        return dict(mean=float(j.mean() - rule), ci=_ci(j), p=float(stats.ttest_1samp(j, rule).pvalue))
    VR = {(tag, m): vs_rule(full, m) for tag, full in (("S", fs), ("N", fn)) for m in ("DP", "Monte Carlo", "Q-learning")}

    # ---- model-predicted versus true advantage of the oracle DP policy over the rule
    def diag(full):
        mo = full["J_oracle_model"]
        return dict(pred=mo["Oracle DP"] - mo["Rule"], true=full["methods"]["Oracle DP"]["J"][0] - full["methods"]["Rule"]["J"][0],
                    mo=mo["Oracle DP"], mr=mo["Rule"], to=full["methods"]["Oracle DP"]["J"][0], tr=full["methods"]["Rule"]["J"][0])
    DS, DN = diag(fs), diag(fn)
    og_s, og_n = np.asarray(fs["optimism_gap"]), np.asarray(fn["optimism_gap"])
    og_p = float(stats.ttest_rel(og_n, og_s).pvalue)

    # ---- learning curves on the oracle MDP
    def curve_stats(full, m):
        c = np.array(full["curves"][m])
        return dict(c0=c[:, 0, 1].mean(), c6=c[:, 1, 1].mean(), c30=c[:, 5, 1].mean(), cend=c[:, -1, 1].mean(),
                    k6=c[0, 1, 0], kend=c[0, -1, 0])
    CS = {m: curve_stats(fs, m) for m in ("Monte Carlo", "Q-learning")}
    CN = {m: curve_stats(fn, m) for m in ("Monte Carlo", "Q-learning")}

    # ---- policies
    rule_pol = np.array([{0: 2, 1: 1, 2: 0}[s // 36] for s in range(108)])
    pol_s, pol_n = np.array(r["oracle_policy_smdp"]), np.array(r["oracle_policy_naive"])
    n_diff_oracle = int((pol_s != pol_n).sum())
    agree_rule_s, agree_rule_n = float((pol_s == rule_pol).mean()), float((pol_n == rule_pol).mean())
    seed0 = {m: float((np.array(fs["policies"][m]) == rule_pol).mean()) for m in ("DP", "Monte Carlo", "Q-learning")}
    n_dp_diff = int((np.array(fs["policies"]["DP"]) != np.array(fn["policies"]["DP"])).sum())

    def cells(pol):
        M = np.zeros((3, 6), int)
        for t in range(3):
            for ab in range(6):
                M[t, ab] = pol[((t * 6 + ab) * 2 + 0) * 3 + 1] + 1
        return M
    Cs, Cn = cells(pol_s), cells(pol_n)
    n_cell_diff = int((Cs != Cn).sum())
    rule_cells = np.array([[3] * 6, [2] * 6, [1] * 6])
    n_cell_rule = int((Cs == rule_cells).sum())
    lowrow = "/".join(f"{x}y" for x in Cs[0])

    agS = {m: np.mean(fs["agree"][m]) for m in ("Monte Carlo", "Q-learning")}
    agN = {m: np.mean(fn["agree"][m]) for m in ("Monte Carlo", "Q-learning")}
    g_lo, g_hi = float(GK[0]), float(GK[-1])
    ddif = {k: 100 * GS[k]["differ"] for k in GK}
    sig = [k for k in GK if GS[k]["diff"]["p"] < 0.05]
    neg = [k for k in GK if GS[k]["diff"]["mean"] < 0]
    Pk = {m: pts[f"{m} (SMDP)"] for m in ("DP", "Monte Carlo", "Q-learning")}
    Pn = {m: pts[f"{m} (naive MDP)"] for m in ("DP", "Monte Carlo", "Q-learning")}
    g9 = GS["0.9"]

    P = Paper("Semi-Markov-Aware Cancer Screening Schedules: Does Modelling the Length of the Screening Interval in the Discount Help Dynamic Programming, Monte Carlo and Q-learning?",
              "cancer screening; semi-Markov decision process; dynamic programming; Monte Carlo control; Q-learning; discounting; offline reinforcement learning",
              running="SMDP-aware screening schedules", refdict=REFS)

    P.abstract(
        "Cancer screening decisions are made at irregular intervals: a patient screened now is next screened in one, two or three years, so the time between decisions "
        "depends on the action. A tabular Markov decision process (MDP) that treats every decision as one time step discounts all intervals alike; a semi-Markov (SMDP) model "
        "discounts each transition by gamma raised to the interval length. We ask whether this correction matters when schedules are learned from logs. On a simulator "
        "mirroring the variables of Breast Cancer Wisconsin and Cervical Cancer Risk tables (the Kaggle site was unreachable, HTTP 403, so no real file was used) we compare "
        f"SMDP-aware and naive versions of dynamic programming (DP), first-visit Monte Carlo control and Q-learning, {nrep} replicates each, plus a sweep of the yearly discount factor from {g_lo:g} to {g_hi:g}. "
        f"The correction did not help DP ({dpd['mean']:+.2f} ± {dpd['ci']:.2f}, {pstr(dpd['p'])}) or Q-learning ({qd['mean']:+.2f} ± {qd['ci']:.2f}, {pstr(qd['p'])}); it improved Monte Carlo "
        f"by {mcd['mean']:+.2f} ± {mcd['ci']:.2f} ({pstr(mcd['p'])}), a result that does not survive a Bonferroni correction over the {ntests} comparisons we ran. In the sweep the share of states in which "
        f"the two DP policies disagree falls from {ddif['0.6']:.1f}% at gamma = {g_lo:g} to {ddif['0.97']:.1f}% at {g_hi:g}, but the return difference is not monotone and its sign is negative "
        f"(SMDP-aware worse) at gamma = 0.75 ({GS['0.75']['diff']['mean']:+.2f}, {pstr(GS['0.75']['diff']['p'])}) and 0.9 ({g9['diff']['mean']:+.2f}, {pstr(g9['diff']['p'])}). "
        f"No learned schedule beat a risk-stratified rule ({rule:.2f}), and the oracle DP, trained on ten times more data, scored {S['Oracle DP']['J']:.2f}. "
        "All results are simulated and carry no clinical meaning.")

    # ------------------------------------------------------------------ 1
    P.h1("Introduction")
    P.p("Screening programmes for breast and cervical cancer ask a question that looks simple and is not: given what is known about a woman today, when should she next be "
        "screened? Annual screening finds more disease but costs more and causes more false alarms; three-yearly screening is cheaper but lets more lesions progress between visits. "
        "The decision is sequential, and unusual in that the action is itself a length of time: choosing a three-year interval means the next decision point is three years away. "
        "Dynamic screening policies have been studied with Markov models of disease progression [@maillart,ayer], and Markov decision processes in medicine are well established [@schaefer,puterman].")
    P.p("Dynamic programming (DP), Monte Carlo (MC) control and Q-learning [@sb,bellman,watkins] assume a clock in which every decision takes one step. When decisions take different "
        "amounts of time, the correct generalisation is the semi-Markov decision process (SMDP), in which the discount of a transition depends on its duration [@puterman,bradtke,sutton99]. "
        "An analyst with a table of screening visits can easily skip this step: build a count-based MDP from (state, action, reward, next state) rows, ignore the gaps between visits, and run value iteration. "
        "The resulting policy is optimal for a model in which a three-year wait is discounted as if it were one year. This paper measures what that shortcut costs, and whether the correct SMDP treatment recovers it.")
    P.p("**Contributions.** (i) A 108-state, 3-action semi-Markov screening simulator with a latent disease stage, in which the interval chosen at each decision (1, 2 or 3 years) is also the "
        f"time that elapses. (ii) A paired comparison of SMDP-aware and naive (duration-ignoring) planning for DP, MC control and Q-learning, {nrep} replicates, scored exactly in the simulator with common random numbers. "
        f"(iii) A sweep of the yearly discount factor from {g_lo:g} to {g_hi:g}, in which both the disagreement between the two policies and the return difference are measured. "
        "(iv) A diagnostic that compares model-predicted with true policy advantage over a risk-stratified rule. (v) Candid negative and null results: the SMDP correction is not "
        "reliably beneficial in this simulator, it is detectably harmful at two of four discount factors, and no learned schedule beats the rule.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached from the build environment (all requests returned HTTP 403), so no real dataset was analysed. The simulator mirrors the "
        "variables of Breast Cancer Wisconsin and Cervical Cancer Risk tables (risk tier, age, prior result) with hand-specified dynamics. Every number below is a simulated result about "
        "algorithms, and no clinical claim is made.")

    # ------------------------------------------------------------------ 2
    P.h1("Related work")
    P.p("**Screening as a sequential decision problem.** Maillart and colleagues assessed dynamic breast cancer screening policies that depend on age and screening history [@maillart]; "
        "Ayer, Alagoz and Stout formulated mammography decisions as a partially observable MDP with a hidden disease state [@ayer]. Our simulator shares their structural features (latent stage, imperfect "
        "sensitivity, false positives, competing mortality) but is far simpler and calibrated to nothing: it is meant to expose algorithmic behaviour, not to estimate optimal intervals.")
    P.p("**Semi-Markov decision processes and RL.** Puterman treats SMDPs as a standard extension in which sojourn times enter the reward and the discounting [@puterman]; Bradtke and Duff derived Q-learning "
        "for continuous-time problems [@bradtke], and the options framework uses the same structure for temporally extended actions [@sutton99]. We use the discrete-time analogue, discount gamma^tau for an interval of tau years, "
        "and test it against the shortcut that ignores tau.")
    P.p("**Learning from logs.** Plug-in planning on logged data can exploit estimation error; remedies include restricting policies to the support of the data or penalising uncertain values [@levine,fujimoto,kumar,jin], "
        "and reviews of RL in healthcare warn against assuming improvements over clinician behaviour [@yu,gottesman]. We use a count threshold, and the simulator gives exact evaluation.")
    background(P)

    # ------------------------------------------------------------------ 4
    P.h1("Problem formulation: a semi-Markov screening simulator")
    P.p("Each simulated patient starts at an age drawn uniformly from 45 to 65 years, with a risk tier of low, middle or high (probabilities 0.55, 0.30, 0.15). The latent disease stage "
        "is 0 (none), 1 (early lesion), 2 (early cancer) or 3 (advanced cancer); a patient begins in stage 0 with probability 0.95, 0.88 or 0.80 for the three tiers, and otherwise in stage 1 (80%) or stage 2 (20%). "
        "At each decision the patient is screened and the clinician chooses the interval until the next screen: 1, 2 or 3 years. The screen is positive with probability 0.60, 0.85 or 0.95 if the stage is 1, 2 or 3, "
        "and with probability 0.03 (a false positive) in stage 0. The reward of a decision is -2 for the screen itself and then depends on the outcome: -3 for a false-positive work-up; +8 for a positive "
        "stage-1 screen (treatment resets the stage to 0); +100 for a positive stage-2 screen, which ends the episode and is what we call an early detection; and +10 for a positive stage-3 screen, a late detection. "
        "During the chosen interval the stage may progress, once per year, with tier-dependent probabilities; a patient in stage 3 presents with symptoms with probability 0.6 per year, "
        "which costs -80 and ends the episode (also counted as late). Other-cause mortality is 0.01 + 0.002 (age - 50) per year after age 50, episodes end at age 75, and at most 12 decisions are taken.")
    P.table([["Element", "Definition"],
             ["State (108)", "risk tier (3) × age band {<50, 50-54, 55-59, 60-64, 65-69, 70+} (6) × previous screen positive (2) × previous interval {1, 2, 3 years} (3)"],
             ["Latent (unobserved)", "disease stage 0 to 3; progression per year 0.03·m (stage 0 to 1), 0.15·k (1 to 2), 0.40 (2 to 3), with tier multipliers m = 1, 2.5, 6 (scaled by age) and k = 1, 1.5, 2.5"],
             ["Actions (3)", "screen again in 1, 2 or 3 years; the interval length tau = 1, 2 or 3 years is the transition duration"],
             ["Reward", "-2 per screen; -3 false positive; +8 stage-1 detection; +100 stage-2 detection (early, ends episode); +10 stage-3 detection; -80 symptomatic presentation"],
             ["Discount", "gamma = 0.90 per year, applied as gamma^tau to a decision of tau years (SMDP-aware) or as gamma to every decision (naive)"],
             ["Horizon", "at most 12 decisions; episode ends at age 75, at death, or at an early or late detection"],
             ["Behaviour policy / rule", "risk-stratified rule: low tier every 3 years, middle every 2, high every 1; behaviour = rule with epsilon = 0.25 random actions"],
             ["Dataset mirrored", "Breast Cancer Wisconsin; Cervical Cancer Risk Factors (risk tier, age, previous result)"]],
            "The semi-Markov screening MDP.", widths=[3.4, 13.0], font=8.5)
    P.p("Two properties matter for interpreting the results. First, the true stage is hidden and the state records only tier, age band and the previous screen's outcome and interval, so the "
        "observed process is not exactly Markov. Second, the reward is a stylised utility: an early detection earns +100 and ends the episode, whereas catching a lesion at stage 1 earns only +8. Policies that screen more often "
        "catch more lesions at stage 1 and therefore record fewer early detections (stage-2 detections) in the tables below. The values are arbitrary and were fixed before the experiments.")

    # ------------------------------------------------------------------ 5 method
    P.h1("Method: SMDP-aware planning and its naive counterpart")
    P.p("Let a decision in state s with action a (interval tau(a) years) yield reward R(s, a), transition to s', and consume a random number of years tau. The SMDP Bellman equation with yearly discount gamma is")
    P.eq("V(s) = max_a [ R(s,a) + Σ_{s'} P(s'|s,a) Γ(s,a,s') V(s') ],      Γ(s,a,s') = E[ γ^τ | s, a, s' ].", "3")
    P.p("From logs, Γ is estimated by the mean of gamma^tau_i over the observed transitions (s, a, s'). In this simulator tau is determined by the action, so Γ(s,a,s') = gamma^{tau(a)} exactly; "
        "the estimate reproduces that value, and nothing in the learning code needs to be told the structure. The **naive MDP** replaces tau by 1 in every logged transition, so Γ = gamma, and Eq. (3) reduces to the "
        "ordinary Bellman equation (1). Planning, MC control and Q-learning are otherwise identical in the two versions: the learners sample transitions from the estimated model and use the estimated "
        "Γ in place of gamma, so the SMDP form of the Q-learning update is")
    P.eq("Q(s,a) ← Q(s,a) + α [ r + γ^τ max_{a'} Q(s',a') − Q(s,a) ]", "4")
    P.p("and the MC return accumulates the product of the interval discounts along the trajectory. For an interval of d years the naive continuation weight is gamma instead of gamma^d, an overstatement by")
    P.eq("ρ(d) = γ / γ^d = γ^{−(d−1)},", "5")
    P.p("which equals one for annual screening and grows with the interval length and the severity of the discount. The naive planner is therefore biased toward long intervals in the sense that it under-discounts "
        "what happens after them. Whether that changes the policy depends on how large continuation values are relative to immediate reward differences, and whether it hurts the true return depends on the model's other errors, so we measure it.")
    P.p(f"For a three-year interval the naive model weights the continuation {', '.join(f'{float(k) ** -2:.2f} times too heavily at gamma = {float(k):g}' for k in GK[:2])}, {float(GK[2]) ** -2:.2f} at {float(GK[2]):g} and {float(GK[3]) ** -2:.2f} at {g_hi:g}. "
        "This framed two hypotheses for the sweep: (H1) the share of states in which the two planners disagree falls as gamma rises; (H2) the return advantage of the SMDP-aware planner, if any, is largest at low gamma. "
        "H1 is supported and H2 is not.")
    P.h2("Algorithm")
    P.code("""Algorithm 1  SMDP-aware vs naive planning from screening logs
 input : logged episodes (state, interval a, reward r, next state, done), yearly discount gamma, support threshold n_min = 10
 1. tau_i = 1 + a_i  (years until the next screen) for every logged decision;  naive variant: tau_i = 1
 2. counts N(s,a,s') and mean reward R(s,a); mean continuation weight G(s,a,s') = mean_i gamma ** tau_i
 3. (s,a) with fewer than n_min visits are unsupported: absorb with the worst reward in the data; add one pseudo-absorption to supported pairs
 4. DP:    V <- max_a [ R + sum_s' P * G * V ]   until convergence;  pi_DP = argmax   (policy iteration is used as a check)
 5. Q / MC: sample (s', r, G) from the estimated model; update Q with r + G * max Q(s') (Eq. 4), or average first-visit returns
 6. in states never observed in the logs, fall back to the modal logged interval
 7. output pi; score in the TRUE simulator (discounted by gamma ** tau) with common random numbers""")
    protocol(P, [["Item", "Setting"],
                 ["Simulator", "semi-Markov screening; latent stage; intervals of 1, 2, 3 years; gamma = 0.90 for the main benchmark"],
                 ["States / actions", "108 / 3"],
                 ["Logged data", f"{fs['config']['n_eps']:,} episodes per replicate from the epsilon-noisy risk-stratified rule"],
                 ["Oracle model", f"{fs['config']['n_oracle']:,} episodes (ten times more data) for exact learning-curve scoring and the oracle DP"],
                 ["Replicates", f"{nrep} independent data draws and learner seeds per condition; SMDP-aware and naive versions use the same logged episodes"],
                 ["Learners", f"DP (value iteration, checked against policy iteration), first-visit MC, Q-learning; {fs['config']['steps']:,} steps each"],
                 ["Scoring", f"{fs['config']['n_eval']:,} episodes per policy in the true simulator (common random numbers)"],
                 ["Fixed schedules", "annual, biennial and triennial screening for all patients, same evaluation"],
                 ["Discount sweep", f"gamma in {{{', '.join(GK)}}}; DP only; {nrep} replicates; 10,000 evaluation episodes; different data seeds from the main benchmark"]],
             fs["config"]["steps"], nrep, fs["config"]["n_eval"],
             extra=f"**Statistical conventions.** Confidence half-widths are 1.96 standard errors over the {nrep} replicates (a t-based interval would be {tfac:.2f} times wider; this paper uses the normal form, not the 't-style' intervals of the generic description above). The oracle DP and the rule are single deterministic evaluations without a replicate-based interval; "
                   f"intervals for fixed schedules are over evaluation episodes. We ran {ntests} paired SMDP-versus-naive comparisons (three solvers and {len(GK)} discount factors), so a Bonferroni threshold is {bonf:.4f}.")

    # ------------------------------------------------------------------ results
    P.h1("Results")
    P.h2("Headline comparison: SMDP-aware versus naive planning")
    rows = [["Method", "Return, SMDP-aware", "Return, naive", "Paired difference (SMDP − naive)", "p", "Screens per episode, SMDP / naive"]]
    for m in ("DP", "Monte Carlo", "Q-learning"):
        d = {"DP": dpd, "Monte Carlo": mcd, "Q-learning": qd}[m]
        rows.append([NICE.get(m, m), pm(S[m]["J"], S[m]["ci"]), pm(N[m]["J"], N[m]["ci"]), pm(d["mean"], d["ci"]), f"{d['p']:.3f}",
                     f"{S[m]['info']['screens']:.2f} / {N[m]['info']['screens']:.2f}"])
    rows.append(["Risk-stratified rule", f"{S['Rule']['J']:.2f}", f"{N['Rule']['J']:.2f}", "-", "-", f"{S['Rule']['info']['screens']:.2f}"])
    rows.append([NICE["Oracle DP"], f"{S['Oracle DP']['J']:.2f}", f"{N['Oracle DP']['J']:.2f}", f"{S['Oracle DP']['J'] - N['Oracle DP']['J']:+.2f}", "-",
                 f"{S['Oracle DP']['info']['screens']:.2f} / {N['Oracle DP']['info']['screens']:.2f}"])
    t1 = P.table(rows, f"True-simulator discounted return (gamma = 0.90; mean ± 95% CI over {nrep} replicates) for SMDP-aware and naive versions of each learner; paired differences use the same logged episodes.",
                 widths=[4.0, 2.7, 2.7, 3.0, 1.2, 3.4], font=8.5)
    P.p(f"Table {t1} is the central comparison. The SMDP-aware version of DP scores {S['DP']['J']:.2f} ± {S['DP']['ci']:.2f} against {N['DP']['J']:.2f} ± {N['DP']['ci']:.2f} for the naive version; "
        f"the paired difference is {dpd['mean']:+.2f} ± {dpd['ci']:.2f} ({pstr(dpd['p'])}), a null result with a point estimate in the unfavourable direction. For Q-learning the difference is {qd['mean']:+.2f} ± {qd['ci']:.2f} "
        f"({pstr(qd['p'])}), again indistinguishable from zero. Only Monte Carlo shows a difference in favour of the SMDP-aware model: {mcd['mean']:+.2f} ± {mcd['ci']:.2f} ({pstr(mcd['p'])}). "
        f"That is nominally significant, but it is one of {ntests} comparisons we report, so it does not survive a Bonferroni threshold of {bonf:.4f}. The honest summary is that the SMDP correction has no reliable effect on DP or Q-learning here, "
        f"and a suggestive effect on one solver.")
    f_ret = P.fig(fig(I, "fig_returns"), "True-simulator return (mean, error bar: 95% interval) for the risk-stratified rule, the SMDP-aware and naive learners, and the three fixed schedules. "
                  "Intervals for learned methods are over replicates; those for fixed schedules are over evaluation episodes.", 8.0)
    LJ = [v["J"] for k, v in pts.items() if k.startswith(("DP", "Monte", "Q-"))]
    LJmin, LJmax = min(LJ), max(LJ)
    P.p(f"Figure {f_ret} adds the fixed schedules. Annual screening is by far the worst policy ({F['Annual']['J']:.2f} ± {F['Annual']['ci']:.2f}); "
        f"biennial screening ({F['Biennial']['J']:.2f} ± {F['Biennial']['ci']:.2f}) matches the rule ({rule:.2f}) within the episode-level noise, and triennial screening "
        f"({F['Triennial']['J']:.2f} ± {F['Triennial']['ci']:.2f}) is somewhat lower. Every learned policy scores between {LJmin:.2f} and {LJmax:.2f}, below biennial screening and around the triennial result. "
        "Their replicate intervals overlap, and a trivially simple fixed schedule is at least as good as anything learned.")

    P.h2("Screening burden and detection")
    rows = [["Policy", "Screens", "Early detections (stage 2)", "Late events", "Return"]]
    order = [("Annual", "Annual (all patients)"), ("Biennial", "Biennial (all patients)"), ("Triennial", "Triennial (all patients)"), ("Rule (SMDP)", "Risk-stratified rule"),
             ("DP (SMDP)", "DP, SMDP-aware"), ("Monte Carlo (SMDP)", "Monte Carlo, SMDP-aware"), ("Q-learning (SMDP)", "Q-learning, SMDP-aware"),
             ("DP (naive MDP)", "DP, naive"), ("Monte Carlo (naive MDP)", "Monte Carlo, naive"), ("Q-learning (naive MDP)", "Q-learning, naive")]
    for k, nm in order:
        v = pts[k]
        rows.append([nm, f"{v['screens']:.2f}", f"{v['early']:.3f}", f"{v['late']:.3f}", f"{v['J']:.2f}"])
    t2 = P.table(rows, "Per-episode outcomes in the true simulator. Early detections are positive stage-2 screens; late events are stage-3 detections plus symptomatic presentations. "
                 "Fixed-schedule returns have episode-level intervals in Figure 1.", widths=[4.6, 2.2, 3.2, 2.4, 2.0], font=8.5)
    P.p(f"Table {t2} shows where the policies sit on the screening-burden axis. Annual screening performs {F['Annual']['screens']:.2f} screens per episode, "
        f"triennial {F['Triennial']['screens']:.2f} and biennial {F['Biennial']['screens']:.2f}; the rule performs {pts['Rule (SMDP)']['screens']:.2f} and the learned policies "
        f"{min(v['screens'] for k, v in pts.items() if k.startswith(('DP', 'Monte', 'Q-')) ):.2f} to {max(v['screens'] for k, v in pts.items() if k.startswith(('DP', 'Monte', 'Q-')) ):.2f}, with early detections nearly flat "
        f"({min(v['early'] for k, v in pts.items() if k.startswith(('DP', 'Monte', 'Q-'))):.3f} to {max(v['early'] for k, v in pts.items() if k.startswith(('DP', 'Monte', 'Q-'))):.3f}) against {F['Biennial']['early']:.3f} for biennial and {F['Triennial']['early']:.3f} for triennial. "
        f"The low early-detection count for annual screening ({F['Annual']['early']:.3f}) reflects the stage definitions, not an advantage: annual screening catches lesions at stage 1, where treatment resets the stage and earns +8, "
        f"so late events fall to {F['Annual']['late']:.3f} from {pts['Rule (SMDP)']['late']:.3f} for the rule, at a cost of {F['Annual']['screens'] - pts['Rule (SMDP)']['screens']:.1f} more screens per episode (-2 each before discounting), consistent with its negative return. "
        f"Triennial screening has the highest late-event rate ({F['Triennial']['late']:.3f}, about {F['Triennial']['late'] / pts['Rule (SMDP)']['late']:.1f} times the rule's). "
        "The learned policies differ little from the rule on either axis.")

    P.h2("Does any learned schedule beat the rule?")
    P.p(f"No learned policy exceeds the rule (return {rule:.2f}; one-sample t-tests of the per-replicate returns over {nrep} replicates). In the SMDP-aware benchmark DP is {abs(VR[('S', 'DP')]['mean']):.2f} below the rule ({pstr(VR[('S', 'DP')]['p'])}), Q-learning "
        f"{abs(VR[('S', 'Q-learning')]['mean']):.2f} below ({pstr(VR[('S', 'Q-learning')]['p'])}) and Monte Carlo {abs(VR[('S', 'Monte Carlo')]['mean']):.2f} below ({pstr(VR[('S', 'Monte Carlo')]['p'])}). "
        f"In the naive benchmark the corresponding gaps are {abs(VR[('N', 'DP')]['mean']):.2f} ({pstr(VR[('N', 'DP')]['p'])}), {abs(VR[('N', 'Q-learning')]['mean']):.2f} ({pstr(VR[('N', 'Q-learning')]['p'])}) and "
        f"{abs(VR[('N', 'Monte Carlo')]['mean']):.2f} ({pstr(VR[('N', 'Monte Carlo')]['p'])}). The tests treat the rule's value, a single deterministic evaluation, as a constant. The logged behaviour policy's modal action scores exactly the same as the rule in every replicate, so it is omitted from the tables. "
        "The learned policies are at best on a par with the rule that generated their data and mostly worse, a pattern familiar from offline reinforcement learning [@levine,gottesman].")

    P.h2("Why the oracle is worse than the rule: a model-versus-truth diagnostic")
    rows = [["Quantity", "SMDP-aware model", "Naive model"],
            ["Model value of the oracle DP policy (oracle MDP)", f"{DS['mo']:.2f}", f"{DN['mo']:.2f}"],
            ["Model value of the rule (oracle MDP)", f"{DS['mr']:.2f}", f"{DN['mr']:.2f}"],
            ["Model-predicted advantage of oracle DP over the rule", f"{DS['pred']:+.2f}", f"{DN['pred']:+.2f}"],
            ["True return of the oracle DP policy", f"{DS['to']:.2f}", f"{DN['to']:.2f}"],
            ["True return of the rule", f"{DS['tr']:.2f}", f"{DN['tr']:.2f}"],
            ["True advantage of oracle DP over the rule", f"{DS['true']:+.2f}", f"{DN['true']:+.2f}"],
            ["Sign of predicted and true advantage agree?", "yes" if DS["pred"] * DS["true"] > 0 else "no", "yes" if DN["pred"] * DN["true"] > 0 else "no"],
            [f"Optimism of DP trained on {fs['config']['n_eps']:,} episodes (estimated minus oracle-model value; mean ± 95% CI)", pm(og_s.mean(), _ci(og_s)), pm(og_n.mean(), _ci(og_n))]]
    t4 = P.table(rows, "Model-predicted versus true policy advantage. Absolute model values are not comparable with simulator values (the model treats the fixed horizon and the age limit as a constant termination hazard), "
                 "but the sign and size of the advantage are informative.", widths=[9.2, 3.6, 3.6], font=8.5)
    P.p(f"Table {t4} is the most informative result of the paper, and it is negative for both modelling choices. Even the oracle model, estimated from {fs['config']['n_oracle']:,} episodes, predicts that its optimal policy beats the rule by "
        f"{DS['pred']:+.2f} (SMDP-aware) or {DN['pred']:+.2f} (naive); in the simulator the same policies are worse than the rule by {abs(DS['true']):.2f} and {abs(DN['true']):.2f}. The sign is wrong in both models, and more data does not fix it: "
        "the oracle is trained on ten times more data than the learners and does worse than they do. The gap is therefore a model gap, not a sampling gap, and the SMDP correction does not remove it: "
        f"the two oracle policies are very close in true return ({DS['to']:.2f} versus {DN['to']:.2f}). A plausible mechanism, consistent with but not proved by these data, is policy-induced distribution shift from the omitted hidden stage: the model learns what a state leads to when the logs came from a risk-stratified rule, "
        "and a policy that chooses differently meets the same observed states with a different mix of hidden stages. We did not test this directly. "
        f"Estimation optimism is a second effect: a DP policy trained on {fs['config']['n_eps']:,} episodes is predicted by its own model to be {og_s.mean():.2f} ± {_ci(og_s):.2f} (SMDP-aware) and {og_n.mean():.2f} ± {_ci(og_n):.2f} (naive) "
        f"better than on the oracle model; the paired difference is {og_n.mean() - og_s.mean():+.2f} ({pstr(og_p)}). This is the standard overfitting of plug-in planning.")

    P.h2("Learning dynamics and agreement among solvers")
    f_learn = P.fig(fig(I, "fig_learning_smdp"), "Learning curves on the oracle MDP (SMDP-aware model; mean ± s.d. over replicates). Both learners rise quickly and plateau below the rule's model value; the oracle DP optimum (dashed) is far above.", 7.0)
    P.p(f"Figure {f_learn} shows that learning dynamics are not where the difficulty lies. Both learners start at {CS['Monte Carlo']['c0']:.1f} (MC) and {CS['Q-learning']['c0']:.1f} (Q-learning) on the oracle MDP, "
        f"reach {CS['Monte Carlo']['c6']:.2f} and {CS['Q-learning']['c6']:.2f} after only {CS['Monte Carlo']['k6']:,.0f} steps, and settle at {CS['Monte Carlo']['cend']:.2f} and {CS['Q-learning']['cend']:.2f} after {CS['Monte Carlo']['kend']:,.0f} "
        f"(naive benchmark: {CN['Monte Carlo']['cend']:.2f} and {CN['Q-learning']['cend']:.2f}, on a different scale because the naive oracle discounts more mildly). Both end below the rule's model value ({DS['mr']:.2f}) and far below the oracle DP optimum ({DS['mo']:.2f}): "
        "the learners converge to the optimum of their own, imperfect, models, as the diagnostic table showed.")
    P.p(f"Q-learning reproduces the DP policy in about {100 * agS['Q-learning']:.0f}% of well-visited states (at least 5 visits) and Monte Carlo in {100 * agS['Monte Carlo']:.0f}%, with nearly the same numbers in the naive model "
        f"({100 * agN['Q-learning']:.0f}% and {100 * agN['Monte Carlo']:.0f}%). Policy iteration and value iteration "
        f"returned policies of the same value in {sum(fs['pi_equals_vi'])} of {len(fs['pi_equals_vi'])} SMDP-aware and {sum(fn['pi_equals_vi'])} of {len(fn['pi_equals_vi'])} naive replicates, so the solvers are correct. "
        "Agreement between solvers is evidence of convergence, not of the quality of the model they converge to.")

    P.h2("What the schedules look like")
    P.p(f"In the stored first replicate the three learned schedules stay close to the behaviour pattern, which is the rule with noise: they agree with the rule in {100 * seed0['DP']:.0f}% (DP), {100 * seed0['Monte Carlo']:.0f}% (MC) "
        f"and {100 * seed0['Q-learning']:.0f}% (Q-learning) of the 108 states. The oracle DP, in contrast, is very different from the rule: it agrees in only {100 * agree_rule_s:.0f}% of states (naive oracle: {100 * agree_rule_n:.0f}%; a random choice would agree in about 33%), "
        "which fits the diagnostic above: it is the optimum of a model whose state omits the disease stage.")
    f_int = P.fig(fig(I, "fig_intervals"), "Oracle DP intervals by risk tier and age band for patients whose previous interval was two years and previous screen negative: SMDP-aware (left) and naive (right).", 9.5)
    P.p(f"Figure {f_int} makes the oracle schedule concrete for one slice of the state space. The two oracle policies are close: they differ in {n_cell_diff} of the 18 cells shown and in {n_diff_oracle} of 108 states overall. In the low-risk row the SMDP-aware oracle chooses "
        f"{lowrow} for the six age bands, while the rule would say 3 years throughout; only {n_cell_rule} of the 18 cells match the rule. The pattern is not monotone in age or risk (high-risk patients aged 65-69 are screened annually, those aged 70 or more every three years), which no clinical rationale would predict; "
        "we read it as fitting noise and misspecification, not as a discovery. The SMDP correction changes one cell in this slice (middle risk, 60-64 years) "
        f"and {n_dp_diff} of the 108 states in the stored DP policy of the first replicate, consistent with a small effect on return at gamma = 0.90.")

    P.h2("Sweeping the discount factor")
    sec_sweep = f"{P.sec}.{P.sub}"
    rows = [["Yearly γ", "SMDP-aware DP return", "Naive DP return", "SMDP − naive (mean ± 95% CI)", "p", "States with different intervals"]]
    for k in GK:
        g = GS[k]
        rows.append([k, f"{g['smdp']:.2f}", f"{g['naive']:.2f}", pm(g["diff"]["mean"], g["diff"]["ci"]), f"{g['diff']['p']:.3f}", f"{100 * g['differ']:.1f}%"])
    t6 = P.table(rows, f"Discount-factor sweep (DP only; {nrep} replicates; same logged episodes for both versions). Returns are on different scales for different gamma, so only the paired difference should be compared across rows. "
                 "The last column is the share of well-visited states (at least 5 visits) in which the two policies differ.", widths=[1.8, 3.0, 2.6, 4.2, 1.2, 3.6], font=8.5)
    f_g = P.fig(fig(I, "fig_gamma_sweep"), "Left: return of SMDP-aware minus naive DP versus the yearly discount factor (95% intervals over replicates). Right: share of well-visited states in which the two policies differ.", 10.5)
    P.p(f"Table {t6} and Figure {f_g} give the answer to the question the paper set out to ask. H1 is supported, monotonically: the two planners choose different intervals in {ddif['0.6']:.1f}% of well-visited states at gamma = {g_lo:g}, "
        f"{ddif['0.75']:.1f}% at 0.75, {ddif['0.9']:.1f}% at 0.9 and only {ddif['0.97']:.1f}% at {g_hi:g}, consistent with Eq. (5), since the naive overstatement factor falls from {g_lo ** -2:.2f} to {g_hi ** -2:.2f}. "
        f"H2 is not supported. The return difference is {GS['0.6']['diff']['mean']:+.2f} ± {GS['0.6']['diff']['ci']:.2f} at gamma = {g_lo:g} ({pstr(GS['0.6']['diff']['p'])}), "
        f"{GS['0.75']['diff']['mean']:+.2f} ± {GS['0.75']['diff']['ci']:.2f} at 0.75 ({pstr(GS['0.75']['diff']['p'])}), {g9['diff']['mean']:+.2f} ± {g9['diff']['ci']:.2f} at 0.9 ({pstr(g9['diff']['p'])}) and "
        f"{GS['0.97']['diff']['mean']:+.2f} ± {GS['0.97']['diff']['ci']:.2f} at {g_hi:g} ({pstr(GS['0.97']['diff']['p'])}). The sign is negative at {len(neg)} of {len(GK)} discount factors, so on average the SMDP-aware policy is "
        f"the worse one, and nominally significant at {len(sig)} of them (gamma = {', '.join(sig)}). Against a Bonferroni threshold of {bonf:.4f} only gamma = 0.9 ({pstr(g9['diff']['p'])}) survives; the gamma = 0.75 result ({pstr(GS['0.75']['diff']['p'])}) does not. "
        f"The gamma = 0.9 entry uses different data seeds and 10,000 rather than {fs['config']['n_eval']:,} evaluation episodes, so it is a separate replication of the main benchmark's DP comparison ({dpd['mean']:+.2f}); it points the same way with a larger magnitude. "
        "The difference is not monotone in gamma: it is largest at the middle values and near zero at the extremes, where at gamma = 0.97 the two policies are almost identical.")
    P.p("What could make a more correct model produce worse policies? We list explanations, none of which we tested. (a) **Compensating errors.** The state omits the disease stage, so the estimated "
        "transition probabilities are policy-dependent and biased; the naive model's heavier weighting of the future after a long interval may offset part of that bias. "
        "(b) **Proximity to the rule.** The naive planner favours long intervals where the continuation is positive, as the rule does for the numerous low-risk patients. "
        f"(c) **Noise.** A t-based interval would be {tfac:.2f} times wider, and the three nominally significant findings are modest against the multiple-comparison burden. "
        "We regard 'the SMDP correction is harmful' as unproven; 'it is not reliably beneficial' is what the data support.")

    # ------------------------------------------------------------------ discussion
    P.h1("Discussion")
    P.p("**What the paper shows.** In a simulator in which the interval is both the action and the elapsed time, the principled SMDP treatment did not improve tabular planning or Q-learning, improved Monte Carlo control by a margin "
        f"that does not survive multiple-comparison correction, and was nominally worse than the shortcut at two of four discount factors. It changes the policy most at low discount factors (from {ddif['0.97']:.1f}% of states at gamma = {g_hi:g} to {ddif['0.6']:.1f}% at {g_lo:g}), but the return effect does not follow. "
        f"At gamma = 0.9 the two treatments differ in only {ddif['0.9']:.1f}% of well-visited states, and the largest return difference in the sweep ({max(abs(GS[k]['diff']['mean']) for k in GK):.2f}) is small next to the model error of {DS['pred'] - DS['true']:.1f} return units in Table {t4}. "
        "Checking the model's ranking of policies matters more than refining its discounting.")
    P.p(f"**The larger lesson is about the model, not the clock.** In Table {t4} the oracle policy from either model is worse than the rule by about {abs(DS['true']):.1f} return units while the model promises a gain of "
        f"{DS['pred']:.1f} to {DN['pred']:.1f}. An error that large swamps the second-order effect of discounting, which is why the SMDP correction cannot show up cleanly. The mechanism is the usual one in offline RL [@levine]: "
        "planning on a model estimated under one policy and deploying another. Restricting to the support of the data [@fujimoto], penalising uncertainty [@kumar,jin] or enriching the state "
        "(for example with time since the last positive result, which we did not test) would likely matter more here than the semi-Markov bookkeeping.")
    P.p(f"**DP, MC and Q-learning.** None of the three solvers dominates. Q-learning reproduces the DP policy in about {100 * agS['Q-learning']:.0f}% of well-visited states; MC is the best learner in the SMDP-aware benchmark and the worst in the naive one, "
        f"a swing of {abs(S['Monte Carlo']['J'] - N['Monte Carlo']['J']):.2f} return units, larger than for the other solvers. Because MC averages complete returns, an interval-discount error could plausibly be amplified there, "
        "but we have not isolated this and would not rest a conclusion on one nominally significant comparison. The rule is difficult to beat, and a clinician-in-the-loop design that adjusts it only where data are dense would be a more defensible use of learning.")
    limitations(P, [
        "**Hidden stage.** The latent disease stage is not in the state, so the observed process is non-Markov. This is probably the main reason the oracle DP fails and limits any conclusion about the SMDP correction in isolation.",
        "**Interval equals duration.** The transition duration is a deterministic function of the action, so the estimated continuation weight is exactly gamma^tau(a). Real intervals vary around the recommendation (late attendance).",
        f"**Few replicates and multiple comparisons.** Five replicates, 1.96-standard-error intervals ({tfac:.2f} times narrower than t-intervals) and {ntests} comparisons make the MC result suggestive at best; the sweep used different data seeds from the main benchmark.",
        "**Arbitrary reward.** The payoff scale (+100, +8, -80) is hand-set; different weights could reorder the fixed schedules, including annual screening.",
        "**Datasets mirrored, not used.** Breast Cancer Wisconsin and Cervical Cancer Risk tables are, to our knowledge, cross-sectional and contain no inter-screening intervals, so they could not identify an SMDP without longitudinal registry data.",
    ])
    ethics_future(P, "(1) Add time since the last positive result to the state and test whether the oracle-versus-rule sign error disappears; "
                  "(2) make intervals random and vary the discount form; (3) repeat the sweep with about 30 replicates to settle the sign at gamma = 0.75 and 0.9; "
                  "(4) apply support-constrained or pessimistic planning; (5) use a longitudinal screening-registry table through the adapter and compare model-predicted advantage with a held-out cohort.")
    P.h1("Conclusion")
    P.p(f"On a simulated screening problem in which each action is also a length of time, treating the problem as a semi-Markov decision process did not reliably improve schedules learned from logs. "
        f"The paired difference was {dpd['mean']:+.2f} for DP ({pstr(dpd['p'])}) and {qd['mean']:+.2f} for Q-learning ({pstr(qd['p'])}); Monte Carlo gained {mcd['mean']:+.2f} ({pstr(mcd['p'])}), which does not survive correction for multiple comparisons. "
        f"Across discount factors from {g_lo:g} to {g_hi:g} the two planners disagree in a shrinking share of states ({ddif['0.6']:.1f}% to {ddif['0.97']:.1f}%) and the return difference is not monotone, being negative at gamma = 0.75 and 0.9. "
        f"The dominant failure is elsewhere: the oracle DP policy scores {S['Oracle DP']['J']:.2f}, below the risk-stratified rule ({rule:.2f}), although its model predicts a gain of {DS['pred']:+.2f}. The finding is a simulation result about algorithms, "
        "to be re-examined on longitudinal screening data with the accompanying adapter, and it supports no clinical conclusion.")

    defense(P, [
        ("Why should I trust a simulator result about screening intervals?",
         "You should not read it as evidence about patients. The Kaggle site returned HTTP 403, so no real table was analysed, and the simulator is hand-specified and uncalibrated. "
         "Its value is that true policy return is measurable, which lets us test whether a modelling choice changes the quality of a learned schedule."),
        ("Is the SMDP correction worthwhile, given your results?",
         f"The data do not show a reliable benefit. At gamma = 0.9 the paired difference is {dpd['mean']:+.2f} ± {dpd['ci']:.2f} for DP ({pstr(dpd['p'])}) and {qd['mean']:+.2f} ± {qd['ci']:.2f} for Q-learning ({pstr(qd['p'])}). "
         f"The only nominally positive result, Monte Carlo at {mcd['mean']:+.2f} ({pstr(mcd['p'])}), is one of {ntests} tests and above the Bonferroni threshold of {bonf:.4f}. The correction is theoretically right and cheap; we could not show that it pays off here."),
        ("Why is the SMDP-aware policy worse at gamma = 0.75 and 0.9?",
         f"We do not know. The differences are {GS['0.75']['diff']['mean']:+.2f} ({pstr(GS['0.75']['diff']['p'])}) and {g9['diff']['mean']:+.2f} ({pstr(g9['diff']['p'])}); only the latter survives a Bonferroni correction. "
         f"Candidate explanations are listed in Section {sec_sweep} and were not tested, so we claim only that the correction is not reliably beneficial."),
        ("Why does the oracle DP lose to the rule when it has ten times more data?",
         f"Its model is wrong in a systematic way. It predicts a {DS['pred']:+.2f} advantage over the rule and delivers {DS['true']:+.2f} (naive: {DN['pred']:+.2f} predicted, {DN['true']:+.2f} delivered; Table {t4}). "
         "The hidden disease stage is left out of the state, so a policy that differs from the logged rule meets a different mix of stages than the model learned."),
        ("Are five replicates enough?",
         f"Enough to see that effects are small, not to settle the sign of marginal ones. Our intervals are 1.96 standard errors; a t-interval would be {tfac:.2f} times wider. For Monte Carlo it would still exclude zero (lower end about {mcd['mean'] - tfac * mcd['ci']:+.2f}) "
         f"but the Bonferroni-adjusted test would not reject. We recommend 30 replicates for a confirmatory run."),
        ("What would be needed to run this on the real Kaggle files?",
         "A longitudinal table with dates of successive screens and results. The Breast Cancer Wisconsin and Cervical Cancer Risk tables are, to our knowledge, cross-sectional and could only supply covariates; "
         "interval durations would have to come from a registry. kaggle_adapter.transitions_from_dataframe then takes the interval as the duration column, and evaluation must be model-based and labelled as an estimate."),
    ])
    repro(P, 5, ["fig_returns", "fig_pareto", "fig_bars_smdp", "fig_learning_smdp", "fig_agreement_smdp", "fig_policies_smdp", "fig_intervals", "fig_gamma_sweep"])
    import domains
    code_appendix(P, 5, [domains.Screening])
    rl_core_appendix(P)
    P.refs()
    path = f"{PAP}/Paper05_Screening_SMDP_Gamma.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
