import numpy as np
from docx_helper import Paper
from paper_common import *

I = 10


def build():
    r = load(I); M, L = r["memory"], r["memoryless"]
    fm, fl = r["mem_full"], r["less_full"]
    P = Paper("Fatigue-Aware Nudging: State Augmentation Prevents Habituation Failures in Dynamic-Programming, Monte Carlo and Q-learning Health Interventions",
              "just-in-time adaptive interventions; wearable data; habituation; state augmentation; dynamic programming; Monte Carlo control; Q-learning",
              running="Fatigue-aware nudging", refdict=REFS)
    adv = lambda full, m: float(np.mean(full["methods"]["Oracle DP"]["Jor"]) - np.mean(full["methods"]["Rule"]["Jor"]))
    adv_true = lambda full: float(np.mean(full["methods"]["Oracle DP"]["J"]) - np.mean(full["methods"]["Rule"]["J"]))
    pm_adv, pm_true = adv(fm, None), adv_true(fm)
    pl_adv, pl_true = adv(fl, None), adv_true(fl)
    rate = r["nudge_rate_by_habituation"]; mix = np.array(r["action_mix_by_habituation"])
    P.abstract(
        "Mobile-health apps send nudges - reminders to sleep, move or relax - but people habituate: the same message sent every day loses its effect. We ask "
        "whether a learned nudging policy must model that habituation explicitly. On a wearable-style simulator (columns mirroring Kaggle sleep-and-lifestyle "
        "and Fitbit tables; the Kaggle site was unreachable, so no real file was used) we formulate daily nudging as a Markov decision process and solve it "
        "by dynamic programming (DP), first-visit Monte Carlo control and Q-learning, with and without a **fatigue-aware state** that appends a four-level habituation band to the "
        f"observed sleep, activity and stress bands. Without the augmentation all three solvers over-nudge ({L['DP']['info']['nudges']:.0f} of 28 days for DP, versus "
        f"{L['Rule']['info']['nudges']:.0f} for an expert rule) and lose {M['Rule']['J'] - L['DP']['J']:.2f} return units to the rule; with it they nudge about {M['DP']['info']['nudges']:.0f} days and are back within "
        f"{abs(M['Rule']['J'] - M['DP']['J']):.2f} of the rule. The augmentation improves DP by {r['mem_vs_less_dp']['mean']:.2f} ± {r['mem_vs_less_dp']['ci']:.2f} ({pstr(r['mem_vs_less_dp']['p'])}), "
        f"Q-learning by {r['mem_vs_less_q']['mean']:.2f} and Monte Carlo by {r['mem_vs_less_mc']['mean']:.2f}. A diagnostic shows why: a memoryless model predicts that the optimal policy beats the rule by "
        f"{pl_adv:+.2f} while it truly loses by {-pl_true:.2f}; the fatigue-aware model predicts {pm_adv:+.2f} against a true {pm_true:+.2f}. The effect is a policy-induced distribution shift. "
        "No learned policy beat the expert rule, and all results are simulated.")

    P.h1("Introduction")
    P.p("Just-in-time adaptive interventions (JITAIs) deliver support - a prompt, a tip, a reminder - when a person is most likely to benefit and able to "
        "respond [@nahum]. The scientific questions are old (what to send, when, to whom), but a new practical one has appeared with the abundance of wearable data: "
        "can a sequential decision policy be learned from the logs of existing app behaviour rather than hand-written? Early field studies with reinforcement "
        "learning suggest the answer can be yes for physical-activity messaging [@yomtov], and the broader literature on dynamic treatment regimes provides the statistical "
        "framework [@murphy03,murphy05].")
    P.p("A recurring obstacle is **habituation**. A nudge that works on day one is less effective on day five if it was also sent on days two to four. The effect is a "
        "dependence of the response on recent nudge history, which is not part of the observed physiological state. A learner that ignores history and treats each "
        "day as a fresh Markov state will see an average nudge response that mixes fresh and habituated contexts. If the logged data came from a cautious clinician "
        "who nudged rarely, the average will be dominated by fresh contexts, and the learner will conclude that nudging is more effective than it really is for a "
        "policy that nudges often. This is an instance of the policy-induced distribution shift that plagues offline reinforcement learning [@levine].")
    P.p("**Contributions.** (i) A transparent simulator of daily nudging with a hidden, exponentially-decaying habituation variable. (ii) A fatigue-aware state, a four-level band "
        "of an exponentially weighted nudge count, that restores the Markov property approximately and costs only a four-fold increase in states. (iii) A head-to-head comparison "
        "of DP, Monte Carlo control and Q-learning with and without the augmentation, scored exactly in the simulator. (iv) A diagnostic - model-predicted versus true advantage over "
        "a reference policy - that exposes the distribution shift directly. (v) A candid negative result: no learned policy beats a well-designed expert rule here, so the value "
        "of augmentation is in preventing a failure, not in creating a win.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached from the build environment, so no real dataset was analysed. The simulator mirrors the variables of "
        "sleep-and-lifestyle and wearable tables (sleep quality, activity, stress) and uses made-up dynamics; the findings concern algorithms, not people.")

    P.h1("Related work")
    P.p("**Adaptive mobile interventions.** The JITAI framework organises decision points, tailoring variables, intervention options and decision rules [@nahum]. Reinforcement "
        "learning is a natural engine for decision rules, and randomised evidence for an RL-driven activity-messaging system in people with diabetes has been reported [@yomtov]. "
        "We do not claim to replicate such trials; we study a mechanism (habituation) that makes a naive RL formulation fail.")
    P.p("**Dynamic treatment regimes.** The statistics literature formalises sequences of decision rules and their estimation by Q-learning and related methods [@murphy03,murphy05]; "
        "the MDP view is identical. Our augmentation is the practitioner's response to a familiar issue in that literature: the decision rule must condition on a sufficient history summary.")
    P.p("**RL in healthcare and offline learning.** Reviews emphasise that learning from observational logs requires care with confounding and coverage [@yu,gottesman,levine]. "
        "A common safeguard is to restrict a policy to the support of the data [@fujimoto]; we use a count threshold of this kind. Our diagnostic isolates a failure that a support "
        "threshold does not catch: every individual (state, action) pair is well supported, but the states themselves are incomplete.")
    background(P)

    P.h1("Problem formulation: a nudging simulator with habituation")
    P.p("Each day the app may send one of four nudges: none, a sleep-hygiene prompt, an activity prompt or a stress-management prompt. Three latent scores in [0, 1] describe the "
        "person: sleep quality, activity and stress. Each reverts toward a personal set point with noise (rates 0.25, 0.25 and 0.20), stress is raised by poor sleep, "
        "and a nudge pushes its target score toward the favourable end by an amount proportional to its **responsiveness** m = exp(−0.9 h). The hidden habituation variable h "
        "increases by one whenever any nudge is sent and decays by a factor 0.7 each day, so h is an exponentially weighted count of recent nudges. A fresh person (h = 0) has "
        "m = 1; after one nudge m is about 0.41; after nudges on three consecutive days h is about 2.2 and m about 0.14.")
    P.table([["Element", "Memoryless state", "Fatigue-aware state"],
             ["Observed bands", "sleep, activity, stress (3 each)", "same, plus habituation band"],
             ["Habituation band", "not observed", "h in [0, 0.5), [0.5, 1.2), [1.2, 2.0), [2.0, ∞)"],
             ["Number of states", "27", "108"],
             ["Actions", "none, sleep, activity, stress nudge", "same"],
             ["Reward per day", "health score − 0.06 · 1[nudge];  health = (sleep + activity + (1 − stress)) / 3", "same"],
             ["Discount, horizon", "γ = 0.97, 28 days", "same"],
             ["Behaviour policy", "nudge the worst-banded domain when it is in the lowest band; ε = 0.3 random actions", "same"],
             ["Dataset mirrored", "Sleep Health and Lifestyle; Fitbit / wearable activity tables", ""]],
            "The two state definitions compared in this paper.", widths=[3.6, 6.4, 6.4], font=8.5)
    P.p("The behaviour policy is deliberately sensible: it nudges only when a domain is poor. Consequently the logs contain few runs of consecutive nudges, so habituation is "
        "rarely visible in the data. This is exactly the situation in which a memoryless learner is most misled, and it is also the situation that real app logs, "
        "produced by cautious rule-based nudging, are likely to resemble.")

    P.h1("Method: fatigue-aware state augmentation")
    P.p("Let h_t be the hidden habituation, h_{t+1} = 0.7 h_t + 1[a_t ≠ none]. The expected effect of a nudge in a given observed state s is")
    P.eq("E[Δ | s, a] = Σ_h p(h | s) · g(a) · exp(−0.9 h),", "3")
    P.p("where p(h | s) is the distribution of habituation among visits to state s and g(a) is the nominal effect size. A memoryless learner estimates (3) with p equal to the "
        "distribution under the behaviour policy, p_b(h | s). A policy that nudges more often visits s with larger h, so the effect it will actually obtain is governed by "
        "p_π(h | s), which is shifted toward larger h and gives a smaller effect. Because the sign of the advantage of a nudge can change when its effect shrinks "
        "(the effect must exceed the cost 0.06), the planned policy can be actively harmful, not merely suboptimal. Augmenting the state with a discretised ĥ makes p(h | s, ĥ) approximately "
        "degenerate, so the estimate of (3) becomes conditional on the right quantity and the planning model is correct whichever policy is evaluated.")
    P.p("The augmentation requires no new learning machinery - the MDP, DP, MC and Q-learning code is unchanged - and increases the number of states only from 27 to 108. In a "
        "real app ĥ is computable from the notification log alone, so it is observable. The cost is statistical: four times as many states to estimate from the same data.")

    P.h2("Algorithm")
    P.code("""Algorithm 1  Fatigue-aware nudging from logs
 input : logged episodes (day, sleep/activity/stress bands, nudge, reward), decay rho = 0.7, band edges E
 1. for every logged day compute h_hat = rho * h_hat_prev + 1[nudge_prev != none]   (from the notification log alone)
 2. state s = (sleep band, activity band, stress band, band(h_hat; E))                 (27 x 4 = 108 states)
 3. count transitions N(s,a,s'), rewards R(s,a); (s,a) with fewer than 10 visits are treated as unsupported
 4. solve the estimated MDP by value iteration -> pi_DP;   optionally learn pi_Q, pi_MC by Q-learning / MC control
 5. in unsupported states fall back to the modal logged action
 6. deploy: each morning recompute s from sensors and the notification log, send nudge pi(s)""")
    P.p("A worked example illustrates the habituation arithmetic. Suppose a person receives a sleep prompt on days 1, 2 and 3. Their habituation is h = 1 after day 1, h = 1.7 after day 2 and "
        "h = 2.19 after day 3, so the responsiveness on day 4 is exp(-0.9 x 2.19) = 0.14, about one seventh of a fresh person. If no further nudge is sent, h decays to 1.53, 1.07 and 0.75 on days 4, 5 and 6, "
        "and responsiveness recovers to 0.25, 0.38 and 0.51. A policy that waits two days between nudges therefore obtains about half the effect of a fresh nudge each time, while a policy that nudges "
        "daily obtains well under a quarter. Since a nudge costs 0.06 and its benefit is the product of effect size and responsiveness, the break-even responsiveness is a property of the cost, and an optimal "
        "policy must be able to see where on this curve the person currently is.")
    protocol(P, [["Item", "Setting"],
                 ["Simulator", "daily nudging; sleep, activity, stress in [0, 1]; hidden habituation h"],
                 ["States / actions", "27 (memoryless) or 108 (fatigue-aware) / 4"],
                 ["Logged data", "300 episodes × 28 days = 8,400 transitions per replicate"],
                 ["Oracle model", "3,000 episodes for exact learning-curve scoring"],
                 ["Replicates", "6 independent data draws and learner seeds per condition"],
                 ["Learners", "DP (value iteration), first-visit Monte Carlo, Q-learning; 150,000 steps each"],
                 ["Scoring", "300 episodes per policy in the true simulator (common random numbers)"]],
             150000, 6, 300)

    P.h1("Results")
    P.h2("Headline comparison")
    rows = [["Method", "Return, memoryless", "Return, fatigue-aware", "Nudge-days, memoryless", "Nudge-days, fatigue-aware"]]
    for k in ("Rule", "Oracle DP", "DP", "Monte Carlo", "Q-learning"):
        rows.append([NICE.get(k, k), pm(L[k]["J"], L[k]["ci"]), pm(M[k]["J"], M[k]["ci"]), f"{L[k]['info']['nudges']:.1f}", f"{M[k]['info']['nudges']:.1f}"])
    t1 = P.table(rows, "True-simulator return (28 days, 95% CI over 6 replicates) and number of days with a nudge, by state definition.", widths=[4.4, 3.0, 3.0, 3.0, 3.0], font=8.5)
    P.p(f"Table {t1} is the headline result. Without the fatigue band, DP scores {L['DP']['J']:.2f} ± {L['DP']['ci']:.2f}, Q-learning {L['Q-learning']['J']:.2f} and Monte Carlo {L['Monte Carlo']['J']:.2f}, all clearly "
        f"below the expert rule ({L['Rule']['J']:.2f}); the cause is visible in the right-hand columns - they nudge on {L['DP']['info']['nudges']:.0f}, {L['Q-learning']['info']['nudges']:.0f} and "
        f"{L['Monte Carlo']['info']['nudges']:.0f} days of 28, three to four times as often as the rule. Even the oracle DP, trained on ten times more data, is worse than the rule "
        f"({L['Oracle DP']['J']:.2f} versus {L['Rule']['J']:.2f}) and nudges on {L['Oracle DP']['info']['nudges']:.0f} days: more data makes the memoryless policy more confidently wrong, "
        f"because the problem is a misspecified model, not a lack of samples. With the fatigue band, DP ({M['DP']['J']:.2f} ± {M['DP']['ci']:.2f}), Monte Carlo ({M['Monte Carlo']['J']:.2f}) and Q-learning "
        f"({M['Q-learning']['J']:.2f}) are all within {max(abs(M['Rule']['J'] - M[k]['J']) for k in ('DP', 'Monte Carlo', 'Q-learning')):.2f} of the rule, and the oracle ({M['Oracle DP']['J']:.2f}) is marginally above it.")
    f1 = P.fig(fig(I, "fig_memory_compare"), "Memoryless (grey) versus fatigue-aware (blue) state: true return, summed health score and number of nudge-days (means with 95% CIs). "
               "Augmentation lifts every learner to the rule's level and halves nudging.", 14.0)
    P.p(f"Figure {f1} shows that the gain does not come from a higher health score alone: summed health is {M['DP']['info']['health']:.2f} for fatigue-aware DP and "
        f"{L['DP']['info']['health']:.2f} for memoryless DP, a difference of {M['DP']['info']['health'] - L['DP']['info']['health']:.2f}, while nudge-days fall from "
        f"{L['DP']['info']['nudges']:.1f} to {M['DP']['info']['nudges']:.1f} (a paired difference of {r['nudges_dp']['mean']:.1f} ± {r['nudges_dp']['ci']:.1f} days, {pstr(r['nudges_dp']['p'])}). "
        f"Paired differences in return across replicates are {r['mem_vs_less_dp']['mean']:.2f} ± {r['mem_vs_less_dp']['ci']:.2f} for DP, "
        f"{r['mem_vs_less_q']['mean']:.2f} ± {r['mem_vs_less_q']['ci']:.2f} for Q-learning and {r['mem_vs_less_mc']['mean']:.2f} ± {r['mem_vs_less_mc']['ci']:.2f} for Monte Carlo "
        f"({pstr(r['mem_vs_less_q']['p'])} and {pstr(r['mem_vs_less_mc']['p'])} respectively), so the benefit is not specific to the solver.")

    P.h2("What the learned policies do over time")
    f2 = P.fig(fig(I, "fig_trajectories"), "Left: probability of a nudge on each day of the 28-day episode. Right: mean health score. The memoryless DP policy keeps "
               "nudging at roughly 50-60% per day throughout; the fatigue-aware policy and the rule settle near 20-35%.", 13.0)
    P.p(f"Figure {f2} shows the mechanism. All policies nudge heavily on day 1 (about 75%), when the person is fresh and the response is largest. The rule and the fatigue-aware "
        "policy then back off to roughly one nudge in four or five days, which lets habituation decay between nudges, and the health score climbs and stays higher. The memoryless "
        "policy never backs off: it continues to nudge about every other day, keeps the person habituated, and its health score drifts down relative to the other two after day 10. "
        "The memoryless planner cannot see the accumulating habituation, so it has no reason to stop.")
    f3 = P.fig(fig(I, "fig_habituation_policy"), "Action mix of the fatigue-aware oracle DP policy by habituation band (states observed in the data). The probability of 'none' rises with habituation.", 8.5)
    P.p(f"Figure {f3} reads the learned policy. In fresh states (band h0) the fatigue-aware policy nudges in {100 * rate[0]:.0f}% of states; in habituated states (band h3) it nudges in "
        f"{100 * rate[3]:.0f}% and in the intermediate bands in {100 * rate[1]:.0f}% and {100 * rate[2]:.0f}%. The monotone decline is the behaviour that a clinician would endorse "
        "(\"give the person a break\"), and it emerged from the reward alone, without being programmed. The decline is gentler than one might expect - even in the most habituated "
        "band the policy still nudges in about half the supported states - because a poor domain in a habituated person can still justify a nudge when the cost is only 0.06.")

    P.h2("Diagnosing the failure: model-predicted versus true advantage")
    rows = [["Quantity", "Memoryless", "Fatigue-aware"],
            ["Model-predicted value of oracle DP minus value of rule (on the oracle MDP)", f"{pl_adv:+.2f}", f"{pm_adv:+.2f}"],
            ["True value of oracle DP minus value of rule (simulator)", f"{pl_true:+.2f}", f"{pm_true:+.2f}"],
            ["Sign agreement", "no" if pl_adv * pl_true < 0 else "yes", "no" if pm_adv * pm_true < 0 else "yes"]]
    t2 = P.table(rows, "Does the model rank policies correctly? Differences in return between the oracle DP policy and the expert rule, as predicted by the estimated model and as measured in the simulator.", widths=[9.4, 3.5, 3.5], font=8.5)
    P.p(f"Table {t2} is the sharpest evidence for the distribution-shift explanation. The memoryless model, even when estimated from 3,000 episodes, predicts that its optimal "
        f"policy beats the rule by {pl_adv:+.2f}; in the simulator the same policy is worse than the rule by {-pl_true:.2f}. The model has the wrong sign. The fatigue-aware model predicts "
        f"{pm_adv:+.2f} and the simulator delivers {pm_true:+.2f}: small, but of the same sign and similar in size. (Absolute model values in this time-limited problem are lower than "
        "simulator values because the model treats the 28-day horizon as a constant termination hazard; this affects both policies equally, so the difference is the meaningful "
        "quantity.) The conclusion is that the memoryless failure is a model failure, and that more data from the same logs cannot fix it.")

    P.h2("Learning curves")
    f4 = P.fig(fig(I, "fig_learning_memory"), "Learning curves on the oracle MDP, fatigue-aware state. Q-learning and Monte Carlo both approach the DP optimum.", 9.5)
    f5 = P.fig(fig(I, "fig_learning_memoryless"), "Learning curves on the oracle MDP, memoryless state. Both learners converge quickly - to a policy that the simulator shows to be poor.", 9.5)
    P.p(f"Figures {f4} and {f5} show that learning dynamics are not the problem. In both state definitions Q-learning and Monte Carlo converge to near the DP optimum of their own "
        f"model within the step budget, agreeing with the DP policy in {100 * np.mean(fm['agree']['Q-learning']):.0f}% (fatigue-aware) and {100 * np.mean(fl['agree']['Q-learning']):.0f}% (memoryless) of "
        "well-visited states for Q-learning. A learner can converge perfectly to the optimum of the wrong model. This is the same lesson as Table 3: algorithmic convergence "
        "diagnostics do not detect a misspecified state.")

    P.h2("Agreement between learners and per-state behaviour")
    rows = [["Learner", "Agreement with DP, memoryless", "Agreement with DP, fatigue-aware"]]
    for k in ("Monte Carlo", "Q-learning"):
        rows.append([k, f"{100 * np.mean(fl['agree'][k]):.0f}%", f"{100 * np.mean(fm['agree'][k]):.0f}%"])
    t3 = P.table(rows, "Fraction of well-visited states in which the learner chooses the same action as DP on the estimated model.", widths=[4.4, 5.8, 5.8], font=8.5)
    P.p(f"Table {t3} shows that Monte Carlo and Q-learning recover the DP policy in most well-visited states under both definitions. High agreement in the memoryless case is a warning rather than a comfort: "
        "all three solvers are optimising the same flawed model and so they agree on the same flawed policy. Agreement among solvers is therefore evidence of convergence, not of correctness.")
    f6 = P.fig(fig(I, "fig_policies_memory"), "Policy strips for the fatigue-aware state: expert rule, DP, Monte Carlo and Q-learning actions by state.", 14.0)
    f7 = P.fig(fig(I, "fig_policies_memoryless"), "Policy strips for the memoryless state. The learned policies nudge in nearly every state.", 14.0)
    P.p(f"Figures {f6} and {f7} display the policies state by state. In the memoryless strips almost every state receives a nudge, which is the visual signature of the over-nudging in Table {t1}; "
        "in the fatigue-aware strips the cells with a high habituation band are more often blank, reflecting the break-giving behaviour of Figure " + str(f3) + ".")
    P.h1("Discussion")
    P.p("**Why no learned policy beats the rule.** The behaviour policy nudges only when a domain is poor, which is close to what the simulator's reward favours; the rule also "
        "uses a single threshold that happens to align with the cost of a nudge. Under such conditions the Bellman-optimal policy is not very different from the rule, and "
        "estimation noise in 8,400 transitions is enough to offset any small advantage. The honest summary is that the fatigue band restores parity with a strong expert rule. "
        "In practice an app's rule will be less well matched, so the room for improvement could be larger - but the simulator cannot show that.")
    P.p("**Design guidance.** (1) Any intervention whose effect decays with exposure needs an exposure summary in the state. It is cheap to compute from the notification log. "
        "(2) Check model-predicted against measured advantage whenever a simulator or randomised holdout exists; a sign disagreement is a red flag that no amount of data will fix. "
        "(3) Prefer an exposure summary with few levels: four bands were enough here, and each added band multiplies the data requirement.")
    P.p("**Algorithm choice.** DP, Q-learning and Monte Carlo all benefit equally from the augmentation, so the state definition matters more than the solver. With the correct state, "
        "Monte Carlo is competitive on this short, 28-step horizon, unlike in longer-horizon problems where its variance dominates.")
    limitations(P, [
        "**Habituation form.** The simulator's habituation is a single exponential memory. Real habituation is person-specific and may recover or sensitise; a sufficient summary could need more than one number.",
        "**Observable exposure.** We assume the habituation band is computed from the log exactly. In practice it is a noisy proxy for the person's true state of attention.",
        "**One nudge per day, no content.** Real systems vary message content, timing and channel; our action set is a coarse abstraction.",
        "**No engagement dynamics.** People can disable notifications after too many nudges. Such drop-out would make over-nudging even worse than the simulator shows.",
    ])
    ethics_future(P, "(1) Replace the exponential habituation by person-specific, possibly non-monotone responsiveness and test whether a learned summary of exposure can replace the hand-built band; "
                  "(2) combine the fatigue band with the support-threshold and robust methods used for other conditions; (3) run the adapter on a real wearable log and compare the model-predicted advantage of the "
                  "fatigue-aware policy to an A/B holdout, the only true test of the diagnostic proposed here.")
    P.h1("Conclusion")
    P.p(f"A learner that treats each day as a fresh Markov state cannot see habituation. On a wearable-style simulator, DP, Monte Carlo control and Q-learning without a fatigue "
        f"band all over-nudge ({L['DP']['info']['nudges']:.0f}, {L['Monte Carlo']['info']['nudges']:.0f} and {L['Q-learning']['info']['nudges']:.0f} of 28 days) and underperform an expert rule; with a four-level habituation band they "
        f"nudge about {M['DP']['info']['nudges']:.0f}-{M['Q-learning']['info']['nudges']:.0f} days and match it. The model-predicted-versus-true advantage diagnostic shows that the memoryless failure is a policy-induced distribution shift, "
        "so it persists with more data. The practical message is simple: give the state a memory of recent exposure. The scientific caveat is equally simple: this is a simulator study, "
        "and no learned policy outperformed the expert rule.")
    defense(P, [
        ("Why should I believe a simulator result about habituation?",
         "You should not believe it as a statement about people. It demonstrates a mechanism - a model that ignores exposure history is misled when logs come from a cautious policy - "
         "and measures how large the consequence can be when ground truth is known. The adapter provided lets the same analysis be run on real logs, where only model-based estimates are possible."),
        ("Is the fatigue band just an extra feature that adds information?",
         f"Yes, and that is the point: it is the minimal piece of history that makes the dynamics approximately Markov. Table {t2} shows its effect on policy ranking, not merely on fit."),
        ("Why did even the oracle memoryless DP lose to the rule?",
         f"Because the model is wrong in a systematic way. The memoryless oracle predicts a {pl_adv:+.2f} advantage over the rule and delivers {pl_true:+.2f}. The error is not statistical noise: it is the habituation distribution that differs between behaviour and learned policy."),
        ("Are the four habituation bands optimal?",
         "They were chosen in advance and not tuned. A finer partition would reduce approximation error and increase data requirements; testing 3, 4 and 6 bands is the obvious sensitivity analysis, and the code accepts any bin edges."),
        ("Why does the rule beat the learned policies, and is that a problem for your claim?",
         "It limits the claim. We claim only that augmentation prevents a failure and restores parity with the rule; we do not claim superiority. The rule matches the simulator's cost structure well, which is a favourable case for the baseline."),
    ])
    repro(P, 10, ["fig_memory_compare", "fig_trajectories", "fig_habituation_policy", "fig_learning_memory", "fig_learning_memoryless"])
    import domains
    code_appendix(P, 10, [domains.Nudge])
    rl_core_appendix(P)
    P.refs()
    path = f"{PAP}/Paper10_FatigueAware_Nudging.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
