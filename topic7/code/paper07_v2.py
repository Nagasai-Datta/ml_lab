"""paper07_v2.py -- version 2 of the paper: Paper 7 and our work written as one paper.

A copy of paper07.build() (Paper 7's builder) in which Paper 7's own text is edited where the two parts meet, so the
paper reads as one study with no contradictions: same title, same experiments, numbers, tables and figures of
Paper 7, our sections placed as 4.2, 5.2, 6.3 and 7.5 to 7.10 without "Extension" labels, and one discussion,
limitations list, future work list, conclusion and appendix set. Version 1 (paper07.py) is not affected.

    python topic7/code/paper07_v2.py      writes topic7/paper/Paper07_v2_edited.docx

check_paper.py lists every paragraph of Paper 7 that this version changes.
"""
import numpy as np
from scipy import stats
from docx_helper import Paper
from paper_common import *
import domains
import paper07_ext as ext

# ----------------------------------------------------------------------------- Paper 7's shared blocks (paper_common.py), edited for version 2
def protocol_v2(P, rows, steps, seeds, n_eval, extra=""):
    P.h1("Experimental protocol")
    P.h2("Data: a simulator and real clinic sessions")
    P.p("**Important disclosure.** The Kaggle website could not be reached from the environment in which the simulator study was built (all requests "
        "returned HTTP 403), so Sections 7.1 to 7.4 run on a transparent, hand-specified stochastic "
        "simulator whose recorded variables mirror the columns of the dataset named in the paper. The simulator plays two roles: it "
        "generates a logged dataset by running an ε-noisy clinician-like behaviour policy (the stand-in for the Kaggle table), and it is the "
        "ground truth against which every learned policy is scored. Using a simulator is a methodological choice with a real benefit "
        "(true policy value is measurable, which is impossible with a static table) and a real cost (the numbers are not clinical "
        "evidence). The table was obtained later. It records no overbooking decisions, so the data adapter of the accompanying code, which converts a table of logged "
        "decisions into the same transition format, does not apply to it; Section 4.2 uses it instead to rebuild real clinic sessions, on which Sections 7.8 to 7.10 score "
        "every policy. All results should be read as a method study, not as a finding about patients.")
    P.h2("Data, solvers and scoring")
    P.table(rows, "Experimental configuration.", widths=[5.0, 11.0])
    P.bullets([
        f"**Learners.** MC control and Q-learning each receive {steps:,} environment steps drawn from the estimated MDP, with an exponentially "
        "decaying ε-greedy schedule (ε from 1.0 to about 0.05) and a visit-count step size α = max(0.02, 1/(1 + n^0.7)) for Q-learning.",
        f"**Replicates.** {seeds} independent data draws (each with its own logged dataset and learner seeds). Intervals are 95% t-style "
        "intervals over replicates; paired differences are tested with a one-sample t-test on the per-replicate differences.",
        f"**Scoring.** Each final policy is run for {n_eval} episodes in the true simulator using common random numbers (episode e of every policy "
        "starts from the same random stream), and the mean discounted return is recorded. Learning curves are scored exactly on a large "
        "'oracle' MDP estimated from ten or more times as much logged data.",
        "**Fallback.** In states never observed in the (supported) data a learned policy defers to the clinician's modal logged action, "
        "so that unsupported states cannot be exploited.",
    ])
    if extra: P.p(extra)


def limitations_v2(P, specific):
    B = ext.res()["B"]
    P.h1("Limitations and threats to validity")
    P.bullets(specific + [
        "**Known clinic rules.** The structured model takes the number of slots, the reward weights and the way extra bookings empty the waiting list as known; if these are "
        "wrong, its policy is optimal for a different clinic.",
        f"**Assumed waiting list.** The table has no waiting list, so the real sessions use a simulated one with the arrival rate and reward weights of Table 1; Table {ext.T['t10']} "
        "varies two of these values, and others were not tested.",
        f"**Sessions from neighbourhood days.** A session is all appointments of one neighbourhood on one day; only {ext.pc(B['data']['share_not_full'], 0)} of these have fewer "
        "than 10 bookings, so the slots filled band rarely changes and adds little to the state.",
        "**Drift and calibration.** The classifier was calibrated before the no-show rate fell, so it overstates the risk in the scoring weeks (Section 7.8).",
        "**Baselines.** The baselines from published work are simplified implementations, not the authors' code, and their threshold and k were chosen on the scoring sessions.",
        "**Wait time.** Waiting is measured by the length of the waiting list; the table records no times within a day, so the time patients spend waiting in the clinic "
        "cannot be measured.",
        "**Simulated, not clinical, evidence.** All numbers come from stylised simulators written for this study, and on real sessions from a simulation built from recorded "
        "appointments with an assumed waiting list and reward. They demonstrate how the "
        "algorithms behave when the ground truth is known; they say nothing about the safety or benefit of any treatment in people.",
        "**State aggregation.** Continuous physiology is discretised into bands. Banding makes the process only approximately Markov, and part of "
        "the gap between the DP policy on the estimated model and the true optimum is a modelling gap rather than a sampling gap.",
        "**Fixed-horizon episodes treated as stationary.** Episodes end after a fixed number of steps but the state does not always contain the "
        "clock, so the estimated model treats termination as a constant hazard. This distorts absolute model values (not true-simulator "
        "scores) and is why model-based value estimates are compared only on a common oracle model.",
        "**Behaviour policy.** The logged data come from a single ε-noisy rule. Coverage of non-rule actions is therefore thin, which is exactly "
        "the offline-RL situation, and results depend on that coverage.",
        "**Hand-set hyper-parameters.** ε schedules, step sizes and the support threshold were fixed in advance and not tuned per paper.",
    ])


def ethics_future_v2(P, future):
    P.h1("Ethical considerations and future work")
    P.p("**Ethics.** The simulator study used no human data. The real sessions come from a public appointment table [@kaggle] in which patients appear only as numeric "
        "identifiers; it was used as published, was not linked to any other source, and is not redistributed with the code. If the method were used for real bookings, three safeguards would be essential: "
        "de-identification and governance approval of the source data; a clinician-in-the-loop design in which the learned policy is advisory, with hard guardrails on unsafe actions; "
        "and prospective, randomised or at least carefully monitored evaluation before any deployment. Reinforcement learning from logs inherits the biases of the clinicians who "
        "produced them, and may under-serve groups that are under-represented in the data; subgroup performance should be reported alongside averages, including the waiting times "
        "of groups with a higher predicted no-show risk, on whom overbooking decisions may fall more often.")
    P.p("**Future work.** " + future)


def repro_v2(P, script, figs):
    P.h1("Appendix B. Reproducibility")
    P.p(f"All results of Sections 7.1 to 7.4 are produced by `python ideas.py {script}` (shared library: rl_core.py, domains.py, common.py). Random "
        "seeds are fixed in the code, so a re-run reproduces every table and figure. Figure files generated by the script: " + ", ".join(figs) + ". "
        "To repeat the simulator study on a real table of logged decisions, convert it with kaggle_adapter.transitions_from_dataframe, then call build_mdp, "
        "value_iteration, mc_control and q_learning exactly as the script does; because a real table has no simulator, report model-based "
        "estimates on a held-out split and label them as estimates.")


def code_appendix_v2(P, idea_no, extra_funcs=()):
    import inspect, ideas
    P.h1("Appendix C. Source code of the experiment")
    P.p("For completeness the exact source of the simulator, the experiment function and the structured model is reproduced below; it is the same code that produced every number "
        "and figure in this paper (the shared solvers value_iteration, mc_control and q_learning are in rl_core.py, and the experiment script of Sections 7.5 to 7.10, "
        "run_topic7.py, is delivered with the code package).")
    for obj in list(extra_funcs):
        P.h3(getattr(obj, "__name__", "code"))
        P.code(inspect.getsource(obj))
    P.h3(f"ideas.idea{idea_no}")
    P.code(inspect.getsource(getattr(ideas, f"idea{idea_no}")))



I = 7


def build():
    ext.MODE = "v2"
    R7 = ext.res(); A7, B7 = R7["A"], R7["B"]; M7, MB7 = A7["methods"], B7["methods"]
    g7 = [B7["demand"][k]["gain_vs_one_session_pct"] for k in B7["demand"]]
    d7 = [A7["paired"][f"Ours: DP vs {k}"]["mean"] for k in ("Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning")]
    x0 = list(A7["data_size"])[0]
    r = load(I); S = r["standard"]; E = r["extra"]; meth = E["methods"]; cfg = E["config"]
    dom = domains.Clinic()
    SCH = r["schedules"]; ALG = ("Q-learning", "Monte Carlo")
    Jopt, Jbeh = r["J_opt"], r["J_beh"]
    nrep = cfg["seeds"]; nS = len(r["t90"]["Q-learning"][SCH[0]])      # replicates in the benchmark / in the schedule study
    tf5 = float(stats.t.ppf(0.975, nrep - 1) / 1.96)
    tf6 = float(stats.t.ppf(0.975, nS - 1) / 1.96)
    rule_J = S["Rule"]["J"]; orc_J = S["Oracle DP"]["J"]
    nst = E["n_states"]

    # ---- benchmark statistics (paired over replicates against the rule)
    def pvr(k, ref="Rule"):
        d = np.array(meth[k]["J"]) - np.array(meth[ref]["J"])
        return float(d.mean()), 1.96 * float(d.std(ddof=1) / np.sqrt(len(d))), float(stats.ttest_1samp(d, 0).pvalue), int((d > 0).sum())
    pv = {k: pvr(k) for k in ("DP", "Monte Carlo", "Q-learning")}
    dq, dm, mq = pvr("DP", "Q-learning"), pvr("DP", "Monte Carlo"), pvr("Monte Carlo", "Q-learning")
    gap_orc = orc_J - rule_J
    inf = {k: S[k]["info"] for k in S}
    sdQ = float(np.std(meth["Q-learning"]["J"], ddof=1)); sdM = float(np.std(meth["Monte Carlo"]["J"], ddof=1))
    Jdp = meth["DP"]["J"]; n_dp_eq_beh = sum(1 for a, b in zip(Jdp, meth["Behavior"]["J"]) if abs(a - b) < 1e-9)

    # ---- model-scale (oracle MDP) numbers
    Jor = {k: float(np.mean(meth[k]["Jor"])) for k in meth}
    gaps = [float(np.mean(meth[k]["J"]) - np.mean(meth[k]["Jor"])) for k in meth]          # true minus oracle-model value
    og = np.array(E["optimism_gap"])
    cq, cm = np.array(E["curves"]["Q-learning"]), np.array(E["curves"]["Monte Carlo"])
    K = cq.shape[1]
    q0, m0 = float(cq[:, 0, 1].mean()), float(cm[:, 0, 1].mean())
    q_last, m_last = float(cq[:, -1, 1].mean()), float(cm[:, -1, 1].mean())
    v0 = (q0 - Jbeh) / (Jopt - Jbeh); offset = v0 / K
    steps_ck = cq[0, :, 0]
    # first checkpoint at which the mean curve is within 1 model unit of its final value
    def settle(c):
        mc = c[:, :, 1].mean(0)
        return float(steps_ck[np.flatnonzero(np.abs(mc - mc[-1]) <= 1.0)[0]])
    set_q, set_m = settle(cq), settle(cm)
    dJor = {k: float(np.mean(meth[k]["Jor"]) - Jbeh) for k in ("DP", "Monte Carlo", "Q-learning")}
    n_neg_dJor = {k: int(sum(1 for x in meth[k]["Jor"] if x < Jbeh)) for k in ("DP", "Monte Carlo", "Q-learning")}

    # ---- policies of replicate 0
    pol = E["policies"]
    n_diff = {k: int(np.sum(np.array(pol[k]) != np.array(pol["Rule"]))) for k in pol if k != "Rule"}
    n_diffb = {k: int(np.sum(np.array(pol[k]) != np.array(pol["Behavior"]))) for k in ("DP", "Monte Carlo", "Q-learning")}
    n_act = {k: np.bincount(pol[k], minlength=dom.nA).tolist() for k in pol}
    agQ, agM = 100 * np.mean(E["agree"]["Q-learning"]), 100 * np.mean(E["agree"]["Monte Carlo"])
    p_off = dom.behavior_eps / dom.nA                      # probability of each specific off-rule action in the logs
    per_pair = cfg["n_eps"] * dom.max_steps / (dom.nS) * p_off          # expected visits to an off-rule pair if visits were spread evenly
    per_pair_or = cfg["n_oracle"] * dom.max_steps / (dom.nS) * p_off

    # ---- schedule study
    AUC, FIN, SH = r["auc"], r["final"], r["shift"]
    n_reach = int(round(sum(r["reach_rate"][a][s] * nS for a in ALG for s in SCH)))
    n_runs = len(ALG) * len(SCH) * nS
    best_auc = {a: max(SCH, key=lambda s: AUC[a][s]["mean"]) for a in ALG}
    worst_auc = {a: min(SCH, key=lambda s: AUC[a][s]["mean"]) for a in ALG}
    best_nom = {a: max(SCH, key=lambda s: SH[a][s]["nominal"]) for a in ALG}
    worst_nom = {a: min(SCH, key=lambda s: SH[a][s]["nominal"]) for a in ALG}
    def sep(a, s1, s2, tab=AUC):               # non-overlap of t-widened intervals (a screening rule, not a test)
        return abs(tab[a][s1]["mean"] - tab[a][s2]["mean"]) > (tab[a][s1]["ci"] + tab[a][s2]["ci"]) * tf6
    fin_sig = [(a, s) for a in ALG for s in SCH if abs(FIN[a][s]["mean"]) > FIN[a][s]["ci"] * tf6]
    max_fin = max(FIN[a][s]["mean"] for a in ALG for s in SCH)
    max_fin_key = max(((a, s) for a in ALG for s in SCH), key=lambda k: FIN[k[0]][k[1]]["mean"])
    auc_rng = {a: (min(AUC[a][s]["mean"] for s in SCH), max(AUC[a][s]["mean"] for s in SCH)) for a in ALG}
    nearQ_sep = [s for s in SCH if s != "Near-greedy (0.02)" and sep("Q-learning", "Near-greedy (0.02)", s)]
    nearM_sep = [s for s in SCH if s != "Near-greedy (0.02)" and sep("Monte Carlo", "Near-greedy (0.02)", s)]
    qopt = 14.0 * (1 - dom.gamma ** dom.max_steps) / (1 - dom.gamma); q_init = 0.5 * qopt
    rho_q = float(stats.spearmanr([AUC["Q-learning"][s]["mean"] for s in SCH], [SH["Q-learning"][s]["nominal"] for s in SCH]).statistic)
    rho_m = float(stats.spearmanr([AUC["Monte Carlo"][s]["mean"] for s in SCH], [SH["Monte Carlo"][s]["nominal"] for s in SCH]).statistic)
    rho_qm = float(stats.spearmanr([SH["Q-learning"][s]["nominal"] for s in SCH], [SH["Monte Carlo"][s]["nominal"] for s in SCH]).statistic)
    nom = {a: [SH[a][s]["nominal"] for s in SCH] for a in ALG}
    spread = {a: max(nom[a]) - min(nom[a]) for a in ALG}
    # noise gauge: the same algorithm with the eps-exp schedule appears in the benchmark (data seeds 1000-1004, 300 eval episodes)
    # and in the schedule study (data seeds 700-705, 120 eval episodes)
    gauge = {a: SH[a]["eps exp. decay"]["nominal"] - S[a]["J"] for a in ALG}

    # ---- shift
    sref = r["shift_ref"]; scen = list(sref.keys())            # nominal, +0.08, -0.05
    nomk, upk, dnk = scen
    labs = list(sref[nomk].keys())                             # Rule, DP, Q (eps exp), MC (eps exp)
    allrows = {"Rule": {k: sref[k]["Rule"] for k in scen}, "DP": {k: sref[k]["DP"] for k in scen}}
    for a in ALG:
        for s in SCH:
            allrows[f"{a}|{s}"] = SH[a][s]
    ruleS = allrows["Rule"]
    cells = [(a, s) for a in ALG for s in SCH]
    n_above = {k: sum(1 for a, s in cells if SH[a][s][k] > ruleS[k]) for k in scen}
    gain_dn = {(a, s): SH[a][s][dnk] - ruleS[dnk] for a, s in cells}
    gain_nom = {(a, s): SH[a][s][nomk] - ruleS[nomk] for a, s in cells}
    drop = {k: v[dnk] - v[nomk] for k, v in allrows.items()}
    up_vals = [v[upk] for v in allrows.values()]
    up_learn = [SH[a][s][upk] for a, s in cells]
    up_dev = [x - ruleS[upk] for x in up_learn]
    rule_up = ruleS[upk] - ruleS[nomk]

    oth = [x for x in SCH if x != "Near-greedy (0.02)"]
    n_sepQ_oth = sum(1 for i, a in enumerate(oth) for b in oth[i + 1:] if sep("Q-learning", a, b))
    eqval = next((a for a, b in zip(Jdp, meth["Behavior"]["J"]) if abs(a - b) < 1e-9), float("nan"))
    # guards for the qualitative wording below: the build fails loudly if the data no longer support it
    assert pv["Monte Carlo"][2] < 0.05 < pv["Q-learning"][2] and pv["DP"][2] > 0.1 and all(v[0] > 0 for v in pv.values())
    assert n_reach == 0 and not fin_sig and rho_q > 0.99 and rho_m < 0.5
    assert best_nom["Q-learning"] == "eps exp. decay" and worst_nom["Monte Carlo"] == "eps exp. decay"
    assert n_above[dnk] == len(cells) and max(abs(x) for x in up_dev) < 1.0
    assert all(v < 0 for v in drop.values()) and all(drop[f"{a}|{s}"] > drop["Rule"] for a, s in cells)
    assert og.max() < 0 and max(dJor.values()) < 0 and nearQ_sep and not nearM_sep
    assert best_auc["Q-learning"] == "eps exp. decay" and worst_auc["Q-learning"] == "Near-greedy (0.02)"
    P = Paper("Overbooking Clinic Appointments with Reinforcement Learning: Exploration Schedules, Learning Efficiency and Robustness to Shifts in the No-Show Rate",
              "appointment scheduling; no-show; overbooking; exploration schedule; epsilon-greedy; optimistic initialisation; learning curves; distribution shift; structured model; real appointment data",
              running="Clinic overbooking, exploration and shift", refdict=REFS)
    ext.add_refs(P)

    P.abstract(
        "Patients who miss appointments leave clinic slots empty, and booking more patients than slots recovers capacity at the risk of overtime and long waits. "
        f"We model a clinic that decides before each session how many extra patients to book as a {nst}-state, {dom.nA}-action Markov decision process on a simulator whose variables mirror the public 'Medical Appointment No Shows' table, "
        "and we rebuild real clinic sessions from that table to test the methods on recorded attendance. "
        f"From {cfg['n_eps']} logged clinic days we learn policies with dynamic programming (DP), Monte Carlo (MC) control and Q-learning, compare five exploration schedules (constant epsilon, exponential decay, a 1/k decay, optimistic initialisation, near-greedy), "
        "and score the final policies in the true simulator, including two shifts of the no-show rate. "
        f"In the true simulator DP, MC and Q-learning scored {S['DP']['J']:.1f}, {S['Monte Carlo']['J']:.1f} and {S['Q-learning']['J']:.1f} against {rule_J:.1f} for the overbooking rule, a gain of {pv['DP'][0]:+.1f} ({pstr(pv['DP'][2])}), {pv['Monte Carlo'][0]:+.1f} ({pstr(pv['Monte Carlo'][2])}) and {pv['Q-learning'][0]:+.1f} ({pstr(pv['Q-learning'][2])}) over {nrep} replicates, "
        f"whereas a DP policy fitted to ten times as much data scored {orc_J:.1f}. "
        f"On the fitted model no schedule reached 90% of the optimal improvement in any of {n_runs} runs, and the schedule differences that can be resolved are small: the near-greedy schedule was clearly slowest for Q-learning "
        f"(normalised area under the learning curve {AUC['Q-learning']['Near-greedy (0.02)']['mean']:.2f} against {AUC['Q-learning'][best_auc['Q-learning']]['mean']:.2f} for the best schedule), but for MC it could not be separated from the others. "
        f"The ranking of schedules by final true return differed between the two algorithms (rank correlation {rho_qm:+.2f} over five schedules). "
        f"When the no-show rate fell by 0.05, all {n_above[dnk]} learned configurations scored above the rule, by {np.mean(list(gain_dn.values())):.1f} units on average against {np.mean(list(gain_nom.values())):.1f} at nominal rates; when it rose by 0.08, they were within {max(abs(x) for x in up_dev):.1f} units of it. "
        f"The limit is the coverage of the logged data, and a structured model that learns only patient behaviour and takes the clinic's rules as known removes it: solved by DP, it scored "
        f"{M7['Ours: DP']['J']['mean']:.1f} with the same {cfg['n_eps']} days ({ext.sg(A7['paired']['Ours: DP vs Sir: DP']['mean'], 1)} over DP on the estimated MDP, p < 0.001, {R7['config']['seeds']} replicates) "
        f"and {A7['data_size'][x0]['Ours: DP']['mean']:.1f} with {x0} days. On {B7['data']['sessions_future']} real clinic sessions it scored {MB7['Ours: DP']['J']['mean']:.1f} against "
        f"{MB7['Rule']['J']['mean']:.1f} for the rule, {MB7['Sir: DP']['J']['mean']:.1f} for DP on the estimated MDP and {MB7['One-session optimiser']['J']['mean']:.1f} for a single session optimiser, "
        f"with a margin over the latter that grows to {g7[-1]:.0f}% when waiting is costlier, and Q-learning came closer to its DP solution than MC. No clinical claims are made.")

    P.h1("Introduction")
    P.p("Missed appointments are costly: a slot reserved for a patient who does not come is lost capacity, and clinics have long responded by overbooking, that is, giving more appointments than slots. "
        "Overbooking trades the value of filled slots against the cost of overtime when everyone arrives and of waiting for those who are not seen [@gupta,zacharias]. "
        "The right amount depends on the no-show rate, which varies with the weekday, the season and the patient mix and drifts over time.")
    P.p("Reinforcement learning (RL) can learn an overbooking rule from logged clinic data without an explicit model of no-show behaviour [@sb]. Three practical questions follow. "
        "First, how quickly and how reliably do standard learners converge, and does the exploration schedule matter? The textbook answer is that it does, because an agent that never tries an action cannot learn its value [@sb,singh00,rmax]. "
        "Second, a policy learned from a log is deployed in a clinic whose no-show rate may differ from that of the log, so how robust is it? Third, if the logs are the limit, can they be used more efficiently, and do the answers hold on real appointments? "
        "This paper studies the first two in a transparent simulator and the third both in the simulator and on clinic sessions rebuilt from a real appointment table.")
    P.p("A feature of the offline setting changes the first question. When the learner is trained on a model fitted from logs, its exploration happens inside that model, and a state-action pair that the logs never contain cannot be discovered by exploring harder. "
        "The exploration schedule then governs how fast the fitted model is solved rather than what information is collected. Part of our contribution is to measure how much the schedule matters under that condition and to report the answer even when it is small. "
        "The same condition points to a remedy: if the clinic's own rules are known, only patient behaviour has to be learned, and every logged session informs the same few quantities.")
    P.p("**Contributions.** (i) A clinic overbooking MDP with a weekday cycle, a no-show state, a waiting list and an operational reward (Section 4.1), and clinic sessions rebuilt from a real appointment table (Section 4.2). "
        "(ii) A normalised learning-efficiency metric, the area under the learning curve and the steps to 90% of the optimal improvement, defined against the clinic's own behaviour and the oracle optimum (Eq. 3 to 5). "
        f"(iii) A controlled comparison of {len(SCH)} exploration schedules for two learners with {nS} replicates each. "
        "(iv) A shift study that evaluates the trained policies when the no-show rate is higher or lower than in the log (Eq. 6). "
        "(v) Candid negative results: no schedule reaches 90%, the fitted-model metric does not agree with true return, and schedule effects on true return are of the same size as the run-to-run noise. "
        "(vi) A structured model in which only patient behaviour is learned (Section 5.2); it removes the data coverage limit in the simulator, and on real sessions it is ahead of the learners "
        "and of baselines from published work (Sections 7.5 to 7.10), where we also compare the convergence of MC and Q-learning.")
    P.p("**Scope and honesty.** The Kaggle site could not be reached when the simulator study was built (all requests returned HTTP 403), so Sections 7.1 to 7.4 analyse no real appointment table. The simulator mirrors the variables of such a table (appointment day, no-show indicator, waiting time) with made-up dynamics. "
        "The table was obtained later and is used to rebuild real clinic sessions (Section 4.2); those sessions keep a simulated waiting list and reward. "
        "Findings concern algorithms, not clinics, and no clinical claim is made.")

    P.h1("Related work")
    P.p("**Appointment scheduling and overbooking.** Operations research has studied appointment systems extensively, including no-shows, overbooking and the trade-off between patient waiting and provider overtime [@gupta]; "
        "analytical results for overbooking with no-shows are given by Zacharias and Pinedo [@zacharias]. Predicted no-show probabilities have been combined with overbooking decisions for a single session, "
        "for example by marginal analysis [@topuz], and fixed risk thresholds for double booking serve as the baseline for learned policies [@amalina]. "
        "Our contribution is not a better scheduling model but a study of how RL learners behave on one, and of how much they gain from a model that knows the clinic's rules.")
    P.p("**Exploration in RL.** Epsilon-greedy exploration, decaying schedules and optimistic initial values are the textbook devices [@sb]. Convergence of on-policy methods needs schedules that are greedy in the limit with infinite exploration (GLIE) [@singh00]; "
        "Q-learning converges under visit-count conditions on the step size [@watkins,evendar]. Model-based methods such as R-max make optimism explicit [@rmax]. "
        "Our schedules are the simple ones a practitioner would try; the 'GLIE-style' schedule here has a floor and is therefore not strictly GLIE.")
    P.p("**Offline RL and shift.** Learning from logs is limited by data coverage and is sensitive to model error, which motivates pessimism and support constraints [@levine,jin,kumar]. Robustness to a change in the environment after deployment is a related but different concern, "
        "and we treat it with the simplest possible experiment: re-evaluating the trained policies in simulators with a shifted no-show rate. "
        "Known structure is an alternative to more data: in a factored MDP the transition model is assembled from separately specified parts of the state [@boutilier], "
        "which is the idea behind the structured model of Section 5.2.")
    background(P)

    P.h1("Problem formulation: a clinic overbooking simulator and real clinic sessions")
    P.h2("The simulator")
    P.p(f"A clinic runs {dom.max_steps} sessions per episode, each with {dom.CAP} regular slots. Before a session the clinic books {dom.CAP} + k patients, where k in {{0, 1, 2, 3, 4}} is the number of extra bookings. "
        f"Each booked patient shows up independently with probability 1 − p, where p is the current no-show rate. The number who show is binomial. "
        f"Overtime is the number of patients above {dom.CAP}, idle is the number of unused slots, and a waiting list of length w (0 to 12) grows by a Poisson({1.5}) number of patients per session and shrinks by min(w, k), the number of extra bookings drawn from it. "
        f"The no-show rate starts near {0.18} (normal with standard deviation 0.03) and relaxes towards a weekday-specific mean: p moves to 0.8 p + 0.2 μ_d plus Gaussian noise with standard deviation 0.02, where the means for the five weekdays are {', '.join(f'{x:.2f}' for x in dom.MU)}. "
        f"The reward per session is shows − 3 × overtime − 0.5 × idle − 0.4 × waiting-list length + 0.3 × (patients drawn from the waiting list). Rewards are discounted with γ = {dom.gamma}.")
    P.table([["Element", "Definition"],
             [f"State ({nst})", f"weekday (5) × no-show band (4: p below {dom.NE[0]}, {dom.NE[0]} to {dom.NE[1]}, {dom.NE[1]} to {dom.NE[2]}, above {dom.NE[2]}) × waiting-list band (5: lengths 0, 1-2, 3-5, 6-8, 9 or more)"],
             [f"Actions ({dom.nA})", f"book {dom.CAP} + k patients, k = 0 to {dom.nA - 1} (labels {', '.join(dom.action_labels)})"],
             ["Reward per session", "shows − 3 overtime − 0.5 idle − 0.4 waiting list + 0.3 drawn from waiting list"],
             ["Discount, horizon", f"γ = {dom.gamma}; {dom.max_steps} sessions per episode"],
             ["Overbooking rule", "book one extra patient when the no-show band is 0 or 1 (p below 0.18), otherwise two extra"],
             ["Behaviour policy", f"the rule with ε = {dom.behavior_eps} uniformly random actions"],
             ["Dataset mirrored", "Medical Appointment No Shows (appointment date, no-show flag, wait)"]],
            "The clinic overbooking MDP.", widths=[3.6, 12.8], font=8.5)
    P.p(f"Three design features matter for what follows. First, the rule only ever uses k = 1 or 2, and the behaviour policy picks each of the other actions with probability ε / {dom.nA} = {p_off:.2f} per session, so the logs contain few examples of overbooking by 3 or 4. "
        f"If visits were spread evenly over the {dom.nS} states, a given off-rule state-action pair would be seen about {per_pair:.1f} times in {cfg['n_eps']} logged episodes and about {per_pair_or:.0f} times in {cfg['n_oracle']:,}; since pairs seen fewer than 10 times are treated as unsupported, "
        "the small dataset cannot support most of the actions the optimum uses, while the large one can. (Visits are not spread evenly, so this is an order-of-magnitude argument.) "
        "Second, the waiting list rewards overbooking beyond the show-rate trade-off, because extra bookings drain it. Third, the no-show band is part of the state, so a policy can in principle adapt to a shifted no-show rate if it has learned what each band calls for.")
    ext.real_data(P)

    P.h1("Method: exploration schedules, shift evaluation and a structured model")
    P.h2("Exploration schedules, normalised learning efficiency and shift evaluation")
    P.p(f"Both learners are trained on a model fitted to the logged episodes and sample experience from that model (never from the simulator). Let k be the step count and T the budget ({cfg['steps']:,} steps). The five schedules are")
    P.eq("constant: ε_k = 0.1;   exponential: ε_k = 0.05 + 0.95 exp(−5k/T);   1/k-type: ε_k = max(0.02, c / (c + k)),  c = T / 50;   near-greedy: ε_k = 0.02,", "3")
    P.p(f"and the optimistic-initialisation schedule uses ε_k = 0.02 together with an initial action value q0 = 0.5 × 14 × (1 − γ^H) / (1 − γ) = {q_init:.1f}, where 14 is an assumed upper bound on the reward per session and H = {dom.max_steps}. "
        f"With c = {cfg['steps'] // 50:,} the 1/k-type schedule equals 0.5 at k = c and reaches its floor only near the end of the budget; because of the floor it is not strictly GLIE [@singh00]. "
        "MC control uses first-visit returns with incremental averaging; Q-learning uses a visit-count step size [@watkins,evendar].")
    P.p("Learning curves are measured by recording, at 26 evenly spaced checkpoints, the value of the greedy policy on a large 'oracle' model (Section 7.1 explains what that value means). The raw values depend on the scale of the problem, so we normalise them against the clinic's own behaviour and against the optimum of the same model:")
    P.eq("v(π) = [J(π) − J_beh] / [J_opt − J_beh],", "4")
    P.p("so that v = 0 is the behaviour policy and v = 1 the model optimum. We summarise a learning curve by its mean over the checkpoints and by the first checkpoint reaching 90% of the possible improvement:")
    P.eq("AUC = (1 / K) Σ_j v(π_{t_j}),      t90 = min { t_j : v(π_{t_j}) ≥ 0.9 },      K = 26.", "5")
    P.p("The AUC rewards fast learning and penalises a slow start, and t90 asks whether the optimum is approached at all. The evaluation under shift is a re-simulation in which the no-show rate is displaced by δ at the start of an episode and in every weekday mean:")
    P.eq("p_0 ← p_0 + δ,      μ_d ← μ_d + δ,      δ ∈ { 0, +0.08, −0.05 }.", "6")
    P.p("The policy, trained on nominal data, is not changed. Because the state contains the no-show band, a shift moves the visited states towards other bands, some of which are rarely seen in the log.")
    P.code("""Algorithm 1  Exploration-schedule study
 input : logged clinic days D (150 episodes), schedule list, learners {Q-learning, MC}, budget T, shift list
 1. fit the MDP from D (pairs seen < 10 times are unsupported; unsupported pairs are absorbing with the worst reward)
 2. for each schedule and each learner:
      run the learner for T steps in the fitted MDP with eps_k (and q0); every T/25 steps record J(greedy policy) on the oracle MDP
      fill unobserved states with the logged modal action; evaluate the final policy in each shifted simulator (120 episodes)
 3. normalise curves with Eq. (4); compute AUC and t90 with Eq. (5)
 4. repeat over independent data draws; report means and 95% intervals""")
    ext.method(P)

    protocol_v2(P, [["Item", "Setting"],
                 ["Simulator", f"clinic overbooking MDP; {nst} states; {dom.nA} actions; {dom.max_steps} sessions per episode"],
                 ["Benchmark", f"{cfg['n_eps']} logged episodes; oracle MDP from {cfg['n_oracle']:,} episodes; {cfg['seeds']} replicates; {cfg['n_eval']} evaluation episodes; {cfg['steps']:,} steps"],
                 ["Learners", f"DP (value and policy iteration agreed in all replicates: {'yes' if all(E['pi_equals_vi']) else 'no'}), first-visit MC, Q-learning"],
                 ["Schedule study", f"{len(SCH)} schedules × 2 learners × {nS} replicates (data seeds 700-705); {cfg['steps']:,} steps; 26 checkpoints; reference values J_beh = {Jbeh:.2f}, J_opt = {Jopt:.2f} on the oracle MDP"],
                 ["Shift study", "no-show offsets 0, +0.08, -0.05; 120 evaluation episodes per run (300 for the rule and the DP policy)"],
                 ["Statistics", f"95% CI = 1.96 standard errors over replicates; where a significance statement is made the interval is widened to a t-interval (factor {tf5:.2f} for {nrep} replicates, {tf6:.2f} for {nS})"],
                 ["Structured model", f"DP, MC and Q-learning on the model of Section 5.2; {R7['config']['seeds']} replicates (data seeds 1000 to {999 + R7['config']['seeds']}); {R7['config']['n_eval_A']} evaluation episodes"],
                 ["Real sessions", f"{B7['data']['sessions_past']} learning and {B7['data']['sessions_future']} scoring sessions (Section 4.2); {R7['config']['n_eval_B']:,} scoring episodes; three baselines from published work (Section 6.3)"]],
             cfg["steps"], cfg["seeds"], cfg["n_eval"],
             extra="In the schedule study the standard benchmark's exponential schedule is one of the five. Final policies in the schedule study use the logged modal action in states with no data.")
    ext.protocol(P)

    P.h1("Results")
    P.h2("Benchmark in the true simulator")
    rows = std_rows(S, ("shows", "overtime", "idle", "wait"), ("Shows", "Overtime", "Idle", "Waiting"), nd=1)
    t1 = P.table(rows, f"Benchmark in the true simulator ({nst} states). Return is discounted; shows, overtime, idle slots and the summed waiting-list length are totals per {dom.max_steps}-session episode ({cfg['seeds']} replicates, 95% CI; rule and oracle are single policies).",
                 widths=[5.0, 3.4, 1.8, 1.9, 1.8, 1.9], font=8.5)
    P.p(f"Table {t1} shows that the learners are ahead of the rule on average but not by much, and not consistently. DP scored {S['DP']['J']:.2f} ± {S['DP']['ci']:.2f}, MC {S['Monte Carlo']['J']:.2f} ± {S['Monte Carlo']['ci']:.2f} and Q-learning {S['Q-learning']['J']:.2f} ± {S['Q-learning']['ci']:.2f} against {rule_J:.2f} for the rule. "
        f"Paired over replicates the differences from the rule are {pv['DP'][0]:+.2f} ± {pv['DP'][1]:.2f} ({pstr(pv['DP'][2])}; ahead in {pv['DP'][3]} of {nrep}) for DP, {pv['Monte Carlo'][0]:+.2f} ± {pv['Monte Carlo'][1]:.2f} ({pstr(pv['Monte Carlo'][2])}; {pv['Monte Carlo'][3]} of {nrep}) for MC and {pv['Q-learning'][0]:+.2f} ± {pv['Q-learning'][1]:.2f} ({pstr(pv['Q-learning'][2])}; {pv['Q-learning'][3]} of {nrep}) for Q-learning. "
        f"With {nrep} replicates a t-interval is {tf5:.2f} times wider than the 1.96-standard-error interval shown, so the p-values (which use the t distribution) are the safer guide: "
        f"only MC is below 0.05, Q-learning is borderline, and DP is not distinguishable from the rule. Among the learners no difference is resolved either (DP minus Q-learning {dq[0]:+.2f} ± {dq[1]:.2f}, {pstr(dq[2])}; DP minus MC {dm[0]:+.2f} ± {dm[1]:.2f}, {pstr(dm[2])}; MC minus Q-learning {mq[0]:+.2f} ± {mq[1]:.2f}, {pstr(mq[2])}). "
        f"The replicate-to-replicate standard deviations are {sdQ:.1f} for Q-learning and {sdM:.1f} for MC, which is the scale of the noise to keep in mind below. "
        f"In {n_dp_eq_beh} of {nrep} replicates the DP policy scored exactly as the logged modal-action policy ({eqval:.2f}), consistent with it having reproduced the behaviour policy there.")
    P.p(f"The oracle DP, fitted to {cfg['n_oracle']:,} episodes, scored {orc_J:.2f}, which is {gap_orc:.1f} above the rule and {orc_J - S['Monte Carlo']['J']:.1f} above the best learner. This is the opposite of the finding in a problem where data volume does not matter: here the headroom is large and the learners trained on {cfg['n_eps']} episodes capture only part of it "
        f"({100 * pv['Monte Carlo'][0] / gap_orc:.0f}% for MC, {100 * pv['Q-learning'][0] / gap_orc:.0f}% for Q-learning, {100 * pv['DP'][0] / gap_orc:.0f}% for DP). The operational columns show how the oracle gets there: it books more patients "
        f"({inf['Oracle DP']['shows']:.0f} shows per episode against {inf['Rule']['shows']:.0f} for the rule), accepts about {inf['Oracle DP']['overtime'] / inf['Rule']['overtime']:.1f} times as much overtime ({inf['Oracle DP']['overtime']:.1f} against {inf['Rule']['overtime']:.1f}), has fewer idle slots ({inf['Oracle DP']['idle']:.1f} against {inf['Rule']['idle']:.1f}) and, above all, a much shorter waiting list ({inf['Oracle DP']['wait']:.0f} against {inf['Rule']['wait']:.0f}). "
        f"The learners move in the same direction but only a little: MC reduces the summed waiting list to {inf['Monte Carlo']['wait']:.0f} and Q-learning to {inf['Q-learning']['wait']:.0f}, with overtime of {inf['Monte Carlo']['overtime']:.1f} and {inf['Q-learning']['overtime']:.1f}.")
    f1 = P.fig(fig(I, "fig_learning"), "Learning curves on the oracle MDP (mean ± s.d. over 5 replicates). Dashed: oracle DP optimum; dotted: rule-based (modal behaviour) policy.", 7.0)
    P.p(f"Figure {f1} shows the learning curves on the oracle model, which is the scale used for all efficiency metrics. Both learners start at {q0:.1f} (the greedy policy of an all-zero value table) and rise to within one model unit of their final value by step {set_q:,.0f} (Q-learning) and {set_m:,.0f} (MC) of the {cfg['steps']:,}, "
        f"then stay flat at {q_last:.1f} and {m_last:.1f}. These plateaus match the mean DP policy value on the same model ({np.mean(meth['DP']['Jor']):.1f}) to within {max(abs(q_last - np.mean(meth['DP']['Jor'])), abs(m_last - np.mean(meth['DP']['Jor']))):.1f}, as expected when learning in a fitted model converges to its solution. "
        f"They lie below the rule's {Jbeh:.1f} and far below the optimum {Jopt:.1f}. The model-scale and true-simulator pictures therefore disagree in sign: on the model DP, MC and Q-learning are {abs(dJor['DP']):.1f}, {abs(dJor['Monte Carlo']):.1f} and {abs(dJor['Q-learning']):.1f} below the rule, "
        f"while in the true simulator they are {pv['DP'][0]:.1f}, {pv['Monte Carlo'][0]:.1f} and {pv['Q-learning'][0]:.1f} above it. In {sum(n_neg_dJor.values())} of {3 * nrep} replicate-learner cells the model value is below the rule's.")
    P.p(f"Understanding why requires looking at what the model-scale value measures. The oracle model treats a state-action pair seen fewer than 10 times as absorbing with the worst reward in the data, so a policy that chooses such an action is penalised on the model whatever it would do in the clinic. "
        f"All the model-scale values are far below their true values: the true return minus the oracle-model value ranges from {min(gaps):.1f} to {max(gaps):.1f} over the six methods, and the data model's value of its own DP policy was {abs(og.max()):.1f} to {abs(og.min()):.1f} below the oracle model's value of the same policy in every replicate (mean optimism gap {og.mean():.1f}). "
        "This is a code-level reading, not a tested explanation, but it implies that the model scale is a pessimistic and distorted version of the true scale; we use it only to compare learning speed.")
    f2 = P.fig(fig(I, "fig_policies"), "Extra bookings chosen in each state (replicate 0) by the behaviour policy, the three learners and the oracle. States are sorted by weekday, then no-show band, then waiting-list band.", 9.0)
    P.p(f"Figure {f2} compares the policies for replicate 0. The rule uses +1 in {n_act['Rule'][1]} states and +2 in {n_act['Rule'][2]}, never anything else. The modal-action behaviour policy differs from it in {n_diff['Behavior']} states, and the learners in {n_diff['DP']} (DP), {n_diff['Monte Carlo']} (MC) and {n_diff['Q-learning']} (Q-learning) of {nst}; "
        f"they differ from the behaviour policy in only {n_diffb['DP']}, {n_diffb['Monte Carlo']} and {n_diffb['Q-learning']} states. The oracle differs from the rule in {n_diff['Oracle DP']} states and is the only policy that uses +3 ({n_act['Oracle DP'][3]} states) or +4 ({n_act['Oracle DP'][4]} states) with any frequency; "
        f"the learners use them in at most {max(n_act[k][3] + n_act[k][4] for k in ('DP', 'Monte Carlo', 'Q-learning'))} states. Action 0 appears in {n_act['Behavior'][0]} states of the behaviour policy, which is consistent with the default action for states absent from the data (default is 0), although we did not check that directly. "
        f"This pattern agrees with the support argument of Section 4: with {cfg['n_eps']} episodes the high-overbooking actions are mostly unsupported, so the learners cannot adopt them, and the benchmark gains of +2 to +6 come from changes to a handful of states. "
        f"Q-learning agreed with DP in {agQ:.0f}% and MC in {agM:.0f}% of the well-visited states, averaged over replicates. Because only replicate 0 is plotted and DP in that replicate scored {Jdp[0]:.1f}, below the rule's {rule_J:.1f}, the figure should not be read as showing the policies that earned the averages in Table {t1}.")

    P.h2("Exploration schedules on the fitted model")
    rows = [["Schedule", "Q-learning AUC", "Q-learning final", "MC AUC", "MC final", "Reached 90%"]]
    for s in SCH:
        nreach = int(round(sum(r["reach_rate"][a][s] * nS for a in ALG)))
        rows.append([s, pm(AUC["Q-learning"][s]["mean"], AUC["Q-learning"][s]["ci"], 2), pm(FIN["Q-learning"][s]["mean"], FIN["Q-learning"][s]["ci"], 2),
                     pm(AUC["Monte Carlo"][s]["mean"], AUC["Monte Carlo"][s]["ci"], 2), pm(FIN["Monte Carlo"][s]["mean"], FIN["Monte Carlo"][s]["ci"], 2), f"{nreach}/{2 * nS}"])
    t2 = P.table(rows, f"Exploration schedules: normalised area under the learning curve (AUC), normalised final value and the number of runs (of {2 * nS}) reaching 90% of the optimal improvement. "
                 f"Values are on the scale of Eq. (4): 0 is the clinic's behaviour and 1 the oracle-model optimum ({nS} replicates, 95% CI).", widths=[3.8, 2.7, 2.7, 2.7, 2.7, 1.8], font=8.5)
    P.p(f"Table {t2} is the central comparison of the schedule study. No run reached the 90% target in any of the {n_runs} runs, so the steps-to-90% metric is undefined for every schedule (the left panel of the efficiency figure in the code output is empty for this reason); "
        f"the best mean final value was {max_fin:+.3f} ({max_fin_key[0]}, {max_fin_key[1]}), so the 0.9 target is about {0.9 / max_fin:.0f} times larger. The reason is visible in Figure {f1}: the learners plateau at the value of the DP solution of the fitted model, which on the model scale is slightly below the rule's, so even a perfect solver of that model cannot approach 1. "
        f"The AUC values are all negative, from {auc_rng['Q-learning'][0]:.2f} to {auc_rng['Q-learning'][1]:.2f} for Q-learning and from {auc_rng['Monte Carlo'][0]:.2f} to {auc_rng['Monte Carlo'][1]:.2f} for MC. "
        f"A part of every AUC is a common offset: the first checkpoint (step 0) has v = {v0:.2f}, which contributes {offset:.3f} to the mean over {K} checkpoints in every run, so the offset is the same for all schedules and does not affect comparisons but does make the absolute AUC look worse than the learning actually is. "
        f"No final value differs from zero once the intervals are widened to t-intervals ({len(fin_sig)} of {2 * len(SCH)} do), so after {cfg['steps']:,} steps all schedules end at about the behaviour policy's model value, with the largest positive mean being {max_fin:+.3f}.")
    f3 = P.fig(fig(I, "fig_schedules"), "Normalised learning curves by exploration schedule (mean ± s.d. over 6 replicates). 0 is the clinic's behaviour and 1 the oracle-model optimum.", 11.0)
    P.p(f"Figure {f3} shows what separates the schedules: almost entirely the first steps and, for the near-greedy schedule, a slow climb. For Q-learning the near-greedy schedule has an AUC of {AUC['Q-learning']['Near-greedy (0.02)']['mean']:.2f} ± {AUC['Q-learning']['Near-greedy (0.02)']['ci']:.2f}, "
        f"against {AUC['Q-learning'][best_auc['Q-learning']]['mean']:.2f} ± {AUC['Q-learning'][best_auc['Q-learning']]['ci']:.2f} for the best schedule ({best_auc['Q-learning']}); its t-widened interval is separated from {len(nearQ_sep)} of the other 4 schedules. "
        f"This is the one robust schedule effect in the study, and it is the textbook one: a learner that explores little needs many more steps to learn the values of actions it rarely tries. The same schedule for MC (AUC {AUC['Monte Carlo']['Near-greedy (0.02)']['mean']:.2f} ± {AUC['Monte Carlo']['Near-greedy (0.02)']['ci']:.2f}) is separated from {len(nearM_sep)} of the other 4 schedules, "
        f"so for MC the evidence does not distinguish it from the rest. Among the other four schedules for Q-learning the AUC differences are between {min(abs(AUC['Q-learning'][a]['mean'] - AUC['Q-learning'][b]['mean']) for a in SCH for b in SCH if a != b and 'Near' not in a + b):.2f} and {max(abs(AUC['Q-learning'][a]['mean'] - AUC['Q-learning'][b]['mean']) for a in SCH for b in SCH if a != b and 'Near' not in a + b):.2f}, {'and none of these pairs is separated by the t-widened intervals' if n_sepQ_oth == 0 else 'and some pairs are separated'}. "
        f"Optimistic initialisation (q0 = {q_init:.0f}, ε = 0.02) gives AUC {AUC['Q-learning']['Optimistic init']['mean']:.2f} for Q-learning against {AUC['Q-learning']['eps=0.1 const']['mean']:.2f} for constant ε, and {AUC['Monte Carlo']['Optimistic init']['mean']:.2f} against {AUC['Monte Carlo']['eps=0.1 const']['mean']:.2f} for MC; with intervals of ±{AUC['Q-learning']['Optimistic init']['ci']:.2f} to ±{AUC['Monte Carlo']['Optimistic init']['ci']:.2f} these do not separate. "
        "Seen from the offline-RL angle this is unsurprising, because exploration inside a fitted model cannot reveal pairs absent from the logs: once the supported part of the model is sampled enough, every schedule converges to the same solution, and the schedule only changes how soon.")

    P.h2("Does the model-scale metric predict true return?")
    P.p(f"Learning efficiency is measured on the model scale, but what matters is true return. For each learner we ranked the five schedules by their AUC and by the true return of the final policy on nominal data (Table {t2 + 1}). "
        f"For Q-learning the two rankings agree exactly (Spearman correlation {rho_q:+.2f} over five schedules); for MC they hardly agree ({rho_m:+.2f}). With only five points no p-value is reported and the numbers are descriptive. "
        f"The best schedule by true return is {best_nom['Q-learning']} for Q-learning (true return {SH['Q-learning'][best_nom['Q-learning']]['nominal']:.1f}) and {best_nom['Monte Carlo']} for MC ({SH['Monte Carlo'][best_nom['Monte Carlo']]['nominal']:.1f}), "
        f"and the worst is {worst_nom['Q-learning']} ({SH['Q-learning'][worst_nom['Q-learning']]['nominal']:.1f}) and {worst_nom['Monte Carlo']} ({SH['Monte Carlo'][worst_nom['Monte Carlo']]['nominal']:.1f}). "
        f"The exponential decay schedule is the best for Q-learning and the worst for MC, and across the five schedules the true-return ranks of the two learners have a correlation of {rho_qm:+.2f}.")
    P.p(f"How much of this is signal? The same learner with the exponential schedule also appears in the benchmark, with different data draws (seeds 1000 to 1004 against 700 to 705) and more evaluation episodes ({cfg['n_eval']} against 120): Q-learning scored {S['Q-learning']['J']:.1f} there and {SH['Q-learning']['eps exp. decay']['nominal']:.1f} here ({gauge['Q-learning']:+.1f}), "
        f"and MC {S['Monte Carlo']['J']:.1f} there and {SH['Monte Carlo']['eps exp. decay']['nominal']:.1f} here ({gauge['Monte Carlo']:+.1f}). Based on the code these are the same configuration, so differences of about {max(abs(gauge['Q-learning']), abs(gauge['Monte Carlo'])):.0f} units arise from the data draws and the evaluation sample alone. "
        f"The spread of true returns across schedules is {spread['Q-learning']:.1f} for Q-learning and {spread['Monte Carlo']:.1f} for MC, which is {spread['Q-learning'] / max(abs(gauge['Q-learning']), abs(gauge['Monte Carlo'])):.1f} and {spread['Monte Carlo'] / max(abs(gauge['Q-learning']), abs(gauge['Monte Carlo'])):.1f} times that discrepancy, so the data do not establish that any schedule changes true return, and the opposite orderings of the two learners are consistent with noise.")

    P.h2("Robustness to a change in the no-show rate")
    names = {"Rule": "Overbooking rule", "DP": "DP (replicate 0)"}
    rows = [["Policy", "Nominal", "No-show +0.08", "No-show -0.05", "Change, -0.05 vs nominal"]]
    for k, v in allrows.items():
        lab = names.get(k, k.replace("|", ": "))
        rows.append([lab, f"{v[nomk]:.1f}", f"{v[upk]:.1f}", f"{v[dnk]:.1f}", f"{v[dnk] - v[nomk]:+.1f}"])
    t3 = P.table(rows, "True return of the final policies under a shifted no-show rate. Entries for Q-learning and MC are means over 6 replicates (120 evaluation episodes each); the rule is evaluated on 300 episodes and DP on one replicate and 300 episodes. No confidence intervals were stored for these entries.",
                 widths=[5.2, 2.2, 2.9, 2.9, 3.2], font=8.5)
    P.p(f"Table {t3} reports true returns when the no-show rate is higher or lower than in the log. Lower no-show rates are worse for everyone: the rule falls by {abs(drop['Rule']):.1f} units, DP by {abs(drop['DP']):.1f}, and the learners by between "
        f"{min(abs(v) for k, v in drop.items() if '|' in k):.1f} and {max(abs(v) for k, v in drop.items() if '|' in k):.1f}. A fall in no-shows means that the booking made for the nominal rate produces more overtime, so this is the direction in which overbooking policies are exposed. "
        f"The learned policies lose less than the rule: all {n_above[dnk]} of {len(cells)} learner-schedule cells are above the rule at -0.05, by {min(gain_dn.values()):.1f} to {max(gain_dn.values()):.1f} units, compared with {n_above[nomk]} of {len(cells)} at nominal rates (by {min(gain_nom.values()):.1f} to {max(gain_nom.values()):.1f}); the mean margin over the rule is {np.mean(list(gain_dn.values())):.1f} at -0.05 against {np.mean(list(gain_nom.values())):.1f} at nominal, so the margin grows when no-shows fall, although part of the margin already exists without any shift. "
        f"At +0.08 the picture is different: the rule gains {rule_up:.1f}, and every learned configuration lies within {max(abs(x) for x in up_dev):.1f} units of it (all entries between {min(up_vals):.1f} and {max(up_vals):.1f}), so the learned policies have no visible advantage or disadvantage when there are more no-shows.")
    f4 = P.fig(fig(I, "fig_shift"), "True return under the nominal and shifted no-show rates for the rule, DP and the exponentially decaying schedule of each learner.", 7.0)
    P.p(f"Figure {f4} shows the same for the four policies that the figure includes. The advantage of the learned policies at -0.05 is the largest result of the shift study ({gain_dn[('Q-learning', 'eps exp. decay')]:+.1f} for Q-learning, {sref[dnk]['DP'] - ruleS[dnk]:+.1f} for DP and {gain_dn[('Monte Carlo', 'eps exp. decay')]:+.1f} for MC relative to the rule), and it is consistent across all ten configurations, which gives some confidence it is not an accident of one schedule. "
        f"Two caveats limit it. First, no intervals were stored, and the benchmark noise gauge above (about {max(abs(gauge['Q-learning']), abs(gauge['Monte Carlo'])):.0f} units) is comparable to some of the advantages; the {n_above[dnk]} of {len(cells)} count is the more reliable evidence than any individual difference. "
        f"Second, the rule is a single policy with no learning noise while the learned entries average over replicates, so the comparison is not symmetric. We did not evaluate the oracle under shift, so we cannot say how far any of these policies is from the best possible policy in the shifted clinics, and we did not identify which states produce the advantage; the learned policies differ from the rule in only {n_diff['DP']} to {n_diff['Monte Carlo']} states in replicate 0 (Figure {f2}), so the effect must come from a small set of cells that the shift makes more frequent, which is a hypothesis.")
    ext.results(P)

    P.h1("Discussion")
    P.p(f"**What the study shows.** (1) Learners trained on {cfg['n_eps']} episodes are ahead of the overbooking rule in the true simulator by {min(v[0] for v in pv.values()):+.1f} to {max(v[0] for v in pv.values()):+.1f} units on average, with only MC clearly so ({pstr(pv['Monte Carlo'][2])}), while a model fitted to ten times the data is ahead by {gap_orc:+.1f}, which points to data coverage rather than the learner as the main barrier (Section 7.6 confirms it by varying the data volume). "
        f"(2) On the fitted model every schedule converges to about the same value, and none reaches 90% of the possible improvement ({n_reach} of {n_runs} runs). (3) The one clearly resolved schedule effect is that a near-greedy Q-learner learns more slowly. "
        f"(4) The learned policies are ahead of the rule at the nominal rate and by a larger average margin when no-shows fall ({np.mean(list(gain_dn.values())):.1f} against {np.mean(list(gain_nom.values())):.1f}; all {n_above[dnk]} of {len(cells)} configurations), and level with it when no-shows rise. "
        f"(5) The coverage barrier can be removed without more data: with the clinic's rules known and only patient behaviour learned, {x0} logged days gave {A7['data_size'][x0]['Ours: DP']['mean']:.1f}, "
        f"more than the oracle DP fitted to {cfg['n_oracle']:,} days ({orc_J:.1f}; Table {ext.T['t7']}). (6) The gain carries over to real patients: on sessions rebuilt from the Medical Appointment No Shows "
        f"table the structured model is ahead of every method in Table {ext.T['t9']}, by {B7['paired_seeds']['Ours: DP vs Sir: DP']['mean']:.2f} over DP on the estimated MDP and by "
        f"{B7['paired_episodes']['Ours: DP vs One-session optimiser']['mean']:.2f} over the strongest baseline from published work. (7) That margin is small in the default setting and grows to "
        f"{g7[-1]:.0f}% when the waiting list is under pressure (Table {ext.T['t10']}). (8) On the structured model Q-learning ends closer to the DP solution than MC in both comparisons.")
    P.p(f"**What it does not show.** It does not show that exploration schedules matter for final performance: the final model-scale values are all within {max(abs(FIN[a][s]['mean']) for a in ALG for s in SCH):.2f} of the behaviour policy's, and the true-return differences across schedules are comparable to the run-to-run discrepancy for the same configuration. "
        f"It does not show that the model-scale metric is a good proxy for true return; its sign disagrees with the true comparison, and its schedule ranking correlates with true return for Q-learning ({rho_q:+.2f}) but hardly for MC ({rho_m:+.2f}). "
        "It does not show which states drive the robustness result of Section 7.4. It does not show that a clinic knows its own rules as exactly as the structured model assumes: "
        "the slot count, the reward weights and the way extra bookings empty the waiting list are inputs, not estimates. It does not show a benefit to real patients: the real sessions "
        "replay recorded attendance, but the waiting list and the reward are still simulated. And the small margin over the single session optimiser in the default setting means that, "
        "in a quiet clinic, a good prediction model with a decision for one session at a time captures most of the gain.")
    P.p("**Practical use.** Because exploration inside a fitted model does not collect new data, practitioners should spend effort on what generates the log. A behaviour policy with deliberate variation in the number of extra bookings, run at low stakes, would give the learner support for the actions the optimum uses; the exploration schedule of the learner cannot substitute for that. "
        "A second way to reach the actions the optimum uses is to estimate less: whatever the clinic knows about its own rules need not be learned from the log. "
        "For deployment the shift study suggests a cheap sanity check: re-evaluate a learned policy against the existing rule under plausible no-show shifts and report the range, not the nominal value alone.")
    limitations_v2(P, [
        f"**Few replicates and no intervals for shift.** The schedule study uses {nS} replicates (t-interval factor {tf6:.2f}) and the shift table stores no intervals; the DP entry is one replicate.",
        "**Unequal evaluation sizes.** Rule and DP in the shift table use 300 evaluation episodes, the learners 120 per run; the benchmark and the schedule study use different data seeds.",
        f"**A model-scale metric.** Learning efficiency is measured on the oracle model, which is pessimistic by construction for pairs with fewer than 10 observations; the AUC also contains a common step-0 offset of {offset:.3f}. The ranking of schedules on this scale need not match the ranking in the clinic.",
        "**Exploration inside a model.** The schedules sample from the fitted MDP and cannot reveal unlogged state-action pairs, so the study cannot say anything about exploration in the real environment, where a clinic would face costs of trying.",
        "**Assumed no-show structure.** The shift is a constant offset to the no-show rate; real shifts also change who is booked and the waiting-list inflow. The simulator's reward weights (3 for overtime, 0.4 for waiting) are invented.",
    ])
    ethics_future_v2(P, "(1) Add an exploration-aware behaviour policy that deliberately varies the extra bookings and test whether the learners then close the gap to the oracle; (2) store per-replicate learning curves and shift results to attach intervals and paired tests; "
                  "(3) evaluate the oracle and a DP policy under shift, and a policy trained on shifted data, to measure the cost of a stale model; (4) compare against a model-based safeguard such as a pessimistic penalty in the supported part of the space; "
                  "(5) measure overtime and waiting from visit timestamps, which the table lacks; (6) estimate the reward weights and the waiting list dynamics from clinic records instead of "
                  "assuming them; (7) let the structured model use each booked patient's own predicted risk at decision time, together with the value of the next state; (8) test the structured "
                  "model in a clinic that records its overbooking decisions, where the empirical MDP and the structured model can be compared on the same real data.")
    P.h1("Conclusion")
    P.p(f"On a simulated clinic, DP, MC and Q-learning trained on {cfg['n_eps']} logged days scored {S['DP']['J']:.1f}, {S['Monte Carlo']['J']:.1f} and {S['Q-learning']['J']:.1f} against {rule_J:.1f} for a conservative overbooking rule and {orc_J:.1f} for DP trained on ten times as much data. "
        f"Five exploration schedules gave learning curves that all plateaued at the same model value: none reached 90% of the possible improvement ({n_reach} of {n_runs}), only the near-greedy Q-learner was clearly slower, and the ordering of schedules on true return differed between learners and was comparable in size to the seed-to-seed discrepancy. "
        f"The learned policies lost less than the rule when no-shows fell by 0.05 (all {n_above[dnk]} of {len(cells)} configurations were ahead) and matched it when they rose by 0.08.")
    P.p(f"The limit is the coverage of the logged data. A structured model that learns only patient behaviour and takes the clinic's rules as known scored {M7['Ours: DP']['J']['mean']:.1f} in the same simulator, "
        f"{min(d7):.1f} to {max(d7):.1f} above the three learners and {A7['paired']['Ours: DP vs Oracle DP (10x data)']['mean']:.1f} above the oracle, and reached that level with {x0} logged days. "
        f"On clinic sessions rebuilt from the Medical Appointment No Shows table it scored {MB7['Ours: DP']['J']['mean']:.1f} against {MB7['Rule']['J']['mean']:.1f} for the rule, "
        f"{MB7['Sir: DP']['J']['mean']:.1f} for DP on the estimated MDP and {MB7['One-session optimiser']['J']['mean']:.1f} for the best baseline from published work, and its margin over that "
        f"baseline grew from {g7[0]:.1f}% to {g7[-1]:.1f}% as the waiting list came under pressure. Q-learning came closer to the DP solution of the structured model than MC in both comparisons. "
        "These are results from a simulator and from a simulation built from real records, not evidence about real clinics.")
    defense(P, [
        ("Why does no schedule reach 90% of the optimum?",
         f"The learners solve a fitted model whose DP solution is worth about {np.mean(meth['DP']['Jor']):.1f} on the oracle model, against a rule value of {Jbeh:.1f} and an optimum of {Jopt:.1f}. They converge to that solution (plateaus {q_last:.1f} and {m_last:.1f}), so the target is out of reach for any schedule; "
         "the limit is the support of the logged data, not exploration."),
        ("If exploration does not matter, why is near-greedy Q-learning slower?",
         f"Near-greedy Q-learning has a mean AUC of {AUC['Q-learning']['Near-greedy (0.02)']['mean']:.2f} against {AUC['Q-learning'][best_auc['Q-learning']]['mean']:.2f} for the best schedule, and a plausible reason (not tested separately) is that it tries few actions early and needs many steps to correct its initial values. "
         "That is a speed effect within a fixed model; it disappears from the final values, which are all indistinguishable from zero."),
        ("Why do the model-scale and true-simulator comparisons disagree?",
         f"The model treats rarely logged pairs as absorbing with the worst reward, so it undervalues every policy by {abs(min(gaps)):.0f} to {abs(max(gaps)):.0f} units and penalises choices the data do not support. On that scale the learners are {min(abs(v) for v in dJor.values()):.1f} to {max(abs(v) for v in dJor.values()):.1f} below the rule, while in the true simulator they are ahead of it. "
         "The model scale is useful for learning speed, not for deciding which policy is better."),
        ("Can we conclude that exponential decay is best for Q-learning?",
         f"No. Its true return in the schedule study ({SH['Q-learning']['eps exp. decay']['nominal']:.1f}) differs by {abs(gauge['Q-learning']):.1f} from the same configuration in the benchmark ({S['Q-learning']['J']:.1f}), which is comparable to the {spread['Q-learning']:.1f}-unit spread across all five Q-learning schedules, and the same schedule is the worst for MC."),
        ("Is the robustness result a reason to deploy the learned policy?",
         f"No. It is a simulated shift of a single parameter, with no intervals, evaluated on a policy that differs from the rule in few states. It shows that the learned policies do not lose more than the rule when no-shows fall, in all {n_above[dnk]} of {len(cells)} configurations, and nothing more."),
        ("Why does the oracle score so much more than the learners?",
         f"With {cfg['n_oracle']:,} episodes the high-overbooking actions are supported (about {per_pair_or:.0f} visits per pair against {per_pair:.1f}), and the oracle uses +3 or +4 in {n_act['Oracle DP'][3] + n_act['Oracle DP'][4]} states. The learners use them in at most {max(n_act[k][3] + n_act[k][4] for k in ('DP', 'Monte Carlo', 'Q-learning'))}."),
    ])
    ext.questions(P, 7)
    repro_v2(P, 7, ["fig_bars", "fig_learning", "fig_policies", "fig_agreement", "fig_schedules", "fig_efficiency", "fig_shift"])
    ext.repro(P)
    code_appendix_v2(P, 7, [domains.Clinic])
    rl_core_appendix(P)
    ext.code(P)
    P.refs()
    path = f"{PAP}/Paper07_v2_edited.docx"
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
