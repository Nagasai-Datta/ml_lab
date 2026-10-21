"""paper07_v3.py -- version 3 of the paper: Paper 7 with every result taken from the real Medical Appointment No Shows
data, and our method compared with Paper 7's methods and with baselines from published work on the same data.

    python topic7/code/paper07_v3.py      writes topic7/paper/Paper07_v3_real_data.docx

Built like Paper 7 (same title, layout code, section order, tables and figures), from out/topic7_real/results.json,
which run_real.py writes. The numbers are never typed: every one is computed from that file, and the qualitative
statements are guarded by asserts, so the build fails loudly if a rerun no longer supports the wording.
Versions 1 and 2 (paper07.py, paper07_v2.py) are not affected.
"""
import inspect, json, os
import numpy as np
from scipy import stats
from docx_helper import Paper
from paper_common import REFS, PAP, OUT, pm, pstr, background, rl_core_appendix
from paper07_ext import num, sg, pmx, pc, ahead
from topic7_refs import OURS
from real_clinic import RealClinic

D = os.path.join(OUT, "topic7_real")
NAME = {"Behavior": "Clinician behaviour (modal logged action)", "Rule": "Guideline rule", "DP": "DP on estimated MDP",
        "Monte Carlo": "Monte Carlo control", "Q-learning": "Q-learning", "Oracle DP": "Oracle DP (10x data)",
        "Ours: DP": "Structured model DP (ours)", "Ours: Monte Carlo": "Monte Carlo control, structured model",
        "Ours: Q-learning": "Q-learning, structured model", "Single session optimiser": "Single session optimiser"}
SIR = ("Behavior", "Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP")


def fig(name):
    return os.path.join(D, f"{name}.png")


def ttest(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), 1.96 * float(d.std(ddof=1) / np.sqrt(len(d))), float(stats.ttest_1samp(d, 0).pvalue), int((d > 0).sum())


# authors, written as an IEEE author block (one column per author) in place of the placeholder line of docx_helper
AUTHORS = [("M Naga Sai Dattu", "23BCE0757", "marupaka.naga2023@vitstudent.ac.in"),
           ("Prakrit Bahl", "23BCE0826", "prakrit.bahl2023@vitstudent.ac.in")]
AFFILIATION = ("School of Computer Science and Engineering (SCOPE)", "Vellore Institute of Technology", "Vellore, India")


def author_block(P, authors=AUTHORS, affiliation=AFFILIATION):
    """Replace docx_helper's '[Author names] | ...' line with a borderless table, one centred column per author:
    name, registration number, school, institute, city and e-mail, as in the IEEE conference template."""
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    doc = P.doc
    old = next(p for p in doc.paragraphs if p.text.startswith("[Author names]"))
    tb = doc.add_table(rows=1, cols=len(authors))
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "nil"); borders.append(e)
    pr = tb._tbl.tblPr                                    # schema order: tblBorders comes before shd, tblLayout, tblCellMar, tblLook
    after = [c for c in pr if c.tag in {qn("w:shd"), qn("w:tblLayout"), qn("w:tblCellMar"), qn("w:tblLook")}]
    after[0].addprevious(borders) if after else pr.append(borders)
    width = Cm(16.4 / len(authors))                       # the full text width, split evenly
    tb.autofit = False
    for col in tb.columns:
        col.width = width
    for c in pr:
        if c.tag == qn("w:tblW"):
            c.set(qn("w:type"), "dxa"); c.set(qn("w:w"), str(int(16.4 / 2.54 * 1440)))
    mar = OxmlElement("w:tblCellMar")                     # small side padding so the school's name fits on one line
    for side in ("left", "right"):
        m = OxmlElement(f"w:{side}"); m.set(qn("w:w"), "28"); m.set(qn("w:type"), "dxa"); mar.append(m)
    look = [c for c in pr if c.tag == qn("w:tblLook")]
    look[0].addprevious(mar) if look else pr.append(mar)
    for cell, (name, reg, mail) in zip(tb.rows[0].cells, authors):
        cell.width = width
        lines = [(name, 11, False), (reg, 10, False), (affiliation[0], 10, True), (affiliation[1], 10, True),
                 (affiliation[2], 10, False), (mail, 10, False)]
        for i, (text, size, italic) in enumerate(lines):
            p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(8 if i == len(lines) - 1 else 0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(text); r.font.size = Pt(size); r.italic = italic
    old._p.addnext(tb._tbl)
    old._p.getparent().remove(old._p)


def build(short=False):
    """short=False writes version 3. short=True (paper07_v4.py) writes version 4: the same paper without Appendix A
    (reviewer questions) and Appendix C (source code); the reproducibility appendix stays and becomes Appendix A."""
    R = json.load(open(os.path.join(D, "results.json")))
    B, Sd, DT, CF = R["benchmark"], R["schedule"], R["data"], R["config"]
    S, meth, cfg = B["summary"], B["methods"], B["config"]
    dom = RealClinic
    nA, CAP, H, gam, beps = dom.nA, dom.CAP, dom.max_steps, dom.gamma, dom.behavior_eps
    nst = B["n_states"]
    SCH = Sd["schedules"]; ALG = ("Q-learning", "Monte Carlo")
    Jopt, Jbeh = Sd["J_opt"], Sd["J_beh"]
    nrep = cfg["seeds"]; nS = len(Sd["t90"]["Q-learning"][SCH[0]])
    tfB = float(stats.t.ppf(0.975, nrep - 1) / 1.96)
    tf6 = float(stats.t.ppf(0.975, nS - 1) / 1.96)
    rule_J, orc_J = S["Rule"]["J"], S["Oracle DP"]["J"]
    J_ = lambda k: S[k]["J"]; ci_ = lambda k: S[k]["ci"]; inf = {k: S[k]["info"] for k in S}

    # ---- benchmark statistics (paired over replicates against the rule), as in Paper 7
    pv = {k: ttest(meth[k]["J"], meth["Rule"]["J"]) for k in ("DP", "Monte Carlo", "Q-learning")}
    dq, dm, mq = ttest(meth["DP"]["J"], meth["Q-learning"]["J"]), ttest(meth["DP"]["J"], meth["Monte Carlo"]["J"]), ttest(meth["Monte Carlo"]["J"], meth["Q-learning"]["J"])
    gap_orc = orc_J - rule_J
    sdQ, sdM = float(np.std(meth["Q-learning"]["J"], ddof=1)), float(np.std(meth["Monte Carlo"]["J"], ddof=1))
    bvr = ttest(meth["Behavior"]["J"], meth["Rule"]["J"])
    # ---- model scale (oracle MDP)
    Jor = {k: float(np.mean(meth[k]["Jor"])) for k in SIR}
    gaps = [float(np.mean(meth[k]["J"]) - np.mean(meth[k]["Jor"])) for k in SIR]
    og = np.array(B["optimism_gap"])
    cq, cm = np.array(B["curves"]["Q-learning"]), np.array(B["curves"]["Monte Carlo"])
    K = cq.shape[1]; xck = cq[0, :, 0]
    q0 = float(cq[:, 0, 1].mean()); v0 = (q0 - Jbeh) / (Jopt - Jbeh); offset = v0 / K
    def settle(c):
        mc_ = c[:, :, 1].mean(0)
        return float(xck[np.flatnonzero(np.abs(mc_ - mc_[-1]) <= 1.0)[0]]), float(mc_[-1]), float(mc_.max()), float(xck[mc_.argmax()])
    sq, sm = settle(cq), settle(cm)
    dJor = {k: Jor[k] - Jbeh for k in ("DP", "Monte Carlo", "Q-learning")}
    n_neg = sum(1 for k in ("DP", "Monte Carlo", "Q-learning") for v in meth[k]["Jor"] if v < Jbeh)
    # ---- policies of replicate 0
    pol = B["policies"]; rl = np.array(pol["Rule"])
    n_diff = {k: int(np.sum(np.array(pol[k]) != rl)) for k in SIR if k != "Rule"}
    n_diffb = {k: int(np.sum(np.array(pol[k]) != np.array(pol["Behavior"]))) for k in ("DP", "Monte Carlo", "Q-learning")}
    n_act = {k: np.bincount(pol[k], minlength=nA).tolist() for k in pol}
    agQ, agM = 100 * np.mean(B["agree"]["Q-learning"]), 100 * np.mean(B["agree"]["Monte Carlo"])
    p_off = beps / nA
    per_pair = cfg["n_eps"] * H / nst * p_off; per_pair_or = cfg["n_oracle"] * H / nst * p_off
    hi_or = n_act["Oracle DP"][3] + n_act["Oracle DP"][4]
    hi_l = max(n_act[k][3] + n_act[k][4] for k in ("DP", "Monte Carlo", "Q-learning"))
    # ---- schedule study
    AUC, FIN, SH = Sd["auc"], Sd["final"], Sd["shift"]
    n_reach = int(round(sum(Sd["reach_rate"][a][s] * nS for a in ALG for s in SCH))); n_runs = len(ALG) * len(SCH) * nS
    best_auc = {a: max(SCH, key=lambda s: AUC[a][s]["mean"]) for a in ALG}
    worst_auc = {a: min(SCH, key=lambda s: AUC[a][s]["mean"]) for a in ALG}
    best_nom = {a: max(SCH, key=lambda s: SH[a][s]["nominal"]) for a in ALG}
    worst_nom = {a: min(SCH, key=lambda s: SH[a][s]["nominal"]) for a in ALG}
    def sep(a, s1, s2, tab=AUC):
        return abs(tab[a][s1]["mean"] - tab[a][s2]["mean"]) > (tab[a][s1]["ci"] + tab[a][s2]["ci"]) * tf6
    NG = "Near-greedy (0.02)"
    fin_pos = [(a, s) for a in ALG for s in SCH if FIN[a][s]["mean"] - FIN[a][s]["ci"] * tf6 > 0]
    fins = [FIN[a][s]["mean"] for a in ALG for s in SCH]
    max_fin_key = max(((a, s) for a in ALG for s in SCH), key=lambda k: FIN[k[0]][k[1]]["mean"]); max_fin = FIN[max_fin_key[0]][max_fin_key[1]]["mean"]
    auc_rng = {a: (min(AUC[a][s]["mean"] for s in SCH), max(AUC[a][s]["mean"] for s in SCH)) for a in ALG}
    nearQ_sep = [s for s in SCH if s != NG and sep("Q-learning", NG, s)]
    nearM_sep = [s for s in SCH if s != NG and sep("Monte Carlo", NG, s)]
    qopt = 14.0 * (1 - gam ** H) / (1 - gam); q_init = 0.5 * qopt
    rho_q = float(stats.spearmanr([AUC["Q-learning"][s]["mean"] for s in SCH], [SH["Q-learning"][s]["nominal"] for s in SCH]).statistic)
    rho_m = float(stats.spearmanr([AUC["Monte Carlo"][s]["mean"] for s in SCH], [SH["Monte Carlo"][s]["nominal"] for s in SCH]).statistic)
    rho_qm = float(stats.spearmanr([SH["Q-learning"][s]["nominal"] for s in SCH], [SH["Monte Carlo"][s]["nominal"] for s in SCH]).statistic)
    nom = {a: [SH[a][s]["nominal"] for s in SCH] for a in ALG}
    spread = {a: max(nom[a]) - min(nom[a]) for a in ALG}
    gauge = {a: SH[a]["eps exp. decay"]["nominal"] - J_(a) for a in ALG}
    gmax = max(abs(v) for v in gauge.values())
    rank_exp_mc = sorted(SCH, key=lambda s: -SH["Monte Carlo"][s]["nominal"]).index("eps exp. decay") + 1
    # ---- shift (Paper 7's Table 5 layout)
    sref = Sd["shift_ref"]; nomk, upk, dnk = list(sref)
    allrows = {"Rule": {k: sref[k]["Rule"] for k in sref}, "DP": {k: sref[k]["DP"] for k in sref}}
    for a in ALG:
        for s in SCH:
            allrows[f"{a}|{s}"] = SH[a][s]
    ruleS = allrows["Rule"]; cells = [(a, s) for a in ALG for s in SCH]
    n_above = {k: sum(1 for a, s in cells if SH[a][s][k] > ruleS[k]) for k in sref}
    g_up = {c: SH[c[0]][c[1]][upk] - ruleS[upk] for c in cells}
    g_dn = {c: SH[c[0]][c[1]][dnk] - ruleS[dnk] for c in cells}
    g_nom = {c: SH[c[0]][c[1]][nomk] - ruleS[nomk] for c in cells}
    drop_up = {k: v[upk] - v[nomk] for k, v in allrows.items()}
    ch_dn = {k: v[dnk] - v[nomk] for k, v in allrows.items()}
    lc = [k for k in allrows if "|" in k]
    # ---- our method, baselines, extra studies
    ep, PS = B["ep_mean"], {}
    for k in ("DP", "Monte Carlo", "Q-learning", "Oracle DP", "Rule"):
        PS[k] = ttest(meth["Ours: DP"]["J"], meth[k]["J"])
    PE = {k: ttest(ep["Ours: DP"], ep[k]) for k in ("Single session optimiser", "Fixed booking", "Threshold double booking", "Rule", "Oracle DP", "DP")}
    qQ, qM = ttest(meth["Ours: Q-learning"]["J"], meth["Q-learning"]["J"]), ttest(meth["Ours: Monte Carlo"]["J"], meth["Monte Carlo"]["J"])
    BL = B["baselines"]; theta, kfix = BL["best_threshold"], BL["best_k"]
    DS = R["data_size"]; xs = list(DS)
    dsm = {x: {m: (float(np.mean(DS[x][m])), float(1.96 * np.std(DS[x][m], ddof=1) / np.sqrt(len(DS[x][m])))) for m in DS[x]} for x in xs}
    dsd = {x: ttest(DS[x]["Ours: DP"], DS[x]["DP"]) for x in xs}
    DM = R["demand"]; dk = list(DM)
    gain = [100 * (np.mean(DM[k]["Ours: DP"]) - DM[k]["Single session optimiser"]) / abs(DM[k]["Single session optimiser"]) for k in dk]
    BS = B["shift"]; shm = {c: {k: float(np.mean(v)) for k, v in BS[c].items()} for c in BS}
    TC = B["true_curves"]
    tsett = {k: settle(np.array(TC[k])) for k in TC}
    nE = cfg["n_eval"]
    N_EST, N_EMP = 12 + 1 + 13, nst * nA * (nst + 1)
    names = dict(NAME); names["Threshold double booking"] = f"Threshold double booking (θ = {theta:.2f})"; names["Fixed booking"] = f"Fixed booking (k = {kfix})"

    # ---- guards: the wording below holds only if these do (the build fails loudly otherwise)
    assert all(v[2] < 0.001 and v[3] == nrep for v in pv.values()) and dq[2] < 0.05 and dm[2] < 0.05 and mq[2] < 0.05
    assert J_("DP") > J_("Q-learning") > J_("Monte Carlo") and abs(bvr[0]) < 0.5
    assert n_reach == 0 and len(fin_pos) == len(fins) and all(v > 0 for v in dJor.values())
    assert best_auc["Q-learning"] == "eps exp. decay" and worst_auc["Q-learning"] == NG and nearQ_sep
    assert n_above[upk] == len(cells) and n_above[dnk] == len(cells) and n_above[nomk] == len(cells)
    assert ruleS[dnk] > ruleS[nomk] > ruleS[upk] and all(drop_up[k] < 0 for k in allrows)
    assert np.mean(list(g_up.values())) > np.mean(list(g_nom.values())) > np.mean(list(g_dn.values()))
    assert J_("Ours: DP") == max(J_(k) for k in S) and all(shm[c]["Ours: DP"] == max(shm[c].values()) for c in shm)
    assert hi_or > hi_l and rho_qm < 0 and rho_m > rho_q > 0
    assert all(a < b for a, b in zip(gain, gain[1:])) and dsm[xs[0]]["DP"][0] < rule_J

    P = Paper("Overbooking Clinic Appointments with Reinforcement Learning: Exploration Schedules, Learning Efficiency and Robustness to Shifts in the No-Show Rate",
              "appointment scheduling; no-show; overbooking; exploration schedule; epsilon-greedy; optimistic initialisation; learning curves; distribution shift; structured model; real appointment data",
              running="Clinic overbooking, exploration and shift", refdict={**REFS, **OURS})
    author_block(P)

    # ================================================================== abstract
    P.abstract(
        "Patients who miss appointments leave clinic slots empty, and booking more patients than slots recovers capacity at the risk of overtime and long waits. "
        f"We model a clinic that decides before each session how many extra patients to book as a Markov decision process with {nst} states and {nA} actions, built from the public 'Medical Appointment No Shows' table: "
        "each session is a real neighbourhood day, attendance is what the table records, the state combines the slots filled, the no-show risk predicted by a classifier and a waiting list, "
        "and every policy is scored on later weeks that it never saw. "
        f"From {cfg['n_eps']} logged clinic days we learn policies with dynamic programming (DP), Monte Carlo (MC) control and Q-learning, compare five exploration schedules (constant epsilon, exponential decay, a 1/k decay, optimistic initialisation, near greedy), "
        "and score the final policies on the real scoring sessions, including two shifts of the no-show rate. "
        f"DP, MC and Q-learning scored {J_('DP'):.1f}, {J_('Monte Carlo'):.1f} and {J_('Q-learning'):.1f} against {rule_J:.1f} for the overbooking rule, a gain of {sg(pv['DP'][0], 1)}, {sg(pv['Monte Carlo'][0], 1)} and {sg(pv['Q-learning'][0], 1)} "
        f"(all p < 0.001) over {nrep} replicates, whereas a DP policy fitted to ten times as much data scored {orc_J:.1f}. "
        f"On the fitted model no schedule reached 90% of the optimal improvement in any of {n_runs} runs; the near greedy schedule was the slowest for Q-learning (normalised area under the learning curve "
        f"{num(AUC['Q-learning'][NG]['mean'])} against {num(AUC['Q-learning'][best_auc['Q-learning']]['mean'])} for the best schedule), and the ranking of schedules by final true return differed between the two "
        f"algorithms (rank correlation {sg(rho_qm)} over five schedules). When the no-show rate rose by 0.08, all {n_above[upk]} learned configurations stayed above the rule, by "
        f"{np.mean(list(g_up.values())):.1f} units on average against {np.mean(list(g_nom.values())):.1f} at the nominal rate; when it fell by 0.05, the rule gained {ch_dn['Rule']:.1f} and they were still ahead by "
        f"{np.mean(list(g_dn.values())):.1f}. A structured model that learns only patient behaviour and takes the clinic's rules as known scored {J_('Ours: DP'):.1f} with the same {cfg['n_eps']} days "
        f"({sg(PS['DP'][0], 1)} over DP, p < 0.001) and {dsm[xs[0]]['Ours: DP'][0]:.1f} with {xs[0]} days, ahead of every other method, including a single session optimiser from published work "
        f"({J_('Single session optimiser'):.1f}), at the nominal and both shifted rates; its margin over that optimiser grows to {gain[-1]:.0f}% when waiting is costlier, and Q-learning came closer to its DP "
        "solution than MC. No clinical claims are made.")

    # ================================================================== 1 Introduction
    P.h1("Introduction")
    P.p("Missed appointments are costly: a slot reserved for a patient who does not come is lost capacity, and clinics have long responded by overbooking, that is, giving more appointments than slots. "
        "Overbooking trades the value of filled slots against the cost of overtime when everyone arrives and of waiting for those who are not seen [@gupta,zacharias]. "
        "The right amount depends on the no-show rate, which varies with the weekday, the season and the patient mix and drifts over time.")
    P.p("Reinforcement learning (RL) can learn an overbooking rule from logged clinic data without an explicit model of no-show behaviour [@sb]. Three practical questions follow. "
        "First, how quickly and how reliably do standard learners converge, and does the exploration schedule matter? The textbook answer is that it does, because an agent that never tries an action cannot learn its value [@sb,singh00,rmax]. "
        "Second, a policy learned from a log is deployed in a clinic whose no-show rate may differ from that of the log, so how robust is it? Third, if the logs are the limit, can they be used more efficiently? "
        "This paper studies all three on clinic sessions rebuilt from a real appointment table, scoring every policy on weeks that were never used for learning.")
    P.p("A feature of the offline setting changes the first question. When the learner is trained on a model fitted from logs, its exploration happens inside that model, and a state and action pair that the logs never contain cannot be discovered by exploring harder. "
        "The exploration schedule then governs how fast the fitted model is solved rather than what information is collected. Part of our contribution is to measure how much the schedule matters under that condition and to report the answer even when it is small. "
        "The same condition points to a remedy: if the clinic's own rules are known, only patient behaviour has to be learned, and every logged session informs the same few quantities.")
    P.p(f"**Contributions.** (i) A clinic overbooking MDP built from the Medical Appointment No Shows table, with a state made of the slots filled, the predicted no-show risk and a waiting list, and an operational reward (Section 4). "
        "(ii) A normalised learning efficiency metric, the area under the learning curve and the steps to 90% of the optimal improvement, defined against the clinic's own behaviour and the oracle optimum (Eq. 3 to 5). "
        f"(iii) A controlled comparison of {len(SCH)} exploration schedules for two learners with {nS} replicates each. "
        "(iv) A shift study that scores the trained policies when the no-show rate is higher or lower than in the log (Eq. 6). "
        "(v) A structured model in which only patient behaviour is learned (Section 5.2), compared with the learners and with baselines from published work (Sections 7.5 to 7.9). "
        "(vi) Candid results where they are small: no schedule reaches 90%, and the margin over the best published baseline is small unless waiting is costly.")
    P.p("**Scope and honesty.** Every result uses the real table, but the table does not record everything a clinic decision needs: it has no overbooking decisions, no waiting list and no times within a day. "
        "The decisions in the logs are therefore generated by a behaviour policy, the waiting list is simulated, and overtime and waiting are counted in patients (Section 6.1). "
        "Findings concern algorithms, not clinics, and no clinical claim is made.")

    # ================================================================== 2 Related work
    P.h1("Related work")
    P.p("**Appointment scheduling and overbooking.** Operations research has studied appointment systems extensively, including no-shows, overbooking and the trade-off between patient waiting and provider overtime [@gupta]; "
        "analytical results for overbooking with no-shows are given by Zacharias and Pinedo [@zacharias]. Predicted no-show probabilities have been combined with overbooking decisions for a single session, "
        "for example by marginal analysis [@topuz], and fixed risk thresholds for double booking serve as the baseline for learned policies [@amalina]. "
        "Our contribution is not a better scheduling model but a study of how RL learners behave on one, and of how much they gain from a model that knows the clinic's rules.")
    P.p("**Exploration in RL.** Epsilon-greedy exploration, decaying schedules and optimistic initial values are the textbook devices [@sb]. Convergence of on-policy methods needs schedules that are greedy in the limit with infinite exploration (GLIE) [@singh00]; "
        "Q-learning converges under visit-count conditions on the step size [@watkins,evendar]. Model-based methods such as R-max make optimism explicit [@rmax]. "
        "Our schedules are the simple ones a practitioner would try; the 'GLIE-style' schedule here has a floor and is therefore not strictly GLIE.")
    P.p("**Offline RL and shift.** Learning from logs is limited by data coverage and is sensitive to model error, which motivates pessimism and support constraints [@levine,jin,kumar]. Robustness to a change in the environment after deployment is a related but different concern, "
        "and we treat it with the simplest possible experiment: scoring the trained policies again on sessions with a shifted no-show rate. "
        "Known structure is an alternative to more data: in a factored MDP the transition model is assembled from separately specified parts of the state [@boutilier], which is the idea behind the structured model of Section 5.2.")
    background(P)

    # ================================================================== 4 Problem formulation
    Sm = DT["summary"]; C = DT["classifier"]; e = DT["risk_edges"]
    P.h1("Problem formulation: a clinic overbooking MDP built from real sessions")
    P.h2("Real clinic sessions from the Medical Appointment No Shows table")
    P.p(f"The Medical Appointment No Shows table [@kaggle] lists 110,527 medical appointments in Brazil, one row per appointment, with the day the appointment was booked, the day it took place, "
        "the patient's age, sex and neighbourhood, an SMS reminder flag, enrolment in a welfare programme, flags for hypertension, diabetes, alcoholism and disability, and whether the patient came. "
        f"After removing one row with a negative age and the 39 appointments on Saturdays, {Sm['rows']:,} appointments of {Sm['patients']:,} patients in {Sm['neighbourhoods']} neighbourhoods remain, "
        f"on {Sm['days']} weekdays between 29 April and 8 June 2016; {pc(Sm['noshow_rate'])} of them were missed.")
    P.p("A classifier predicts the no-show probability of each appointment from what is known when it is booked: the days between booking and appointment, the patient's age, sex, SMS flag, welfare flag "
        "and the four condition flags, the weekday, the numbers of the patient's earlier appointments and earlier no-shows in the table, and the neighbourhood. "
        f"It is a gradient boosting model [@friedman] fitted on the {C['n_fit']:,} appointments before 9 May 2016, calibrated by isotonic regression [@zadrozny] on the {C['n_cal']:,} appointments from 9 to 15 May, "
        f"and applied to the {C['n_test']:,} appointments from 16 May onwards; both steps use scikit-learn [@pedregosa]. On these appointments its area under the ROC curve is {C['auc_test']:.3f}, its Brier score "
        f"{C['brier']:.3f}, and its mean predicted no-show probability {pc(C['mean_pred'])} against {pc(C['mean_obs'])} observed. No information from later weeks enters the classifier.")
    P.p("The appointments from 16 May onwards are grouped into clinic sessions, one for each neighbourhood and day with at least 5 appointments. "
        f"Sessions from 16 to 25 May ({DT['sessions_past']}) are used only for learning and sessions from 30 May to 8 June ({DT['sessions_future']}) only for scoring. "
        f"The no-show rate fell from {pc(DT['noshow_past'])} in the first period to {pc(DT['noshow_future'])} in the second, so the scoring weeks already differ from the learning weeks.")
    P.h2("The overbooking MDP")
    P.p(f"A clinic runs {H} sessions per episode, each with {CAP} regular slots. An episode draws its sessions at random from the real sessions of one period, keeping the weekday order. "
        f"The first min({CAP}, n) appointments of a session fill its regular slots, and before the session the clinic books k extra patients, k in {{0, 1, 2, 3, 4}}, "
        "drawn from all appointments on the same weekday of the same period. Whether each patient shows up is what the table records. "
        f"Overtime is the number of patients above {CAP}, idle is the number of unused slots, and a waiting list of length w (0 to 12) grows by a Poisson(1.5) number of patients per session and shrinks by "
        "min(w, k), the number of extra bookings drawn from it; the table has no waiting list, so this is the one simulated part of the clinic. "
        f"The reward per session is shows − 3 × overtime − 0.5 × idle − 0.4 × waiting list length + 0.3 × (patients drawn from the waiting list). Rewards are discounted with γ = {gam}.")
    t_mdp = P.table([["Element", "Definition"],
                     [f"State ({nst})", f"slots filled (3: 5 to 7, 8 or 9, 10) × predicted no-show risk band (4: quartiles of the session mean, cut points {e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}) × waiting list band (5: lengths 0, 1 or 2, 3 to 5, 6 to 8, 9 or more)"],
                     [f"Actions ({nA})", f"book k extra patients, k = 0 to {nA - 1} (labels {', '.join(dom.action_labels)})"],
                     ["Reward per session", "shows − 3 overtime − 0.5 idle − 0.4 waiting list + 0.3 drawn from waiting list"],
                     ["Discount, horizon", f"γ = {gam}; {H} sessions per episode"],
                     ["Overbooking rule", "book one extra patient when the predicted risk band is 0 or 1, otherwise two extra"],
                     ["Behaviour policy", f"the rule with ε = {beps} uniformly random actions"],
                     ["Data", f"Medical Appointment No Shows: {DT['sessions_past']} learning sessions (16 to 25 May 2016), {DT['sessions_future']} scoring sessions (30 May to 8 June 2016)"]],
                    "The clinic overbooking MDP.", widths=[3.6, 12.8], font=8.5)
    P.p(f"Three design features matter for what follows. First, the rule only ever uses k = 1 or 2, and the behaviour policy picks each of the other actions with probability ε / {nA} = {p_off:.2f} per session, so the logs contain few examples of overbooking by 3 or 4. "
        f"If visits were spread evenly over the {nst} states, a given state and action pair outside the rule would be seen about {per_pair:.1f} times in {cfg['n_eps']} logged episodes and about {per_pair_or:.0f} times in {cfg['n_oracle']:,}; since pairs seen fewer than 10 times are treated as unsupported, "
        "the small dataset cannot support most of the actions the optimum uses, while the large one can. (Visits are not spread evenly, so this is an order of magnitude argument.) "
        "Second, the waiting list rewards overbooking beyond the show rate trade-off, because extra bookings drain it. Third, the risk band in the state comes from the classifier, not from the attendance itself, "
        "so a change in how often patients come that the classifier has not seen does not change the state.")

    # ================================================================== 5 Method
    P.h1("Method: exploration schedules, shift evaluation and a structured model")
    P.h2("Exploration schedules, normalised learning efficiency and shift evaluation")
    P.p(f"Both learners are trained on a model fitted to the logged episodes and sample experience from that model (never from the scoring sessions). Let k be the step count and T the budget ({cfg['steps']:,} steps). The five schedules are")
    P.eq("constant: ε_k = 0.1;   exponential: ε_k = 0.05 + 0.95 exp(−5k/T);   1/k-type: ε_k = max(0.02, c / (c + k)),  c = T / 50;   near-greedy: ε_k = 0.02,", "3")
    P.p(f"and the optimistic initialisation schedule uses ε_k = 0.02 together with an initial action value q0 = 0.5 × 14 × (1 − γ^H) / (1 − γ) = {q_init:.1f}, where 14 is an assumed upper bound on the reward per session and H = {H}. "
        f"With c = {cfg['steps'] // 50:,} the 1/k schedule equals 0.5 at k = c and reaches its floor only near the end of the budget; because of the floor it is not strictly GLIE [@singh00]. "
        "MC control uses first visit returns with incremental averaging; Q-learning uses a visit count step size [@watkins,evendar].")
    P.p("Learning curves are measured by recording, at 26 evenly spaced checkpoints, the value of the greedy policy on a large 'oracle' model (Section 7.1 explains what that value means). The raw values depend on the scale of the problem, so we normalise them against the clinic's own behaviour and against the optimum of the same model:")
    P.eq("v(π) = [J(π) − J_beh] / [J_opt − J_beh],", "4")
    P.p("so that v = 0 is the behaviour policy and v = 1 the model optimum. We summarise a learning curve by its mean over the checkpoints and by the first checkpoint reaching 90% of the possible improvement:")
    P.eq("AUC = (1 / K) Σ_j v(π_{t_j}),      t90 = min { t_j : v(π_{t_j}) ≥ 0.9 },      K = 26.", "5")
    r_nom, r_sh = DT["session_rate_future"], DT["session_rate_shifted"]
    P.p("The AUC rewards fast learning and penalises a slow start, and t90 asks whether the optimum is approached at all. The table cannot change how often patients come, so the evaluation under shift changes "
        "the recorded outcomes of the scoring sessions instead: with r the no-show rate of the scoring sessions, each recorded show becomes a no-show with probability δ / (1 − r) when δ > 0, and each recorded "
        "no-show becomes a show with probability −δ / r when δ < 0:")
    P.eq("Pr(show → no-show) = δ / (1 − r)  if δ > 0,      Pr(no-show → show) = −δ / r  if δ < 0,      δ ∈ { 0, +0.08, −0.05 }.", "6")
    P.p(f"This moves the no-show rate of the scoring sessions from {pc(r_nom)} to {pc(r_sh[upk])} and {pc(r_sh[dnk])}. The policy, trained on nominal data, is not changed, and neither are the predicted risks, "
        "so the shift does not move the visited states: a policy can only notice it through the waiting list.")
    P.code("""Algorithm 1  Exploration schedule study
 input : logged clinic days D (150 episodes from the learning sessions), schedule list, learners {Q-learning, MC}, budget T, shift list
 1. fit the MDP from D (pairs seen < 10 times are unsupported; unsupported pairs are absorbing with the worst reward)
 2. for each schedule and each learner:
      run the learner for T steps in the fitted MDP with eps_k (and q0); every T/25 steps record J(greedy policy) on the oracle MDP
      fill unobserved states with the logged modal action; score the final policy on the scoring sessions and their shifted versions (120 episodes)
 3. normalise curves with Eq. (4); compute AUC and t90 with Eq. (5)
 4. repeat over independent data draws; report means and 95% intervals""")
    P.h2("A structured model of the clinic solved by DP")
    P.p(f"Section 4.2 explains what limits any learner that works on the empirical MDP of Section 3: it estimates P(s' | s, a) separately for each of the {nst * nA} pairs (s, a), and the pairs that the optimum uses are rarely logged. "
        "A structured model estimates far fewer quantities. The clinic's own rules are taken as known: the 10 regular slots, the reward of Table 1, and the fact that k extra bookings take min(w, k) patients off "
        "a waiting list of length w. Only patient behaviour is learned: how likely the booked patients are to come, which the classifier of Section 4.1 provides, how often each combination of slots filled and "
        "risk band occurs, and the mean number λ of patients who join the waiting list in a session. A model assembled in this way from separately specified parts of the state is a factored MDP [@boutilier]; "
        f"here {N_EST} numbers are estimated from the logs, against {N_EMP:,} transition probabilities in the empirical MDP.")
    P.p("Write the state as s = (c, v), where c is one of the 12 combinations of slots filled and risk band and v the waiting list band. From the logged sessions, f̂(c) is the share of sessions in cell c "
        "(with 0.5 added to every cell); λ̂ is the mean of w_{t+1} − (w_t − min(w_t, k_t)) over consecutive sessions in which the list did not reach its cap of 12; and π(w | v), the share of each exact list "
        "length w among the logged sessions in band v, links the band to the length. With (x)⁺ = max(0, x), the expected reward of booking k extra patients is")
    P.eq("R̃(s, k) = mean_{i ∈ L_c} E[ N_i − 3 (N_i − 10)⁺ − 0.5 (10 − N_i)⁺ ]  +  E_w[ 0.3 min(w, k) − 0.4 w ],", "7")
    P.p(f"where L_c holds the logged sessions of cell c (up to 400), and N_i, the number of patients who come, is a sum of independent Bernoulli variables: one for each booked patient of session i, with the "
        f"session's mean predicted attendance probability, and one for each extra patient, with probability {DT['q_extra']:.3f}, one minus the mean predicted risk in the learning period; its distribution is "
        "computed exactly by convolution. The second expectation is over w drawn from π(w | v). Tomorrow's patients do not depend on today's booking, so the next state follows as")
    P.eq("P̃(c', v' | s, k) = f̂(c') Pr[ band(min(12, w − min(w, k) + A)) = v' ],", "8")
    P.p("with w drawn from π(w | v) and the number of new patients A drawn from Poisson(λ̂). Eq. (7) and (8) give a reward and a transition for every pair (s, k), whether or not the logs contain it, "
        "so no pair is unsupported and no fallback is needed. The model is solved with value iteration (Eq. 1), the solver of Section 3. The same model can also be given to MC control and Q-learning in "
        "place of the empirical MDP, which separates the effect of the model from that of the learner.")
    P.code("""Algorithm 2  Structured model DP
 input : logged clinic days D; predicted no-show risks of the booked patients; the clinic's rules (10 slots, reward of Table 1, waiting list emptied by extra bookings)
 1. estimate from D: the cell shares f(c), the arrival rate λ and π(w|v)
 2. for every state s and every k = 0 to 4 compute R(s,k) with Eq. (7) and P(s'|s,k) with Eq. (8)
 3. solve the model by value iteration (Eq. 1) and use the greedy policy in every state
 4. optionally, train MC control or Q-learning on the same model instead of step 3""")

    # ================================================================== 6 Protocol
    P.h1("Experimental protocol")
    P.h2("Data: real sessions")
    P.p("**Important disclosure.** Every result in this paper "
        "comes from the real table, and the scoring sessions are never used for learning. The table cannot provide everything the study needs, and the gaps are filled as follows. "
        "(i) It records no overbooking decisions, so the logs are generated by running the behaviour policy of Table 1 over the real learning sessions: the decisions are generated, the attendance is recorded. "
        "(ii) It has no waiting list, so the waiting list of Section 4.2 is simulated. (iii) It has no clinics or slots, so a session is one neighbourhood on one day with 10 regular slots. "
        "(iv) It has no times within a day, so overtime and waiting are counted in patients, not minutes. "
        f"(v) It covers {Sm['days']} weekdays, so {cfg['n_eps']} or {cfg['n_oracle']:,} logged days are runs of {H} sessions drawn from the {DT['sessions_past']} learning sessions, not distinct days. "
        "(vi) Its no-show rate cannot be set, so the shift study changes recorded outcomes (Eq. 6). The results should be read as a method study on recorded attendance, not as a finding about patients.")
    P.h2("Data, solvers and scoring")
    P.table([["Item", "Setting"],
             ["Data", f"{DT['sessions_past']} learning and {DT['sessions_future']} scoring sessions; {nst} states; {nA} actions; {H} sessions per episode"],
             ["Benchmark", f"{cfg['n_eps']} logged episodes; oracle MDP from {cfg['n_oracle']:,} episodes; {nrep} replicates; {nE:,} scoring episodes; {cfg['steps']:,} steps"],
             ["Learners", f"DP (value and policy iteration agreed in all replicates: {'yes' if all(B['pi_equals_vi']) else 'no'}), first visit MC, Q-learning; DP, MC and Q-learning on the structured model"],
             ["Schedule study", f"{len(SCH)} schedules × 2 learners × {nS} replicates (data seeds 700 to {699 + nS}); {cfg['steps']:,} steps; 26 checkpoints; reference values J_beh = {Jbeh:.2f}, J_opt = {Jopt:.2f} on the oracle MDP"],
             ["Shift study", f"no-show offsets 0, +0.08, -0.05 (Eq. 6); {CF['sched_eval']} scoring episodes per run in the schedule study ({CF['shift_ref_eval']} for the rule and the DP policy), {nE:,} for every policy of Section 7.7"],
             ["Further studies", f"logged days {', '.join(xs[:-1])} and {int(xs[-1]):,} ({CF['size_seeds']} replicates); waiting cost 0.4 or 1.0 and 1.5 or 2.5 new patients per session ({CF['demand_seeds']} replicates, {CF['n_eval_demand']} episodes)"],
             ["Statistics", f"95% CI = 1.96 standard errors over replicates; where a significance statement is made the interval is widened to a t interval (factor {tfB:.2f} for {nrep} replicates, {tf6:.2f} for {nS})"]],
            "Experimental configuration.", widths=[5.0, 11.0])
    P.bullets([
        f"**Learners.** MC control and Q-learning each receive {cfg['steps']:,} steps drawn from the estimated MDP, with an exponentially decaying ε schedule (ε from 1.0 to about 0.05) and a visit count step size α = max(0.02, 1/(1 + n^0.7)) for Q-learning.",
        f"**Replicates.** {nrep} independent data draws (each with its own logged dataset and learner seeds). Intervals are 95% intervals over replicates; paired differences are tested with a one sample t test on the differences for each replicate.",
        f"**Scoring.** Each final policy is run for {nE:,} episodes of {H} scoring sessions with common random numbers (episode e brings the same patients and the same attendance for every policy), and the mean discounted return is recorded. "
        "Learning curves are scored exactly on a large 'oracle' MDP estimated from ten times as much logged data.",
        "**Fallback.** In states never observed in the (supported) data a learned policy defers to the clinician's modal logged action, so that unsupported states cannot be exploited.",
    ])
    P.p("In the schedule study the benchmark's exponential schedule is one of the five. Final policies in the schedule study use the logged modal action in states with no data.")
    P.h2("Baselines from published work and further comparisons")
    P.bullets([
        "**Baselines from published work.** (i) A single session optimiser books the k that maximises the expected reward of the current session, computed from each booked patient's predicted risk; "
        "this is marginal analysis of the kind used for overbooking under predicted no-shows [@topuz]. (ii) Threshold double booking fills empty regular slots and books one extra patient for every booked "
        "patient whose predicted risk is at least θ, the fixed threshold rule against which learned double booking policies are compared [@amalina]. (iii) A fixed number k of extra bookings in every session. "
        "The threshold θ (0.20, 0.25, 0.30 or 0.35) and the fixed k (0 to 4) were chosen on the scoring sessions, which favours these baselines.",
        f"**Structured model.** DP, MC and Q-learning on the model of Section 5.2, learned from the same {nrep} logged datasets as the benchmark and scored on the same {nE:,} episodes.",
        f"**Convergence.** During learning, the greedy policy of MC and Q-learning is also scored on {CF['ckpt_eps']} scoring episodes at each of the 26 checkpoints, on the estimated MDP and on the structured model.",
        f"**Statistics.** Differences between two learned policies are tested over replicates; differences from a single policy (the rule, the oracle and the baselines) are also tested over the {nE:,} paired scoring episodes.",
    ])

    # ================================================================== 7 Results
    P.h1("Results")
    P.h2("Benchmark on the real scoring sessions")
    order3 = ["Behavior", "Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP"]
    rows = [["Method", "Return (mean ± 95% CI)", "Shows", "Overtime", "Idle", "Waiting"]]
    for k in order3:
        rows.append([NAME[k], pm(J_(k), ci_(k), 1)] + [f"{inf[k][i]:.1f}" for i in ("shows", "overtime", "idle", "wait")])
    t1 = P.table(rows, f"Benchmark on the real scoring sessions ({nst} states). Return is discounted; shows, overtime, idle slots and the summed waiting list length are totals per {H} session episode "
                       f"({nrep} replicates, 95% CI; rule and oracle are single policies).", widths=[5.0, 3.4, 1.8, 1.9, 1.8, 1.9], font=8.5)
    P.p(f"Table {t1} shows that all three learners are well ahead of the rule. DP scored {pm(J_('DP'), ci_('DP'))}, MC {pm(J_('Monte Carlo'), ci_('Monte Carlo'))} and Q-learning {pm(J_('Q-learning'), ci_('Q-learning'))} against {rule_J:.2f} for the rule. "
        f"Paired over replicates the differences from the rule are {sg(pv['DP'][0])} ± {pv['DP'][1]:.2f} ({pstr(pv['DP'][2])}; ahead in {pv['DP'][3]} of {nrep}) for DP, {sg(pv['Monte Carlo'][0])} ± {pv['Monte Carlo'][1]:.2f} "
        f"({pstr(pv['Monte Carlo'][2])}; {pv['Monte Carlo'][3]} of {nrep}) for MC and {sg(pv['Q-learning'][0])} ± {pv['Q-learning'][1]:.2f} ({pstr(pv['Q-learning'][2])}; {pv['Q-learning'][3]} of {nrep}) for Q-learning. "
        f"With {nrep} replicates a t interval is only {tfB:.2f} times wider than the interval shown. Among the learners the differences are resolved as well: DP minus Q-learning {sg(dq[0])} ± {dq[1]:.2f} ({pstr(dq[2])}), "
        f"DP minus MC {sg(dm[0])} ± {dm[1]:.2f} ({pstr(dm[2])}) and MC minus Q-learning {sg(mq[0])} ± {mq[1]:.2f} ({pstr(mq[2])}), so DP is ahead of Q-learning, which is ahead of MC. "
        f"The replicate to replicate standard deviations are {sdQ:.1f} for Q-learning and {sdM:.1f} for MC. The modal logged action scored {J_('Behavior'):.2f}, {abs(bvr[0]):.2f} below the rule, "
        "consistent with it reproducing the rule almost everywhere.")
    P.p(f"The oracle DP, fitted to {cfg['n_oracle']:,} episodes, scored {orc_J:.2f}, which is {gap_orc:.1f} above the rule and {orc_J - J_('DP'):.1f} above the best learner. The learners trained on {cfg['n_eps']} episodes "
        f"capture most of that headroom ({100 * pv['DP'][0] / gap_orc:.0f}% for DP, {100 * pv['Q-learning'][0] / gap_orc:.0f}% for Q-learning, {100 * pv['Monte Carlo'][0] / gap_orc:.0f}% for MC), because the rule "
        f"is conservative for real attendance: with about {pc(DT['noshow_future'], 0)} of patients not coming, one or two extra bookings leave {inf['Rule']['idle']:.1f} idle slots per episode and a summed waiting list of {inf['Rule']['wait']:.0f}. "
        f"The operational columns show how the oracle gets further: it books more patients ({inf['Oracle DP']['shows']:.0f} shows per episode against {inf['Rule']['shows']:.0f} for the rule), accepts about "
        f"{inf['Oracle DP']['overtime'] / inf['Rule']['overtime']:.1f} times as much overtime ({inf['Oracle DP']['overtime']:.1f} against {inf['Rule']['overtime']:.1f}), has fewer idle slots ({inf['Oracle DP']['idle']:.1f}) "
        f"and a much shorter waiting list ({inf['Oracle DP']['wait']:.0f}). The learners move in the same direction but stop short: DP reduces the summed waiting list to {inf['DP']['wait']:.0f}, Q-learning to "
        f"{inf['Q-learning']['wait']:.0f} and MC to {inf['Monte Carlo']['wait']:.0f}, with overtime of {inf['DP']['overtime']:.1f}, {inf['Q-learning']['overtime']:.1f} and {inf['Monte Carlo']['overtime']:.1f}.")
    f1 = P.fig(fig("fig_learning"), f"Learning curves on the oracle MDP (mean ± s.d. over {nrep} replicates). Dashed: oracle DP optimum; dotted: the overbooking rule.", 7.0)
    P.p(f"Figure {f1} shows the learning curves on the oracle model, which is the scale used for all efficiency metrics. Both learners start at {q0:.1f} (the greedy policy of a value table of zeros). "
        f"MC is within one model unit of its final value by step {sm[0]:,.0f} and stays at {sm[1]:.1f}; Q-learning peaks at {sq[2]:.1f} by step {sq[3]:,.0f} and settles at {sq[1]:.1f} from step {sq[0]:,.0f} of the {cfg['steps']:,}. "
        f"The mean DP policy value on the same model is {Jor['DP']:.1f}, so Q-learning ends {Jor['DP'] - sq[1]:.1f} below the DP solution of its fitted model and MC {Jor['DP'] - sm[1]:.1f} below it. "
        f"All three lie above the rule's {Jbeh:.1f} and far below the optimum {Jopt:.1f}. On this scale DP, MC and Q-learning are {dJor['DP']:.1f}, {dJor['Monte Carlo']:.1f} and {dJor['Q-learning']:.1f} above the rule, "
        f"and on the scoring sessions they are {pv['DP'][0]:.1f}, {pv['Monte Carlo'][0]:.1f} and {pv['Q-learning'][0]:.1f} above it, so the two scales agree in sign and in the order of the learners; "
        f"{n_neg} of {3 * nrep} replicate and learner cells have a model value below the rule's.")
    P.p(f"The model scale is still pessimistic. The oracle model treats a state and action pair seen fewer than 10 times as absorbing with the worst reward in the data, so a policy that chooses such an action is penalised on the model whatever it would do in the clinic. "
        f"The return on the scoring sessions minus the oracle model value ranges from {min(gaps):.1f} to {max(gaps):.1f} over the six methods, and the data model's value of its own DP policy was {abs(og.max()):.1f} to {abs(og.min()):.1f} "
        f"below the oracle model's value of the same policy in every replicate (mean optimism gap {num(og.mean(), 1)}). We therefore use the model scale only to compare learning speed.")
    f2 = P.fig(fig("fig_policies"), "Extra bookings chosen in each state (replicate 0) by the behaviour policy, the three learners and the oracle, for the states observed in the oracle data.", 9.0)
    P.p(f"Figure {f2} compares the policies for replicate 0. The rule uses +1 in {n_act['Rule'][1]} states and +2 in {n_act['Rule'][2]}, never anything else. The modal action behaviour policy differs from it in {n_diff['Behavior']} states, "
        f"and the learners in {n_diff['DP']} (DP), {n_diff['Monte Carlo']} (MC) and {n_diff['Q-learning']} (Q-learning) of {nst}; they differ from the behaviour policy in {n_diffb['DP']}, {n_diffb['Monte Carlo']} and {n_diffb['Q-learning']} states. "
        f"The oracle differs from the rule in {n_diff['Oracle DP']} states and uses +3 ({n_act['Oracle DP'][3]} states) or +4 ({n_act['Oracle DP'][4]} states); the learners use them in at most {hi_l} states. "
        f"This agrees with the support argument of Section 4.2: with {cfg['n_eps']} episodes the high overbooking actions are mostly unsupported, so the learners cannot adopt them. "
        f"Q-learning agreed with DP in {agQ:.0f}% and MC in {agM:.0f}% of the well visited states, averaged over replicates. Only replicate 0 is plotted, where DP scored {meth['DP']['J'][0]:.1f}.")

    P.h2("Exploration schedules on the fitted model")
    rows = [["Schedule", "Q-learning AUC", "Q-learning final", "MC AUC", "MC final", "Reached 90%"]]
    for s in SCH:
        nreach = int(round(sum(Sd["reach_rate"][a][s] * nS for a in ALG)))
        rows.append([s, pmx(AUC["Q-learning"][s]["mean"], AUC["Q-learning"][s]["ci"]), pmx(FIN["Q-learning"][s]["mean"], FIN["Q-learning"][s]["ci"]),
                     pmx(AUC["Monte Carlo"][s]["mean"], AUC["Monte Carlo"][s]["ci"]), pmx(FIN["Monte Carlo"][s]["mean"], FIN["Monte Carlo"][s]["ci"]), f"{nreach}/{2 * nS}"])
    t2 = P.table(rows, f"Exploration schedules: normalised area under the learning curve (AUC), normalised final value and the number of runs (of {2 * nS}) reaching 90% of the optimal improvement. "
                       f"Values are on the scale of Eq. (4): 0 is the clinic's behaviour and 1 the oracle model optimum ({nS} replicates, 95% CI).", widths=[3.8, 2.7, 2.7, 2.7, 2.7, 1.8], font=8.5)
    P.p(f"Table {t2} is the central comparison of the schedule study. No run reached the 90% target in any of the {n_runs} runs, so the steps to 90% metric is undefined for every schedule; "
        f"the best mean final value was {max_fin:.3f} ({max_fin_key[0]}, {max_fin_key[1]}), so the 0.9 target is about {0.9 / max_fin:.0f} times larger. The reason is visible in Figure {f1}: the learners plateau "
        f"near the value of the DP solution of the fitted model ({Jor['DP']:.1f}), well below the oracle optimum ({Jopt:.1f}), so even a perfect solver of that model cannot approach 1. "
        f"Unlike the target, the improvement over the behaviour policy is clear: all {len(fin_pos)} final values stay above zero once the intervals are widened to t intervals, from {min(fins):.2f} to {max(fins):.2f}. "
        f"The AUC values range from {num(auc_rng['Q-learning'][0])} to {num(auc_rng['Q-learning'][1])} for Q-learning and from {num(auc_rng['Monte Carlo'][0])} to {num(auc_rng['Monte Carlo'][1])} for MC. "
        f"A part of every AUC is a common offset: the first checkpoint (step 0) has v = {num(v0)}, which contributes {num(offset, 3)} to the mean over {K} checkpoints in every run, so the offset is the same for all schedules and does not affect comparisons.")
    f3 = P.fig(fig("fig_schedules"), f"Normalised learning curves by exploration schedule (mean ± s.d. over {nS} replicates). 0 is the clinic's behaviour and 1 the oracle model optimum.", 11.0)
    oth = [x for x in SCH if x != NG]
    n_sepQ_oth = sum(1 for i, a in enumerate(oth) for b in oth[i + 1:] if sep("Q-learning", a, b))
    dQ = [abs(AUC["Q-learning"][a]["mean"] - AUC["Q-learning"][b]["mean"]) for i, a in enumerate(oth) for b in oth[i + 1:]]
    P.p(f"Figure {f3} shows what separates the schedules: mostly the first steps and, for the near greedy schedule, a slow climb. For Q-learning the near greedy schedule has an AUC of {pmx(AUC['Q-learning'][NG]['mean'], AUC['Q-learning'][NG]['ci'])}, "
        f"against {pmx(AUC['Q-learning'][best_auc['Q-learning']]['mean'], AUC['Q-learning'][best_auc['Q-learning']]['ci'])} for the best schedule ({best_auc['Q-learning']}); its t widened interval is separated from {len(nearQ_sep)} of the other 4 schedules. "
        "This is the most robust schedule effect in the study, and it is the textbook one: a learner that explores little needs many more steps to learn the values of actions it rarely tries. "
        f"The same schedule for MC (AUC {pmx(AUC['Monte Carlo'][NG]['mean'], AUC['Monte Carlo'][NG]['ci'])}) is separated from {len(nearM_sep)} of the other 4 schedules, so for MC the evidence does not distinguish it from the rest. "
        f"Among the other four schedules for Q-learning the AUC differences are between {min(dQ):.2f} and {max(dQ):.2f}, and {n_sepQ_oth} of these {len(dQ)} pairs {'is' if n_sepQ_oth == 1 else 'are'} separated by the t widened intervals. "
        f"Optimistic initialisation (q0 = {q_init:.0f}, ε = 0.02) gives AUC {num(AUC['Q-learning']['Optimistic init']['mean'])} for Q-learning against {num(AUC['Q-learning']['eps=0.1 const']['mean'])} for constant ε, "
        f"and {num(AUC['Monte Carlo']['Optimistic init']['mean'])} against {num(AUC['Monte Carlo']['eps=0.1 const']['mean'])} for MC. "
        "Seen from the offline RL angle the absence of large effects is unsurprising, because exploration inside a fitted model cannot reveal pairs absent from the logs: once the supported part of the model is sampled enough, every schedule converges towards the same solution, and the schedule mainly changes how soon.")

    P.h2("Does the model scale metric predict true return?")
    P.p(f"Learning efficiency is measured on the model scale, but what matters is the return on the scoring sessions. For each learner we ranked the five schedules by their AUC and by the return of the final policy at the nominal rate (Table {t2 + 1}). "
        f"For MC the two rankings agree exactly (Spearman correlation {sg(rho_m)} over five schedules); for Q-learning they mostly agree ({sg(rho_q)}). With only five points no p-value is reported and the numbers are descriptive. "
        f"The best schedule by return is {best_nom['Q-learning']} for Q-learning ({SH['Q-learning'][best_nom['Q-learning']]['nominal']:.1f}) and {best_nom['Monte Carlo']} for MC ({SH['Monte Carlo'][best_nom['Monte Carlo']]['nominal']:.1f}), "
        f"and the worst is {worst_nom['Q-learning']} ({SH['Q-learning'][worst_nom['Q-learning']]['nominal']:.1f}) and {worst_nom['Monte Carlo']} ({SH['Monte Carlo'][worst_nom['Monte Carlo']]['nominal']:.1f}). "
        f"The exponential decay schedule is the best for Q-learning but ranks {rank_exp_mc} of 5 for MC, and across the five schedules the ranks of the two learners have a correlation of {sg(rho_qm)}.")
    P.p(f"How much of this is signal? The same learner with the exponential schedule also appears in the benchmark, with different data draws (seeds 1000 to {999 + nrep} against 700 to {699 + nS}) and more scoring episodes ({nE:,} against {CF['sched_eval']}): "
        f"Q-learning scored {J_('Q-learning'):.1f} there and {SH['Q-learning']['eps exp. decay']['nominal']:.1f} here ({sg(gauge['Q-learning'], 1)}), and MC {J_('Monte Carlo'):.1f} there and {SH['Monte Carlo']['eps exp. decay']['nominal']:.1f} here ({sg(gauge['Monte Carlo'], 1)}). "
        f"Differences of about {gmax:.1f} units therefore arise from the data draws and the scoring sample alone. The spread of returns across schedules is {spread['Q-learning']:.1f} for Q-learning and {spread['Monte Carlo']:.1f} for MC, "
        f"{spread['Q-learning'] / gmax:.1f} and {spread['Monte Carlo'] / gmax:.1f} times that discrepancy, so the schedule may change the final return of MC by a few units, while for Q-learning the evidence is weak; "
        "the opposite orderings of the two learners argue against reading either ranking as general.")

    P.h2("Robustness to a change in the no-show rate")
    nm5 = {"Rule": "Overbooking rule", "DP": "DP (replicate 0)"}
    rows = [["Policy", "Nominal", "No-show +0.08", "No-show -0.05", "Change, +0.08", "Change, -0.05"]]
    for k, v in allrows.items():
        rows.append([nm5.get(k, k.replace("|", ": ")), f"{v[nomk]:.1f}", f"{v[upk]:.1f}", f"{v[dnk]:.1f}", sg(v[upk] - v[nomk], 1), sg(v[dnk] - v[nomk], 1)])
    t3 = P.table(rows, f"Return on the scoring sessions of the final policies under a shifted no-show rate (Eq. 6). Entries for Q-learning and MC are means over {nS} replicates ({CF['sched_eval']} scoring episodes each); "
                       f"the rule is scored on {CF['shift_ref_eval']} episodes and DP on one replicate and {CF['shift_ref_eval']} episodes. Section 7.7 gives intervals for every policy of the benchmark.",
                 widths=[4.6, 1.9, 2.4, 2.4, 2.4, 2.4], font=8.5)
    P.p(f"Table {t3} reports returns when the no-show rate is higher or lower than in the scoring weeks. On real attendance the exposed direction is a rise in no-shows: the rule loses {abs(drop_up['Rule']):.1f} units, DP {abs(drop_up['DP']):.1f}, "
        f"and the learners between {min(abs(drop_up[k]) for k in lc):.1f} and {max(abs(drop_up[k]) for k in lc):.1f}, because a booking made for the nominal rate leaves more slots idle. "
        f"The learned policies lose less than the rule: all {n_above[upk]} of {len(cells)} learner and schedule cells are above the rule at +0.08, by {min(g_up.values()):.1f} to {max(g_up.values()):.1f} units "
        f"(mean {np.mean(list(g_up.values())):.1f}), against {n_above[nomk]} of {len(cells)} at the nominal rate (by {min(g_nom.values()):.1f} to {max(g_nom.values()):.1f}; mean {np.mean(list(g_nom.values())):.1f}). "
        f"A fall in no-shows helps the rule, which gains {ch_dn['Rule']:.1f} because its cautious booking now fills more slots, while the learned policies, which already book more, change by "
        f"{num(min(ch_dn[k] for k in lc), 1)} to {sg(max(ch_dn[k] for k in lc), 1)}; all {n_above[dnk]} cells stay above the rule, by a smaller mean margin of {np.mean(list(g_dn.values())):.1f}.")
    f4 = P.fig(fig("fig_shift"), "Return on the scoring sessions under the nominal and shifted no-show rates for the rule, DP and the exponentially decaying schedule of each learner.", 7.0)
    P.p(f"Figure {f4} shows the same for the four policies that the figure includes. Two caveats limit the shift study. First, the shift changes recorded outcomes at random and leaves the predicted risks unchanged, "
        "so it tests a change that the classifier has not seen; a real change in patient behaviour may also change who is booked. Second, the rule is a single policy with no learning noise while the learned "
        "entries average over replicates, so the comparison is not symmetric. Section 7.7 repeats the study with intervals for every policy, including the oracle.")

    # ================================================================== 7.5 to 7.9: structured model, baselines, further studies
    P.h2("The structured model and baselines from published work")
    order6 = ["Rule", "Threshold double booking", "Fixed booking", "Single session optimiser", "DP", "Monte Carlo", "Q-learning", "Oracle DP",
              "Ours: Monte Carlo", "Ours: Q-learning", "Ours: DP"]
    rows = [["Method", "Return (mean ± 95% CI)", "Shows", "Overtime", "Idle", "Waiting", "Utilisation"]]
    for k in order6:
        rows.append([names[k], pm(J_(k), ci_(k), 1)] + [f"{inf[k][i]:.1f}" for i in ("shows", "overtime", "idle", "wait")] + [pc(1 - inf[k]["idle"] / (CAP * H))])
    t6 = P.table(rows, f"All methods on the real scoring sessions. Columns as in Table {t1}; utilisation is the share of the {CAP} regular slots in use ({nrep} replicates, 95% CI; the rule, the three baselines "
                       "and the oracle are single policies).", widths=[5.0, 3.0, 1.6, 1.7, 1.5, 1.7, 1.9], font=8.5)
    others = [k for k in order6 if k != "Ours: DP"]
    n_ahead = min(ahead(meth["Ours: DP"]["J"], meth[k]["J"]) for k in ("DP", "Monte Carlo", "Q-learning"))
    so = PE["Single session optimiser"]
    P.p(f"Sections 7.1 to 7.4 show that the learners are held back by the support of the logged data. Table {t6} adds the structured model of Section 5.2 and the baselines of Section 6.3. "
        f"With the same {cfg['n_eps']} logged days per replicate, DP on the structured model scored {pmx(J_('Ours: DP'), ci_('Ours: DP'))}: {sg(PS['DP'][0])} ± {PS['DP'][1]:.2f} above DP on the estimated MDP, "
        f"{sg(PS['Q-learning'][0])} ± {PS['Q-learning'][1]:.2f} above Q-learning and {sg(PS['Monte Carlo'][0])} ± {PS['Monte Carlo'][1]:.2f} above MC (all {pstr(max(PS[k][2] for k in ('DP', 'Q-learning', 'Monte Carlo')))}, "
        f"ahead in {n_ahead} of {nrep} replicates against each), and {sg(PE['Oracle DP'][0])} above the oracle fitted to ten times as much data ({pstr(PE['Oracle DP'][2])}, paired over {nE:,} episodes). "
        f"Its interval of ±{ci_('Ours: DP'):.2f} shows that the policy hardly depends on the data draw. Of the baselines from published work, threshold double booking scored {J_('Threshold double booking'):.2f}, "
        f"a fixed booking of {kfix} extra patients {J_('Fixed booking'):.2f} and the single session optimiser {J_('Single session optimiser'):.2f}; the structured model is ahead of them by "
        f"{sg(PE['Threshold double booking'][0])}, {sg(PE['Fixed booking'][0])} and {sg(so[0])} ± {so[1]:.2f} ({pstr(so[2])}, paired over {nE:,} episodes). The last margin is "
        f"{100 * so[0] / J_('Single session optimiser'):.1f}% of the return.")
    w_ = lambda k: inf[k]["wait"]; o_ = lambda k: inf[k]["overtime"]; u_ = lambda k: 1 - inf[k]["idle"] / (CAP * H)
    assert w_("Ours: DP") == min(w_(k) for k in order6) and u_("Threshold double booking") == max(u_(k) for k in order6)
    f5 = P.fig(fig("fig_metrics"), f"Utilisation of the regular slots, overtime patients and waiting list per session for every method (95% CI over {nrep} replicates; single policies have no interval). "
               "Orange: baselines from published work; diamonds: policies on the structured model.", 15.5)
    P.p("The single session optimiser is a strong baseline: it uses each booked patient's own predicted risk, while the structured model sees only the band of the session's mean risk. "
        "The structured model is ahead because it also values the waiting list it leaves for later sessions, which the optimiser counts only through the 0.3 per patient drawn today; Section 7.8 shows that "
        f"this matters more when the waiting list is under pressure. Figure {f5} shows the operational side. Threshold double booking uses the most regular slots ({pc(u_('Threshold double booking'))}) at the cost of "
        f"{o_('Threshold double booking'):.1f} overtime patients per episode; the structured model uses {pc(u_('Ours: DP'))} of the slots with {o_('Ours: DP'):.1f} overtime patients, about as many as the oracle "
        f"({o_('Oracle DP'):.1f}), and has the shortest waiting list of all methods ({w_('Ours: DP'):.1f} per episode against {w_('Single session optimiser'):.1f} for the single session optimiser and {w_('Rule'):.1f} for the rule). "
        f"The classifier was fitted before the drift described in Section 4.1: its mean predicted no-show probability for the test weeks is {pc(C['mean_pred'])} against {pc(C['mean_obs'])} observed, so the structured "
        "model plans with risks that are slightly too high; we did not test how much a recalibrated classifier would add.")

    P.h2("How much logged data is needed")
    rows = [["Logged days", "DP on estimated MDP", "Structured model DP", "Difference (paired)"]]
    for x in xs:
        rows.append([f"{int(x):,}", pmx(*dsm[x]["DP"]), pmx(*dsm[x]["Ours: DP"]), f"{sg(dsd[x][0])} ± {dsd[x][1]:.2f} ({pstr(dsd[x][2])})"])
    t7 = P.table(rows, f"Return on the scoring sessions of DP on the estimated MDP and on the structured model for different amounts of logged data ({CF['size_seeds']} replicates per size, 95% CI; "
                       "differences paired over replicates).", widths=[2.6, 4.0, 4.0, 5.8], font=8.5)
    f6 = P.fig(fig("fig_datasize"), f"Return on the scoring sessions against the number of logged clinic days (log scale; 95% CI over {CF['size_seeds']} replicates). Dotted: guideline rule; dashed: oracle DP of Table {t1}.", 9.5)
    s_ = [dsm[x]["DP"][0] for x in xs]; o_ds = [dsm[x]["Ours: DP"][0] for x in xs]
    P.p(f"Table {t7} and Figure {f6} vary the number of logged days. DP on the estimated MDP rises from {s_[0]:.2f} with {int(xs[0])} days, below the rule's {rule_J:.2f}, to {s_[2]:.2f} with {int(xs[2])} and "
        f"{s_[-1]:.2f} with {int(xs[-1]):,}, which is the data coverage effect anticipated in Section 4.2; since the logged days are drawn from the same {DT['sessions_past']} learning sessions, more days here mean "
        f"more of the rarely logged actions, not new patients. DP on the structured model scores {o_ds[0]:.2f} with {int(xs[0])} days and {o_ds[-1]:.2f} with {int(xs[-1]):,}: the quantities it estimates are already "
        f"precise with {int(xs[0])} days. Even with {int(xs[-1]):,} days the estimated MDP remains {dsd[xs[-1]][0]:.2f} ± {dsd[xs[-1]][1]:.2f} behind ({pstr(dsd[xs[-1]][2])}).")

    P.h2("The structured model under a shifted no-show rate")
    ck = list(BS)
    rows = [["Policy", "Nominal", "No-show +0.08", "No-show -0.05", "Change, +0.08", "Change, -0.05"]]
    for k in order6:
        v = {c: BS[c][k] for c in ck}
        cell = lambda c: pmx(np.mean(v[c]), 1.96 * np.std(v[c], ddof=1) / np.sqrt(len(v[c])) if len(v[c]) > 1 else 0.0, 1)
        rows.append([names[k], cell(ck[0]), cell(ck[1]), cell(ck[2]), sg(np.mean(v[ck[1]]) - np.mean(v[ck[0]]), 1), sg(np.mean(v[ck[2]]) - np.mean(v[ck[0]]), 1)])
    t8 = P.table(rows, f"Return on the scoring sessions under a shifted no-show rate (Eq. 6) for every policy of Table {t6}, trained at the nominal rate ({nrep} replicates, {nE:,} scoring episodes each, 95% CI; "
                       "single policies have no interval).", widths=[4.6, 2.3, 2.3, 2.3, 2.3, 2.3], font=8.5)
    up_, dn_ = ck[1], ck[2]
    P.p(f"Table {t8} repeats the shift study of Section 7.4 for every policy, with {nrep} replicates and {nE:,} scoring episodes. The structured model is ahead of every other policy at all three rates: "
        f"{shm[up_]['Ours: DP']:.2f} at +0.08 against {max(v for k, v in shm[up_].items() if k != 'Ours: DP'):.2f} for the next best, and {shm[dn_]['Ours: DP']:.2f} at -0.05 against "
        f"{max(v for k, v in shm[dn_].items() if k != 'Ours: DP'):.2f}. It loses {shm[ck[0]]['Ours: DP'] - shm[up_]['Ours: DP']:.1f} when no-shows rise, less than the rule ({shm[ck[0]]['Rule'] - shm[up_]['Rule']:.1f}) "
        f"and DP on the estimated MDP ({shm[ck[0]]['DP'] - shm[up_]['DP']:.1f}), and {shm[ck[0]]['Ours: DP'] - shm[dn_]['Ours: DP']:.1f} when they fall, which is consistent with it booking more extra patients than any learner "
        f"(+4 in {n_act['Ours: DP'][4]} of {nst} states in replicate 0), so that fewer no-shows turn some of them into overtime. Threshold double booking is the one policy that gains when no-shows rise "
        f"({sg(shm[up_]['Threshold double booking'] - shm[ck[0]]['Threshold double booking'], 1)}), since it books the most, but it stays below the structured model. As in Section 7.4, the predicted risks "
        "do not change, so no policy can see the shift in its state; a classifier recalibrated on the shifted weeks would let the structured model adjust, which we did not test.")

    P.h2("When planning across sessions matters")
    rows = [["Waiting cost", "New patients per session", "DP on estimated MDP", "Best fixed booking", "Single session optimiser", "Structured model DP", "Gain over single session optimiser"]]
    for k, g in zip(dk, gain):
        d_ = DM[k]
        rows.append([f"{d_['wait_cost']}", f"{d_['arrivals']}", pmx(np.mean(d_["DP"]), 1.96 * np.std(d_["DP"], ddof=1) / np.sqrt(len(d_["DP"])), 1), f"{d_['Best fixed booking']:.1f}",
                     f"{d_['Single session optimiser']:.1f}", pmx(np.mean(d_["Ours: DP"]), 1.96 * np.std(d_["Ours: DP"], ddof=1) / np.sqrt(len(d_["Ours: DP"])), 1), f"{sg(g, 1)}%"])
    t9 = P.table(rows, f"Return on the scoring sessions when the waiting list is under more pressure ({CF['demand_seeds']} replicates, {CF['n_eval_demand']} scoring episodes; 95% CI for the two DP methods, "
                       f"the other policies are single policies). The first row is the setting of Table {t6}.", widths=[1.9, 2.1, 2.7, 2.2, 2.4, 2.7, 2.4], font=8.5)
    last = DM[dk[-1]]
    assert last["Single session optimiser"] < last["Best fixed booking"]
    P.p(f"Table {t9} varies the two quantities that make the waiting list costly. In the setting of Table {t6} the structured model is {gain[0]:.1f}% ahead of the single session optimiser (with "
        f"{CF['n_eval_demand']} scoring episodes instead of {nE:,}). The gain grows to {gain[1]:.1f}% when waiting costs 1.0 per patient, to {gain[2]:.1f}% with 2.5 new patients per session, and to "
        f"{gain[3]:.1f}% with both. In that last setting the single session optimiser ({last['Single session optimiser']:.2f}) falls below the best fixed booking ({last['Best fixed booking']:.2f}), DP on the "
        f"estimated MDP falls to {np.mean(last['DP']):.2f}, and the structured model scores {np.mean(last['Ours: DP']):.2f}. When the waiting list grows quickly or costs more, today's booking decides how long "
        "the list will be in later sessions, and only a policy that plans across sessions can weigh that.")

    P.h2("Convergence of Monte Carlo and Q-learning")
    f7 = P.fig(fig("fig_convergence"), f"Return on the scoring sessions of the greedy policy during learning (mean ± s.d. over {nrep} replicates; {CF['ckpt_eps']} scoring episodes per checkpoint). "
               "Solid: learners on the estimated MDP; dashed: on the structured model; pink dashed: DP on the structured model; dotted: guideline rule. The axis is cut below, so the value at step 0 is not shown.", 12.0)
    ts = tsett
    P.p(f"Figure {f7} shows how MC control and Q-learning converge when their greedy policies are scored on the scoring sessions rather than on the oracle model. On the estimated MDP, Q-learning peaks at "
        f"{ts['Q-learning|sir'][2]:.1f} after {ts['Q-learning|sir'][3]:,.0f} steps and ends at {ts['Q-learning|sir'][1]:.1f}, and MC ends at {ts['Monte Carlo|sir'][1]:.1f}. On the structured model Q-learning ends at "
        f"{ts['Q-learning|ours'][1]:.1f} and MC at {ts['Monte Carlo|ours'][1]:.1f}. In the final scores of Table {t6}, Q-learning on the structured model is ahead of Q-learning on the estimated MDP by "
        f"{sg(qQ[0])} ± {qQ[1]:.2f} ({pstr(qQ[2])}) and MC by {sg(qM[0])} ± {qM[1]:.2f} ({pstr(qM[2])}). Neither reaches DP on the same model within {cfg['steps']:,} steps ({J_('Ours: DP'):.2f}), so on this "
        f"problem the model matters more than the learner, and on both models Q-learning ends closer to the DP solution than MC (by {J_('Ours: Q-learning') - J_('Ours: Monte Carlo'):.2f} on the structured model and "
        f"{J_('Q-learning') - J_('Monte Carlo'):.2f} on the estimated MDP). A plausible reason, not tested separately, is that MC learns from the return of whole episodes of {H} sessions, whose variance is large "
        "compared with the differences between bookings (Section 3), while Q-learning updates each value from the next step.")
    assert shm[ck[0]]["Ours: DP"] - shm[up_]["Ours: DP"] < min(shm[ck[0]][k] - shm[up_][k] for k in ("Rule", "DP"))
    assert [k for k in shm[up_] if shm[up_][k] > shm[ck[0]][k]] == ["Threshold double booking"]
    assert spread["Monte Carlo"] / gmax > 2.5 > spread["Q-learning"] / gmax
    assert min(FIN["Q-learning"][s]["mean"] for s in SCH if s != NG) <= FIN["Q-learning"][NG]["mean"] <= max(FIN["Q-learning"][s]["mean"] for s in SCH if s != NG)

    # ================================================================== 8 Discussion
    P.h1("Discussion")
    P.p(f"**What the study shows.** (1) On real attendance, learners trained on {cfg['n_eps']} episodes are well ahead of the overbooking rule, by {min(v[0] for v in pv.values()):+.1f} to {max(v[0] for v in pv.values()):+.1f} "
        f"units (all {pstr(max(v[2] for v in pv.values()))}), because the rule books too few patients for a no-show rate near {pc(DT['noshow_future'], 0)}; a model fitted to ten times the data is ahead by {gap_orc:+.1f}, "
        f"and Section 7.6 shows the estimated MDP improving with more logged days, which points to data coverage as what keeps the learners short of the oracle. "
        f"(2) On the fitted model every schedule ends above the behaviour policy, and none reaches 90% of the possible improvement ({n_reach} of {n_runs} runs). "
        "(3) The most clearly resolved schedule effect is that a near greedy Q-learner learns more slowly. "
        f"(4) The learned policies lose less than the rule when no-shows rise (mean margin {np.mean(list(g_up.values())):.1f} against {np.mean(list(g_nom.values())):.1f}; all {n_above[upk]} of {len(cells)} configurations), "
        f"and stay ahead when no-shows fall, by less ({np.mean(list(g_dn.values())):.1f}). "
        f"(5) The coverage barrier can be removed without more data: with the clinic's rules known and only patient behaviour learned, {xs[0]} logged days gave {o_ds[0]:.1f}, more than the oracle DP ({orc_J:.1f}). "
        f"(6) The structured model is ahead of every method in Table {t6}, by {PS['DP'][0]:.2f} over DP on the estimated MDP and by {so[0]:.2f} over the strongest baseline from published work, and stays ahead at both shifted rates. "
        f"(7) That margin is small in the default setting and grows to {gain[-1]:.0f}% when the waiting list is under pressure (Table {t9}). (8) Q-learning ends closer to the DP solution than MC on both models.")
    P.p("**What it does not show.** It does not show that exploration schedules matter much for final performance: the differences across schedules are a few units, of the order of the discrepancy between "
        f"data draws for the same configuration. It does not show that the model scale metric is a reliable proxy for the return on real sessions; it agreed in sign here, and its schedule ranking agreed for MC ({sg(rho_m)}) "
        f"and less for Q-learning ({sg(rho_q)}), but the model undervalues every policy by {min(gaps):.0f} to {max(gaps):.0f} units. It does not show that a clinic knows its own rules as exactly as the structured model "
        "assumes: the slot count, the reward weights and the way extra bookings empty the waiting list are inputs, not estimates. It does not show a benefit to patients: attendance is recorded, but the "
        "decisions are generated and the waiting list is simulated. And the small margin over the single session optimiser in the default setting means that, in a quiet clinic, a good prediction model "
        "with a decision for one session at a time captures most of the gain.")
    P.p("**Practical use.** Because exploration inside a fitted model does not collect new data, practitioners should spend effort on what generates the log. A behaviour policy with deliberate variation in the number of extra bookings, "
        "run at low stakes, would give the learner support for the actions the optimum uses; the exploration schedule of the learner cannot substitute for that. A second way to reach those actions is to "
        "estimate less: whatever the clinic knows about its own rules need not be learned from the log. For deployment the shift study suggests a cheap check: score a learned policy against the existing "
        "rule under plausible no-show shifts and report the range, not the nominal value alone.")

    # ================================================================== 9 Limitations
    P.h1("Limitations and threats to validity")
    P.bullets([
        f"**Generated decisions and a simulated waiting list.** The table records attendance only. The decisions in the logs come from the behaviour policy of Table 1, and the waiting list, its {1.5} new patients per session "
        "and the reward weights (3 for overtime, 0.4 for waiting) are assumptions; Section 7.8 varies two of them.",
        f"**Few distinct days.** The table covers {Sm['days']} weekdays. Logged days are runs drawn from {DT['sessions_past']} learning sessions, so larger logs repeat the same patients, and the scoring weeks are a single period of {DT['sessions_future']} sessions.",
        f"**Sessions from neighbourhood days.** A session is all appointments of one neighbourhood on one day; only {pc(DT['share_not_full'], 0)} of these have fewer than 10 bookings, so the slots filled band rarely changes and adds little to the state.",
        "**Shift by changed outcomes.** The shift study flips recorded outcomes at random and keeps the predicted risks, so it tests one kind of change; a real change in patient behaviour may also change who is booked and the waiting list inflow.",
        "**Drift and calibration.** The classifier was calibrated before the no-show rate fell, so it overstates the risk in the scoring weeks (Section 7.5).",
        f"**Few replicates in the schedule study.** It uses {nS} replicates (t interval factor {tf6:.2f}) and {CF['sched_eval']} scoring episodes per run; the benchmark and the schedule study use different data seeds.",
        "**A model scale metric.** Learning efficiency is measured on the oracle model, which is pessimistic by construction for pairs with fewer than 10 observations. The ranking of schedules on this scale need not match the ranking in the clinic.",
        "**Exploration inside a model.** The schedules sample from the fitted MDP and cannot reveal unlogged state and action pairs, so the study cannot say anything about exploration in a real clinic, where trying an action has a cost.",
        "**Known clinic rules.** The structured model takes the number of slots, the reward weights and the way extra bookings empty the waiting list as known; if these are wrong, its policy is optimal for a different clinic.",
        "**Baselines.** The baselines from published work are simplified implementations, not the authors' code, and their threshold and k were chosen on the scoring sessions.",
        "**Wait time.** Waiting is measured by the length of the waiting list; the table records no times within a day, so the time patients spend waiting in the clinic cannot be measured.",
        "**Fixed horizon episodes treated as stationary.** Episodes end after a fixed number of steps but the state does not contain the clock, so the estimated model treats termination as a constant hazard. "
        "This distorts absolute model values (not returns on the scoring sessions) and is why model based value estimates are compared only on a common oracle model.",
        "**Hand set hyperparameters.** ε schedules, step sizes and the support threshold were fixed in advance and not tuned.",
    ])

    # ================================================================== 10 Ethics and future work
    P.h1("Ethical considerations and future work")
    P.p("**Ethics.** The study uses a public appointment table [@kaggle] in which patients appear only as numeric identifiers; it was used as published, was not linked to any other source, and is not redistributed "
        "with the code. If the method were used for real bookings, three safeguards would be essential: de-identification and governance approval of the source data; a clinician-in-the-loop design in which the "
        "learned policy is advisory, with hard guardrails on unsafe actions; and prospective, randomised or at least carefully monitored evaluation before any deployment. Reinforcement learning from logs inherits "
        "the biases of the clinicians who produced them, and may under-serve groups that are under-represented in the data; subgroup performance should be reported alongside averages, including the waiting "
        "times of groups with a higher predicted no-show risk, on whom overbooking decisions may fall more often.")
    P.p("**Future work.** (1) Add a behaviour policy that deliberately varies the extra bookings and test whether the learners then close the gap to the oracle; (2) evaluate a policy trained on shifted data, "
        "to measure the cost of a stale model; (3) compare against a pessimistic penalty in the supported part of the space; (4) measure overtime and waiting from visit timestamps, which the table lacks; "
        "(5) estimate the reward weights and the waiting list dynamics from clinic records instead of assuming them; (6) let the structured model use each booked patient's own predicted risk at decision time, "
        "together with the value of the next state, and recalibrate the classifier as the no-show rate drifts; (7) test the structured model in a clinic that records its overbooking decisions.")

    # ================================================================== 11 Conclusion
    P.h1("Conclusion")
    P.p(f"On clinic sessions rebuilt from the Medical Appointment No Shows table, DP, MC and Q-learning trained on {cfg['n_eps']} logged days scored {J_('DP'):.1f}, {J_('Monte Carlo'):.1f} and {J_('Q-learning'):.1f} against "
        f"{rule_J:.1f} for a conservative overbooking rule and {orc_J:.1f} for DP trained on ten times as much data. Five exploration schedules gave learning curves that ended above the behaviour policy but short "
        f"of 90% of the possible improvement ({n_reach} of {n_runs}); only the near greedy Q-learner was clearly slower, and the ordering of schedules on return differed between learners. "
        f"The learned policies lost less than the rule when no-shows rose by 0.08 (all {n_above[upk]} of {len(cells)} configurations were ahead) and stayed ahead when they fell by 0.05.")
    P.p(f"The limit is the coverage of the logged data. A structured model that learns only patient behaviour and takes the clinic's rules as known scored {J_('Ours: DP'):.1f}, {min(PS[k][0] for k in ('DP', 'Q-learning', 'Monte Carlo')):.1f} to "
        f"{max(PS[k][0] for k in ('DP', 'Q-learning', 'Monte Carlo')):.1f} above the three learners and {PE['Oracle DP'][0]:.1f} above the oracle, reached that level with {xs[0]} logged days, and was ahead of the best baseline from published work "
        f"({J_('Single session optimiser'):.1f}) at the nominal and both shifted rates, by a margin that grew from {gain[0]:.1f}% to {gain[-1]:.1f}% as the waiting list came under pressure. Q-learning came closer "
        "to the DP solution than MC. Attendance in these results is recorded, but the decisions are generated and the waiting list is simulated, so they are not evidence about real clinics.")

    # ================================================================== appendices
    repro = (f"All results in this paper are produced by `python run_real.py` (shared library: rl_core.py, common.py, ideas.py; data and models: noshow_data.py, real_clinic.py, planner.py), which reads the table from "
        "data/KaggleV2-May-2016.csv. Random seeds are fixed in the code, so a rerun reproduces every table and figure. Figure files generated by the script: fig_learning, fig_policies, fig_schedules, fig_shift, "
        "fig_metrics, fig_datasize, fig_convergence and fig_demand; `python run_real.py --figures` redraws them from the stored results.")
    if short:                                             # version 4: no reviewer questions and no source code
        P.h1("Appendix A. Reproducibility")
        P.p(repro)
        P.refs()
        path = os.path.join(PAP, "Paper07_v4_real_data_short.docx")
        P.save(path)
        return path
    P.h1("Appendix A. Anticipated reviewer questions")
    P.p("The following questions are the ones a reviewer or examiner is most likely to raise. Each answer is tied to a number or figure produced by the code.")
    qas = [
        ("Why does no schedule reach 90% of the optimum?",
         f"The learners solve a fitted model whose DP solution is worth about {Jor['DP']:.1f} on the oracle model, against a rule value of {Jbeh:.1f} and an optimum of {Jopt:.1f}. They end near that solution "
         f"(Q-learning {sq[1]:.1f}, MC {sm[1]:.1f}), so the target is out of reach for any schedule; the limit is the support of the logged data, not exploration."),
        ("If exploration matters little, why is near greedy Q-learning slower?",
         f"Near greedy Q-learning has a mean AUC of {num(AUC['Q-learning'][NG]['mean'])} against {num(AUC['Q-learning'][best_auc['Q-learning']]['mean'])} for the best schedule, and a plausible reason (not tested separately) is that "
         "it tries few actions early and needs many steps to correct its initial values. That is a speed effect within a fixed model; its final value is within the range of the other schedules."),
        ("Why are the learners so far ahead of the rule on real data?",
         f"The rule books one or two extra patients, and with about {pc(DT['noshow_future'], 0)} of patients not coming that leaves {inf['Rule']['idle']:.1f} idle slots per episode and a summed waiting list of "
         f"{inf['Rule']['wait']:.0f}. The learners book more in some states and cut the waiting list to {min(inf[k]['wait'] for k in ('DP', 'Q-learning', 'Monte Carlo')):.0f} to {max(inf[k]['wait'] for k in ('DP', 'Q-learning', 'Monte Carlo')):.0f}."),
        ("Can we conclude that exponential decay is best for Q-learning?",
         f"No. Its return in the schedule study ({SH['Q-learning']['eps exp. decay']['nominal']:.1f}) differs by {abs(gauge['Q-learning']):.1f} from the same configuration in the benchmark ({J_('Q-learning'):.1f}), "
         f"which is comparable to the {spread['Q-learning']:.1f} unit spread across all five Q-learning schedules, and the same schedule ranks {rank_exp_mc} of 5 for MC."),
        ("Is the robustness result a reason to deploy the learned policy?",
         f"No. It is a change of a single quantity made by flipping recorded outcomes, scored on one period of real sessions. It shows that the learned policies lose less than the rule when no-shows rise, in all "
         f"{n_above[upk]} of {len(cells)} configurations, and nothing more."),
        ("Why does the oracle score more than the learners?",
         f"With {cfg['n_oracle']:,} episodes the high overbooking actions are supported (about {per_pair_or:.0f} visits per pair against {per_pair:.1f}), and the oracle uses +3 or +4 in {hi_or} states. The learners use them in at most {hi_l}."),
        ("Why does the structured model score more than the oracle fitted to ten times as much data?",
         "The oracle still estimates P(s' | s, a) pair by pair, so with 1,500 days some pairs that the optimum uses remain below the support threshold and the rest carry sampling error. "
         f"The structured model estimates {N_EST} numbers that every logged session informs and computes the rest from the clinic's rules; its return varies by ±{ci_('Ours: DP'):.2f} over {nrep} data draws."),
        ("Is the comparison fair to the learners?",
         "They receive the same logged days, the same solver for DP and the same scoring episodes. What the structured model has in addition is the clinic's rules, which Section 9 lists as a limitation: "
         "a clinic that cannot state them cannot use it."),
        ("Why is the margin over the single session optimiser so small?",
         f"In the default setting the waiting list stays short under both policies ({w_('Single session optimiser') / H:.2f} and {w_('Ours: DP') / H:.2f} patients per session), so little is left to plan for, "
         f"and the optimiser uses each patient's own risk while the structured model uses a band. When the waiting list is under pressure the margin reaches {gain[-1]:.1f}% (Table {t9})."),
        ("Which parts of the results are real?",
         "Attendance is real: every patient in a scored session is a recorded appointment and comes exactly when the table says so, and the scoring weeks are never used for learning. The overbooking decisions "
         "in the logs, the waiting list and the reward weights are not in the table and are generated or assumed (Section 6.1)."),
    ]
    for i, (q, a) in enumerate(qas, 1):
        P.p(f"**Q{i}. {q}**")
        P.p(a, indent=True)
    P.h1("Appendix B. Reproducibility")
    P.p(repro)
    import run_real, planner, real_clinic, noshow_data
    P.h1("Appendix C. Source code of the experiment")
    P.p("For completeness the exact source of the clinic built from real sessions, the experiment functions and the structured model is reproduced below; it is the same code that produced every number and "
        "figure in this paper (the shared solvers value_iteration, mc_control and q_learning are in rl_core.py and are delivered with the code package).")
    P.h3("real_clinic.RealClinic")
    P.code(inspect.getsource(real_clinic.RealClinic))
    P.h3("run_real: shift, benchmark and schedule study")
    for fn in (run_real.flip_outcomes, run_real.benchmark, run_real.schedule_study):
        P.code(inspect.getsource(fn))
    P.h3("planner: structured model")
    for fn in (planner.session_value, planner.waiting_list_model, planner.waiting_step, planner.real_model):
        P.code(inspect.getsource(fn))
    P.h3("noshow_data: classifier and sessions")
    for fn in (noshow_data.fit_predictor, noshow_data.build_sessions):
        P.code(inspect.getsource(fn))
    rl_core_appendix(P)
    P.refs()
    path = os.path.join(PAP, "Paper07_v3_real_data.docx")
    P.save(path)
    return path


if __name__ == "__main__":
    print(build())
