"""paper07_ext.py -- the Topic 7 extension of Paper 7: every section, table and figure we add to the paper.

paper07.build() calls these functions at the end of the matching sections (each call is one added line marked
"Topic 7 extension"); no existing sentence, table or figure of Paper 7 is changed here. Every number comes from
out/topic7/results.json (written by run_topic7.py) or from the data module, never typed by hand.

  add_refs        merge our references (topic7_refs.py) into the paper's reference list
  real_data       Section 4.1    the real table, the classifier and the clinic sessions
  method          Section 5.1    the structured model, Eq. (7) and (8), Algorithm 2
  protocol        Section 6.3    comparisons, baselines from published work, scoring
  results         Sections 7.5 to 7.10, Tables 6 to 10, Figures 5 to 8
  discussion, limitations, ethics_future, conclusion   closing paragraphs or bullets of Sections 8 to 11
  questions, repro, code   additions to Appendices A, B and C
"""
import inspect, json, os
import numpy as np
from paper_common import OUT, PAP, pstr
from topic7_refs import OURS

D = os.path.join(OUT, "topic7")
_R = {}
T = {}          # numbers of our tables and figures, filled by results()
MODE = "v1"     # "v1": Paper 7 unchanged plus our sections, headed "Extension: ..."
                # "v2": one paper (paper07_v2.py); no "Extension" labels, our sections are 4.2, 5.2 and 6.3


def sec(name):
    return {"v1": {"data": "4.1", "method": "5.1", "protocol": "6.3"},
            "v2": {"data": "4.2", "method": "5.2", "protocol": "6.3"}}[MODE][name]


def head(title):
    return "Extension: " + title[0].lower() + title[1:] if MODE == "v1" else title


def ours():
    return f"Sections {sec('data')}, {sec('method')}, {sec('protocol')} and 7.5 to 7.10"


def res():
    if not _R:
        with open(os.path.join(D, "results.json")) as f:
            _R.update(json.load(f))
    return _R


# ----------------------------------------------------------------------------- number formats (negatives with a hyphen, as in Paper 7)
def num(x, nd=2):
    return f"{'-' if x < 0 else ''}{abs(x):.{nd}f}"


def sg(x, nd=2):
    return f"{'+' if x >= 0 else '-'}{abs(x):.{nd}f}"


def pmx(x, ci, nd=2):
    return f"{num(x, nd)} ± {ci:.{nd}f}"


def pc(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def ahead(a, b):
    return int(np.sum(np.array(a) > np.array(b)))


NAME = {"Rule": "Guideline rule", "Sir: DP": "DP on estimated MDP", "Sir: Monte Carlo": "Monte Carlo control",
        "Sir: Q-learning": "Q-learning", "Oracle DP (10x data)": "Oracle DP (10x data)",
        "Ours: Monte Carlo": "Monte Carlo control, structured model", "Ours: Q-learning": "Q-learning, structured model",
        "Ours: DP": "Structured model DP (ours)", "One-session optimiser": "Single session optimiser"}
ORDER_A = ["Rule", "Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning", "Oracle DP (10x data)",
           "Ours: Monte Carlo", "Ours: Q-learning", "Ours: DP"]


def keys_B(MB):
    th = next(k for k in MB if k.startswith("Threshold")); fx = next(k for k in MB if k.startswith("Fixed"))
    return th, fx, float(th.split(">= ")[1].rstrip(")")), int(fx.split("+")[1].rstrip(")"))


def op_rows(M, order, names):
    """rows in the layout of Table 3 (totals per 40 session episode) plus utilisation"""
    rows = [["Method", "Return (mean ± 95% CI)", "Shows", "Overtime", "Idle", "Waiting", "Utilisation"]]
    for k in order:
        v = M[k]
        rows.append([names[k], pmx(v["J"]["mean"], v["J"]["ci"], 1), f"{v['shows']['mean']:.1f}",
                     f"{40 * v['overtime']['mean']:.1f}", f"{40 * v['idle']['mean']:.1f}", f"{40 * v['waiting']['mean']:.1f}",
                     pc(v["utilization"]["mean"])])
    return rows


def settle(curves):
    """first checkpoint at which the mean curve is within 1 unit of its final value (as in Section 7.1)"""
    arr = np.array(curves); mc = arr[:, :, 1].mean(0); x = arr[0, :, 0]
    return float(x[np.flatnonzero(np.abs(mc - mc[-1]) <= 1.0)[0]]), float(mc[-1]), float(mc.max()), float(x[mc.argmax()])


# ----------------------------------------------------------------------------- references
def add_refs(P):
    P.refdict = {**P.refdict, **OURS}


# ----------------------------------------------------------------------------- Section 4.1
def real_data(P):
    assert P.sec == 4 and P.sub == (0 if MODE == "v1" else 1)
    B = res()["B"]; S = B["data"]["summary"]; C = B["data"]["classifier"]; e = B["data"]["risk_edges"]
    n_raw, n_neg, n_sat = 110527, 1, 39          # rows of the CSV, removed rows (checked in noshow_data.load)
    assert n_raw - n_neg - n_sat == S["rows"]
    P.h2(head("Real clinic sessions from the Medical Appointment No Shows table"))
    opener = (f"Sections 4 to 7.4 use the simulator only. The extension reported in {ours()} also uses the Medical Appointment No Shows table [@kaggle], "
              "which was obtained after the original study was built." if MODE == "v1" else
              "The simulator of Section 4.1 has invented dynamics. To test the methods on recorded attendance, the same decision problem is also rebuilt from the "
              "Medical Appointment No Shows table [@kaggle], whose variables the simulator mirrors.")
    P.p(opener + f" The table lists {n_raw:,} medical appointments in Brazil, one row per appointment, with the day the appointment was booked, "
        "the day it took place, the patient's age, sex and neighbourhood, an SMS reminder flag, enrolment in a welfare programme, flags for hypertension, diabetes, alcoholism and disability, "
        f"and whether the patient came. After removing one row with a negative age and the {n_sat} appointments on Saturdays, {S['rows']:,} appointments of {S['patients']:,} patients in "
        f"{S['neighbourhoods']} neighbourhoods remain, on {S['days']} weekdays between 29 April and 8 June 2016; {pc(S['noshow_rate'])} of them were missed.")
    P.p("The table records whether each patient came, but no overbooking decision and no waiting list, so it does not contain the transitions (s, a, r, s') of Section 3. "
        "It is used in two other ways. First, a classifier predicts the no-show probability of each appointment from what is known when it is booked: the days between booking and "
        "appointment, the patient's age, sex, SMS flag, welfare flag and the four condition flags, the weekday, the numbers of the patient's earlier appointments and earlier no-shows in the table, "
        f"and the neighbourhood. The classifier is a gradient boosting model [@friedman] fitted on the {C['n_fit']:,} appointments before 9 May 2016, calibrated by isotonic regression "
        f"[@zadrozny] on the {C['n_cal']:,} appointments from 9 to 15 May, and applied to the {C['n_test']:,} appointments from 16 May onwards; both steps use scikit-learn [@pedregosa]. "
        f"On these appointments its area under the ROC curve is {C['auc_test']:.3f}, its Brier score {C['brier']:.3f}, and its mean predicted no-show probability {pc(C['mean_pred'])} "
        f"against {pc(C['mean_obs'])} observed. No information from later weeks enters the classifier.")
    P.p("Second, the appointments from 16 May onwards are grouped into clinic sessions, one for each neighbourhood and day with at least 5 appointments. The first min(10, n) appointments "
        "of a session fill its regular slots, patients for extra bookings are drawn from all appointments on the same weekday of the same period, and each patient shows up exactly when "
        f"the table says so. Sessions from 16 to 25 May ({B['data']['sessions_past']}) are used only for learning and sessions from 30 May to 8 June ({B['data']['sessions_future']}) only "
        f"for scoring. The no-show rate fell from {pc(B['data']['noshow_past'])} in the first period to {pc(B['data']['noshow_future'])} in the second, a real shift in the direction that "
        "Section 7.4 found hardest for overbooking policies.")
    P.p("The state combines the three quantities that matter for the booking decision: the number of regular slots filled (3 bands: 5 to 7, 8 or 9, all 10), the predicted no-show "
        "risk of the booked patients (4 bands, the quartiles of the session mean of the predicted probabilities in the learning period, with cut points "
        f"{e[0]:.3f}, {e[1]:.3f} and {e[2]:.3f}) and the waiting list band of Table 1 (5 bands), giving 60 states. Actions, reward, discount and horizon are those of Table 1, "
        "and the overbooking rule reads the risk band as Table 1 reads the no-show band: one extra booking in the two lower bands, otherwise two. The table has no waiting list, "
        f"so the waiting list of Section {'4' if MODE == 'v1' else '4.1'} is kept, with Poisson(1.5) new patients per session; this part of the clinic is assumed, and Section 7.9 varies it. "
        f"Only {pc(B['data']['share_not_full'], 0)} of the learning sessions have fewer than 10 bookings, so the slots filled band rarely changes.")


# ----------------------------------------------------------------------------- Section 5.1
N_EST = 4 + 5 * 4 * 4 + 1 + 13          # q_b, T_d(b'|b), lambda, pi(w|v) over the 13 list lengths
N_EMP = 100 * 5 * 101                   # P(s'|s,a) of the empirical MDP, terminal state included


def method(P):
    assert P.sec == 5 and P.sub == (0 if MODE == "v1" else 1)
    qx = res()["B"]["data"]["q_extra"]
    P.h2(head("A structured model of the clinic solved by DP"))
    opener = ("Section 7.1 shows that the learners trained on 150 logged days capture only part of the improvement that the oracle reaches, and Section 4 gives the reason: the empirical MDP "
              "of Section 3 estimates P(s' | s, a) separately for each of the 500 pairs (s, a), and the pairs that the optimum uses are rarely logged. The extension estimates far fewer quantities. "
              if MODE == "v1" else
              "Section 4.1 explains what limits any learner that works on the empirical MDP of Section 3: it estimates P(s' | s, a) separately for each of the 500 pairs (s, a), "
              "and the pairs that the optimum uses are rarely logged. A structured model estimates far fewer quantities. ")
    P.p(opener + "The clinic's own rules are taken as known: the 10 regular slots, the reward of Table 1, and the fact that k extra bookings take min(w, k) patients off a waiting list of length w. "
        "Only patient behaviour is learned from the logs: the probability q_b that a booked patient comes when the no-show band is b, the probability T_d(b' | b) that the band moves from b "
        "to b' after a session on weekday d, and the mean number λ of patients who join the waiting list in a session. A model assembled in this way from separately specified parts of the "
        f"state is a factored MDP [@boutilier]; here {N_EST} numbers are estimated, against {N_EMP:,} transition probabilities in the empirical MDP.")
    P.p("From the logged sessions, q̂_b is the number of patients who came divided by the number booked, over all sessions in band b whatever the action; T̂_d(b' | b) is a count with "
        "0.25 added to every cell; λ̂ is the mean of w_{t+1} − (w_t − min(w_t, k_t)) over consecutive sessions in which the list did not reach its cap of 12; and π(w | v), the share of each "
        "exact list length w among the logged sessions in waiting list band v, links the band to the length. With (x)⁺ = max(0, x), the expected reward of booking k extra "
        "patients in state s = (d, b, v) is")
    P.eq("R̃(s, k) = E[ N − 3 (N − 10)⁺ − 0.5 (10 − N)⁺ ]  +  E_w[ 0.3 min(w, k) − 0.4 w ],", "7")
    P.p("where N, the number of patients who come, is a sum of independent Bernoulli variables, one for each booked patient, each with attendance probability q̂_b; its distribution "
        "is computed exactly by convolution. The second expectation is over the list length w drawn from π(w | v). The next state s' = (d + 1 mod 5, b', v') follows from the same parts:")
    P.eq("P̃(s' | s, k) = T̂_d(b' | b) Pr[ band(min(12, w − min(w, k) + A)) = v' ],", "8")
    P.p("with w drawn from π(w | v) and the number of new patients A drawn from Poisson(λ̂).")
    P.p("In comparison B the state is (slots filled, risk band, waiting list band). The first term of Eq. (7) is then averaged over the logged sessions of the same slots and risk cell "
        "(up to 400), each represented by its number of booked patients and their mean predicted attendance probability from the classifier of Section " + sec("data") + "; extra patients come with "
        f"probability {qx:.3f}, one minus the mean predicted risk in the learning period. Tomorrow's patients do not depend on today's booking, so in Eq. (8) T̂_d(b' | b) is replaced by the "
        "logged frequency of each slots and risk cell (with 0.5 added to every cell), there is no weekday, and the waiting list part is unchanged.")
    P.p("Eq. (7) and (8) give a reward and a transition for every pair (s, k), whether or not the logs contain it, so no pair is unsupported and no fallback is needed. The model is solved "
        "with value iteration (Eq. 1), the solver of Section 3. The same model can also be given to MC control and Q-learning in place of the empirical MDP, which separates the effect of "
        "the model from that of the learner.")
    P.code("""Algorithm 2  Structured model DP (extension)
 input : logged clinic days D; the clinic's rules (10 slots, reward of Table 1, waiting list emptied by extra bookings)
 1. estimate patient behaviour from D: q_b for each no-show band, T_d(b'|b), the arrival rate λ and π(w|v)
    (comparison B: attendance probabilities from the calibrated classifier of Section """ + sec("data") + """)
 2. for every state s and every k = 0 to 4 compute R(s,k) with Eq. (7) and P(s'|s,k) with Eq. (8)
 3. solve the model by value iteration (Eq. 1) and use the greedy policy in every state
 4. optionally, train MC control or Q-learning on the same model instead of step 3""")


# ----------------------------------------------------------------------------- Section 6.3
def protocol(P):
    assert P.sec == 6 and P.sub == 2
    cfg = res()["config"]
    P.h2(head("Comparisons, baselines and scoring"))
    P.p(("The extension makes two comparisons." if MODE == "v1" else "Sections 7.5 to 7.10 make two comparisons.")
        + " Comparison A repeats the benchmark of Section 7.1 in the same simulator and adds the structured model; comparison B builds the clinic of "
        f"Section {sec('data')} from real sessions and adds three baselines from published work. All other settings are those of Table 2.")
    P.bullets([
        f"**Replicates.** Comparison A uses {cfg['seeds']} data draws with data seeds 1000 to {999 + cfg['seeds']}; the first 5 are those of Table 3 and reproduce it exactly, and the other "
        f"{cfg['seeds'] - 5} narrow the intervals. Comparison B uses {cfg['seeds']} logged datasets of {cfg['n_logged']} episodes, generated in the learning sessions by the behaviour policy of Table 1.",
        f"**Scoring.** Comparison A scores every policy on {cfg['n_eval_A']} episodes of the true simulator, as in Section 6.2. Comparison B scores every policy on {cfg['n_eval_B']:,} episodes "
        "of 40 sessions drawn from the scoring sessions, with common random numbers, so that episode e brings the same patients and the same attendance for every policy.",
        "**Baselines from published work (comparison B).** (i) A single session optimiser books the k that maximises the expected reward of the current session, computed from each booked "
        "patient's predicted risk; this is marginal analysis of the kind used for overbooking under predicted no-shows [@topuz]. (ii) Threshold double booking fills empty regular slots and "
        "books one extra patient for every booked patient whose predicted risk is at least θ, the fixed threshold rule against which learned double booking policies are compared [@amalina]. "
        "(iii) A fixed number k of extra bookings in every session. The threshold θ (0.20, 0.25, 0.30 or 0.35) and the fixed k (0 to 4) were chosen on the scoring sessions, which favours "
        "these baselines.",
        f"**Data volume and demand.** Comparison A repeats DP on the estimated MDP and on the structured model with {', '.join(f'{x:,}' for x in cfg['sizes'][:-1])} and {cfg['sizes'][-1]:,} "
        f"logged days ({cfg['size_seeds']} replicates each). Comparison B repeats both with a waiting cost of 0.4 or 1.0 per patient and 1.5 or 2.5 new patients per session "
        f"({cfg['demand_seeds']} replicates, {cfg['n_eval_demand']} scoring episodes).",
        f"**Convergence.** MC control and Q-learning run for {cfg['steps']:,} steps on the empirical MDP and on the structured model, with the schedule and step size of Section 6.2; "
        f"at 26 checkpoints the true return of the greedy policy is measured on {cfg['ckpt_eps']} episodes of the clinic used for scoring.",
        "**Statistics.** Intervals are 95% intervals over replicates. Differences between two learned policies are tested with a t test on the paired differences over replicates; "
        f"differences from a single policy in comparison B are also tested over the {cfg['n_eval_B']:,} paired scoring episodes.",
    ])


# ----------------------------------------------------------------------------- Sections 7.5 to 7.10
def results(P):
    assert P.sec == 7 and P.sub == 4 and P.ntab == 5 and P.nfig == 4      # Tables 1 to 5 and Figures 1 to 4 are Paper 7's
    from ideas import paired
    R = res(); A, B = R["A"], R["B"]; M, MB = A["methods"], B["methods"]; PA = A["paired"]
    J = lambda m, k: m[k]["J"]["mean"]; CI = lambda m, k: m[k]["J"]["ci"]; seed = lambda m, k: m[k]["J"]["per_seed"]
    th, fx, theta, kfix = keys_B(MB)
    names = dict(NAME); names[th] = f"Threshold double booking (θ = {theta:.2f})"; names[fx] = f"Fixed booking (k = {kfix})"
    learners = ("Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning")

    # ---- 7.5 comparison A
    P.h2(head("The structured model in the true simulator"))
    t6 = P.table(op_rows(M, ORDER_A, names),
                 "Comparison A: the benchmark of Table 3 with the structured model added. Return is discounted; shows, overtime, idle slots and the summed waiting list length are totals "
                 f"per episode of 40 sessions, and utilisation is the share of the 10 regular slots in use ({R['config']['seeds']} replicates, 95% CI; rule and oracle are single policies).",
                 widths=[5.0, 3.0, 1.6, 1.7, 1.5, 1.7, 1.9], font=8.5)
    first5 = {k: float(np.mean(seed(M, k)[:5])) for k in learners}
    spread = max(J(M, k) for k in learners) - min(J(M, k) for k in learners)
    n_ahead = min(ahead(seed(M, "Ours: DP"), seed(M, k)) for k in learners)
    P.p(("Sections 7.1 to 7.4 show that the learners are held back by the support of the logged data. " if MODE == "v2" else "")
        + f"Table {t6} shows the benchmark of Table 3 with the structured model added. With the same 150 logged days per replicate, DP on the structured model scored "
        f"{pmx(J(M, 'Ours: DP'), CI(M, 'Ours: DP'))}, against {pmx(J(M, 'Sir: DP'), CI(M, 'Sir: DP'))} for DP on the estimated MDP, {pmx(J(M, 'Sir: Monte Carlo'), CI(M, 'Sir: Monte Carlo'))} "
        f"for MC and {pmx(J(M, 'Sir: Q-learning'), CI(M, 'Sir: Q-learning'))} for Q-learning. Paired over {R['config']['seeds']} replicates it is ahead of the three learners by "
        + ", ".join(f"{sg(PA[f'Ours: DP vs {k}']['mean'])} ± {PA[f'Ours: DP vs {k}']['ci']:.2f}" for k in learners[:2])
        + f" and {sg(PA['Ours: DP vs Sir: Q-learning']['mean'])} ± {PA['Ours: DP vs Sir: Q-learning']['ci']:.2f} (all {pstr(max(PA[f'Ours: DP vs {k}']['p'] for k in learners))}), "
        f"in {n_ahead} of {R['config']['seeds']} replicates against each, and {sg(PA['Ours: DP vs Oracle DP (10x data)']['mean'])} ahead of the oracle DP fitted to ten times as much data "
        f"({pstr(PA['Ours: DP vs Oracle DP (10x data)']['p'])}). Its interval of ±{CI(M, 'Ours: DP'):.2f} shows that the policy hardly depends on the data draw. Over the first 5 replicates "
        f"the three learners score exactly as in Table 3 ({first5['Sir: DP']:.2f}, {first5['Sir: Monte Carlo']:.2f} and {first5['Sir: Q-learning']:.2f}); over all {R['config']['seeds']} "
        f"they are within {spread:.1f} of each other, so the added replicates do not change the picture of Section 7.1.")
    v = lambda k, m: M[k][m]["mean"]
    est = A["estimates"]; q = np.mean([x["show_up"] for x in est], 0); lam = float(np.mean([x["arrivals"] for x in est]))
    P.p(f"The operational columns show that the structured model goes further along the route of the oracle: {v('Ours: DP', 'shows'):.1f} patients seen per episode against "
        f"{v('Rule', 'shows'):.1f} for the rule and {v('Sir: DP', 'shows'):.1f} for DP on the estimated MDP, {pc(v('Ours: DP', 'utilization'))} of the regular slots in use against "
        f"{pc(v('Rule', 'utilization'))} and {pc(v('Sir: DP', 'utilization'))}, and a summed waiting list of {40 * v('Ours: DP', 'waiting'):.1f} against {40 * v('Rule', 'waiting'):.1f} "
        f"and {40 * v('Sir: DP', 'waiting'):.1f}. The price is overtime: {40 * v('Ours: DP', 'overtime'):.1f} patients per episode against {40 * v('Rule', 'overtime'):.1f} for the rule and "
        f"{40 * v('Sir: DP', 'overtime'):.1f} for DP on the estimated MDP, slightly below the oracle's {40 * v('Oracle DP (10x data)', 'overtime'):.1f}. The estimated arrival rate was "
        f"{lam:.2f} patients per session on average over replicates, against 1.5 in the simulator, and the estimated attendance probabilities fell from {q[0]:.2f} in the lowest no-show "
        f"band to {q[-1]:.2f} in the highest.")
    assert 40 * v("Ours: DP", "overtime") < 40 * v("Oracle DP (10x data)", "overtime")

    # ---- 7.6 data size
    P.h2(head("How much logged data is needed"))
    DS = A["data_size"]; xs = list(DS)
    rows = [["Logged days", "DP on estimated MDP", "Structured model DP", "Difference (paired)"]]
    for x in xs:
        d = DS[x]
        rows.append([f"{int(x):,}", pmx(d["Sir: DP"]["mean"], d["Sir: DP"]["ci"]), pmx(d["Ours: DP"]["mean"], d["Ours: DP"]["ci"]),
                     f"{sg(d['diff']['mean'])} ± {d['diff']['ci']:.2f} ({pstr(d['diff']['p'])})"])
    t7 = P.table(rows, f"True return of DP on the estimated MDP and on the structured model for different amounts of logged data ({R['config']['size_seeds']} replicates per size, 95% CI; "
                       "differences paired over replicates).", widths=[2.6, 4.0, 4.0, 5.8], font=8.5)
    f5 = P.fig(os.path.join(D, "fig_A_datasize.png"), f"True return against the number of logged clinic days (log scale; 95% CI over {R['config']['size_seeds']} replicates). "
               "Dotted: guideline rule; dashed: oracle DP of Table 3.", 9.5)
    s, o = [DS[x]["Sir: DP"]["mean"] for x in xs], [DS[x]["Ours: DP"]["mean"] for x in xs]
    assert s[0] < J(M, "Rule") and max(o) - min(o) < 0.1
    P.p(f"Table {t7} and Figure {f5} vary the number of logged days. DP on the estimated MDP rises from {s[0]:.2f} with {int(xs[0])} days, below the rule's {J(M, 'Rule'):.2f}, to "
        f"{s[2]:.2f} with {int(xs[2])}, {s[3]:.2f} with {int(xs[3])} and {s[4]:.2f} with {int(xs[4]):,}, which is the data coverage effect anticipated in Section 4: more days bring more of "
        f"the pairs that the optimum uses above the support threshold. DP on the structured model scores {o[0]:.2f} with {int(xs[0])} days and {o[-1]:.2f} with {int(xs[-1]):,}; the "
        f"quantities it estimates are already precise with {int(xs[0])} days, and further data change its return by less than 0.1. Even with {int(xs[-1]):,} days the estimated MDP remains "
        f"{DS[xs[-1]]['diff']['mean']:.2f} ± {DS[xs[-1]]['diff']['ci']:.2f} behind ({pstr(DS[xs[-1]]['diff']['p'])}).")

    # ---- 7.7 shift
    P.h2(head("Robustness to a change in the no-show rate"))
    SH = A["shift"]; nom, up, dn = list(SH)
    rows = [["Policy", "Nominal", "No-show +0.08", "No-show -0.05", "Change, -0.05 vs nominal"]]
    for k in ORDER_A:
        rows.append([names[k], pmx(SH[nom][k]["mean"], SH[nom][k]["ci"], 1), pmx(SH[up][k]["mean"], SH[up][k]["ci"], 1), pmx(SH[dn][k]["mean"], SH[dn][k]["ci"], 1),
                     sg(SH[dn][k]["mean"] - SH[nom][k]["mean"], 1)])
    t8 = P.table(rows, f"True return under a shifted no-show rate for every policy of Table {t6}, trained on nominal data ({R['config']['seeds']} replicates, "
                       f"{R['config']['n_eval_A']} evaluation episodes each, 95% CI; rule and oracle are single policies).", widths=[5.0, 2.6, 2.9, 2.9, 3.0], font=8.5)
    loss = {k: SH[nom][k]["mean"] - SH[dn][k]["mean"] for k in ORDER_A}
    assert loss["Ours: DP"] < min(loss[k] for k in ORDER_A if k != "Ours: DP")
    P.p(f"Table {t8} repeats the shift study of Section 7.4 for every policy of Table {t6}, now with {R['config']['seeds']} replicates and intervals. When the no-show rate falls by 0.05, "
        f"the rule loses {loss['Rule']:.1f}, the oracle {loss['Oracle DP (10x data)']:.1f} and the three learners {min(loss[k] for k in learners):.1f} to {max(loss[k] for k in learners):.1f}, "
        f"while DP on the structured model loses {loss['Ours: DP']:.1f} ({pmx(SH[dn]['Ours: DP']['mean'], SH[dn]['Ours: DP']['ci'])} against {SH[nom]['Ours: DP']['mean']:.2f}); in the shifted "
        f"clinic it is {SH[dn]['Ours: DP']['mean'] - SH[dn]['Oracle DP (10x data)']['mean']:.2f} ahead of the oracle and {SH[dn]['Ours: DP']['mean'] - SH[dn]['Sir: DP']['mean']:.2f} ahead of DP "
        f"on the estimated MDP. When the rate rises by 0.08, it scores {pmx(SH[up]['Ours: DP']['mean'], SH[up]['Ours: DP']['ci'])} against {SH[up]['Rule']['mean']:.2f} for the rule and "
        f"{pmx(SH[up]['Sir: DP']['mean'], SH[up]['Sir: DP']['ci'])} for DP on the estimated MDP. The policy is not retrained; a plausible reason for the small loss, not tested separately, "
        "is that the no-show band is part of the state and Eq. (7) and (8) value every band and every k, including bands that the nominal logs rarely visit, so the policy already specifies "
        "what to book when no-shows are fewer.")

    # ---- 7.8 comparison B
    P.h2(head("Real clinic sessions"))
    ORDER_B = ["Rule", th, fx, "One-session optimiser", "Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning", "Oracle DP (10x data)",
               "Ours: Monte Carlo", "Ours: Q-learning", "Ours: DP"]
    t9 = P.table(op_rows(MB, ORDER_B, names),
                 f"Comparison B: real clinic sessions from the scoring period (30 May to 8 June 2016). Columns as in Table {t6}; {R['config']['seeds']} replicates of the logged data, "
                 f"{R['config']['n_eval_B']:,} scoring episodes, 95% CI; the rule, the three baselines and the oracle are single policies.",
                 widths=[5.0, 3.0, 1.6, 1.7, 1.5, 1.7, 1.9], font=8.5)
    PS, PE = B["paired_seeds"], B["paired_episodes"]
    gain_one = PE["Ours: DP vs One-session optimiser"]
    others = [k for k in ORDER_B if k != "Ours: DP"]
    assert all(J(MB, "Ours: DP") > J(MB, k) for k in others)
    P.p(f"Table {t9} reports comparison B. The rule scored {J(MB, 'Rule'):.2f}. The learners of Section 3, trained on 150 logged days of real sessions, scored "
        f"{pmx(J(MB, 'Sir: DP'), CI(MB, 'Sir: DP'))} (DP), {pmx(J(MB, 'Sir: Monte Carlo'), CI(MB, 'Sir: Monte Carlo'))} (MC) and {pmx(J(MB, 'Sir: Q-learning'), CI(MB, 'Sir: Q-learning'))} "
        f"(Q-learning), and the oracle {J(MB, 'Oracle DP (10x data)'):.2f}. Of the baselines from published work, threshold double booking scored {J(MB, th):.2f}, a fixed booking of {kfix} extra "
        f"patients {J(MB, fx):.2f} and the single session optimiser {J(MB, 'One-session optimiser'):.2f}. DP on the structured model scored {pmx(J(MB, 'Ours: DP'), CI(MB, 'Ours: DP'))}: "
        f"{sg(PE['Ours: DP vs Rule']['mean'])} above the rule, {sg(PS['Ours: DP vs Sir: DP']['mean'])} ± {PS['Ours: DP vs Sir: DP']['ci']:.2f} above DP on the estimated MDP "
        f"({pstr(PS['Ours: DP vs Sir: DP']['p'])}, paired over replicates), {sg(PE[f'Ours: DP vs {fx}']['mean'])} above the best fixed booking and {sg(gain_one['mean'])} ± {gain_one['ci']:.2f} "
        f"above the single session optimiser ({pstr(gain_one['p'])}, paired over {gain_one['n']:,} episodes). The last margin is {100 * gain_one['mean'] / J(MB, 'One-session optimiser'):.1f}% of the return.")
    w = lambda k: 40 * MB[k]["waiting"]["mean"]; ot = lambda k: 40 * MB[k]["overtime"]["mean"]; u = lambda k: MB[k]["utilization"]["mean"]
    assert w("Ours: DP") == min(w(k) for k in ORDER_B) and u(th) == max(u(k) for k in ORDER_B) and ot(th) == max(ot(k) for k in ORDER_B)
    next_ot = max(ot(k) for k in ORDER_B if k != th)
    f6 = P.fig(os.path.join(D, "fig_B_metrics.png"), f"Utilisation of the regular slots, overtime patients and waiting list per session in comparison B (95% CI over {R['config']['seeds']} "
               "replicates; single policies have no interval). Orange: baselines from published work; diamonds: policies on the structured model.", 15.5)
    P.p("The single session optimiser is a strong baseline: it uses each booked patient's own predicted risk, while the structured model sees only the band of the session's mean risk. "
        "The structured model is ahead because it values the waiting list it leaves for later sessions, which the optimiser counts only through the 0.3 per patient drawn today; Section 7.9 "
        f"shows that this matters more when the waiting list is under pressure. Figure {f6} shows the operational side. Threshold double booking uses the most regular slots "
        f"({pc(u(th))}) at the cost of {ot(th):.1f} overtime patients per episode, {ot(th) / next_ot:.1f} times the next highest; the structured model uses {pc(u('Ours: DP'))} of the slots "
        f"with {ot('Ours: DP'):.1f} overtime patients and has the shortest waiting list of all methods ({w('Ours: DP'):.1f} per episode against {w('One-session optimiser'):.1f} for the single "
        f"session optimiser and {w('Rule'):.1f} for the rule).")
    Cl = B["data"]["classifier"]
    P.p(f"The classifier was fitted before the drift described in Section {sec('data')}: its mean predicted no-show probability for the test weeks is {pc(Cl['mean_pred'])} against "
        f"{pc(Cl['mean_obs'])} observed, and the scoring sessions have {pc(B['data']['noshow_future'])}. The structured model therefore plans with risks that are slightly too high, "
        "yet it is ahead of every other method; we did not test how much a recalibrated classifier would add.")

    # ---- 7.9 demand
    P.h2(head("When planning across sessions matters"))
    DM = B["demand"]; dk = list(DM)
    rows = [["Waiting cost", "New patients per session", "DP on estimated MDP", "Best fixed booking", "Single session optimiser", "Structured model DP", "Gain over single session optimiser"]]
    for key in dk:
        d = DM[key]; wc, lm = key.replace("wait cost ", "").split(", arrivals ")
        rows.append([wc, lm, pmx(d["Sir: DP"]["mean"], d["Sir: DP"]["ci"], 1), f"{d['Best fixed overbooking']:.1f}", f"{d['One-session optimiser']:.1f}",
                     pmx(d["Ours: DP"]["mean"], d["Ours: DP"]["ci"], 1), f"{sg(d['gain_vs_one_session_pct'], 1)}%"])
    t10 = P.table(rows, f"True return in comparison B when the waiting list is under more pressure ({R['config']['demand_seeds']} replicates, {R['config']['n_eval_demand']} scoring episodes; "
                        f"95% CI for the two DP methods, the other policies are single policies). The first row is the setting of Table {t9}.",
                  widths=[1.9, 2.1, 2.7, 2.2, 2.4, 2.7, 2.4], font=8.5)
    g = [DM[k]["gain_vs_one_session_pct"] for k in dk]; last = DM[dk[-1]]
    assert all(a < b for a, b in zip(g, g[1:])) and last["One-session optimiser"] < last["Best fixed overbooking"]
    P.p(f"Table {t10} varies the two quantities that make the waiting list costly. In the setting of Table {t9} the structured model is {g[0]:.1f}% ahead of the single session optimiser "
        f"(with {R['config']['n_eval_demand']} scoring episodes instead of {R['config']['n_eval_B']:,}). The gain grows to {g[1]:.1f}% when waiting costs 1.0 per patient, to {g[2]:.1f}% with "
        f"2.5 new patients per session, and to {g[3]:.1f}% with both. In that last setting the single session optimiser ({last['One-session optimiser']:.2f}) falls below the best fixed booking "
        f"({last['Best fixed overbooking']:.2f}), DP on the estimated MDP falls to {pmx(last['Sir: DP']['mean'], last['Sir: DP']['ci'])}, and the structured model scores "
        f"{pmx(last['Ours: DP']['mean'], last['Ours: DP']['ci'])}. When the waiting list grows quickly or costs more, today's booking decides how long the list will be in later sessions, "
        "and only a policy that plans across sessions can weigh that.")

    # ---- 7.10 convergence
    P.h2(head("Convergence of Monte Carlo and Q-learning"))
    f7 = P.fig(os.path.join(D, "fig_A_convergence.png"), f"Comparison A: true return of the greedy policy during learning (mean ± s.d. over {R['config']['seeds']} replicates; "
               f"{R['config']['ckpt_eps']} evaluation episodes per checkpoint). Solid: learners on the estimated MDP; dashed: on the structured model; pink dashed: DP on the structured model; "
               "dotted: guideline rule. The axis is cut below, so the value at step 0 is not shown.", 12.0)
    f8 = P.fig(os.path.join(D, "fig_B_convergence.png"), "Comparison B: the same as Figure " + str(f7) + " on real clinic sessions.", 12.0)
    cA, cB = A["curves"], B["curves"]
    qs_A = settle(cA["Q-learning|sir"]); ms_A = settle(cA["Monte Carlo|sir"]); qo_A = settle(cA["Q-learning|ours"]); mo_A = settle(cA["Monte Carlo|ours"])
    qs_B = settle(cB["Q-learning|sir"]); ms_B = settle(cB["Monte Carlo|sir"]); qo_B = settle(cB["Q-learning|ours"]); mo_B = settle(cB["Monte Carlo|ours"])
    qA, mA = PA["Ours: Q-learning vs Sir: Q-learning"], PA["Ours: Monte Carlo vs Sir: Monte Carlo"]
    qB = paired(seed(MB, "Ours: Q-learning"), seed(MB, "Sir: Q-learning")); mB = paired(seed(MB, "Ours: Monte Carlo"), seed(MB, "Sir: Monte Carlo"))
    assert qA["p"] < 0.001 and qB["p"] < 0.001 and mA["p"] > 0.05 and mB["p"] < 0.05 and qo_A[1] > mo_A[1] and qo_B[1] > mo_B[1]
    P.p(f"Figures {f7} and {f8} show how MC control and Q-learning converge, on the estimated MDP and on the structured model. On the estimated MDP of comparison A, Q-learning peaks at "
        f"{qs_A[2]:.1f} after {qs_A[3]:,.0f} steps and ends at {qs_A[1]:.1f}, and MC ends at {ms_A[1]:.1f}. On the structured model Q-learning is within one unit of its final value "
        f"{qo_A[1]:.1f} after {qo_A[0]:,.0f} steps, while MC ends at {mo_A[1]:.1f}. Comparison B shows the same order: on the structured model Q-learning ends at {qo_B[1]:.1f} and MC at "
        f"{mo_B[1]:.1f}, on the estimated MDP at {qs_B[1]:.1f} and {ms_B[1]:.1f}. In the final scores of Tables {t6} and {t9}, Q-learning on the structured model is ahead of Q-learning on the "
        f"estimated MDP by {sg(qA['mean'])} ± {qA['ci']:.2f} in comparison A ({pstr(qA['p'])}) and {sg(qB['mean'])} ± {qB['ci']:.2f} in comparison B ({pstr(qB['p'])}). For MC the gain "
        f"is {sg(mB['mean'])} ± {mB['ci']:.2f} in comparison B ({pstr(mB['p'])}) but only {sg(mA['mean'])} ± {mA['ci']:.2f} in comparison A ({pstr(mA['p'])}).")
    P.p(f"Neither learner reaches DP on the same model within {R['config']['steps']:,} steps ({J(M, 'Ours: DP'):.2f} in comparison A, {J(MB, 'Ours: DP'):.2f} in comparison B), so on this "
        f"problem the model matters more than the learner, and on the structured model Q-learning ends closer to the DP solution than MC in both comparisons (by "
        f"{J(M, 'Ours: Q-learning') - J(M, 'Ours: Monte Carlo'):.2f} in A and {J(MB, 'Ours: Q-learning') - J(MB, 'Ours: Monte Carlo'):.2f} in B). A plausible reason, not tested separately, is that MC learns from "
        "the return of whole episodes of 40 sessions, whose variance is large compared with the differences between bookings (Section 3), while Q-learning updates each value from the next step.")
    T.update(t6=t6, t7=t7, t8=t8, t9=t9, t10=t10, f5=f5, f6=f6, f7=f7, f8=f8, qA=qA, mA=mA,
             dQM_A=J(M, "Ours: Q-learning") - J(M, "Ours: Monte Carlo"), dQM_B=J(MB, "Ours: Q-learning") - J(MB, "Ours: Monte Carlo"))


# ----------------------------------------------------------------------------- Sections 8 to 11
def discussion(P):
    assert P.sec == 8
    R = res(); A, B = R["A"], R["B"]; M, MB = A["methods"], B["methods"]; DS = A["data_size"]; x0 = list(DS)[0]
    g = [B["demand"][k]["gain_vs_one_session_pct"] for k in B["demand"]]
    P.p(f"**What the extension shows.** (1) The data coverage barrier of point (1) above can be removed without more data: with the clinic's rules known and only patient behaviour learned, "
        f"{x0} logged days gave {DS[x0]['Ours: DP']['mean']:.1f}, more than the oracle DP fitted to 1,500 days ({M['Oracle DP (10x data)']['J']['mean']:.1f}; Table {T['t7']}). "
        "(2) The gain carries over to real patients: on sessions rebuilt from the Medical Appointment No Shows table the structured model is ahead of every method in Table "
        f"{T['t9']}, by {B['paired_seeds']['Ours: DP vs Sir: DP']['mean']:.2f} over DP on the estimated MDP and by {B['paired_episodes']['Ours: DP vs One-session optimiser']['mean']:.2f} "
        f"over the strongest baseline from published work. (3) The margin over that baseline is small in the default setting and grows to {g[-1]:.0f}% when the waiting list is under "
        f"pressure (Table {T['t10']}). (4) On the structured model Q-learning ends closer to the DP solution than MC in both comparisons.")
    P.p("**What the extension does not show.** It does not show that a clinic knows its own rules as exactly as the model assumes: the slot count, the reward weights and the way extra "
        "bookings empty the waiting list are inputs, not estimates. It does not show a benefit to real patients: comparison B replays recorded attendance, but the waiting list and the "
        "reward are still simulated. And the small margin over the single session optimiser in the default setting means that, in a quiet clinic, a good prediction model with a decision "
        "for one session at a time captures most of the gain.")
    P.p("**Practical use, continued.** Besides a behaviour policy with more varied bookings, a second way to reach the actions the optimum uses is to estimate less: whatever the clinic "
        "knows about its own rules need not be learned from the log.")


def limitations(P):
    assert P.sec == 9
    B = res()["B"]
    P.bullets([
        "**Known clinic rules (extension).** The structured model takes the number of slots, the reward weights and the way extra bookings empty the waiting list as known; if these are "
        "wrong, its policy is optimal for a different clinic.",
        f"**Assumed waiting list (extension).** The table has no waiting list, so comparison B simulates one with the arrival rate and reward weights of Table 1; Table {T['t10']} varies "
        "two of these values, and others were not tested.",
        f"**Sessions from neighbourhood days (extension).** A session is all appointments of one neighbourhood on one day; only {pc(B['data']['share_not_full'], 0)} of these have fewer "
        "than 10 bookings, so the slots filled band rarely changes and adds little to the state.",
        "**Drift and calibration (extension).** The classifier was calibrated before the no-show rate fell, so it overstates the risk in the scoring weeks (Section 7.8).",
        "**Baselines (extension).** The baselines from published work are simplified implementations written for this extension, not the authors' code, and their threshold and k were "
        "chosen on the scoring sessions.",
        "**Wait time (extension).** Waiting is measured by the length of the waiting list, as in Table 3; the table records no times within a day, so the time patients spend waiting in "
        "the clinic cannot be measured.",
    ])


def ethics_future(P):
    assert P.sec == 10
    P.p("**Ethics of the extension.** The extension uses a public appointment table [@kaggle] in which patients appear only as numeric identifiers. It was used as published, was not "
        "linked to any other source, and is not redistributed with the code. The safeguards above apply unchanged to any use of the planner for real bookings; in particular, overbooking "
        "decisions informed by predicted no-shows may fall more often on groups with a higher predicted risk, so their waiting times should be reported separately.")
    P.p("**Future work of the extension.** (6) Estimate the reward weights and the waiting list dynamics from clinic records instead of assuming them; (7) let the planner use each booked "
        "patient's own predicted risk at decision time, as the single session optimiser does, together with the value of the next state; (8) test the structured model in a clinic that "
        "records its overbooking decisions, where the empirical MDP and the structured model can be compared on the same real data.")


def conclusion(P):
    assert P.sec == 11
    R = res(); A, B = R["A"], R["B"]; M, MB = A["methods"], B["methods"]; PA = A["paired"]; DS = A["data_size"]
    g = [B["demand"][k]["gain_vs_one_session_pct"] for k in B["demand"]]
    d = [PA[f"Ours: DP vs {k}"]["mean"] for k in ("Sir: DP", "Sir: Monte Carlo", "Sir: Q-learning")]
    P.p(f"In the extension, a structured model that learns only patient behaviour and takes the clinic's rules as known scored {M['Ours: DP']['J']['mean']:.1f} in the same simulator, "
        f"{min(d):.1f} to {max(d):.1f} above the three learners and {PA['Ours: DP vs Oracle DP (10x data)']['mean']:.1f} above the oracle, and reached that level with {list(DS)[0]} logged days. "
        f"On clinic sessions rebuilt from the Medical Appointment No Shows table it scored {MB['Ours: DP']['J']['mean']:.1f} against {MB['Rule']['J']['mean']:.1f} for the rule, "
        f"{MB['Sir: DP']['J']['mean']:.1f} for DP on the estimated MDP and {MB['One-session optimiser']['J']['mean']:.1f} for the best baseline from published work, and its margin over that "
        f"baseline grew from {g[0]:.1f}% to {g[-1]:.1f}% as the waiting list came under pressure. Q-learning came closer to the DP solution of the structured model than MC in both comparisons. "
        "Comparison B replays recorded attendance but simulates the waiting list and the reward, so these results are not evidence about real clinics either.")


# ----------------------------------------------------------------------------- appendices
def questions(P, start):
    R = res(); A, B = R["A"], R["B"]; MB = B["methods"]; g = [B["demand"][k]["gain_vs_one_session_pct"] for k in B["demand"]]
    qas = [
        ("Why does the structured model score more than the oracle fitted to ten times as much data?",
         "The oracle still estimates P(s' | s, a) pair by pair, so with 1,500 days some pairs that the optimum uses remain below the support threshold and the rest carry sampling error. "
         f"The structured model estimates {N_EST} numbers that every logged session informs and computes the rest from the clinic's rules; its return varies by ±{A['methods']['Ours: DP']['J']['ci']:.2f} "
         f"over {R['config']['seeds']} data draws."),
        ("Is the comparison fair to the learners of Section 7.1?",
         "They receive the same logged days, the same solver for DP and the same scoring episodes. What the structured model has in addition is the clinic's rules, which Section 9 lists "
         "as a limitation: a clinic that cannot state them cannot use it."),
        ("Why is the margin over the single session optimiser so small?",
         f"In the default setting the waiting list stays short under both policies ({MB['One-session optimiser']['waiting']['mean']:.2f} and {MB['Ours: DP']['waiting']['mean']:.2f} patients "
         "per session), so little is left to plan for, and the optimiser uses each patient's own risk while the structured model uses a band. When the waiting list is under pressure the "
         f"margin reaches {g[-1]:.1f}% (Table {T['t10']})."),
        ("Why does Q-learning come closer to the DP solution than MC?",
         f"On the structured model Q-learning ends {T['dQM_A']:.2f} above MC in comparison A and {T['dQM_B']:.2f} in comparison B, and in comparison A MC gains only "
         f"{T['mA']['mean']:.2f} from the better model ({pstr(T['mA']['p'])}). A plausible reason, not tested separately, is the variance of whole episode returns in MC (Section 3); DP on the same model shows that the model itself supports {A['methods']['Ours: DP']['J']['mean']:.2f}."),
        ("Are the outcomes in comparison B real?",
         "Attendance is real: every patient in a scored session is a recorded appointment and comes exactly when the table says so. The waiting list, its arrivals and the reward weights "
         "are simulated, so comparison B is a simulation built from real records."),
    ]
    for i, (q, a) in enumerate(qas, start):
        P.p(f"**Q{i}. {q}**")
        P.p(a, indent=True)


def repro(P):
    P.p((f"The extension ({ours()}) is produced" if MODE == "v1" else f"{ours()} are produced") + " by `python run_topic7.py` (new modules noshow_data.py, real_clinic.py and planner.py), which reuses "
        "rl_core.py, domains.py, common.py and ideas.py unchanged and reads the table from data/KaggleV2-May-2016.csv. Random seeds are fixed in the code, so a rerun reproduces every "
        f"number of {'the extension' if MODE == 'v1' else 'those sections'}. Figure files used here: fig_A_datasize, fig_B_metrics, fig_A_convergence, fig_B_convergence; `python run_topic7.py --figures` redraws them from the "
        "stored results.")


def code(P):
    import planner, real_clinic, noshow_data
    if MODE == "v1":
        P.p("The source of the extension's model, of the clinic built from real sessions and of the classifier follows; the experiment script run_topic7.py is delivered with the code.")
    P.h3(head("Planner (structured model)") if MODE == "v1" else "planner: structured model")
    for fn in (planner.session_value, planner.waiting_list_model, planner.waiting_step, planner.clinic_model, planner.real_model):
        P.code(inspect.getsource(fn))
    P.h3("Extension: real_clinic.RealClinic" if MODE == "v1" else "real_clinic.RealClinic")
    P.code(inspect.getsource(real_clinic.RealClinic))
    P.h3("Extension: noshow_data (classifier and sessions)" if MODE == "v1" else "noshow_data: classifier and sessions")
    for fn in (noshow_data.fit_predictor, noshow_data.build_sessions):
        P.code(inspect.getsource(fn))


def output_path(path):
    """version 1 of the extended paper; Paper 7's own docx in paper/ is left as it was handed out"""
    return os.path.join(PAP, "Paper07_v1_additions.docx")
