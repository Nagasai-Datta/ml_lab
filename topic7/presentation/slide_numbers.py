"""slide_numbers.py -- every number, table row and reference on the slides, computed the way the paper computes them.

    python topic7/presentation/slide_numbers.py      writes topic7/presentation/slide_numbers.json

Reads code/out/topic7_real/results.json (written by run_real.py), code/out/idea07/results.json (Paper 7's own results,
for the slide on what Paper 7 did) and the reference list and reviewer questions of the built paper (version 3).
build_slides.js then turns slide_numbers.json into the .pptx; no number on a slide is typed by hand.
"""
import json, os, re, sys
import numpy as np
from scipy import stats
from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, "..", "code"))
sys.path.insert(0, CODE)
from paper_common import pm, pstr          # noqa: E402  (same number formats as the paper)
from paper07_ext import num, sg, pc        # noqa: E402

OUT = os.path.join(CODE, "out")
V3 = os.path.normpath(os.path.join(HERE, "..", "paper", "Paper07_v3_real_data.docx"))
CAP, H, NA = 10, 40, 5                     # regular slots, sessions per episode, actions (real_clinic.RealClinic)
NAME = {"Rule": "Guideline rule", "Threshold double booking": "Threshold double booking", "Fixed booking": "Fixed booking",
        "Single session optimiser": "Single session optimiser", "DP": "DP on estimated MDP", "Monte Carlo": "Monte Carlo control",
        "Q-learning": "Q-learning", "Oracle DP": "Oracle DP (10x data)", "Ours: Monte Carlo": "Monte Carlo, structured model",
        "Ours: Q-learning": "Q-learning, structured model", "Ours: DP": "Structured model DP (ours)"}


def ttest(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), float(stats.ttest_1samp(d, 0).pvalue)


def paper_items(path):
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs]
    refs = [t for t in paras if re.match(r"^\[\d+\] ", t)]
    qa = []
    for i, t in enumerate(paras):
        m = re.match(r"^Q(\d+)\. (.*)$", t)
        if m:
            qa.append({"q": m.group(2), "a": paras[i + 1]})
    return refs, qa


def main():
    R = json.load(open(os.path.join(OUT, "topic7_real", "results.json")))
    P7 = json.load(open(os.path.join(OUT, "idea07", "results.json")))["standard"]
    B, Sd, DT = R["benchmark"], R["schedule"], R["data"]
    S, meth, cfg, ds = B["summary"], B["methods"], B["config"], DT["summary"]
    nst = B["n_states"]
    J = lambda k: S[k]["J"]
    inf = {k: S[k]["info"] for k in S}

    order = ["Rule", "Threshold double booking", "Fixed booking", "Single session optimiser", "DP", "Monte Carlo", "Q-learning",
             "Oracle DP", "Ours: Monte Carlo", "Ours: Q-learning", "Ours: DP"]
    BL = B["baselines"]
    names = dict(NAME)
    names["Threshold double booking"] = f"Threshold double booking (θ = {BL['best_threshold']:.2f})"
    names["Fixed booking"] = f"Fixed booking (k = {BL['best_k']})"
    table = [{"name": names[k], "ours": k.startswith("Ours"), "paper7": k in ("Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP"),
              "J": f"{J(k):.1f} ± {S[k]['ci']:.2f}" if S[k]["ci"] > 0 else f"{J(k):.1f}",
              "util": pc(1 - inf[k]["idle"] / (CAP * H)), "over": f"{inf[k]['overtime'] / H:.2f}", "wait": f"{inf[k]['wait'] / H:.2f}"} for k in order]

    PS = {k: ttest(meth["Ours: DP"]["J"], meth[k]["J"]) for k in ("DP", "Monte Carlo", "Q-learning", "Rule")}
    ep = B["ep_mean"]
    PE = {k: ttest(ep["Ours: DP"], ep[k]) for k in ("Single session optimiser", "Oracle DP")}
    pv = {k: ttest(meth[k]["J"], meth["Rule"]["J"]) for k in ("DP", "Monte Carlo", "Q-learning")}

    # data volume
    DS = R["data_size"]
    size = [{"days": f"{int(x):,}", "dp": pm(float(np.mean(v["DP"])), float(1.96 * np.std(v["DP"], ddof=1) / np.sqrt(len(v["DP"]))), 1),
             "ours": f"{np.mean(v['Ours: DP']):.1f}"} for x, v in DS.items()]

    # no-show shift, every method averaged over replicates
    BS = B["shift"]; conds = list(BS)
    rates = [pc(DT["session_rate_future"])] + [pc(DT["session_rate_shifted"][c]) for c in conds[1:]]
    shift_rows = []
    for k in ("Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP", "Single session optimiser", "Ours: DP"):
        shift_rows.append({"name": names[k], "ours": k == "Ours: DP", "vals": [f"{np.mean(BS[c][k]):.1f}" for c in conds]})
    best_all = all(np.mean(BS[c]["Ours: DP"]) == max(np.mean(v) for v in BS[c].values()) for c in conds)

    # busier clinic
    DM = R["demand"]
    demand = []
    for k, v in DM.items():
        so, ours = v["Single session optimiser"], float(np.mean(v["Ours: DP"]))
        demand.append({"cost": f"{v['wait_cost']}", "arrivals": f"{v['arrivals']}", "dp": f"{np.mean(v['DP']):.1f}",
                       "fixed": f"{v['Best fixed booking']:.1f}", "single": f"{so:.1f}", "ours": f"{ours:.1f}",
                       "gain": f"{100 * (ours - so) / abs(so):+.1f}%".replace("+", "+")})

    # convergence of MC and Q-learning (true return of the greedy policy, mean over replicates)
    TC = {k: np.array(v) for k, v in B["true_curves"].items()}
    final = {k: float(v[:, -1, 1].mean()) for k, v in TC.items()}
    def settle(c):
        m = c[:, :, 1].mean(0)
        return int(c[0, np.flatnonzero(np.abs(m - m[-1]) <= 1.0)[0], 0])
    conv = {k: {"final": f"{final[k]:.1f}", "settle": f"{settle(TC[k]):,}"} for k in TC}

    # exploration schedules (Paper 7's schedule study, on the real data)
    SCH = Sd["schedules"]; nS = len(Sd["t90"]["Q-learning"][SCH[0]])
    n_reach = int(round(sum(Sd["reach_rate"][a][s] * nS for a in ("Q-learning", "Monte Carlo") for s in SCH)))
    AUC = Sd["auc"]; NG = "Near-greedy (0.02)"
    bestQ = max(SCH, key=lambda s: AUC["Q-learning"][s]["mean"])
    rho_qm = float(stats.spearmanr([Sd["shift"]["Q-learning"][s]["nominal"] for s in SCH], [Sd["shift"]["Monte Carlo"][s]["nominal"] for s in SCH]).statistic)

    refs, qa = paper_items(V3)
    lead = ds["by_lead"]
    out = {
        "data": {"appointments": f"{ds['rows']:,}", "patients": f"{ds['patients']:,}", "neighbourhoods": ds["neighbourhoods"], "days": ds["days"],
                 "noshow": pc(ds["noshow_rate"]), "lead_same": pc(lead["same day"]), "lead_31": pc(lead["31+ days"]),
                 "weekday_lo": pc(min(ds["by_weekday"].values())), "weekday_hi": pc(max(ds["by_weekday"].values())),
                 "auc": f"{DT['classifier']['auc_test']:.3f}", "pred": pc(DT["classifier"]["mean_pred"]), "obs": pc(DT["classifier"]["mean_obs"]),
                 "sessions_past": DT["sessions_past"], "sessions_future": DT["sessions_future"],
                 "noshow_past": pc(DT["noshow_past"]), "noshow_future": pc(DT["noshow_future"]), "not_full": pc(DT["share_not_full"], 0)},
        "mdp": {"states": nst, "actions": NA, "slots": CAP, "sessions": H, "days": cfg["n_eps"], "oracle_days": f"{cfg['n_oracle']:,}",
                "seeds": cfg["seeds"], "episodes": f"{cfg['n_eval']:,}", "steps": f"{cfg['steps']:,}",
                "n_est": 12 + 1 + 13, "n_emp": f"{nst * NA * (nst + 1):,}"},
        "paper7": {k: f"{P7[k]['J']:.1f}" for k in ("Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP")},
        "paper7_gain": {k: sg(P7[k]["J"] - P7["Rule"]["J"], 1) for k in ("DP", "Monte Carlo", "Q-learning", "Oracle DP")},
        "real": {k: f"{J(k):.1f}" for k in S},
        "gain_rule": {k: sg(v[0], 1) for k, v in pv.items()},
        "table": table,
        "ours_vs": {k: {"d": sg(v[0], 1), "p": pstr(v[1])} for k, v in PS.items()},
        "ours_vs_ep": {k: {"d": sg(v[0], 2), "p": pstr(v[1]), "pct": f"{100 * v[0] / J(k):.1f}%"} for k, v in PE.items()},
        "size": size, "rates": rates, "shift_conds": ["Nominal", "No-show +0.08", "No-show -0.05"], "shift": shift_rows, "shift_best_all": best_all,
        "demand": demand, "conv": conv,
        "sched": {"reach": n_reach, "runs": 2 * len(SCH) * nS, "auc_ng": num(AUC["Q-learning"][NG]["mean"]), "auc_best": num(AUC["Q-learning"][bestQ]["mean"]),
                  "best": bestQ, "rho": sg(rho_qm)},
        "refs": refs, "qa": qa,
    }
    path = os.path.join(HERE, "slide_numbers.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    return path


if __name__ == "__main__":
    print(main())
