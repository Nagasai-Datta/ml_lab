"""defend_visuals.py -- defence visualisations for any paper:  python3 defend_visuals.py <k>
Reads out/ideaKK/results.json (made by `python3 ideas.py k`) and draws, besides the figures the experiment
already saved, a 'defence dashboard': every (method/condition) entry carrying a return J with a 95% CI,
sorted, so an examiner can see at a glance which differences exceed their uncertainty. It also prints
the paired tests stored in the results (mean, CI, p) -- the numbers to quote when defending a claim."""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def walk(o, path=""):
    if isinstance(o, dict):
        if "J" in o and "ci" in o and np.isscalar(o["J"]):
            yield path, float(o["J"]), float(o["ci"])
        if {"mean", "ci", "p"} <= set(o):
            yield ("PAIRED", path, o)
        for k, v in o.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(o, list) and len(o) < 40:
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")

k = int(sys.argv[1]); d = f"out/idea{k:02d}"
r = json.load(open(f"{d}/results.json"))
rows, paired = [], []
for x in walk(r):
    (paired if x[0] == "PAIRED" else rows).append(x)
print(f"== paired tests in idea {k} (difference, 95% CI half-width, p) ==")
for _, path, o in paired:
    print(f"{path:70s} {o['mean']:+8.3f} ± {o['ci']:.3f}  p={o['p']:.4f}  n={o.get('n')}")
rows = rows[:30]
if rows:
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(rows) + 1.2))
    ax.errorbar([t[1] for t in rows], range(len(rows)), xerr=[t[2] for t in rows], fmt="o", capsize=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([t[0].strip("/")[-48:] for t in rows], fontsize=6)
    ax.set_xlabel("true return (95% CI)"); ax.set_title(f"Defence dashboard, idea {k}")
    fig.tight_layout(); fig.savefig(f"{d}/defence_dashboard.png", dpi=200); print("saved", f"{d}/defence_dashboard.png")
