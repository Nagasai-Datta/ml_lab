"""Real-data feasibility probe for Topic 7.

1. Train a no-show classifier on the early weeks of the real Kaggle CSV (temporal split).
2. Build a trace-driven clinic: each session is a REAL neighbourhood-day from the later weeks; the 10
   regular patients and up to 4 extra bookings are real appointments with their real show/no-show.
3. Compare, in that same environment:
     - Paper 7 rule and Paper-7-style learners (state uses the last session's observed no-show rate)
     - prior-work baselines re-implemented: threshold double-booking (Amalina & An 2026 style) and
       one-step marginal analysis on predicted probabilities (Topuz et al. 2024 style)
     - ours: prediction-based risk state + DP, and the abstraction-vote ensemble
"""
import sys, itertools, time
import _paths  # adds the faculty code package to sys.path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from rl_core import Domain, build_mdp, value_iteration, q_learning, fill_policy, behavior_mode, ci95

CSV = str(_paths.CSV)
SPLIT = pd.Timestamp("2016-05-16")

# ---------------------------------------------------------------- 1. classifier
df = pd.read_csv(CSV)
df["y"] = (df["No-show"] == "Yes").astype(int)
df["AD"] = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None)
df["SD"] = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
df["lead"] = (df["AD"] - df["SD"].dt.normalize()).dt.days.clip(lower=0)
df = df[(df.Age >= 0) & (df.AD.dt.dayofweek < 5)].sort_values(["AD", "SD"]).reset_index(drop=True)
df["dow"] = df.AD.dt.dayofweek
df["prev_appts"] = df.groupby("PatientId").cumcount()
df["prev_noshow"] = df.groupby("PatientId")["y"].cumsum() - df["y"]
df["nb"] = df["Neighbourhood"].astype("category").cat.codes
F = ["lead", "Age", "SMS_received", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
     "dow", "prev_appts", "prev_noshow", "nb"]
df["male"] = (df.Gender == "M").astype(int); F.append("male")
tr, te = df[df.AD < SPLIT], df[df.AD >= SPLIT].copy()
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, categorical_features=[F.index("nb")], random_state=0)
CAL = pd.Timestamp("2016-05-09")
fit, cal = tr[tr.AD < CAL], tr[tr.AD >= CAL]
clf.fit(fit[F], fit["y"])
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(cal[F])[:, 1], cal["y"])
te["p"] = iso.predict(clf.predict_proba(te[F])[:, 1])
print(f"classifier: train {len(tr)} rows, test {len(te)} rows, test AUC {roc_auc_score(te.y, te.p):.3f}, "
      f"mean p {te.p.mean():.3f} vs actual {te.y.mean():.3f}")

# ---------------------------------------------------------------- 2. sessions
sess = {d: [] for d in range(5)}
for (nb, ad), g in te.groupby(["Neighbourhood", "AD"]):
    if len(g) >= 14:
        sess[ad.dayofweek].append((g["p"].to_numpy(), g["y"].to_numpy()))
print("real sessions with >=14 appointments per weekday:", {d: len(v) for d, v in sess.items()})
panel_means = []
rng0 = np.random.default_rng(0)
for d in range(5):
    for p, y in sess[d]:
        for _ in range(20):
            idx = rng0.permutation(len(p))[:10]; panel_means.append((p[idx].mean(), y[idx].mean()))
panel_means = np.array(panel_means)
RE = np.quantile(panel_means[:, 0], [0.25, 0.5, 0.75])      # predicted-risk band edges
HE = np.quantile(panel_means[:, 1], [0.25, 0.5, 0.75])      # observed-rate band edges
print("risk band edges", RE.round(3), " observed-rate band edges", HE.round(3))


class RealClinic(Domain):
    """Paper 7's clinic, but sessions, patients and show/no-show outcomes come from the real CSV."""
    nA, gamma, max_steps, CAP = 5, 0.97, 40, 10
    nS = 5 * 4 * 5
    info_keys = ["shows", "overtime", "idle", "wait"]

    def __init__(self, state="pred"):
        self.state = state               # "pred" = predicted risk of today's panel (ours); "hist" = last session's observed rate (paper-style)

    def _draw(self, rng):
        p, y = sess[self.t % 5][rng.integers(len(sess[self.t % 5]))]
        o = rng.permutation(len(p))
        self.pp, self.yy = p[o], y[o]
        self.band_pred = int(np.searchsorted(RE, self.pp[:10].mean(), side="right"))

    def _s(self):
        b = self.band_pred if self.state == "pred" else self.band_hist
        return (self.t % 5 * 4 + b) * 5 + int(np.searchsorted([1, 3, 6, 9], self.w, side="right"))

    def reset(self, rng):
        self.t, self.w, self.band_hist = 0, int(rng.integers(0, 6)), int(rng.integers(4))
        self._draw(rng)
        return self._s()

    def step(self, k, rng):
        booked = self.CAP + k
        shows = int((1 - self.yy[:booked]).sum())
        over, idle = max(0, shows - self.CAP), max(0, self.CAP - shows)
        drawn = min(self.w, k)
        r = 1.0 * shows - 3.0 * over - 0.5 * idle - 0.4 * self.w + 0.3 * drawn
        self.band_hist = int(np.searchsorted(HE, self.yy[:10].mean(), side="right"))
        self.w = int(np.clip(self.w - drawn + rng.poisson(1.5), 0, 12))
        self.t += 1
        self._draw(rng)
        return self._s(), r, self.t >= self.max_steps, dict(shows=shows, overtime=over, idle=idle, wait=self.w), 1

    def rule(self, s):
        return 1 if (s // 5) % 4 <= 1 else 2


def evaluate(dom, act, seed=12345, n_eps=300):
    """act(dom, s) -> k.  Common random numbers across policies (episode e uses stream [seed, e])."""
    out = np.zeros(n_eps)
    for e in range(n_eps):
        rng = np.random.default_rng([seed, e]); s = dom.reset(rng); disc = 1.0
        for _ in range(dom.max_steps):
            s, r, done, _, _ = dom.step(int(act(dom, s)), rng); out[e] += disc * r; disc *= dom.gamma
            if done: break
    return out.mean()


def rollout_dom(dom, rng, n):
    rows = []
    for e in range(n):
        s = dom.reset(rng)
        for t in range(dom.max_steps):
            a = dom.behavior(s, rng)
            s2, r, done, info, _ = dom.step(a, rng)
            last = done or t == dom.max_steps - 1
            rows.append((s, a, r, s2, float(last), 1.0, e, t)); s = s2
            if last: break
    a = np.array(rows, float)
    return dict(s=a[:, 0].astype(int), a=a[:, 1].astype(int), r=a[:, 2], s2=a[:, 3].astype(int),
                done=a[:, 4], dur=a[:, 5], ep=a[:, 6].astype(int), t=a[:, 7].astype(int))


# ---------------------------------------------------------------- prior-work baselines
def myopic(dom, s):
    """one-step marginal analysis on predicted probabilities (extras assumed at the panel's mean risk)"""
    p10 = dom.pp[:10]; pe = p10.mean(); best, bk = -1e9, 0
    for k in range(5):
        ps = np.concatenate([1 - p10, np.full(k, 1 - pe)])
        dist = np.array([1.0])
        for q in ps: dist = np.convolve(dist, [1 - q, q])
        n = np.arange(len(dist))
        val = (dist * (n - 3 * np.maximum(0, n - 10) - 0.5 * np.maximum(0, 10 - n))).sum() + 0.3 * min(dom.w, k)
        if val > best: best, bk = val, k
    return bk


def threshold(theta):
    return lambda dom, s: min(4, int((dom.pp[:10] >= theta).sum()))


class Spec:
    def __init__(self, nS, nA, gamma):
        self.nS, self.nA, self.gamma, self.chan = nS, nA, gamma, []


SIZES = (5, 4, 5)
def feats(s): d, rest = divmod(s, 20); p, w = divmod(rest, 5); return (d, p, w)
def code(f, keep):
    c = 0
    for i in keep: c = c * SIZES[i] + f[i]
    return c
COMBOS = [k for r in (3, 2, 1) for k in itertools.combinations(range(3), r)]


def abstraction_vote(data, dom):
    V = np.zeros((dom.nS, dom.nA))
    for keep in COMBOS:
        enc = np.array([code(feats(s), keep) for s in range(dom.nS)])
        d = dict(data); d["s"] = enc[data["s"]]; d["s2"] = enc[np.minimum(data["s2"], dom.nS - 1)]
        ma = build_mdp(d, Spec(int(np.prod([SIZES[i] for i in keep])), dom.nA, dom.gamma))
        pa, _, _ = value_iteration(ma)
        V[np.arange(dom.nS), pa[enc[np.arange(dom.nS)]]] += 1
    return V.argmax(1)


