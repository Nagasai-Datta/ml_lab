"""noshow_data.py -- the real Kaggle table "Medical Appointment No Shows" (topic7/data/KaggleV2-May-2016.csv).

What it does
  * load()            : read and clean the CSV, build per-appointment features
  * fit_predictor()   : train a no-show classifier on the early weeks, calibrate it on the next week,
                        and predict the no-show probability of every later appointment
  * build_sessions()  : group the later weeks into clinic sessions (one neighbourhood on one day)

Time split (no information from the future leaks into the model):
  fit   : appointments before 9 May 2016
  calib : 9 May to 15 May 2016 (isotonic calibration, so predicted risks match observed rates)
  test  : 16 May 2016 onwards -> the sessions the clinic simulation is built from
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "KaggleV2-May-2016.csv"))
CAL_START, TEST_START = pd.Timestamp("2016-05-09"), pd.Timestamp("2016-05-16")
FEATURES = ["lead", "Age", "male", "SMS_received", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
            "Handcap", "dow", "prev_appts", "prev_noshow", "nb"]


def load(csv=CSV):
    """one row per appointment, weekdays only, with the features a clinic knows at booking time"""
    if not os.path.exists(csv):
        raise FileNotFoundError(f"{csv} not found; see topic7/data/README.md for how to get it")
    df = pd.read_csv(csv)
    df["y"] = (df["No-show"] == "Yes").astype(int)                       # 1 = patient did not come
    df["AD"] = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None)
    df["SD"] = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
    df["lead"] = (df["AD"] - df["SD"].dt.normalize()).dt.days.clip(lower=0)   # days waited for the appointment
    df = df[(df.Age >= 0) & (df.AD.dt.dayofweek < 5)].sort_values(["AD", "SD"]).reset_index(drop=True)
    df["dow"] = df.AD.dt.dayofweek
    df["male"] = (df.Gender == "M").astype(int)
    df["prev_appts"] = df.groupby("PatientId").cumcount()                 # earlier appointments of this patient
    df["prev_noshow"] = df.groupby("PatientId")["y"].cumsum() - df["y"]  # earlier no-shows (excludes this one)
    df["nb"] = df["Neighbourhood"].astype("category").cat.codes
    return df


def fit_predictor(df):
    """gradient-boosted trees + isotonic calibration; returns the test part with a column p (predicted no-show)"""
    fit, cal, test = df[df.AD < CAL_START], df[(df.AD >= CAL_START) & (df.AD < TEST_START)], df[df.AD >= TEST_START].copy()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         categorical_features=[FEATURES.index("nb")], random_state=0)
    clf.fit(fit[FEATURES], fit["y"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(cal[FEATURES])[:, 1], cal["y"])
    raw = clf.predict_proba(test[FEATURES])[:, 1]
    test["p"] = iso.predict(raw)
    info = dict(n_fit=len(fit), n_cal=len(cal), n_test=len(test),
                auc_test=float(roc_auc_score(test["y"], raw)),
                mean_pred=float(test["p"].mean()), mean_obs=float(test["y"].mean()),
                brier=float(np.mean((test["p"] - test["y"]) ** 2)))
    return test, info


def build_sessions(test, min_size=5):
    """sessions[d] = list of (p, y) arrays for every neighbourhood-day on weekday d with >= min_size appointments;
    pool[d] = (p, y) of all appointments on weekday d (patients pulled in from the waiting list are drawn from it)"""
    sessions, pool = {d: [] for d in range(5)}, {}
    for (nb, ad), g in test.groupby(["Neighbourhood", "AD"]):
        if len(g) >= min_size:
            sessions[ad.dayofweek].append((g["p"].to_numpy(), g["y"].to_numpy()))
    for d in range(5):
        g = test[test.dow == d]
        pool[d] = (g["p"].to_numpy(), g["y"].to_numpy())
    return sessions, pool


def dataset_summary(df):
    """the facts quoted in the paper's data section"""
    lead_band = pd.cut(df.lead, [-1, 0, 7, 30, 1000], labels=["same day", "1-7 days", "8-30 days", "31+ days"])
    return dict(rows=int(len(df)), patients=int(df.PatientId.nunique()), neighbourhoods=int(df.Neighbourhood.nunique()),
                days=int(df.AD.nunique()), first=str(df.AD.min().date()), last=str(df.AD.max().date()),
                noshow_rate=float(df.y.mean()),
                by_weekday={int(k): float(v) for k, v in df.groupby("dow").y.mean().items()},
                by_lead={str(k): float(v) for k, v in df.groupby(lead_band, observed=True).y.mean().items()})
