"""
domains.py -- ten stylised medical decision processes (one per paper).

IMPORTANT: these are *simulators whose columns mirror the named Kaggle datasets*,
not the Kaggle data itself (Kaggle was unreachable from the build environment).
They are transparent, hand-specified stochastic models used to (a) generate a
logged "clinician" dataset, and (b) act as ground truth so every learned policy
can be evaluated exactly.  Nothing here is clinical advice.
"""
import numpy as np
from rl_core import Domain, band


# =============================================================================
# 1. Diabetes digital twin  (mirrors: Diabetes CGM datasets)
class Diabetes(Domain):
    name = "diabetes"
    nA, gamma, max_steps = 4, 0.97, 24
    info_keys = ["in_range", "hypo", "insulin"]
    GE = [70, 100, 140, 180, 250]
    nS = 6 * 3 * 2
    MEALS = {7, 13, 19}
    action_labels = ["0u", "1u", "2u", "3u"]

    def __init__(self, sens=1.0, noise=10.0):
        self.sens, self.noise = sens, noise
        self.state_labels = [f"g{g}-tr{t}-m{m}" for g in range(6) for t in range(3) for m in range(2)]

    def _s(self):
        gb = band(self.g, self.GE)
        d = self.g - self.gp
        tr = 0 if d < -15 else (2 if d > 15 else 1)
        return (gb * 3 + tr) * 2 + (1 if self.t in self.MEALS else 0)

    def reset(self, rng):
        self.g = float(np.clip(rng.normal(120, 20), 50, 300))
        self.gp, self.t = self.g, 0
        return self._s()

    def step(self, a, rng):
        meal = self.t in self.MEALS
        base = 180.0 - 40.0 * a * self.sens
        gn = self.g + 0.35 * (base - self.g) + (55.0 * rng.uniform(0.7, 1.3) if meal else 0.0) \
            + rng.normal(0, self.noise)
        gn = float(np.clip(gn, 30, 400))
        if gn < 54: r = -10.0
        elif gn < 70: r = -4.0
        elif gn <= 180: r = 1.0
        elif gn <= 250: r = -1.0
        else: r = -3.0
        r -= 0.05 * a
        self.gp, self.g, self.t = self.g, gn, self.t + 1
        info = dict(in_range=float(70 <= gn <= 180), hypo=float(gn < 70), insulin=float(a))
        return self._s(), r, self.t >= self.max_steps, info, 1

    def rule(self, s):
        gb, rest = divmod(s // 2, 3)[0], s % 2
        meal = s % 2
        if gb <= 1: return 0
        if gb == 2: return 1 if meal else 0
        if gb == 3: return 2 if meal else 1
        if gb == 4: return 2
        return 3


# =============================================================================
# 2. Sepsis ICU  (mirrors: PhysioNet/CinC 2019 sepsis, eICU demo)
class Sepsis(Domain):
    name = "sepsis"
    nA, gamma, max_steps = 9, 0.99, 24
    info_keys = ["died", "survived", "fluid", "vaso"]
    behavior_eps = 0.25
    action_labels = [f"f{f}v{v}" for f in range(3) for v in range(3)]

    def __init__(self, gran=3):
        self.gran = gran
        g = gran
        self.me = np.linspace(48, 92, g + 1)[1:-1]
        self.le = np.geomspace(0.8, 7.0, g + 1)[1:-1]
        self.se = np.linspace(2, 13, g + 1)[1:-1]
        self.nS = g ** 3

    def _s(self):
        g = self.gran
        return (band(self.map, self.me) * g + band(self.lac, self.le)) * g + band(self.sofa, self.se)

    def reset(self, rng):
        self.map = float(np.clip(rng.normal(62, 8), 40, 100))
        self.lac = float(np.clip(rng.lognormal(np.log(2.8), 0.4), 0.6, 12))
        self.sofa = float(rng.uniform(3, 9))
        self.t = 0
        return self._s()

    def step(self, a, rng):
        f, v = divmod(a, 3)
        tgt = 74 - 1.6 * self.sofa + 7 * v + 2.5 * f
        mp = self.map + 0.5 * (tgt - self.map) + rng.normal(0, 3)
        mp = float(np.clip(mp, 30, 110))
        ls = 0.8 + 0.25 * self.sofa + 0.08 * max(65 - mp, 0) + 0.25 * (v == 2)
        lc = float(max(0.4, self.lac + 0.4 * (ls - self.lac) + rng.normal(0, 0.25)))
        sf = self.sofa + 0.5 * (lc > 3.5) + 0.3 * (mp < 60) - 0.45 * (mp >= 68 and lc < 2.5) \
            - 0.2 * (mp >= 65) + 0.5 * (f == 2 and self.sofa >= 8) + rng.normal(0, 0.3)
        sf = float(np.clip(sf, 0, 20))
        self.map, self.lac, self.sofa, self.t = mp, lc, sf, self.t + 1
        r, done = -0.1, False
        died = survived = 0.0
        if sf >= 14 or mp < 42:
            r, done, died = -100.0, True, 1.0
        elif sf <= 2 and lc < 2.0:
            r, done, survived = 100.0, True, 1.0
        elif self.t >= self.max_steps:
            r, done = 40 - 5 * sf, True
            survived = float(sf < 6)
        info = dict(died=died, survived=survived, fluid=float(f), vaso=float(v))
        return self._s(), r, done, info, 1

    def rule(self, s):
        g = self.gran
        mb = s // (g * g)
        lb = (s // g) % g
        mfrac = mb / max(g - 1, 1)
        lfrac = lb / max(g - 1, 1)
        v = 2 if mfrac < 0.2 else (1 if mfrac < 0.5 else 0)
        f = 2 if lfrac >= 0.85 else (1 if lfrac >= 0.5 else 0)
        return f * 3 + v


# =============================================================================
# 3. Hospital capacity  (mirrors: Hospital length-of-stay / Healthcare analytics)
class Hospital(Domain):
    name = "hospital"
    nA, gamma, max_steps = 3, 0.97, 28
    info_keys = ["admitted", "overflow", "diverted", "early_dc", "queue"]
    CAP = 40
    LAM = np.array([9.5, 9.0, 8.5, 8.5, 9.0, 7.0, 6.5])
    W = 3                       # occupancy bucket width (beds)
    QE = [1, 3, 6, 10]
    action_labels = ["admit", "early-discharge", "divert"]

    def __init__(self, surge=1.0):
        self.surge = surge
        self.nOB = 40 // self.W + 1
        self.nQ = len(self.QE) + 1
        self.nS = 7 * self.nOB * self.nQ

    def _s(self):
        return (self.dow * self.nOB + min(self.occ // self.W, self.nOB - 1)) * self.nQ + band(self.q, self.QE)

    def reset(self, rng):
        self.occ = int(rng.integers(20, 40))
        self.q = int(rng.integers(0, 5))
        self.dow = int(rng.integers(7))
        self.t = 0
        return self._s()

    def step(self, a, rng):
        arr = int(rng.poisson(self.LAM[self.dow] * self.surge))
        div = int(rng.binomial(arr, 0.5)) if a == 2 else 0
        self.q += arr - div
        over = max(0, self.q - 15)
        self.q -= over
        adm = min(self.q, self.CAP - self.occ)
        self.q -= adm
        occ_mid = self.occ + adm
        rate = 0.28 if a == 1 else 0.2
        dis = int(rng.binomial(occ_mid, rate))
        early = max(0, dis - int(round(0.2 * occ_mid))) if a == 1 else 0
        self.occ = occ_mid - dis
        r = 1.0 * adm - 0.3 * self.q - 5.0 * over - 2.0 * div - 1.6 * early
        self.dow = (self.dow + 1) % 7
        self.t += 1
        info = dict(admitted=adm, overflow=over, diverted=div, early_dc=early, queue=self.q)
        return self._s(), r, self.t >= self.max_steps, info, 1

    def rule(self, s):
        occb = (s // self.nQ) % self.nOB
        qb = s % self.nQ
        occ = occb * self.W
        if occ >= 35 and qb >= 2: return 2
        if occ >= 30: return 1
        return 0


# =============================================================================
# 4. Hypertension titration  (mirrors: Heart Disease UCI / Cardiovascular Disease)
class Hypertension(Domain):
    name = "hypertension"
    nA, gamma, max_steps = 3, 0.97, 24
    info_keys = ["hypo", "unsafe_up", "in_target", "changes", "dose"]
    SE = [110, 120, 130, 140, 160]
    BASE = [140.0, 152.0, 166.0]
    nS = 6 * 3 * 4
    action_labels = ["down", "hold", "up"]

    def _s(self):
        return (band(self.sbp, self.SE) * 3 + self.risk) * 4 + self.dose

    def reset(self, rng):
        self.risk = int(rng.integers(3))
        self.sens = float(np.clip(rng.normal(1.0, 0.3), 0.5, 1.8))
        self.sbp = float(self.BASE[self.risk] + rng.normal(0, 8))
        self.dose, self.t = 0, 0
        return self._s()

    def step(self, a, rng):
        b0 = band(self.sbp, self.SE)
        unsafe = float(a == 2 and b0 <= 1)
        nd = int(np.clip(self.dose + a - 1, 0, 3))
        changed = float(nd != self.dose)
        self.dose = nd
        tgt = self.BASE[self.risk] - 11.0 * self.dose * self.sens
        self.sbp = float(self.sbp + 0.5 * (tgt - self.sbp) + rng.normal(0, 4))
        x = self.sbp
        if x < 105: r = -8.0
        elif x < 110: r = -0.5
        elif x <= 130: r = 1.0
        elif x < 140: r = 0.4
        elif x < 160: r = -0.3
        else: r = -1.5
        r -= 0.15 * self.dose + 0.1 * changed
        self.t += 1
        info = dict(hypo=float(x < 105), unsafe_up=unsafe, in_target=float(110 <= x <= 130),
                    changes=changed, dose=float(self.dose))
        return self._s(), r, self.t >= self.max_steps, info, 1

    def rule(self, s):
        sb = s // 12
        if sb >= 4: return 2
        if sb == 0: return 0
        return 1

    def mask(self):
        m = np.ones((self.nS + 1, self.nA), dtype=bool)
        for s in range(self.nS):
            sb, dose = s // 12, s % 4
            if sb <= 1: m[s, 2] = False          # never up-titrate when SBP < 120
            if sb >= 4: m[s, 0] = False          # never down-titrate when SBP >= 140
        return m


# =============================================================================
# 5. Cancer screening (semi-Markov)  (mirrors: Breast Cancer Wisconsin / Cervical Cancer Risk)
class Screening(Domain):
    name = "screening"
    nA, gamma, max_steps = 3, 0.90, 12
    info_keys = ["screens", "early", "late", "lastdur"]
    AGE_E = [50, 55, 60, 65, 70]
    nS = 3 * 6 * 2 * 3
    action_labels = ["1y", "2y", "3y"]
    behavior_eps = 0.25

    def __init__(self, gamma=0.90):
        self.gamma = gamma

    def _s(self):
        return ((self.tier * 6 + band(self.age, self.AGE_E)) * 2 + self.lastpos) * 3 + self.lastd

    def reset(self, rng):
        self.tier = int(rng.choice(3, p=[0.55, 0.3, 0.15]))
        self.age = float(rng.uniform(45, 65))
        p0 = [0.95, 0.88, 0.8][self.tier]
        u = rng.random()
        self.stage = 0 if u < p0 else (1 if rng.random() < 0.8 else 2)
        self.lastpos, self.lastd = 0, 0
        self.n = 0
        self.yr = 0
        self.U = rng.random((4, 120))      # pre-drawn shocks: common random numbers across policies
        return self._s()

    def step(self, a, rng):
        d = a + 1
        r = -2.0
        screens, early, late = 1.0, 0.0, 0.0
        sens = {0: 0.0, 1: 0.6, 2: 0.85, 3: 0.95}[self.stage]
        pos = self.U[3, self.n] < (sens if self.stage > 0 else 0.03)
        done = False
        self.lastpos = 0
        if pos:
            if self.stage == 0:
                r -= 3.0                           # false positive work-up
            elif self.stage == 1:
                r += 8.0; self.stage = 0; self.lastpos = 1
            elif self.stage == 2:
                r += 100.0; early = 1.0; done = True
            else:
                r += 10.0; late = 1.0; done = True
        self.lastd = a
        if not done:
            for _ in range(d):
                tm = (1.0, 2.5, 6.0)[self.tier] * (1 + (self.age - 50) / 20.0)
                u = self.U[0, self.yr]
                if self.stage == 0 and u < 0.03 * tm: self.stage = 1
                elif self.stage == 1 and u < 0.15 * (1.0, 1.5, 2.5)[self.tier]: self.stage = 2
                elif self.stage == 2 and u < 0.40: self.stage = 3
                if self.stage == 3 and self.U[1, self.yr] < 0.6:
                    r -= 80.0; late = 1.0; done = True; break
                self.age += 1
                self.yr += 1
                if self.U[2, self.yr - 1] < 0.01 + 0.002 * max(self.age - 50, 0):
                    done = True; break
            if self.age >= 75:
                done = True
        self.n += 1
        info = dict(screens=screens, early=early, late=late, lastdur=float(d))
        return self._s(), r, done or self.n >= self.max_steps, info, d

    def rule(self, s):
        tier = s // 36
        return {0: 2, 1: 1, 2: 0}[tier]


# =============================================================================
# 6. CKD monitoring/treatment  (mirrors: Chronic Kidney Disease dataset)
class CKD(Domain):
    name = "ckd"
    nA, gamma, max_steps = 3, 0.98, 60
    info_keys = ["c1", "c2", "failure"]
    chan = ["c1", "c2"]
    GE = list(range(15, 88, 4))
    RED = [0.0, 0.25, 0.5]
    BUR = [0.0, 0.15, 0.5]
    nS = (len(GE) + 1) * 2 * 3
    action_labels = ["routine", "enhanced", "intensive"]

    def _s(self):
        return (band(self.egfr, self.GE) * 2 + self.prot) * 3 + self.last

    def reset(self, rng):
        self.egfr = float(rng.uniform(25, 85))
        self.prot = int(rng.random() < 0.4)
        self.last, self.t = 0, 0
        return self._s()

    def step(self, a, rng):
        base = 0.45 if self.prot == 0 else 1.1
        acc = 1.4 if self.egfr < 30 else 1.0
        self.egfr = float(self.egfr - base * (1 - self.RED[a]) * acc + rng.normal(0, 0.6))
        if self.prot == 1:
            self.prot = int(not (rng.random() < 0.04 + 0.1 * a))
        else:
            self.prot = int(rng.random() < 0.05 - 0.015 * a)
        self.last, self.t = a, self.t + 1
        c1 = self.egfr / 60.0
        c2 = self.BUR[a]
        fail = self.egfr < 15
        if fail: c1 -= 20.0
        died = (not fail) and rng.random() < (0.015 + 0.01 * (self.egfr < 30))   # competing mortality
        info = dict(c1=c1, c2=c2, failure=float(fail))
        return self._s(), c1 - c2, fail or died or self.t >= self.max_steps, info, 1

    def rule(self, s):
        gb = s // 6
        mid = 0.0 if gb == 0 else self.GE[gb - 1] + 2.0
        if mid < 45: return 2
        if mid < 60: return 1
        return 0


# =============================================================================
# 7. Clinic overbooking  (mirrors: Medical Appointment No Shows)
class Clinic(Domain):
    name = "clinic"
    nA, gamma, max_steps = 5, 0.97, 40
    info_keys = ["shows", "overtime", "idle", "wait"]
    CAP = 10
    NE = [0.12, 0.18, 0.24]
    nS = 5 * 4 * 5
    action_labels = [f"+{k}" for k in range(5)]
    MU = np.array([0.22, 0.17, 0.17, 0.18, 0.20])

    def __init__(self, shift=0.0):
        self.shift = shift

    def _s(self):
        return (self.t % 5 * 4 + band(self.p, self.NE)) * 5 + band(self.w, [1, 3, 6, 9])

    def reset(self, rng):
        self.p = float(np.clip(0.18 + self.shift + rng.normal(0, 0.03), 0.02, 0.6))
        self.w, self.t = int(rng.integers(0, 6)), 0
        return self._s()

    def step(self, k, rng):
        booked = self.CAP + k
        shows = int(rng.binomial(booked, 1 - self.p))
        over = max(0, shows - self.CAP)
        idle = max(0, self.CAP - shows)
        drawn = min(self.w, k)
        r = 1.0 * shows - 3.0 * over - 0.5 * idle - 0.4 * self.w + 0.3 * drawn
        self.w = int(np.clip(self.w - drawn + rng.poisson(1.5), 0, 12))
        mu = self.MU[self.t % 5] + self.shift
        self.p = float(np.clip(0.8 * self.p + 0.2 * mu + rng.normal(0, 0.02), 0.02, 0.6))
        self.t += 1
        info = dict(shows=shows, overtime=over, idle=idle, wait=self.w)
        return self._s(), r, self.t >= self.max_steps, info, 1

    def rule(self, s):
        pb = (s // 5) % 4          # conservative practice: overbook by one, or two at high no-show rates
        return 1 if pb <= 1 else 2


# =============================================================================
# 8. Discharge planning  (mirrors: Diabetes 130-US Hospitals)
class Discharge(Domain):
    name = "discharge"
    nA, gamma, max_steps = 3, 0.99, 10
    info_keys = ["readmit", "los", "followup"]
    BASE = np.array([0.06, 0.10, 0.18, 0.30])
    STAB = np.array([1.9, 1.0, 0.55])
    nS = 4 * 9 * 3
    action_labels = ["extend", "discharge", "discharge+follow-up"]
    behavior_eps = 0.25

    def __init__(self, readmit_scale=1.0):
        self.rs = readmit_scale

    def _s(self):
        return (self.risk * 9 + min(self.los, 8)) * 3 + self.stab

    def reset(self, rng):
        self.risk = int(rng.choice(4, p=[0.35, 0.3, 0.2, 0.15]))
        self.stab = int(rng.choice(3, p=[0.45, 0.4, 0.15]))
        self.los = 0
        return self._s()

    def step(self, a, rng):
        self.los += 1
        if self.los >= 9 and a == 0:
            a = 1
        if a == 0:
            up = 0.4 - 0.05 * self.risk
            down = 0.05 + 0.03 * self.risk
            u = rng.random()
            if self.stab < 2 and u < up: self.stab += 1
            elif self.stab > 0 and u > 1 - down: self.stab -= 1
            return self._s(), -1.0, False, dict(los=1.0), 1
        p = self.BASE[self.risk] * self.STAB[self.stab] * (0.6 if a == 2 else 1.0) * self.rs
        re = float(rng.random() < min(p, 0.95))
        r = -25.0 * re - (2.0 if a == 2 else 0.0)
        return self._s(), r, True, dict(readmit=re, los=1.0, followup=float(a == 2)), 1

    def rule(self, s):
        risk, los, stab = s // 27, (s // 3) % 9, s % 3
        if stab == 2 and los >= 2: return 1 if risk <= 1 else 2
        if los >= 6: return 2
        return 0


# =============================================================================
# 9. Stroke rehabilitation  (mirrors: Stroke Prediction + telemonitoring/rehab data)
class Rehab(Domain):
    name = "rehab"
    nA, gamma, max_steps = 4, 0.995, 12
    info_keys = ["gain", "injury"]
    SE = [15, 30, 42, 52, 60]
    FE = [0.35, 0.7]
    nS = 6 * 3 * 12
    action_labels = ["rest", "light", "moderate", "high"]
    GAIN = [0.5, 1.5, 3.0, 4.5]
    FAT = [-0.35, -0.10, 0.15, 0.40]

    def __init__(self, sparse=False, H=12):
        self.sparse = sparse
        self.max_steps = H
        self.nS = 6 * 3 * H

    def _s(self):
        return (band(self.score, self.SE) * 3 + band(self.fat, self.FE)) * self.max_steps + self.t

    def reset(self, rng):
        self.score = float(rng.uniform(12, 40))
        self.s0 = self.score
        self.fat, self.t = float(rng.uniform(0, 0.4)), 0
        return self._s()

    def step(self, a, rng):
        g = self.GAIN[a] * (1 - self.score / 66.0) * (1.1 - self.fat) + rng.normal(0, 0.4)
        fat = float(np.clip(self.fat + self.FAT[a] + rng.normal(0, 0.07), 0, 1))
        inj = float(fat > 0.85 and a >= 2)
        if inj: g -= 2.5
        old = self.score
        self.score = float(np.clip(self.score + g, 0, 66))
        self.fat, self.t = fat, self.t + 1
        done = self.t >= self.max_steps
        gain = self.score - old
        r = gain if not self.sparse else ((self.score - self.s0) if done else 0.0)
        return self._s() if not done else 0, r, done, dict(gain=gain, injury=inj), 1

    def rule(self, s):
        fb = (s // self.max_steps) % 3
        return 1 if fb == 2 else (2 if fb == 1 else 3)


# =============================================================================
# 10. Lifestyle nudging  (mirrors: Sleep Health & Lifestyle / Fitbit wearable data)
class Nudge(Domain):
    name = "nudge"
    nA, gamma, max_steps = 4, 0.97, 28
    info_keys = ["health", "nudges"]
    action_labels = ["none", "sleep", "activity", "stress"]
    behavior_eps = 0.3

    def __init__(self, memory=True):
        self.memory = memory
        self.nS = 27 * (4 if memory else 1)

    def _s(self):
        b = (band(self.sl, [0.4, 0.65]) * 3 + band(self.ac, [0.4, 0.65])) * 3 + band(self.st, [0.4, 0.65])
        if not self.memory:
            return b
        return b * 4 + band(self.h, [0.5, 1.2, 2.0])

    def reset(self, rng):
        self.sl, self.ac, self.st = (float(np.clip(rng.normal(0.5, 0.15), 0, 1)),
                                     float(np.clip(rng.normal(0.4, 0.15), 0, 1)),
                                     float(np.clip(rng.normal(0.55, 0.15), 0, 1)))
        self.h, self.t = 0.0, 0
        return self._s()

    def step(self, a, rng):
        m = float(np.exp(-0.9 * self.h))
        n = lambda: rng.normal(0, 0.08)
        sl = self.sl + 0.25 * (0.55 - self.sl) + 0.5 * m * (a == 1) * (1 - self.sl) + 0.05 * (self.ac - 0.4) + n()
        ac = self.ac + 0.25 * (0.40 - self.ac) + 0.55 * m * (a == 2) * (1 - self.ac) + n()
        st = self.st + 0.20 * (0.50 - self.st) - 0.5 * m * (a == 3) * self.st + 0.15 * (0.5 - self.sl) + n()
        self.sl, self.ac, self.st = (float(np.clip(x, 0, 1)) for x in (sl, ac, st))
        self.h = 0.7 * self.h + float(a > 0)
        self.t += 1
        health = (self.sl + self.ac + (1 - self.st)) / 3.0
        r = health - 0.06 * (a > 0)
        return self._s(), r, self.t >= self.max_steps, dict(health=health, nudges=float(a > 0)), 1

    def rule(self, s):
        b = s // 4 if self.memory else s
        sl, ac, st = b // 9, (b // 3) % 3, b % 3
        # nudge the worst domain if it is in the lowest band
        worst = min((sl, 1), (ac, 2), (2 - st, 3))
        return worst[1] if worst[0] == 0 else 0


FACTORIES = dict(diabetes=Diabetes, sepsis=Sepsis, hospital=Hospital,
                 hypertension=Hypertension, screening=Screening, ckd=CKD,
                 clinic=Clinic, discharge=Discharge, rehab=Rehab, nudge=Nudge)
