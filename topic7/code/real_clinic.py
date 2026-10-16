"""real_clinic.py -- Paper 7's clinic, rebuilt from real Kaggle sessions (comparison B).

Same decision and reward as the Clinic simulator (domains.Clinic), but the patients are real:
  * each session is a real neighbourhood-day; its first min(10, n) appointments fill the regular slots
  * patients pulled in from the waiting list by overbooking are real appointments from the same weekday
  * whether each patient shows up is what actually happened in the data
State, as the problem statement asks: slots filled x predicted no-show risk band x waiting-list band
  slots filled band : 5-7 | 8-9 | 10 regular slots booked                (3)
  risk band         : quartiles of the session's mean predicted no-show  (4)
  waiting band      : 0 | 1-2 | 3-5 | 6-8 | 9+ patients waiting         (5)   -> 60 states
Action k = 0..4 extra bookings. Reward (as in domains.Clinic):
  shows - 3 x overtime - 0.5 x idle slots - c_wait x waiting list + 0.3 x patients drawn from the list
Built on rl_core.Domain, so rollout / build_mdp / eval_policy / q_learning / mc_control all work unchanged.
"""
import numpy as np
from rl_core import Domain, band

SLOT_E = [8, 10]            # slots filled 5-7 -> 0, 8-9 -> 1, 10 -> 2
WAIT_E = [1, 3, 6, 9]       # same waiting bands as domains.Clinic


class RealClinic(Domain):
    name = "real clinic"
    nA, gamma, max_steps, CAP = 5, 0.97, 40, 10
    nS = 3 * 4 * 5
    info_keys = ["shows", "overtime", "idle", "wait", "booked", "drawn", "p_reg_mean"]
    action_labels = [f"+{k}" for k in range(5)]

    def __init__(self, sessions, pool, risk_edges, lam=1.5, wait_cost=0.4):
        self.sessions, self.pool, self.RE = sessions, pool, np.asarray(risk_edges)
        self.lam, self.wait_cost = lam, wait_cost

    # ----------------------------------------------------------------- state
    def _draw(self, rng):
        """pick today's real session and the real patients who would come off the waiting list"""
        d = self.t % 5
        p, y = self.sessions[d][rng.integers(len(self.sessions[d]))]
        o = rng.permutation(len(p))[:self.CAP]
        self.p_reg, self.y_reg = p[o], y[o]                     # regular slots (up to 10)
        pp, yy = self.pool[d]
        e = rng.integers(len(pp), size=self.nA - 1)
        self.p_ext, self.y_ext = pp[e], yy[e]                   # up to 4 extra patients
        self.n_reg = len(o)

    def _s(self):
        sb = band(self.n_reg, SLOT_E)
        rb = int(np.searchsorted(self.RE, self.p_reg.mean(), side="right"))
        return (sb * 4 + rb) * 5 + band(self.w, WAIT_E)

    @staticmethod
    def decode(s):
        sb, rest = divmod(s, 20)
        rb, wb = divmod(rest, 5)
        return sb, rb, wb

    # ----------------------------------------------------------------- dynamics
    def reset(self, rng):
        self.t, self.w = 0, int(rng.integers(0, 6))
        self._draw(rng)
        return self._s()

    def step(self, k, rng):
        shows = int((1 - self.y_reg).sum() + (1 - self.y_ext[:k]).sum())
        over, idle = max(0, shows - self.CAP), max(0, self.CAP - shows)
        drawn = min(self.w, k)
        r = 1.0 * shows - 3.0 * over - 0.5 * idle - self.wait_cost * self.w + 0.3 * drawn
        self.w = int(np.clip(self.w - drawn + rng.poisson(self.lam), 0, 12))
        self.t += 1
        info = dict(shows=shows, overtime=over, idle=idle, wait=self.w, booked=self.n_reg + k, drawn=drawn,
                    p_reg_mean=float(self.p_reg.mean()))
        self._draw(rng)
        return self._s(), r, self.t >= self.max_steps, info, 1

    def rule(self, s):
        """the overbooking rule of domains.Clinic, read on this state: +1 when predicted no-shows are low, otherwise +2"""
        return 1 if self.decode(s)[1] <= 1 else 2


def risk_edges(sessions, rng=None, draws=20):
    """quartiles of a session's mean predicted no-show (the risk-band cut points)"""
    rng = rng or np.random.default_rng(0)
    m = [p[rng.permutation(len(p))[:10]].mean() for d in sessions for p, _ in sessions[d] for _ in range(draws)]
    return np.quantile(m, [0.25, 0.5, 0.75])
