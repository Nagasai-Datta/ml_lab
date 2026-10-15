"""Where does the multi-day lookahead matter? Vary the waiting-list cost and arrival rate; compare PDP with the
static one-session optimiser (the Yogarajan / Topuz-style baseline) using paired episodes."""
import sys
import _paths  # adds topic7/code (our copy of sir's code) to sys.path
from real_env import *
import exp07_real_pdp as P

def run(WC, LAM, n=800):
    class C(RealClinic):
        def step(self, k, rng):
            booked = self.CAP + k
            shows = int((1 - self.yy[:booked]).sum())
            over, idle = max(0, shows - self.CAP), max(0, self.CAP - shows)
            drawn = min(self.w, k)
            r = shows - 3.0 * over - 0.5 * idle - WC * self.w + 0.3 * drawn
            self.w = int(np.clip(self.w - drawn + rng.poisson(LAM), 0, 12)); self.t += 1; self._draw(rng)
            return self._s(), r, self.t >= self.max_steps, {}, 1
    pois = stats.poisson.pmf(np.arange(0, 15), LAM); pois[-1] += 1 - pois.sum()
    NW = np.zeros((13, 5, 13))
    for w in range(13):
        for k in range(5):
            for a, pa in enumerate(pois): NW[w, k, min(12, w - min(w, k) + a)] += pa
    rng = np.random.default_rng(7); R = {}
    for d in range(5):
        R[d] = np.array([[P.imm(p[rng.permutation(len(p))[:10]], k) for k in range(5)]
                         for p, _ in (sess[d][rng.integers(len(sess[d]))] for _ in range(300))])
    V = np.zeros((41, 13))
    for t in range(39, -1, -1):
        cont = 0.97 * NW @ V[t + 1]
        for w in range(13):
            V[t, w] = (R[t % 5] + 0.3 * np.minimum(w, np.arange(5)) - WC * w + cont[w]).max(1).mean()
    pdp = lambda dom, s: int(np.argmax([P.imm(dom.pp[:10], k) + 0.3 * min(dom.w, k) + 0.97 * NW[dom.w, k] @ V[min(dom.t + 1, 40)] for k in range(5)]))
    dom = C("pred")
    def ret(act):
        out = np.zeros(n)
        for e in range(n):
            g = np.random.default_rng([777, e]); s = dom.reset(g); disc = 1.0
            for _ in range(40):
                s, r, done, _, _ = dom.step(int(act(dom, s)), g); out[e] += disc * r; disc *= 0.97
                if done: break
        return out
    a, b = ret(pdp), ret(myopic)
    d = a - b
    print(f"waiting cost {WC:.1f}, arrivals/session {LAM:.1f}:  PDP {a.mean():7.2f}  static {b.mean():7.2f}  "
          f"gain {d.mean():+5.2f} ± {ci95(d):.2f} ({100*d.mean()/abs(b.mean()):.1f}%)", flush=True)

for WC, LAM in ((0.4, 1.5), (1.0, 1.5), (0.4, 2.5), (1.0, 2.5), (2.0, 2.5)):
    run(WC, LAM)
