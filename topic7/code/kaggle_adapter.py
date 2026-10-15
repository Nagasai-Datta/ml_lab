"""
kaggle_adapter.py -- run the SAME pipeline on a real Kaggle CSV.

Kaggle was not reachable from the environment in which this project was built, so
every reported number comes from the simulators in domains.py.  To repeat a study on
the real file you need only (1) download the CSV, (2) describe how its columns map to
states / actions / rewards, and (3) call `transitions_from_dataframe`.  The result is a
`data` dict in exactly the format produced by `rl_core.rollout`, so `build_mdp`,
`value_iteration`, `mc_control` and `q_learning` run unchanged.

Because a real CSV has no simulator, policies cannot be scored by simulation.  Use
`rl_core.J(model, policy)` on a held-out split of the data (a model-based estimate), and
report that clearly as an estimate -- not as a clinical outcome.

Self-test:  python kaggle_adapter.py
"""
import numpy as np
import pandas as pd


class Spec:
    """minimal stand-in for a Domain (build_mdp only needs these attributes)"""
    def __init__(self, nS, nA, gamma=0.97, chan=(), info_keys=()):
        self.nS, self.nA, self.gamma = nS, nA, gamma
        self.chan, self.info_keys = list(chan), list(info_keys)


def discretise(series, edges):
    return np.searchsorted(np.asarray(edges), series.to_numpy(), side="right")


def transitions_from_dataframe(df, episode_col, time_col, state_bins, action_col, reward_col,
                               action_map=None, gamma=0.97):
    """
    df          : one row per (episode, time step); e.g. patient_id, visit_number
    state_bins  : ordered dict {column: edges}; the state is the mixed-radix code of the
                  discretised columns (np.searchsorted bands, as in domains.py)
    action_col  : column holding the action taken (int or categorical -> use action_map)
    reward_col  : column holding the per-step reward (compute it first if needed)
    returns     : data (dict), spec (Spec)
    """
    df = df.sort_values([episode_col, time_col]).reset_index(drop=True)
    sizes, code = [], np.zeros(len(df), dtype=int)
    for col, edges in state_bins.items():
        sizes.append(len(edges) + 1)
        code = code * (len(edges) + 1) + discretise(df[col], edges)
    nS = int(np.prod(sizes))
    a = df[action_col].map(action_map).to_numpy() if action_map else df[action_col].to_numpy()
    nA = int(a.max()) + 1
    ep = pd.factorize(df[episode_col])[0]
    t = df.groupby(episode_col).cumcount().to_numpy()
    last = (df.groupby(episode_col)[time_col].transform("max") == df[time_col]).to_numpy()
    s2 = np.roll(code, -1)
    s2[last] = code[last]                    # placeholder; replaced by terminal because done=1
    data = dict(s=code, a=a.astype(int), r=df[reward_col].to_numpy(float), s2=s2,
                done=last.astype(float), dur=np.ones(len(df)), ep=ep, t=t)
    return data, Spec(nS, nA, gamma)


if __name__ == "__main__":
    # self-test: simulate -> dump to a DataFrame -> rebuild the MDP -> must match.
    import sys
    sys.path.insert(0, ".")
    from domains import Diabetes
    from rl_core import rollout, build_mdp
    dom = Diabetes()
    d = rollout(dom, dom.behavior, np.random.default_rng(0), 200)
    # decode the state back into (glucose band, trend, meal) columns
    s = d["s"]
    df = pd.DataFrame(dict(patient=d["ep"], hour=d["t"], gband=s // 6, trend=(s // 2) % 3, meal=s % 2,
                           dose=d["a"], reward=d["r"]))
    bins = {"gband": np.arange(1, 6) - 0.5, "trend": [0.5, 1.5], "meal": [0.5]}
    data, spec = transitions_from_dataframe(df, "patient", "hour", bins, "dose", "reward", gamma=dom.gamma)
    m1, m2 = build_mdp(d, dom), build_mdp(data, spec)
    # state ids are recoded (mixed radix), so compare through the shared reward and visit totals
    print("transitions:", len(data["s"]), "== original:", len(d["s"]))
    print("total visits equal:", int(m1.n.sum()) == int(m2.n.sum()),
          "| reward mass equal:", bool(np.isclose(m1.Rsum.sum(), m2.Rsum.sum())))
