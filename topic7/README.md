# Topic 7: appointment overbooking and no-show management

Everything for Topic 7 lives in this folder: sir's Paper 7 material (copied unchanged from `Projects/`) and our extension of it.

> 7. Appointment Overbooking and No-Show Management
> - Dataset: Medical Appointment No Shows
> - MDP: state is slots filled plus predicted no-show risk band; action is overbook level; reward is utilization minus patient waiting and overtime.
> - Futuristic angle: Autonomous clinic scheduling that adapts to patient behavior.
> - Evaluation: Utilization, wait time, overtime, convergence of Monte Carlo versus Q-learning.

## Layout

```
topic7/
  paper/
    Paper07_Clinic_Overbooking_Exploration.docx   sir's Paper 7 (our results are added to it; his text stays as written)
    Problem_statements.docx                       sir's problem statements (Topic 7 is the seventh)
  code/
    sir's files (unchanged apart from two path lines, see below)
      rl_core.py domains.py common.py ideas.py    shared library, simulator, experiments (idea 7)
      paper07.py paper_common.py docx_helper.py   builds the Paper 7 docx from out/idea07/results.json
      paper07_run.py defend_visuals.py            full rerun; defence dashboard (run defend_visuals.py from inside code/)
      kaggle_adapter.py README.md                 sir's adapter and code README
      out/idea07/                                 sir's Paper 7 results.json and figures
      explanations/                               sir's Paper 7 results text and paired tests
    our files (new)
      noshow_data.py                              real Kaggle data: cleaning, no-show classifier, clinic sessions
      real_clinic.py                              the clinic rebuilt from real sessions (comparison B)
      planner.py                                  our method + baselines from published work
      run_topic7.py                               runs every experiment, writes out/topic7/
      out/topic7/                                 our results.json and figures
  data/
    KaggleV2-May-2016.csv                         real no-show dataset (not in git, see data/README.md)
    README.md                                     source, columns, hash
```

## Our method

Sir's learners (DP, Monte Carlo, Q-learning) estimate every transition of every (state, action) pair from 150 logged clinic days. Most pairs are seen fewer than 10 times, so the support rule sends them back to the overbooking rule, and the learned policies stay close to it.

Our planner splits the problem in two:

- **Learned from data:** how patients behave. On sir's simulator: show-up chance per no-show band, how the no-show band moves between sessions, waiting-list arrivals per session. On the real data: each patient's no-show probability from a classifier trained on earlier weeks (gradient-boosted trees, isotonic calibration, test AUC 0.73).
- **Known:** the clinic's own rules. 10 regular slots, overtime and idle costs, the waiting list is drained by extra bookings, the reward formula.

The two are combined into an MDP that sir's `value_iteration` solves. Every logged session informs the same few patient numbers, so they are estimated precisely from little data, and every action's value can be computed even if the logs never tried it.

## Results (`python topic7/code/run_topic7.py`, 20 seeds)

**A. Sir's own simulator** (his data seeds and scoring; the first 5 seeds reproduce his published numbers exactly)

| Method | Return | Utilization | Overtime / session | Waiting list / session |
|---|---|---|---|---|
| Sir: rule | 160.27 | 91.5% | 0.25 | 4.87 |
| Sir: DP | 164.53 | 91.8% | 0.30 | 4.19 |
| Sir: Monte Carlo | 165.46 | 91.8% | 0.32 | 3.99 |
| Sir: Q-learning | 166.02 | 91.8% | 0.31 | 3.99 |
| Sir: oracle DP (10x data) | 179.01 | 93.2% | 0.54 | 1.81 |
| **Ours** | **184.10** | 94.0% | 0.52 | 1.65 |

+19.6 over sir's DP (p < 0.001). Ours reaches 184.0 from 25 logged days; sir's DP needs 1,500 days to reach 177.1. When the no-show rate shifts by +0.08 or -0.05, ours stays at 179 to 183 while sir's methods fall to 142 to 145 in the harder case.

**B. Real Kaggle sessions** (learned from 16 to 25 May 2016, scored on 30 May to 8 June)

| Method | Return | Utilization | Overtime / session | Waiting list / session |
|---|---|---|---|---|
| Sir: rule | 150.82 | 88.9% | 0.20 | 5.28 |
| Threshold double-booking (best threshold) | 164.53 | 95.5% | 1.02 | 1.76 |
| Fixed overbooking (best level, +2) | 174.61 | 91.4% | 0.37 | 2.45 |
| One-session optimiser | 182.03 | 93.8% | 0.51 | 1.83 |
| Sir: DP | 170.28 | 90.4% | 0.39 | 2.46 |
| **Ours** | **182.54** | 93.8% | 0.53 | 1.67 |

+12.3 over sir's DP, +31.7 over his rule, +0.5 over the one-session optimiser at normal demand (p < 0.001). The gap over the one-session optimiser grows as the clinic gets busier: +1.7%, +5.3% and +33.3%.

**Monte Carlo vs Q-learning:** on our model, Q-learning gets much closer to the DP answer than Monte Carlo (176 vs 166 in A, 175 vs 169 in B).

## Caveats

- Our planner takes the clinic's own rules as known and learns only patient behaviour.
- In the real data about 9% of sessions have fewer than 10 regular bookings, so the slots-filled band rarely changes.
- The real no-show rate drifted from 21.2% (learning weeks) to 18.5% (test weeks); the classifier predicts 22.0% on average for the test weeks, so the planner is slightly cautious there.
- At normal demand the margin over the one-session optimiser is small (+0.3%); the large gains are over Paper 7's methods and in busier clinics.
- The baselines from published work are our simplified re-implementations.
- Comparison B is still a simulation, built from real records with Paper 7's reward weights.

## Run (from the repo root, with the virtual environment active)

All of our experiments (about 10 to 30 minutes; writes `topic7/code/out/topic7/`):

```
python topic7/code/run_topic7.py
```

A two-minute smoke test (writes `topic7/code/out/topic7_quick/`, not committed):

```
python topic7/code/run_topic7.py --quick
```

Sir's Paper 7 experiment, figures and docx (about 2 minutes):

```
python topic7/code/paper07_run.py
```

## What differs from sir's copy

When copied, `code/`, `paper/` and the files in them were byte-identical to `Projects/`. The only change to sir's files is two path lines: `common.py` and `paper_common.py` used absolute paths from the machine the package was built on (`/home/claude/project/...`); they now point to `code/out/` and `paper/`. Rebuilding Paper 7 from the stored results gives text identical to sir's docx. Everything else of ours is in new files.

```
git diff --no-index Projects/code_package/code_package/common.py topic7/code/common.py
git diff --no-index Projects/code_package/code_package/paper_common.py topic7/code/paper_common.py
```
