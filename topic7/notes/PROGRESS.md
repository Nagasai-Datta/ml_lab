# Topic 7 progress notes

Last updated: 16 Oct 2026 (Step 1: final code and 20-seed results).

## 1. The task

Sir gave ten problem statements, each with a finished paper and code. We pick one, propose an algorithm that beats the paper's methods, show why it wins with the same kind of metrics the paper reports, and add our results to the paper. Later, a presentation: what the paper was about, what was already there, what we did, how much it improved and why.

## 1b. Topic 7 problem statement (sir's criteria, from Problem_statements.docx)

> 7. Appointment Overbooking and No-Show Management
> - Dataset: Medical Appointment No Shows
> - MDP: state is slots filled plus predicted no-show risk band; action is overbook level; reward is utilization minus patient waiting and overtime.
> - Futuristic angle: Autonomous clinic scheduling that adapts to patient behavior.
> - Evaluation: Utilization, wait time, overtime, convergence of Monte Carlo versus Q-learning.

Checklist: what sir's Paper 7 covers, and what our version must cover.

| Item | Sir's Paper 7 | Our version | Status |
|---|---|---|---|
| Dataset | simulator that copies the columns; no real data | real CSV: no-show model + real clinic sessions | done (Step 1) |
| State: slots filled | not in his state (weekday, no-show band, waiting band) | slots-filled band (5-7, 8-9, 10 booked) in comparison B | done (Step 1) |
| State: predicted no-show risk band | his band is the simulator's true rate, not a prediction | band of the classifier's predicted risk for the session's patients | done (Step 1) |
| Action: overbook level | 0 to 4 extra bookings | same | done |
| Reward: utilization minus waiting and overtime | shows − 3·overtime − 0.5·idle − 0.4·waiting list + 0.3·drawn | same reward, so results compare directly | done |
| Futuristic angle: adapts to patient behaviour | partly (no-show shift test) | adapts per session to predicted risk and to the waiting list; shift test (A) and real drift between log and test weeks (B) | done (Step 1) |
| Evaluation: utilization | reports shows and idle slots, not utilization as such | utilization % of the 10 regular slots | done (Step 1) |
| Evaluation: wait time | waiting-list length | waiting list per session; real booking lead time reported in the data summary | done (Step 1) |
| Evaluation: overtime | reported | overtime patients per session | done (Step 1) |
| Evaluation: convergence of Monte Carlo vs Q-learning | learning curves + exploration-schedule study | MC and Q-learning on sir's model and on ours, true-return learning curves against our DP | done (Step 1) |

## 2. Decision: Topic 7 (appointment overbooking and no-show management)

- Largest gap left on the table in the paper itself: rule 160.3, paper's best learner 166.6, oracle 179.0. The paper's learners capture only 10 to 34% of the possible improvement.
- Small, fast problem (100 states, 5 actions); experiments run in seconds to minutes.
- The only topic whose named dataset is real, large and dated, so we can test on real clinic days. Most other datasets have one row per patient and no timeline (heart disease 303 rows, CKD 400, breast cancer 569, stroke 5,110, sleep 374), and the sepsis data has no treatment columns.

## 3. What the faculty package is

- All ten papers use hand-written simulators that copy the Kaggle columns. No real data was used (the README says Kaggle returned HTTP 403 when it was built).
- The code reproduces Paper 7 exactly: DP 162.19, Q-learning 164.36, Monte Carlo 166.57 (exp02 log).
- `paper07.py` builds the Paper 7 docx from `out/idea07/results.json`. After the path fix, rebuilding gives text identical to the handed-out docx, so we can add our sections through the same code.
- `kaggle_adapter.py` cannot be used on the no-show CSV: the CSV records appointments and show/no-show only, not any overbooking decision.

## 4. What the real data says (KaggleV2-May-2016.csv)

- 110,527 appointments, 62,299 patients, 81 neighbourhoods, 27 days, 20.2% no-shows.
- Weekday barely matters: 19.4% (Thu) to 21.2% (Fri). Paper 7's simulator uses 17% to 22%.
- A neighbourhood's no-show rate today says almost nothing about tomorrow (day-to-day correlation 0.04). Paper 7's simulator assumes strong carry-over (0.8), so its "no-show band" state is unrealistic.
- What predicts no-shows: booking lead time (4.7% same day, 33% for 30+ days ahead), past no-shows, age, SMS. Our classifier: test AUC 0.73 (trained on days before 9 May, calibrated on 9 to 15 May, tested from 16 May).

## 5. Prior work

On this dataset:

| Work | What it does | Relation to us |
|---|---|---|
| Kaggle notebooks, most GitHub repos | exploration and no-show prediction | prediction only |
| Yogarajan12/clinic-overbooking-optimization (course project, 2025) | stacked ensemble, isotonic calibration, Poisson-binomial optimiser for one session | our "static one-session" baseline; we don't claim this part |
| Chandancndn/clinical-appointment-management-system (mini project, Oct 2026) | prediction (AUC 0.727), SimPy session simulation, threshold double-booking, REINFORCE choosing slots within one session; attendance drawn from the model's own risk | RL on this dataset exists, but only within one session |
| cnrasili/noshow-smart-scheduling | prediction, rule-based selective overbooking, SimPy | no RL, no results |
| Isha290105/No-Show-Prediction | prediction, fixed vs model-based overbooking simulation | dataset not named |

Other data: Amalina & An 2026 (PPO, private Middle East data), Topuz et al. 2024 (ML plus marginal analysis), Valenzuela-Núñez et al. 2024 (Chile, SVM plus integer programming), Han et al. 2024 (chemotherapy), Leiva-Araos et al. 2025 (Chile), Deina et al. 2024 (two Brazilian hospitals, prediction only; full text not read, Springer rate-limited).

Not done on this dataset as far as we found:
1. Planning across sessions: today's overbooking drains tomorrow's waiting list, solved with DP or Q-learning.
2. Evaluation on real recorded clinic days with real show/no-show outcomes.
3. Comparison with the faculty's Paper 7 learners.

## 6. Results so far (prototypes; logs in `topic7/prototypes/logs/`)

### 6a. Paper 7's own simulator (same 5 data seeds as the paper)

| Method | Return | vs paper |
|---|---|---|
| Rule | 160.3 | |
| Paper DP / Q-learning / MC | 162.2 / 164.4 / 166.6 | baseline |
| DP, weekday dropped from state | 177.4 | +15.2 over paper DP (p = 0.001) |
| Q-learning, weekday dropped | 176.0 | +11.6 over paper Q-learning (p = 0.006) |
| Majority vote of DP over all 7 feature subsets | 174.6 | +12.4 over paper DP (p = 0.008) |
| Oracle DP (10x data) | 179.0 | |

Source: exp01, exp02, exp03, exp04.

### 6b. Clinic built from real sessions (Paper 7 reward, real patients and outcomes)

| Policy | Return |
|---|---|
| Ours: prediction + DP lookahead over the waiting list (PDP) | 182.2 |
| Static one-session optimiser (Yogarajan / Topuz style) | 181.3 |
| Always overbook by 2 | 176.8 |
| Threshold double-booking, best threshold (Amalina / CAMS style) | 165.0 |
| Weekday-dropped vote from 6a, moved to real data * | 161.6 |
| Paper 7 DP / Q-learning * | 159.3 / 159.7 |
| Paper 7 rule | 154.9 |

PDP minus static optimiser: +0.89 ± 0.14 (p < 0.001, 1,500 paired episodes). Sources: exp08 (PDP, static, +2, rule; 1,500 episodes), exp07 (threshold; 300 episodes).

\* From exp06, the first try: uncalibrated classifier and slightly different risk bands, 300 evaluation episodes. The final runs will put every method in one environment with the same evaluation.

### 6c. When planning across sessions matters (exp09)

| Waiting cost | New patients per session | Ours | Static one-session | Gain |
|---|---|---|---|---|
| 0.4 (paper default) | 1.5 (paper default) | 182.2 | 181.3 | +0.5% |
| 1.0 | 1.5 | 159.2 | 153.9 | +3.5% |
| 0.4 | 2.5 | 169.5 | 156.4 | +8.4% |
| 1.0 | 2.5 | 130.1 | 86.7 | +50% |
| 2.0 | 2.5 | 65.2 | -29.5 | large |

Story: one-session decisions are fine when the clinic is quiet and break when the waiting list builds up. The CSV has no waiting-list data, so demand levels are assumptions we vary openly.

### 6d. Step 1 final results (`run_topic7.py`, 20 seeds; outputs in `topic7/code/out/topic7/`)

**A. Sir's simulator** (his data seeds 1000-1019, his scoring: 300 episodes). The first 5 seeds reproduce
his published numbers exactly (DP 162.19, Monte Carlo 166.57, Q-learning 164.36, rule 160.27, oracle 179.01).

| Method | Return | Utilization | Overtime / session | Waiting list / session |
|---|---|---|---|---|
| Rule | 160.27 | 91.5% | 0.25 | 4.87 |
| Sir: DP | 164.53 ± 1.35 | 91.8% | 0.30 | 4.19 |
| Sir: Monte Carlo | 165.46 ± 2.23 | 91.8% | 0.32 | 3.99 |
| Sir: Q-learning | 166.02 ± 1.47 | 91.8% | 0.31 | 3.99 |
| Oracle DP (10x data) | 179.01 | 93.2% | 0.54 | 1.81 |
| Ours: Monte Carlo (on our model) | 166.30 ± 1.80 | 92.8% | 0.64 | 2.50 |
| Ours: Q-learning (on our model) | 176.25 ± 0.56 | 92.8% | 0.53 | 2.00 |
| **Ours: DP** | **184.10 ± 0.01** | 94.0% | 0.52 | 1.65 |

Ours vs sir's DP +19.56 (p = 5e-17), vs his Q-learning +18.08, vs his oracle +5.09.
Data size (sir's DP vs ours): 25 days 158.5 vs 184.0; 150 days 163.6 vs 184.1; 1,500 days 177.1 vs 184.1.
No-show shift: +0.08: ours 179.2, sir's DP 165.9; -0.05: ours 182.8, sir's DP 142.3, rule 136.7.

**B. Real Kaggle sessions** (logs 16-25 May, scored on 30 May - 8 Jun, 1,000 episodes; classifier AUC 0.728).

| Method | Return | Utilization | Overtime / session | Waiting list / session |
|---|---|---|---|---|
| Rule | 150.82 | 88.9% | 0.20 | 5.28 |
| Threshold double-booking (p >= 0.25) | 164.53 | 95.5% | 1.02 | 1.76 |
| Fixed overbooking (+2) | 174.61 | 91.4% | 0.37 | 2.45 |
| One-session optimiser | 182.03 | 93.8% | 0.51 | 1.83 |
| Sir: DP | 170.28 ± 0.99 | 90.4% | 0.39 | 2.46 |
| Sir: Monte Carlo | 162.80 ± 2.26 | 89.2% | 0.38 | 2.86 |
| Sir: Q-learning | 167.05 ± 2.11 | 90.1% | 0.42 | 2.53 |
| Oracle DP (10x data) | 173.62 | 91.5% | 0.54 | 1.72 |
| Ours: Monte Carlo | 169.32 ± 1.70 | 92.4% | 0.65 | 2.00 |
| Ours: Q-learning | 175.16 ± 1.05 | 92.8% | 0.59 | 1.81 |
| **Ours: DP** | **182.54 ± 0.01** | 93.8% | 0.53 | 1.67 |

Ours vs one-session optimiser +0.51 (p = 4e-6, paired over 1,000 episodes), vs fixed +2 +7.92, vs sir's DP +12.25,
vs rule +31.72. Busier clinic (ours vs one-session optimiser): +0.2%, +1.7%, +5.3%, +33.3% for
(wait cost, arrivals) = (0.4, 1.5), (1.0, 1.5), (0.4, 2.5), (1.0, 2.5).

## 7. Plan and status

1. Step 1, code + first run: new files `noshow_data.py`, `real_clinic.py`, `planner.py`, `run_topic7.py` (sir's files untouched). Done.
2. Step 2, clean-up: delete `prototypes/`, fold these notes into `topic7/README.md`.
3. Step 3, paper: extend `paper07.py` (keep sir's results, add our method, results for A and B, why it wins, limitations, anticipated questions, references), rebuild the docx. Needs author names and department.
4. Step 4, checks: every number in the paper against `out/topic7/results.json`; full rerun from a fresh clone.
5. Step 5, presentation (.pptx).

## 8. Caveats to keep in the paper

- Our planner assumes the clinic's own rules are known (slots, overtime and idle costs, how the waiting list empties) and learns only patient behaviour; state this plainly.
- In the real data only about 9% of sessions have fewer than 10 regular bookings, so the slots-filled band rarely changes.
- The real no-show rate drifted from 21.2% (log weeks) to 18.5% (test weeks); the classifier predicts 22.0% on average for the test weeks, so our planner is slightly cautious there.
- Margin over the best static baseline is small at the paper's default settings (+0.5%); the large gains are over Paper 7 and under heavier demand.
- Baselines from other papers are our simplified re-implementations, not their code.
- The real-session clinic is still a simulator built from real records, with Paper 7's reward weights.
- The dataset licence was not confirmed; keep the repo private or check before publishing the CSV.

## Sources

- Kaggle dataset: https://www.kaggle.com/datasets/joniarroba/noshowappointments
- Amalina & An 2026: https://arxiv.org/abs/2603.07270
- Topuz et al. 2024: https://link.springer.com/article/10.1007/s10479-023-05799-0
- Valenzuela-Núñez et al. 2024: https://www.researchgate.net/publication/377163945
- Han et al. 2024: https://researchwith.montclair.edu/en/publications/a-multi-appointment-patient-scheduling-system-with-machine-learni/
- Leiva-Araos et al. 2025: https://link.springer.com/article/10.1007/s10916-025-02143-w
- Deina et al. 2024: https://doaj.org/article/eda8c2b9fd644d93be1e047e2e149d42
- Zacharias & Pinedo 2014: https://www.researchgate.net/publication/262232413
- Yogarajan12 repo: https://github.com/Yogarajan12/clinic-overbooking-optimization
- Chandancndn repo: https://github.com/Chandancndn/clinical-appointment-management-system
- cnrasili repo: https://github.com/cnrasili/noshow-smart-scheduling
- Isha290105 repo: https://github.com/Isha290105/No-Show-Prediction
