# Topic 7 progress notes

Last updated: 15 Oct 2026 (problem-statement checklist added).

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
| Dataset | simulator that copies the columns; no real data | real CSV: no-show model + real clinic sessions | done in prototypes |
| State: slots filled | not in his state (weekday, no-show band, waiting band) | add slots filled (booked regular slots per session) | to do |
| State: predicted no-show risk band | his band is the simulator's true rate, not a prediction | band of the classifier's predicted risk for the session's patients | done in prototypes |
| Action: overbook level | 0 to 4 extra bookings | same | done |
| Reward: utilization minus waiting and overtime | shows − 3·overtime − 0.5·idle − 0.4·waiting list + 0.3·drawn | same reward, so results compare directly | done |
| Futuristic angle: adapts to patient behaviour | partly (no-show shift test) | adapts per session to predicted risk and to the waiting list; add shift test | partly |
| Evaluation: utilization | reports shows and idle slots, not utilization as such | report utilization % (patients seen ÷ capacity) | to do |
| Evaluation: wait time | waiting-list length | waiting-list length; consider real lead time (ScheduledDay to AppointmentDay) as a real wait measure | to do |
| Evaluation: overtime | reported | report | to do (prototypes only printed return) |
| Evaluation: convergence of Monte Carlo vs Q-learning | learning curves + exploration-schedule study | run MC and Q-learning on our new state, show their learning curves against our DP | to do |

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

## 7. Plan for the edit work (not started)

1. Add our methods to the faculty code (new functions next to `ideas.py` idea 7), keeping everything else as is.
2. Rerun Paper 7's experiment plus ours with more seeds (20), the no-show shift test, and the demand sweep.
3. Extend `paper07.py` so the paper gains our sections, tables, figures and references, then rebuild the docx.
4. Presentation: paper summary, what was there, what we did, improvement and why.

## 8. Caveats to keep in the paper

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
