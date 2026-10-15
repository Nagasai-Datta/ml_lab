# Ten RL-in-medicine papers: code package
Requires Python 3.9+, numpy, scipy, pandas, matplotlib, python-docx.
- rl_core.py: empirical MDP estimation, value/policy iteration, robust DP, first-visit Monte Carlo control, Q-learning, evaluation.
- domains.py: the ten simulators (mirroring the variables of the named Kaggle datasets). common.py: benchmark + figure helpers.
- ideas.py: the ten experiments (`python3 ideas.py k`, k=1..10). Results land in out/ideaNN/ (results.json + figures).
- defend_visuals.py: defence dashboard and paired-test printout (`python3 defend_visuals.py k`).
- paperKK.py: builds each Word paper from results.json. paperKK_run.py runs experiment -> defence visuals -> paper.
- kaggle_adapter.py: converts a real Kaggle CSV to the same transition format (see its docstring).
- explanations/: results-section text of each paper and the paired-test printouts.
Honesty note: Kaggle was unreachable when this was built (HTTP 403), so every result comes from simulators. They are method studies, not clinical evidence.
