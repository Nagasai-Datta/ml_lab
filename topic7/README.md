# Topic 7: appointment overbooking and no-show management

Everything for Topic 7 lives in this folder: sir's Paper 7 material (copied from `Projects/`) and our work.

```
topic7/
  paper/
    Paper07_Clinic_Overbooking_Exploration.docx   sir's Paper 7 (our edited version will be rebuilt here)
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
      out/topic7/                                 our results.json and figures (created by run_topic7.py)
  data/
    KaggleV2-May-2016.csv                         real no-show dataset (in the zip, not in git)
    README.md                                     source, columns, hash, why not in git
  prototypes/                                     early experiments that picked the topic and method (removed in step 2)
  notes/PROGRESS.md                               decisions, prior work, results, plan
```

## Our method in one paragraph

Sir's learners estimate every transition of every (state, action) pair from 150 logged clinic days, so most pairs are seen too rarely to trust and the policy falls back to the overbooking rule. Our planner learns only how patients behave (show-up chance, how the no-show level moves, waiting-list arrivals) and takes the clinic's own rules as known (10 slots, overtime and idle costs, the waiting list is drained by extra bookings). The two are combined into an MDP that sir's `value_iteration` solves. On the real data the show-up chances come from a no-show classifier trained on earlier weeks.

## Run (from the repo root, with the virtual environment active)

All of our experiments (about 15 to 30 minutes; writes `topic7/code/out/topic7/`):

```
python topic7/code/run_topic7.py
```

A two-minute smoke test with a few seeds (writes `topic7/code/out/topic7_quick/`, not committed):

```
python topic7/code/run_topic7.py --quick
```

Sir's Paper 7 experiment, figures and docx (about 2 minutes; writes `topic7/code/out/idea07/` and `topic7/paper/`):

```
python topic7/code/paper07_run.py
```

Rebuild only the Paper 7 docx from the stored results:

```
python topic7/code/paper07.py
```

## What differs from sir's copy

When copied, `code/`, `paper/` and the files in them were byte-identical to `Projects/`. Since then, one change to sir's files: `common.py` and `paper_common.py` used absolute paths from the machine the package was built on (`/home/claude/project/...`). They now point to `code/out/` and `paper/`. Rebuilding Paper 7 from the stored results gives text identical to sir's docx. Everything else of ours is in new files.

To see exactly how our copy of any of sir's files differs from his original:

```
git diff --no-index Projects/code_package/code_package/common.py topic7/code/common.py
git diff --no-index Projects/code_package/code_package/paper_common.py topic7/code/paper_common.py
```
