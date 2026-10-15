# Topic 7: appointment overbooking and no-show management

Everything for Topic 7 lives in this folder: sir's Paper 7 material (copied from `Projects/`) and our work.

```
topic7/
  paper/
    Paper07_Clinic_Overbooking_Exploration.docx   sir's Paper 7 (our edited version will be rebuilt here)
    Problem_statements.docx                       sir's problem statements (Topic 7 is the seventh)
  code/                                           sir's code needed for Paper 7
    rl_core.py domains.py common.py ideas.py      shared library, simulator, experiments (idea 7)
    paper07.py paper_common.py docx_helper.py     builds the Paper 7 docx from out/idea07/results.json
    paper07_run.py defend_visuals.py              full rerun; defence dashboard (run defend_visuals.py from inside code/)
    kaggle_adapter.py README.md                   sir's adapter and code README
    out/idea07/                                   sir's Paper 7 results.json and figures
    explanations/                                 sir's Paper 7 results text and paired tests
  data/
    KaggleV2-May-2016.csv                         real no-show dataset (in the zip, not in git)
    README.md                                     source, columns, hash, why not in git
  prototypes/                                     our experiments exp01 to exp09
    logs/                                         their output
  notes/PROGRESS.md                               decisions, prior work, results so far, plan
```

## What differs from sir's copy

When copied, `code/`, `paper/` and the files in them were byte-identical to `Projects/`. Since then, one change: `common.py` and `paper_common.py` used absolute paths from the machine the package was built on (`/home/claude/project/...`). They now point to `code/out/` and `paper/`. Rebuilding Paper 7 from the stored results gives text identical to sir's docx.

To see exactly how our copy of any file differs from sir's original, compare it with the one in `Projects/`:

```
git diff --no-index Projects/code_package/code_package/common.py topic7/code/common.py
git diff --no-index Projects/code_package/code_package/paper_common.py topic7/code/paper_common.py
```

The same works for any file we change later (for example `ideas.py` or `paper07.py`).

## Run (from the repo root, with the virtual environment active)

Rebuild the Paper 7 docx from the stored results (a few seconds; writes `topic7/paper/`):

```
python3 topic7/code/paper07.py
```

Rerun sir's whole Paper 7 experiment, figures and docx (about 10 to 15 minutes; writes `topic7/code/out/idea07/` and `topic7/paper/`):

```
python3 topic7/code/paper07_run.py
```

Our prototypes:

```
python3 topic7/prototypes/exp07_real_pdp.py
```
