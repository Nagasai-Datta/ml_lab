# ml_lab

Course project on reinforcement learning in medicine. Our topic is **Topic 7: appointment overbooking and no-show management**. We build on the faculty's Paper 7 and its code, beat the paper's results with a better method, explain why it wins, and add our results to the paper.

## Layout

```
Projects/      sir's full package for all ten topics, exactly as handed out
topic7/        everything for Topic 7; all our work happens here (see topic7/README.md)
requirements.txt
```

`Projects/` is reference only and never edited. To check that nothing in it has changed since it was first committed:

```
git log --oneline -- Projects
```

shows only the first commit.

## Setup (Mac)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then see `topic7/README.md` for how to run things.
