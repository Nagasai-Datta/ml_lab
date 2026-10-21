"""paper07_v4.py -- version 4 of the paper: version 3 without Appendix A (anticipated reviewer questions) and
Appendix C (source code of the experiment). Everything else, including the authors, the tables, the figures and the
references, is built by the same code as version 3, so the two never disagree; the reproducibility appendix stays and
becomes Appendix A.

    python topic7/code/paper07_v4.py      writes topic7/paper/Paper07_v4_real_data_short.docx

Reads out/topic7_real/results.json, which run_real.py writes. Versions 1, 2 and 3 are not affected.
"""
import paper07_v3


def build():
    return paper07_v3.build(short=True)


if __name__ == "__main__":
    print(build())
