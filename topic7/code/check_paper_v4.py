"""check_paper_v4.py -- checks version 4 of the paper (version 3 without the reviewer questions and the source code).

    python topic7/code/check_paper_v4.py

  1. the same checks as version 3: no em or en dashes, no "sir", added hyphenated words are names, titles, page
     ranges or the CSV file name only, and 9 tables, 7 figures and 20 references
  2. version 4 is exactly version 3 with Appendix A (questions) and Appendix C (code) removed and Appendix B renamed
     Appendix A: every other paragraph, table cell, figure and reference is identical and in the same order
Run paper07_v3.py and paper07_v4.py first.
"""
import os, sys
from check_paper import items
import check_paper_v3

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.normpath(os.path.join(HERE, "..", "paper", "Paper07_v4_real_data_short.docx"))


def expected_from_v3(v3):
    """version 3 items with the reviewer questions and the source code removed"""
    heads = [t for _, t in v3]
    qa = heads.index("Appendix A. Anticipated reviewer questions")
    repro = heads.index("Appendix B. Reproducibility")
    code = heads.index("Appendix C. Source code of the experiment")
    refs = heads.index("References")
    return v3[:qa] + [("text", "Appendix A. Reproducibility")] + v3[repro + 1:code] + v3[refs:]


def main():
    ok = check_paper_v3.main(V4, "Version 4")
    v3, v4 = items(check_paper_v3.V3), items(V4)
    same = v4 == expected_from_v3(v3)
    gone = [t for _, t in v4 if t.startswith(("Appendix A. Anticipated", "Appendix C. Source code")) or t.startswith("Q1. ")]
    print(f"\nVersion 4 = version 3 without the questions and the code: {'yes' if same else 'NO'} "
          f"({len(v3) - len(v4)} of {len(v3)} items removed); leftover question or code headings: {len(gone)}")
    ok = ok and same and not gone
    print("\nALL VERSION 4 CHECKS PASSED" if ok else "\nVERSION 4 CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
