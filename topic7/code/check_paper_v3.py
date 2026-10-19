"""check_paper_v3.py -- checks version 3 of the paper (every result from the real data).

    python topic7/code/check_paper_v3.py

  1. no em or en dashes and the word "sir" nowhere in the text
  2. every hyphenated word that version 3 adds beyond Paper 7's own text is listed (they should be names, reference
     titles, page ranges or the CSV file name only)
  3. counts: tables, figures and references, and how many of Paper 7's paragraphs are kept word for word
Every number in version 3 is computed by paper07_v3.py from out/topic7_real/results.json; none is typed by hand.
"""
import os, re, sys
from check_paper import items, hyphen_words, align, ORIG

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.normpath(os.path.join(HERE, "..", "paper", "Paper07_v3_real_data.docx"))
ALLOWED = re.compile(r"^(Q-learning\.?|Q-learner|[Nn]o-shows?|[Ss]cikit-learn|double-booking|multi-objective|KaggleV2-May-2016\.csv\.?|\d+-\d+)$")


def main():
    o, v = items(ORIG), items(V3)
    prose = [t for k, t in v if k != "code"]
    dashes = [t for t in prose if "—" in t or "–" in t]
    sir = [t for t in prose if re.search(r"\bsir\b", t, re.I)]
    ho, hv = hyphen_words([t for k, t in o if k != "code"]), hyphen_words(prose)
    new = {w: n - ho.get(w, 0) for w, n in sorted(hv.items()) if n - ho.get(w, 0) > 0}
    odd = {w: n for w, n in new.items() if not ALLOWED.match(w)}
    pos, _ = align(o, v)
    kept = sum(1 for p in pos if p is not None)
    n_tab = len({t.split(".")[0] for k, t in v if k == "text" and re.match(r"^Table \d+\. ", t)})
    n_fig = len({t.split(".")[0] for k, t in v if k == "text" and re.match(r"^Figure \d+\. ", t)})
    n_ref = sum(1 for k, t in v if k == "text" and re.match(r"^\[\d+\] ", t))
    print(f"Version 3: {len(v)} items; {n_tab} tables, {n_fig} figures, {n_ref} references")
    print(f"  em or en dashes: {len(dashes)}; 'sir': {len(sir)}")
    print("  hyphenated words added beyond Paper 7's text:", ", ".join(f"{w} ({n})" for w, n in new.items()) or "none")
    print("  of which not a name, title, page range or file name:", ", ".join(odd) or "none")
    print(f"  Paper 7 items kept word for word: {kept} of {len(o)} (headings, unchanged sentences, references and code)")
    ok = not dashes and not sir and not odd and n_tab == 9 and n_fig == 7 and n_ref == 20
    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
