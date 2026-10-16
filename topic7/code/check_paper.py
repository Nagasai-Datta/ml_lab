"""check_paper.py -- checks the two extended versions of Paper 7 against the docx that was handed out.

    python topic7/code/check_paper.py

  1. version 1 (Paper07_v1_additions.docx): every paragraph, table cell and caption of the original appears unchanged
     and in the same order (only additions)
  2. version 2 (Paper07_v2_edited.docx, Paper 7 and our work as one paper): lists every original paragraph that was
     edited; everything else of Paper 7 must be unchanged and in order (citation numbers aside, since they renumber)
  3. our added text: no em or en dashes, the word "sir" nowhere, and a list of every hyphenated word
"""
import os, re, sys
from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
PAP = os.path.normpath(os.path.join(HERE, "..", "paper"))
ORIG = os.path.normpath(os.path.join(HERE, "..", "..", "Projects", "papers", "Paper07_Clinic_Overbooking_Exploration.docx"))  # as handed out
V1 = os.path.join(PAP, "Paper07_v1_additions.docx")
V2 = os.path.join(PAP, "Paper07_v2_edited.docx")


def items(path):
    """body in reading order: (kind, text) for paragraphs and table cells; kind 'code' for source listings"""
    doc = Document(path); out = []
    body = doc.element.body
    for el in body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "p":
            fonts = {r.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii")
                     for r in el.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts")}
            text = "".join(t.text or "" for t in el.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
            has_img = any(True for _ in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"))
            if has_img:
                out.append(("image", "[image]"))
            elif text.strip():
                out.append(("code" if "Courier New" in fonts else "text", text))
        elif tag == "tbl":
            for tc in el.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                out.append(("cell", "".join(t.text or "" for t in tc.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))))
    return out


def align(orig, new):
    """greedy subsequence match: returns indices in new of each orig item (None if missing) and the unmatched new items"""
    j, pos = 0, []
    for k, t in orig:
        jj = j
        while jj < len(new) and new[jj] != (k, t):
            jj += 1
        if jj < len(new):
            pos.append(jj); j = jj + 1
        else:
            pos.append(None)
    used = {p for p in pos if p is not None}
    return pos, [new[i] for i in range(len(new)) if i not in used]


def hyphen_words(texts):
    out = {}
    for t in texts:
        for w in re.findall(r"[A-Za-z0-9.]+(?:-[A-Za-z0-9.]+)+", t):
            out[w] = out.get(w, 0) + 1
    return out


def report_added(added, label, minus=()):
    """added: (kind, text) items; minus: original texts whose hyphenated words are not ours (edited paragraphs)"""
    prose = [t for k, t in added if k != "code"]
    bad = [t for t in prose if "—" in t or "–" in t]
    sir = [t for t in prose if re.search(r"\bsir\b", t, re.I)]
    hyph, theirs = hyphen_words(prose), hyphen_words(minus)
    hyph = {w: n - theirs.get(w, 0) for w, n in hyph.items() if n - theirs.get(w, 0) > 0}
    print(f"  {label}: {len(added)} added items; em or en dashes: {len(bad)}; 'sir': {len(sir)}")
    print("  hyphenated words in our prose:", ", ".join(f"{w} ({n})" for w, n in sorted(hyph.items())) or "none")
    return not bad and not sir


def main():
    ok = True
    o = items(ORIG)
    v1 = items(V1)
    pos, added = align(o, v1)
    missing = [o[i] for i, p in enumerate(pos) if p is None]
    print(f"Version 1: {len(o)} original items, {len(o) - len(missing)} found unchanged and in order, {len(missing)} missing")
    for k, t in missing:
        print("   MISSING:", t[:160])
    ok &= not missing
    ok &= report_added(added, "version 1")
    if os.path.exists(V2):
        # version 2 is one paper: references are renumbered, so citation numbers are ignored when comparing
        norm = lambda its: [(k, re.sub(r"\[\d+(?:,\d+)*\]", "[n]", t)) for k, t in its]
        on, v2 = norm(o), norm(items(V2))
        pos2, added2 = align(on, v2)
        changed = [o[i][1] for i, p in enumerate(pos2) if p is None]
        print(f"\nVersion 2: {len(on) - len(changed)} of {len(on)} original items unchanged and in order (citation numbers aside); "
              f"{len(changed)} original paragraphs edited:")
        for t in changed:
            print("   EDITED:", t[:150] + ("..." if len(t) > 150 else ""))
        ok &= report_added(added2, "version 2 (new and edited text)", minus=changed)
    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
