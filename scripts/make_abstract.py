"""Render abstract.pdf (one page) from the text below.  Numbers are taken from
runs_manhattan/report and runs/metric_selection; update the text if they change."""
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

TITLE = ("Maximum-Entropy Countersign Pools: "
         "a Confusability-Graph Analysis of Korean Spoken Authentication Words")

ABSTRACT = [
    "Countersigns are the last line of friend-or-foe authentication in low visibility, yet their security has "
    "never been systematically analyzed. In practice words recur too often, some are obsolete or "
    "unrecognizable, and others are easily misheard — silent failures that accumulate into a wartime "
    "vulnerability. Under Kerckhoffs's assumption, a scheme's security reduces to the entropy of the word pool, "
    "log<sub>2</sub><i>N</i>. Countersign design therefore becomes a constrained optimization problem: build the "
    "largest word set that remains recognizable, pronounceable, and phonetically distinguishable. From a "
    "688,129-entry Korean subtitle corpus we keep 2–3-syllable common nouns at least as frequent as a "
    "national basic-vocabulary list (20,262 words), represent each word by the articulatory features of its "
    "pronounced form, and quantify pairwise confusability by the Manhattan distance between feature vectors, "
    "selected over a calibrated edit distance and two alternatives by how well each predicts misrecognition, "
    "using paired bootstrap tests on a shared set of 250 word pairs. "
    "The valid pool is a maximum independent set of the resulting confusability graph, solved by integer "
    "programming and greedy heuristics at the knee of the size–threshold trade-off. The final pool of "
    "6,536 words carries 12.67 bits of daily entropy, within 0.15 bits of the solver's upper bound. "
    "Relaxing the frequency cutoff trades entropy for familiarity, from 12.67 bits to 12.02 and 9.65 bits "
    "as the cutoff rises. Two "
    "validations support it: the distance tracks confusions on two independent speech-synthesis–"
    "recognition channels (Spearman ρ = −0.43 and −0.28), and in noise the pool is misheard "
    "less often than equal-size random or most-frequent word lists (0.80 vs. 0.85 vs. 0.90 at −5 dB). "
    "Using only public data, we contribute the first quantitative security analysis of countersign pools and "
    "a ready-to-use word list.",
]

KEYWORDS = ("Keywords: phonological confusability; Korean; maximum independent set; "
            "speech-in-noise; word-list design; countersign")


def main():
    out = Path(__file__).resolve().parent.parent / "abstract.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm, title="Abstract",
                            subject=TITLE)
    t = ParagraphStyle("t", fontName="Times-Bold", fontSize=14, leading=18, alignment=TA_CENTER)
    h = ParagraphStyle("h", fontName="Times-Bold", fontSize=11, leading=14, alignment=TA_CENTER)
    b = ParagraphStyle("b", fontName="Times-Roman", fontSize=11, leading=14.5, alignment=TA_JUSTIFY,
                       spaceAfter=7)
    k = ParagraphStyle("k", parent=b, fontName="Times-Italic", fontSize=10)
    story = [Paragraph(TITLE, t), Spacer(1, 18), Paragraph("Abstract", h), Spacer(1, 8)]
    story += [Paragraph(p, b) for p in ABSTRACT]
    story += [Spacer(1, 4), Paragraph(KEYWORDS, k)]
    doc.build(story)
    print(out)


if __name__ == "__main__":
    main()
