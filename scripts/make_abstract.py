"""Render abstract.pdf (one page) from the text below.  Numbers are taken from
runs_manhattan/report and runs/metric_selection; update the text if they change."""
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

TITLE = ("Large Pools of Acoustically Distinct Korean Words for Spoken Countersigns: "
         "a Confusability-Graph Pipeline with Machine-Listener Validation")

ABSTRACT = [
    "Spoken challenge–response words (countersigns) must be numerous enough to resist guessing yet hard to "
    "mishear over noisy radio. We present a reproducible pipeline that builds such word pools for Korean and "
    "quantifies the trade-off between pool size and confusability.",

    "Candidates are 2–3-syllable common nouns lemmatised from OpenSubtitles Korean (Kiwi), kept if at least as "
    "frequent as the bottom <i>x</i>% of a national basic-vocabulary list (<i>x</i> = 5, 10, 50; 20,262 words at "
    "<i>x</i> = 5). Words are converted to their pronounced surface form (g2pk2) and described by four "
    "articulatory features per segment. To choose the distance, four candidates — a GLM-calibrated edit "
    "distance and Manhattan, Euclidean and cosine distances on fixed-length syllable-slot vectors — were scored "
    "by Spearman correlation with pair confusion measured by a machine listener (text-to-speech → speech "
    "recognition, 0 and −5 dB) on one shared set of 250 pairs, with paired bootstrap tests. Manhattan "
    "(ρ = −0.43) and the calibrated edit distance (ρ = −0.38) were statistically tied, and remained tied on a "
    "second, held-out listener (−0.28 vs −0.38); Euclidean and cosine were significantly worse on both. "
    "Manhattan was adopted as the simpler metric.",

    "For each threshold τ, words closer than τ are joined in a confusability graph and the pool is a large "
    "independent set, found by integer programming where tractable and by greedy, randomised-forest and "
    "no-swap greedy heuristics otherwise; τ is set at the knee of pool size N(τ). At <i>x</i> = 5 the pool "
    "holds 6,536 words (12.67 bits), within 726 words (0.15 bits) of the solver's upper bound. On the "
    "held-out listener at −5 dB, in-pool confusion was 0.80 for the optimised pool versus 0.85 for an "
    "equal-size random pool and 0.90 for the most frequent words; results at <i>x</i> = 10 were similar, while "
    "at <i>x</i> = 50 the pools did not differ. A plain greedy pool performed as well as the optimised one.",

    "Thresholding a simple articulatory distance therefore yields large pools that, at the stricter "
    "frequency cutoffs, are modestly less confusable than frequency-based or random pools; exact "
    "optimisation adds words, not robustness. All validation uses synthetic speech and automatic recognition; confirmation with human "
    "listeners remains future work.",
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
