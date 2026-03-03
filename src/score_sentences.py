"""
score_sentences.py
Computes sentence-level scores and QA report.
"""

import csv
import sys
import spacy
from pathlib import Path
from sacrebleu.metrics import CHRF
from bert_score import score as bert_score_fn

sys.path.insert(0, "src")
from qa_glossary import load_glossary, load_nlp, run_qa

TRANSLATIONS_FILE = Path("data/processed/translations.csv")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "qa_report.csv"

def load_translations(path):
    sources, references, hypotheses = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row["de"])
            references.append(row["ref_en"])
            hypotheses.append(row["hyp_en"])
    print(f"Loaded {len(sources)} sentence pairs")
    return sources, references, hypotheses

def sentence_chrf(hypothesis, reference):
    chrf = CHRF()
    score = chrf.sentence_score(hypothesis, [reference])
    return round(score.score, 2)

def compute_bertscore(hypotheses, references):
    print("Computing BERTScore...")
    P, R, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
    p_list  = [round(float(v), 4) for v in P]
    r_list  = [round(float(v), 4) for v in R]
    f1_list = [round(float(v), 4) for v in F1]
    return p_list, r_list, f1_list

def classify_error(chrf_score, bertscore_f1, qa_result):
    has_qa_miss = len(qa_result["glossary_coverage"]["misses"]) > 0
    low_chrf    = chrf_score < 40
    low_bert    = bertscore_f1 < 0.92
    issues = sum([has_qa_miss, low_chrf, low_bert])
    if issues == 0:
        return "GOOD"
    elif issues > 1:
        return "REVIEW"
    elif has_qa_miss:
        return "TERMINOLOGY"
    elif low_bert:
        return "SEMANTIC"
    else:
        return "FLUENCY"

def build_report(sources, references, hypotheses, glossary, nlp):
    print("Computing chrF scores...")
    chrf_scores = [sentence_chrf(h, r) for h, r in zip(hypotheses, references)]
    p_scores, r_scores, f1_scores = compute_bertscore(hypotheses, references)
    print("Running glossary QA...")
    qa_results = []
    for source, hypothesis in zip(sources, hypotheses):
        qa_result = run_qa(source, hypothesis, glossary, nlp, verbose=False)
        qa_results.append(qa_result)
    print("Classifying errors...")
    rows = []
    for i, (src, ref, hyp) in enumerate(zip(sources, references, hypotheses)):
        chrf   = chrf_scores[i]
        bsp    = p_scores[i]
        bsr    = r_scores[i]
        bsf1   = f1_scores[i]
        qa     = qa_results[i]
        category = classify_error(chrf, bsf1, qa)
        misses = [m["de_term"] for m in qa["glossary_coverage"]["misses"]]
        hits   = [h["de_term"] for h in qa["glossary_coverage"]["hits"]]
        rows.append({
            "source":                 src,
            "reference":              ref,
            "hypothesis":             hyp,
            "chrf":                   chrf,
            "bertscore_p":            bsp,
            "bertscore_r":            bsr,
            "bertscore_f1":           bsf1,
            "glossary_hits":          "|".join(hits),
            "glossary_misses":        "|".join(misses),
            "capitalised_noun_count": qa["capitalisation"]["count"],
            "separable_verb_count":   qa["separable_verbs"]["count"],
            "error_category":         category
        })
    return rows

def save_report(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source", "reference", "hypothesis",
        "chrf", "bertscore_p", "bertscore_r", "bertscore_f1",
        "glossary_hits", "glossary_misses",
        "capitalised_noun_count", "separable_verb_count",
        "error_category"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Report saved to {output_path}")

def print_summary(rows):
    total = len(rows)
    categories = {}
    chrf_scores = [r["chrf"] for r in rows]
    bert_scores = [r["bertscore_f1"] for r in rows]
    for row in rows:
        cat = row["error_category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\n--- QA Report Summary ---")
    print(f"Total sentences : {total}")
    print(f"Avg chrF        : {round(sum(chrf_scores) / total, 2)}")
    print(f"Avg BERTScore F1: {round(sum(bert_scores) / total, 4)}")
    print("\nError Categories:")
    for cat, count in sorted(categories.items()):
        pct = round(count / total * 100, 1)
        print(f"  {cat} -- {count} sentences ({pct}%)")

if __name__ == "__main__":
    glossary = load_glossary()
    nlp      = load_nlp()
    sources, references, hypotheses = load_translations(TRANSLATIONS_FILE)
    rows = build_report(sources, references, hypotheses, glossary, nlp)
    save_report(rows, OUTPUT_FILE)
    print_summary(rows)
    print("\n✅ QA report complete!")
