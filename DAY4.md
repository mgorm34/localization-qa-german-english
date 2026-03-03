# Day 4 Results — Sentence-Level Scoring and Error Analysis

## What We Built
- `src/score_sentences.py` — Sentence-level chrF, BERTScore and glossary QA pipeline
- `results/qa_report.csv` — Full 995-sentence QA report
- Error classification system with 5 categories: GOOD, FLUENCY, SEMANTIC, TERMINOLOGY, REVIEW

## Scoring Results

| Metric | Value |
|--------|-------|
| Total sentences | 995 |
| Avg chrF | 59.65 |
| Avg BERTScore F1 | 0.9544 |

## Error Categories

| Category | Count | Percentage |
|----------|-------|------------|
| GOOD | 871 | 87.5% |
| FLUENCY | 64 | 6.4% |
| SEMANTIC | 33 | 3.3% |
| REVIEW | 27 | 2.7% |
| TERMINOLOGY | 0 | 0% |

## Error Analysis — 10 Worst Sentences

### Error Type 1: Idiom and Figurative Language Failure
Model translates literally when idiomatic translation is required.
- Langer Atem macht sich bezahlt -> "Long breath pays off" (should be "Endurance pays dividends")
- Das Spiel war eine Hausnummer -> "a house number" (untranslatable without context)

### Error Type 2: Reference Worse Than Model (4 sentences)
Low chrF score is misleading — the model produced a better translation than the reference.
- Niemand weiss warum -> REF: "Nobody can answer questions about why" / HYP: "No one knows why" (HYP better)
- Wasser ist weiterhin kostenlos -> REF: "It will still give away water" / HYP: "Water is still free" (HYP better)

### Error Type 3: Tense and Register Mismatch
German present perfect does not map cleanly to English.
- Das habe ich verboten -> "I forbade that" (archaic) vs "I have forbidden that" (better)

### Error Type 4: Semantic Drift and Pronoun Ambiguity
German Sie is ambiguous (they vs formal you) causing wrong subject translation.
- Sie benoetigen -> "You need" instead of "They require"

### Error Type 5: Tautology
Verb and noun redundancy in output.
- Tombola verlost -> "a raffle was raffled" instead of "there was a raffle"

## Key Insight
A low chrF score does not always indicate a bad translation.
In 4 of the 10 worst-scoring sentences the model output was
actually superior to the reference translation. This highlights
a fundamental limitation of reference-based evaluation and points
to the need for multiple references or human evaluation alongside
automatic metrics.

## Next Steps (Day 5)
- Build Gradio demo wrapping the full pipeline
- Add domain-matched tech data to validate glossary QA
- Explore MQM error framework for more granular classification
