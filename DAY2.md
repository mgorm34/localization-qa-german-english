# Day 2 Results — Baseline German→English Translation

## Model
- **Model**: Helsinki-NLP/opus-mt-de-en (MarianMT)
- **Dataset**: WMT14 DE-EN (test split, 995 sentences)

## Evaluation Results

| Metric | Score |
|--------|-------|
| BLEU | 24.31 |
| BERTScore Precision | 0.9544 |
| BERTScore Recall | 0.9513 |
| BERTScore F1 | 0.9528 |

## What These Scores Mean
- **BLEU 24.31** is a solid baseline — the industry standard 
  for news translation sits between 20–30
- **BERTScore F1 0.9528** indicates strong semantic similarity 
  between model output and reference translations

## Linguistic Observations
- The model handles formal news text well
- Long complex German sentences are generally preserved
- Some loss of nuance in idiomatic expressions

## Files
- `data/processed/wmt14_de_en_1k.csv` — Test set (995 pairs)
- `data/processed/translations.csv` — Model translations
- `src/prepare_data.py` — Data prep script
- `src/run_translation.py` — Translation script
- `src/evaluate.py` — Evaluation script

## Next Steps (Day 3)
- Error analysis on low scoring sentences
- Compare with a second model
- Build Gradio demo
