"""
run_translation.py
Runs baseline German→English translation using Helsinki-NLP/opus-mt-de-en
Input:  data/processed/wmt14_de_en_1k.csv
Output: data/processed/translations.csv
"""

import csv
import time
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
MODEL_NAME   = "Helsinki-NLP/opus-mt-de-en"
INPUT_FILE   = Path("data/processed/wmt14_de_en_1k.csv")
OUTPUT_FILE  = Path("data/processed/translations.csv")
BATCH_SIZE   = 16
MAX_LENGTH   = 512
# ────────────────────────────────────────────────────────


def load_model():
    """Load MarianMT tokenizer and model."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model     = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model


def load_csv(input_path: Path) -> tuple[list[str], list[str]]:
    """Load German source and English reference sentences from CSV."""
    sources, references = [], []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row["de"].strip())
            references.append(row["en"].strip())
    print(f"Loaded {len(sources)} sentence pairs from {input_path}")
    return sources, references


def translate_batch(
    texts: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    batch_size: int = BATCH_SIZE
) -> list[str]:
    """Translate a list of German sentences in batches."""
    translations = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        outputs = model.generate(**inputs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


def save_results(
    sources: list[str],
    references: list[str],
    hypotheses: list[str],
    output_path: Path
):
    """Save source, reference and hypothesis to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["de", "ref_en", "hyp_en"])
        writer.writeheader()
        for de, ref, hyp in zip(sources, references, hypotheses):
            writer.writerow({"de": de, "ref_en": ref, "hyp_en": hyp})

    print(f"Saved {len(hypotheses)} translations to {output_path}")


def preview_results(
    sources: list[str],
    references: list[str],
    hypotheses: list[str],
    n: int = 5
):
    """Print a quick preview of translations."""
    print(f"\n--- Preview (first {n} translations) ---")
    for de, ref, hyp in zip(sources[:n], references[:n], hypotheses[:n]):
        print(f"DE  : {de}")
        print(f"REF : {ref}")
        print(f"HYP : {hyp}")
        print()


if __name__ == "__main__":
    start = time.time()

    # Load model
    tokenizer, model = load_model()

    # Load data
    sources, references = load_csv(INPUT_FILE)

    # Translate
    print(f"\nTranslating {len(sources)} sentences in batches of {BATCH_SIZE}...")
    hypotheses = translate_batch(sources, tokenizer, model)

    # Save and preview
    save_results(sources, references, hypotheses, OUTPUT_FILE)
    preview_results(sources, references, hypotheses)

    elapsed = time.time() - start
    print(f"✅ Translation complete in {elapsed:.1f}s")
