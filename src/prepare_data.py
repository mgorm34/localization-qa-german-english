"""
prepare_data.py
Downloads WMT14 DE-EN dataset and creates a reproducible
1k-2k sentence test set saved to data/processed/
"""

import csv
import random
from pathlib import Path
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────
SEED        = 42
NUM_SAMPLES = 1000
RAW_DIR     = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "wmt14_de_en_1k.csv"
# ────────────────────────────────────────────────────────


def download_wmt14(split: str = "test"):
    """Download WMT14 DE-EN dataset."""
    print(f"Downloading WMT14 DE-EN ({split} split)...")
    dataset = load_dataset("wmt14", "de-en", split=split)
    print(f"Total sentences available: {len(dataset)}")
    return dataset


def sample_pairs(dataset, num_samples: int = NUM_SAMPLES, seed: int = SEED):
    """Randomly sample sentence pairs."""
    random.seed(seed)
    total = len(dataset)
    indices = random.sample(range(total), min(num_samples, total))
    
    pairs = []
    for i in indices:
        item = dataset[i]
        de = item["translation"]["de"].strip()
        en = item["translation"]["en"].strip()
        # Basic quality filter - skip very short or very long sentences
        if 3 < len(de.split()) < 100:
            pairs.append({"de": de, "en": en})
    
    print(f"Sampled {len(pairs)} sentence pairs after filtering.")
    return pairs


def save_to_csv(pairs: list[dict], output_path: Path):
    """Save sentence pairs to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["de", "en"])
        writer.writeheader()
        writer.writerows(pairs)
    
    print(f"Saved {len(pairs)} pairs to {output_path}")


def preview(pairs: list[dict], n: int = 5):
    """Print a quick preview of the data."""
    print(f"\n--- Preview (first {n} pairs) ---")
    for pair in pairs[:n]:
        print(f"DE: {pair['de']}")
        print(f"EN: {pair['en']}")
        print()


if __name__ == "__main__":
    # Run the full pipeline
    dataset = download_wmt14(split="test")
    pairs   = sample_pairs(dataset, num_samples=NUM_SAMPLES)
    save_to_csv(pairs, OUTPUT_FILE)
    preview(pairs)
    print("✅ Data prep complete!")
