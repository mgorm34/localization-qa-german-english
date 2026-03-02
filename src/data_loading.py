"""
data_loading.py
Loads and previews German-English parallel datasets.
"""

from datasets import load_dataset


def load_wmt(split: str = "test", num_samples: int = 100):
    """
    Load WMT14 German-English dataset.

    Args:
        split: Dataset split - 'train', 'validation', or 'test'
        num_samples: Number of samples to load

    Returns:
        List of dicts with 'de' and 'en' keys
    """
    print(f"Loading WMT14 DE-EN ({split} split)...")
    dataset = load_dataset("wmt14", "de-en", split=split)
    samples = dataset.select(range(num_samples))

    pairs = [
        {
            "de": item["translation"]["de"],
            "en": item["translation"]["en"]
        }
        for item in samples
    ]

    print(f"Loaded {len(pairs)} sentence pairs.")
    return pairs


def load_opensubtitles(num_samples: int = 100):
    """
    Load OpenSubtitles German-English dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of dicts with 'de' and 'en' keys
    """
    print("Loading OpenSubtitles DE-EN...")
    dataset = load_dataset(
        "Helsinki-NLP/opus_opensubtitles",
        lang1="de",
        lang2="en",
        split="train"
    )
    samples = dataset.select(range(num_samples))

    pairs = [
        {
            "de": item["translation"]["de"],
            "en": item["translation"]["en"]
        }
        for item in samples
    ]

    print(f"Loaded {len(pairs)} sentence pairs.")
    return pairs


if __name__ == "__main__":
    # Quick preview
    wmt_data = load_wmt(split="test", num_samples=5)
    print("\n--- WMT14 Sample ---")
    for pair in wmt_data:
        print(f"DE: {pair['de']}")
        print(f"EN: {pair['en']}")
        print()
