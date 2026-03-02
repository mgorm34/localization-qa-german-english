"""
translate.py
Translates German text to English using Helsinki-NLP opus-mt model.
"""

from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm


MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"


def load_model():
    """Load the MarianMT tokenizer and model."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    print("Model loaded successfully.")
    return tokenizer, model


def translate(texts: list[str], tokenizer, model, batch_size: int = 16) -> list[str]:
    """
    Translate a list of German sentences to English.

    Args:
        texts: List of German strings
        tokenizer: MarianTokenizer instance
        model: MarianMTModel instance
        batch_size: Number of sentences per batch

    Returns:
        List of translated English strings
    """
    translations = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        outputs = model.generate(**inputs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


if __name__ == "__main__":
    from data_loading import load_wmt

    tokenizer, model = load_model()

    data = load_wmt(split="test", num_samples=5)
    german_texts = [pair["de"] for pair in data]
    references = [pair["en"] for pair in data]

    print("\n--- Translating ---")
    translated = translate(german_texts, tokenizer, model)

    for de, ref, hyp in zip(german_texts, references, translated):
        print(f"DE  : {de}")
        print(f"REF : {ref}")
        print(f"HYP : {hyp}")
        print()
