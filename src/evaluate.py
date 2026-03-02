"""
evaluate.py
Evaluates translation quality using sacreBLEU and BERTScore.
"""

import sacrebleu
from bert_score import score as bert_score_fn


def evaluate_bleu(hypotheses: list[str], references: list[str]) -> dict:
    """
    Compute BLEU score.

    Args:
        hypotheses: List of model translations
        references: List of reference translations

    Returns:
        Dict with BLEU score and details
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    result = {
        "bleu": round(bleu.score, 2),
        "details": str(bleu)
    }
    print(f"BLEU Score: {result['bleu']}")
    return result


def evaluate_bertscore(hypotheses: list[str], references: list[str], lang: str = "en") -> dict:
    """
    Compute BERTScore.

    Args:
        hypotheses: List of model translations
        references: List of reference translations
        lang: Target language code

    Returns:
        Dict with precision, recall, and F1
    """
    print("Computing BERTScore (this may take a moment)...")
    P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    result = {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4)
    }
    print(f"BERTScore — P: {result['precision']} | R: {result['recall']} | F1: {result['f1']}")
    return result


if __name__ == "__main__":
    from data_loading import load_wmt
    from translate import load_model, translate

    data = load_wmt(split="test", num_samples=20)
    german_texts = [pair["de"] for pair in data]
    references = [pair["en"] for pair in data]

    tokenizer, model = load_model()
    hypotheses = translate(german_texts, tokenizer, model)

    print("\n--- Evaluation Results ---")
    evaluate_bleu(hypotheses, references)
    evaluate_bertscore(hypotheses, references)
