
import sys
import math
import torch
import gradio as gr
from sacrebleu.metrics import CHRF
from bert_score import score as bert_score_fn
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, "src")
from qa_glossary import load_glossary, load_nlp, run_qa

DE_EN_MODEL = "Helsinki-NLP/opus-mt-de-en"
EN_DE_MODEL = "Helsinki-NLP/opus-mt-en-de"
GPT2_MODEL  = "gpt2"

print("Loading DE->EN model...")
tokenizer = MarianTokenizer.from_pretrained(DE_EN_MODEL)
model     = MarianMTModel.from_pretrained(DE_EN_MODEL)
model.eval()

print("Loading EN->DE back-translation model...")
back_tokenizer = MarianTokenizer.from_pretrained(EN_DE_MODEL)
back_model     = MarianMTModel.from_pretrained(EN_DE_MODEL)
back_model.eval()

print("Loading GPT2 for perplexity...")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_MODEL)
gpt2_model     = GPT2LMHeadModel.from_pretrained(GPT2_MODEL)
gpt2_model.eval()

print("Loading spaCy...")
nlp = load_nlp()
print("Loading glossary...")
glossary = load_glossary()
print("All models loaded!")


def translate_text(text, tok, mod):
    inputs  = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = mod.generate(**inputs)
    decoded = tok.batch_decode(outputs, skip_special_tokens=True)
    return next(iter(decoded))


def compute_chrf(hypothesis, reference):
    if not reference.strip():
        return None
    chrf  = CHRF()
    score = chrf.sentence_score(hypothesis, [reference])
    return round(score.score, 2)


def compute_bert(hypothesis, reference):
    if not reference.strip():
        return None
    P, R, F1 = bert_score_fn([hypothesis], [reference], lang="en", verbose=False)
    f1_val   = next(iter(F1.tolist()))
    return round(f1_val, 4)


def compute_perplexity(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = gpt2_model(**inputs, labels=inputs["input_ids"]).loss
    return round(math.exp(loss.item()), 2)


def compute_back_translation(source_de, hypothesis_en):
    back  = translate_text(hypothesis_en, back_tokenizer, back_model)
    chrf  = CHRF()
    score = chrf.sentence_score(back, [source_de])
    return round(score.score, 2), back


def classify_error(chrf_score, bertscore_f1, qa_result, perplexity, back_score):
    has_qa_miss = len(qa_result["glossary_coverage"]["misses"]) > 0
    if chrf_score is not None and bertscore_f1 is not None:
        low_chrf = chrf_score < 40
        low_bert = bertscore_f1 < 0.92
    else:
        low_chrf = back_score is not None and back_score < 40
        low_bert = perplexity is not None and perplexity > 200
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


def format_glossary_results(qa_result):
    hits   = qa_result["glossary_coverage"]["hits"]
    misses = qa_result["glossary_coverage"]["misses"]
    cap    = qa_result["capitalisation"]["count"]
    sep    = qa_result["separable_verbs"]["count"]
    lines  = []
    if hits:
        lines.append("GLOSSARY HITS:")
        for h in hits:
            lines.append(f"  OK: {h['de_term']} -> {h['en_gloss']}")
    if misses:
        lines.append("GLOSSARY MISSES:")
        for m in misses:
            lines.append(f"  MISS: {m['de_term']} -> expected {m['en_gloss']}")
    if not hits and not misses:
        lines.append("No glossary terms found in this sentence.")
    lines.append(f"Capitalised nouns detected : {cap}")
    lines.append(f"Separable verbs detected   : {sep}")
    return "\n".join(lines)


def run_pipeline(german_text, reference_text):
    if not german_text.strip():
        return "", "", "", ""
    hypothesis   = translate_text(german_text, tokenizer, model)
    chrf_score   = compute_chrf(hypothesis, reference_text)
    bert_score   = compute_bert(hypothesis, reference_text)
    perplexity   = compute_perplexity(hypothesis)
    back_score, back_translation = compute_back_translation(german_text, hypothesis)
    qa_result    = run_qa(german_text, hypothesis, glossary, nlp, verbose=False)
    qa_text      = format_glossary_results(qa_result)
    category     = classify_error(chrf_score, bert_score, qa_result, perplexity, back_score)
    score_lines  = []
    if chrf_score is not None:
        score_lines.append(f"chrF Score          : {chrf_score} / 100  (reference-based)")
    if bert_score is not None:
        score_lines.append(f"BERTScore F1        : {bert_score} / 1.0  (reference-based)")
    score_lines.append(f"Perplexity          : {perplexity}  (good: <100 | ok: 100-300 | poor: 300+)")
    score_lines.append(f"Back-translation    : {back_score} / 100  (good: 60+ | partial: 30-60 | poor: <30)")
    score_lines.append(f"Back-translated DE  : {back_translation}")
    return hypothesis, "\n".join(score_lines), qa_text, category


SCORE_GUIDE = """| Metric | Range | Good | Moderate | Poor |
|--------|-------|------|----------| -----|
| chrF | 0-100 | 60+ | 30-60 | below 30 |
| BERTScore F1 | 0-1 | 0.92+ | 0.85-0.92 | below 0.85 |
| Perplexity | 0-1000+ | below 100 | 100-300 | above 300 |
| Back-translation | 0-100 | 60+ | 30-60 | below 30 |
"""

ERROR_KEY = """| Category | Meaning |
|----------|---------|
| GOOD | No issues detected |
| FLUENCY | Surface form differs but meaning preserved |
| SEMANTIC | Meaning has drifted from reference |
| TERMINOLOGY | Glossary term missing from translation |
| REVIEW | Multiple issues - recommend human review |
"""

with gr.Blocks(title="German-English MT QA Demo") as demo:
    gr.Markdown("# German to English Translation QA Demo")
    gr.Markdown(
        "Translate a German sentence and get automatic quality analysis. "
        "Reference translation is optional - reference-free scores are always computed."
    )
    with gr.Row():
        with gr.Column():
            german_input = gr.Textbox(
                label="German Source Text",
                placeholder="Enter German text here...",
                lines=4
            )
            reference_input = gr.Textbox(
                label="Reference Translation (optional)",
                placeholder="Enter a reference to compute chrF and BERTScore...",
                lines=4
            )
            run_btn = gr.Button("Translate and Analyse", variant="primary")
        with gr.Column():
            translation_output = gr.Textbox(label="MT Translation", lines=4)
            scores_output      = gr.Textbox(label="Scores", lines=6)
            qa_output          = gr.Textbox(label="Glossary QA Results", lines=6)
            category_output    = gr.Textbox(label="Error Category")

    gr.Markdown("## Score Guide")
    gr.Markdown(SCORE_GUIDE)
    gr.Markdown("## Error Category Key")
    gr.Markdown(ERROR_KEY)
    gr.Markdown("## Example Sentences")
    gr.Examples(
        examples=[
            ["Langer Atem macht sich bezahlt.", "Endurance pays dividends."],
            ["Die Benutzeroberfläche ist sehr intuitiv.", "The user interface is very intuitive."],
            ["Beim Herunterladen der Datei trat ein Fehler auf.", "An error occurred while downloading the file."],
            ["Niemand weiss, warum es geschieht.", "Nobody knows why it happens."],
            ["Das Passwort muss mindestens acht Zeichen lang sein.", "The password must be at least eight characters long."],
        ],
        inputs=[german_input, reference_input]
    )
    run_btn.click(
        fn=run_pipeline,
        inputs=[german_input, reference_input],
        outputs=[translation_output, scores_output, qa_output, category_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
