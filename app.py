import sys
import math
import torch
import gradio as gr
from sacrebleu.metrics import CHRF
from bert_score import score as bert_score_fn
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, "src")
from qa_glossary import load_glossary, load_nlp, run_qa

MODELS = {
    "DE to EN": {
        "translation": "Helsinki-NLP/opus-mt-de-en",
        "back_translation": "Helsinki-NLP/opus-mt-en-de",
        "source_lang": "de",
        "target_lang": "en",
    },
    "EN to DE": {
        "translation": "Helsinki-NLP/opus-mt-en-de",
        "back_translation": "Helsinki-NLP/opus-mt-de-en",
        "source_lang": "en",
        "target_lang": "de",
    },
}

loaded_models = {}

def get_model(model_name):
    if model_name not in loaded_models:
        print(f"Loading {model_name}...")
        tok = MarianTokenizer.from_pretrained(model_name)
        mod = MarianMTModel.from_pretrained(model_name)
        mod.eval()
        loaded_models[model_name] = (tok, mod)
    return loaded_models[model_name]

print("Loading GPT2...")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

print("Loading spaCy...")
nlp = load_nlp()
print("Loading glossary...")
glossary = load_glossary()

print("Loading translation models...")
get_model(MODELS["DE to EN"]["translation"])
get_model(MODELS["DE to EN"]["back_translation"])
get_model(MODELS["EN to DE"]["translation"])
get_model(MODELS["EN to DE"]["back_translation"])
print("All models loaded!")

def translate_single(text, tok, mod):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = mod.generate(**inputs)
    first = outputs[0]
    return tok.decode(first, skip_special_tokens=True)


def translate_multiple(text, tok, mod, num_options=3):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = mod.generate(
        **inputs,
        num_beams=max(num_options * 3, 10),
        num_return_sequences=num_options,
        early_stopping=True,
        length_penalty=1.0,
    )
    translations = []
    seen = set()
    for i in range(len(outputs)):
        decoded = tok.decode(outputs[i], skip_special_tokens=True)
        if decoded not in seen:
            seen.add(decoded)
            translations.append(decoded)
    if not translations:
        translations = [translate_single(text, tok, mod)]
    return translations


def compute_chrf(hypothesis, reference):
    if not reference or not reference.strip():
        return None
    chrf = CHRF()
    score = chrf.sentence_score(hypothesis, [reference])
    return round(score.score, 2)


def compute_bert(hypothesis, reference, lang="en"):
    if not reference or not reference.strip():
        return None
    P, R, F1 = bert_score_fn([hypothesis], [reference], lang=lang, verbose=False)
    f1_list = F1.tolist()
    return round(f1_list[0], 4)


def compute_perplexity(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = gpt2_model(**inputs, labels=inputs["input_ids"]).loss
    return round(math.exp(loss.item()), 2)


def compute_back_translation(source, hypothesis, back_tok, back_mod):
    back = translate_single(hypothesis, back_tok, back_mod)
    chrf = CHRF()
    score = chrf.sentence_score(back, [source])
    return round(score.score, 2), back


def score_translation(source, hypothesis, reference, direction_config, back_tok, back_mod):
    target_lang = direction_config["target_lang"]
    chrf_score = compute_chrf(hypothesis, reference)
    bert_sc = compute_bert(hypothesis, reference, lang=target_lang)
    if target_lang == "en":
        perplexity = compute_perplexity(hypothesis)
    else:
        perplexity = None
    back_score, back_text = compute_back_translation(source, hypothesis, back_tok, back_mod)
    return {
        "chrf": chrf_score,
        "bertscore": bert_sc,
        "perplexity": perplexity,
        "back_score": back_score,
        "back_text": back_text,
    }

def classify_error(scores, qa_result):
    has_qa_miss = len(qa_result["glossary_coverage"]["misses"]) > 0
    chrf_score = scores["chrf"]
    bertscore = scores["bertscore"]
    perplexity = scores["perplexity"]
    back_score = scores["back_score"]
    if chrf_score is not None and bertscore is not None:
        low_chrf = chrf_score < 40
        low_bert = bertscore < 0.92
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


def format_scores(scores):
    lines = []
    if scores["chrf"] is not None:
        lines.append("chrF Score          : " + str(scores["chrf"]) + " / 100  (reference-based)")
    if scores["bertscore"] is not None:
        lines.append("BERTScore F1        : " + str(scores["bertscore"]) + " / 1.0  (reference-based)")
    if scores["perplexity"] is not None:
        lines.append("Perplexity          : " + str(scores["perplexity"]) + "  (good: <100 | ok: 100-300 | poor: 300+)")
    lines.append("Back-translation    : " + str(scores["back_score"]) + " / 100  (good: 60+ | partial: 30-60 | poor: <30)")
    lines.append("Back-translated     : " + scores["back_text"])
    NL = chr(10)
    return NL.join(lines)


def format_glossary_results(qa_result):
    hits = qa_result["glossary_coverage"]["hits"]
    misses = qa_result["glossary_coverage"]["misses"]
    cap = qa_result["capitalisation"]["count"]
    sep = qa_result["separable_verbs"]["count"]
    lines = []
    if hits:
        lines.append("GLOSSARY HITS:")
        for h in hits:
            lines.append("  OK: " + h["de_term"] + " -> " + h["en_gloss"])
    if misses:
        lines.append("GLOSSARY MISSES:")
        for m in misses:
            lines.append("  MISS: " + m["de_term"] + " -> expected " + m["en_gloss"])
    if not hits and not misses:
        lines.append("No glossary terms found in this sentence.")
    lines.append("Capitalised nouns detected : " + str(cap))
    lines.append("Separable verbs detected   : " + str(sep))
    NL = chr(10)
    return NL.join(lines)


def format_option(i, translation, scores, category):
    num = str(i + 1)
    NL = chr(10)
    header = "=== Option " + num + " ===  [" + category + "]"
    return header + NL + translation + NL + NL + format_scores(scores)


def empty_qa():
    return {
        "glossary_coverage": {"hits": [], "misses": []},
        "capitalisation": {"count": 0},
        "separable_verbs": {"count": 0},
    }

def run_pipeline(source_text, reference_text, direction):
    if not source_text.strip():
        return "", "", "", ""
    config = MODELS[direction]
    trans_tok, trans_mod = get_model(config["translation"])
    back_tok, back_mod = get_model(config["back_translation"])
    translations = translate_multiple(source_text, trans_tok, trans_mod, num_options=3)
    all_options = []
    best_translation = ""
    best_score = -1
    for i, hyp in enumerate(translations):
        scores = score_translation(source_text, hyp, reference_text, config, back_tok, back_mod)
        if direction == "DE to EN":
            qa_result = run_qa(source_text, hyp, glossary, nlp, verbose=False)
        else:
            qa_result = empty_qa()
        category = classify_error(scores, qa_result)
        all_options.append(format_option(i, hyp, scores, category))
        if scores["back_score"] > best_score:
            best_score = scores["back_score"]
            best_translation = hyp
    NL = chr(10)
    options_text = (NL + NL).join(all_options)
    if direction == "DE to EN":
        qa_result = run_qa(source_text, best_translation, glossary, nlp, verbose=False)
        qa_text = format_glossary_results(qa_result)
    else:
        qa_text = "Glossary QA currently available for DE to EN only." + NL + "EN to DE glossary coming soon."
    return best_translation, options_text, qa_text, direction


def rescore_edited(source_text, edited_translation, reference_text, direction):
    if not source_text.strip() or not edited_translation.strip():
        return "", ""
    config = MODELS[direction]
    back_tok, back_mod = get_model(config["back_translation"])
    scores = score_translation(source_text, edited_translation, reference_text, config, back_tok, back_mod)
    if direction == "DE to EN":
        qa_result = run_qa(source_text, edited_translation, glossary, nlp, verbose=False)
    else:
        qa_result = empty_qa()
    category = classify_error(scores, qa_result)
    NL = chr(10)
    score_text = "[" + category + "]" + NL + NL + format_scores(scores)
    if direction == "DE to EN":
        qa_text = format_glossary_results(qa_result)
    else:
        qa_text = "Glossary QA currently available for DE to EN only."
    return score_text, qa_text

SCORE_GUIDE = """| Metric | Range | Good | Moderate | Poor |
|--------|-------|------|----------|------|
| chrF | 0-100 | 60+ | 30-60 | below 30 |
| BERTScore F1 | 0-1 | 0.92+ | 0.85-0.92 | below 0.85 |
| Perplexity | 0-1000+ | below 100 | 100-300 | above 300 |
| Back-translation | 0-100 | 60+ | 30-60 | below 30 |
"""

ERROR_KEY = """| Category | Meaning |
|----------|----------|
| GOOD | No issues detected |
| FLUENCY | Surface form differs but meaning preserved |
| SEMANTIC | Meaning has drifted from reference |
| TERMINOLOGY | Glossary term missing from translation |
| REVIEW | Multiple issues - recommend human review |
"""

with gr.Blocks(title="DE EN Translation QA") as demo:
    gr.Markdown("# German - English Translation QA Tool")
    gr.Markdown(
        "Translate between German and English with automatic quality scoring. "
        "Get multiple translation options, edit any translation and rescore instantly. "
        "Reference translation is optional."
    )

    with gr.Row():
        direction = gr.Dropdown(
            choices=["DE to EN", "EN to DE"],
            value="DE to EN",
            label="Translation Direction",
            scale=1,
        )

    gr.Markdown("## 1. Translate")
    with gr.Row():
        with gr.Column():
            source_input = gr.Textbox(
                label="Source Text",
                placeholder="Enter German or English text here...",
                lines=4,
            )
            reference_input = gr.Textbox(
                label="Reference Translation (optional)",
                placeholder="Enter a reference to compute chrF and BERTScore...",
                lines=4,
            )
            translate_btn = gr.Button("Translate and Score", variant="primary", size="lg")

        with gr.Column():
            best_output = gr.Textbox(label="Best Translation", lines=4)
            options_output = gr.Textbox(label="All Translation Options with Scores", lines=20)
            qa_output = gr.Textbox(label="Glossary QA Results", lines=6)

    translate_btn.click(
        fn=run_pipeline,
        inputs=[source_input, reference_input, direction],
        outputs=[best_output, options_output, qa_output, direction],
    )

    gr.Markdown("## 2. Edit and Rescore")
    gr.Markdown("Copy a translation above or write your own, edit it, and see how scores change.")

    with gr.Row():
        with gr.Column():
            edited_input = gr.Textbox(
                label="Edited Translation",
                placeholder="Paste and edit a translation here...",
                lines=4,
            )
            rescore_btn = gr.Button("Rescore Edited Translation", variant="secondary")

        with gr.Column():
            rescore_output = gr.Textbox(label="Updated Scores", lines=8)
            rescore_qa_output = gr.Textbox(label="Updated Glossary QA", lines=6)

    rescore_btn.click(
        fn=rescore_edited,
        inputs=[source_input, edited_input, reference_input, direction],
        outputs=[rescore_output, rescore_qa_output],
    )

    with gr.Accordion("Score Guide", open=False):
        gr.Markdown(SCORE_GUIDE)
    with gr.Accordion("Error Category Key", open=False):
        gr.Markdown(ERROR_KEY)

    gr.Markdown("## Example Sentences")
    gr.Examples(
        examples=[
            ["Die Sicherheit des Systems wurde ueberprueft.", "The security of the system was checked.", "DE to EN"],
            ["Langer Atem macht sich bezahlt.", "Endurance pays dividends.", "DE to EN"],
            ["Die Benutzeroberflaeche ist sehr intuitiv.", "The user interface is very intuitive.", "DE to EN"],
            ["Das Passwort muss mindestens acht Zeichen lang sein.", "The password must be at least eight characters long.", "DE to EN"],
            ["The weather forecast predicts rain tomorrow.", "", "EN to DE"],
            ["Please restart the application after updating.", "", "EN to DE"],
        ],
        inputs=[source_input, reference_input, direction],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
