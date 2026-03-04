import sys
import math
import torch
import gradio as gr
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

sys.path.insert(0, "src")
from qa_glossary import load_glossary, load_nlp, run_qa

# === MODEL CONFIG ===
MODELS = {
    "DE to EN": {
        "translation": "Helsinki-NLP/opus-mt-de-en",
        "back_translation": "Helsinki-NLP/opus-mt-en-de",
        "source_lang": "de",
        "target_lang": "en",
        "mask_model": "bert-base-uncased",
        "perplexity_model": "gpt2",
    },
    "EN to DE": {
        "translation": "Helsinki-NLP/opus-mt-en-de",
        "back_translation": "Helsinki-NLP/opus-mt-de-en",
        "source_lang": "en",
        "target_lang": "de",
        "mask_model": "bert-base-german-cased",
        "perplexity_model": "dbmdz/german-gpt2",
    },
}

# === MODEL CACHE ===
loaded_models = {}

def get_model(model_name, model_class, tokenizer_class):
    if model_name not in loaded_models:
        print(f"Loading {model_name}...")
        tok = tokenizer_class.from_pretrained(model_name)
        mod = model_class.from_pretrained(model_name)
        mod.eval()
        loaded_models[model_name] = (tok, mod)
    return loaded_models[model_name]

def get_translation_model(model_name):
    return get_model(model_name, MarianMTModel, MarianTokenizer)

def get_mask_model(model_name):
    return get_model(model_name, AutoModelForMaskedLM, AutoTokenizer)

def get_perplexity_model(model_name):
    return get_model(model_name, GPT2LMHeadModel, GPT2TokenizerFast)

# === LOAD ALL MODELS AT STARTUP ===
print("Loading spaCy...")
nlp_de = load_nlp()

print("Loading English spaCy...")
import spacy
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")

print("Loading glossary...")
glossary = load_glossary()

print("Loading translation models...")
get_translation_model(MODELS["DE to EN"]["translation"])
get_translation_model(MODELS["DE to EN"]["back_translation"])
get_translation_model(MODELS["EN to DE"]["translation"])
get_translation_model(MODELS["EN to DE"]["back_translation"])

print("Loading perplexity models...")
get_perplexity_model(MODELS["DE to EN"]["perplexity_model"])
get_perplexity_model(MODELS["EN to DE"]["perplexity_model"])

print("Loading mask-fill models...")
get_mask_model(MODELS["DE to EN"]["mask_model"])
get_mask_model(MODELS["EN to DE"]["mask_model"])

print("All models loaded!")


# === TRANSLATION FUNCTIONS ===
def translate_single(text, tok, mod):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = mod.generate(**inputs)
    return tok.decode(outputs[0], skip_special_tokens=True)


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


# === SCORING FUNCTIONS ===
def compute_perplexity(text, direction):
    config = MODELS[direction]
    tok, mod = get_perplexity_model(config["perplexity_model"])
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = mod(**inputs, labels=inputs["input_ids"]).loss
    return round(math.exp(loss.item()), 2)


def compute_back_translation(source, hypothesis, back_tok, back_mod):
    back = translate_single(hypothesis, back_tok, back_mod)
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    score = chrf.sentence_score(back, [source])
    return round(score.score, 2), back


def score_translation(source, hypothesis, direction, back_tok, back_mod):
    perplexity = compute_perplexity(hypothesis, direction)
    back_score, back_text = compute_back_translation(source, hypothesis, back_tok, back_mod)
    return {
        "perplexity": perplexity,
        "back_score": back_score,
        "back_text": back_text,
    }


def format_scores(scores):
    lines = []
    lines.append("Perplexity          : " + str(scores["perplexity"]) + "  (good: <100 | ok: 100-300 | poor: 300+)")
    lines.append("Back-translation    : " + str(scores["back_score"]) + " / 100  (good: 60+ | partial: 30-60 | poor: <30)")
    lines.append("Back-translated     : " + scores["back_text"])
    NL = chr(10)
    return NL.join(lines)


# === TRANSLATION FUNCTIONS ===
def translate_single(text, tok, mod):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = mod.generate(**inputs)
    return tok.decode(outputs[0], skip_special_tokens=True)


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


# === SCORING FUNCTIONS ===
def compute_perplexity(text, direction):
    config = MODELS[direction]
    tok, mod = get_perplexity_model(config["perplexity_model"])
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = mod(**inputs, labels=inputs["input_ids"]).loss
    return round(math.exp(loss.item()), 2)


def compute_back_translation(source, hypothesis, back_tok, back_mod):
    back = translate_single(hypothesis, back_tok, back_mod)
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    score = chrf.sentence_score(back, [source])
    return round(score.score, 2), back


def score_translation(source, hypothesis, direction, back_tok, back_mod):
    perplexity = compute_perplexity(hypothesis, direction)
    back_score, back_text = compute_back_translation(source, hypothesis, back_tok, back_mod)
    return {
        "perplexity": perplexity,
        "back_score": back_score,
        "back_text": back_text,
    }


def format_scores(scores):
    lines = []
    lines.append("Perplexity          : " + str(scores["perplexity"]) + "  (good: <100 | ok: 100-300 | poor: 300+)")
    lines.append("Back-translation    : " + str(scores["back_score"]) + " / 100  (good: 60+ | partial: 30-60 | poor: <30)")
    lines.append("Back-translated     : " + scores["back_text"])
    NL = chr(10)
    return NL.join(lines)


# === TRANSLATION FUNCTIONS ===
def translate_single(text, tok, mod):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = mod.generate(**inputs)
    return tok.decode(outputs[0], skip_special_tokens=True)


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


# === SCORING FUNCTIONS ===
def compute_perplexity(text, direction):
    config = MODELS[direction]
    tok, mod = get_perplexity_model(config["perplexity_model"])
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = mod(**inputs, labels=inputs["input_ids"]).loss
    return round(math.exp(loss.item()), 2)


def compute_back_translation(source, hypothesis, back_tok, back_mod):
    back = translate_single(hypothesis, back_tok, back_mod)
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    score = chrf.sentence_score(back, [source])
    return round(score.score, 2), back


def score_translation(source, hypothesis, direction, back_tok, back_mod):
    perplexity = compute_perplexity(hypothesis, direction)
    back_score, back_text = compute_back_translation(source, hypothesis, back_tok, back_mod)
    return {
        "perplexity": perplexity,
        "back_score": back_score,
        "back_text": back_text,
    }


def format_scores(scores):
    lines = []
    lines.append("Perplexity          : " + str(scores["perplexity"]) + "  (good: <100 | ok: 100-300 | poor: 300+)")
    lines.append("Back-translation    : " + str(scores["back_score"]) + " / 100  (good: 60+ | partial: 30-60 | poor: <30)")
    lines.append("Back-translated     : " + scores["back_text"])
    NL = chr(10)
    return NL.join(lines)


# === TRANSLATION FUNCTIONS ===
def translate_single(text, tok, mod):
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = mod.generate(**inputs)
    return tok.decode(outputs[0], skip_special_tokens=True)


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


# === SCORING FUNCTIONS ===
def compute_perplexity(text, direction):
    config = MODELS[direction]
    tok, mod = get_perplexity_model(config["perplexity_model"])
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = mod(**inputs, labels=inputs["input_ids"]).loss
    return round(math.exp(loss.item()), 2)


def compute_back_translation(source, hypothesis, back_tok, back_mod):
    back = translate_single(hypothesis, back_tok, back_mod)
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    score = chrf.sentence_score(back, [source])
    return round(score.score, 2), back


def score_translation(source, hypothesis, direction, back_tok, back_mod):
    perplexity = compute_perplexity(hypothesis, direction)
    back_score, back_text = compute_back_translation(source, hypothesis, back_tok, back_mod)
    return {
        "perplexity": perplexity,
        "back_score": back_score,
        "back_text": back_text,
    }


def format_scores(scores):
    lines = []
    lines.append("Perplexity          : " + str(scores["perplexity"]) + "  (good: <100 | ok: 100-300 | poor: 300+)")
    lines.append("Back-translation    : " + str(scores["back_score"]) + " / 100  (good: 60+ | partial: 30-60 | poor: <30)")
    lines.append("Back-translated     : " + scores["back_text"])
    NL = chr(10)
    return NL.join(lines)


# === WORD ALTERNATIVES ===
def get_word_alternatives(sentence, word_index, direction, top_k=5):
    config = MODELS[direction]
    target_lang = config["target_lang"]
    tok, mod = get_mask_model(config["mask_model"])

    words = sentence.split()
    if word_index < 0 or word_index >= len(words):
        return []

    original_word = words[word_index]

    # Create masked sentence
    masked_words = words.copy()
    masked_words[word_index] = tok.mask_token
    masked_sentence = " ".join(masked_words)

    inputs = tok(masked_sentence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mod(**inputs)

    logits = outputs.logits
    # Find position of mask token in input ids
    input_ids = inputs["input_ids"][0]
    mask_token_id = tok.mask_token_id
    mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

    if len(mask_positions) == 0:
        return []

    mask_pos = mask_positions[0].item()
    mask_logits = logits[0, mask_pos]
    probs = torch.softmax(mask_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k * 3)

    alternatives = []
    for prob, idx in zip(top_probs, top_indices):
        token = tok.decode([idx.item()]).strip()
        # Skip subwords, empty, punctuation, same word
        if not token or token.startswith("##") or token.startswith("_"):
            continue
        if token.lower() == original_word.lower():
            continue
        if not token.isalpha():
            continue
        alternatives.append({
            "word": token,
            "confidence": round(prob.item() * 100, 1),
        })
        if len(alternatives) >= top_k:
            break

    return alternatives


def replace_word_and_adjust(sentence, word_index, new_word, source_text, direction):
    words = sentence.split()
    if word_index < 0 or word_index >= len(words):
        return sentence

    # Simple replacement first
    words[word_index] = new_word
    simple_result = " ".join(words)

    # Try constrained retranslation for more natural result
    try:
        config = MODELS[direction]
        tok, mod = get_translation_model(config["translation"])
        back_tok, back_mod = get_translation_model(config["back_translation"])

        inputs = tok(source_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Generate multiple candidates
        outputs = mod.generate(
            **inputs,
            num_beams=15,
            num_return_sequences=10,
            early_stopping=True,
            length_penalty=1.0,
        )

        # Find best candidate containing the new word
        best_candidate = None
        best_score = -1

        for i in range(len(outputs)):
            candidate = tok.decode(outputs[i], skip_special_tokens=True)
            if new_word.lower() in candidate.lower():
                # Score by back-translation similarity
                back = translate_single(candidate, back_tok, back_mod)
                from sacrebleu.metrics import CHRF
                chrf = CHRF()
                score = chrf.sentence_score(back, [source_text]).score
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

        if best_candidate and best_score > 30:
            return best_candidate
        else:
            return simple_result

    except Exception:
        return simple_result


def format_alternatives(sentence, word_index, alternatives):
    words = sentence.split()
    if word_index < 0 or word_index >= len(words):
        return "Invalid word index"

    original = words[word_index]
    lines = []
    lines.append("Original word: " + original)
    lines.append("")
    lines.append("Alternatives:")
    for i, alt in enumerate(alternatives):
        num = str(i + 1)
        lines.append("  " + num + ". " + alt["word"] + " (" + str(alt["confidence"]) + "% confidence)")
    if not alternatives:
        lines.append("  No alternatives found for this word.")
    NL = chr(10)
    return NL.join(lines)


# === SYNTAX ANALYSIS ===
def get_nlp_for_lang(lang):
    if lang == "de":
        return nlp_de
    else:
        return nlp_en


def analyze_syntax(sentence, direction):
    config = MODELS[direction]
    target_lang = config["target_lang"]
    nlp = get_nlp_for_lang(target_lang)
    doc = nlp(sentence)

    rows = []
    for token in doc:
        morph = str(token.morph) if token.morph else ""
        rows.append({
            "Word": token.text,
            "Lemma": token.lemma_,
            "POS": token.pos_,
            "Tag": token.tag_,
            "Dependency": token.dep_,
            "Head": token.head.text,
            "Morphology": morph,
        })
    return rows


def format_syntax_table(rows):
    if not rows:
        return "No analysis available."

    headers = ["Word", "Lemma", "POS", "Tag", "Dependency", "Head", "Morphology"]
    col_widths = {}
    for h in headers:
        col_widths[h] = len(h)
    for row in rows:
        for h in headers:
            val = str(row[h])
            if len(val) > col_widths[h]:
                col_widths[h] = len(val)

    # Build header line
    header_parts = []
    separator_parts = []
    for h in headers:
        header_parts.append(h.ljust(col_widths[h]))
        separator_parts.append("-" * col_widths[h])

    NL = chr(10)
    lines = []
    lines.append(" | ".join(header_parts))
    lines.append("-+-".join(separator_parts))

    for row in rows:
        row_parts = []
        for h in headers:
            row_parts.append(str(row[h]).ljust(col_widths[h]))
        lines.append(" | ".join(row_parts))

    return NL.join(lines)


def format_morphology_detail(sentence, direction):
    config = MODELS[direction]
    target_lang = config["target_lang"]
    nlp = get_nlp_for_lang(target_lang)
    doc = nlp(sentence)

    NL = chr(10)
    lines = []
    for token in doc:
        if token.pos_ in ("PUNCT", "SPACE"):
            continue
        morph_dict = token.morph.to_dict()
        if morph_dict:
            lines.append(token.text + " (" + token.pos_ + "):")
            for key, value in morph_dict.items():
                lines.append("    " + key + ": " + value)
        else:
            lines.append(token.text + " (" + token.pos_ + "): no morphology data")
        lines.append("")

    if not lines:
        return "No morphological data available."
    return NL.join(lines)




# === ERROR CLASSIFICATION ===
def classify_error(scores, qa_result):
    has_qa_miss = len(qa_result["glossary_coverage"]["misses"]) > 0
    perplexity = scores["perplexity"]
    back_score = scores["back_score"]

    high_perplexity = perplexity is not None and perplexity > 200
    low_back = back_score is not None and back_score < 40

    issues = sum([has_qa_miss, high_perplexity, low_back])
    if issues == 0:
        return "GOOD"
    elif issues > 1:
        return "REVIEW"
    elif has_qa_miss:
        return "TERMINOLOGY"
    elif high_perplexity:
        return "FLUENCY"
    else:
        return "SEMANTIC"


def empty_qa():
    return {
        "glossary_coverage": {"hits": [], "misses": []},
        "capitalisation": {"count": 0},
        "separable_verbs": {"count": 0},
    }


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


# === MAIN PIPELINE ===
def run_pipeline(source_text, direction):
    if not source_text.strip():
        return "", "", "", ""
    config = MODELS[direction]
    trans_tok, trans_mod = get_translation_model(config["translation"])
    back_tok, back_mod = get_translation_model(config["back_translation"])

    translations = translate_multiple(source_text, trans_tok, trans_mod, num_options=3)
    all_options = []
    best_translation = ""
    best_score = -1

    for i, hyp in enumerate(translations):
        scores = score_translation(source_text, hyp, direction, back_tok, back_mod)
        if direction == "DE to EN":
            qa_result = run_qa(source_text, hyp, glossary, nlp_de, verbose=False)
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
        qa_result = run_qa(source_text, best_translation, glossary, nlp_de, verbose=False)
        qa_text = format_glossary_results(qa_result)
    else:
        qa_text = "Glossary QA currently available for DE to EN only."

    return best_translation, options_text, qa_text, direction


def rescore_edited(source_text, edited_translation, direction):
    if not source_text.strip() or not edited_translation.strip():
        return "", ""
    config = MODELS[direction]
    back_tok, back_mod = get_translation_model(config["back_translation"])
    scores = score_translation(source_text, edited_translation, direction, back_tok, back_mod)
    if direction == "DE to EN":
        qa_result = run_qa(source_text, edited_translation, glossary, nlp_de, verbose=False)
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


# === WORD CLICK HANDLER ===
def handle_word_click(sentence, word_index_str, direction):
    if not sentence.strip() or not word_index_str.strip():
        return "", ""
    try:
        word_index = int(word_index_str)
    except ValueError:
        return "Please enter a valid number.", ""

    alternatives = get_word_alternatives(sentence, word_index, direction)
    alt_text = format_alternatives(sentence, word_index, alternatives)
    return alt_text, ""


def handle_word_replace(sentence, word_index_str, new_word, source_text, direction):
    if not sentence.strip() or not word_index_str.strip() or not new_word.strip():
        return ""
    try:
        word_index = int(word_index_str)
    except ValueError:
        return sentence
    result = replace_word_and_adjust(sentence, word_index, new_word, source_text, direction)
    return result


def handle_syntax_analysis(sentence, direction):
    if not sentence.strip():
        return "", ""
    rows = analyze_syntax(sentence, direction)
    table = format_syntax_table(rows)
    morphology = format_morphology_detail(sentence, direction)
    return table, morphology


def number_words(sentence):
    if not sentence.strip():
        return ""
    words = sentence.split()
    NL = chr(10)
    lines = []
    for i, word in enumerate(words):
        lines.append(str(i) + ": " + word)
    return NL.join(lines)


# === SCORE GUIDE ===
SCORE_GUIDE = """| Metric | Range | Good | Moderate | Poor |
|--------|-------|------|----------|------|
| Perplexity | 0-1000+ | below 100 | 100-300 | above 300 |
| Back-translation | 0-100 | 60+ | 30-60 | below 30 |
"""

ERROR_KEY = """| Category | Meaning |
|----------|----------|
| GOOD | No issues detected |
| FLUENCY | Surface form or grammar issues |
| SEMANTIC | Meaning has drifted from source |
| TERMINOLOGY | Glossary term missing from translation |
| REVIEW | Multiple issues - recommend human review |
"""

# === GRADIO UI ===
with gr.Blocks(title="DE EN Translation QA") as demo:
    gr.Markdown("# German - English Translation QA Tool")
    gr.Markdown(
        "Translate between German and English with quality scoring, "
        "word-level alternatives, and syntactic analysis."
    )

    # Store direction as state
    direction_state = gr.State("DE to EN")

    with gr.Row():
        direction = gr.Dropdown(
            choices=["DE to EN", "EN to DE"],
            value="DE to EN",
            label="Translation Direction",
            scale=1,
        )

    # === SECTION 1: TRANSLATE ===
    gr.Markdown("## 1. Translate")
    with gr.Row():
        with gr.Column():
            source_input = gr.Textbox(
                label="Source Text",
                placeholder="Enter German or English text here...",
                lines=4,
            )
            translate_btn = gr.Button("Translate and Score", variant="primary", size="lg")

        with gr.Column():
            best_output = gr.Textbox(label="Best Translation", lines=4)
            options_output = gr.Textbox(label="All Translation Options with Scores", lines=20)
            qa_output = gr.Textbox(label="Glossary QA Results", lines=6)

    translate_btn.click(
        fn=run_pipeline,
        inputs=[source_input, direction],
        outputs=[best_output, options_output, qa_output, direction_state],
    )

    # === SECTION 2: EDIT AND RESCORE ===
    gr.Markdown("## 2. Edit and Rescore")
    gr.Markdown("Copy a translation above, edit it, and see how scores change.")

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
        inputs=[source_input, edited_input, direction],
        outputs=[rescore_output, rescore_qa_output],
    )

    # === SECTION 3: WORD ALTERNATIVES ===
    gr.Markdown("## 3. Word Alternatives")
    gr.Markdown("See numbered words in your translation, pick a word by number, and explore alternatives.")

    with gr.Row():
        with gr.Column():
            word_sentence_input = gr.Textbox(
                label="Translation to Explore",
                placeholder="Paste a translation here...",
                lines=2,
            )
            show_words_btn = gr.Button("Show Numbered Words")
            numbered_words_output = gr.Textbox(label="Numbered Words", lines=6)

            word_index_input = gr.Textbox(
                label="Word Number",
                placeholder="Enter the number of the word to explore...",
                lines=1,
            )
            find_alts_btn = gr.Button("Find Alternatives", variant="primary")

        with gr.Column():
            alternatives_output = gr.Textbox(label="Word Alternatives", lines=10)

            new_word_input = gr.Textbox(
                label="Replacement Word",
                placeholder="Type or paste an alternative word...",
                lines=1,
            )
            replace_btn = gr.Button("Replace and Adjust Sentence", variant="secondary")
            adjusted_output = gr.Textbox(label="Adjusted Translation", lines=4)

    show_words_btn.click(
        fn=number_words,
        inputs=[word_sentence_input],
        outputs=[numbered_words_output],
    )

    find_alts_btn.click(
        fn=handle_word_click,
        inputs=[word_sentence_input, word_index_input, direction],
        outputs=[alternatives_output, adjusted_output],
    )

    replace_btn.click(
        fn=handle_word_replace,
        inputs=[word_sentence_input, word_index_input, new_word_input, source_input, direction],
        outputs=[adjusted_output],
    )

    # === SECTION 4: SYNTAX ANALYSIS ===
    gr.Markdown("## 4. Syntax Analysis")
    gr.Markdown("Analyze the syntactic structure, POS tags, and morphology of any translation.")

    with gr.Row():
        with gr.Column():
            syntax_input = gr.Textbox(
                label="Sentence to Analyze",
                placeholder="Paste a translation here...",
                lines=2,
            )
            analyze_btn = gr.Button("Analyze Syntax", variant="primary")

        with gr.Column():
            syntax_output = gr.Textbox(label="Syntax Table (POS, Dependencies)", lines=12)
            morph_output = gr.Textbox(label="Morphological Detail", lines=12)

    analyze_btn.click(
        fn=handle_syntax_analysis,
        inputs=[syntax_input, direction],
        outputs=[syntax_output, morph_output],
    )

    # === GUIDES ===
    with gr.Accordion("Score Guide", open=False):
        gr.Markdown(SCORE_GUIDE)
    with gr.Accordion("Error Category Key", open=False):
        gr.Markdown(ERROR_KEY)

    # === EXAMPLES ===
    gr.Markdown("## Example Sentences")
    gr.Examples(
        examples=[
            ["Die Sicherheit des Systems wurde ueberprueft.", "DE to EN"],
            ["Langer Atem macht sich bezahlt.", "DE to EN"],
            ["Die Benutzeroberflaeche ist sehr intuitiv.", "DE to EN"],
            ["Das Passwort muss mindestens acht Zeichen lang sein.", "DE to EN"],
            ["The weather forecast predicts rain tomorrow.", "EN to DE"],
            ["Please restart the application after updating.", "EN to DE"],
        ],
        inputs=[source_input, direction],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
