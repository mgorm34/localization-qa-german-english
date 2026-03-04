import sys
import math
import torch
import gradio as gr
from sacrebleu.metrics import CHRF
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
print("Loading spaCy DE...")
nlp_de = load_nlp()

print("Loading spaCy EN...")
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


# === TRANSLATION ===
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


# === SCORING ===
def compute_perplexity(text, direction):
    config = MODELS[direction]
    tok, mod = get_perplexity_model(config["perplexity_model"])
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = mod(**inputs, labels=inputs["input_ids"]).loss
    return round(math.exp(loss.item()), 2)


def compute_accuracy(source, hypothesis, back_tok, back_mod):
    back = translate_single(hypothesis, back_tok, back_mod)
    chrf = CHRF()
    score = chrf.sentence_score(back, [source])
    return round(score.score, 2)


def perplexity_label(score):
    if score < 100:
        return "Good"
    elif score < 300:
        return "Moderate"
    else:
        return "Poor"


def accuracy_label(score):
    if score >= 60:
        return "Good"
    elif score >= 30:
        return "Moderate"
    else:
        return "Poor"


# === WORD ALTERNATIVES ===
def get_word_alternatives(sentence, word_index, direction, top_k=5):
    config = MODELS[direction]
    tok, mod = get_mask_model(config["mask_model"])

    words = sentence.split()
    if word_index < 0 or word_index >= len(words):
        return []

    original_word = words[word_index]

    masked_words = words.copy()
    masked_words[word_index] = tok.mask_token
    masked_sentence = " ".join(masked_words)

    inputs = tok(masked_sentence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mod(**inputs)

    logits = outputs.logits
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

    words[word_index] = new_word
    simple_result = " ".join(words)

    try:
        config = MODELS[direction]
        tok, mod = get_translation_model(config["translation"])
        back_tok, back_mod = get_translation_model(config["back_translation"])

        inputs = tok(source_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = mod.generate(
            **inputs,
            num_beams=15,
            num_return_sequences=10,
            early_stopping=True,
            length_penalty=1.0,
        )

        best_candidate = None
        best_score = -1

        for i in range(len(outputs)):
            candidate = tok.decode(outputs[i], skip_special_tokens=True)
            if new_word.lower() in candidate.lower():
                back = translate_single(candidate, back_tok, back_mod)
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
        morph_dict = token.morph.to_dict()
        grammar_parts = []
        for key, value in morph_dict.items():
            grammar_parts.append(key + "=" + value)
        grammar = ", ".join(grammar_parts) if grammar_parts else "-"

        rows.append([
            token.text,
            token.lemma_,
            token.pos_,
            grammar,
        ])
    return rows


# === MAIN PIPELINE ===
def run_translate(source_text, direction):
    if not source_text.strip():
        return "", "", gr.update(visible=False), gr.update(visible=False)

    config = MODELS[direction]
    trans_tok, trans_mod = get_translation_model(config["translation"])
    back_tok, back_mod = get_translation_model(config["back_translation"])

    translation = translate_single(source_text, trans_tok, trans_mod)

    ppl = compute_perplexity(translation, direction)
    acc = compute_accuracy(source_text, translation, back_tok, back_mod)

    score_text = "Perplexity: " + str(ppl) + " (" + perplexity_label(ppl) + ")    |    Accuracy: " + str(acc) + "/100 (" + accuracy_label(acc) + ")"

    return translation, score_text, gr.update(visible=False), gr.update(visible=False)


# === WORD CLICK HANDLERS ===
def show_word_buttons(translation):
    if not translation.strip():
        return gr.update(visible=False)
    words = translation.split()
    button_labels = []
    for i, w in enumerate(words):
        button_labels.append(w)
    return gr.update(visible=True, choices=button_labels, value=None)


def on_word_selected(translation, selected_word, direction):
    if not translation.strip() or selected_word is None:
        return gr.update(choices=[], visible=False)

    words = translation.split()
    word_index = -1
    for i, w in enumerate(words):
        if w == selected_word:
            word_index = i
            break

    if word_index == -1:
        return gr.update(choices=[], visible=False)

    alternatives = get_word_alternatives(translation, word_index, direction)
    if not alternatives:
        return gr.update(choices=["No alternatives found"], visible=True)

    choices = []
    for alt in alternatives:
        choices.append(alt["word"] + " (" + str(alt["confidence"]) + "%)")
    return gr.update(choices=choices, visible=True)


def on_alternative_selected(translation, selected_word, selected_alt, source_text, direction):
    if not translation.strip() or not selected_word or not selected_alt:
        return translation, ""

    if selected_alt == "No alternatives found":
        return translation, ""

    new_word = selected_alt.split(" (")[0]

    words = translation.split()
    word_index = -1
    for i, w in enumerate(words):
        if w == selected_word:
            word_index = i
            break

    if word_index == -1:
        return translation, ""

    adjusted = replace_word_and_adjust(translation, word_index, new_word, source_text, direction)

    config = MODELS[direction]
    back_tok, back_mod = get_translation_model(config["back_translation"])
    ppl = compute_perplexity(adjusted, direction)
    acc = compute_accuracy(source_text, adjusted, back_tok, back_mod)
    score_text = "Perplexity: " + str(ppl) + " (" + perplexity_label(ppl) + ")    |    Accuracy: " + str(acc) + "/100 (" + accuracy_label(acc) + ")"

    return adjusted, score_text


def on_analyze_click(translation, direction):
    if not translation.strip():
        return gr.update(visible=False)

    rows = analyze_syntax(translation, direction)
    headers = ["Word", "Lemma", "POS", "Grammar"]
    return gr.update(value={"headers": headers, "data": rows}, visible=True)


# === GRADIO UI ===
with gr.Blocks(title="DE-EN Translation QA", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# German - English Translation QA")
    gr.Markdown("Translate, explore word alternatives, and analyze grammar.")

    with gr.Row():
        direction = gr.Dropdown(
            choices=["DE to EN", "EN to DE"],
            value="DE to EN",
            label="Direction",
            scale=1,
        )

    source_input = gr.Textbox(
        label="Source Text",
        placeholder="Enter text to translate...",
        lines=3,
    )

    translate_btn = gr.Button("Translate", variant="primary", size="lg")

    score_display = gr.Textbox(
        label="Quality Scores",
        interactive=False,
        lines=1,
    )

    with gr.Row():
        translation_output = gr.Textbox(
            label="Translation (click a word below to explore alternatives)",
            lines=3,
            interactive=True,
            scale=4,
        )
        analyze_btn = gr.Button("Analyze", variant="secondary", scale=1)

    # Word selection
    word_selector = gr.Radio(
        label="Click a word to see alternatives",
        choices=[],
        visible=False,
    )

    # Alternatives dropdown
    alt_selector = gr.Radio(
        label="Select an alternative",
        choices=[],
        visible=False,
    )

    # Syntax table
    syntax_table = gr.Dataframe(
        headers=["Word", "Lemma", "POS", "Grammar"],
        label="Syntax Analysis",
        visible=False,
        interactive=False,
    )

    # === EVENT HANDLERS ===
    translate_btn.click(
        fn=run_translate,
        inputs=[source_input, direction],
        outputs=[translation_output, score_display, word_selector, syntax_table],
    ).then(
        fn=show_word_buttons,
        inputs=[translation_output],
        outputs=[word_selector],
    )

    word_selector.change(
        fn=on_word_selected,
        inputs=[translation_output, word_selector, direction],
        outputs=[alt_selector],
    )

    alt_selector.change(
        fn=on_alternative_selected,
        inputs=[translation_output, word_selector, alt_selector, source_input, direction],
        outputs=[translation_output, score_display],
    )

    analyze_btn.click(
        fn=on_analyze_click,
        inputs=[translation_output, direction],
        outputs=[syntax_table],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
