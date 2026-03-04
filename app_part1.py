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
    first = outputs<a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[0]</a>
    return tok.decode(first, skip_special_tokens=True)