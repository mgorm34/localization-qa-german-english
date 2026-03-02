"""
qa_glossary.py
Glossary-driven QA module for German to English translation.
Uses spaCy POS tagging to avoid false matches on inflected forms.
"""

import re
import json
import csv
import spacy
from pathlib import Path

GLOSSARY_PATH = Path("data/glossary.json")

TERM_POS = {
    "NOUN": [
        "Anwendung", "Benutzeroberfläche", "Fehlermeldung",
        "Datei", "Ordner", "Passwort", "Netzwerk", "Drucker",
        "Tastatur", "Bildschirm", "Speicher", "Betriebssystem",
        "Schnittstelle", "Datenbank", "Verschlüsselung",
        "Aktualisierung", "Systemsteuerung", "Aufgabenmanager",
        "Zwischenablage", "Benutzerkontosteuerung", "Zugriffsrechte",
        "Sicherheit", "Einstellung"
    ],
    "VERB": [
        "herunterladen", "hochladen", "aufrufen",
        "einloggen", "ausloggen", "anmelden", "abmelden"
    ]
}

def load_glossary(path=GLOSSARY_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_nlp():
    print("Loading spaCy German model...")
    nlp = spacy.load("de_core_news_sm")
    print("spaCy model loaded.")
    return nlp

def get_expected_pos(de_term):
    for pos, terms in TERM_POS.items():
        if de_term in terms:
            return pos
    return "NOUN"

def smart_term_match(de_term, source_doc):
    expected_pos = get_expected_pos(de_term)
    de_term_lower = de_term.lower()
    for token in source_doc:
        if token.text.lower() == de_term_lower:
            if token.pos_ == expected_pos:
                return True
            else:
                return False
    return False

def check_glossary_coverage(source_doc, hypothesis, glossary):
    hits = []
    misses = []
    for de_term, en_gloss in glossary.items():
        if smart_term_match(de_term, source_doc):
            valid_translations = [t.strip() for t in en_gloss.split("|")]
            matched = any(t.lower() in hypothesis.lower() for t in valid_translations)
            if matched:
                hits.append({"de_term": de_term, "en_gloss": en_gloss, "status": "found"})
            else:
                misses.append({"de_term": de_term, "en_gloss": en_gloss, "status": "missing"})
    total = len(hits) + len(misses)
    coverage = round(len(hits) / total * 100, 1) if total > 0 else None
    return {"hits": hits, "misses": misses, "coverage": coverage, "total_terms_found": total}

def check_capitalisation(source_doc):
    capitalised = []
    for token in source_doc:
        if token.pos_ == "NOUN" and token.text[:1].isupper():
            capitalised.append(token.text)
    return {
        "capitalised_nouns": capitalised,
        "count": len(capitalised),
        "note": "German nouns are always capitalised"
    }

def check_separable_verbs(source_doc, glossary):
    prefixes = ["herunter","hoch","auf","ein","aus","an","ab","zu","vor","nach"]
    detected = []
    verb_terms = {
        de_term: en_gloss
        for de_term, en_gloss in glossary.items()
        if get_expected_pos(de_term) == "VERB"
    }
    for token in source_doc:
        token_lower = token.text.lower()
        if token.pos_ in ["VERB", "NOUN"]:
            for de_term, en_gloss in verb_terms.items():
                if token_lower == de_term.lower():
                    detected.append({"verb": de_term, "gloss": en_gloss, "pos_found": token.pos_})
    return {"separable_verbs_detected": detected, "count": len(detected)}

def run_qa(source, hypothesis, glossary, nlp, verbose=True):
    source_doc = nlp(source)
    coverage       = check_glossary_coverage(source_doc, hypothesis, glossary)
    capitalisation = check_capitalisation(source_doc)
    separable      = check_separable_verbs(source_doc, glossary)
    results = {
        "source": source,
        "hypothesis": hypothesis,
        "glossary_coverage": coverage,
        "capitalisation": capitalisation,
        "separable_verbs": separable
    }
    if verbose:
        print(f"DE  : {source}")
        print(f"HYP : {hypothesis}")
        print(f"Coverage: {coverage['coverage']}% ({len(coverage['hits'])} hits, {len(coverage['misses'])} misses)")
        if coverage["misses"]:
            print("Misses:")
            for m in coverage["misses"]:
                print(f"  MISS: {m['de_term']} -> expected {m['en_gloss']}")
        if separable["separable_verbs_detected"]:
            print(f"Separable verbs: {separable['count']}")
        print()
    return results

def run_qa_batch(sources, hypotheses, glossary, nlp, verbose=False):
    results = []
    flagged = 0
    for source, hypothesis in zip(sources, hypotheses):
        result = run_qa(source, hypothesis, glossary, nlp, verbose=verbose)
        results.append(result)
        if result["glossary_coverage"]["misses"]:
            flagged += 1
    print("\n--- Batch QA Summary ---")
    print(f"Total sentences : {len(results)}")
    print(f"Flagged (misses): {flagged}")
    print(f"Clean           : {len(results) - flagged}")
    return results

if __name__ == "__main__":
    glossary = load_glossary()
    print(f"Loaded glossary with {len(glossary)} terms")
    nlp = load_nlp()
    sources, hypotheses = [], []
    with open("data/processed/translations.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row["de"])
            hypotheses.append(row["hyp_en"])
    print(f"Loaded {len(sources)} translation pairs")
    first_source = next(iter(sources))
    first_hyp = next(iter(hypotheses))
    print("--- Single Sentence QA Example ---")
    run_qa(first_source, first_hyp, glossary, nlp, verbose=True)
    print("--- Running Batch QA ---")
    run_qa_batch(sources, hypotheses, glossary, nlp, verbose=False)