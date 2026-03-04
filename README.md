---
title: Localization QA
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# German-English Translation QA

An end-to-end machine translation quality assurance pipeline.

## Features
- German to English translation using Helsinki-NLP opus-mt
- Reference-free scoring (perplexity + back-translation)
- Reference-based scoring (chrF + BERTScore)
- POS-aware glossary QA using spaCy
- Automatic error classification

## How to Use
1. Enter a German sentence
2. Optionally add a reference translation
3. Click Translate and Analyse
4. Review scores and QA results
