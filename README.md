---
title: Localization QA German English
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
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
