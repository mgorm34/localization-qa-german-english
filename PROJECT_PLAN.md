# Project Plan: Localization QA German–English

## Timeline: March 2 – March 9

## Day 1 (Mar 2) – Setup and Scope
- [x] Create repository skeleton
- [x] Define data sources and licenses
- [x] Set up environment and folder structure
- [x] Write README with reproduction steps
- [x] Decide on tech stack

## Tech Stack
- Python 3.10
- HuggingFace Transformers (Helsinki-NLP/opus-mt-de-en)
- sacreBLEU for BLEU score evaluation
- BERTScore for semantic evaluation
- Gradio for interactive UI demo
- Optional: Docker for reproducibility

## Goals
- Evaluate German–English MT quality on open datasets
- Identify common translation errors and failure modes
- Build an interactive demo for qualitative evaluation
