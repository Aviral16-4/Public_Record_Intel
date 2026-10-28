
# AI-Based Public Record Intelligence Pipeline

A Python-based intelligence pipeline that aggregates publicly available news reports and analyzes reported legal incidents using NLP and fuzzy name matching.

## Features
- Google News RSS data collection
- NLP-based entity recognition
- Fuzzy name matching and identity confidence scoring
- Classification of reported incidents
- Severity scoring and financial amount extraction
- Structured JSON report generation
- Low-confidence results flagged for manual review
- Gotham-inspired dark theme for the project interface

## Tech Stack
Python, spaCy, RapidFuzz, Feedparser, Requests, Trafilatura

## Installation
```bash
py -m pip install feedparser requests trafilatura spacy rapidfuzz python-dateutil
py -m spacy download en_core_web_sm
```

## Disclaimer
This project analyzes reported mentions from public sources. Results are not verified criminal records, and identity matches and allegations must be independently verified against reliable official sources.
