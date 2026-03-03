# TF-IDF Competitor Analysis App

A Streamlit app that scrapes and compares TF-IDF keyword scores across your URL and up to 10 competitor URLs.

## Features
- Bulk URL input (paste one per line)
- Scrapes body text, headings (H1–H6), meta title & description
- Unigram, bigram and trigram analysis
- Auto language detection with English translation for non-English terms
- Keyword gap analysis vs competitor aggregate
- CSV export

## Stack
- Python 3.11+
- Streamlit
- scikit-learn, NLTK, pandas, BeautifulSoup4, Playwright, deep-translator

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
