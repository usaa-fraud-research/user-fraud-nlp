Fraud Detection • Semantic Search • Streamlit Dashboard • Supabase Vector DB

This project analyzes CFPB (Consumer Financial Protection Bureau) articles to detect fraud patterns, compute embeddings, and power a semantic search engine using Supabase pgvector + OpenAI.
A Streamlit dashboard displays the full pipeline.

⸻

📁 Project Structure
user-fraud-nlp/
│
├── cfpb_articles.py          # Scraper for CFPB newsroom/blog/enforcement
├── articles_supabase.py      # Upload scraped data to Supabase
├── llm_embedding.py          # Generate & store embeddings in Supabase
├── semantic_search.py        # Cached semantic search using pgvector
├── fraud_dashboard.py        # Streamlit app (Scraper • Fraud Detection • Analysis)
│
├── data/                     # (ignored) scraped CSVs
├── .env                      # (ignored) secrets
├── .gitignore                # ignore env, data, pycache, venv
├── pyproject.toml            # dependencies (uv)
└── uv.lock

🚀 Pipeline Overview

1️⃣ CFPB Scraper

Scrapes articles from:
	•	CFPB Newsroom
	•	CFPB Blog
	•	Enforcement Actions

Extracts:
title, date, url, text, source

2️⃣ Fraud Detection

Each article is tagged using regex-based patterns:

Examples:
	•	identity theft
	•	account takeover
	•	card fraud
	•	wire/Zelle fraud
	•	phishing / smishing / vishing
	•	crypto fraud
	•	romance scams
	•	loan/investment/insurance/job scams

Outputs:
	•	fraud_type
	•	fraud_tags
	•	summary

⸻

3️⃣ Supabase Storage

Tables used:
	•	cfpb_articles (fraud metadata + embeddings)
	•	search_queries (cached query embeddings)

⸻

4️⃣ Embeddings Pipeline

Uses OpenAI’s text-embedding-3-small (1536-dim) to embed article text.

5️⃣ Semantic Search Engine

Built with:
	•	Query embedding (cached)
	•	Supabase RPC + pgvector
	•	Cosine similarity
	•	Optional filters: year, keyword, threshold

6️⃣ Streamlit Dashboard

All results visualized in fraud_dashboard.py:

Tabs:
	•	Week 2: Scraper viewer
	•	Week 3: Fraud keywords, charts, word cloud
	•	Week 4: Trend analysis + Semantic Search

🔐 Environment Setup

Create .env:
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
OPENAI_API_KEY=your_key

🧹 .gitignore
.env
data/
__pycache__/
*.pyc
.venv/