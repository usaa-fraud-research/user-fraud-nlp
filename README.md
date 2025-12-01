🛡️ CFPB Fraud Intelligence Dashboard

Fraud Detection • Semantic Search • Streamlit Dashboard • Supabase Vector DB

A full NLP pipeline that scrapes CFPB articles, detects fraud patterns, generates embeddings, performs semantic search, runs ML fraud classification, and visualizes everything inside a professional Streamlit dashboard.

🚀 Quick Start

1️⃣ Install Dependencies
    uv sync

2️⃣ Create a .env File
    SUPABASE_URL=your-url
    SUPABASE_KEY=your-key
    OPENAI_API_KEY=your-openai-key

🏗️ System Architecture
   flowchart TD
    A[CFPB Scraper] --> B[Supabase: cfpb_articles]
    B --> C[OpenAI Embeddings\n1536-dim]
    C --> D[pgvector Similarity Search]
    B --> E[SVM ]
    E --> F[Fraud Predictions]
    D --> G[Streamlit Dashboard]
    F --> G

🖥️ Screenshots & UI (Placeholders — Replace in repo)

Dashboard Home
Semantic Search Page
ML Alerts

user-fraud-nlp/
│
├── cfpb_articles.py        # Scraper for CFPB Newsroom/Blog/Enforcement
├── articles_supabase.py    # Upload scraped data → Supabase
├── llm_embedding.py        # Generate + store embeddings
├── semantic_search.py      # Cached semantic search using pgvector
├── ml_train.py             # Train ML fraud classifier (LogReg/SVM)
├── fraud_dashboard.py      # Streamlit dashboard
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── fraud_type_svm.joblib
│
├── data/                   # (ignored) scraped CSV files
├── txt/                    # Summaries and text dumps
│
├── .env.example
├── .gitignore
├── pyproject.toml          # Dependencies for uv
└── uv.lock

This project automates the entire pipeline from raw CFPB articles → fraud insights, combining scraping, NLP, vector search, and machine learning.

Example Raw Article Snippet: The Bureau filed a complaint alleging unauthorized transfers via Zelle...

Example Transformed Output:
{
  "fraud_type": "reg_e",
  "fraud_tags": ["unauthorized_transfer", "zelle_fraud"],
  "summary": "Unauthorized account withdrawals via payment app."
}

Embedding Example (shortened):

embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
).data[0].embedding

ML Prediction Example:
pred = model.predict([embedding])[0]
# → "identity_theft"

🔎 Full Pipeline Overview

1️⃣ CFPB Scraper

Scrapes from:
	•	CFPB Newsroom
	•	CFPB Blog
	•	CFPB Enforcement Actions

Extracts:
title, date, url, text, source.

⸻

2️⃣ Fraud Detection (Regex Rule-Based)

Fraud types identified include:
Category         example

Identity Theft:  account takeover, stolen info

Payment App Fraud: Zelle/ACH unauthorized transfers
Card Fraud: debit/credit disputes
Loan/Investment Scams: payday, student loan, mortgage
Crypto Fraud: crypto exchanges, transfers
Romance / Social Scams: impersonation, fake profiles
UDAP: deceptive abusive practices

Outputs: fraud_type, fraud_tags, summary.

3️⃣ Supabase Storage

Tables:
	•	cfpb_articles — all article metadata + embeddings + ML predictions
	•	search_queries — cached query embeddings to save OpenAI cost

⸻

4️⃣ Embedding Pipeline
	•	Model: text-embedding-3-small
	•	1536-dimensional vector
	•	Stored directly in Supabase (pgvector column)

⸻

5️⃣ Semantic Search Engine

Tools:
	•	Cached query embeddings
	•	match_cfpb_articles RPC
	•	pgvector cosine similarity
	•	Filters:
	•	by year
	•	by keyword
	•	min similarity threshold

6️⃣ Streamlit Dashboard

Tabs:
Tab                 Contents
Week 2: Scraper:    Browse raw scraped articles
Week 3: Fraud Detection:  Keyword charts, word cloud, tag frequencies
Week 4: Analysis:    Trends, bar charts, ML analytics
Semantic Search:     Preset scenarios + custom query search
ML Alerts:        High-risk fraud notifications

📊 Findings & Why This Project Matters

✔️ 1. Fraud patterns become visible

Charts and word clouds reveal dominant fraud categories (Zelle, identity theft, UDAP).

✔️ 2. Semantic search finds similar cases even with different wording

Example:

“unauthorized zelle transfer”

Matches:
	•	ACH errors
	•	account takeover
	•	unauthorized withdrawals

✔️ 3. ML classifier identifies fraud types automatically

Even if CFPB didn’t tag it.

✔️ 4. High-Priority Alerts

Shows articles involving:
	•	Regulation E
	•	Crypto fraud
	•	Wire transfer fraud
	•	Identity theft

These surface instantly in the dashboard.

✔️ 5. Supabase + Streamlit = Real production workflow

This demonstrates real-world:
	•	ETL pipeline
	•	Vector database setup
	•	LLM embeddings
	•	ML training
	•	UI front-end

⸻

🧾 .gitignore
.env
data/
__pycache__/
*.pyc
.venv/
uv.lock



⸻

🎉 Summary

This project demonstrates end-to-end NLP + ML engineering, including scraping, embedding, vector search, classification, and dashboarding — all using modern tools (Supabase, OpenAI, Streamlit, uv).
