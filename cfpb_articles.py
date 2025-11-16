import os
import re
import time
import argparse
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from readability import Document
from dateutil import parser as dtparse
from tqdm import tqdm
from urllib.parse import urlparse
from datetime import datetime, timedelta

# =====================================================================
# 1. Enhanced Fraud + Regulatory Violation Tagging
# =====================================================================

FRAUD_PATTERNS = {
    # --- True consumer frauds ---
    "identity_theft": r"\b(identity theft|stolen identity|synthetic identity)\b",
    "account_takeover": r"\b(account takeover|ATO|SIM swap|credential stuffing)\b",
    "check_fraud": r"\b(check fraud|fake check|counterfeit check|check kiting)\b",
    "card_fraud": r"\b(credit|debit) card (fraud|skimming|cloning|chargeback)\b",
    "wire_transfer": r"\b(wire fraud|unauthorized wire|zelle|money transfer fraud)\b",
    "phishing": r"\b(phish|phishing|smishing|vishing)\b",
    "crypto": r"\b(crypto|bitcoin|ethereum).*(fraud|scam|rug)\b",
    "romance": r"\b(romance scam|catfish)\b",
    "pig_butchering": r"\bpig[- ]?butcher(ing)?\b",
    "money_laundering": r"\b(money laundering|laundered|AML)\b",

    # --- New CFPB-specific regulatory violations ---
    "udap": r"\b(unfair|deceptive|abusive)\b",
    "reg_e": r"\b(Reg(ulation)? E|Electronic Fund Transfer Act|EFTA|unauthorized transfer|error resolution)\b",
    "reg_z": r"\b(Reg(ulation)? Z|Truth in Lending Act|billing error)\b",
    "fcra": r"\b(FCRA|credit reporting|reinvestigate|consumer report|inaccurate information)\b",
    "debt_collection": r"\b(debt collection|collector|FDCPA|collection lawsuit)\b",
    "loan_servicing": r"\b(servicing|loan modification|forbearance|deferral)\b",
    "mortgage_misconduct": r"\b(mortgage|refinance|APR|closing disclosure|loan estimate)\b",
    "auto_lending": r"\b(auto loan|vehicle financing|dealer markup|indirect auto)\b",
    "student_loan": r"\b(student loan|servicer|FAFSA|borrower defense)\b",
    "remittance": r"\b(remittance|exchange rate|transfer fee|prepaid card)\b",
    "credit_furnishing": r"\b(furnish|furnisher|credit bureau|data furnishing)\b",
    "payments_app_failure": r"\b(Cash App|peer-to-peer|P2P|dispute investigation)\b",
    "privacy_data_abuse": r"\b(data (harvesting|sharing|breach)|privacy violation)\b",

    # Fallback
    "generic": r"\b(fraud|scam|scheme|deceptive)\b",
}

FRAUD_REGEX = {k: re.compile(v, re.I) for k, v in FRAUD_PATTERNS.items()}


def classify_violation(hits):
    """
    Prioritized classification including CFPB regulatory domains.
    """
    priority = [
        # Highest – true fraud categories
        "identity_theft", "account_takeover", "card_fraud", "check_fraud",
        "wire_transfer", "phishing", "crypto", "pig_butchering", "romance",
        "money_laundering",

        # Next – CFPB regulatory categories
        "fcra", "reg_e", "reg_z", "udap", "Debt_collection",
        "loan_servicing", "mortgage_misconduct", "auto_lending",
        "student_loan", "credit_furnishing", "payments_app_failure",
        "privacy_data_abuse", "remittance"
    ]

    for p in priority:
        if p in hits:
            return p
    return "generic"


def tag_fraud(text: str):
    t = text or ""
    hits = [k for k, rx in FRAUD_REGEX.items() if rx.search(t)]
    fraud_type = classify_violation(hits)
    return hits, fraud_type


def summarize_lead(txt: str, n: int = 3) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (txt or "").strip())
    return " ".join(parts[:n])


# =====================================================================
# 2. CFPB Feed Scraping + Parsing
# =====================================================================

CFPB_FEEDS = [
    "https://www.consumerfinance.gov/about-us/newsroom/feed/",
    "https://www.consumerfinance.gov/about-us/blog/feed/",
    "https://www.consumerfinance.gov/enforcement/actions/feed/",
]

HEADERS = {"User-Agent": "UNCC-USAA-FraudResearch/1.0 (+edu)"}


def clean_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    txt = " ".join(soup.get_text(" ").split())
    return txt


def fetch_article_text(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        doc = Document(r.text)
        main_html = doc.summary()
        txt = clean_text(main_html)
        if len(txt) < 400:
            txt = " ".join(
                p.get_text(" ", strip=True)
                for p in BeautifulSoup(r.text, "lxml").find_all("p")
            )
        return txt.strip()
    except Exception:
        return ""


def parse_date(entry) -> str:
    for k in ("published", "updated", "pubDate"):
        if entry.get(k):
            try:
                return dtparse.parse(entry[k]).date().isoformat()
            except Exception:
                pass
    return datetime.utcnow().date().isoformat()


def load_items(feeds, per_feed=2000):
    for feed in feeds:
        try:
            d = feedparser.parse(feed)
            domain = urlparse(feed).netloc
            for e in d.entries[:per_feed]:
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                if title and link:
                    yield title, link, parse_date(e), domain
        except Exception:
            continue


# =====================================================================
# 3. Scraper + Tagging Pipeline
# =====================================================================

def scrape_cfpb(limit: int = 25, days: int = 365, feeds=None) -> pd.DataFrame:
    feeds = feeds or CFPB_FEEDS
    results = []
    cutoff = datetime.utcnow().date() - timedelta(days=days)

    with tqdm(total=limit, desc="CFPB articles", unit="art") as bar:
        for title, link, date_iso, domain in load_items(feeds, per_feed=max(1000, limit)):
            try:
                d = dtparse.parse(date_iso).date()
            except Exception:
                d = datetime.utcnow().date()

            # if d < cutoff:
            #     continue

            text = fetch_article_text(link)
            if not text:
                continue

            results.append({
                "source": domain,
                "date": d.isoformat(),
                "title": title,
                "url": link,
                "text": text,
            })

            bar.n = min(len(results), limit)
            bar.refresh()

            if len(results) >= limit * 2:
                break
            time.sleep(0.2)

    if not results:
        return pd.DataFrame(columns=["source", "date", "title", "url", "text"])

    df = pd.DataFrame(results)

    # Apply new classification system
    texts = df["text"].astype(str)
    tag_hits = texts.apply(lambda x: tag_fraud(x)[0])
    df["fraud_type"] = texts.apply(lambda x: tag_fraud(x)[1])
    df["fraud_tags"] = tag_hits.apply(lambda xs: ", ".join(xs))
    df["summary"] = texts.apply(summarize_lead)

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop_duplicates(subset=["title", "url"])
    df = df.sort_values("date_dt", ascending=False).drop(columns=["date_dt"])
    return df.head(limit)


# =====================================================================
# 4. CLI Entry Point
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Scrape CFPB articles (newsroom/blog/enforcement).")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    df = scrape_cfpb(limit=args.limit, days=args.days)
    print(f"Found {len(df)} CFPB articles.")

    if len(df):
        out = args.out or f"data/processed/cfpb_articles_{datetime.now().date():%Y%m%d}.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()