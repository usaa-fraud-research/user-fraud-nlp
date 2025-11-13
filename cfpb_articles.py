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

# --- Fraud tagging helpers ---
FRAUD_PATTERNS = {
    "identity_theft": r"\b(identity\s*theft|stolen\s*identity|synthetic\s*identity)\b",
    "account_takeover": r"\b(account\s*takeover|ATO|SIM\s*swap|credential\s*stuffing)\b",
    "check_fraud": r"\b(check\s*fraud|fake\s*check|counterfeit\s*check|check\s*kiting)\b",
    "card_fraud": r"\b(credit|debit)\s*card\s*(fraud|skimming|cloning|chargeback)\b",
    "wire_transfer": r"\b(wire\s*fraud|unauthorized\s*wire|zelle|money\s*transfer\s*fraud)\b",
    "phishing": r"\b(phish(ing|ed)?|smish(ing)?|vish(ing)?)\b",
    "crypto": r"\b(crypto|bitcoin|ether(eum)?)\b.*\b(fraud|scam|rug)\b",
    "romance": r"\b(romance\s*scam|catfish(ing)?)\b",
    "pig_butchering": r"\bpig[-\s]?butcher(ing)?\b",
    "money_laundering": r"\b(money\s*launder(ing|er|ed)|AML)\b",

    # 🆕 New subtypes
    "investment_fraud": r"\b(investment|ponzi|securities)\s*(fraud|scheme|scam)\b",
    "loan_fraud": r"\b(auto|student|mortgage|loan)\s*(fraud|scam|scheme)\b",
    "insurance_fraud": r"\b(insurance)\s*(fraud|scam|scheme)\b",
    "employment_scam": r"\b(job|employment)\s*(scam|scheme|fraud)\b",
    "charity_scam": r"\b(charity|donation)\s*(scam|scheme|fraud)\b",

    # Fallback
    "generic": r"\b(fraud|scam|deceptive|scheme|scamming)\b",
}

FRAUD_REGEX = {k: re.compile(v, re.I) for k, v in FRAUD_PATTERNS.items()}


results = []  # Initialize results as an empty list
df = pd.DataFrame(results)
print("Raw scraped rows:", len(df))  # 👈 all within feeds

# --- Smart fraud tagging (single block) ---
if not df.empty:
    ...
    df = df[df["fraud_type"] != "not_fraud"].copy()
    ...
    print("Fraud-tagged rows:", len(df))  # 👈 after filter


def tag_fraud(text: str):
    t = text or ""
    hits = [k for k, rx in FRAUD_REGEX.items() if rx.search(t)]
    priority = [
        "account_takeover", "identity_theft", "check_fraud", "card_fraud",
        "wire_transfer", "phishing", "crypto", "pig_butchering",
        "romance", "money_laundering", "generic"
    ]
    primary = next((p for p in priority if p in hits), None)
    return hits, (primary or "not_fraud")


def summarize_lead(txt: str, n: int = 3) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (txt or "").strip())
    return " ".join(parts[:n])


# --- CFPB feeds (newsroom + blog + enforcement highlights) ---
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
    """Fetch a page and return readable text."""
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
    """Yield tuples (title, link, date, feed_domain)."""
    for feed in feeds:
        try:
            d = feedparser.parse(feed)
            domain = urlparse(feed).netloc
            for e in d.entries[:per_feed]:
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                if not title or not link:
                    continue
                yield title, link, parse_date(e), domain
        except Exception:
            continue


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

            #if d < cutoff: 
                #continue

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

    # --- Smart fraud tagging (single block) ---
    if not df.empty:
        texts = df["text"].astype(str)
        tag_hits = texts.apply(lambda x: tag_fraud(x)[0])
        df["fraud_type"] = texts.apply(lambda x: tag_fraud(x)[1])
        df = df[df["fraud_type"] != "not_fraud"].copy()
        df["fraud_tags"] = tag_hits.apply(lambda xs: ", ".join(xs))
        df["summary"] = texts.apply(summarize_lead)

    # --- Clean and sort ---
    df = df.drop_duplicates(subset=["title", "url"])
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date_dt", ascending=False).drop(columns=["date_dt"])
    return df.head(limit)


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