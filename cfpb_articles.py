# cfpb_articles.py
import os
import re
import time
import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from readability import Document
from dateutil import parser as dtparse
from tqdm import tqdm
from urllib.parse import urlparse, urljoin
from datetime import datetime
# =====================================================================
# 1. Enhanced Fraud + Regulatory Violation Tagging (your logic)
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

    # --- CFPB-specific regulatory violations ---
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
        "fcra", "reg_e", "reg_z", "udap", "debt_collection",
        "loan_servicing", "mortgage_misconduct", "auto_lending",
        "student_loan", "credit_furnishing", "payments_app_failure",
        "privacy_data_abuse", "remittance",
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
# 2. HTTP helpers & text cleanup
# =====================================================================

HEADERS = {"User-Agent": "UNCC-USAA-FraudResearch/1.0 (+edu)"}

DATE_RE = re.compile(
    r"\b("
    r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
    r")\s+\d{1,2},\s+\d{4}\b",
    re.I,
)


def clean_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    txt = " ".join(soup.get_text(" ").split())
    return txt


def fetch_url(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def parse_article_date(html: str):
    """
    Grab the first 'JAN 10, 2025' style date on the page.
    Works for Newsroom, Blog, and (usually) Enforcement pages.
    """
    m = DATE_RE.search(html)
    if not m:
        return None
    try:
        return dtparse.parse(m.group(0)).date()
    except Exception:
        return None


def parse_article_title(html: str):
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    # fallback: <title>
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def extract_article(url: str):
    """
    Fetch full article page and return dict:
    source, date, title, url, text
    """
    try:
        html = fetch_url(url)
    except Exception:
        return None

    # Main text via readability
    try:
        doc = Document(html)
        main_html = doc.summary()
    except Exception:
        main_html = html

    text = clean_text(main_html)
    if len(text) < 400:
        # fallback: concatenate all <p> tags
        soup = BeautifulSoup(html, "lxml")
        text = " ".join(
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
        )
        text = " ".join(text.split())

    date_obj = parse_article_date(html)
    title = parse_article_title(html)

    return {
        "source": urlparse(url).netloc,
        "date": date_obj.isoformat() if date_obj else None,
        "title": title,
        "url": url,
        "text": text,
    }


# =====================================================================
# 3. Crawl CFPB archive index pages (Newsroom, Blog, Enforcement)
# =====================================================================

SECTIONS = [
    {
        "name": "newsroom",
        "template": "https://www.consumerfinance.gov/about-us/newsroom/?page={page}",
        "href_prefix": "/about-us/newsroom/",
        # current site says 85 pages; give a little headroom
        "max_pages": 90,
    },
    {
        "name": "blog",
        "template": "https://www.consumerfinance.gov/about-us/blog/?page={page}",
        "href_prefix": "/about-us/blog/",
        # blog shows 68 pages; give headroom
        "max_pages": 75,
    },
    {
        "name": "enforcement",
        "template": "https://www.consumerfinance.gov/enforcement/actions/?page={page}",
        "href_prefix": "/enforcement/actions/",
        # enforcement shows 16 pages; give headroom
        "max_pages": 25,
    },
]


def gather_section_urls(section) -> set:
    """
    Crawl paginated index pages and collect article URLs for one section.
    """
    base_template = section["template"]
    href_prefix = section["href_prefix"]
    max_pages = section["max_pages"]

    urls = set()
    for page in range(1, max_pages + 1):
        index_url = base_template.format(page=page)
        try:
            html = fetch_url(index_url)
        except Exception:
            # stop if this page blows up – usually means we've gone too far
            break

        soup = BeautifulSoup(html, "lxml")
        new_count = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith(href_prefix):
                continue
            # filter out the bare section root, e.g. "/about-us/newsroom/"
            if href.rstrip("/") == href_prefix.rstrip("/"):
                continue

            full = urljoin("https://www.consumerfinance.gov", href)
            if full not in urls:
                urls.add(full)
                new_count += 1

        # If we got no *new* URLs from this page, it's probably beyond the archive
        if new_count == 0:
            # but don't be too aggressive: continue a bit more if you want
            # here we just break for simplicity
            break

        # be nice
        time.sleep(0.2)

    return urls


def gather_all_urls() -> list:
    """
    Collect URLs for all sections.
    """
    all_urls = set()
    for sec in SECTIONS:
        print(f"🔎 Crawling {sec['name']} index pages…")
        urls = gather_section_urls(sec)
        print(f"  → {len(urls)} URLs from {sec['name']}")
        all_urls.update(urls)
    print(f"🌐 Total unique article URLs found: {len(all_urls)}")
    return sorted(all_urls)


# =====================================================================
# 4. Full archive scraper + fraud tagging
# =====================================================================

def scrape_cfpb_archive(start_year: int, end_year: int | None, max_articles: int | None):
    urls = gather_all_urls()
    results = []

    pbar = tqdm(urls, desc="CFPB articles", unit="art")
    for url in pbar:
        article = extract_article(url)
        if article is None:
            continue

        # Date filtering by year
        date_str = article["date"]
        year_ok = True
        year_val = None
        if date_str:
            try:
                year_val = dtparse.parse(date_str).year
            except Exception:
                year_val = None

        if year_val is not None:
            if year_val < start_year:
                year_ok = False
            if end_year is not None and year_val > end_year:
                year_ok = False

        if not year_ok:
            continue

        text = article["text"]
        hits, fraud_type = tag_fraud(text)
        article["fraud_type"] = fraud_type
        article["fraud_tags"] = ", ".join(hits)
        article["summary"] = summarize_lead(text)

        results.append(article)

        if max_articles is not None and len(results) >= max_articles:
            break

        # slow it down a bit so we don't hammer CFPB
        time.sleep(0.2)

    if not results:
        return pd.DataFrame(
            columns=["source", "date", "title", "url", "text", "fraud_type", "fraud_tags", "summary"]
        )

    df = pd.DataFrame(results)

    # clean + sort
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop_duplicates(subset=["title", "url"])
    df = df.sort_values("date_dt", ascending=False).drop(columns=["date_dt"])

    return df


# =====================================================================
# 5. CLI Entry Point
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Scrape CFPB Newsroom/Blog/Enforcement archive (HTML) and tag fraud."
    )
    current_year = datetime.utcnow().year

    ap.add_argument("--start-year", type=int, default=2020,
                    help="Earliest year to keep (default: 2020)")
    ap.add_argument("--end-year", type=int, default=current_year,
                    help=f"Latest year to keep (default: {current_year})")
    ap.add_argument("--max-articles", type=int, default=None,
                    help="Optional hard cap on number of articles (for testing)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV path (default: data/processed/cfpb_articles_YYYYMMDD_full.csv)")

    args = ap.parse_args()

    print(f"📆 Filtering to years {args.start_year}–{args.end_year}")
    if args.max_articles:
        print(f"⛔ Limiting to {args.max_articles} articles (testing mode)")

    df = scrape_cfpb_archive(
        start_year=args.start_year,
        end_year=args.end_year,
        max_articles=args.max_articles,
    )

    print(f"Found {len(df)} CFPB articles in selected year range.")

    if len(df):
        out = args.out or f"data/processed/cfpb_articles_{datetime.now().date():%Y%m%d}_full.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print(f"💾 Saved → {out}")
    else:
        print("⚠️ No matching articles scraped.")


if __name__ == "__main__":
    main()