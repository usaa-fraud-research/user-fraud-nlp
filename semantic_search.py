# semantic_search.py
# ------------------------------------------------------------
# Purpose:
#   This script performs *semantic search* over CFPB articles stored in Supabase.
#   It embeds user queries with OpenAI and uses pgvector similarity search.
#   We resave it in Supabase so repeated searches don’t re-call OpenAI.
# ------------------------------------------------------------

import os, hashlib, re
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# === Step 1: Load environment variables (.env file) ===
# ------------------------------------------------------
load_dotenv()
sb  = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "text-embedding-3-small"   # OpenAI’s 1536-dimensional embedding model


# === Step 2: Helper functions for caching ===
# ------------------------------------------------------
# These handle normalization and hashing of the search query.
# We create a short unique “query hash” so we can easily find
# previously embedded queries in the Supabase cache.

def _normalize(q: str) -> str:
    """Normalize a query: lowercase, remove extra spaces."""
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    return q

def _qhash(q: str) -> str:
    """Create a unique SHA256 hash of the normalized query text."""
    return hashlib.sha256(q.encode("utf-8")).hexdigest()


# === Step 3: Embedding cache system ===
# ------------------------------------------------------
# This function checks if a query already exists in the 'search_queries'
# table in Supabase. If it does, it reuses the saved embedding.
# If not, it calls OpenAI, generates an embedding, and saves it for future use.

def get_or_create_query_embedding(query: str) -> list[float]:
    """
    Return cached embedding if present; otherwise call OpenAI and cache it.
    This saves money and speeds up repeated searches.
    """
    norm = _normalize(query)
    h = _qhash(norm)

    # --- 1️⃣ Try to find cached embedding ---
    res = sb.table("search_queries").select("embedding, uses").eq("query_hash", h).limit(1).execute()
    if res.data:
        # Found in cache → update 'uses' counter and timestamp (optional)
        sb.table("search_queries").update({
            "uses": (res.data[0]["uses"] or 0) + 1,
            "last_used_at": "now()"
        }).eq("query_hash", h).execute()
        print(f"✅ Cache hit for query: '{query}'")
        return res.data[0]["embedding"]

    # --- 2️⃣ Cache miss → create embedding via OpenAI ---
    print(f"🚀 Generating new embedding for query: '{query}'")
    emb = client.embeddings.create(model=MODEL, input=norm).data[0].embedding

    # --- 3️⃣ Save new embedding in Supabase ---
    sb.table("search_queries").insert({
        "query_text": norm,
        "query_hash": h,
        "embedding": emb
    }).execute()

    return emb


# === Step 4: Semantic search using Supabase RPC (remote procedure call) ===
# ------------------------------------------------------
# The pgvector search happens inside Supabase through the
# 'match_cfpb_articles' function.
# This function retrieves similar articles based on cosine similarity.

def search(query: str, top_k: int = 20, threshold: float = 0.55,
           year: int | None = None, keyword: str | None = None):
    """
    Perform semantic search for fraud-related CFPB articles.

    Parameters:
      query      → The natural language text to search for.
      top_k      → Max number of results to return.
      threshold  → Minimum similarity (0–1); higher = stricter.
      year       → Optional year filter.
      keyword    → Optional keyword filter for titles.
    """

    # --- Step 1: Get embedding (cached or new) ---
    qvec = get_or_create_query_embedding(query)

    # --- Step 2: Run Supabase vector search via RPC ---
    rpc = sb.rpc("match_cfpb_articles", {
        "query_embedding": qvec,
        "match_count": 100,
        "match_threshold": threshold
    }).execute()

    rows = rpc.data or []

    # --- Step 3: Optional local filters (year, keyword) ---
    if year:
        rows = [r for r in rows if r.get("date") and str(year) in str(r["date"])]
    if keyword:
        rows = [r for r in rows if keyword.lower() in (r.get("title") or "").lower()]

    # --- Step 4: Sort and show top results ---
    rows = sorted(rows, key=lambda r: r["similarity"], reverse=True)[:top_k]

    print(f"\n=== Semantic Search Results for: '{query}' ===\n")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r['title']}  ({r.get('date','N/A')})")
        print(f"    sim={r['similarity']:.3f}  {r['url']}")
    print(f"\nTotal matches: {len(rows)}")

    return rows


# === Step 5: Example usage ===
# ------------------------------------------------------
# This section runs when I execute the file directly.
# It includes sample queries that show how semantic search works.
# You can comment/uncomment lines below to test different queries.

if __name__ == "__main__":
    search("zelle unauthorized transfer", top_k=10)
    search("identity theft on credit cards", top_k=10)
    search("romance scam refund policy", keyword="CFPB", top_k=10)