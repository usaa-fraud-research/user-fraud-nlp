import os, math
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Model using 1536-dim embeddings; smaller and cheaper

MODEL = "text-embedding-3-small"  # 1536-dim
BATCH = 50
FETCH_LIMIT = 500   # how many rows to embed per run

# ============================================================
#  STEP 1 — Fetch missing rows (no embeddings yet)
# ------------------------------------------------------------
# Limiting the result set prevents overloading the API when large datasets exist.
# Fetch rows from Supabase that don't yet have embeddings.
# We only select 'url' and 'text' because 'url' is unique
# and can safely be used for conflict resolution.
# ============================================================
def fetch_missing():
    res = (
        supabase.table("cfpb_articles")
        .select("url, text")
        .is_("embedding", "null")
        .limit(FETCH_LIMIT)
        .execute()
    )
    return res.data or []

# ============================================================
#  STEP 2 — Embed text content using OpenAI
# ------------------------------------------------------------
# Here we send each article’s text to OpenAI’s embedding model.
# OpenAI returns a high-dimensional vector (1536 numbers per article)
# that captures the article’s semantic meaning.
# These vectors can later be used for:
#   - semantic search
#   - clustering similar fraud topics
#   - feeding into ML models
# =====================================
def embed_texts(texts):
    # Truncate to control cost; model handles long text but keep it practical
    inputs = [(t or "")[:6000] for t in texts]
    resp = client.embeddings.create(model=MODEL, input=inputs)
    return [d.embedding for d in resp.data]

# ============================================================
#  STEP 3 — Upload embeddings back to Supabase
# ------------------------------------------------------------
# After generating embeddings, we update each article record in Supabase.
# We use “upsert” which means insert new data or update existing data if the ID already exists.
# This keeps the table clean and ensures that each article only has one embedding.
# matching on 'url' (since it’s unique) instead of 'id'.
# ======================================================
def upsert_embeddings(rows):
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        embeddings = embed_texts([r["text"] or "" for r in chunk])

        # Build payload using URL instead of ID
        payload = [{"url": chunk[j]["url"], "embedding": embeddings[j]} for j in range(len(chunk))]

        # Upsert rows by matching the unique URL field
        supabase.table("cfpb_articles").upsert(payload, on_conflict="url").execute()

        print(f"✅ Upserted {i + len(chunk)}/{len(rows)} embeddings")
# ============================================================
#  STEP 4 — Run everything together
# ------------------------------------------------------------
# The main execution flow:
# Find all articles missing embeddings.
# Generate OpenAI embeddings in batches.
# Upload them to Supabase for storage and retrieval.
# This ensures the entire dataset can be used later for semantic search and for our ML training.
# ============================================================
if __name__ == "__main__":
    rows = fetch_missing()
    print(f"Rows needing embeddings: {len(rows)}")
    if rows:
        upsert_embeddings(rows)
        print("✅ Done.")
    else:
        print("Nothing to embed. ✅")