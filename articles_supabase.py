import os
import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv  

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Path to CSV 
CSV_PATH = r"C:\Dev\user-fraud-nlp\data\processed\cfpb_articles_20251116.csv"

# Upload function 
def upload_csv(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"Uploading {len(df)} rows from {CSV_PATH}...")

    # ------------------------------------------------------------------
    # ✅ FIX: Replace NaN with None so JSON upload to Supabase works
    # Supabase/PostgREST does NOT accept NaN, Infinity, or -Infinity.
    # pandas → Python None → valid JSON null
    # ------------------------------------------------------------------
    df = df.replace({np.nan: None})
    df = df.where(pd.notnull(df), None)

    # Only include the relevant columns
    cols = ["source", "date", "title", "url", "text", "fraud_type", "fraud_tags", "summary"]
    records = df[cols].to_dict(orient="records")

    # Upload in batches (avoids rate limits)
    BATCH = 50
    for i in range(0, len(records), BATCH):
        batch = records[i:i+BATCH]
        supabase.table("cfpb_articles").upsert(batch, on_conflict="url").execute()
        print(f"✅ Uploaded {len(batch)} rows")

    print("🎉 All data uploaded successfully!")

if __name__ == "__main__":
    upload_csv(CSV_PATH)
