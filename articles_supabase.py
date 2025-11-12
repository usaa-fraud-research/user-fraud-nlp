import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv  
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# Path to CSV 
CSV_PATH = "/Users/gustave/Desktop/untitled folder 2/usaa_nlp/user-fraud-nlp/data/processed/cfpb_articles_20251111.csv"

# Upload function 
def upload_csv(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"Uploading {len(df)} rows from {CSV_PATH}...")

    # Only include the relevant columns
    cols = ["source", "date", "title", "url", "text", "fraud_type", "fraud_tags", "summary"]
    records = df[cols].to_dict(orient="records")

    # Upload in batches (avoids rate limits)
    # Upload data to Supabase in small chunks
    # BATCH = 50 means we upload 50 rows at a time instead of everything at once.
    # This prevents Supabase API timeouts or payload errors when the dataset grows larger.
    # For example, if we scrape hundreds of articles later, it will upload them in safe, smaller pieces.
    # Since we currently have ~20 articles, this runs just once — but it keeps the pipeline scalable.
    BATCH = 50
    for i in range(0, len(records), BATCH):
        batch = records[i:i+BATCH]
        supabase.table("cfpb_articles").upsert(batch).execute()
        print(f"✅ Uploaded {len(batch)} rows")

    print("🎉 All data uploaded successfully!")

if __name__ == "__main__":
    upload_csv(CSV_PATH)