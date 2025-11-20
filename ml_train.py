"""
ml_train.py

Train a simple ML model on top of OpenAI embeddings stored in Supabase.
Goal:
  - Use cfpb_articles.embedding as features
  - Use fraud_type as label
  - Save trained model to disk for later scoring (ml_alerts.py)
"""

import os
from collections import Counter
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # comes with scikit-learn dependency

MODEL_PATH = "models/fraud_type_logreg.joblib"


def get_client():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)


def _parse_embedding(e):
    """
    Ensure each embedding is a list[float].

    Handles:
      - already-a-list
      - JSON string like "[0.1, 0.2, ...]"
      - plain string "0.1,0.2,..." as fallback
    """
    if isinstance(e, list):
        return e
    if isinstance(e, np.ndarray):
        return e.tolist()
    if isinstance(e, str):
        s = e.strip()
        try:
            # try JSON first
            val = json.loads(s)
            if isinstance(val, list):
                return val
        except json.JSONDecodeError:
            pass

        # fallback: split on commas
        s = s.strip("[]")
        return [float(x) for x in s.split(",") if x.strip()]

    raise ValueError(f"Unsupported embedding type: {type(e)}")
    

def fetch_training_data():
    """
    Pull embeddings + labels from Supabase.

    - require embedding IS NOT NULL
    - drop rows where fraud_type is null or 'not_fraud'
    - drop labels with < 2 samples (cannot be stratified)
    - PARSE embeddings from strings -> list[float]
    """
    sb = get_client()
    res = (
        sb.table("cfpb_articles")
        .select("id, fraud_type, embedding")
        .not_.is_("embedding", "null")
        .execute()
    )

    rows = res.data or []
    if not rows:
        raise RuntimeError("No rows with embeddings found in cfpb_articles")

    df = pd.DataFrame(rows)

    # basic filtering
    df = df.dropna(subset=["fraud_type"])
    df = df[df["fraud_type"] != "not_fraud"].copy()

    # drop rare labels
    counts = Counter(df["fraud_type"])
    keep_labels = {label for label, c in counts.items() if c >= 2}
    dropped = [label for label, c in counts.items() if c < 2]
    if dropped:
        print("Dropping rare labels (count < 2):", dropped)
        df = df[df["fraud_type"].isin(keep_labels)].copy()

    if df.empty:
        raise RuntimeError("After dropping rare labels, no data left to train on.")

    # 🔑 parse embedding strings into real vectors
    df["embedding_parsed"] = df["embedding"].apply(_parse_embedding)
    X = np.vstack(df["embedding_parsed"].to_list())  # shape: (N, 1536)
    y = df["fraud_type"].astype(str).values

    print(f"Training set size: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Unique labels (after filter): {sorted(set(y))}")

    return X, y


def train_and_save_model():
    """Train a logistic regression model and save it to disk."""
    X, y = fetch_training_data()

    # If there is only 1 label left, just fit on all data (no test split)
    unique_labels = np.unique(y)
    if len(unique_labels) == 1:
        print("Only one label present; training on full data (no test split).")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,  # safe now because min class count >= 2
        )

        clf = LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        print("\n=== Classification report (fraud_type) ===")
        print(classification_report(y_test, y_pred))

    # Ensure models/ directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\n✅ Saved model → {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model() 