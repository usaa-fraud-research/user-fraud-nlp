# fraud_dashboard.py
import os
from datetime import date, timedelta
import calendar

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
from dotenv import load_dotenv
from supabase import create_client, Client
from semantic_search import search as semantic_search
import streamlit.components.v1 as components 
from joblib import load 

# Preset semantic search scenarios (high-level questions)
SEMANTIC_PRESETS = {
    "🔁 Zelle / payment app scams": "unauthorized zelle or payment app transfers and how the bank resolved them",
    "🪪 Identity theft & account takeover": "identity theft on credit cards or bank accounts and how it was resolved",
    "🏠 Mortgage & home lending issues": "mortgage servicing errors, foreclosure, escrow problems, misleading home loans",
    "🎓 Student loan servicing problems": "student loan servicing issues, misapplied payments, and forgiveness confusion",
    "📞 Debt collection harassment": "aggressive or illegal debt collection tactics and consumer protections",
    "💳 Credit reporting errors (FCRA)": "credit report errors, disputes under FCRA, and correction outcomes",
    "🌍 Remittances / international transfers": "remittance transfer problems, high fees, or lost international payments",
    "⚖️ UDAP / deceptive practices": "unfair, deceptive, or abusive acts and practices in banking or lending",
}


# ------------------------------------------------------------
# Supabase client setup
# ------------------------------------------------------------
def get_client() -> Client:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data():
    supabase = get_client()
    res = supabase.table("cfpb_articles").select("*").execute()
    df = pd.DataFrame(res.data)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    return df


# ------------------------------------------------------------
# cache wrapper so repeated searches don't re-run
# ------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def run_cached_semantic(query, top_k, threshold, year, keyword):
    return semantic_search(
        query=query,
        top_k=top_k,
        threshold=threshold,
        year=year,
        keyword=keyword,
    )
@st.cache_resource
def load_ml_model():
    """Load the trained fraud-type classifier (SVM or logistic)."""
    model_path = "models/svm_fraud_type.joblib"  # or your actual filename
    return load(model_path)



# ------------------------------------------------------------
# Main Streamlit App
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="USAA Fraud NLP Dashboard", layout="wide")
    st.title("🧠 USAA Fraud Detection — Full Project Dashboard")

    df = load_data()
    if df.empty:
        st.error("❌ No data found in Supabase. Please run scraper + upload first.")
        return
    st.success(f"✅ Loaded {len(df)} CFPB articles") 
    # --- Simple ML predictions (if embeddings are available) ---
    ml_available = False
    if "embedding" in df.columns:
        try:
            model = load_ml_model()
            # embeddings come back as list -> stack into matrix
            emb_matrix = np.vstack(df["embedding"].values)
            df["ml_pred"] = model.predict(emb_matrix)
            ml_available = True
        except Exception as e:
            st.warning(f"ML model could not run: {e}")

    week2, week3, week4, semtab = st.tabs(
        [
            "📄 Week 2: Scraper",
            "🔍 Week 3: Fraud Detection",
            "📊 Week 4: Analysis",
            "🔎 Semantic Search",
        ]
    )

    # =======================
    # WEEK 2 — SCRAPER
    # =======================
    with week2:
        st.header("Week 2 — Build Scraper")
        st.write("Goal: Get articles from CFPB source and clean text for analysis.")
        st.write(f"**Total Articles Scraped:** {len(df)}")
        st.dataframe(df[["title", "date", "url"]].head(10))

        st.subheader("Example Article Text Preview")
        sample = df.sample(1).iloc[0]
        st.markdown(f"**Title:** {sample['title']}")
        st.markdown(f"**Date:** {sample['date'].date()}")
        st.markdown(f"**URL:** [Read Article]({sample['url']})")
        st.text_area("Excerpt", (sample.get("text") or "")[:600] + "...")

    # ==========================
    # WEEK 3 — FRAUD DETECTION
    # ==========================
    with week3:
        st.header("Week 3 — Fraud Detection")
        st.write("Goal: Identify and tag fraud-related articles.")

        st.metric(
            "Distinct Fraud Types",
            int(df["fraud_type"].dropna().nunique()),
        )
        st.metric(
            "Total Fraud-Tagged Articles",
            int((df["fraud_type"] != "not_fraud").sum()),
        )

        st.subheader("Fraud Tags by Frequency")
        tags_col = df["fraud_tags"].dropna().tolist()
        tag_list = [
            t.strip()
            for tags_str in tags_col
            for t in tags_str.split(",")
            if t.strip()
        ]
        tag_counts = Counter(tag_list)
        kw_df = pd.DataFrame(
            tag_counts.most_common(10), columns=["Keyword", "Count"]
        )
        st.bar_chart(kw_df.set_index("Keyword"))
        st.dataframe(kw_df)

        st.subheader("Word Cloud of Fraud Keywords")
        if tag_list:
            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate(" ".join(tag_list))
            st.image(wordcloud.to_array(), width=800)
                # ==========================
        # 🔔 Simple ML Alerts
        # ==========================
        st.subheader("🔔 ML Alerts (high-priority fraud types)")

        if not ml_available:
            st.caption("ML model not available or no embeddings column in Supabase.")
        else:
            HIGH_PRIORITY = ["reg_e", "identity_theft", "wire_transfer",
                             "crypto", "udap"]

            alerts = (
                df[df["ml_pred"].isin(HIGH_PRIORITY)]
                .sort_values("date", ascending=False)
                .head(10)
            )

            if alerts.empty:
                st.info("No high-priority ML alerts right now.")
            else:
                for _, row in alerts.iterrows():
                    st.markdown(
                        f"**{row['title']}** — *{row['ml_pred']}* "
                        f"({row['date'].date()})  \n"
                        f"{row.get('summary','')}  \n"
                        f"[Read more]({row['url']})"
                    )

    # ==========================
    # WEEK 4 — ANALYSIS
    # ==========================
    with week4:
        st.header("Week 4 — Full Pipeline & Trend Analysis")
        st.write(
            "Goal: Complete pipeline (scrape → detect → summarize) and identify top trends."
        )

        col1, col2 = st.columns(2)
        fraud_types = df["fraud_type"].dropna().unique().tolist()
        selected_types = col1.multiselect(
            "Filter by Fraud Type", fraud_types, default=fraud_types
        )
        date_min, date_max = df["date"].min().date(), df["date"].max().date()
        date_range = col2.slider(
            "Select Date Range", date_min, date_max, (date_min, date_max)
        )

        colD = st.columns(1)[0]
        enable_year = colD.toggle("Filter by specific year", value=False)
        year = None
        if enable_year:
            default_year = int(df["date"].dt.year.min()) if not df.empty else 2020
            year = colD.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                value=default_year,
                step=1,
            )

        filtered_df = df[
            (df["fraud_type"].isin(selected_types))
            & (
                df["date"].between(
                    pd.to_datetime(date_range[0]),
                    pd.to_datetime(date_range[1]),
                )
            )
        ]
        if year:
            filtered_df = filtered_df[filtered_df["date"].dt.year == year]

        c3, c4, c5 = st.columns(3)
        c3.metric("Articles in Range", len(filtered_df))
        c4.metric("Fraud Types Displayed", len(selected_types))
        c5.metric("Date Range", f"{date_range[0]} → {date_range[1]}")

        st.subheader("🔑 Top Keywords / Phrases")
        if not filtered_df.empty:
            tags = filtered_df["fraud_tags"].dropna().tolist()
            all_tags = [
                t.strip()
                for tags_str in tags
                for t in tags_str.split(",")
                if t.strip()
            ]
            tag_counts = Counter(all_tags)
            kw_df = pd.DataFrame(
                tag_counts.most_common(10), columns=["Keyword", "Count"]
            )
            fig_kw = px.bar(
                kw_df,
                x="Keyword",
                y="Count",
                title="Top Fraud Keywords / Phrases",
                color="Count",
            )
            st.plotly_chart(fig_kw, use_container_width=True)

        st.subheader("🚨 Top Fraud Types")
        fraud_trend = (
            filtered_df.groupby("fraud_type")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="Count")
        )
        if not fraud_trend.empty:
            fig_ft = px.bar(
                fraud_trend,
                x="fraud_type",
                y="Count",
                color="Count",
                title="Fraud Type Frequency",
            )
            st.plotly_chart(fig_ft, use_container_width=True)

        # ==========================
        # GitHub-Style Fraud Activity Heatmap (dynamic by selected year)
        # ==========================
        st.subheader("GitHub-Style Fraud Activity (Heatmap by Daily Count)")

        # --- Detect if the user selected a specific year ---
        if enable_year and year:
            start_day = pd.Timestamp(year=year, month=1, day=1)
            end_day = pd.Timestamp(year=year, month=12, day=31)
        else:
            # default GitHub style: last 52 weeks
            today = date.today()
            weekday_offset = (today.weekday() + 1) % 7  # align to Sunday
            this_week_start = today - timedelta(days=weekday_offset)
            start_day = this_week_start - timedelta(weeks=52)
            end_day = this_week_start + timedelta(days=6)

        # Build the date range
        all_days = pd.date_range(start=start_day, end=end_day, freq="D")

        # Daily count of articles
        day_counts = (
            df.groupby(df["date"].dt.date)
              .size()
              .to_dict()
        )

        # GitHub intensity color scale
        def intensity_color(count):
            if count == 0:
                return "#161b22"  # empty
            elif count == 1:
                return "#0e4429"  # light
            elif count <= 3:
                return "#006d32"  # mid
            elif count <= 6:
                return "#26a641"  # high
            else:
                return "#39d353"  # very high

        # Build HTML
        html = """
        <style>
        .calendar-container {
            width: 100%;
            overflow-x: auto;
        }
        .calendar-wrapper {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            width: 100%;
        }
        .calendar {
            display: grid;
            grid-template-rows: repeat(7, 14px);
            grid-auto-flow: column;
            grid-auto-columns: 1fr;
            grid-gap: 2px;
            width: 100%;
        }
        .daybox {
            width: 100%;
            height: 14px;
            border-radius: 2px;
            cursor: pointer;
        }
        .daybox:hover {
            outline: 1px solid #58a6ff;
        }
        .month-labels {
            display: flex;
            justify-content: space-between;
            margin-left: 24px;
            font-size: 10px;
            color: #8b949e;
            width: calc(100% - 24px);
        }
        .weekday-labels {
            display: grid;
            grid-template-rows: repeat(7, 16px);
            grid-gap: 2px;
            font-size: 10px;
            color: #8b949e;
            margin-top: 14px;
        }
        .legend {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 10px;
            color: #8b949e;
            margin-top: 10px;
            margin-left: 20px;
            flex-wrap: wrap;
        }
        .legend-box {
            width: 12px;
            height: 12px;
            border-radius: 2px;
            display: inline-block;
        }
        </style>

        <div class='month-labels'>
        """

        # Generate month labels dynamically based on selected range
        unique_months = sorted(set((d.year, d.month) for d in all_days))
        for (y, m) in unique_months:
            html += f"<span>{calendar.month_abbr[m]}</span>"
        html += "</div><div class='calendar-container'><div class='calendar-wrapper'>"

        # Weekday labels (Mon-Wed-Fri like GitHub)
        weekday_labels = ["Mon", "", "Wed", "", "Fri", "", ""]
        html += "<div class='weekday-labels'>"
        for label in weekday_labels:
            html += f"<span>{label}</span>"
        html += "</div>"

        # Build day grid
        html += "<div class='calendar' id='fraudCalendar'>"
        for d in all_days:
            dt = d.date()
            count = day_counts.get(dt, 0)
            color = intensity_color(count)
            title = f"{dt}: {count} article(s)"

            html += (
                f"<div class='daybox' title='{title}' data-day='{dt}' "
                f"style='background:{color};'></div>"
            )
        html += "</div></div></div>"

        # Legend
        html += """
        <div class='legend'>
          <span class='legend-box' style='background:#161b22;'></span>0
          <span class='legend-box' style='background:#0e4429;'></span>1
          <span class='legend-box' style='background:#006d32;'></span>2–3
          <span class='legend-box' style='background:#26a641;'></span>4–6
          <span class='legend-box' style='background:#39d353;'></span>7+
        </div>
        """

        # JS Click Handling
        html += """
        <script>
        const boxes = document.querySelectorAll('.daybox');
        boxes.forEach(b => {
            b.addEventListener('click', () => {
                const day = b.dataset.day;
                const frame = window.parent;
                frame.postMessage({type:'select-day', day:day}, '*');
            });
        });
        </script>
        """

        components.html(html, height=270)

        components.html(
            """
            <script>
            window.addEventListener('message', (event) => {
                if (event.data.type === 'select-day') {
                    const d = event.data.day;
                    const qs = new URLSearchParams(window.location.search);
                    qs.set('day', d);
                    window.location.search = qs.toString();
                }
            });
            </script>
            """,
            height=0,
        )

        # Drill-down article list
        params = st.query_params
        if "day" in params:
            chosen_date = pd.to_datetime(params["day"]).date()
            st.markdown(f"### Articles on {chosen_date}")
            day_articles = df[df["date"].dt.date == chosen_date]
            if day_articles.empty:
                st.info("No CFPB articles found for this day.")
            else:
                for _, r in day_articles.iterrows():
                    st.markdown(
                        f"**{r['title']}** — *{r['fraud_type']}*  \n"
                        f"{r.get('summary', '')}  \n"
                        f"[Read More]({r['url']})"
                    )

                # ==========================
        # 🔥 Top 3 Trending Fraud Types (Year-Aware)
        # ==========================
        st.subheader("🔥 Top 3 Trending Fraud Types")

        # Use the selected year if enabled; otherwise fallback to current year
        selected_year = year if enable_year and year else pd.Timestamp.now().year

        year_df = df[df["date"].dt.year == selected_year]

        FRAUD_EXPLANATIONS = {
            "fcra": "Fair Credit Reporting Act — protects consumers from inaccurate or unfair credit reporting.",
            "udap": "Unfair, Deceptive, or Abusive Acts or Practices — broad deceptive financial behavior.",
            "reg_e": "Regulation E — protects consumers from unauthorized electronic funds transfers.",
            "mortgage_misconduct": "Violations related to mortgage servicing, payments, escrow, or disclosures.",
            "loan_servicing": "Issues involving loan servicing, payments, or servicing errors.",
            "identity_theft": "Fraud involving stolen personal information.",
            "debt_collection": "Illegal or abusive debt collection practices.",
            "generic": "General fraud issue not classified under a specific law or category."
        }

        if year_df.empty:
            st.warning(f"No fraud articles found for {selected_year}.")
        else:
            top_year = (
                year_df["fraud_type"]
                .value_counts()
                .reset_index()
            )
            top_year.columns = ["fraud_type", "count"]

            top3 = top_year.head(3)

            st.markdown(f"### 📆 Showing trends for **{selected_year}**")
            st.markdown("---")

            # Clickable top 3
            for _, row in top3.iterrows():
                ftype = row["fraud_type"]
                count = row["count"]

                explanation = FRAUD_EXPLANATIONS.get(ftype, "No description available.")

                if st.button(f"{ftype.upper()} — {count} mentions", key=f"type_{ftype}_{selected_year}"):
                    st.query_params["selected_type"] = ftype
                    st.query_params["selected_year"] = selected_year

                st.caption(explanation)
                st.markdown("---")

        # ==========================
        # 📰 Filtered Article List (by click + year)
        # ==========================
        st.subheader("📰 Filtered Articles")

        selected_type = st.query_params.get("selected_type", None)
        selected_year_param = st.query_params.get("selected_year", None)

        # Validate year param from URL
        try:
            selected_year_param = int(selected_year_param) if selected_year_param else None
        except:
            selected_year_param = None

        # Determine the final year to show
        article_year = selected_year_param or selected_year

        # Filter base dataset to that year
        article_df = df[df["date"].dt.year == article_year]

        # If a fraud type is selected, filter further
        if selected_type:
            article_df = article_df[article_df["fraud_type"] == selected_type]
            st.info(f"Showing **{selected_type.upper()}** articles for **{article_year}**")
        else:
            st.info(f"Showing articles for **{article_year}**")

        # Display articles
        for _, row in article_df.head(10).iterrows():
            st.markdown(
                f"**{row['title']}** — *{row['fraud_type']}* ({row['date'].date()})  \n"
                f"{row.get('summary','')}  \n"
                f"[Read More]({row['url']})"
            )


        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download Filtered Data as CSV",
            data=csv,
            file_name="fraud_analysis_filtered.csv",
        )

           # ==========================
    # SEMANTIC SEARCH (embeddings)
    # ==========================
    with semtab:
        st.header("🔎 Semantic Search — Embeddings + pgvector")
        st.write(
            "Use semantic search to find CFPB articles that **mean** the same thing as your question, "
            "even if they use different words."
        )

        # 1️⃣ Choose a preset or type your own question
        st.markdown("### 1️⃣ Pick a scenario or ask your own question")

        colA, colB = st.columns([3, 1])

        # Preset dropdown
        preset_label = colA.selectbox(
            "Common fraud scenarios",
            options=["(None — I will type my own)"] + list(SEMANTIC_PRESETS.keys()),
            help="Pick a high-level scenario to auto-fill the search intent.",
        )

        preset_query = None
        if preset_label != "(None — I will type my own)":
            preset_query = SEMANTIC_PRESETS[preset_label]

        # Custom query overrides preset if filled
        custom_query = colA.text_input(
            "Or type your own question",
            placeholder="e.g., unauthorized Zelle transfer dispute",
            help=(
                "This text is embedded with OpenAI and compared to all CFPB article embeddings. "
                "If you type something here, it overrides the preset scenario above."
            ),
        )

        # Decide which query to actually send to the model
        if custom_query.strip():
            effective_query = custom_query.strip()
            source_label = "Using your custom question"
        elif preset_query:
            effective_query = preset_query
            source_label = f"Using preset scenario: {preset_label}"
        else:
            effective_query = ""
            source_label = "No query selected yet"

        top_k = colB.slider(
            "Top K results",
            5,
            40,
            10,
            step=5,
            help="How many top-ranked matches to return (sorted by semantic similarity).",
        )

        st.caption(f"🔍 {source_label}")

        # 2️⃣ Advanced search settings
        st.markdown("### 2️⃣ Tune search settings (optional)")

        colC, colD = st.columns([1, 1])
        threshold = colC.number_input(
            "Min similarity (0–1)",
            0.0,
            1.0,
            0.55,
            0.01,
            help=(
                "Semantic similarity threshold. "
                "Higher = stricter (fewer but more relevant results). "
                "Lower = more results, but some may be loosely related."
            ),
        )
        enable_year = colD.toggle(
            "Filter by year",
            value=False,
            help=(
                "Turn this on to only show articles whose publication year matches the 'Year' field below. "
                "Turn off for no year filter."
            ),
        )

        year = None
        if enable_year:
            default_year = 2020
            year = colD.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                value=default_year,
                step=1,
                help="Only articles published in this calendar year will be returned.",
            )

        keyword = st.text_input(
            "Optional keyword filter (title contains)",
            help=(
                "Extra plain-text filter applied *after* semantic search. "
                "If set, only articles whose **title** contains this word or phrase are kept. "
                "Leave blank for no extra filter."
            ),
        )

        # 3️⃣ Run search
        st.markdown("### 3️⃣ Run search")

        if st.button("Run semantic search", type="primary"):
            if not effective_query:
                st.warning(
                    "Please either pick a scenario from the dropdown **or** type your own question."
                )
            else:
                with st.spinner("Searching…"):
                    try:
                        results = run_cached_semantic(
                            query=effective_query,
                            top_k=top_k,
                            threshold=threshold,
                            year=year,
                            keyword=(keyword.strip() or None),
                        )
                    except Exception as e:
                        st.error(f"Search failed: {e}")
                        results = []

                if not results:
                    st.info(
                        "No matches (try lowering similarity threshold, removing the year filter, "
                        "or clearing the keyword filter)."
                    )
                else:
                    res_df = pd.DataFrame(results)
                    if "similarity" in res_df.columns:
                        res_df["similarity"] = res_df["similarity"].map(
                            lambda x: round(float(x), 3)
                        )
                    keep_cols = [
                        c
                        for c in [
                            "title",
                            "fraud_type",
                            "date",
                            "similarity",
                            "url",
                            "summary",
                        ]
                        if c in res_df.columns
                    ]
                    st.success(f"Found {len(res_df)} matches")
                    st.dataframe(
                        res_df[keep_cols],
                        use_container_width=True,
                        height=420,
                    )

                    st.markdown("**Quick links (top 5):**")
                    for _, r in res_df.head(5).iterrows():
                        st.markdown(
                            f"- [{r.get('title','(no title)')}]({r.get('url','#')}) "
                            f"— sim {r.get('similarity','?')}"
                        )
# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
