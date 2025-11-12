# fraud_dashboard.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
from dotenv import load_dotenv
from supabase import create_client, Client
from semantic_search import search as semantic_search
f
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
# The semantic_search() already caches embeddings in Supabase,
# this just avoids redoing the RPC + filtering for the same UI inputs.
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

    # --------------------------------------------------------
    # Create Tabs
    # --------------------------------------------------------
    week2, week3, week4, semtab = st.tabs([
        "📄 Week 2: Scraper",
        "🔍 Week 3: Fraud Detection",
        "📊 Week 4: Analysis",
        "🔎 Semantic Search"
    ])

    # =======================
    # 🗞️ WEEK 2 — SCRAPER
    # =======================
    with week2:
        st.header("Week 2 — Build Scraper")
        st.write("Goal: Get articles from CFPB source and clean text for analysis.")
        st.write(f"**Total Articles Scraped:** {len(df)}")
        st.dataframe(df[["title", "date", "url"]].head(10), width="stretch")

        st.subheader("Example Article Text Preview")
        sample = df.sample(1).iloc[0]
        st.markdown(f"**Title:** {sample['title']}")
        st.markdown(f"**Date:** {sample['date'].date()}")
        st.markdown(f"**URL:** [Read Article]({sample['url']})")
        st.text_area("Excerpt", (sample.get("text") or "")[:600] + "...")

    # ==========================
    # 🔍 WEEK 3 — FRAUD DETECTION
    # ==========================
    with week3:
        st.header("Week 3 — Fraud Detection")
        st.write("Goal: Identify and tag fraud-related articles.")
        st.metric("Distinct Fraud Types", int(df["fraud_type"].dropna().nunique()))
        st.metric("Total Fraud-Tagged Articles", int((df["fraud_type"] != "not_fraud").sum()))

        st.subheader("Fraud Tags by Frequency")
        tags = df["fraud_tags"].dropna().tolist()
        tag_list = [t.strip() for tags_str in tags for t in tags_str.split(",") if t.strip()]
        tag_counts = Counter(tag_list)
        kw_df = pd.DataFrame(tag_counts.most_common(10), columns=["Keyword", "Count"])
        st.bar_chart(kw_df.set_index("Keyword"))
        st.dataframe(kw_df, width="stretch")

        st.subheader("Word Cloud of Fraud Keywords")
        if tag_list:
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tag_list))
            st.image(wordcloud.to_array(), width="stretch")

    # ==========================
    # 📊 WEEK 4 — ANALYSIS
    # ==========================
    with week4:
        st.header("Week 4 — Full Pipeline & Trend Analysis")
        st.write("Goal: Complete pipeline (scrape → detect → summarize) and identify top trends.")

    # ---------- Filters ----------
        col1, col2 = st.columns(2)
        fraud_types = df["fraud_type"].dropna().unique().tolist()
        selected_types = col1.multiselect("Filter by Fraud Type", fraud_types, default=fraud_types)
        date_min, date_max = df["date"].min().date(), df["date"].max().date()
        date_range = col2.slider("Select Date Range", date_min, date_max, (date_min, date_max))

    # Optional Year Filter
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
                step=1
        )

    # Apply filters
    filtered_df = df[
        (df["fraud_type"].isin(selected_types)) &
        (df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    ]
    if year:
        filtered_df = filtered_df[df["date"].dt.year == year]


        # Overview
        c3, c4, c5 = st.columns(3)
        c3.metric("Articles in Range", len(filtered_df))
        c4.metric("Fraud Types Displayed", len(selected_types))
        c5.metric("Date Range", f"{date_range[0]} → {date_range[1]}")

        # Top Keywords
        st.subheader("🔑 Top Keywords / Phrases")
        if not filtered_df.empty:
            tags = filtered_df["fraud_tags"].dropna().tolist()
            all_tags = [t.strip() for tags_str in tags for t in tags_str.split(",") if t.strip()]
            tag_counts = Counter(all_tags)
            kw_df = pd.DataFrame(tag_counts.most_common(10), columns=["Keyword", "Count"])
            fig_kw = px.bar(kw_df, x="Keyword", y="Count", title="Top Fraud Keywords / Phrases", color="Count")
            st.plotly_chart(fig_kw, width="stretch")

        # Top Fraud Types
        st.subheader("🚨 Top Fraud Types")
        fraud_trend = (
            filtered_df.groupby("fraud_type")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="Count")
        )
        if not fraud_trend.empty:
            fig_ft = px.bar(fraud_trend, x="fraud_type", y="Count", color="Count", title="Fraud Type Frequency")
            st.plotly_chart(fig_ft, width="stretch")

                # ==========================
        # GitHub-Style Fraud Activity Calendar (fully proportional + full-width + interactive)
        # ==========================
        from datetime import date, timedelta
        import calendar
        import streamlit.components.v1 as components

        st.subheader("GitHub-Style Fraud Activity")

        # --- Build 1-year range and fraud counts/types ---
        today = date.today()
        start_date = today - timedelta(days=365)
        all_days = pd.date_range(start=start_date, end=today, freq="D")

        # Group articles by day and collect fraud types
        day_types = (
            df.groupby(df["date"].dt.date)["fraud_type"]
            .agg(list)
            .to_dict()
        )

        # --- Fixed color palette for four fraud types ---
        fraud_colors = {
            "generic": "#39d353",          # bright green
            "wire_transfer": "#26a641",    # dark green
            "identity_theft": "#2188ff",   # blue
            "money_laundering": "#ff7b72", # orange-red
        }

        def color_for_day(types):
            if not types:
                return "#161b22"  # empty background
            t = types[0]
            return fraud_colors.get(t, "#8b949e")  # gray fallback

        # --- CSS and HTML Grid ---
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
            grid-template-columns: repeat(53, 1fr);
            grid-auto-rows: 14px;
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
            font-size: 10px;
            color: #8b949e;
            margin-left: 24px;
            margin-right: 10px;
            width: calc(100% - 34px);
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
        # Month labels (Jan–Dec)
        for m in range(1, 13):
            html += f"<span>{calendar.month_abbr[m]}</span>"
        html += "</div><div class='calendar-container'><div class='calendar-wrapper'>"

        # Dynamically render Mon/Wed/Fri on alternating rows
        weekday_labels = ["Mon", "Wed", "Fri"]
        html += "<div class='weekday-labels'>"
        for i in range(7):
            # Only show Mon/Wed/Fri on alternating (odd) rows
            label = weekday_labels[i % 3] if i % 2 == 1 else ""
            html += f"<span>{label}</span>"
        html += "</div>"

        # Main 53x7 grid
        html += "<div class='calendar' id='fraudCalendar'>"
        for d in all_days:
            types = day_types.get(d.date(), [])
            color = color_for_day(types)
            title = f"{d.date()}: {'None' if not types else ', '.join(types)}"
            html += f"<div class='daybox' title='{title}' data-day='{d.date()}' style='background:{color};'></div>"
        html += "</div></div></div>"

        # Color legend
        html += "<div class='legend'>"
        for ftype, color in fraud_colors.items():
            label = ftype.replace('_', ' ').title()
            html += f"<span class='legend-box' style='background:{color};'></span>{label}"
        html += "<span class='legend-box' style='background:#161b22;'></span>None"
        html += "</div>"

        # JS click → update Streamlit query params (inline update)
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

        # JS listener to sync with Streamlit query params
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

        # Drill-down section (same as before)
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

        
        # --- Daily Trending Fraud Type (GitHub-style summary) ---
        st.subheader("🔥 Today's Trending Fraud Type")
        if not df.empty:
            today = pd.Timestamp.now().normalize()
            daily_df = df[df["date"].dt.date == today.date()]

            if daily_df.empty:
                st.warning("No fraud articles found for today.")
            else:
                top_today = (
                    daily_df["fraud_type"]
                    .value_counts()
                    .reset_index()
                    .rename(columns={"index": "fraud_type", "fraud_type": "count"})
                )
                most_common_type = top_today.iloc[0]["fraud_type"]
                count_today = top_today.iloc[0]["count"]

                st.markdown(f"""
                <div style='
                    display:flex;
                    align-items:center;
                    background-color:#0d1117;
                    border:1px solid #30363d;
                    border-radius:8px;
                    padding:10px 16px;
                    margin-top:12px;
                '>
                    <img src='https://img.icons8.com/ios-filled/50/79c0ff/fire-element.png' width='26' style='margin-right:10px;'/>
                    <span style='font-size:1.1em;color:#79c0ff;font-weight:600'>
                        🔥 Today's trending fraud type: <b style='color:#58a6ff'>{most_common_type.replace("_"," ").title()}</b>
                        &nbsp;({count_today} mentions)
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # Recent Articles
        st.subheader("📰 Recent Fraud Articles")
        for _, row in filtered_df.head(5).iterrows():
            st.markdown(
                f"**{row['title']}** — *{row['fraud_type']}* ({row['date'].date()})  \n"
                f"{row.get('summary','')}  \n[Read More]({row['url']})"
            )

        # Download
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Filtered Data as CSV", data=csv, file_name="fraud_analysis_filtered.csv")

    # ==========================
    # 🔎 SEMANTIC SEARCH (embeddings)
    # ==========================
    with semtab:
        st.header("Semantic Search — Embedding Powered")
        st.write("Type a natural-language query. We embed it with OpenAI and run pgvector similarity in Supabase.")

        colA, colB = st.columns([3, 1])
        query = colA.text_input("Search query", placeholder="e.g., zelle unauthorized transfer")
        top_k = colB.slider("Top K", 5, 40, 10, step=5)

        colC, colD = st.columns([1, 1])
        threshold = colC.number_input("Min similarity (0–1)", 0.0, 1.0, 0.55, 0.01)
        enable_year = colD.toggle("Filter by year", value=False)
        year = None
        if enable_year:
    # pick a safe default from your data if you like; 2020 is fine fallback
            default_year = 2020
            year = colD.number_input("Year", min_value=1900, max_value=2100, value=default_year, step=1)

        keyword = st.text_input("Optional keyword filter (title contains)")

        if st.button("Run semantic search", type="primary"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Searching…"):
                    try:
                        results = run_cached_semantic(
                            query=query.strip(),
                            top_k=top_k,
                            threshold=threshold,
                            year=year,
                            keyword=(keyword.strip() or None),
                        )
                    except Exception as e:
                        st.error(f"Search failed: {e}")
                        results = []

                if not results:
                    st.info("No matches (try lowering similarity threshold or removing filters).")
                else:
                    res_df = pd.DataFrame(results)
                    # format similarity nicely if present
                    if "similarity" in res_df.columns:
                        res_df["similarity"] = res_df["similarity"].map(lambda x: round(float(x), 3))
                    keep_cols = [c for c in ["title","fraud_type","date","similarity","url","summary"] if c in res_df.columns]
                    st.success(f"Found {len(res_df)} matches")
                    st.dataframe(res_df[keep_cols], width="stretch", height=420)

                    # quick links
                    for _, r in res_df.head(5).iterrows():
                        st.markdown(f"- [{r.get('title','(no title)')}]({r.get('url','#')}) — sim {r.get('similarity','?')}")

# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------
if __name__ == "__main__":
    main()