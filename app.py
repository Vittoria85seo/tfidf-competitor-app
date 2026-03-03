import streamlit as st
import pandas as pd
import plotly.express as px

from scraper import scrape_url
from processor import compute_tfidf
from translator import has_non_english, translate_terms

st.set_page_config(
    page_title="TF-IDF Competitor Analysis",
    page_icon="🔍",
    layout="wide",
)

st.title("TF-IDF Competitor Analysis")
st.markdown(
    "Compare keyword usage between your page and up to 10 competitors. "
    "**First line = your URL**, remaining lines = competitor URLs."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    presence_pct = st.slider(
        "Min. competitor presence (%)",
        min_value=10, max_value=80, value=30, step=5,
        help="Only show terms found on at least this % of competitor pages",
    )

    custom_sw = st.text_input(
        "Extra stopwords to exclude (comma-separated)",
        placeholder="e.g. brand, cityname, promo",
    )

    top_n = st.slider("Max results to display", 10, 100, 50, step=10)

    st.markdown("---")
    st.caption(
        "Red rows = competitors mention this term more than you.  \n"
        "Green rows = you mention this term more than competitors."
    )

# ── URL input ─────────────────────────────────────────────────────────────────
url_input = st.text_area(
    "Paste URLs — one per line",
    height=220,
    placeholder=(
        "https://yoursite.com/page\n"
        "https://competitor1.com/page\n"
        "https://competitor2.com/page\n"
        "..."
    ),
)

run = st.button("Run Analysis", type="primary", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if run:
    urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]

    if len(urls) < 2:
        st.error("Please enter at least 2 URLs (your URL + at least 1 competitor).")
        st.stop()

    my_url = urls[0]
    all_urls = [my_url] + urls[1:11]  # cap at 10 competitors

    # Scraping
    st.markdown("---")
    progress = st.progress(0)
    status = st.empty()
    scraped = []

    for idx, url in enumerate(all_urls):
        label = "Your page" if idx == 0 else f"Competitor {idx}"
        status.text(f"Fetching {label}: {url}")
        scraped.append(scrape_url(url))
        progress.progress((idx + 1) / len(all_urls))

    progress.empty()
    status.empty()

    # Surface scrape errors
    for d in scraped:
        if d["error"]:
            tag = "Your page" if d["url"] == my_url else "Competitor"
            st.warning(f"Scrape issue — {tag}: {d['url']}\n{d['error']}")

    # TF-IDF
    custom_words = [w.strip() for w in custom_sw.split(",") if w.strip()] if custom_sw else None
    with st.spinner("Computing TF-IDF scores…"):
        try:
            df = compute_tfidf(
                scraped,
                my_url=my_url,
                presence_threshold=presence_pct / 100,
                custom_stopwords=custom_words,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if df.empty:
        st.warning(
            "No terms passed the filters. "
            "Try lowering the competitor presence slider or check that your URLs scraped successfully."
        )
        st.stop()

    df = df.head(top_n)

    # Translation
    with st.spinner("Detecting languages…"):
        non_english = has_non_english(scraped)

    if non_english:
        with st.spinner("Translating non-English terms…"):
            translations = translate_terms(df["Keyword / Phrase"].tolist())
            df.insert(1, "English Translation", df["Keyword / Phrase"].map(translations))

    # Cache in session state
    st.session_state["df"] = df
    st.session_state["my_url"] = my_url
    st.session_state["scraped"] = scraped

# ── Display ───────────────────────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state["df"]
    my_url = st.session_state["my_url"]

    st.markdown("---")

    # Summary metrics
    n_missing = int((df["Found In My Page"] == "No").sum())
    n_underused = int(
        ((df["Found In My Page"] == "Yes") &
        (df["Avg Mentions (Competitors)"] > df["Mentions (My Page)"])).sum()
    )
    n_strong = int(
        (df["Mentions (My Page)"] >= df["Avg Mentions (Competitors)"]).sum()
    )
    n_total = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Terms analysed", n_total)
    c2.metric("Missing from your page", n_missing)
    c3.metric("Underused vs competitors", n_underused)
    c4.metric("You outperform competitors", n_strong)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Full Results", "Gap Analysis", "Chart"])

    # Helper: strip internal column
    def clean(frame):
        return frame.drop(columns=["_opportunity"], errors="ignore").copy()

    fmt = {"% Competitors Using": "{:.1f}%"}

    # Row colouring: red = competitors mention more, green = you mention more
    def _color_row(row):
        if row.get("Mentions (My Page)", 0) < row.get("Avg Mentions (Competitors)", 0):
            return ["background-color: #ffe0e0"] * len(row)
        if row.get("Mentions (My Page)", 0) > row.get("Avg Mentions (Competitors)", 0):
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    with tab1:
        st.subheader("Full Results")
        display = clean(df)
        styled = display.style.apply(_color_row, axis=1).format(fmt)
        st.dataframe(styled, use_container_width=True, height=520)

        csv = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="tfidf_analysis.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Missing terms — not found on your page at all")
        missing = clean(df[df["Found In My Page"] == "No"])
        if missing.empty:
            st.success("No gaps — your page covers all high-frequency competitor terms.")
        else:
            st.dataframe(
                missing.style.apply(_color_row, axis=1).format(fmt),
                use_container_width=True, height=400,
            )

        st.subheader("Underused terms — present but mentioned less than competitors")
        underused = clean(df[
            (df["Found In My Page"] == "Yes") &
            (df["Avg Mentions (Competitors)"] > df["Mentions (My Page)"])
        ])
        if underused.empty:
            st.info("No underused terms found.")
        else:
            st.dataframe(
                underused.style.apply(_color_row, axis=1).format(fmt),
                use_container_width=True, height=350,
            )

    with tab3:
        st.subheader("Top 20 terms by competitor usage")
        chart_df = df.head(20).copy()
        kw_col = "English Translation" if "English Translation" in chart_df.columns else "Keyword / Phrase"
        # Fall back to original phrase if English Translation is empty
        chart_df["label"] = chart_df.apply(
            lambda r: r[kw_col] if r.get(kw_col) else r["Keyword / Phrase"], axis=1
        ).str[:35]

        # Melt into long format so we get two bars per term
        chart_long = chart_df[["label", "Mentions (My Page)", "Avg Mentions (Competitors)"]].melt(
            id_vars="label", var_name="Source", value_name="Mentions"
        )
        fig = px.bar(
            chart_long,
            x="Mentions",
            y="label",
            color="Source",
            orientation="h",
            barmode="group",
            color_discrete_map={
                "Mentions (My Page)": "#3498db",
                "Avg Mentions (Competitors)": "#e74c3c",
            },
            labels={"label": "Keyword", "Mentions": "Times mentioned"},
            title="Your page (blue) vs competitor average (red)",
        )
        fig.update_layout(yaxis={"autorange": "reversed"}, height=600, legend_title="")
        st.plotly_chart(fig, use_container_width=True)
