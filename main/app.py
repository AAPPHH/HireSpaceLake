import streamlit as st
import duckdb
from main.config import APP_CONFIG, DUCKLAKE_CATALOG, EMBED_CONFIG, LAKE_DIR
from main.embed import load_model, embed_documents
from main.data import init_ducklake
import os


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_connection():
    con = duckdb.connect()
    init_ducklake(con)
    return con


def run_app():
    st.set_page_config(page_title=APP_CONFIG["title"], layout="wide")
    st.title(APP_CONFIG["title"])
    model, tokenizer = get_model()
    con = get_connection()
    mode = st.selectbox("Suchmodus", ["Finde passende Jobs", "Finde passende CVs"])
    top_k = st.slider("Top-K Ergebnisse", min_value=1, max_value=50, value=APP_CONFIG["top_k"])
    query_text = st.text_area(
        "CV eingeben" if mode == "Finde passende Jobs" else "Job Description eingeben",
        height=300,
    )
    if st.button("Matching starten") and query_text.strip():
        query_emb = embed_documents(
            [query_text.strip()], model, tokenizer, config=EMBED_CONFIG, query=True
        )
        emb_list = query_emb[0].tolist()
        target_type = "job" if mode == "Finde passende Jobs" else "cv"
        embedding_col = "lora_embedding"
        fallback = con.execute(
            f"SELECT COUNT(*) FROM lake.main.documents WHERE type = ? AND {embedding_col} IS NOT NULL",
            [target_type],
        ).fetchone()[0]
        if fallback == 0:
            embedding_col = "raw_embedding"
        results = con.execute(
            f"""
            SELECT id, text,
                   array_cosine_similarity({embedding_col}, ?::FLOAT[1024]) AS similarity
            FROM lake.main.documents
            WHERE type = ? AND {embedding_col} IS NOT NULL
            ORDER BY similarity DESC
            LIMIT ?
            """,
            [emb_list, target_type, top_k],
        ).fetchall()
        if not results:
            st.warning("Keine Ergebnisse gefunden.")
        else:
            for rank, (doc_id, text, sim) in enumerate(results, 1):
                with st.expander(f"#{rank} — Similarity: {sim:.4f} (ID: {doc_id})"):
                    st.text(text[:2000])


if __name__ == "__main__":
    run_app()
