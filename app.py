import json
import streamlit as st
import duckdb
import numpy as np
from config import ACTIVE_MODEL
from main import init_ducklake, load_model, embed_documents


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_connection():
    con = duckdb.connect()
    init_ducklake(con)
    return con


# ── Tab: Match ──────────────────────────────────────────────────────────────────

def tab_match(con, model):
    direction = st.radio("Richtung", ["Job → Passende CVs", "CV → Passende Jobs"])
    top_k = st.slider("Top-K", 1, 50, 10)
    query_text = st.text_area("Query eingeben")

    if st.button("Match"):
        if not query_text.strip():
            st.warning("Bitte Query eingeben.")
            return

        query_emb = embed_documents([query_text], model, query=True)
        query_emb_list = query_emb[0].tolist()
        target_type = "cv" if "CVs" in direction else "job"
        model_version = ACTIVE_MODEL["name"]

        results = con.execute(
            """
            SELECT d.id, d.doc_type, d.text,
                   list_cosine_similarity(e.embedding, ?::FLOAT[]) AS similarity
            FROM lake.main.embeddings e
            JOIN lake.main.documents d ON e.doc_id = d.id
            WHERE d.doc_type = ? AND e.model_version = ?
            ORDER BY similarity DESC
            LIMIT ?
            """,
            [query_emb_list, target_type, model_version, top_k],
        ).fetchall()

        if not results:
            st.info("Keine Ergebnisse gefunden.")
            return

        for rank, row in enumerate(results, 1):
            doc_id, doc_type, text, similarity = row
            truncated = text[:500] + "..." if len(text) > 500 else text
            with st.expander(f"Rank {rank} | ID {doc_id} | Similarity: {similarity:.4f}"):
                st.text(truncated)


# ── Tab: Explore ────────────────────────────────────────────────────────────────

def tab_explore(con):
    model_version = ACTIVE_MODEL["name"]

    count = con.execute(
        "SELECT COUNT(*) FROM lake.main.clusters WHERE model_version = ?",
        [model_version],
    ).fetchone()[0]

    if count == 0:
        st.info("Keine Cluster vorhanden. Pipeline mit cluster=True ausfuehren.")
        return

    rows = con.execute(
        """
        SELECT e.embedding, c.cluster_id, d.doc_type
        FROM lake.main.clusters c
        JOIN lake.main.embeddings e ON c.doc_id = e.doc_id AND c.model_version = e.model_version
        JOIN lake.main.documents d ON c.doc_id = d.id
        WHERE c.model_version = ?
        """,
        [model_version],
    ).fetchall()

    if not rows:
        st.info("Keine Daten fuer Visualisierung.")
        return

    embeddings = np.array([row[0] for row in rows])
    cluster_ids = [row[1] for row in rows]
    doc_types = [row[2] for row in rows]

    try:
        from umap import UMAP
    except ImportError:
        st.warning("umap-learn nicht installiert. Installieren mit: pip install umap-learn")
        return

    reducer = UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)

    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": [str(c) for c in cluster_ids],
            "type": doc_types,
        })
        fig = px.scatter(
            df, x="x", y="y", color="cluster", symbol="type",
            title="Embedding Clusters (UMAP)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))
        markers = {"cv": "o", "job": "^"}
        for dt in set(doc_types):
            mask = [i for i, t in enumerate(doc_types) if t == dt]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[cluster_ids[i] for i in mask],
                marker=markers.get(dt, "o"),
                label=dt, alpha=0.7,
            )
        ax.legend()
        ax.set_title("Embedding Clusters (UMAP)")
        st.pyplot(fig)


# ── Tab: Metrics ────────────────────────────────────────────────────────────────

def tab_metrics(con):
    rows = con.execute(
        "SELECT id, model_version, config, metrics, created_at FROM lake.main.experiments ORDER BY created_at DESC"
    ).fetchall()

    if not rows:
        st.info("Keine Experiments vorhanden.")
        return

    table_data = []
    for row in rows:
        exp_id, model_ver, config_raw, metrics_raw, created_at = row
        metrics = json.loads(metrics_raw) if isinstance(metrics_raw, str) else metrics_raw
        entry = {
            "id": exp_id,
            "model_version": model_ver,
            "created_at": str(created_at),
        }
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                entry[k] = v
        table_data.append(entry)

    st.dataframe(table_data)

    if len(rows) > 1:
        recall_keys = [k for k in table_data[0] if k.startswith("recall")]
        if recall_keys:
            import pandas as pd
            chart_data = pd.DataFrame([
                {
                    "model_version": d["model_version"],
                    **{k: d.get(k, 0) for k in recall_keys},
                }
                for d in table_data
            ]).set_index("model_version")
            st.bar_chart(chart_data)


# ── Tab: Data ───────────────────────────────────────────────────────────────────

def tab_data(con):
    st.subheader("Documents")
    try:
        doc_counts = con.execute(
            "SELECT doc_type, COUNT(*) AS cnt FROM lake.main.documents GROUP BY doc_type"
        ).fetchdf()
        st.dataframe(doc_counts)
    except Exception:
        st.info("Keine Documents vorhanden.")

    st.subheader("Pairs")
    try:
        pair_counts = con.execute(
            "SELECT label_source, split, COUNT(*) AS cnt FROM lake.main.pairs GROUP BY label_source, split"
        ).fetchdf()
        st.dataframe(pair_counts)
    except Exception:
        st.info("Keine Pairs vorhanden.")

    st.subheader("Embeddings")
    try:
        emb_counts = con.execute(
            "SELECT model_version, COUNT(*) AS cnt FROM lake.main.embeddings GROUP BY model_version"
        ).fetchdf()
        st.dataframe(emb_counts)
    except Exception:
        st.info("Keine Embeddings vorhanden.")

    st.subheader("Clusters")
    try:
        cluster_counts = con.execute(
            "SELECT model_version, COUNT(*) AS cnt FROM lake.main.clusters GROUP BY model_version"
        ).fetchdf()
        st.dataframe(cluster_counts)
    except Exception:
        st.info("Keine Clusters vorhanden.")

    st.subheader("SQL Query")
    sql = st.text_area("Custom SQL", key="custom_sql")
    if st.button("Execute"):
        if not sql.strip():
            st.warning("Bitte SQL eingeben.")
            return
        try:
            result = con.execute(sql).fetchdf()
            st.dataframe(result)
        except Exception as e:
            st.error(str(e))


# ── Entry Point ─────────────────────────────────────────────────────────────────

def run_app():
    st.set_page_config(page_title="TalentLake - CV-Job Matching", layout="wide")
    st.title("TalentLake - CV-Job Matching")
    model = get_model()
    con = get_connection()
    tabs = st.tabs(["Match", "Explore", "Metrics", "Data"])
    with tabs[0]:
        tab_match(con, model)
    with tabs[1]:
        tab_explore(con)
    with tabs[2]:
        tab_metrics(con)
    with tabs[3]:
        tab_data(con)


if __name__ == "__main__":
    run_app()
