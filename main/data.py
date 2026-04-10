import duckdb
from datasets import load_dataset
from main.config import (
    SEP_TOKEN, DATASET_NAME, DATASET_TEXT_COL, DATASET_SCORE_COL,
    DUCKLAKE_CATALOG, DUCKLAKE_TABLES, LAKE_DIR,
)
import os


def load_dataset_splits():
    ds = load_dataset(DATASET_NAME, split="train")
    records = []
    for row in ds:
        text = row[DATASET_TEXT_COL]
        parts = text.split(SEP_TOKEN)
        if len(parts) != 2:
            continue
        cv, job = parts[0].strip(), parts[1].strip()
        score = float(row[DATASET_SCORE_COL])
        records.append({"cv": cv, "job": job, "ats_score": score})
    return records


def init_ducklake(con):
    con.execute("INSTALL ducklake;")
    con.execute("LOAD ducklake;")
    os.makedirs(LAKE_DIR, exist_ok=True)
    con.execute(f"ATTACH 'ducklake:{DUCKLAKE_CATALOG}' AS lake;")
    con.execute("USE lake.main;")
    for table_sql in DUCKLAKE_TABLES.values():
        con.execute(table_sql)


def store_documents(con, documents):
    for doc in documents:
        con.execute(
            "INSERT INTO lake.main.documents (id, type, text) VALUES (?, ?, ?)",
            [doc["id"], doc["type"], doc["text"]],
        )


def store_pairs(con, pairs):
    for p in pairs:
        con.execute(
            "INSERT INTO lake.main.pairs (cv_id, job_id, label, source) VALUES (?, ?, ?, ?)",
            [p["cv_id"], p["job_id"], p["label"], p["source"]],
        )


def get_documents(con, doc_type=None):
    if doc_type:
        result = con.execute(
            "SELECT * FROM lake.main.documents WHERE type = ?", [doc_type]
        )
    else:
        result = con.execute("SELECT * FROM lake.main.documents")
    cols = [desc[0] for desc in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]


def get_pairs(con, source=None):
    if source:
        result = con.execute(
            "SELECT * FROM lake.main.pairs WHERE source = ?", [source]
        )
    else:
        result = con.execute("SELECT * FROM lake.main.pairs")
    cols = [desc[0] for desc in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]
