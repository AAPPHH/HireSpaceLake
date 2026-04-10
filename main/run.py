import duckdb
import subprocess
import sys
from main.config import DUCKLAKE_CATALOG, EMBED_CONFIG, APP_CONFIG
from main.data import load_dataset_splits, init_ducklake, store_documents, store_pairs
from main.embed import load_model, update_embeddings
from main.cluster import cluster_documents, generate_pairs_from_clusters
from main.train import train, setup_lora_model
from main.evaluate import evaluate_model, compare_models


def run_pipeline():
    con = duckdb.connect()
    init_ducklake(con)

    print("[1/6] Loading dataset")
    records = load_dataset_splits()
    cv_docs = []
    job_docs = []
    dataset_pairs = []
    for i, rec in enumerate(records):
        cv_id = i * 2
        job_id = i * 2 + 1
        cv_docs.append({"id": cv_id, "type": "cv", "text": rec["cv"]})
        job_docs.append({"id": job_id, "type": "job", "text": rec["job"]})
        dataset_pairs.append({
            "cv_id": cv_id, "job_id": job_id,
            "label": rec["ats_score"], "source": "dataset",
        })
    existing = con.execute(
        "SELECT COUNT(*) FROM lake.main.documents"
    ).fetchone()[0]
    if existing == 0:
        store_documents(con, cv_docs + job_docs)
        store_pairs(con, dataset_pairs)
        print(f"  Stored {len(cv_docs)} CVs, {len(job_docs)} Jobs, {len(dataset_pairs)} pairs")
    else:
        print(f"  Documents already loaded ({existing} rows)")

    print("[2/6] Frozen baseline embeddings")
    model, tokenizer = load_model()
    update_embeddings(con, model, tokenizer, column="raw_embedding")
    print("  Raw embeddings computed")

    print("[3/6] Evaluating frozen baseline")
    frozen_metrics = evaluate_model(con, model, tokenizer, embedding_column="raw_embedding")
    for k, v in frozen_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("[4/6] Clustering + pair generation")
    cluster_documents(con, embedding_column="raw_embedding")
    generated = generate_pairs_from_clusters(con)
    print(f"  Generated {len(generated)} cluster-based pairs")

    print("[5/6] LoRA finetuning")
    lora_model = train(con)
    if lora_model is not None:
        update_embeddings(con, model, tokenizer, column="lora_embedding")
        print("  LoRA embeddings computed")

        print("[6/6] Evaluating LoRA model")
        lora_metrics = evaluate_model(con, model, tokenizer, embedding_column="lora_embedding")
        for k, v in lora_metrics.items():
            print(f"  {k}: {v:.4f}")
        compare_models(con, frozen_metrics, lora_metrics)
    else:
        print("[6/6] Skipped LoRA evaluation (no model)")

    con.close()
    print("Pipeline complete")
    return frozen_metrics, lora_metrics if lora_model else {}


def launch_app():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "main/app.py", "--server.port", str(APP_CONFIG["port"]),
    ])


if __name__ == "__main__":
    frozen, lora = run_pipeline()
    answer = input("Launch Streamlit demo? [y/N] ")
    if answer.strip().lower() == "y":
        launch_app()
