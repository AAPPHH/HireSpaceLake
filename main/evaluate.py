import json
import numpy as np
from main.config import EVAL_CONFIG, TRAIN_CONFIG
from main.data import get_documents, get_pairs
from main.embed import embed_documents


def compute_recall_at_k(query_embeddings, doc_embeddings, labels, k):
    sims = query_embeddings @ doc_embeddings.T
    recall_sum = 0.0
    for i in range(len(query_embeddings)):
        top_k_indices = np.argsort(sims[i])[::-1][:k]
        relevant = set(np.where(labels[i] > 0)[0])
        if not relevant:
            continue
        hits = len(relevant.intersection(set(top_k_indices)))
        recall_sum += hits / min(len(relevant), k)
    return recall_sum / len(query_embeddings)


def compute_mrr(query_embeddings, doc_embeddings, labels):
    sims = query_embeddings @ doc_embeddings.T
    mrr_sum = 0.0
    for i in range(len(query_embeddings)):
        ranked_indices = np.argsort(sims[i])[::-1]
        relevant = set(np.where(labels[i] > 0)[0])
        for rank, idx in enumerate(ranked_indices, 1):
            if idx in relevant:
                mrr_sum += 1.0 / rank
                break
    return mrr_sum / len(query_embeddings)


def compute_ndcg(query_embeddings, doc_embeddings, labels, k):
    sims = query_embeddings @ doc_embeddings.T
    ndcg_sum = 0.0
    for i in range(len(query_embeddings)):
        top_k_indices = np.argsort(sims[i])[::-1][:k]
        dcg = 0.0
        for rank, idx in enumerate(top_k_indices):
            dcg += labels[i][idx] / np.log2(rank + 2)
        ideal_gains = np.sort(labels[i])[::-1][:k]
        idcg = 0.0
        for rank, g in enumerate(ideal_gains):
            idcg += g / np.log2(rank + 2)
        if idcg > 0:
            ndcg_sum += dcg / idcg
        else:
            ndcg_sum += 0.0
    return ndcg_sum / len(query_embeddings)


def evaluate_model(con, model, tokenizer, embedding_column="raw_embedding"):
    cvs = get_documents(con, doc_type="cv")
    jobs = get_documents(con, doc_type="job")
    if not cvs or not jobs:
        return {}
    pairs = get_pairs(con)
    pair_set = {}
    for p in pairs:
        pair_set[(p["cv_id"], p["job_id"])] = float(p["label"])
    cv_texts = [c["text"] for c in cvs]
    job_texts = [j["text"] for j in jobs]
    config = {"batch_size": EVAL_CONFIG["batch_size"], "normalize": True, "device": "cuda"}
    query_embeddings = embed_documents(cv_texts, model, tokenizer, config=config, query=True)
    doc_embeddings = embed_documents(job_texts, model, tokenizer, config=config, query=False)
    n_queries = len(cvs)
    n_docs = len(jobs)
    labels = np.zeros((n_queries, n_docs))
    for qi, cv in enumerate(cvs):
        for di, job in enumerate(jobs):
            key = (cv["id"], job["id"])
            if key in pair_set:
                labels[qi][di] = pair_set[key]
    results = {}
    for k in EVAL_CONFIG["k_values"]:
        if "recall" in EVAL_CONFIG["metrics"]:
            results[f"recall@{k}"] = compute_recall_at_k(query_embeddings, doc_embeddings, labels, k)
        if "ndcg" in EVAL_CONFIG["metrics"]:
            results[f"ndcg@{k}"] = compute_ndcg(query_embeddings, doc_embeddings, labels, k)
    if "mrr" in EVAL_CONFIG["metrics"]:
        results["mrr"] = compute_mrr(query_embeddings, doc_embeddings, labels)
    return results


def compare_models(con, frozen_metrics, lora_metrics):
    print(f"{'Metric':<15} {'Frozen':>10} {'LoRA':>10} {'Delta':>10}")
    print("-" * 47)
    all_keys = sorted(set(list(frozen_metrics.keys()) + list(lora_metrics.keys())))
    for key in all_keys:
        fv = frozen_metrics.get(key, 0.0)
        lv = lora_metrics.get(key, 0.0)
        delta = lv - fv
        sign = "+" if delta >= 0 else ""
        print(f"{key:<15} {fv:>10.4f} {lv:>10.4f} {sign}{delta:>9.4f}")
    comparison = {
        "frozen": frozen_metrics,
        "lora": lora_metrics,
        "delta": {k: lora_metrics.get(k, 0) - frozen_metrics.get(k, 0) for k in all_keys},
    }
    try:
        max_id = con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM lake.main.experiments"
        ).fetchone()[0]
    except Exception:
        max_id = 0
    con.execute(
        "INSERT INTO lake.main.experiments (id, model_version, metrics_json, config_json) VALUES (?, ?, ?, ?)",
        [max_id + 1, "comparison_frozen_vs_lora", json.dumps(comparison), json.dumps(EVAL_CONFIG)],
    )
    return comparison
