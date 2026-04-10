import os
import json
import numpy as np
import torch
import duckdb
import datasets

from config import (
    ACTIVE_MODEL, LORA, LAKE, CLUSTER, EVAL,
    SEP_TOKEN, DATASET_NAME, DATASET_FALLBACK, DUCKLAKE_TABLES,
)

# --- DUCKLAKE ---

def init_ducklake(con):
    os.makedirs(LAKE["data_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(LAKE["catalog"]), exist_ok=True)
    con.execute("INSTALL ducklake")
    con.execute("LOAD ducklake")
    con.execute(f"ATTACH 'ducklake:{LAKE['catalog']}' AS lake")
    con.execute("USE lake.main")
    for table_name, ddl in DUCKLAKE_TABLES.items():
        con.execute(ddl)

def get_connection():
    con = duckdb.connect()
    init_ducklake(con)
    return con

# --- DATA ---

def load_data(con):
    count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if count > 0:
        print(f"[data] already loaded ({count} documents)")
        return {"documents": count, "status": "skipped"}

    try:
        ds = datasets.load_dataset(DATASET_NAME, split="train")
    except Exception:
        print(f"[data] fallback to {DATASET_FALLBACK}")
        ds = datasets.load_dataset(DATASET_FALLBACK, split="train")

    doc_rows = []
    pair_indices = []

    for i, row in enumerate(ds):
        text = row.get("text", "")
        parts = text.split(SEP_TOKEN)
        if len(parts) != 2:
            continue
        cv_text, job_text = parts[0].strip(), parts[1].strip()
        if not cv_text or not job_text:
            continue

        score = float(row.get("ats_score", 0.0))
        if score > 1.0:
            score = score / 100.0

        cv_id = i * 2
        job_id = i * 2 + 1
        doc_rows.append((cv_id, "cv", cv_text, "huggingface", None))
        doc_rows.append((job_id, "job", job_text, "huggingface", None))
        pair_indices.append((cv_id, job_id, score))

    rng = np.random.RandomState(42)
    indices = np.arange(len(pair_indices))
    rng.shuffle(indices)

    n = len(indices)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = np.empty(n, dtype=object)
    splits[indices[:train_end]] = "train"
    splits[indices[train_end:val_end]] = "val"
    splits[indices[val_end:]] = "test"

    for cv_id, doc_type, text, source, ts in doc_rows:
        con.execute(
            "INSERT INTO documents (id, doc_type, text, source, created_at) VALUES (?, ?, ?, ?, ?)",
            [cv_id, doc_type, text, source, ts],
        )

    for idx, (cv_id, job_id, score) in enumerate(pair_indices):
        con.execute(
            "INSERT INTO pairs (cv_id, job_id, label, label_source, split) VALUES (?, ?, ?, ?, ?)",
            [cv_id, job_id, score, "dataset", splits[idx]],
        )

    print(f"[data] loaded {len(doc_rows)} documents, {len(pair_indices)} pairs")
    return {"documents": len(doc_rows), "pairs": len(pair_indices)}

# --- EMBEDDING ---

def load_model():
    name = ACTIVE_MODEL["name"]
    if name.startswith("sentence-transformers/"):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)
    else:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        return (model, tokenizer)

def embed_documents(texts, model, query=False):
    from sentence_transformers import SentenceTransformer

    if isinstance(model, SentenceTransformer):
        prefix = ACTIVE_MODEL["instruction_prefix"] if query and ACTIVE_MODEL["instruction_prefix"] else ""
        if prefix:
            texts = [prefix + t for t in texts]
        embs = model.encode(
            texts,
            batch_size=ACTIVE_MODEL["batch_size"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embs, dtype=np.float32)

    transformer_model, tokenizer = model
    device = next(transformer_model.parameters()).device
    batch_size = ACTIVE_MODEL["batch_size"]
    max_length = ACTIVE_MODEL["max_length"]

    if query and ACTIVE_MODEL["instruction_prefix"]:
        texts = [ACTIVE_MODEL["instruction_prefix"] + t for t in texts]

    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = transformer_model(**encoded)
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_embs = output.last_hidden_state * attention_mask
        summed = token_embs.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        norms = mean_pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
        normalized = mean_pooled / norms
        all_embs.append(normalized.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)

def embed_all(con, model):
    model_version = ACTIVE_MODEL["name"]
    rows = con.execute(
        "SELECT d.id, d.text FROM documents d "
        "WHERE d.id NOT IN (SELECT doc_id FROM embeddings WHERE model_version = ?)",
        [model_version],
    ).fetchall()

    if not rows:
        print(f"[embed] all documents already embedded for {model_version}")
        return 0

    doc_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    batch_size = ACTIVE_MODEL["batch_size"]
    total = 0

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_ids = doc_ids[start:start + batch_size]
        embs = embed_documents(batch_texts, model, query=False)
        for doc_id, emb in zip(batch_ids, embs):
            con.execute(
                "INSERT INTO embeddings (doc_id, model_version, embedding, created_at) VALUES (?, ?, ?, ?)",
                [doc_id, model_version, emb.tolist(), None],
            )
        total += len(batch_texts)

    print(f"[embed] embedded {total} documents with {model_version}")
    return total

# --- CLUSTERING ---

def cluster_documents(con):
    model_version = ACTIVE_MODEL["name"]
    rows = con.execute(
        "SELECT doc_id, embedding FROM embeddings WHERE model_version = ?",
        [model_version],
    ).fetchall()

    if not rows:
        print("[cluster] no embeddings found")
        return np.array([])

    doc_ids = [r[0] for r in rows]
    emb_matrix = np.array([r[1] for r in rows], dtype=np.float32)

    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=CLUSTER["min_cluster_size"],
        min_samples=CLUSTER["min_samples"],
        metric=CLUSTER["metric"],
    )
    labels = clusterer.fit_predict(emb_matrix)

    con.execute("DELETE FROM clusters WHERE model_version = ?", [model_version])

    cluster_ids_set = set(labels)
    cluster_ids_set.discard(-1)

    centroids = set()
    for cid in cluster_ids_set:
        mask = labels == cid
        cluster_embs = emb_matrix[mask]
        cluster_doc_ids = np.array(doc_ids)[mask]
        mean_vec = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - mean_vec, axis=1)
        centroid_idx = np.argmin(dists)
        centroids.add(int(cluster_doc_ids[centroid_idx]))

    for doc_id, label in zip(doc_ids, labels):
        con.execute(
            "INSERT INTO clusters (doc_id, model_version, cluster_id, is_centroid) VALUES (?, ?, ?, ?)",
            [doc_id, model_version, int(label), doc_id in centroids],
        )

    n_clusters = len(cluster_ids_set)
    n_noise = int(np.sum(labels == -1))
    print(f"[cluster] {n_clusters} clusters, {n_noise} noise points")
    return labels

def generate_pairs_from_clusters(con, num_negatives=3):
    model_version = ACTIVE_MODEL["name"]
    rows = con.execute(
        "SELECT c.doc_id, c.cluster_id, d.doc_type FROM clusters c "
        "JOIN documents d ON c.doc_id = d.id "
        "WHERE c.model_version = ? AND c.cluster_id >= 0",
        [model_version],
    ).fetchall()

    cluster_cvs = {}
    cluster_jobs = {}
    for doc_id, cluster_id, doc_type in rows:
        if doc_type == "cv":
            cluster_cvs.setdefault(cluster_id, []).append(doc_id)
        else:
            cluster_jobs.setdefault(cluster_id, []).append(doc_id)

    rng = np.random.RandomState(42)
    pairs = []
    all_cluster_ids = sorted(set(cluster_cvs.keys()) | set(cluster_jobs.keys()))

    for cid in all_cluster_ids:
        cvs = cluster_cvs.get(cid, [])
        jobs = cluster_jobs.get(cid, [])
        for cv_id in cvs:
            for job_id in jobs:
                pairs.append({"cv_id": cv_id, "job_id": job_id, "label": 1.0})

    all_jobs_by_cluster = {cid: jobs for cid, jobs in cluster_jobs.items() if jobs}
    other_cluster_ids = list(all_jobs_by_cluster.keys())

    for cid in all_cluster_ids:
        cvs = cluster_cvs.get(cid, [])
        neg_clusters = [c for c in other_cluster_ids if c != cid]
        if not neg_clusters:
            continue
        for cv_id in cvs:
            chosen = rng.choice(neg_clusters, size=min(num_negatives, len(neg_clusters)), replace=False)
            for neg_cid in chosen:
                neg_job = rng.choice(all_jobs_by_cluster[neg_cid])
                pairs.append({"cv_id": cv_id, "job_id": int(neg_job), "label": 0.0})

    for p in pairs:
        con.execute(
            "INSERT INTO pairs (cv_id, job_id, label, label_source, split) VALUES (?, ?, ?, ?, ?)",
            [p["cv_id"], p["job_id"], p["label"], "cluster", "train"],
        )

    print(f"[cluster] generated {len(pairs)} pairs from clusters")
    return pairs

# --- TRAINING ---

def train_model(con):
    print(f"[train] placeholder -- config: rank={LORA['rank']}, lr={LORA['lr']}, epochs={LORA['epochs']}")
    model_version = ACTIVE_MODEL["name"]
    exp_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM experiments").fetchone()[0]
    con.execute(
        "INSERT INTO experiments (id, model_version, config, metrics, created_at) VALUES (?, ?, ?, ?, ?)",
        [exp_id, model_version, json.dumps(LORA), json.dumps({"status": "placeholder"}), None],
    )

# --- EVALUATION ---

def compute_recall_at_k(query_emb, doc_emb, labels, k):
    sim = query_emb @ doc_emb.T
    n_queries = sim.shape[0]
    hits = 0
    for i in range(n_queries):
        top_k = np.argsort(sim[i])[::-1][:k]
        relevant = np.where(labels[i] > 0)[0]
        if len(relevant) == 0:
            continue
        if len(set(top_k) & set(relevant)) > 0:
            hits += 1
    n_with_relevant = np.sum(np.any(labels > 0, axis=1))
    if n_with_relevant == 0:
        return 0.0
    return hits / n_with_relevant

def compute_mrr(query_emb, doc_emb, labels):
    sim = query_emb @ doc_emb.T
    n_queries = sim.shape[0]
    reciprocal_ranks = []
    for i in range(n_queries):
        ranking = np.argsort(sim[i])[::-1]
        relevant = set(np.where(labels[i] > 0)[0])
        if not relevant:
            continue
        for rank, idx in enumerate(ranking, 1):
            if idx in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
    if not reciprocal_ranks:
        return 0.0
    return float(np.mean(reciprocal_ranks))

def evaluate(con, model):
    model_version = ACTIVE_MODEL["name"]

    test_pairs = con.execute(
        "SELECT cv_id, job_id, label FROM pairs WHERE split = 'test'"
    ).fetchall()

    if not test_pairs:
        print("[eval] no test pairs found")
        return {}

    cv_ids_set = sorted(set(r[0] for r in test_pairs))
    job_ids_set = sorted(set(r[1] for r in test_pairs))

    cv_id_to_idx = {cid: idx for idx, cid in enumerate(cv_ids_set)}
    job_id_to_idx = {jid: idx for idx, jid in enumerate(job_ids_set)}

    label_matrix = np.zeros((len(cv_ids_set), len(job_ids_set)), dtype=np.float32)
    for cv_id, job_id, label in test_pairs:
        label_matrix[cv_id_to_idx[cv_id], job_id_to_idx[job_id]] = label

    cv_embs = {}
    job_embs = {}

    if cv_ids_set:
        placeholders = ",".join(["?"] * len(cv_ids_set))
        rows = con.execute(
            f"SELECT doc_id, embedding FROM embeddings WHERE model_version = ? AND doc_id IN ({placeholders})",
            [model_version] + cv_ids_set,
        ).fetchall()
        for doc_id, emb in rows:
            cv_embs[doc_id] = np.array(emb, dtype=np.float32)

    if job_ids_set:
        placeholders = ",".join(["?"] * len(job_ids_set))
        rows = con.execute(
            f"SELECT doc_id, embedding FROM embeddings WHERE model_version = ? AND doc_id IN ({placeholders})",
            [model_version] + job_ids_set,
        ).fetchall()
        for doc_id, emb in rows:
            job_embs[doc_id] = np.array(emb, dtype=np.float32)

    missing_cv = [cid for cid in cv_ids_set if cid not in cv_embs]
    missing_job = [jid for jid in job_ids_set if jid not in job_embs]

    if missing_cv or missing_job:
        missing_ids = missing_cv + missing_job
        placeholders = ",".join(["?"] * len(missing_ids))
        rows = con.execute(
            f"SELECT id, text FROM documents WHERE id IN ({placeholders})",
            missing_ids,
        ).fetchall()
        id_to_text = {r[0]: r[1] for r in rows}

        if missing_cv:
            texts = [id_to_text[cid] for cid in missing_cv]
            embs = embed_documents(texts, model, query=True)
            for cid, emb in zip(missing_cv, embs):
                cv_embs[cid] = emb

        if missing_job:
            texts = [id_to_text[jid] for jid in missing_job]
            embs = embed_documents(texts, model, query=False)
            for jid, emb in zip(missing_job, embs):
                job_embs[jid] = emb

    query_matrix = np.stack([cv_embs[cid] for cid in cv_ids_set])
    doc_matrix = np.stack([job_embs[jid] for jid in job_ids_set])

    metrics = {}
    for k in EVAL["recall_k"]:
        metrics[f"recall@{k}"] = compute_recall_at_k(query_matrix, doc_matrix, label_matrix, k)
    metrics["mrr"] = compute_mrr(query_matrix, doc_matrix, label_matrix)

    exp_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM experiments").fetchone()[0]
    con.execute(
        "INSERT INTO experiments (id, model_version, config, metrics, created_at) VALUES (?, ?, ?, ?, ?)",
        [exp_id, model_version, json.dumps({"model": ACTIVE_MODEL["name"], "stage": "eval"}), json.dumps(metrics), None],
    )

    return metrics

# --- PIPELINE ---

def run_pipeline(cluster=False):
    con = get_connection()
    print("[1/5] Loading data")
    load_data(con)
    print("[2/5] Embedding documents")
    model = load_model()
    embed_all(con, model)
    if cluster:
        print("[3/5] Clustering")
        cluster_documents(con)
        generate_pairs_from_clusters(con)
    else:
        print("[3/5] Clustering -- skipped")
    print("[4/5] Training")
    train_model(con)
    print("[5/5] Evaluating")
    metrics = evaluate(con, model)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    con.close()
    return metrics

if __name__ == "__main__":
    run_pipeline()
