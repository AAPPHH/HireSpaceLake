import numpy as np
import hdbscan
from main.config import CLUSTER_CONFIG
from main.data import store_pairs


def cluster_documents(con, embedding_column="raw_embedding"):
    rows = con.execute(
        f"SELECT id, {embedding_column} FROM lake.main.documents WHERE {embedding_column} IS NOT NULL"
    ).fetchall()
    if not rows:
        return np.array([])
    ids = [r[0] for r in rows]
    embeddings = np.array([list(r[1]) for r in rows], dtype=np.float32)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=CLUSTER_CONFIG["min_cluster_size"],
        min_samples=CLUSTER_CONFIG["min_samples"],
        metric=CLUSTER_CONFIG["metric"],
        cluster_selection_method=CLUSTER_CONFIG["cluster_selection_method"],
    )
    labels = clusterer.fit_predict(embeddings)
    for idx, doc_id in enumerate(ids):
        con.execute(
            "UPDATE lake.main.documents SET cluster_id = ? WHERE id = ?",
            [int(labels[idx]), doc_id],
        )
    return labels


def get_cluster_centroids(con, embedding_column="raw_embedding"):
    rows = con.execute(
        f"SELECT cluster_id, {embedding_column} FROM lake.main.documents "
        f"WHERE cluster_id IS NOT NULL AND cluster_id >= 0 AND {embedding_column} IS NOT NULL"
    ).fetchall()
    clusters = {}
    for cluster_id, emb in rows:
        clusters.setdefault(int(cluster_id), []).append(list(emb))
    centroids = {}
    for cluster_id, embs in clusters.items():
        centroids[cluster_id] = np.mean(embs, axis=0)
    return centroids


def generate_pairs_from_clusters(con, num_negatives=3):
    rows = con.execute(
        "SELECT id, type, cluster_id FROM lake.main.documents "
        "WHERE cluster_id IS NOT NULL AND cluster_id >= 0"
    ).fetchall()
    clusters = {}
    for doc_id, doc_type, cluster_id in rows:
        clusters.setdefault(int(cluster_id), []).append((doc_id, doc_type))
    all_cluster_ids = list(clusters.keys())
    pairs = []
    for cluster_id, members in clusters.items():
        cvs = [m[0] for m in members if m[1] == "cv"]
        jobs = [m[0] for m in members if m[1] == "job"]
        for cv_id in cvs:
            for job_id in jobs:
                pairs.append({
                    "cv_id": cv_id, "job_id": job_id,
                    "label": 1.0, "source": "cluster",
                })
        other_clusters = [c for c in all_cluster_ids if c != cluster_id]
        if not other_clusters:
            continue
        for cv_id in cvs:
            neg_clusters = np.random.choice(
                other_clusters,
                size=min(num_negatives, len(other_clusters)),
                replace=False,
            )
            for neg_c in neg_clusters:
                neg_jobs = [m[0] for m in clusters[neg_c] if m[1] == "job"]
                if neg_jobs:
                    neg_job = np.random.choice(neg_jobs)
                    pairs.append({
                        "cv_id": cv_id, "job_id": int(neg_job),
                        "label": 0.0, "source": "cluster",
                    })
    if pairs:
        store_pairs(con, pairs)
    return pairs
