import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "max_length": 256,
    "instruction_prefix": "",
    "batch_size": 64,
}

HARRIER = {
    "name": "microsoft/harrier-oss-v1-0.6b",
    "embedding_dim": 1024,
    "max_length": 32768,
    "instruction_prefix": "Instruct: Match this CV to relevant job postings\nQuery: ",
    "batch_size": 8,
}

LORA = {
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lr": 2e-5,
    "epochs": 10,
    "warmup_ratio": 0.1,
}

LAKE = {
    "catalog": os.path.join(BASE_DIR, "lake", "talentlake.ducklake"),
    "data_dir": os.path.join(BASE_DIR, "lake", "data"),
}

CLUSTER = {
    "min_cluster_size": 5,
    "min_samples": 3,
    "metric": "euclidean",
}

EVAL = {
    "recall_k": [1, 5, 10],
}

ACTIVE_MODEL = MODEL

SEP_TOKEN = " SEP "
DATASET_NAME = "0xnbk/resume-ats-score-v1-en"
DATASET_FALLBACK = "cnamuangtoun/resume-job-description-fit"

DUCKLAKE_TABLES = {
    "documents": """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER,
            doc_type VARCHAR,
            text VARCHAR,
            source VARCHAR,
            created_at TIMESTAMP
        )
    """,
    "pairs": """
        CREATE TABLE IF NOT EXISTS pairs (
            cv_id INTEGER,
            job_id INTEGER,
            label FLOAT,
            label_source VARCHAR,
            split VARCHAR
        )
    """,
    "embeddings": """
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id INTEGER,
            model_version VARCHAR,
            embedding FLOAT[],
            created_at TIMESTAMP
        )
    """,
    "clusters": """
        CREATE TABLE IF NOT EXISTS clusters (
            doc_id INTEGER,
            model_version VARCHAR,
            cluster_id INTEGER,
            is_centroid BOOLEAN
        )
    """,
    "experiments": """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER,
            model_version VARCHAR,
            config JSON,
            metrics JSON,
            created_at TIMESTAMP
        )
    """,
}
