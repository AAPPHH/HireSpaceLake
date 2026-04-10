import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAKE_DIR = os.path.join(BASE_DIR, "lake")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DUCKLAKE_CATALOG = os.path.join(LAKE_DIR, "talentlake.ducklake")

MODEL_NAME = "microsoft/Harrier-OSS-v1-0.6b"
EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 32768

QUERY_PREFIX = "Represent this document for retrieval: "
SEP_TOKEN = " SEP "

DATASET_NAME = "0xnbk/resume-ats-score-v1-en"
DATASET_TEXT_COL = "text"
DATASET_SCORE_COL = "ats_score"

DUCKLAKE_TABLES = {
    "documents": """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER,
            type VARCHAR,
            text VARCHAR,
            raw_embedding FLOAT[],
            lora_embedding FLOAT[],
            cluster_id INTEGER,
            created_at TIMESTAMP
        )
    """,
    "pairs": """
        CREATE TABLE IF NOT EXISTS pairs (
            cv_id INTEGER,
            job_id INTEGER,
            label FLOAT,
            source VARCHAR,
            created_at TIMESTAMP
        )
    """,
    "experiments": """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER,
            model_version VARCHAR,
            metrics_json VARCHAR,
            config_json VARCHAR,
            created_at TIMESTAMP
        )
    """,
}

EMBED_CONFIG = {
    "batch_size": 32,
    "normalize": True,
    "device": "cuda",
}

CLUSTER_CONFIG = {
    "min_cluster_size": 15,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}

TRAIN_CONFIG = {
    "epochs": 10,
    "lr": 2e-5,
    "batch_size": 16,
    "warmup_ratio": 0.1,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "v_proj"],
    "quantization_bits": 4,
    "loss": "MultipleNegativesRankingLoss",
    "eval_steps": 100,
    "save_steps": 500,
    "resume": None,
}

EVAL_CONFIG = {
    "k_values": [1, 5, 10, 20],
    "metrics": ["recall", "mrr", "ndcg"],
    "batch_size": 64,
}

APP_CONFIG = {
    "title": "TalentLake - CV-Job Matching",
    "top_k": 10,
    "port": 8501,
}
