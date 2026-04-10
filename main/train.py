import json
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from main.config import (
    MODEL_NAME, TRAIN_CONFIG, CHECKPOINT_DIR, EMBEDDING_DIM,
)
from main.data import get_pairs, get_documents
import os


def create_training_pairs(con):
    pairs = get_pairs(con)
    if not pairs:
        return []
    docs = get_documents(con)
    doc_map = {d["id"]: d["text"] for d in docs}
    examples = []
    for p in pairs:
        cv_text = doc_map.get(p["cv_id"])
        job_text = doc_map.get(p["job_id"])
        if cv_text and job_text:
            examples.append(InputExample(texts=[cv_text, job_text], label=float(p["label"])))
    return examples


def setup_lora_model(model_name=None):
    model_name = model_name or MODEL_NAME
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=TRAIN_CONFIG["lora_rank"],
        lora_alpha=TRAIN_CONFIG["lora_alpha"],
        target_modules=TRAIN_CONFIG["lora_target_modules"],
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    st_model = SentenceTransformer(modules=[peft_model])
    return st_model


def train(con, model=None):
    if model is None:
        model = setup_lora_model()
    examples = create_training_pairs(con)
    if not examples:
        print("No training pairs found")
        return None
    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=TRAIN_CONFIG["batch_size"],
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(
        len(train_dataloader) * TRAIN_CONFIG["epochs"] * TRAIN_CONFIG["warmup_ratio"]
    )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"lora_{int(time.time())}"
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=TRAIN_CONFIG["epochs"],
        warmup_steps=warmup_steps,
        output_path=checkpoint_path,
        evaluation_steps=TRAIN_CONFIG["eval_steps"],
        save_best_model=True,
        show_progress_bar=True,
    )
    metrics = {"checkpoint": checkpoint_path, "num_pairs": len(examples)}
    config_dump = json.dumps(TRAIN_CONFIG)
    metrics_dump = json.dumps(metrics)
    try:
        max_id = con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM lake.main.experiments"
        ).fetchone()[0]
    except Exception:
        max_id = 0
    con.execute(
        "INSERT INTO lake.main.experiments (id, model_version, metrics_json, config_json) VALUES (?, ?, ?, ?)",
        [max_id + 1, f"lora_{TRAIN_CONFIG['lora_rank']}r", metrics_dump, config_dump],
    )
    print(f"Training complete: {len(examples)} pairs, saved to {checkpoint_path}")
    return model
