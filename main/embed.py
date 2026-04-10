import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from main.config import (
    MODEL_NAME, EMBEDDING_DIM, MAX_SEQ_LENGTH,
    QUERY_PREFIX, EMBED_CONFIG,
)


def load_model(device=None):
    device = device or EMBED_CONFIG["device"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def embed_documents(texts, model, tokenizer, config=None, query=False):
    config = config or EMBED_CONFIG
    device = config.get("device", "cuda")
    normalize = config.get("normalize", True)
    batch_size = config.get("batch_size", 32)
    if query:
        texts = [QUERY_PREFIX + t for t in texts]
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_SEQ_LENGTH, return_tensors="pt",
            ).to(device)
            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / count
            if normalize:
                mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(mean_pooled.cpu().float().numpy())
    return np.concatenate(all_embeddings, axis=0)


def update_embeddings(con, model, tokenizer, column="raw_embedding"):
    rows = con.execute(
        f"SELECT id, text FROM lake.main.documents WHERE {column} IS NULL"
    ).fetchall()
    if not rows:
        return
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    config = EMBED_CONFIG.copy()
    batch_size = config.get("batch_size", 32)
    embeddings = embed_documents(texts, model, tokenizer, config=config)
    for idx, doc_id in enumerate(ids):
        emb_list = embeddings[idx].tolist()
        con.execute(
            f"UPDATE lake.main.documents SET {column} = ? WHERE id = ?",
            [emb_list, doc_id],
        )
