# TalentLake

CV-Job Matching via Embedding Retrieval. CVs und Jobanzeigen werden in einen gemeinsamen Vektorraum projiziert, Matching laeuft ueber Cosine Similarity. Semi-supervised Labeling Pipeline: HDBSCAN-Cluster → Centroid-Labels → Paar-Generierung → LoRA-Finetuning.

## Quickstart

```bash
pip install torch sentence-transformers transformers accelerate duckdb datasets hdbscan streamlit
python main.py
streamlit run app.py
```

## Architektur

```
HuggingFace Dataset
    │
    │  load_data() — parst "text SEP text" → CV + Job Dokumente
    ▼
┌──────────────────────────────────────────────────────────────┐
│  DuckLake (lake/talentlake.ducklake)                         │
│                                                              │
│  documents ─── embeddings ─── clusters                       │
│      │              │                                        │
│      └── pairs      └── experiments                          │
└──────────────────────────────────────────────────────────────┘
    │
    │  embed_all() — Sentence Transformer oder HF AutoModel
    ▼
┌──────────────────────┐
│  Embedding Space      │
│  (384-dim oder        │
│   1024-dim)           │
└──────────────────────┘
    │
    │  list_cosine_similarity() — Retrieval in SQL
    ▼
┌──────────────────────┐       ┌──────────────────────────┐
│  Matching / Eval      │       │  Streamlit App            │
│  Recall@1/5/10, MRR   │       │  Match | Explore |        │
│                        │       │  Metrics | Data           │
└──────────────────────┘       └──────────────────────────┘
```

## Dateien

| Datei | Funktion |
|-------|----------|
| `config.py` | Config-Dicts fuer Model, LoRA, DuckLake, Cluster, Eval; Schema-Definitionen; `ACTIVE_MODEL` Switch |
| `main.py` | Monolithische Pipeline: Daten laden, embedden, clustern, trainieren, evaluieren |
| `app.py` | Streamlit App mit 4 Tabs: Match (Retrieval), Explore (UMAP-Cluster), Metrics (Experiment-Vergleich), Data (Tabellen + Custom SQL) |

## DuckLake Schema

| Tabelle | Spalten | Zweck |
|---------|---------|-------|
| `documents` | id, doc_type, text, source, created_at | CVs und Jobs als Rohdokumente |
| `pairs` | cv_id, job_id, label, label_source, split | Ground-Truth-Paare mit ATS-Score (0-1), 80/10/10 Split |
| `embeddings` | doc_id, model_version, embedding FLOAT[], created_at | Vektoren pro Modellversion, alte bleiben erhalten |
| `clusters` | doc_id, model_version, cluster_id, is_centroid | HDBSCAN-Zuordnung mit Centroid-Markierung |
| `experiments` | id, model_version, config JSON, metrics JSON, created_at | Experiment-Tracking mit Config und Metriken |

Kein PRIMARY KEY, kein REFERENCES, kein DEFAULT — DuckLake-Limitierungen. Embeddings als `FLOAT[]` (variable length).

## Model Switch

```python
# config.py, Zeile 45
ACTIVE_MODEL = MODEL    # all-MiniLM-L6-v2 (384-dim, CPU)
ACTIVE_MODEL = HARRIER  # microsoft/harrier-oss-v1-0.6b (1024-dim, GPU)
```

Sonst nichts aendern. `embed_all()` ist model-agnostisch, alte Embeddings bleiben mit `model_version` getrackt. Neue werden daneben geschrieben.

## Pipeline Steps

`python main.py` fuehrt `run_pipeline()` aus:

1. **load_data** — HuggingFace Dataset laden, `" SEP "`-Token splitten, CV/Job-Dokumente + Paare in DuckLake schreiben. Idempotent (skipped wenn Daten existieren).
2. **embed_all** — Alle Dokumente ohne Embedding fuer aktives Modell embedden. SentenceTransformer `.encode()` oder HF AutoModel mit Mean Pooling + L2-Normalisierung.
3. **cluster** — HDBSCAN im Embedding-Space, Centroids berechnen, Intra-/Cross-Cluster-Paare generieren. Default: uebersprungen.
4. **train** — Placeholder. Loggt LoRA-Config (rank=16, alpha=32, lr=2e-5) als Experiment.
5. **evaluate** — Test-Split Embeddings laden, Cosine Similarity Matrix berechnen, Recall@1/5/10 und MRR ausgeben. Ergebnis in `experiments` gespeichert.

## Metriken

| Metrik | Bedeutung |
|--------|-----------|
| **Recall@k** (k=1,5,10) | Anteil der Queries, bei denen mindestens ein relevantes Dokument in den Top-k ist |
| **MRR** (Mean Reciprocal Rank) | Mittlerer Kehrwert des Rangs des ersten relevanten Treffers |

Baseline-Werte (Placeholder-Modell ohne Finetuning) zeigen die Retrieval-Qualitaet eines generischen Sentence Transformers auf CV-Job-Matching. Niedrige Werte sind erwartet — das Modell wurde nicht auf diese Domaene trainiert. Nach LoRA-Finetuning sollten alle Metriken steigen.

## Clustering + Semi-supervised Labeling

```python
run_pipeline(cluster=True)
```

1. HDBSCAN clustert Embeddings → `clusters` Tabelle
2. Centroids (naechster Punkt zum Cluster-Mittel) werden markiert
3. Mensch labelt 20-30 Centroids statt tausende Paare
4. Intra-Cluster-Paare = positiv (label=1.0), Cross-Cluster-Paare = negativ (label=0.0, 3 Negatives pro CV)
5. Iterativ: nach Finetuning neue Embeddings → neue Cluster → Labels verfeinern

## Hardware

| Setup | Modell | Anmerkung |
|-------|--------|-----------|
| CPU (beliebig) | all-MiniLM-L6-v2 | 384-dim, batch_size=64, laeuft ueberall |
| GPU (RTX 4080+) | Harrier-OSS-v1-0.6b | 1024-dim, float16, batch_size=8, braucht ~6 GB VRAM |

## Tech Stack

- Python 3.11+
- PyTorch + CUDA (optional)
- sentence-transformers v5.4+
- transformers + accelerate (fuer Harrier)
- DuckDB + DuckLake Extension
- HDBSCAN
- Streamlit
- NumPy, scikit-learn
- peft/QLoRA (geplant, fuer LoRA-Finetuning)
