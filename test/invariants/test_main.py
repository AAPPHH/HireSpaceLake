import sys
import os
import importlib
import unittest
import tempfile
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

for mod_name in [
    "datasets",
    "sentence_transformers",
    "sentence_transformers.util",
    "transformers",
    "hdbscan",
    "peft",
    "bitsandbytes",
    "accelerate",
    "torch",
]:
    if mod_name not in sys.modules:
        mock = MagicMock()
        mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
        sys.modules[mod_name] = mock

import numpy as np
import duckdb
from config import ACTIVE_MODEL, SEP_TOKEN, EVAL, DUCKLAKE_TABLES

try:
    from main import (
        compute_recall_at_k,
        compute_mrr,
        embed_documents,
    )
    MAIN_AVAILABLE = True
except ImportError:
    MAIN_AVAILABLE = False


def make_con(tmp_dir):
    catalog = os.path.join(tmp_dir, "test.ducklake")
    con = duckdb.connect(":memory:")
    con.execute("INSTALL ducklake; LOAD ducklake;")
    con.execute(f"ATTACH 'ducklake:{catalog}' AS lake;")
    con.execute("USE lake.main;")
    for table_sql in DUCKLAKE_TABLES.values():
        con.execute(table_sql)
    return con


class TestDuckLakeSchema(unittest.TestCase):
    def test_creates_all_five_tables(self):
        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog='lake'"
            ).fetchall()
            names = {t[0] for t in tables}
            for expected in ["documents", "pairs", "embeddings", "clusters", "experiments"]:
                self.assertIn(expected, names)
        finally:
            con.close()

    def test_documents_roundtrip(self):
        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            con.execute(
                "INSERT INTO documents (id, doc_type, text, source, created_at) VALUES (?, ?, ?, ?, ?)",
                [1, "cv", "Python developer with 5 years experience", "test", None],
            )
            con.execute(
                "INSERT INTO documents (id, doc_type, text, source, created_at) VALUES (?, ?, ?, ?, ?)",
                [2, "job", "Looking for senior Python developer", "test", None],
            )
            rows = con.execute("SELECT id, doc_type, text FROM documents ORDER BY id").fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], 1)
            self.assertEqual(rows[0][1], "cv")
            self.assertEqual(rows[1][0], 2)
            self.assertEqual(rows[1][1], "job")
        finally:
            con.close()

    def test_pairs_roundtrip(self):
        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            con.execute(
                "INSERT INTO pairs (cv_id, job_id, label, label_source, split) VALUES (?, ?, ?, ?, ?)",
                [1, 2, 0.85, "dataset", "train"],
            )
            con.execute(
                "INSERT INTO pairs (cv_id, job_id, label, label_source, split) VALUES (?, ?, ?, ?, ?)",
                [3, 4, 0.5, "dataset", "test"],
            )
            rows = con.execute("SELECT cv_id, job_id, label FROM pairs WHERE split = 'test'").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], 3)
            self.assertAlmostEqual(rows[0][2], 0.5, places=5)
        finally:
            con.close()

    def test_embeddings_roundtrip(self):
        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            emb = [float(x) for x in range(384)]
            con.execute(
                "INSERT INTO embeddings (doc_id, model_version, embedding, created_at) VALUES (?, ?, ?, ?)",
                [1, "test-model", emb, None],
            )
            rows = con.execute("SELECT doc_id, embedding FROM embeddings WHERE doc_id = 1").fetchall()
            self.assertEqual(len(rows), 1)
            retrieved = rows[0][1]
            self.assertEqual(len(retrieved), 384)
            self.assertAlmostEqual(retrieved[0], 0.0, places=5)
            self.assertAlmostEqual(retrieved[383], 383.0, places=5)
        finally:
            con.close()

    def test_experiments_json(self):
        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            config_data = {"rank": 16, "alpha": 32}
            metrics_data = {"recall@1": 0.75, "mrr": 0.82}
            con.execute(
                "INSERT INTO experiments (id, model_version, config, metrics, created_at) VALUES (?, ?, ?, ?, ?)",
                [1, "test-model", json.dumps(config_data), json.dumps(metrics_data), None],
            )
            rows = con.execute("SELECT config, metrics FROM experiments WHERE id = 1").fetchall()
            self.assertEqual(len(rows), 1)
            parsed_config = json.loads(rows[0][0])
            parsed_metrics = json.loads(rows[0][1])
            self.assertEqual(parsed_config["rank"], 16)
            self.assertAlmostEqual(parsed_metrics["mrr"], 0.82, places=5)
        finally:
            con.close()


@unittest.skipUnless(MAIN_AVAILABLE, "main.py not available yet")
class TestMetrics(unittest.TestCase):
    def test_recall_perfect(self):
        n = 5
        emb = np.eye(n, dtype=np.float32)
        labels = np.eye(n, dtype=np.float32)
        recall = compute_recall_at_k(emb, emb, labels, 1)
        self.assertAlmostEqual(recall, 1.0, places=5)

    def test_mrr_perfect(self):
        n = 5
        emb = np.eye(n, dtype=np.float32)
        labels = np.eye(n, dtype=np.float32)
        mrr = compute_mrr(emb, emb, labels)
        self.assertAlmostEqual(mrr, 1.0, places=5)

    def test_metrics_in_range(self):
        rng = np.random.RandomState(42)
        n_q, n_d = 20, 30
        query_emb = rng.randn(n_q, 64).astype(np.float32)
        doc_emb = rng.randn(n_d, 64).astype(np.float32)
        labels = (rng.rand(n_q, n_d) > 0.8).astype(np.float32)
        for k in EVAL["recall_k"]:
            recall = compute_recall_at_k(query_emb, doc_emb, labels, k)
            self.assertGreaterEqual(recall, 0.0)
            self.assertLessEqual(recall, 1.0)
        mrr = compute_mrr(query_emb, doc_emb, labels)
        self.assertGreaterEqual(mrr, 0.0)
        self.assertLessEqual(mrr, 1.0)

    def test_recall_no_relevant(self):
        n_q, n_d = 5, 5
        emb = np.eye(n_q, dtype=np.float32)
        labels = np.zeros((n_q, n_d), dtype=np.float32)
        recall = compute_recall_at_k(emb, emb, labels, 1)
        self.assertAlmostEqual(recall, 0.0, places=5)

    def test_cosine_identical(self):
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        sim = float(vec @ vec)
        self.assertAlmostEqual(sim, 1.0, places=5)


@unittest.skipUnless(MAIN_AVAILABLE, "main.py not available yet")
class TestEmbedding(unittest.TestCase):
    def test_embed_shape_matches_config(self):
        dim = ACTIVE_MODEL["embedding_dim"]
        n = 3
        fake_output = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(fake_output, axis=1, keepdims=True)
        fake_output = fake_output / norms

        mock_model = MagicMock()
        mock_model.__class__ = type("SentenceTransformer", (), {})
        mock_model.encode = MagicMock(return_value=fake_output)

        st_mock = sys.modules["sentence_transformers"]
        original_st = st_mock.SentenceTransformer
        st_mock.SentenceTransformer = type(mock_model)

        try:
            result = embed_documents(["text1", "text2", "text3"], mock_model, query=False)
            self.assertEqual(result.shape, (n, dim))
        finally:
            st_mock.SentenceTransformer = original_st

    def test_embed_normalized(self):
        dim = ACTIVE_MODEL["embedding_dim"]
        n = 4
        fake_output = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(fake_output, axis=1, keepdims=True)
        fake_output = fake_output / norms

        mock_model = MagicMock()
        mock_model.__class__ = type("SentenceTransformer", (), {})
        mock_model.encode = MagicMock(return_value=fake_output)

        st_mock = sys.modules["sentence_transformers"]
        original_st = st_mock.SentenceTransformer
        st_mock.SentenceTransformer = type(mock_model)

        try:
            result = embed_documents(["a", "b", "c", "d"], mock_model, query=False)
            result_norms = np.linalg.norm(result, axis=1)
            for norm_val in result_norms:
                self.assertAlmostEqual(float(norm_val), 1.0, places=3)
        finally:
            st_mock.SentenceTransformer = original_st

    def test_query_prefix_applied(self):
        dim = ACTIVE_MODEL["embedding_dim"]
        fake_output = np.random.randn(2, dim).astype(np.float32)

        mock_model = MagicMock()
        mock_model.__class__ = type("SentenceTransformer", (), {})
        mock_model.encode = MagicMock(return_value=fake_output)

        st_mock = sys.modules["sentence_transformers"]
        original_st = st_mock.SentenceTransformer
        st_mock.SentenceTransformer = type(mock_model)

        try:
            with patch.dict(ACTIVE_MODEL, {"instruction_prefix": "Search: "}):
                embed_documents(["hello", "world"], mock_model, query=True)
                call_args = mock_model.encode.call_args
                texts_passed = call_args[0][0]
                for t in texts_passed:
                    self.assertTrue(t.startswith("Search: "), f"Missing prefix in: {t}")
        finally:
            st_mock.SentenceTransformer = original_st


class TestScoreNormalization(unittest.TestCase):
    def test_score_above_one_normalized(self):
        score = 85.0
        if score > 1.0:
            score = score / 100.0
        self.assertAlmostEqual(score, 0.85, places=5)

    def test_score_below_one_unchanged(self):
        score = 0.5
        if score > 1.0:
            score = score / 100.0
        self.assertAlmostEqual(score, 0.5, places=5)


class TestSepTokenSplit(unittest.TestCase):
    def test_sep_split(self):
        text = "CV text here" + SEP_TOKEN + "Job description here"
        parts = text.split(SEP_TOKEN)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "CV text here")
        self.assertEqual(parts[1], "Job description here")

    def test_sep_split_no_token(self):
        text = "No separator in this text"
        parts = text.split(SEP_TOKEN)
        self.assertEqual(len(parts), 1)


if __name__ == "__main__":
    unittest.main()
