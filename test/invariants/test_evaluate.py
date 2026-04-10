import os, sys, unittest, importlib
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

for mod_name in ["datasets", "transformers", "transformers.utils"]:
    if mod_name not in sys.modules:
        mock = MagicMock()
        mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
        sys.modules[mod_name] = mock

import numpy as np
from main.evaluate import compute_recall_at_k, compute_mrr, compute_ndcg


class TestRecallAtK(unittest.TestCase):
    def test_perfect_recall(self):
        n = 5
        query_emb = np.eye(n, dtype=np.float32)
        doc_emb = np.eye(n, dtype=np.float32)
        labels = np.eye(n, dtype=np.float32)
        recall = compute_recall_at_k(query_emb, doc_emb, labels, k=1)
        self.assertEqual(recall, 1.0)

    def test_no_relevant_docs(self):
        query_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        doc_emb = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        labels = np.array([[0.0, 0.0]], dtype=np.float32)
        recall = compute_recall_at_k(query_emb, doc_emb, labels, k=2)
        self.assertEqual(recall, 0.0)


class TestMRR(unittest.TestCase):
    def test_perfect_mrr(self):
        n = 5
        query_emb = np.eye(n, dtype=np.float32)
        doc_emb = np.eye(n, dtype=np.float32)
        labels = np.eye(n, dtype=np.float32)
        mrr = compute_mrr(query_emb, doc_emb, labels)
        self.assertEqual(mrr, 1.0)


class TestNDCG(unittest.TestCase):
    def test_perfect_ndcg(self):
        n = 5
        query_emb = np.eye(n, dtype=np.float32)
        doc_emb = np.eye(n, dtype=np.float32)
        labels = np.eye(n, dtype=np.float32)
        ndcg = compute_ndcg(query_emb, doc_emb, labels, k=1)
        self.assertAlmostEqual(ndcg, 1.0, places=5)


class TestMetricsRange(unittest.TestCase):
    def test_metrics_in_unit_range(self):
        rng = np.random.RandomState(42)
        n_q, n_d = 10, 20
        query_emb = rng.randn(n_q, 64).astype(np.float32)
        doc_emb = rng.randn(n_d, 64).astype(np.float32)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        doc_emb = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True)
        labels = np.zeros((n_q, n_d), dtype=np.float32)
        for i in range(n_q):
            labels[i, i % n_d] = 1.0
        recall = compute_recall_at_k(query_emb, doc_emb, labels, k=5)
        mrr = compute_mrr(query_emb, doc_emb, labels)
        ndcg = compute_ndcg(query_emb, doc_emb, labels, k=5)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        self.assertGreaterEqual(mrr, 0.0)
        self.assertLessEqual(mrr, 1.0)
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)
        sim = (v_norm @ v_norm.T)[0, 0]
        self.assertAlmostEqual(float(sim), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
