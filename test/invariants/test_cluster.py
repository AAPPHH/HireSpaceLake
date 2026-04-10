import os, sys, unittest, tempfile, importlib
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if "hdbscan" not in sys.modules:
    mock = MagicMock()
    mock.__spec__ = importlib.machinery.ModuleSpec("hdbscan", None)
    sys.modules["hdbscan"] = mock

import numpy as np
import duckdb
from main.config import DUCKLAKE_TABLES


def make_con(tmp_dir):
    catalog = os.path.join(tmp_dir, "test.ducklake")
    con = duckdb.connect(":memory:")
    con.execute("INSTALL ducklake;")
    con.execute("LOAD ducklake;")
    con.execute(f"ATTACH 'ducklake:{catalog}' AS lake;")
    con.execute("USE lake.main;")
    for table_sql in DUCKLAKE_TABLES.values():
        con.execute(table_sql)
    return con


def insert_doc_with_embedding(con, doc_id, doc_type, text, embedding, cluster_id=None):
    emb_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    con.execute(
        "INSERT INTO lake.main.documents (id, type, text, raw_embedding, cluster_id) VALUES (?, ?, ?, ?, ?)",
        [doc_id, doc_type, text, emb_list, cluster_id],
    )


class TestClusterCentroids(unittest.TestCase):
    def test_centroid_is_mean(self):
        from main.cluster import get_cluster_centroids

        tmp = tempfile.mkdtemp()
        try:
            con = make_con(tmp)
            emb1 = np.array([1.0] * 1024, dtype=np.float32)
            emb2 = np.array([3.0] * 1024, dtype=np.float32)
            emb3 = np.array([2.0] * 1024, dtype=np.float32)
            insert_doc_with_embedding(con, 1, "cv", "cv1", emb1, cluster_id=0)
            insert_doc_with_embedding(con, 2, "cv", "cv2", emb2, cluster_id=0)
            insert_doc_with_embedding(con, 3, "job", "job1", emb3, cluster_id=0)
            centroids = get_cluster_centroids(con)
            self.assertIn(0, centroids)
            expected = np.mean([emb1, emb2, emb3], axis=0)
            np.testing.assert_allclose(centroids[0], expected, atol=1e-5)
            con.close()
        except Exception:
            con.close()
            raise

    def test_multiple_clusters(self):
        from main.cluster import get_cluster_centroids

        tmp = tempfile.mkdtemp()
        try:
            con = make_con(tmp)
            for i in range(3):
                emb = np.array([float(i)] * 1024, dtype=np.float32)
                insert_doc_with_embedding(con, i + 1, "cv", f"cv_{i}", emb, cluster_id=0)
            for i in range(3):
                emb = np.array([float(i + 10)] * 1024, dtype=np.float32)
                insert_doc_with_embedding(con, i + 4, "job", f"job_{i}", emb, cluster_id=1)
            centroids = get_cluster_centroids(con)
            self.assertEqual(len(centroids), 2)
            self.assertIn(0, centroids)
            self.assertIn(1, centroids)
            con.close()
        except Exception:
            con.close()
            raise


class TestGeneratePairs(unittest.TestCase):
    def test_pairs_have_cluster_source(self):
        from main.cluster import generate_pairs_from_clusters
        from main.data import get_pairs

        tmp = tempfile.mkdtemp()
        try:
            con = make_con(tmp)
            emb = np.zeros(1024, dtype=np.float32)
            insert_doc_with_embedding(con, 1, "cv", "cv1", emb, cluster_id=0)
            insert_doc_with_embedding(con, 2, "job", "job1", emb, cluster_id=0)
            insert_doc_with_embedding(con, 3, "cv", "cv2", emb, cluster_id=1)
            insert_doc_with_embedding(con, 4, "job", "job2", emb, cluster_id=1)
            pairs = generate_pairs_from_clusters(con, num_negatives=1)
            self.assertGreater(len(pairs), 0)
            self.assertTrue(all(p["source"] == "cluster" for p in pairs))
            db_pairs = get_pairs(con, source="cluster")
            self.assertEqual(len(db_pairs), len(pairs))
            con.close()
        except Exception:
            con.close()
            raise

    def test_positive_pairs_intra_cluster(self):
        from main.cluster import generate_pairs_from_clusters

        tmp = tempfile.mkdtemp()
        try:
            con = make_con(tmp)
            emb = np.zeros(1024, dtype=np.float32)
            insert_doc_with_embedding(con, 1, "cv", "cv1", emb, cluster_id=0)
            insert_doc_with_embedding(con, 2, "job", "job1", emb, cluster_id=0)
            insert_doc_with_embedding(con, 3, "job", "job2", emb, cluster_id=0)
            pairs = generate_pairs_from_clusters(con, num_negatives=0)
            positive_pairs = [p for p in pairs if p["label"] == 1.0]
            self.assertEqual(len(positive_pairs), 2)
            self.assertTrue(all(p["cv_id"] == 1 for p in positive_pairs))
            job_ids = {p["job_id"] for p in positive_pairs}
            self.assertEqual(job_ids, {2, 3})
            con.close()
        except Exception:
            con.close()
            raise

    def test_negative_pairs_cross_cluster(self):
        from main.cluster import generate_pairs_from_clusters

        tmp = tempfile.mkdtemp()
        try:
            con = make_con(tmp)
            np.random.seed(42)
            emb = np.zeros(1024, dtype=np.float32)
            insert_doc_with_embedding(con, 1, "cv", "cv1", emb, cluster_id=0)
            insert_doc_with_embedding(con, 2, "job", "job1", emb, cluster_id=0)
            insert_doc_with_embedding(con, 3, "cv", "cv2", emb, cluster_id=1)
            insert_doc_with_embedding(con, 4, "job", "job2", emb, cluster_id=1)
            pairs = generate_pairs_from_clusters(con, num_negatives=1)
            negative_pairs = [p for p in pairs if p["label"] == 0.0]
            self.assertGreater(len(negative_pairs), 0)
            for p in negative_pairs:
                cv_cluster = con.execute(
                    "SELECT cluster_id FROM lake.main.documents WHERE id = ?", [p["cv_id"]]
                ).fetchone()[0]
                job_cluster = con.execute(
                    "SELECT cluster_id FROM lake.main.documents WHERE id = ?", [p["job_id"]]
                ).fetchone()[0]
                self.assertNotEqual(cv_cluster, job_cluster)
            con.close()
        except Exception:
            con.close()
            raise


if __name__ == "__main__":
    unittest.main()
