import os, sys, unittest, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import duckdb
from main.config import SEP_TOKEN, DUCKLAKE_TABLES


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


class TestSepTokenSplit(unittest.TestCase):
    def test_split_produces_two_parts(self):
        text = "Some CV text here" + SEP_TOKEN + "Some Job description"
        parts = text.split(SEP_TOKEN)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "Some CV text here")
        self.assertEqual(parts[1], "Some Job description")

    def test_split_no_sep_token(self):
        text = "No separator in this text"
        parts = text.split(SEP_TOKEN)
        self.assertEqual(len(parts), 1)

    def test_split_multiple_sep_tokens(self):
        text = "A" + SEP_TOKEN + "B" + SEP_TOKEN + "C"
        parts = text.split(SEP_TOKEN)
        self.assertEqual(len(parts), 3)


class TestInitDucklake(unittest.TestCase):
    def test_creates_all_tables(self):
        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_catalog = 'lake'"
            ).fetchall()
            table_names = {t[0] for t in tables}
            self.assertIn("documents", table_names)
            self.assertIn("pairs", table_names)
            self.assertIn("experiments", table_names)
        finally:
            con.close()

    def test_documents_table_has_expected_columns(self):
        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            cols = con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'documents' AND table_catalog = 'lake'"
            ).fetchall()
            col_names = {c[0] for c in cols}
            for expected in ["id", "type", "text", "raw_embedding", "lora_embedding", "cluster_id", "created_at"]:
                self.assertIn(expected, col_names)
        finally:
            con.close()


class TestStoreGetRoundtrip(unittest.TestCase):
    def test_store_and_get_documents(self):
        from main.data import store_documents, get_documents

        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            docs = [
                {"id": 1, "type": "cv", "text": "Python developer with 5 years"},
                {"id": 2, "type": "job", "text": "Looking for senior Python dev"},
            ]
            store_documents(con, docs)
            result = get_documents(con)
            self.assertEqual(len(result), 2)
            types = {r["type"] for r in result}
            self.assertEqual(types, {"cv", "job"})
        finally:
            con.close()

    def test_get_documents_filtered(self):
        from main.data import store_documents, get_documents

        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            docs = [
                {"id": 1, "type": "cv", "text": "CV text"},
                {"id": 2, "type": "job", "text": "Job text"},
                {"id": 3, "type": "cv", "text": "Another CV"},
            ]
            store_documents(con, docs)
            cvs = get_documents(con, doc_type="cv")
            self.assertEqual(len(cvs), 2)
            self.assertTrue(all(d["type"] == "cv" for d in cvs))
        finally:
            con.close()

    def test_store_and_get_pairs(self):
        from main.data import store_documents, store_pairs, get_pairs

        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            store_documents(con, [
                {"id": 1, "type": "cv", "text": "CV"},
                {"id": 2, "type": "job", "text": "Job"},
            ])
            pairs = [{"cv_id": 1, "job_id": 2, "label": 0.85, "source": "dataset"}]
            store_pairs(con, pairs)
            result = get_pairs(con)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["cv_id"], 1)
            self.assertEqual(result[0]["job_id"], 2)
            self.assertEqual(result[0]["source"], "dataset")
        finally:
            con.close()

    def test_get_pairs_filtered_by_source(self):
        from main.data import store_documents, store_pairs, get_pairs

        tmp = tempfile.mkdtemp()
        con = make_con(tmp)
        try:
            store_documents(con, [
                {"id": 1, "type": "cv", "text": "CV"},
                {"id": 2, "type": "job", "text": "Job"},
            ])
            store_pairs(con, [
                {"cv_id": 1, "job_id": 2, "label": 1.0, "source": "cluster"},
                {"cv_id": 1, "job_id": 2, "label": 0.5, "source": "manual"},
            ])
            cluster_pairs = get_pairs(con, source="cluster")
            self.assertEqual(len(cluster_pairs), 1)
            self.assertEqual(cluster_pairs[0]["source"], "cluster")
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
