import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    MODEL, HARRIER, LORA, LAKE, CLUSTER, EVAL,
    ACTIVE_MODEL, SEP_TOKEN, DATASET_NAME, DATASET_FALLBACK, DUCKLAKE_TABLES,
)


class TestConfigStructure(unittest.TestCase):
    def test_active_model_has_required_keys(self):
        for key in ["name", "embedding_dim", "max_length", "instruction_prefix", "batch_size"]:
            self.assertIn(key, ACTIVE_MODEL)

    def test_model_harrier_same_keys(self):
        self.assertEqual(set(MODEL.keys()), set(HARRIER.keys()))

    def test_model_dimensions(self):
        self.assertEqual(MODEL["embedding_dim"], 384)
        self.assertEqual(HARRIER["embedding_dim"], 1024)

    def test_ducklake_tables_complete(self):
        for name in ["documents", "pairs", "embeddings", "clusters", "experiments"]:
            self.assertIn(name, DUCKLAKE_TABLES)

    def test_schema_no_primary_key(self):
        for ddl in DUCKLAKE_TABLES.values():
            self.assertNotIn("PRIMARY KEY", ddl.upper())

    def test_schema_no_references(self):
        for ddl in DUCKLAKE_TABLES.values():
            self.assertNotIn("REFERENCES", ddl.upper())

    def test_schema_no_default_now(self):
        for ddl in DUCKLAKE_TABLES.values():
            self.assertNotIn("DEFAULT NOW()", ddl.upper())
            self.assertNotIn("DEFAULT CURRENT_TIMESTAMP", ddl.upper())

    def test_schema_variable_length_arrays(self):
        import re
        for ddl in DUCKLAKE_TABLES.values():
            matches = re.findall(r'FLOAT\[\d+\]', ddl)
            self.assertEqual(matches, [], f"Fixed-size array found: {matches}")

    def test_eval_config(self):
        self.assertEqual(EVAL["recall_k"], [1, 5, 10])

    def test_lora_config(self):
        self.assertEqual(LORA["rank"], 16)
        self.assertEqual(LORA["alpha"], 32)


if __name__ == "__main__":
    unittest.main()
