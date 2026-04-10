import os, sys, unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from unittest.mock import MagicMock
from main.config import QUERY_PREFIX, EMBEDDING_DIM


def make_mock_model_tokenizer(dim=EMBEDDING_DIM):
    tokenizer = MagicMock()

    def tokenize_fn(texts, **kwargs):
        batch_size = len(texts)
        seq_len = 10
        result = MagicMock()
        att = torch.ones(batch_size, seq_len, dtype=torch.long)
        ids = torch.ones(batch_size, seq_len, dtype=torch.long)
        result.__getitem__ = lambda self, key: {"input_ids": ids, "attention_mask": att}[key]
        result.keys = lambda: ["input_ids", "attention_mask"]
        result.to = lambda device: result
        result["attention_mask"] = att
        result["input_ids"] = ids
        return result

    tokenizer.side_effect = tokenize_fn

    model = MagicMock()

    def forward_fn(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        output = MagicMock()
        output.last_hidden_state = torch.randn(batch_size, seq_len, dim)
        return output

    model.side_effect = forward_fn
    model.__call__ = forward_fn
    return model, tokenizer


class TestEmbedOutputShape(unittest.TestCase):
    def test_shape_single(self):
        from main.embed import embed_documents

        model, tokenizer = make_mock_model_tokenizer()
        config = {"batch_size": 4, "normalize": True, "device": "cpu"}
        result = embed_documents(["hello"], model, tokenizer, config=config)
        self.assertEqual(result.shape, (1, EMBEDDING_DIM))

    def test_shape_batch(self):
        from main.embed import embed_documents

        model, tokenizer = make_mock_model_tokenizer()
        config = {"batch_size": 2, "normalize": True, "device": "cpu"}
        texts = ["text one", "text two", "text three"]
        result = embed_documents(texts, model, tokenizer, config=config)
        self.assertEqual(result.shape, (3, EMBEDDING_DIM))

    def test_shape_larger_batch(self):
        from main.embed import embed_documents

        model, tokenizer = make_mock_model_tokenizer()
        config = {"batch_size": 4, "normalize": False, "device": "cpu"}
        texts = [f"text {i}" for i in range(10)]
        result = embed_documents(texts, model, tokenizer, config=config)
        self.assertEqual(result.shape, (10, EMBEDDING_DIM))


class TestNormalizedEmbeddings(unittest.TestCase):
    def test_normalized_norm_close_to_one(self):
        from main.embed import embed_documents

        model, tokenizer = make_mock_model_tokenizer()
        config = {"batch_size": 4, "normalize": True, "device": "cpu"}
        texts = [f"text {i}" for i in range(5)]
        result = embed_documents(texts, model, tokenizer, config=config)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_unnormalized_norm_not_one(self):
        from main.embed import embed_documents

        model, tokenizer = make_mock_model_tokenizer()
        config = {"batch_size": 4, "normalize": False, "device": "cpu"}
        texts = [f"text {i}" for i in range(5)]
        result = embed_documents(texts, model, tokenizer, config=config)
        norms = np.linalg.norm(result, axis=1)
        self.assertFalse(np.allclose(norms, 1.0, atol=1e-3))


class TestQueryPrefix(unittest.TestCase):
    def test_query_prefix_prepended(self):
        from main.embed import embed_documents

        model, tokenizer = make_mock_model_tokenizer()
        config = {"batch_size": 4, "normalize": True, "device": "cpu"}
        texts = ["find a python developer"]
        embed_documents(texts, model, tokenizer, config=config, query=True)
        call_args = tokenizer.call_args[0][0]
        self.assertEqual(len(call_args), 1)
        self.assertTrue(call_args[0].startswith(QUERY_PREFIX))
        self.assertEqual(call_args[0], QUERY_PREFIX + "find a python developer")

    def test_no_prefix_without_query_flag(self):
        from main.embed import embed_documents

        model, tokenizer = make_mock_model_tokenizer()
        config = {"batch_size": 4, "normalize": True, "device": "cpu"}
        texts = ["find a python developer"]
        embed_documents(texts, model, tokenizer, config=config, query=False)
        call_args = tokenizer.call_args[0][0]
        self.assertEqual(call_args[0], "find a python developer")


if __name__ == "__main__":
    unittest.main()
