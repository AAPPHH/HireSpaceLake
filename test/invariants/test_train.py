import os, sys, unittest, importlib
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

for mod_name in [
    "datasets",
    "sentence_transformers",
    "sentence_transformers.training_args",
    "sentence_transformers.losses",
    "peft",
    "bitsandbytes",
    "accelerate",
]:
    if mod_name not in sys.modules:
        mock = MagicMock()
        mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
        sys.modules[mod_name] = mock

from sentence_transformers import InputExample

InputExample = type("InputExample", (), {"__init__": lambda self, texts=None, label=0.0: setattr(self, "texts", texts) or setattr(self, "label", label)})
sys.modules["sentence_transformers"].InputExample = InputExample

from main.config import TRAIN_CONFIG
from main.train import create_training_pairs


class TestLoraConfig(unittest.TestCase):
    def test_rank_and_alpha(self):
        self.assertEqual(TRAIN_CONFIG["lora_rank"], 16)
        self.assertEqual(TRAIN_CONFIG["lora_alpha"], 32)
        self.assertEqual(TRAIN_CONFIG["lora_target_modules"], ["q_proj", "v_proj"])
        self.assertEqual(TRAIN_CONFIG["quantization_bits"], 4)


class TestCreateTrainingPairs(unittest.TestCase):
    def test_reads_from_ducklake(self):
        con = MagicMock()
        mock_pairs = [
            {"cv_id": 0, "job_id": 1, "label": 0.85, "source": "dataset", "created_at": None},
            {"cv_id": 2, "job_id": 3, "label": 0.6, "source": "cluster", "created_at": None},
        ]
        mock_docs = [
            {"id": 0, "type": "cv", "text": "Python developer with 5 years experience"},
            {"id": 1, "type": "job", "text": "Senior Python engineer needed"},
            {"id": 2, "type": "cv", "text": "Java developer fresh graduate"},
            {"id": 3, "type": "job", "text": "Junior Java developer position"},
        ]
        with patch("main.train.get_pairs", return_value=mock_pairs), \
             patch("main.train.get_documents", return_value=mock_docs):
            examples = create_training_pairs(con)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].texts[0], "Python developer with 5 years experience")
        self.assertEqual(examples[0].texts[1], "Senior Python engineer needed")
        self.assertAlmostEqual(examples[0].label, 0.85)
        self.assertAlmostEqual(examples[1].label, 0.6)

    def test_empty_pairs(self):
        con = MagicMock()
        with patch("main.train.get_pairs", return_value=[]):
            examples = create_training_pairs(con)
        self.assertEqual(examples, [])

    def test_missing_doc(self):
        con = MagicMock()
        mock_pairs = [
            {"cv_id": 0, "job_id": 99, "label": 0.5, "source": "dataset", "created_at": None},
        ]
        mock_docs = [
            {"id": 0, "type": "cv", "text": "Some CV"},
        ]
        with patch("main.train.get_pairs", return_value=mock_pairs), \
             patch("main.train.get_documents", return_value=mock_docs):
            examples = create_training_pairs(con)
        self.assertEqual(len(examples), 0)


if __name__ == "__main__":
    unittest.main()
