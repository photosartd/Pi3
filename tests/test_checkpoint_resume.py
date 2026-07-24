import tempfile
import unittest
from pathlib import Path

from trainers.base_trainer_accelerate import (
    checkpoint_epoch_from_path,
    latest_accelerate_checkpoint,
    next_epoch_after_checkpoint,
    resolve_resume_checkpoint,
)


class CheckpointResumeTest(unittest.TestCase):
    def test_checkpoint_epoch_parsing(self):
        self.assertEqual(checkpoint_epoch_from_path("/tmp/ckpts/checkpoint_9"), 9)
        self.assertEqual(checkpoint_epoch_from_path("checkpoint_0003"), 3)
        self.assertIsNone(checkpoint_epoch_from_path("/tmp/ckpts/best_model"))
        self.assertIsNone(checkpoint_epoch_from_path("/tmp/ckpts/checkpoint_latest"))

    def test_latest_checkpoint_uses_highest_epoch_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint_2").mkdir()
            (root / "checkpoint_10").mkdir()
            (root / "checkpoint_bad").mkdir()
            (root / "best_model").mkdir()
            self.assertEqual(latest_accelerate_checkpoint(root), str(root / "checkpoint_10"))

    def test_resume_resolution_honors_explicit_auto_and_disabled_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint_0").mkdir()
            (root / "checkpoint_4").mkdir()

            self.assertEqual(resolve_resume_checkpoint(root, resume=None, auto_resume=True), str(root / "checkpoint_4"))
            self.assertIsNone(resolve_resume_checkpoint(root, resume=None, auto_resume=False))
            self.assertEqual(resolve_resume_checkpoint(root, resume="auto", auto_resume=False), str(root / "checkpoint_4"))
            self.assertEqual(resolve_resume_checkpoint(root, resume="/tmp/exact/checkpoint_1"), "/tmp/exact/checkpoint_1")

    def test_checkpoint_resume_starts_after_saved_epoch(self):
        self.assertEqual(next_epoch_after_checkpoint("/tmp/ckpts/checkpoint_0"), 1)
        self.assertEqual(next_epoch_after_checkpoint("/tmp/ckpts/checkpoint_29"), 30)
        self.assertEqual(next_epoch_after_checkpoint("/tmp/ckpts/best_model"), 0)


if __name__ == "__main__":
    unittest.main()
