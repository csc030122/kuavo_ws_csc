from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from yolo_catchdata.output_lock import acquire_output_lock


class OutputLockTest(unittest.TestCase):
    def test_second_writer_is_rejected_and_lock_releases_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            first = acquire_output_lock(root, "first")
            holder = json.loads((root / ".collection.lock").read_text())
            self.assertEqual(holder["owner"], "first")
            with self.assertRaisesRegex(RuntimeError, "已有采集进程持锁"):
                acquire_output_lock(root, "second")
            first.close()
            second = acquire_output_lock(root, "second")
            second.close()


if __name__ == "__main__":
    unittest.main()
