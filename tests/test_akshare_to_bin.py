import tempfile
import unittest
from pathlib import Path

from examples.akshareToBin import collect_parquet_files, default_qlib_dir


class TestAkshareToBin(unittest.TestCase):
    def test_collect_parquet_files_from_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir)
            first = src_dir / "first.parquet"
            second = src_dir / "second.parquet"
            ignored = src_dir / "ignored.csv"
            first.touch()
            second.touch()
            ignored.touch()

            self.assertEqual(collect_parquet_files(src_dir), [first, second])

    def test_default_qlib_dir_for_directory_is_under_source_directory(self):
        src_dir = Path("qlib") / "data" / "20260506"

        self.assertEqual(default_qlib_dir(src_dir), src_dir / "qlib_data")

    def test_default_qlib_dir_for_file_is_under_parent_directory(self):
        src_file = Path("qlib") / "data" / "20260506" / "301217.parquet"

        self.assertEqual(default_qlib_dir(src_file), src_file.parent / "qlib_data")


if __name__ == "__main__":
    unittest.main()
