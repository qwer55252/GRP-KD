import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReportTests(unittest.TestCase):
    def test_fraction_conversion_and_missing_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "result.json"
            report.write_text(json.dumps({"wer_unit": "fraction", "splits": {"test-clean": {"wer": 0.12}}}))
            result = subprocess.run([sys.executable, "cmd/compare_results.py", "--results", str(report),
                                     "--training-set", "ls100"], capture_output=True, text=True, check=True)
            self.assertIn("| test-clean | 11.42 | 12.00 | +0.58 |", result.stdout)
            self.assertIn("Not evaluated", result.stdout)

    def test_percent_input_is_not_misinterpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "result.json"
            report.write_text(json.dumps({"wer_unit": "percent", "splits": {}}))
            result = subprocess.run([sys.executable, "cmd/compare_results.py", "--results", str(report),
                                     "--training-set", "ls100"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
