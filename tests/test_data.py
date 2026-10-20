import json
from pathlib import Path
import tempfile
import unittest
from grp_kd.data import validate_manifest


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.audio = self.root / "sample.flac"
        self.audio.touch()
        self.manifest = self.root / "manifest.jsonl"
        self.row = {"audio_filepath": str(self.audio), "duration": 3.6, "text": "hello world"}

    def write(self, row):
        self.manifest.write_text(json.dumps(row) + "\n")

    def test_valid_manifest_reports_hours(self):
        self.write(self.row)
        result = validate_manifest(self.manifest)
        self.assertEqual(result["utterances"], 1)
        self.assertAlmostEqual(result["hours"], 0.001)

    def test_missing_audio_has_line_context(self):
        self.audio.unlink()
        self.write(self.row)
        with self.assertRaisesRegex(ValueError, r"manifest.jsonl:1:"):
            validate_manifest(self.manifest)

    def test_invalid_durations_are_rejected(self):
        for duration in (0, -1, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                self.write({**self.row, "duration": duration})
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    validate_manifest(self.manifest)

    def test_empty_manifest_is_rejected(self):
        self.manifest.write_text("")
        with self.assertRaisesRegex(ValueError, "Empty manifest"):
            validate_manifest(self.manifest)

    def test_unnormalized_text_is_rejected(self):
        self.write({**self.row, "text": "HELLO"})
        with self.assertRaisesRegex(ValueError, "normalize"):
            validate_manifest(self.manifest)

    def test_relative_paths_are_rejected(self):
        self.write({**self.row, "audio_filepath": "sample.flac"})
        with self.assertRaisesRegex(ValueError, "absolute"):
            validate_manifest(self.manifest)
