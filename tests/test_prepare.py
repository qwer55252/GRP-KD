import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("soundfile"), "soundfile is needed for audio fixture tests")
class LibriSpeechPreparationTests(unittest.TestCase):
    def run_fixture(self, channels):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chapter = root / "LibriSpeech" / "test-clean" / "1" / "2"
            chapter.mkdir(parents=True)
            (chapter / "1-2.trans.txt").write_text("1-2-0001 HELLO WORLD\n")
            shape = (16000,) if channels == 1 else (16000, channels)
            sf.write(chapter / "1-2-0001.flac", np.zeros(shape), 16000)
            output = root / "manifests"
            result = subprocess.run([sys.executable, "cmd/prepare_librispeech.py", "--root",
                                     str(root / "LibriSpeech"), "--splits", "test-clean",
                                     "--output-dir", str(output)], capture_output=True, text=True)
            if channels == 1:
                self.assertEqual(result.returncode, 0, result.stderr)
                row = json.loads((output / "test-clean.jsonl").read_text())
                self.assertEqual(row["text"], "hello world")
                self.assertEqual(row["duration"], 1.0)
                self.assertTrue(Path(row["audio_filepath"]).is_absolute())
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Expected mono 16 kHz audio", result.stderr)
                self.assertFalse((output / "test-clean.jsonl").exists())

    def test_real_flac_manifest(self):
        self.run_fixture(1)

    def test_stereo_input_is_rejected(self):
        self.run_fixture(2)
