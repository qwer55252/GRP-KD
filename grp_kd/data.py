"""Validate NeMo manifests without importing the training stack."""
import json
import math
from pathlib import Path


def validate_manifest(path):
    """Require nonempty JSONL with absolute audio paths and normalized transcripts."""
    path = Path(path).resolve(strict=True)
    count = 0
    seconds = 0.0
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                audio = Path(row["audio_filepath"])
                duration = float(row["duration"])
                text = row["text"]
                if not audio.is_absolute() or not audio.is_file():
                    raise ValueError("audio_filepath must name an existing absolute file")
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("duration must be finite and positive")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text must be a nonempty transcript")
                if text != text.lower().strip():
                    raise ValueError("normalize text to lowercase with no surrounding whitespace")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            count += 1
            seconds += duration
    if not count:
        raise ValueError(f"Empty manifest: {path}")
    return {"path": str(path), "utterances": count, "hours": seconds / 3600}
