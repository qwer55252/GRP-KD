#!/usr/bin/env python3
"""Build manifests from an extracted official LibriSpeech directory."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing train-clean-100, etc.")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import soundfile as sf
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        transcripts = sorted((args.root / split).rglob("*.trans.txt"))
        if not transcripts:
            raise FileNotFoundError(f"No transcripts found for {args.root / split}")
        output = args.output_dir / f"{split}.jsonl"
        temporary = output.with_suffix(".tmp")
        count = 0
        with temporary.open("w", encoding="utf-8") as stream:
            for transcript in transcripts:
                for line in transcript.read_text().splitlines():
                    utterance, text = line.split(maxsplit=1)
                    audio = transcript.parent / f"{utterance}.flac"
                    info = sf.info(audio)
                    if info.samplerate != 16000 or info.channels != 1:
                        raise ValueError(f"Expected mono 16 kHz audio: {audio}")
                    row = {"audio_filepath": str(audio.resolve()), "duration": info.duration,
                           "text": text.lower().strip()}
                    stream.write(json.dumps(row) + "\n")
                    count += 1
        temporary.replace(output)
        print(f"{output}: {count} utterances")


if __name__ == "__main__":
    main()
