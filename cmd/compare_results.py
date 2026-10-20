#!/usr/bin/env python3
"""Compare measured WER with the GRP-KD rows of paper Table 1."""
import argparse
import json
from pathlib import Path

PAPER = {
    "ls100": {"dev-clean": 11.18, "dev-other": 27.22, "test-clean": 11.42,
              "test-other": 27.79, "gs-dev": 44.19, "gs-test": 41.76},
    "ls960": {"dev-clean": 7.02, "dev-other": 16.31, "test-clean": 7.14,
              "test-other": 16.49, "gs-dev": 35.58, "gs-test": 33.37},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--training-set", choices=PAPER, required=True)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    if report.get("wer_unit") != "fraction":
        parser.error("Expected an evaluation report with wer_unit='fraction'")
    print("| Split | Paper WER (%) | Measured WER (%) | Difference (pp) |")
    print("| --- | ---: | ---: | ---: |")
    for split, target in PAPER[args.training_set].items():
        row = report["splits"].get(split)
        if row is None:
            print(f"| {split} | {target:.2f} | Not evaluated | — |")
        else:
            value = 100 * row["wer"]
            print(f"| {split} | {target:.2f} | {value:.2f} | {value-target:+.2f} |")


if __name__ == "__main__":
    main()
