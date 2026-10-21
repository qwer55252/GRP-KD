# GRP-KD

**Knowledge Distillation via Generative Reconstruction Pathways for End-to-End Automatic Speech Recognition**

**ICASSP 2026** · Sang-Heon Jeong*, Dong-Hyun Kim*, and Joon-Hyuk Chang

Hanyang University · *Equal contribution

[Paper](https://ieeexplore.ieee.org/abstract/document/11463778) · [Setup](docs/setup.md) · [Data](docs/data.md) · [Reproduction guide](docs/reproducibility.md) · [Paper results](docs/results.md) · [Citation](#citation)

GRP-KD transfers representations from a Conformer-CTC teacher to a compact student through generative reconstruction with flow matching and diffusion. The paper reports a **3.7M-parameter student** compared with a **13.2M-parameter teacher**. Reconstruction modules are used for training; the exported student performs ordinary greedy CTC recognition.

> **Checkpoint availability:** Pretrained GRP-KD checkpoints are not distributed. Train the provided **ver5** recipe to create your own checkpoint, then evaluate the exported student. The pretrained NeMo teacher is downloaded automatically on the first training run. Full training is not a ten-minute quickstart.
>
> **Reproduction status:** The release preserves the author-selected ver5 model computation. Paper/code differences and outstanding GPU validation are documented in the [reproduction guide](docs/reproducibility.md). The WERs below are published results, not measurements from this cleanup.

## Reported results

WER (%) from Table 1; lower is better.

| Training data | LS test-clean | LS test-other | GS dev | GS test |
| --- | ---: | ---: | ---: | ---: |
| LibriSpeech 100h | 11.42 | 27.79 | 44.19 | 41.76 |
| LibriSpeech 960h | 7.14 | 16.49 | 35.58 | 33.37 |

See [all reported results](docs/results.md) for baselines, domain shift, and ablations.

## Setup

Use Linux and an NVIDIA GPU for training. The paper used one RTX 4090, batch size 32, and 100 epochs. The commands below define a **candidate environment**, not a recovered lockfile from the original experiments.

```bash
git clone https://github.com/qwer55252/GRP-KD.git
cd GRP-KD
conda env create -f environment.yml
conda activate grp-kd
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c constraints.txt -r requirements.txt
```

The official NeMo 2.5.0 package is pinned explicitly; the historical bundled tree is incomplete and is not installed. See [setup and validation](docs/setup.md) before launching a full experiment.

## Prepare LibriSpeech

Download and extract the official [LibriSpeech archives](https://www.openslr.org/12), then create manifests:

```bash
python cmd/prepare_librispeech.py --root /path/to/LibriSpeech \
  --splits train-clean-100 dev-clean dev-other test-clean test-other \
  --output-dir data/manifests
```

The manifest builder checks audio sample rate and channels, and lowercases transcripts. See [data preparation](docs/data.md) for LS960 and GigaSpeech.

## Train ver5 and create a checkpoint

```bash
# One-batch integration check before full training.
bash cmd/train_ver5_ls100.sh --fast-dev-run

# Full LibriSpeech 100h run.
bash cmd/train_ver5_ls100.sh

# After preparing train-960.jsonl:
bash cmd/train_ver5_ls960.sh
```

A completed run writes:

```text
outputs/ver5-ls100/
├── checkpoints/last.ckpt  # Full training state, including teacher and pathways
├── student.nemo          # Student-only model for evaluation and inference
├── model_config.yaml     # Resolved NeMo architecture and optimizer settings
├── run.json              # Recipe, seed, manifest summaries, and runtime versions
└── metrics/              # Local CSV training logs; no W&B account required
```

Resume with `bash cmd/train_ver5_ls100.sh --resume /path/to/last.ckpt`. The batch size and epoch count can be overridden with `--batch-size` and `--epochs`; overrides change the experiment. The smoke run does not export a checkpoint.

## Evaluate and compare with Table 1

```bash
bash cmd/evaluate_librispeech.sh
python cmd/compare_results.py \
  --results outputs/ver5-ls100/librispeech.json --training-set ls100
```

For LS960:

```bash
MODEL=outputs/ver5-ls960/student.nemo \
RESULTS=outputs/ver5-ls960/librispeech.json \
  bash cmd/evaluate_librispeech.sh
python cmd/compare_results.py \
  --results outputs/ver5-ls960/librispeech.json --training-set ls960
```

Evaluation loads only `student.nemo`, uses greedy CTC decoding without an external language model, and records model/manifest checksums and WER in JSON. Use `--accelerator cpu` for CPU evaluation. Missing splits are explicitly marked as not evaluated in the comparison table.

## Repository layout

```text
GRP-KD/
├── grp_kd/               # Model definitions, training, evaluation, manifest validation
├── cmd/                  # Separate data, training, evaluation, and comparison commands
├── configs/              # Explicit ver5 experiment settings
├── docs/                 # Setup, data, results, provenance, and reproduction limitations
├── tests/                # Lightweight validation tests
├── third_party/NeMo/     # Historical NeMo snapshot and its license notices
├── environment.yml
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

Earlier exploratory scripts and notebooks remain available in [the original research snapshot](https://github.com/qwer55252/GRP-KD/tree/b995ba3). See [migration notes](docs/migration.md) for the mapping to the public entry points.

## License

Original research code is released under [Apache-2.0](LICENSE). Vendored NeMo retains its own [Apache-2.0 license](third_party/NeMo/LICENSE) and notices. Dataset and pretrained teacher terms are governed by their providers.

## Citation

```bibtex
@inproceedings{jeong2026grpkd,
  title     = {Knowledge Distillation via Generative Reconstruction Pathways for End-to-End Automatic Speech Recognition},
  author    = {Jeong, Sang-Heon and Kim, Dong-Hyun and Chang, Joon-Hyuk},
  booktitle = {2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```
