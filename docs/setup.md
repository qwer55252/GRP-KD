# Environment and validation

The paper reports a single NVIDIA RTX 4090. Linux with an NVIDIA GPU is the training target. macOS is suitable for documentation and lightweight tests; CUDA training is unavailable there.

## Candidate environment

Follow the commands in the [README](../README.md#setup). Python 3.10, PyTorch/torchaudio 2.5.1 with CUDA 12.4, Lightning 2.4.0, and Transformers 4.53.3 form a candidate environment based on the official NeMo 2.5.0 dependency requirements. They are **not verified original experiment versions**. The exact original environment was not included in the research snapshot.

PyTorch installation commands come from the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/). The bundled tree identifies itself as `2.5.0rc0` but is incomplete: `nemo/collections/asr/parts` is missing. It cannot serve as the installable dependency. Setup therefore pins the [official NeMo 2.5.0 release](https://pypi.org/project/nemo-toolkit/2.5.0/). This replacement is a compatibility candidate, not proof of equivalence to the original runtime.

Run these checks after installation:

```bash
python -m pip check
python -c "import torch, nemo, lightning; print(torch.__version__, torch.cuda.is_available(), nemo.__file__, lightning.__version__)"
python -m grp_kd.train --help
python -m grp_kd.evaluate --help
python -m unittest discover -s tests -v
```

`nemo.__file__` should point into your environment's installed package, not the historical tree. The first training run downloads `stt_en_conformer_ctc_small` through NeMo. Network access and writable cache storage are required.

The development container retains the research repository's `nvcr.io/nvidia/nemo:25.02` image as an alternative starting point. It does not automatically install this package or certify the environment. Install `requirements.txt` with `constraints.txt` inside it and run the checks above.

## GPU integration check

Prepare real train/validation manifests and run:

```bash
bash cmd/train_ver5_ls100.sh --fast-dev-run
```

This executes one training/validation batch, downloads the teacher if necessary, and checks the integration. It does not validate final WER, save an export, or replace a complete training run. Use a separate output directory for trials with changed hyperparameters.

After a successful full run, record the complete environment:

```bash
python -m pip freeze > outputs/ver5-ls100/requirements-resolved.txt
nvidia-smi > outputs/ver5-ls100/gpu.txt
git rev-parse HEAD > outputs/ver5-ls100/revision.txt
```

## Validation performed during repository cleanup

The release was prepared on a Mac without PyTorch/NeMo or a CUDA GPU. Syntax checks, lightweight data/report tests, CLI help, and preservation of the model computation were checked locally. Dependency installation, GPU training, student export, and NeMo evaluation still require the integration check above. No new WER measurements are claimed.
