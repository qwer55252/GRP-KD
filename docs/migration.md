# Research snapshot and public layout

The original research tree is preserved in Git at commit `b995ba3`. This release removes exploratory scripts and notebooks from the active root and retains the full vendored NeMo source under `third_party/NeMo`. No Git history is rewritten.

| Original location | Public location |
| --- | --- |
| `asr_train_diffm.py` model classes | `grp_kd/models.py` |
| `asr_train_diffm.py` training orchestration | `grp_kd/train.py` |
| `scripts/train/diffm_libri100_ver5.sh` | `cmd/train_ver5_ls100.sh` + `configs/ls100.json` |
| `scripts/train/diffm_libri960_ver5.sh` | `cmd/train_ver5_ls960.sh` + `configs/ls960.json` |
| `asr_inference_diffm.py` | `grp_kd/evaluate.py` for exported students |
| Dataset download scripts | Explicit manifest preparation; see `docs/data.md` |
| `NeMo/` | `third_party/NeMo/` (contents unchanged) |
| Exploratory baselines, domain-shift scripts, notebooks | Original Git snapshot |

The public training entry point covers the selected LibriSpeech ver5 recipe. It does not claim to replace every historical experiment. Model computation is retained; path handling, data ingestion, logging, seeding, and export are separated from it.

To inspect an original file without changing your checkout:

```bash
git show b995ba3:scripts/train/diffm_libri100_ver5.sh
git show b995ba3:asr_train_diffm.py
```

The repository was renamed from `KD-via-FM-in-ASR` to `GRP-KD`. Update an existing clone:

```bash
git remote set-url origin https://github.com/qwer55252/GRP-KD.git
```
