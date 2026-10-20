# Reproducing the experiments

## Checkpoints and recipe selection

**Train checkpoints locally using the provided ver5 scripts. No pretrained GRP-KD weights are distributed.** The author selected the saved ver5 recipe as the checkpoint creation path for this release.

The public commands are `cmd/train_ver5_ls100.sh` and `cmd/train_ver5_ls960.sh`. Their configurations preserve the original saved scripts' effective settings: 100 epochs, batch size 32, 8 FM steps, 9 diffusion steps, latent dimension 96, logit KD temperature 1, and weight 0.1. The optimizer and scheduler are inherited from the pretrained teacher configuration, as in the original training script, and saved in `model_config.yaml`.

The new entry points use explicit manifests, a default seed of 42, local CSV logs, temporary tokenizer extraction directories, and a student-only export. The original scripts did not specify a seed. These workflow changes have not yet been validated on a GPU. Use the original snapshot when investigating historical run behavior.

## Paper-to-command map

| Paper item | Recipe / command | Status |
| --- | --- | --- |
| Table 1, GRP-KD LS100 | `bash cmd/train_ver5_ls100.sh`, then `bash cmd/evaluate_librispeech.sh` | Author-selected ver5; full rerun pending |
| Table 1, GRP-KD LS960 | `bash cmd/train_ver5_ls960.sh`, then LS960 evaluation in README | Author-selected ver5; full rerun pending |
| Table 1, GS dev/test | `python -m grp_kd.evaluate` with `gs-dev` / `gs-test` manifests | Historical preprocessing must be matched |
| Table 1, baselines | Historical `asr_train.py` and `scripts/train/` | Not migrated or revalidated |
| Table 2, GS XS domain shift | Historical `asr_train_diffm_GS.py` and `DS_GSxs_diffm_ver5.sh` | Separate training behavior; not represented by LS configs |
| Table 3, ablations | Historical numbered variants and baseline scripts | Final row mapping requires confirmation |
| Figure 1 | Architecture illustration in the paper | No experiment required |
| Figure 2 | Feature heatmaps and UMAP | Exact inputs, extraction layer, and plotting recipe not supplied |

The comparison command reports observed WER beside published values. It does not synthesize scores or declare successful reproduction based on configuration alone.

## Scientific differences requiring explicit interpretation

The model code is preserved from research commit `b995ba3`, with English documentation. The following differences were found during inspection and have **not** been silently corrected:

1. **Path topology.** The paper describes parallel FM and diffusion objectives. The saved `ver5` implementation computes student projection → noise adapter → denoiser → FM and a post-FM loss, plus teacher reconstruction. `ver4` has separate pre-FM and post-diffusion losses. The release uses the author-selected `ver5`; it does not relabel `ver4` as the paper model.
2. **Layer reduction.** The paper averages losses over layers; the implementation sums the reconstruction and pathway terms over layers. Changing this changes the effective objective weights.
3. **FM objective.** The implementation's loss uses its final predicted velocity and a shape transformation of the original student input. It does not directly compute MSE on the final Euler-transported state as the paper equation suggests. Euler times descend from 1 to 1/K.
4. **Teacher mode.** Teacher parameters are frozen, but the original distillation model registers the teacher as a submodule and does not override `train()` to keep it in evaluation mode. This historical behavior remains unchanged.
5. **Environment and measurements.** The bundled NeMo tree is missing its ASR `parts` package. Setup uses the official 2.5.0 package as a candidate replacement. No original dependency lockfile, run seeds, or released final checkpoints are present. The candidate environment and new workflow require a complete GPU run before numerical equivalence can be asserted.

## Inference and checkpoint formats

`last.ckpt` is a Lightning training checkpoint and includes the teacher and reconstruction modules. `student.nemo` contains only the standard NeMo CTC student. They are not interchangeable. The new evaluator accepts the student export, checks all student state keys during export, and never instantiates a GRP-KD teacher or pathway module.

The export uses the final in-memory trained state. `last.ckpt` is the last training state, not a best-validation checkpoint. Resuming uses the supplied checkpoint for training and exports the final state after continuation.

## Before claiming a reproduced paper result

Run the GPU integration check, complete training, record the environment and revision, evaluate the same data splits with the original normalization, and retain JSON results and checksums. Resolve the scientific differences above when interpreting any gap from the paper. The original published scores remain reference values until measured again.
