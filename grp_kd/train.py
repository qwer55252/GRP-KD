"""Train a historical reconstruction variant from portable NeMo manifests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from grp_kd.data import validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.config.read_text())
    if args.epochs is not None:
        recipe["epochs"] = args.epochs
    if args.batch_size is not None:
        recipe["batch_size"] = args.batch_size
    manifests = [validate_manifest(args.train_manifest), validate_manifest(args.val_manifest)]
    if args.resume and not args.resume.is_file():
        parser.error("--resume must name an existing Lightning checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import lightning as pl
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    import nemo.collections.asr as nemo_asr
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    from nemo.utils.app_state import AppState
    from omegaconf import OmegaConf
    from grp_kd.models import DistilFlowMatchingCTCModelBPE

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA GPU with a CUDA-enabled PyTorch installation.")
    pl.seed_everything(args.seed, workers=True)
    metadata = {"recipe": recipe, "seed": args.seed, "manifests": manifests,
                "torch": torch.__version__, "lightning": pl.__version__,
                "fast_dev_run": args.fast_dev_run, "resume": str(args.resume) if args.resume else None}
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checkpoint = ModelCheckpoint(dirpath=str(args.output_dir / "checkpoints"),
                                 save_last=True, save_top_k=0)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=recipe["epochs"],
                         logger=CSVLogger(str(args.output_dir), name="metrics"),
                         callbacks=[checkpoint], fast_dev_run=args.fast_dev_run)
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=recipe["teacher"], map_location="cuda:0", trainer=trainer)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    # Keep tokenizer artifacts available while constructing and exporting models.
    with tempfile.TemporaryDirectory(prefix="grpkd-nemo-") as archive:
        source = AppState().get_model_metadata_from_guid(teacher.model_guid).restoration_path
        SaveRestoreConnector()._unpack_nemo_file(source, out_folder=archive)
        teacher._save_restore_connector.model_extracted_dir = archive
        AppState().nemo_file_folder = archive
        cfg = deepcopy(teacher.cfg)
        # New models need local tokenizer artifacts for a self-contained export.
        # TAR_PATH artifacts alone have no restoration archive in a new student.
        for key in teacher.artifacts:
            source_path = str(OmegaConf.select(cfg, key))
            artifact = Path(archive) / Path(source_path.removeprefix("nemo:")).name
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing teacher artifact: {artifact}")
            OmegaConf.update(cfg, key, str(artifact))
        for split, manifest in (("train_ds", args.train_manifest),
                                ("validation_ds", args.val_manifest),
                                ("test_ds", args.val_manifest)):
            ds = cfg[split]
            ds.is_tarred = False
            ds.tarred_audio_filepaths = None
            ds.manifest_filepath = str(manifest.resolve())
            ds.sample_rate = 16000
            ds.batch_size = recipe["batch_size"]
        cfg.encoder.d_model //= 2
        cfg.encoder.n_heads //= 2
        cfg.decoder.feat_in //= 2
        OmegaConf.save(cfg, args.output_dir / "model_config.yaml")
        flow_cfg = {"meta_encoder_type": "mlp", "hidden_dim": recipe["latent_dim"],
                    "training_sampling": recipe["flow_steps"],
                    "inference_sampling": recipe["flow_steps"],
                    "shape_transform": "linear", "noise_schedule": "rectified"}
        model = DistilFlowMatchingCTCModelBPE(
            cfg, trainer, teacher, version=recipe["model_version"],
            student_dim=cfg.encoder.d_model, teacher_dim=teacher.cfg.encoder.d_model,
            latent_dim=recipe["latent_dim"], kd_alpha=recipe["kd_alpha"],
            kd_temperature=recipe["kd_temperature"], flow_cfg=flow_cfg,
            diffkd_cfg={"diffusion_steps": recipe["diffusion_steps"]})
        trainer.fit(model, ckpt_path=str(args.resume.resolve()) if args.resume else None)
        if args.fast_dev_run:
            return
        # Export only the ordinary CTC model. Teacher and pathway weights stay in .ckpt.
        export_cfg = deepcopy(cfg)
        for split in ("train_ds", "validation_ds", "test_ds"):
            export_cfg[split].manifest_filepath = None
        student = nemo_asr.models.EncDecCTCModelBPE(cfg=export_cfg)
        state = model.state_dict()
        student.load_state_dict({key: state[key] for key in student.state_dict()}, strict=True)
        student.save_to(str(args.output_dir / "student.nemo"))
        print(f"Student exported to {args.output_dir / 'student.nemo'}")


if __name__ == "__main__":
    main()
