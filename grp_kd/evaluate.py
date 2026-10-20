"""Evaluate an exported student with greedy CTC decoding and save WER as JSON."""
import argparse
import hashlib
import json
from pathlib import Path
from grp_kd.data import validate_manifest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Exported student.nemo file")
    parser.add_argument("--manifest", action="append", required=True, metavar="SPLIT=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("--model must name an existing .nemo file")
    manifests = {}
    for item in args.manifest:
        if "=" not in item:
            parser.error("--manifest requires SPLIT=PATH")
        split, path = item.split("=", 1)
        if not split or split in manifests:
            parser.error("Split names must be nonempty and unique")
        manifests[split] = validate_manifest(path)

    import lightning as pl
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(args.model), map_location="cpu")
    model.change_decoding_strategy(OmegaConf.create({"strategy": "greedy_batch"}))
    model.eval()
    trainer = pl.Trainer(accelerator=args.accelerator, devices=1, logger=False,
                         enable_checkpointing=False)
    report = {"model": str(args.model.resolve()), "model_sha256": sha256(args.model),
              "decoding": "greedy_ctc", "wer_unit": "fraction", "splits": {}}
    for split, info in manifests.items():
        model.setup_test_data(OmegaConf.create({
            "manifest_filepath": info["path"], "sample_rate": 16000,
            "batch_size": args.batch_size, "shuffle": False, "num_workers": 0,
            "is_tarred": False}))
        metrics = trainer.test(model, dataloaders=model.test_dataloader(), verbose=False)[0]
        wer = metrics.get("test_wer", metrics.get("wer"))
        if wer is None:
            raise RuntimeError(f"No WER metric returned for {split}: {metrics}")
        report["splits"][split] = {**info, "manifest_sha256": sha256(info["path"]), "wer": float(wer)}
        print(f"{split}: WER = {float(wer):.2%}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
