"""Reproducible validation-only hyperparameter sweeps for a single forecasting task.

Example (Windows CMD, working directory is repository root):
  python scripts/tune.py --dataset Weather --pred_len 192 --mode refine \
      --checkpoint checkpoints/pretrained/TimeDO_Weather_192.pt --latent_mode d1 \
      --learning_rates 0.0001 0.0003 --batch_sizes 256 512 \
      --freeze_choices true false --max_trials 8 --execute

Candidate training never evaluates the test split.  Run run.py --mode evaluate
on the selected checkpoint only AFTER tuning concludes, with matching architecture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from timedo.presets import get_config  # noqa: E402


def _bool(raw: str) -> bool:
    if raw.lower() in {"true", "1", "yes"}:
        return True
    if raw.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep without using test for selection")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", type=int, required=True, choices=(96, 192, 336, 720))
    parser.add_argument("--mode", choices=("train", "refine"), default="refine")
    parser.add_argument("--checkpoint", type=Path, help="Required for Weather S4->d1; optional for archived ETTh init")
    parser.add_argument("--latent_mode", choices=("none", "d1", "rank"), default="d1")
    parser.add_argument("--learning_rates", type=float, nargs="+", help="For refine: latent_learning_rate; for train: learning_rate")
    parser.add_argument("--batch_sizes", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, nargs="+", help="Refine epochs or train epochs")
    parser.add_argument("--freeze_choices", type=_bool, nargs="+", help="Refine only, e.g. true false")
    parser.add_argument("--mse_weights", type=float, nargs="+", help="Latent state MSE penalty")
    parser.add_argument("--cos_weights", type=float, nargs="+", help="Latent state cosine penalty")
    parser.add_argument("--evaluation_scale", type=float, help="Fixed validation selection and final inference scale")
    parser.add_argument("--data_root", type=Path, help="Dataset folder or TIMEDO_DATA_ROOT")
    parser.add_argument("--config_dir", type=Path, default=ROOT / "configs", help="Optional config directory for temporary experiments")
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs" / "tuning")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--max_trials", type=int, default=24)
    parser.add_argument("--execute", action="store_true", help="Without this option only list planned experiments")
    return parser


def build_trials(args: argparse.Namespace) -> list[dict]:
    defaults = get_config(args.dataset, args.pred_len, args.mode, config_dir=args.config_dir)
    rates = args.learning_rates or [defaults["latent_learning_rate" if args.mode == "refine" else "learning_rate"]]
    batches = args.batch_sizes or [defaults["batch_size"]]
    epochs = args.epochs or [defaults.get("ref_epochs", 5) if args.mode == "refine" else defaults.get("train_epochs", 30)]
    freezes = args.freeze_choices if args.freeze_choices is not None else [defaults.get("freeze_backbone", False)]
    if args.mode == "train":
        freezes = [False]
    mse_weights = args.mse_weights or [defaults["latent_mse_weight"]]
    cos_weights = args.cos_weights or [defaults["latent_cos_weight"]]
    options = []
    for lr, batch, epoch, freeze, mse, cos in itertools.product(
        rates, batches, epochs, freezes, mse_weights, cos_weights
    ):
        item = {
            "rate": float(lr), "batch_size": int(batch), "epochs": int(epoch),
            "freeze_backbone": bool(freeze), "latent_mse_weight": float(mse),
            "latent_cos_weight": float(cos), "latent_mode": args.latent_mode,
        }
        if item["rate"] <= 0 or item["batch_size"] <= 0 or item["epochs"] <= 0:
            raise ValueError("All rates, batch sizes, and epochs must be positive")
        if item["latent_mse_weight"] < 0 or item["latent_cos_weight"] < 0:
            raise ValueError("Penalty weights must be nonnegative")
        if item not in options:
            options.append(item)
    return options


def write_summary(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fields = ["id", "status", "val_mse", "val_mae", "best_epoch", "mode", "rate", "batch_size", "epochs", "freeze_backbone", "latent_mode", "latent_mse_weight", "latent_cos_weight", "checkpoint", "log"]
    temporary = path / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(temporary, path / "summary.csv")
    ranked = sorted((x for x in rows if x["status"] == "ok"), key=lambda x: x["val_mse"])
    (path / "best_by_validation.json").write_text(json.dumps(ranked[0] if ranked else {}, indent=2), encoding="utf-8")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_trials < 1:
        raise SystemExit("--max_trials must be positive")
    if args.mode == "train" and args.checkpoint:
        raise SystemExit("--checkpoint only works with --mode refine")
    defaults = get_config(args.dataset, args.pred_len, args.mode, config_dir=args.config_dir)
    cp = args.checkpoint
    if args.mode == "refine" and cp is None:
        cp = ROOT / "checkpoints" / "initializations" / f"TimeDO_{args.dataset}_{args.pred_len}.pt"
    if cp is not None:
        cp = cp.expanduser().resolve()
        if not cp.is_file():
            raise SystemExit(f"Missing initialization checkpoint: {cp}")
        digest = hashlib.sha256()
        with cp.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        source_digest = digest.hexdigest()
    else:
        source_digest = "random_initialization"
    scale = args.evaluation_scale if args.evaluation_scale is not None else defaults.get("evaluation_scale", 1.0)
    trials = build_trials(args)
    config_path = args.config_dir.expanduser().resolve() / f"{args.dataset.lower()}.yaml"
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    default_data = ROOT.parent / "Datasets" if (ROOT.parent / "Datasets").is_dir() else ROOT / "data"
    data_path = (args.data_root.expanduser().resolve() if args.data_root else Path(os.getenv("TIMEDO_DATA_ROOT", default_data)).expanduser().resolve())
    if len(trials) > args.max_trials:
        raise SystemExit(f"Refusing {len(trials)} experiments (> --max_trials={args.max_trials}). Narrow the grid or raise the cap explicitly.")
    root = args.output_root.expanduser().resolve() / args.dataset / f"pred_{args.pred_len}"
    print(f"Planned {len(trials)} {args.mode} runs for {args.dataset}/{args.pred_len}; validation-only selection")
    for number, trial in enumerate(trials, 1):
        print(f"  [{number:02d}] {trial}")
    if not args.execute:
        print("DRY RUN: add --execute to train candidates; no dataset accessed and no weights changed")
        return
    rows=[]
    for number, trial in enumerate(trials, 1):
        identifier = hashlib.sha256(json.dumps({**trial,"source_sha":source_digest,"selection_scale":scale,"config_sha":config_sha,"data_root":str(data_path)},sort_keys=True).encode()).hexdigest()[:12]
        target = root / f"trial_{identifier}"
        result_path = target / args.dataset / f"pred_{args.pred_len}" / args.mode / "result.json"
        cp_path = result_path.with_name("checkpoint.pt")
        log_path = target / "stdout.log"
        entry = {**trial,"mode":args.mode,"id":identifier,"status":"unknown","val_mse":float('inf'),"val_mae":float('inf'),"best_epoch":"","checkpoint":str(cp_path),"log":str(log_path)}
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("test") is None and cp_path.is_file():
                print(f"[{number}/{len(trials)}] Reusing {identifier}")
                entry.update(status="ok",val_mse=payload["validation"]["mse"],val_mae=payload["validation"]["mae"],best_epoch=payload["best_epoch"])
                rows.append(entry);write_summary(root,rows);continue
        command=[sys.executable,str(ROOT/"run.py"),"--dataset",args.dataset,"--pred_len",str(args.pred_len),
                 "--mode",args.mode,"--device",args.device,"--output_root",str(target),
                 "--latent_mode",args.latent_mode,"--batch_size",str(trial["batch_size"]),
                 "--latent_mse_weight",str(trial["latent_mse_weight"]),
                 "--latent_cos_weight",str(trial["latent_cos_weight"]),
                 "--checkpoint_policy","validation","--save_epoch_snapshots","--skip_test",
                 "--config_dir",str(args.config_dir.expanduser().resolve()),
                 "--selection_scale",str(scale),"--evaluation_scale",str(scale),
                 "--freeze_backbone",str(trial["freeze_backbone"]).lower()]
        command += (["--latent_learning_rate",str(trial["rate"]),"--ref_epochs",str(trial["epochs"])] if args.mode == "refine"
                    else ["--learning_rate",str(trial["rate"]),"--train_epochs",str(trial["epochs"])])
        if cp is not None: command += ["--checkpoint",str(cp)]
        if args.data_root:command += ["--data_root",str(data_path)]
        target.mkdir(parents=True,exist_ok=True)
        print(f"[{number}/{len(trials)}] Training {identifier}", flush=True)
        with log_path.open("w",encoding="utf-8") as out:
            result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=False)
        if result.returncode != 0 or not result_path.exists():
            entry["status"]="failed"
            rows.append(entry);write_summary(root,rows)
            raise RuntimeError(f"Trial {identifier} failed; inspect {log_path}")
        payload=json.loads(result_path.read_text(encoding="utf-8"))
        if payload["test"] is not None:
            raise RuntimeError(f"Trial {identifier} unexpectedly evaluated test split")
        entry.update(status="ok",val_mse=payload["validation"]["mse"],val_mae=payload["validation"]["mae"],best_epoch=payload["best_epoch"])
        print(f"  VAL MSE={entry['val_mse']:.9f} best_epoch={entry['best_epoch']}",flush=True)
        rows.append(entry);write_summary(root,rows)
    good=sorted([r for r in rows if r["status"]=="ok"],key=lambda r:r["val_mse"])
    if good:
        print(f"Best by validation: {good[0]['id']} MSE={good[0]['val_mse']:.9f}\nCheckpoint: {good[0]['checkpoint']}")
        print("No candidate test metrics computed. Evaluate the selected checkpoint separately.")

if __name__ == "__main__":
    main()
