"""Validate the published checkpoints and compact release manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from timedo.config import ModelConfig
from timedo.models.forecaster import TimeDO
from timedo.models.encoder import parse_dimensions
from timedo.presets import get_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = ROOT / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("pretrained", [])
    if len(entries) != int(manifest.get("item_count", len(entries))):
        raise AssertionError("manifest item_count does not match pretrained entries")

    for entry in entries:
        path = ROOT / Path(entry["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise AssertionError(f"{entry['record_id']} SHA256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
            raise AssertionError(f"{entry['record_id']} is not a self-describing checkpoint")
        config = payload.get("model_config")
        if not isinstance(config, dict):
            raise AssertionError(f"{entry['record_id']} has no model_config")
        if int(config.get("pred_len", -1)) != int(entry["pred_len"]):
            raise AssertionError(f"{entry['record_id']} horizon mismatch")
        resolved = get_config(entry["dataset"], int(entry["pred_len"]))
        model_config = ModelConfig.from_namespace({**resolved, "enc_in": int(config["enc_in"])})
        for name in (
            "seq_len", "pred_len", "enc_in", "period_len", "basis_num",
            "householder_reflections", "use_period_norm",
            "individual", "latent_mode",
        ):
            if getattr(model_config, name) != config[name]:
                raise AssertionError(f"{entry['record_id']} config mismatch for {name}")
        segments = (int(config['seq_len']) + int(config['period_len']) - 1) // int(config['period_len'])
        saved_dims = parse_dimensions(segments, int(config['basis_num']), config['time_do_dims'])
        resolved_dims = parse_dimensions(segments, model_config.basis_num, model_config.time_do_dims)
        if saved_dims != resolved_dims:
            raise AssertionError(f"{entry['record_id']} encoder dimensions mismatch")
        model = TimeDO(model_config)
        model.load_state_dict(payload["model_state"], strict=True)
        print(f"OK {entry['record_id']:14s} SHA256={actual[:16]}...")

    for entry in manifest.get("initializations", []):
        path = ROOT / Path(entry["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != entry["sha256"]:
            raise AssertionError(f"initialization {entry['file']} SHA256 mismatch")

    print(
        f"PASS: {len(entries)} pretrained checkpoints and "
        f"{len(manifest.get('initializations', []))} initialization checkpoints verified."
    )


if __name__ == "__main__":
    main()
