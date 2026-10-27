"""Single source of truth: one portable YAML file per forecasting dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
HORIZONS = (96, 192, 336, 720)
DATASETS = (
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Solar",
    "Electricity", "Traffic", "Weather",
)


def load_dataset_config(dataset: str, config_dir: Path | None = None) -> dict[str, Any]:
    candidates = {name.lower(): name for name in DATASETS}
    try:
        canonical = candidates[dataset.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}; options: {DATASETS}") from exc
    root = Path(config_dir) if config_dir is not None else CONFIG_DIR
    path = root / f"{canonical.lower()}.yaml"
    with path.open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict) or spec.get("dataset") != canonical:
        raise ValueError(f"Malformed/mismatched dataset YAML: {path}")
    if not isinstance(spec.get("common"), dict) or not isinstance(spec.get("horizons"), dict):
        raise ValueError(f"Missing common/horizons sections: {path}")
    if set(spec['horizons']) != set(HORIZONS):
        raise ValueError(f"Each YAML must specify horizons {HORIZONS}: {path}")
    return spec


def get_config(dataset: str, pred_len: int, mode: str = "evaluate", *,
               config_dir: Path | None = None) -> dict[str, Any]:
    if mode not in ("evaluate", "train", "refine"):
        raise ValueError(f"Invalid mode: {mode}")
    spec = load_dataset_config(dataset, config_dir)
    if pred_len not in HORIZONS:
        raise ValueError(f"Invalid horizon: {pred_len}")
    settings = {**spec["common"], **spec["horizons"][pred_len]}
    # Rank appears ONCE in the YAML. Last encoder width is derived, not tuned twice.
    hidden = settings.pop("encoder_hidden_dims", None)
    if hidden is not None:
        if not isinstance(hidden, list):
            raise ValueError("encoder_hidden_dims must be a list")
        settings["time_do_dims"] = ','.join(str(x) for x in [*hidden, settings["basis_num"]])
    if "time_do_dims" not in settings:
        raise ValueError(f"Missing encoder_hidden_dims for {dataset}/{pred_len}")
    settings.update(dataset=spec["dataset"], pred_len=pred_len, data_file=spec["data_file"])
    settings.pop("reference_metrics_exploratory", None)
    if mode == "train":
        settings.update(freeze_backbone=False, evaluation_scale=settings.get('latent_transition_scale', 1.0),
                        checkpoint_policy="validation", exploratory=False,
                        selection_protocol="validation MSE")
    elif mode == 'refine':
        settings.update(checkpoint_policy='fixed', exploratory=True,
                        selection_protocol='refine; model selection requires protocol audit')
    else:
        settings.update(checkpoint_policy='fixed',
                        exploratory=bool(settings.get('exploratory', spec['status'] != 'validation')),
                        selection_protocol=settings.get('selection_protocol', f"{spec['status']} saved checkpoint"))
    return settings
