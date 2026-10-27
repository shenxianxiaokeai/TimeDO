"""Training and evaluation entry points for TimeDO.

Model construction, optimization, checkpoint handling, and metric reporting
live here so that ``run.py`` only describes experiments.  The evaluation
path calls ``model(history)`` exclusively: future windows are supplied only
while optimizing the latent-state objective.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ..config import ModelConfig
from ..data import build_loaders, infer_channels
from ..models.encoder import parse_dimensions
from .losses import TimeDOLoss
from .metrics import StreamingMetrics
from ..models.forecaster import TimeDO


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without consuming their RNG streams."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a concrete device."""

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    cuda_available = torch.cuda.is_available()
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return torch.device("cuda" if device != "cpu" and cuda_available else "cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: Any, device: torch.device | str) -> dict[str, float]:
    """Evaluate a loader using history only and element-weighted metrics."""

    model.eval()
    metrics = StreamingMetrics()
    for history, target in loader:
        history = history.float().to(device)
        target = target.float().to(device)
        prediction = model(history)
        metrics.update(prediction, target)
    return metrics.compute()


def _json_safe(value: Any) -> Any:
    """Convert common experiment values into JSON-compatible containers."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def _config_dict(config: Any) -> dict[str, Any]:
    return asdict(config) if is_dataclass(config) else vars(config).copy()


def _read_checkpoint(
    model: nn.Module,
    path: Path,
    device: torch.device,
    *,
    allow_new_transition: bool = False,
) -> dict[str, Any]:
    """Load an explicit checkpoint, accepting both bundles and state dicts."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a state dictionary: {path}")
    state = payload.get("model_state", payload)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint has no model weights: {path}")
    stored_config = payload.get("model_config") if "model_state" in payload else None
    current_config = getattr(model, "config", None)
    if isinstance(stored_config, Mapping) and current_config is not None:
        architectural_fields = (
            "seq_len", "pred_len", "enc_in", "period_len", "basis_num",
            "householder_reflections", "use_period_norm", "individual", "latent_mode",
        )
        for name in architectural_fields:
            if name not in stored_config or not hasattr(current_config, name):
                continue
            saved = stored_config[name]
            current = getattr(current_config, name)
            if name == "latent_mode" and allow_new_transition and saved == "none":
                continue
            if saved != current:
                raise ValueError(
                    f"Checkpoint {name}={saved!r} disagrees with requested {current!r}: {path}"
                )
        if "time_do_dims" in stored_config:
            saved_segments = (
                int(stored_config.get("seq_len", current_config.seq_len))
                + int(stored_config.get("period_len", current_config.period_len)) - 1
            ) // int(stored_config.get("period_len", current_config.period_len))
            saved_dimensions = parse_dimensions(
                saved_segments,
                int(stored_config.get("basis_num", current_config.basis_num)),
                stored_config["time_do_dims"],
            )
            current_segments = (current_config.seq_len + current_config.period_len - 1) // current_config.period_len
            current_dimensions = parse_dimensions(
                current_segments, current_config.basis_num, current_config.time_do_dims
            )
            if saved_dimensions != current_dimensions:
                raise ValueError(
                    f"Checkpoint hierarchy {saved_dimensions} disagrees with requested "
                    f"{current_dimensions}: {path}"
                )
    # Strict loading prevents an accidentally different encoder/decoder from
    # being accepted as a valid continuation or reference evaluation.
    if allow_new_transition:
        missing, unexpected = model.load_state_dict(state, strict=False)
        transition_keys = {key for key in model.state_dict() if key.startswith("transition.")}
        if unexpected or (missing and set(missing) != transition_keys):
            raise RuntimeError(
                f"Invalid continuation checkpoint: missing={missing}, unexpected={unexpected}"
            )
        if missing:
            print("Initialized a new zero-residual transition over the loaded encoder/decoder")
    else:
        model.load_state_dict(state, strict=True)
    metadata = payload.get("metadata", {}) if "model_state" in payload else {}
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _save_checkpoint(model: nn.Module, config: Any, path: Path, metadata: Mapping[str, Any]) -> None:
    """Save a self-describing checkpoint with an atomic final replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": _json_safe(_config_dict(config)),
            "metadata": _json_safe(dict(metadata)),
        },
        temporary,
    )
    temporary.replace(path)


def _set_transition_scale(model: nn.Module, config: Any, value: float) -> Any:
    """Apply a residual multiplier without altering the learned weights."""

    model.set_transition_scale(float(value))
    return replace(config, latent_transition_scale=float(value))


def run_experiment(
    params: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
    *,
    mode: str = "evaluate",
    checkpoint_path: Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Run one TimeDO benchmark configuration.

    ``evaluate`` requires an explicit checkpoint and never constructs future
    latent targets.  ``train`` starts from random initialization and normally
    chooses the checkpoint with the best validation MSE.  ``refine`` requires
    a checkpoint and may freeze the encoder/decoder; its default is to retain
    the final epoch, matching a fixed continuation recipe.  Test metrics are
    computed once after training has finished.

    Extra experiment keys in ``params`` include ``ref_epochs`` (continuation
    length), ``checkpoint_policy`` (``validation`` or ``fixed``), and optional
    ``evaluation_scale``.  The latter changes only the residual multiplier
    after optimization, while ``latent_transition_scale`` controls training.
    """

    if mode not in {"evaluate", "train", "refine"}:
        raise ValueError("mode must be 'evaluate', 'train', or 'refine'")
    if mode in {"evaluate", "refine"} and checkpoint_path is None:
        raise ValueError(f"mode={mode!r} requires an explicit checkpoint_path")
    if mode == "train" and checkpoint_path is not None:
        raise ValueError("Use mode='refine' when starting from a checkpoint")

    values = dict(params)
    dataset = str(values.get("dataset", "ETTh1"))
    seed = int(values.get("seed", 2027))
    set_seed(seed, bool(values.get("deterministic", True)))
    active_device = resolve_device(device)
    channels = infer_channels(Path(data_root), dataset, data_file=values.get("data_file"))
    if "enc_in" in values and int(values["enc_in"]) != channels:
        raise ValueError(f"enc_in={values['enc_in']} disagrees with the {channels} CSV variables")
    values["enc_in"] = channels
    namespace = SimpleNamespace(**values)
    model_config = ModelConfig.from_namespace(namespace)
    loaders = build_loaders(
        Path(data_root), dataset, model_config.seq_len, model_config.pred_len,
        int(values.get("batch_size", 64)), int(values.get("num_workers", 0)),
        bool(values.get("pin_memory", False)) and active_device.type == "cuda",
        data_file=values.get("data_file"),
    )
    # Construct loaders before the model, and do not supply a separate
    # DataLoader generator: deterministic historical RNG consumption matters.
    model = TimeDO(model_config).to(active_device).float()
    provenance: dict[str, Any] = {}
    if checkpoint_path is not None:
        provenance = _read_checkpoint(
            model, Path(checkpoint_path), active_device, allow_new_transition=mode == "refine"
        )
        if provenance.get("dataset") not in {None, dataset}:
            raise ValueError(f"Checkpoint belongs to {provenance['dataset']}, requested {dataset}")

    run_dir = Path(output_root) / dataset / f"pred_{model_config.pred_len}" / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    saved_checkpoint = run_dir / "checkpoint.pt"
    result_path = run_dir / "result.json"
    policy = str(values.get("checkpoint_policy", "fixed" if mode == "refine" else "validation"))
    if policy not in {"validation", "fixed"}:
        raise ValueError("checkpoint_policy must be 'validation' or 'fixed'")
    epochs = int(values.get("ref_epochs", values.get("train_epochs", 40))) if mode == "refine" else int(values.get("train_epochs", 40))
    patience = int(values.get("patience", 16))
    if mode != "evaluate" and (epochs <= 0 or patience <= 0):
        raise ValueError("training epochs and patience must be positive")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"TimeDO (Deep Orthogonality) | {dataset} | horizon={model_config.pred_len} "
        f"| mode={mode} | parameters={parameter_count} | device={active_device}"
    )
    started = time.time()
    epoch_history: list[dict[str, Any]] = []
    save_snapshots = bool(values.get("save_epoch_snapshots", False)) and mode != "evaluate"
    skip_test = bool(values.get("skip_test", False)) and mode != "evaluate"
    selection_scale = values.get("selection_scale")
    best_epoch: int | None = None
    effective_learning_rate: float | None = None
    if mode != "evaluate":
        freeze = mode == "refine" and bool(values.get("freeze_backbone", False))
        if freeze:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if not hasattr(model, "transition") or model.transition is None:
                raise ValueError("freeze_backbone requires a model with a latent transition")
            for parameter in model.transition.parameters():
                parameter.requires_grad_(True)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("No trainable model parameters remain")
        requested_rate = values.get("latent_learning_rate") if mode == "refine" else values.get("learning_rate", 0.1)
        effective_learning_rate = float(1e-3 if requested_rate is None else requested_rate)
        if effective_learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        decay = float(values.get("lr_decay", 0.8))
        if not 0 < decay <= 1:
            raise ValueError("lr_decay must lie in (0, 1]")
        optimizer = torch.optim.Adam(trainable, lr=effective_learning_rate)
        criterion = TimeDOLoss(model_config)
        best_validation = float("inf")
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            batches = 0
            for history, target in loaders["train"]:
                history = history.float().to(active_device)
                target = target.float().to(active_device)
                optimizer.zero_grad(set_to_none=True)
                if model_config.latent_mode == "none":
                    prediction, states = model(history), None
                else:
                    prediction, states = model.forecast_with_states(history, target)
                losses = criterion(prediction, target, states)
                loss = losses["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach())
                batches += 1
            # Preserve the benchmark scheduler: epochs 1--3 use the initial
            # rate, and decay begins after the third optimization pass.
            if epoch >= 3:
                for group in optimizer.param_groups:
                    group["lr"] = effective_learning_rate * decay ** (epoch - 2)
            # Optional selection under the same residual scale planned for inference.
            # Restore training scale before saving the snapshot or next epoch.
            if selection_scale is not None and model_config.latent_mode != "none":
                old_scale = model.latent_transition_scale
                model.set_transition_scale(float(selection_scale))
                try:
                    validation = evaluate(model, loaders["val"], active_device)
                finally:
                    model.set_transition_scale(old_scale)
            else:
                validation = evaluate(model, loaders["val"], active_device)
            row = {"epoch": epoch, "train_loss": train_loss / batches, "validation": validation}
            epoch_history.append(row)
            if save_snapshots:
                _save_checkpoint(model, model_config, run_dir / "snapshots" / f"epoch_{epoch:02d}.pt", {
                    "dataset": dataset, "seed": seed, "mode": mode, "epoch": epoch,
                    "validation": validation, "selection_scale": selection_scale,
                })
            print(
                f"epoch={epoch:02d} train={row['train_loss']:.7f} "
                f"val_mse={validation['mse']:.7f} val_mae={validation['mae']:.7f}"
            )
            if policy == "fixed" or validation["mse"] < best_validation:
                best_validation = validation["mse"]
                best_epoch = epoch
                stale = 0
                _save_checkpoint(model, model_config, saved_checkpoint, {
                    "dataset": dataset, "seed": seed, "mode": mode,
                    "epoch": epoch, "checkpoint_policy": policy,
                })
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        if best_epoch is None:
            raise RuntimeError("Training completed without producing a checkpoint")
        _read_checkpoint(model, saved_checkpoint, active_device)

    if model_config.latent_mode != "none" and values.get("evaluation_scale") is not None:
        model_config = _set_transition_scale(model, model_config, float(values["evaluation_scale"]))
    validation = evaluate(model, loaders["val"], active_device)
    test = None if skip_test else evaluate(model, loaders["test"], active_device)
    diagnostics = model.diagnostics() if hasattr(model, "diagnostics") else {}
    result: dict[str, Any] = {
        "method": "TimeDO",
        "full_name": "Deep Orthogonality",
        "dataset": dataset,
        "pred_len": model_config.pred_len,
        "mode": mode,
        "seed": seed,
        "device": str(active_device),
        "parameters": parameter_count,
        "model_config": _config_dict(model_config),
        "experiment_config": values,
        "checkpoint_policy": policy if mode != "evaluate" else "provided",
        "best_epoch": best_epoch,
        "effective_learning_rate": effective_learning_rate,
        "initialized_from": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None,
        "provenance": provenance,
        "selection_protocol": values.get("selection_protocol", "provided_reference" if mode == "evaluate" else policy),
        "validation": validation,
        "test": test,
        "diagnostics": diagnostics,
        "history": epoch_history,
        "elapsed_seconds": time.time() - started,
    }
    if mode != "evaluate":
        # Include the final evaluation multiplier in the saved configuration,
        # so reloading the artifact does not depend on a hidden CLI override.
        _save_checkpoint(model, model_config, saved_checkpoint, {
            "dataset": dataset, "seed": seed, "mode": mode,
            "epoch": best_epoch, "checkpoint_policy": policy,
            "selection_protocol": result["selection_protocol"],
            "experiment_config": values,
            "initialization_metadata": provenance,
            "validation": validation, "test": test,
        })
        result["checkpoint"] = str(saved_checkpoint.resolve())
    result = _json_safe(result)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = validation if test is None else test
    metric_label = "VAL_ONLY" if test is None else "RESULT"
    print(f"{metric_label} mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}")
    print(f"Saved {result_path}")
    return result


__all__ = ["evaluate", "resolve_device", "run_experiment", "set_seed"]
