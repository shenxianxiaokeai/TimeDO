"""TimeDO entry point; dataset and horizon hyperparameters live in configs/*.yaml."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from timedo.presets import DATASETS, HORIZONS, get_config

ROOT = Path(__file__).resolve().parent
PRETRAINED_DIR = ROOT / 'checkpoints' / 'pretrained'
INITIALIZATION_DIR = ROOT / 'checkpoints' / 'initializations'


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "on"}:
        return True
    if value.lower() in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TimeDO forecasting from dataset/horizon YAML configurations")
    parser.add_argument('--mode', choices=('evaluate', 'train', 'refine'), default='evaluate')
    parser.add_argument('--dataset', nargs='+', choices=(*DATASETS, 'all'), default=['ETTh1', 'ETTh2'])
    parser.add_argument('--pred_len', nargs='+', type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument('--data_root', type=Path, help='CSV directory; or set TIMEDO_DATA_ROOT')
    parser.add_argument('--output_root', type=Path, default=ROOT / 'outputs' / 'runs')
    parser.add_argument('--config_dir', type=Path, default=ROOT / 'configs')
    parser.add_argument('--checkpoint', type=Path, help='override checkpoint; one dataset and horizon only')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--show_config', action='store_true')
    parser.add_argument('--checkpoint_policy', choices=('validation', 'fixed'), default=None,
                        help='How train/refine selects a saved checkpoint; refine defaults to fixed')
    parser.add_argument('--save_epoch_snapshots', action='store_true', help='Keep checkpoint per epoch for offline analysis')
    parser.add_argument('--skip_test', action='store_true', help='Tune using validation only; evaluate selected checkpoint separately')
    parser.add_argument('--selection_scale', type=float, default=None, help='Optional residual scale for validation-based checkpoint selection')
    for name in ('seed','train_epochs','patience','batch_size','num_workers','ref_epochs',
                 'seq_len','period_len','basis_num','householder_reflections'):
        parser.add_argument(f'--{name}', type=int, default=None)
    for name in ('learning_rate','lr_decay','latent_learning_rate','latent_mse_weight',
                 'latent_cos_weight','shallow_weight','obs_loss_weight',
                 'latent_transition_scale','evaluation_scale'):
        parser.add_argument(f'--{name}', type=float, default=None)
    for name in ('individual','shallow_alignment','use_period_norm', 'freeze_backbone',
                 'deterministic','pin_memory'):
        parser.add_argument(f'--{name}', type=str_to_bool, default=None)
    parser.add_argument('--time_do_dims', help='explicit full widths incl. final rank; experimental override')
    parser.add_argument('--latent_mode', choices=('d1','rank','none'))
    return parser


def resolve_data_root(explicit: Path | None) -> Path:
    """Resolve input data: CLI > environment > workspace Datasets > project data/."""
    if explicit is not None:
        location = explicit
    elif os.environ.get('TIMEDO_DATA_ROOT'):
        location = Path(os.environ['TIMEDO_DATA_ROOT'])
    else:
        workspace_data = ROOT.parent / 'Datasets'
        location = workspace_data if workspace_data.is_dir() else ROOT / 'data'
    return location.expanduser().resolve()


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = build_parser().parse_args(argv)
    horizons = tuple(dict.fromkeys(args.pred_len))
    if 'all' in args.dataset:
        datasets = DATASETS
        if args.mode == 'evaluate':
            datasets = tuple(
                name for name in DATASETS
                if all((PRETRAINED_DIR / f"TimeDO_{name}_{horizon}.pt").is_file() for horizon in horizons)
            )
            if not datasets:
                raise SystemExit('No published checkpoints found for the requested horizons.')
            print(f"Evaluating datasets with published weights: {', '.join(datasets)}")
    else:
        datasets = tuple(dict.fromkeys(args.dataset))
    if args.checkpoint and len(datasets)*len(horizons) != 1:
        raise SystemExit('--checkpoint requires one dataset and one horizon')
    excluded = {'mode','dataset','pred_len','data_root','output_root','config_dir','checkpoint','device','show_config'}
    overrides = {k:v for k,v in vars(args).items() if k not in excluded and v is not None}
    tasks = [{**get_config(name,horizon,args.mode,config_dir=args.config_dir),**overrides}
             for name in datasets for horizon in horizons]
    # Keep CLI rank overrides as a single source of truth, just like YAML.
    if args.basis_num is not None and args.time_do_dims is None:
        for params in tasks:
            widths = [int(x) for x in str(params['time_do_dims']).split(',')]
            params['time_do_dims'] = ','.join(map(str, [*widths[:-1], args.basis_num]))
    for params in tasks:
        if args.checkpoint_policy is not None:
            params['selection_protocol'] = ('validation MSE' if args.checkpoint_policy == 'validation' else 'fixed final epoch')
    if args.mode == 'evaluate' and (args.skip_test or args.save_epoch_snapshots):
        raise SystemExit('--skip_test and --save_epoch_snapshots are only for train/refine')
    if args.mode == 'train' and args.evaluation_scale is None:
        for params in tasks:
            params['evaluation_scale'] = params['latent_transition_scale']
    if args.show_config:
        print(json.dumps(tasks,indent=2,ensure_ascii=False))
        return tasks
    from timedo.training.engine import run_experiment
    root = resolve_data_root(args.data_root)
    results=[]
    for params in tasks:
        file=f"TimeDO_{params['dataset']}_{params['pred_len']}.pt"
        if args.mode == 'train':
            checkpoint = None
        elif args.checkpoint:
            checkpoint = args.checkpoint.expanduser().resolve()
        else:
            checkpoint = (PRETRAINED_DIR if args.mode == 'evaluate' else INITIALIZATION_DIR) / file
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {args.mode} checkpoint: {checkpoint}. "
                                    "Untrained tasks have no published weight; use --mode train first.")
        results.append(run_experiment(params,root,args.output_root.expanduser().resolve(),
                                      mode=args.mode,checkpoint_path=checkpoint,device=args.device))
    print('\nTimeDO completed:')
    for result in results:
        metrics = result["test"] if result["test"] is not None else result["validation"]
        label = "MSE" if result["test"] is not None else "VAL_MSE"
        print(f"{result['dataset']:12s} {result['pred_len']:3d} "
              f"{label}={metrics['mse']:.6f} MAE={metrics['mae']:.6f}")
    return results

if __name__=='__main__':
    main()
