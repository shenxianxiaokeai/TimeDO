"""Validate and publish one TimeDO checkpoint; no fixed machine-specific paths.

Usage: python scripts/publish.py --dataset Weather --pred_len 720 \
    --checkpoint /path/to/selected.pt --data_root /path/to/csvs --apply
Without --apply, the command validates and reports only; with --apply it
updates the selected YAML, checkpoint, and compact checkpoint manifest.
Git is the recommended backup/version history system.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from timedo.config import ModelConfig
from timedo.presets import get_config,load_dataset_config
from timedo.training.engine import run_experiment


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(content,encoding='utf-8')
    os.replace(tmp,path)


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, obj):
    atomic_text(path,json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description='Validate, test and release one task from its best checkpoint')
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--pred_len',type=int,required=True,choices=(96,192,336,720))
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--data_root',type=Path,help='or TIMEDO_DATA_ROOT; defaults to ../Datasets')
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='auto')
    parser.add_argument('--evaluation_scale',type=float,default=None,help='explicit tested inference scale')
    parser.add_argument('--apply',action='store_true',help='write release files only after successful verification')
    args=parser.parse_args(argv)
    cp=args.checkpoint.expanduser().resolve()
    if not cp.is_file():raise FileNotFoundError(cp)
    spec=load_dataset_config(args.dataset)
    name=spec['dataset']
    base=get_config(name,args.pred_len)
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or not isinstance(payload.get('model_state'),dict) or not isinstance(payload.get('model_config'),dict):
        raise ValueError('Publish requires a self-describing checkpoint with model_state and model_config.')
    stored=payload['model_config']
    if int(stored['pred_len'])!=args.pred_len:
        raise ValueError('Checkpoint horizon mismatch')
    meta=payload.get('metadata',{})
    if isinstance(meta,dict) and meta.get('dataset') not in (None,name):
        raise ValueError('Checkpoint dataset mismatch')
    # Training configuration from actual saved metadata wins over provisional YAML.
    recipe=meta.get('experiment_config',{}) if isinstance(meta,dict) else {}
    if not isinstance(recipe,dict):recipe={}
    candidate={**base,**{k:v for k,v in recipe.items() if k in base},**stored}
    candidate['dataset']=name;candidate['pred_len']=args.pred_len;candidate['data_file']=spec['data_file']
    candidate['evaluation_scale']=(args.evaluation_scale if args.evaluation_scale is not None
                                   else base.get('evaluation_scale',stored['latent_transition_scale']))
    ModelConfig.from_namespace(candidate)
    default_data = ROOT.parent/'Datasets' if (ROOT.parent/'Datasets').is_dir() else ROOT/'data'
    root=args.data_root or (Path(os.environ['TIMEDO_DATA_ROOT']) if os.getenv('TIMEDO_DATA_ROOT') else default_data)
    with tempfile.TemporaryDirectory(prefix='timedo_release_check_') as temp:
        result=run_experiment(candidate,root,Path(temp),mode='evaluate',checkpoint_path=cp,device=args.device)
    checksum=sha(cp)
    print(f"Verified {name}/{args.pred_len}: MSE={result['test']['mse']:.12f} MAE={result['test']['mae']:.12f} sha256={checksum}")
    if not args.apply:
        print('DRY RUN ONLY; rerun with --apply to publish.')
        return
    # Only the target dataset YAML is touched. Final rank occurs once in the YAML.
    horizon_config=dict(spec['horizons'][args.pred_len])
    for key,value in recipe.items():
        if key in base and key not in ('dataset','pred_len','data_file','time_do_dims','enc_in'):
            horizon_config[key]=value
    for key,value in stored.items():
        if key in ModelConfig.__dataclass_fields__ and key not in ('enc_in','pred_len','time_do_dims'):
            horizon_config[key]=value
    from timedo.models.encoder import parse_dimensions
    dims=parse_dimensions((stored['seq_len']+stored['period_len']-1)//stored['period_len'],
                           stored['basis_num'],stored['time_do_dims'])
    horizon_config.pop('time_do_dims',None)
    horizon_config['encoder_hidden_dims']=list(dims[1:-1])
    horizon_config['evaluation_scale']=candidate['evaluation_scale']
    spec['horizons'][args.pred_len]=horizon_config
    spec['status']='partially verified' if spec['status']=='provisional' else spec['status']
    # Prepare file contents before changing the release weights.
    config_path=ROOT/'configs'/f'{name.lower()}.yaml'
    yaml_text=yaml.safe_dump(spec,allow_unicode=True,sort_keys=False)
    manifest_path=ROOT/'checkpoints/manifest.json'
    manifest=json_load(manifest_path)
    entries=manifest.setdefault('pretrained',[])
    task_id=f'{name}_{args.pred_len}'
    entry=next((item for item in entries if item.get('record_id')==task_id),None)
    if entry is None:
        entry={'record_id':task_id};entries.append(entry)
    published=f'checkpoints/pretrained/TimeDO_{name}_{args.pred_len}.pt'
    entry.update(dataset=name,pred_len=args.pred_len,file=published,
                 sha256=checksum,model_config=stored,
                 test_metrics=dict(result['test']),
                 selection_protocol=result.get('selection_protocol'))
    manifest['item_count']=len(entries)
    dest=ROOT/published
    dest.parent.mkdir(parents=True,exist_ok=True)
    staged=dest.with_suffix('.pt.tmp')
    shutil.copyfile(cp,staged)
    if sha(staged)!=checksum:
        staged.unlink(missing_ok=True)
        raise ValueError('Copied weight SHA256 did not match source')
    os.replace(staged,dest)
    atomic_text(config_path,yaml_text)
    atomic_json(manifest_path,manifest)
    print(f'Published {task_id} to {dest}; config and checkpoint manifest synchronized.')

if __name__=='__main__':main()
