"""事前固定したプロトコルの全件採点。最終実行はロックし、上書きしない。"""
import argparse
import hashlib
import json
import platform
from datetime import datetime,timezone
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
from evaluation.backtest import select_model,metrics
from forecasting.data import read_csv
from forecasting.models import predict

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'evaluation'/'protocol.json'
OUTPUT=ROOT/'docs'/'evaluation'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_files():
    files=[]
    for folder in ['forecasting','evaluation','scripts']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.suffix in ['.py','.json'] and '__pycache__' not in p.parts)
    files.extend((ROOT/'data'/'demo').rglob('*.csv'))
    files.extend((ROOT/'data'/'demo'/'metadata').glob('*.json'))
    files.extend([ROOT/'requirements.txt',ROOT/'requirements-dev.txt',ROOT/'docs'/'specification.md',
                  ROOT/'docs'/'design-decisions.md',ROOT/'app'/'__init__.py'])
    files.extend(p for p in (ROOT/'tests').glob('test_*.py'))
    return sorted(set(files))


def freeze():
    path=OUTPUT/'frozen-manifest.json'
    if path.exists():
        raise RuntimeError('凍結ファイルは上書きできません。')
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
              'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','Flask']},
              'protocol':json.loads(PROTOCOL.read_text(encoding='utf-8')),
              'files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in frozen_files()}}
    dump(path,manifest)
    print(f'Frozen {len(manifest["files"])} files.')


def verify():
    manifest=json.loads((OUTPUT/'frozen-manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        if sha(ROOT/name)!=expected: raise RuntimeError(f'凍結後に変更されています: {name}')
    expected_files=set(manifest['files'])
    actual_files={str(p.relative_to(ROOT)).replace('\\','/') for p in frozen_files()}
    if expected_files!=actual_files: raise RuntimeError('凍結対象ファイル一覧が変わっています。')
    return manifest


def aggregate(rows):
    frame=pd.DataFrame(rows)
    summaries=[]
    for scenario in ['all']+list(frame.scenario.unique()):
        sub=frame if scenario=='all' else frame[frame.scenario==scenario]
        for horizon in [1,7,30]:
            for model in ['weekday4','mean7','ridge','selected']:
                part=sub[(sub.horizon==horizon)&(sub.model==model)]
                if part.empty: continue
                base=sub[(sub.horizon==horizon)&(sub.model=='weekday4')].runtime_mae.mean()
                mae=float(part.runtime_mae.mean())
                item={'scenario':scenario,'horizon':horizon,'model':model,'cases':len(part),
                      'runtime_mae':mae,'runtime_mae_median':float(part.runtime_mae.median()),
                      'runtime_mae_worst':float(part.runtime_mae.max()),
                      'worst_case':part.loc[part.runtime_mae.idxmax(),'case'],
                      'volume_mae':float(part.volume_mae.mean()),
                      'runtime_rmse':float(part.runtime_rmse.mean()),
                      'endpoint_abs_error':float(part.endpoint_abs_error.mean()),
                      'latest_start_mae':float(part.latest_start_mae.mean()),
                      'improvement_pct':None if base==0 else float(100*(base-mae)/base),
                      'active_runtime_mae':None if part.active_runtime_mae.dropna().empty else float(part.active_runtime_mae.mean()),
                      'active_volume_mae':None if part.active_volume_mae.dropna().empty else float(part.active_volume_mae.mean())}
                for key in ['tp','tn','fn','fp','active_tp','active_tn','active_fn','active_fp','days','active_days']:
                    item[key]=int(part[key].sum())
                item['deadline_recall']=None if item['tp']+item['fn']==0 else item['tp']/(item['tp']+item['fn'])
                item['deadline_precision']=None if item['tp']+item['fp']==0 else item['tp']/(item['tp']+item['fp'])
                summaries.append(item)
    return summaries


def run(stage):
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if stage=='final': verify()
    out=OUTPUT/stage
    out.mkdir(parents=True,exist_ok=True)
    # 排他的な作成。途中失敗も残す。再実行で同じ最終結果を上書きしない。
    with (out/'run-lock.json').open('x',encoding='utf-8') as file:
        json.dump({'started_at':datetime.now(timezone.utc).isoformat(),'protocol_hash':sha(PROTOCOL)},file)
    rows=[];failures=[];predictions=[];selections={}
    for scenario in protocol['scenarios']:
        for seed in protocol[f'{stage}_seeds']:
            case=f'{scenario}-{seed}'
            try:
                history=read_csv((ROOT/'data'/'demo'/'observed'/f'{case}.csv').read_bytes())
                if len(history)!=protocol['observations']: raise ValueError('観測数がプロトコルと不一致')
                selection=select_model(history)
                # 全候補の予測を確定してから、初めて採点用未来CSVを開く。
                forecasts={model:predict(history,30,model) for model in protocol['models']}
                predictions.append({'case':case,'selection':selection,'predictions':forecasts})
                selections[case]=selection['selected_model']
                truth=read_csv((ROOT/'data'/'demo'/'truth'/f'{case}.csv').read_bytes(),minimum=30)
                for model,result in forecasts.items():
                    for horizon in protocol['horizons']:
                        score=metrics(truth.iloc[:horizon],result['forecast'][:horizon],protocol['settings'])
                        deadline=score.pop('deadline');active=score.pop('active_deadline')
                        row={'case':case,'scenario':scenario,'seed':seed,'model':model,
                             'selected_model':selection['selected_model'],'horizon':horizon,
                             'fit_ms':result['fit_ms'],'predict_ms':result['predict_ms'],**score,
                             **deadline,**{f'active_{k}':v for k,v in active.items()}}
                        rows.append(row)
                        if model==selection['selected_model']: rows.append({**row,'model':'selected'})
            except Exception as exc:
                failures.append({'case':case,'error':f'{type(exc).__name__}: {exc}'})
            print(f'{stage}: {case}',flush=True)
    if stage=='final': verify()
    expected=len(protocol['scenarios'])*len(protocol[f'{stage}_seeds'])
    report={'stage':stage,'synthetic_only':True,'created_at':datetime.now(timezone.utc).isoformat(),
            'expected_cases':expected,'predicted_cases':len(predictions),'failed_cases':len(failures),
            'failures':failures,'selection_counts':{m:list(selections.values()).count(m) for m in protocol['models']},
            'summary':aggregate(rows) if rows else [],
            'notes':['合成データのみ。実業務精度ではない。','同じケース内の日次と1/7/30日先は独立標本ではない。',
                     'α・特徴・選択基準は採点後に変更しない。','処理件数の予測誤差を含む所要時間評価。',
                     'ケースごとの指標を同じ重みで平均。RMSEはケースRMSEの平均。',
                     'active指標は実件数が正の日だけ。全0ケースはactive MAEをN/Aとする。']}
    dump(out/'predictions.json',predictions)
    pd.DataFrame(rows).to_csv(out/'cases.csv',index=False,lineterminator='\n')
    dump(out/'report.json',report)
    print(f'{stage}: {expected} cases, {len(failures)} failures.')
    if failures: raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--stage',choices=['exploration','final'])
    args=parser.parse_args()
    if sum([args.freeze,args.verify,bool(args.stage)])!=1: parser.error('操作を1つ指定してください。')
    if args.freeze: freeze()
    elif args.verify:
        manifest=verify();print(f'Verified {len(manifest["files"])} files.')
    else: run(args.stage)


if __name__=='__main__': main()
