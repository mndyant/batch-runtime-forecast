"""既存の最終結果を上書きせず、凍結済みの予測と採点を再現照合する。"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.benchmark import verify
from evaluation.backtest import select_model,metrics
from forecasting.models import predict
from forecasting.data import read_csv


def main():
    manifest=verify()
    saved=json.loads((ROOT/'docs/evaluation/final/predictions.json').read_text(encoding='utf-8'))
    cases=pd.read_csv(ROOT/'docs/evaluation/final/cases.csv')
    settings=manifest['protocol']['settings']
    for case in saved:
        name=case['case']
        observed=read_csv((ROOT/'data/demo/observed'/f'{name}.csv').read_bytes())
        assert select_model(observed)==case['selection'],f'{name}: selection'
        forecasts={model:predict(observed,30,model) for model in manifest['protocol']['models']}
        for model,result in forecasts.items():
            expected=case['predictions'][model]['forecast']
            assert [r['date'] for r in result['forecast']]==[r['date'] for r in expected]
            np.testing.assert_allclose(
                [[r['records'],r['runtime_minutes']] for r in result['forecast']],
                [[r['records'],r['runtime_minutes']] for r in expected],rtol=1e-9,atol=1e-8)
        truth=read_csv((ROOT/'data/demo/truth'/f'{name}.csv').read_bytes(),minimum=30)
        for model,result in forecasts.items():
            for horizon in [1,7,30]:
                score=metrics(truth.iloc[:horizon],result['forecast'][:horizon],settings)
                expected=cases[(cases['case']==name)&(cases.model==model)&(cases.horizon==horizon)].iloc[0]
                for key in ['volume_mae','runtime_mae','runtime_rmse','endpoint_abs_error','latest_start_mae']:
                    assert math.isclose(score[key],float(expected[key]),rel_tol=1e-9,abs_tol=1e-8),(name,model,key)
                for key,value in score['deadline'].items():
                    assert value==expected[key],(name,model,key)
    print(f'PASS: {len(saved)} cases, 3 models, 1/7/30-day scores reproduced; original files unchanged.')


if __name__=='__main__': main()
