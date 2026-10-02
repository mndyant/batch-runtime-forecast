"""固定の時間分割・過去だけの手法選択・全候補の同条件比較。"""
from time import perf_counter
import numpy as np
import pandas as pd
from forecasting.data import COLUMNS
from forecasting.models import MODELS, LABELS, predict
from forecasting.planning import daily_plan

DEFAULT_SETTINGS={'horizon':7,'start_time':'22:00','deadline_time':'00:00','buffer_minutes':15.,'stress_pct':120.}


def origins(n):
    end=n-60
    development=list(range(max(60,end-60),end-29,7))[:5]
    audit=list(range(end,end+29,7))
    if not development or end<90:
        raise ValueError('評価には150点以上必要です。')
    return development,audit


def predict_at(frame,origin,horizon,model):
    # 正解を受け取るのは評価器だけ。モデルには起点以前の3列のみ。
    return predict(frame.loc[:,COLUMNS].iloc[:origin].copy(),horizon,model)


def metrics(actual,prediction,settings=None):
    settings=settings or DEFAULT_SETTINGS
    if len(actual)!=len(prediction) or not len(actual):
        raise ValueError('実績と予測の行数が一致しません。')
    dates=[str(d.date()) for d in actual.date]
    if dates!=[p['date'] for p in prediction]:
        raise ValueError('実績と予測の日付が一致しません。')
    av=actual.records.to_numpy(dtype=float);at=actual.runtime_minutes.to_numpy(dtype=float)
    pv=np.array([p['records'] for p in prediction]);pt=np.array([p['runtime_minutes'] for p in prediction])
    ve=pv-av;te=pt-at
    counts={'tp':0,'tn':0,'fn':0,'fp':0}
    active_counts=dict(counts)
    completion=[];latest=[]
    for i,day in enumerate(dates):
        a=daily_plan(day,float(at[i]),settings['start_time'],settings['deadline_time'],settings['buffer_minutes'])
        p=daily_plan(day,float(pt[i]),settings['start_time'],settings['deadline_time'],settings['buffer_minutes'])
        key=('tp' if p['status']=='late' else 'fn') if a['status']=='late' else ('fp' if p['status']=='late' else 'tn')
        counts[key]+=1
        if av[i]>0: active_counts[key]+=1
        difference=a['slack_minutes']-p['slack_minutes']
        completion.append(difference);latest.append(-difference)
    active=av>0
    return {'days':len(actual),'volume_mae':float(np.mean(abs(ve))),
            'runtime_mae':float(np.mean(abs(te))),'runtime_rmse':float(np.sqrt(np.mean(te**2))),
            'endpoint_error':float(te[-1]),'endpoint_abs_error':float(abs(te[-1])),
            'completion_error_mean':float(np.mean(completion)),
            'latest_start_mae':float(np.mean(np.abs(latest))),
            'deadline':counts,'active_deadline':active_counts,'active_days':int(sum(active)),
            'active_volume_mae':float(np.mean(abs(ve[active]))) if active.any() else None,
            'active_runtime_mae':float(np.mean(abs(te[active]))) if active.any() else None}


def select_model(history,development_end=None):
    end=len(history)-60 if development_end is None else development_end
    development,_=origins(end+60)
    if end>len(history):
        raise ValueError('開発区間が観測件数を超えています。')
    scores={}
    for model in MODELS:
        errors=[]
        for origin in development:
            result=predict_at(history,origin,7,model)
            errors.append(metrics(history.iloc[origin:origin+7],result['forecast'])['runtime_mae'])
        scores[model]=float(np.mean(errors))
    threshold=min(scores.values())*1.05+.1
    selected=next(model for model in MODELS if scores[model]<=threshold)
    return {'selected_model':selected,'scores':scores,'development_end':end,'origins':development}


def backtest(history,settings=None):
    settings=settings or DEFAULT_SETTINGS
    selection=select_model(history)
    _,audit=origins(len(history))
    rows=[];comparison=[]
    for origin in audit:
        for model in MODELS:
            result=predict_at(history,origin,30,model)
            for horizon in [7,30]:
                rows.append({'model':model,'horizon':horizon,'origin':origin,
                             **metrics(history.iloc[origin:origin+horizon],result['forecast'][:horizon],settings),
                             'fit_ms':result['fit_ms'],'predict_ms':result['predict_ms']})
            if origin==audit[-1] and model==selection['selected_model']:
                for (_,a),p in zip(history.iloc[origin:origin+30].iterrows(),result['forecast']):
                    comparison.append({'date':p['date'],'actual_records':float(a.records),
                                       'predicted_records':p['records'],'actual_runtime':float(a.runtime_minutes),
                                       'predicted_runtime':p['runtime_minutes']})
    summary=[]
    for horizon in [7,30]:
        baseline=np.mean([r['runtime_mae'] for r in rows if r['model']=='weekday4' and r['horizon']==horizon])
        for model in MODELS:
            subset=[r for r in rows if r['model']==model and r['horizon']==horizon]
            item={'model':model,'model_label':LABELS[model],'horizon':horizon,'folds':len(subset)}
            for key in ['volume_mae','runtime_mae','runtime_rmse','endpoint_abs_error','latest_start_mae']:
                item[key]=float(np.mean([r[key] for r in subset]))
            item['improvement_pct']=None if baseline==0 else 100*(baseline-item['runtime_mae'])/baseline
            item['deadline']={key:sum(r['deadline'][key] for r in subset) for key in ['tp','tn','fn','fp']}
            summary.append(item)
    return {'selected_model':selection['selected_model'],
            'selection_rule':'開発区間の7日先・所要時間MAE。最良値×1.05＋0.1分以内は同曜日平均→7日平均→回帰の順。監査結果は選択に使いません。',
            'selection_scores':selection['scores'],'development_end':selection['development_end'],
            'development_origins':[str(history.date.iloc[o-1].date()) for o in selection['origins']],
            'summary':summary,'audit_origins':[str(history.date.iloc[o-1].date()) for o in audit],
            'comparison_origin':str(history.date.iloc[audit[-1]-1].date()),
            'comparison_model':selection['selected_model'],'selected_comparison':comparison,
            'fold_results':rows}
