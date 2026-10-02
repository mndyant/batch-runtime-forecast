"""曜日平均、7日平均、件数→時間の再帰リッジ。外部ファイルを参照しない。"""
from time import perf_counter
import numpy as np
import pandas as pd
from forecasting.data import COLUMNS

MODELS = ('weekday4','mean7','ridge')
LABELS = {'weekday4':'直近4週の同曜日平均（基準）','mean7':'直近7日平均','ridge':'カレンダー付き回帰'}
RIDGE_ALPHA = 10.
RIDGE_WINDOW = 120
LAGS = 28


def calendar(day):
    day = pd.Timestamp(day)
    return [float(day.dayofweek==i) for i in range(7)]+[
        np.sin(2*np.pi*day.day/31),np.cos(2*np.pi*day.day/31),
        float(day.days_in_month-day.day<3),float(day.day<=3),day.toordinal()/365.]


def volume_features(day, volumes):
    return calendar(day)+[volumes[-1],volumes[-7],float(np.mean(volumes[-7:])),float(np.mean(volumes[-28:]))]


def runtime_features(day, volumes, times, volume):
    # 学習時は同日の観測件数、将来はその日の予測件数だけを渡す。
    return calendar(day)+[volume, times[-1],times[-7],float(np.mean(times[-7:])),
                          float(np.mean(times[-28:]))]


def fit_ridge(x, y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    mean=x.mean(axis=0);std=x.std(axis=0);std[std<1e-8]=1.
    z=(x-mean)/std
    center=float(y.mean())
    coefficient=np.linalg.solve(z.T@z+RIDGE_ALPHA*np.eye(z.shape[1]),z.T@(y-center))
    return {'mean':mean,'std':std,'coefficient':coefficient,'center':center}


def apply_ridge(model, x):
    value=float(((np.asarray(x)-model['mean'])/model['std'])@model['coefficient']+model['center'])
    if not np.isfinite(value):
        raise ValueError('有限の予測値を計算できませんでした。')
    return max(0., value)


def serialize(model):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()}


def predict(history,horizon,model='weekday4'):
    if list(history.columns)!=COLUMNS or len(history)<60:
        raise ValueError('予測器には観測済みの3列と60点以上を渡してください。')
    if model not in MODELS or not 1<=horizon<=30:
        raise ValueError('予測モデルまたは期間が不正です。')
    start=perf_counter()
    h=history.iloc[-RIDGE_WINDOW:].copy() if model=='ridge' else history.copy()
    values=h.records.to_numpy(dtype=float).tolist()
    times=h.runtime_minutes.to_numpy(dtype=float).tolist()
    dates=pd.date_range(history.date.iloc[-1]+pd.Timedelta(days=1),periods=horizon)
    parameters={'model':model,'fit_count':len(h),'fit_start':str(h.date.iloc[0].date()),
                'fit_end':str(h.date.iloc[-1].date())}
    if model=='ridge':
        vx=[volume_features(h.date.iloc[i],values[:i]) for i in range(LAGS,len(h))]
        tx=[runtime_features(h.date.iloc[i],values[:i],times[:i],values[i]) for i in range(LAGS,len(h))]
        vm=fit_ridge(vx,values[LAGS:]);tm=fit_ridge(tx,times[LAGS:])
        parameters.update(alpha=RIDGE_ALPHA,training_rows=len(vx),volume=serialize(vm),runtime=serialize(tm))
    elif model=='weekday4':
        by_weekday={}
        for weekday in range(7):
            past=h[h.date.dt.dayofweek==weekday].iloc[-4:]
            by_weekday[weekday]=(float(past.records.mean()),float(past.runtime_minutes.mean()))
        parameters['weekday_means']={str(k):list(v) for k,v in by_weekday.items()}
    else:
        means=(float(np.mean(values[-7:])),float(np.mean(times[-7:])))
        parameters['means']=list(means)
    fit_ms=(perf_counter()-start)*1000; start=perf_counter()
    forecasts=[]
    for day in dates:
        if model=='ridge':
            volume=apply_ridge(vm,volume_features(day,values))
            runtime=apply_ridge(tm,runtime_features(day,values,times,volume))
            # 再帰入力に追加するのは予測値。将来の実績はAPIに存在しない。
            values.append(volume);times.append(runtime)
        elif model=='weekday4':
            volume,runtime=by_weekday[day.dayofweek]
        else:
            volume,runtime=means
        if volume>1e9 or runtime>10080:
            raise ValueError('予測値が初版の対応範囲（10億件・10080分）を超えました。過大な値を丸めず計算を中止しました。')
        forecasts.append({'date':str(day.date()),'records':volume,'runtime_minutes':runtime})
    return {'forecast':forecasts,'parameters':parameters,'fit_ms':fit_ms,'predict_ms':(perf_counter()-start)*1000}
