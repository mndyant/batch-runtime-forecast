"""日次件数・所要時間の入力検証。減少は許容し、補間しない。"""
import io
import re
import numpy as np
import pandas as pd

COLUMNS = ['date', 'records', 'runtime_minutes']


def read_csv(content, minimum=150):
    try:
        frame = pd.read_csv(io.StringIO(content.decode('utf-8-sig')), dtype=str, keep_default_na=False)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('UTF-8のCSVを指定してください。列はdate,records,runtime_minutesです。') from exc
    if list(frame.columns) != COLUMNS:
        raise ValueError('列をdate,records,runtime_minutesの順にしてください。件数は件、所要時間は分です。')
    if not frame.date.map(lambda s: bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}',s))).all():
        raise ValueError('日付を欠損のないYYYY-MM-DD形式にしてください。')
    try:
        frame['date'] = pd.to_datetime(frame.date,format='%Y-%m-%d',errors='raise')
    except ValueError as exc:
        raise ValueError('存在しない日付が含まれています。') from exc
    if frame.date.duplicated().any():
        raise ValueError('日付が重複しています。1日1行にしてください。')
    for name, limit in [('records',1e9),('runtime_minutes',10080)]:
        frame[name] = pd.to_numeric(frame[name],errors='coerce')
        if not np.isfinite(frame[name]).all() or not frame[name].between(0,limit).all():
            raise ValueError(f'{name}は0以上{limit:g}以下の有限の数値にしてください。欠損や単位文字は使用できません。')
    if not np.equal(frame.records, np.floor(frame.records)).all():
        raise ValueError('recordsは0以上の整数で指定してください。')
    if ((frame.records>0)&(frame.runtime_minutes==0)).any():
        raise ValueError('処理件数が正の日は、所要時間を0より大きい分数にしてください。')
    frame=frame.sort_values('date').reset_index(drop=True)
    if not minimum<=len(frame)<=3650:
        raise ValueError(f'日次データが{len(frame)}点です。最低{minimum}点、推奨180点、最大3650点です。')
    if not frame.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError('欠損日があります。休止日も0件として1日1行必要です。自動補間はしません。')
    return frame


def records(frame):
    return [{'date':str(d.date()),'records':float(v),'runtime_minutes':float(t)}
            for d,v,t in frame[COLUMNS].itertuples(index=False,name=None)]
