"""上下する業務量と非線形な処理時間の合成データ。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCENARIOS={'weekday':'曜日差','monthend':'月末集中','trend':'長期的な増減',
           'slowdown':'処理速度の低下','burst':'突発的な件数増加','idle':'低負荷・休止'}


def generate(scenario,seed,start='2026-01-01',days=180,future=30):
    if scenario not in SCENARIOS or days<150 or future<1:
        raise ValueError('シナリオ、観測150日以上、未来1日以上を指定してください。')
    rng=np.random.default_rng(np.random.SeedSequence([seed,list(SCENARIOS).index(scenario)]))
    dates=pd.date_range(start,periods=days+future);t=np.arange(len(dates))
    factors=np.array([1.18,1.05,1.,1.08,1.3,.45,.3])
    base=28000*factors[dates.dayofweek]
    volume=base*rng.lognormal(-.5*.10**2,.10,len(dates))
    volume+=1300*np.sin(t/8)
    events=[]
    speed=np.ones(len(dates))
    if scenario=='monthend':
        volume*=np.where((dates.days_in_month-dates.day)<3,2.2,1.)
    elif scenario=='trend':
        volume*=.8+.002*t+.3*np.sin(t/27)
    elif scenario=='slowdown':
        change=int(rng.integers(105,145));speed[t>=change]=.65;events=[change]
    elif scenario=='burst':
        events=sorted(rng.choice(np.arange(15,len(dates)),size=7,replace=False).tolist())
        volume[events]*=rng.uniform(2.2,3.5,len(events))
    elif scenario=='idle':
        volume*=.45
        volume[((t//12)%3==1)|(dates.dayofweek>=5)]=0.
    volume=np.maximum(0,np.rint(volume))
    # 一定比例だけにはせず、高負荷時の待ち時間を非線形に加える。
    delay=np.maximum(volume-35000,0)**1.15/900
    runtime=np.maximum(0,18+volume/(400*speed)+delay+rng.normal(0,4,len(dates)))
    runtime=np.where(volume==0,0,runtime)
    frame=pd.DataFrame({'date':dates,'records':volume.astype(int),'runtime_minutes':np.round(runtime,4)})
    metadata={'kind':'合成データ','scenario':scenario,'seed':seed,'start':start,'days':days,'future':future,
              'generator_version':1,'events_index':events,'nominal_records':28000,'records_per_minute':400,
              'overhead_minutes':18,'runtime_noise_std':4,'stream':[seed,list(SCENARIOS).index(scenario)]}
    return frame.iloc[:days].copy(),frame.iloc[days:].copy(),metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/demo'))
    parser.add_argument('--scenario',choices=SCENARIOS)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--start',default='2026-01-01')
    parser.add_argument('--days',type=int,default=180)
    args=parser.parse_args()
    for scenario in ([args.scenario] if args.scenario else SCENARIOS):
        for seed in ([args.seed] if args.seed is not None else list(range(5))+list(range(100,120))):
            obs,truth,meta=generate(scenario,seed,args.start,args.days)
            for folder,frame in [('observed',obs),('truth',truth)]:
                (args.output/folder).mkdir(parents=True,exist_ok=True)
                frame.to_csv(args.output/folder/f'{scenario}-{seed}.csv',index=False,date_format='%Y-%m-%d',lineterminator='\n')
            (args.output/'metadata').mkdir(parents=True,exist_ok=True)
            (args.output/'metadata'/f'{scenario}-{seed}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('合成データを観測・正解・メタデータに分離して保存しました。')


if __name__=='__main__':
    main()
