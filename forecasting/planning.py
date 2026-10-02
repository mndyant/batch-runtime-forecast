"""日本時間・日跨ぎを明示した計画。予測の非整数分は秒単位で切り上げ。"""
import math
from datetime import datetime,timedelta,timezone

JST=timezone(timedelta(hours=9))


def daily_plan(day,runtime,start_time,deadline_time,buffer_minutes):
    date=datetime.strptime(day,'%Y-%m-%d').date()
    start=datetime.combine(date,datetime.strptime(start_time,'%H:%M').time(),JST)
    deadline=datetime.combine(date,datetime.strptime(deadline_time,'%H:%M').time(),JST)
    if deadline<=start:
        deadline+=timedelta(days=1)
    duration=timedelta(seconds=math.ceil(runtime*60))
    finish=start+duration
    latest=deadline-duration-timedelta(minutes=buffer_minutes)
    slack=(deadline-finish).total_seconds()/60
    status='late' if slack<0 else 'tight' if slack<buffer_minutes else 'ok'
    return {'start_at':start.isoformat(timespec='seconds'),'deadline_at':deadline.isoformat(timespec='seconds'),
            'finish_at':finish.isoformat(timespec='seconds'),'latest_start_at':latest.isoformat(timespec='seconds'),
            'slack_minutes':slack,'status':status}


def schedule(forecast,settings,multiplier=1.):
    return [{**row,'runtime_minutes':row['runtime_minutes']*multiplier,
             **daily_plan(row['date'],row['runtime_minutes']*multiplier,settings['start_time'],
                          settings['deadline_time'],settings['buffer_minutes'])} for row in forecast]


def overview(rows):
    worst=min(rows,key=lambda row:row['slack_minutes'])
    return {'late_days':sum(r['status']=='late' for r in rows),'tight_days':sum(r['status']=='tight' for r in rows),
            'worst_date':worst['date'],'min_slack_minutes':worst['slack_minutes'],
            'latest_start_at':worst['latest_start_at']}
