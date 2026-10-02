"""ローカル専用Flaskアプリ。入力データの外部送信は行わない。"""
import math
import re
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from forecasting.data import read_csv, records
from forecasting.models import predict, LABELS
from forecasting.planning import schedule, overview
from evaluation.backtest import backtest, DEFAULT_SETTINGS
from scripts.generate_demo import SCENARIOS

ROOT=Path(__file__).resolve().parents[1]


def settings_from(form):
    result={}
    try:
        for key in ['horizon','buffer_minutes','stress_pct']:
            value=float(form.get(key,DEFAULT_SETTINGS[key]))
            if not math.isfinite(value): raise ValueError()
            result[key]=value
        if result['horizon'] not in [1,7,30]: raise ValueError()
        result['horizon']=int(result['horizon'])
        if not 0<=result['buffer_minutes']<=1440 or not 1<=result['stress_pct']<=500: raise ValueError()
        for key in ['start_time','deadline_time']:
            value=form.get(key,DEFAULT_SETTINGS[key])
            if not re.fullmatch(r'([01]\d|2[0-3]):[0-5]\d',value): raise ValueError()
            result[key]=value
    except (TypeError,ValueError):
        raise ValueError('設定を確認してください。予測期間1/7/30日、安全余裕0〜1440分、所要時間倍率1〜500%、時刻HH:MMで指定します。')
    return result


def create_app():
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=2*1024*1024

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/sample/<key>')
    def sample(key):
        if key not in SCENARIOS:
            return jsonify(error='サンプルが見つかりません。'),404
        return send_file(ROOT/'data'/'demo'/'observed'/f'{key}-0.csv',as_attachment=True,
                         download_name=f'batch-{key}-180days.csv',mimetype='text/csv')

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error='CSVは2MB以下にしてください。'),413

    @app.post('/api/forecast')
    def forecast():
        try:
            settings=settings_from(request.form)
            uploaded=request.files.get('file')
            if uploaded and uploaded.filename:
                data=uploaded.read()
                source='アップロードCSV（利用者提供）'
            else:
                sample_key=request.form.get('sample','weekday')
                if sample_key not in SCENARIOS:
                    raise ValueError('サンプルを選択してください。')
                data=(ROOT/'data'/'demo'/'observed'/f'{sample_key}-0.csv').read_bytes()
                source=f'合成データ：{SCENARIOS[sample_key]}（seed 0）'
            history=read_csv(data)
            evaluation=backtest(history,settings)
            selected=evaluation['selected_model']
            result=predict(history,settings['horizon'],selected)
            plan=schedule(result['forecast'],settings)
            summary=overview(plan)
            scenarios=[{'label':f'所要時間 {factor*100:g}%','multiplier':factor,
                        **overview(schedule(result['forecast'],settings,factor))}
                       for factor in [.8,1.,settings['stress_pct']/100]]
            items=[{'label':'入力','value':f'{source} / {len(history)}日'},
                   {'label':'観測最終日','value':str(history.date.iloc[-1].date())},
                   {'label':'予測手法','value':LABELS[selected]},
                   {'label':'予測期間','value':f'{plan[0]["date"]} 〜 {plan[-1]["date"]}'},
                   {'label':'締切超過見込み','value':f'{summary["late_days"]}日'},
                   {'label':'安全余裕不足','value':f'{summary["tight_days"]}日'},
                   {'label':'最も余裕が少ない業務日','value':summary['worst_date']},
                   {'label':'同日の最終着手日時','value':summary['latest_start_at']},
                   {'label':'設定','value':f'開始 {settings["start_time"]} / 締切 {settings["deadline_time"]} / 安全余裕 {settings["buffer_minutes"]:g}分（日本時間）'}]
            limitations=[
                '合成データによる評価です。実業務データでの精度・有効性は未検証です。',
                '未来の処理件数も予測するため、件数と所要時間の両方の誤差を含みます。',
                '再帰予測は長い期間ほど誤差が積み重なる場合があります。突発増加は事前情報なしでは予見できません。',
                '倍率比較は仮定の変更です。統計的予測区間や資源増設の効果ではありません。',
                '暦日単位です。祝日、ジョブ依存、並列実行、24時間を超える処理の重なり・待ち行列は未考慮です。',
                '監査foldは日付が重なります。独立した試行数や実運用の精度保証として扱えません。',
                '7日先誤差で手法を選択します。30日先で最良の手法を選ぶ保証はありません。']
            if len(history)<180:
                limitations.append('180点未満のため開発起点が少なく、手法比較は小標本です。')
            headline=f'{settings["horizon"]}日間の予測：締切超過 {summary["late_days"]}日、安全余裕不足 {summary["tight_days"]}日'
            body='バッチ処理計画サマリ\n'+headline+'\n\n'+'\n'.join(f'{i["label"]}：{i["value"]}' for i in items)
            body+='\n\n評価・限界\n'+'\n'.join(limitations)
            return jsonify(source=source,as_of=str(history.date.iloc[-1].date()),observations=len(history),
                           settings=settings,selected_model=selected,model_label=LABELS[selected],
                           history=records(history),forecast=plan,overview=summary,
                           summary={'headline':headline,'items':items,'text':body},
                           scenarios=scenarios,evaluation=evaluation,limitations=limitations)
        except (ValueError,OverflowError) as exc:
            return jsonify(error=str(exc) if isinstance(exc,ValueError) else '予測値または日付が計算範囲を超えました。入力を確認してください。'),400
    return app
