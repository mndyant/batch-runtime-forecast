import json
from pathlib import Path
from unittest.mock import patch
import pytest
from evaluation import benchmark
from forecasting.data import read_csv
from forecasting.models import predict
from scripts.generate_demo import generate


def test_manifest_tampering_and_no_overwrite(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('frozen',encoding='utf-8')
    protocol=tmp_path/'protocol.json';protocol.write_text('{}',encoding='utf-8')
    monkeypatch.setattr(benchmark,'ROOT',tmp_path)
    monkeypatch.setattr(benchmark,'OUTPUT',tmp_path/'evaluation')
    monkeypatch.setattr(benchmark,'PROTOCOL',protocol)
    monkeypatch.setattr(benchmark,'frozen_files',lambda:[source,protocol])
    benchmark.freeze()
    benchmark.verify()
    with pytest.raises(RuntimeError): benchmark.freeze()
    source.write_text('changed',encoding='utf-8')
    with pytest.raises(RuntimeError,match='source.py'): benchmark.verify()


def test_benchmark_truth_after_predictions_and_lock(tmp_path,monkeypatch):
    protocol=json.loads(benchmark.PROTOCOL.read_text(encoding='utf-8'))
    protocol['scenarios']=['weekday'];protocol['exploration_seeds']=[0]
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol),encoding='utf-8')
    obs,truth,_=generate('weekday',0)
    for folder,frame in [('observed',obs),('truth',truth)]:
        target=tmp_path/'data'/'demo'/folder/'weekday-0.csv'
        target.parent.mkdir(parents=True)
        frame.to_csv(target,index=False,date_format='%Y-%m-%d')
    monkeypatch.setattr(benchmark,'ROOT',tmp_path)
    monkeypatch.setattr(benchmark,'OUTPUT',tmp_path/'output')
    monkeypatch.setattr(benchmark,'PROTOCOL',path)
    predicted=[]
    def guarded_predict(history,horizon,model):
        assert len(history)==180
        result=predict(history,horizon,model)
        predicted.append(model)
        return result
    def guarded_read(content,minimum=150):
        if minimum==30:
            assert predicted==protocol['models']
        return read_csv(content,minimum)
    monkeypatch.setattr(benchmark,'predict',guarded_predict)
    monkeypatch.setattr(benchmark,'read_csv',guarded_read)
    benchmark.run('exploration')
    report=tmp_path/'output'/'exploration'/'report.json'
    result=json.loads(report.read_text(encoding='utf-8'))
    assert result['failed_cases']==0 and result['expected_cases']==1
    before=report.read_bytes()
    with pytest.raises(FileExistsError): benchmark.run('exploration')
    assert report.read_bytes()==before


def test_failed_case_is_retained(tmp_path,monkeypatch):
    protocol=json.loads(benchmark.PROTOCOL.read_text(encoding='utf-8'))
    protocol['scenarios']=['weekday'];protocol['exploration_seeds']=[0]
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol),encoding='utf-8')
    monkeypatch.setattr(benchmark,'ROOT',tmp_path)
    monkeypatch.setattr(benchmark,'OUTPUT',tmp_path/'output')
    monkeypatch.setattr(benchmark,'PROTOCOL',path)
    with pytest.raises(SystemExit): benchmark.run('exploration')
    result=json.loads((tmp_path/'output'/'exploration'/'report.json').read_text(encoding='utf-8'))
    assert result['failed_cases']==1 and result['failures'][0]['case']=='weekday-0'
