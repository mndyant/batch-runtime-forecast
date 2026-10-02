import io
import pytest
from app import create_app,settings_from
from scripts.generate_demo import generate


@pytest.fixture
def client():
    app=create_app()
    app.config['TESTING']=True
    return app.test_client()


@pytest.mark.parametrize('form',[{'horizon':'2'},{'horizon':'nan'},{'buffer_minutes':'-1'},
                                {'stress_pct':'501'},{'start_time':'25:00'},{'deadline_time':'7:00'}])
def test_invalid_settings(form):
    with pytest.raises(ValueError): settings_from(form)


def test_api_sample_upload_and_plan(client):
    sample=client.get('/api/sample/weekday')
    assert sample.status_code==200 and len(sample.data.splitlines())==181
    response=client.post('/api/forecast',data={'file':(io.BytesIO(sample.data),'history.csv'),'horizon':'1'})
    assert response.status_code==200
    body=response.json
    assert len(body['history'])==180 and len(body['forecast'])==1
    assert body['forecast'][0]['date']>body['as_of']
    assert body['forecast'][0]['latest_start_at'].endswith('+09:00')
    assert len(body['evaluation']['selected_comparison'])==30
    assert len(body['evaluation']['summary'])==6
    assert '実業務' in body['summary']['text']
    assert '利用者提供' in body['source']
    assert all(r['date']<=body['as_of'] for r in body['evaluation']['selected_comparison'])


def test_api_errors(client):
    assert client.post('/api/forecast',data={'sample':'../truth'}).status_code==400
    assert client.get('/api/sample/bogus').status_code==404
    result=client.post('/api/forecast',data={'file':(io.BytesIO(b'bad,data'),'bad.csv')})
    assert result.status_code==400 and 'error' in result.json
    result=client.post('/api/forecast',data={'file':(io.BytesIO(b'x'*(2*1024*1024+1)),'large.csv')})
    assert result.status_code==413


def test_out_of_scope_prediction_is_error_not_clipping():
    from unittest.mock import patch
    from forecasting.models import predict
    history,_,_=generate('weekday',0)
    with patch('forecasting.models.apply_ridge',return_value=1e10):
        with pytest.raises(ValueError,match='対応範囲'):
            predict(history,30,'ridge')
