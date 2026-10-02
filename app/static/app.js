'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const form = $('forecast-form');
  const exports = ['export-txt', 'export-csv', 'export-json'];
  const models = {weekday4: '直近4週の同曜日平均（基準）', mean7: '直近7日平均', ridge: 'カレンダー付き回帰'};
  const statusLabels = {late: '締切超過', tight: '安全余裕不足', ok: '余裕あり'};
  const chartMetric = {forecast: 'runtime', comparison: 'runtime'};
  let result = null;
  let busy = false;
  let stale = true;
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num = (value, digits = 1) => Number.isFinite(Number(value)) && value !== null ? Number(value).toLocaleString('ja-JP', {maximumFractionDigits: digits}) : '—';
  const time = (value) => value ? String(value).replace('T', ' ').replace(/(\d{2}:\d{2}):\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?$/, (_, hhmm) => hhmm).replace(/[+-]\d{2}:?\d{2}$/, '') : '—';
  const modelLabel = (value) => models[value] || value || '—';
  const signedMinutes = (value) => `${Number(value) > 0 ? '+' : ''}${num(value)}分`;
  function exportState() { exports.forEach(id => { $(id).disabled = busy || stale || !result; }); }
  function fileState() {
    const file = $('file').files[0];
    $('file-name').textContent = file ? file.name : 'ファイル未選択';
    $('clear-file').disabled = !file || busy;
  }
  function markStale() {
    stale = true;
    $('stale').hidden = !result;
    $('error').hidden = true;
    $('run-status').textContent = result ? '入力条件を変更しました。再実行が必要です。' : '入力条件を指定して予測を実行してください。';
    exportState();
    fileState();
  }
  function setBusy(value) {
    busy = value;
    $('input-fields').disabled = value;
    $('run').textContent = value ? '計算中…' : '予測を実行';
    document.querySelector('.result-area').setAttribute('aria-busy', String(value));
    document.querySelectorAll('[data-chart]').forEach(button => { button.disabled = value; });
    exportState();
    fileState();
  }
  async function run(event) {
    if (event) event.preventDefault();
    if (busy || !form.reportValidity()) return;
    const request = new FormData(form);
    if (!$('file').files.length) request.delete('file');
    stale = true;
    $('error').hidden = true;
    $('run-status').textContent = '予測と過去区間の監査評価を計算しています。';
    setBusy(true);
    try {
      const response = await fetch('/api/forecast', {method: 'POST', body: request});
      let data;
      try { data = await response.json(); } catch (_) { throw new Error('結果を読み取れませんでした。サーバーの稼働状態を確認してください。'); }
      if (!response.ok) throw new Error(typeof data.error === 'string' ? data.error : data.error?.message || data.message || '入力データ・条件を確認してください。');
      if (!Array.isArray(data.forecast) || !data.forecast.length) throw new Error('予測結果が空です。入力データを確認してください。');
      result = data;
      render();
      stale = false;
      $('stale').hidden = true;
      $('run-status').textContent = `${data.forecast.length}日分の予測を更新しました。`;
    } catch (error) {
      $('error').textContent = error.message || '予測に失敗しました。再実行してください。';
      $('error').hidden = false;
      $('stale').hidden = !result;
      $('run-status').textContent = '予測を更新できませんでした。';
    } finally { setBusy(false); }
  }
  function render() {
    const {overview, forecast, settings, evaluation} = result;
    $('empty-state').hidden = true;
    $('results').hidden = false;
    $('data-context').textContent = `入力: ${result.source} ｜ 観測末日: ${result.as_of} ｜ ${result.observations}日分 ｜ 予測: ${settings.horizon}日先 ｜ 採用手法: ${result.model_label || modelLabel(result.selected_model)}`;
    $('late-days').textContent = `${num(overview.late_days, 0)}日`;
    $('tight-days').textContent = `${num(overview.tight_days, 0)}日`;
    $('min-slack').textContent = signedMinutes(overview.min_slack_minutes);
    $('worst-date').textContent = `対象日 ${overview.worst_date}`;
    $('latest-start').textContent = time(overview.latest_start_at);
    $('late-days').closest('.metric').classList.toggle('warning', overview.late_days > 0);
    $('tight-days').closest('.metric').classList.toggle('caution', overview.tight_days > 0);
    $('min-slack').closest('.metric').classList.toggle('warning', overview.min_slack_minutes < 0);
    $('headline').textContent = result.summary.headline;
    $('summary-items').innerHTML = (result.summary.items || []).map(item => `<div><dt>${esc(item.label)}</dt><dd>${esc(item.value)}</dd></div>`).join('');
    $('forecast-table').querySelector('tbody').innerHTML = forecast.map(row => `<tr data-date="${esc(row.date)}"><td>${esc(row.date)}</td><td class="numeric">${num(row.records, 0)}件</td><td class="numeric">${num(row.runtime_minutes)}分</td><td class="date-cell">${esc(time(row.start_at))}</td><td class="date-cell">${esc(time(row.finish_at))}</td><td class="date-cell">${esc(time(row.deadline_at))}</td><td class="numeric ${row.slack_minutes < 0 ? 'negative' : ''}">${signedMinutes(row.slack_minutes)}</td><td class="date-cell">${esc(time(row.latest_start_at))}</td><td><span class="status-pill ${['ok','tight','late'].includes(row.status) ? row.status : 'tight'}">${esc(statusLabels[row.status] || row.status)}</span></td></tr>`).join('');
    $('scenario-table').querySelector('tbody').innerHTML = (result.scenarios || []).map(row => `<tr><td>${esc(row.label)}</td><td class="numeric">${num(row.multiplier * 100, 0)}%</td><td class="numeric">${num(row.late_days, 0)}日</td><td class="numeric">${num(row.tight_days, 0)}日</td><td class="numeric ${row.min_slack_minutes < 0 ? 'negative' : ''}">${signedMinutes(row.min_slack_minutes)}</td><td class="date-cell">${esc(time(row.latest_start_at))}</td></tr>`).join('');
    $('comparison-context').textContent = `監査起点: ${evaluation.comparison_origin} ｜ ${modelLabel(evaluation.comparison_model)} ｜ 比較: ${evaluation.selected_comparison.length}日間`;
    $('model-context').textContent = `採用手法: ${result.model_label || modelLabel(result.selected_model)} / 監査起点: ${(evaluation.audit_origins || []).join('、')}`;
    $('selection-rule').textContent = evaluation.selection_rule;
    $('evaluation-table').querySelector('tbody').innerHTML = (evaluation.summary || []).map(row => `<tr><td>${esc(modelLabel(row.model))}</td><td class="numeric">${num(row.horizon, 0)}日</td><td class="numeric">${num(row.volume_mae)}件</td><td class="numeric">${num(row.runtime_mae)}分</td><td class="numeric">${num(row.runtime_rmse)}分</td><td class="numeric">${num(row.endpoint_abs_error)}分</td><td class="numeric">${row.improvement_pct == null ? 'N/A' : `${num(row.improvement_pct)}%`}</td><td class="numeric">${num(row.folds, 0)}</td></tr>`).join('');
    $('limitations').innerHTML = (result.limitations || []).map(item => `<li>${esc(item)}</li>`).join('');
    drawCharts();
  }
  function drawCharts() {
    if (!result) return;
    const history = result.history.slice(-28);
    const future = result.forecast;
    const metric = chartMetric.forecast === 'runtime' ? 'runtime_minutes' : 'records';
    const dates = [...history.map(r => r.date), ...future.map(r => r.date)];
    const observed = history.map((r, i) => ({i, value: r[metric], date: r.date}));
    const predicted = future.map((r, i) => ({i: history.length + i, value: r[metric], date: r.date}));
    if (observed.length) predicted.unshift({...observed[observed.length - 1], connector: true});
    const start = result.settings.start_time.split(':').map(Number);
    const end = result.settings.deadline_time.split(':').map(Number);
    let windowMinutes = end[0] * 60 + end[1] - start[0] * 60 - start[1];
    if (windowMinutes <= 0) windowMinutes += 1440;
    $('deadline-legend').hidden = metric !== 'runtime_minutes';
    drawChart('forecast-chart', dates, [{points: observed, color: '#64778e', name: '観測実績'}, {points: predicted, color: '#235bc0', name: '今後の予測', dashed: true}], metric === 'records' ? '件数（件）' : '所要時間（分）', {boundary: history.length - .5, threshold: metric === 'runtime_minutes' ? windowMinutes : null});
    const rows = result.evaluation.selected_comparison;
    const runtime = chartMetric.comparison === 'runtime';
    drawChart('comparison-chart', rows.map(r => r.date), [{points: rows.map((r, i) => ({i, date:r.date, value: runtime ? r.actual_runtime : r.actual_records})), color:'#64778e', name:'実績'}, {points: rows.map((r, i) => ({i, date:r.date, value: runtime ? r.predicted_runtime : r.predicted_records})), color:'#235bc0', name:'当時の予測', dashed:true}], runtime ? '所要時間（分）' : '件数（件）');
  }
  function drawChart(target, dates, series, unit, options = {}) {
    const narrow = $(target).clientWidth < 500;
    const width = Math.max(310, $(target).clientWidth || 700), height = narrow ? 218 : 250;
    const margin = {left: narrow ? 48 : 58, right: 14, top: 30, bottom: 35};
    const plotWidth = width - margin.left - margin.right, plotHeight = height - margin.top - margin.bottom;
    const values = series.flatMap(s => s.points.map(p => Number(p.value))).filter(Number.isFinite);
    if (options.threshold != null) values.push(options.threshold);
    const max = Math.max(...values, 1) * 1.12;
    const x = i => margin.left + (dates.length <= 1 ? plotWidth / 2 : i * plotWidth / (dates.length - 1));
    const y = value => margin.top + plotHeight * (1 - Number(value) / max);
    let parts = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(unit)}の日別グラフ"><title>${esc(unit)}の日別グラフ</title><text x="0" y="14" fill="#596b7e" font-size="10">${esc(unit)}</text>`];
    for (let n = 0; n <= 4; n++) {
      const value = max * n / 4, py = y(value);
      parts.push(`<line x1="${margin.left}" y1="${py}" x2="${width - margin.right}" y2="${py}" stroke="#e5ebf2"/><text x="${margin.left - 7}" y="${py + 3}" text-anchor="end" fill="#596b7e" font-size="9">${num(value, 0)}</text>`);
    }
    if (options.boundary != null && options.boundary >= 0) {
      const bx = x(options.boundary);
      parts.push(`<rect x="${bx}" y="${margin.top}" width="${width - margin.right - bx}" height="${plotHeight}" fill="#edf3ff" opacity=".6"/><line x1="${bx}" y1="${margin.top}" x2="${bx}" y2="${height - margin.bottom}" stroke="#a2b9dd" stroke-dasharray="3 3"/><text x="${Math.min(bx + 5, width - 60)}" y="24" fill="#235bc0" font-size="9">予測開始</text>`);
    }
    if (options.threshold != null) parts.push(`<line x1="${margin.left}" y1="${y(options.threshold)}" x2="${width - margin.right}" y2="${y(options.threshold)}" stroke="#b07628" stroke-dasharray="5 4"><title>締切まで ${num(options.threshold)}分</title></line>`);
    series.forEach(s => {
      const valid = s.points.filter(p => Number.isFinite(Number(p.value)));
      parts.push(`<polyline points="${valid.map(p => `${x(p.i)},${y(p.value)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dashed ? 'stroke-dasharray="5 3"' : ''}/>`);
      valid.filter(p => !p.connector).forEach(p => parts.push(`<circle cx="${x(p.i)}" cy="${y(p.value)}" r="${valid.length === 1 ? 4 : 2.5}" fill="${s.color}"><title>${esc(p.date)} / ${esc(s.name)} / ${num(p.value)}${unit.startsWith('件数') ? '件' : '分'}</title></circle>`));
    });
    const ticks = dates.length > 2 ? [0, Math.floor((dates.length - 1) / 2), dates.length - 1] : [...dates.keys()];
    ticks.forEach((i, k) => parts.push(`<text x="${x(i)}" y="${height - 14}" text-anchor="${k === 0 ? 'start' : k === ticks.length - 1 ? 'end' : 'middle'}" fill="#596b7e" font-size="9">${esc(dates[i])}</text>`));
    parts.push('</svg>');
    $(target).innerHTML = parts.join('');
  }
  function download(content, suffix, mime) {
    if (!result || busy || stale) return;
    const blob = new Blob([content], {type: mime});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url;
    link.download = `batch-forecast-${result.as_of}.${suffix}`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const csvValue = (value) => `"${String(value ?? '').replace(/"/g, '""')}"`;
  form.addEventListener('submit', run);
  form.addEventListener('input', markStale);
  form.addEventListener('change', markStale);
  $('clear-file').addEventListener('click', () => { $('file').value = ''; markStale(); });
  $('reset').addEventListener('click', () => { HTMLFormElement.prototype.reset.call(form); fileState(); markStale(); run(); });
  $('sample-download').addEventListener('click', () => {
    const link = document.createElement('a'); link.href = `/api/sample/${encodeURIComponent($('sample').value)}`; link.download = `observed-${$('sample').value}.csv`; document.body.append(link); link.click(); link.remove();
  });
  document.querySelectorAll('[data-chart]').forEach(button => button.addEventListener('click', () => {
    chartMetric[button.dataset.chart] = button.dataset.metric;
    document.querySelectorAll(`[data-chart="${button.dataset.chart}"]`).forEach(peer => { const active = peer === button; peer.classList.toggle('active', active); peer.setAttribute('aria-pressed', String(active)); });
    drawCharts();
  }));
  $('export-txt').addEventListener('click', () => download(`バッチ完了時刻・締切予測\n合成データで評価 / 実データ未検証\n${$('data-context').textContent}\n開始 ${result.settings.start_time} / 締切 ${result.settings.deadline_time} / 安全余裕 ${result.settings.buffer_minutes}分\n\n${result.summary.text}\n\n制約\n${result.limitations.join('\n')}\n`, 'txt', 'text/plain;charset=utf-8'));
  $('export-csv').addEventListener('click', () => {
    const columns = ['date','records','runtime_minutes','start_at','deadline_at','finish_at','latest_start_at','slack_minutes','status'];
    download('\ufeff' + [columns.join(','), ...result.forecast.map(row => columns.map(key => csvValue(row[key])).join(','))].join('\r\n') + '\r\n', 'csv', 'text/csv;charset=utf-8');
  });
  $('export-json').addEventListener('click', () => download(JSON.stringify(result, null, 2) + '\n', 'json', 'application/json;charset=utf-8'));
  let resizeTimer;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawCharts, 100); });
  fileState();
  run();
})();
