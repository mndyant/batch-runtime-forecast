"""Real-browser verification against an already running local server.

Usage: python tests/ui_check.py [--url http://127.0.0.1:5001]
                              [--browser-path C:/path/to/chrome.exe]
Requires playwright; never starts the server or evaluates final holdout data.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import traceback
from datetime import date, timedelta
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5001')
    parser.add_argument('--browser-path', help='Optional installed Chromium/Chrome executable')
    args = parser.parse_args()
    report_dir = ROOT / 'docs' / 'verification'
    screenshots = ROOT / 'docs' / 'screenshots'
    report_dir.mkdir(parents=True, exist_ok=True)
    screenshots.mkdir(parents=True, exist_ok=True)
    report = {'url': args.url, 'checks': [], 'console_errors': [], 'expected_console_errors': [], 'screenshots': [], 'passed': False}

    def checked(name, details=None):
        entry = {'name': name, 'passed': True}
        if details is not None:
            entry['details'] = details
        report['checks'].append(entry)
        print(f'PASS: {name}', flush=True)

    def screenshot(page, name):
        path = screenshots / name
        page.screenshot(path=str(path), full_page=True)
        report['screenshots'].append(str(path.relative_to(ROOT)))

    def exports_disabled(page):
        for key in ('txt', 'csv', 'json'):
            expect(page.locator(f'#export-{key}')).to_be_disabled()

    def exports_enabled(page):
        for key in ('txt', 'csv', 'json'):
            expect(page.locator(f'#export-{key}')).to_be_enabled()

    def run(page):
        with page.expect_response(lambda response: response.url.endswith('/api/forecast') and response.request.method == 'POST', timeout=120000) as pending:
            page.locator('#run').click()
        response = pending.value
        expect(page.locator('#run')).to_be_enabled(timeout=120000)
        return response

    try:
        with sync_playwright() as playwright:
            options = {'headless': True}
            if args.browser_path:
                options['executable_path'] = args.browser_path
            browser = playwright.chromium.launch(**options)
            context = browser.new_context(viewport={'width': 1280, 'height': 900}, accept_downloads=True)
            page = context.new_page()
            page.on('pageerror', lambda error: report['console_errors'].append(str(error)))
            def console_message(message):
                if message.type != 'error':
                    return
                # Chromium logs the deliberately rejected CSV as an HTTP resource error.
                bucket = 'expected_console_errors' if 'status of 400' in message.text or 'status of 422' in message.text else 'console_errors'
                report[bucket].append(message.text)
            page.on('console', console_message)
            with page.expect_response(lambda response: response.url.endswith('/api/forecast') and response.request.method == 'POST', timeout=120000) as pending:
                page.goto(args.url, wait_until='domcontentloaded')
            initial = pending.value
            assert initial.ok, initial.text()
            initial_data = initial.json()
            expect(page.locator('#results')).to_be_visible(timeout=120000)
            exports_enabled(page)
            expect(page.locator('#forecast-table tbody tr')).to_have_count(7)
            expect(page.locator('#forecast-chart svg')).to_be_visible()
            expect(page.locator('#comparison-chart svg')).to_be_visible()
            assert len(initial_data['evaluation']['selected_comparison']) == 30
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Desktop body overflows'
            checked('Initial API forecast and desktop layout', {'forecast_days': 7, 'comparison_days': 30})
            screenshot(page, 'desktop-weekday.png')

            # Check the exact paired dates displayed in the historical comparison.
            comparison = initial_data['evaluation']['selected_comparison']
            expected_dates = [row['date'] for row in comparison]
            actual_titles = page.locator('#comparison-chart circle title').all_text_contents()
            assert len(actual_titles) == len(comparison) * 2
            for comparison_date in expected_dates:
                assert sum(title.startswith(comparison_date + ' / ') for title in actual_titles) == 2
            forecast_titles = page.locator('#forecast-chart circle title').all_text_contents()
            future_dates = {row['date'] for row in initial_data['forecast']}
            assert not any(title.split(' / ')[0] in future_dates and '観測実績' in title for title in forecast_titles)
            checked('Historical paired dates and absence of future actuals')

            page.locator('[data-chart="forecast"][data-metric="records"]').click()
            expect(page.locator('#forecast-chart svg')).to_have_attribute('aria-label', '件数（件）の日別グラフ')
            expect(page.locator('#deadline-legend')).to_be_hidden()
            page.locator('[data-chart="comparison"][data-metric="records"]').click()
            expect(page.locator('#comparison-chart svg')).to_have_attribute('aria-label', '件数（件）の日別グラフ')
            page.locator('[data-chart="forecast"][data-metric="runtime"]').click()
            page.locator('[data-chart="comparison"][data-metric="runtime"]').click()
            checked('Both chart metric switches')

            # Preserve download bytes, then parse each real export.
            for suffix in ('txt', 'csv', 'json'):
                with page.expect_download() as download_event:
                    page.locator(f'#export-{suffix}').click()
                download = download_event.value
                path = report_dir / download.suggested_filename
                download.save_as(str(path))
                assert path.stat().st_size > 20
                content = path.read_text(encoding='utf-8-sig')
                if suffix == 'json':
                    parsed = json.loads(content)
                    assert parsed == initial_data
                elif suffix == 'csv':
                    parsed = list(csv.DictReader(io.StringIO(content)))
                    assert len(parsed) == 7
                    assert [row['date'] for row in parsed] == [row['date'] for row in initial_data['forecast']]
                    assert float(parsed[0]['runtime_minutes']) == initial_data['forecast'][0]['runtime_minutes']
                else:
                    assert '合成データで評価 / 実データ未検証' in content
                    assert initial_data['summary']['text'] in content
            checked('TXT, CSV and JSON downloads parsed against API result')

            # A setting edit invalidates all downloads immediately.
            page.locator('#horizon').select_option('30')
            expect(page.locator('#stale')).to_be_visible()
            exports_disabled(page)
            response = run(page)
            assert response.ok, response.text()
            data30 = response.json()
            expect(page.locator('#forecast-table tbody tr')).to_have_count(30)
            assert len(data30['evaluation']['selected_comparison']) == 30
            exports_enabled(page)
            checked('30-day forecast and stale result protection')
            screenshot(page, 'desktop-30days.png')

            page.locator('#horizon').select_option('1')
            response = run(page)
            assert response.ok, response.text()
            expect(page.locator('#forecast-table tbody tr')).to_have_count(1)
            assert len(response.json()['evaluation']['selected_comparison']) == 30
            checked('1-day forecast keeps the independent 30-day comparison')

            # Stop one real request temporarily to inspect controls while work is pending.
            held = []
            def hold(route):
                held.append(route)
            page.route('**/api/forecast', hold)
            page.locator('#run').click()
            expect(page.locator('#run')).to_be_disabled()
            expect(page.locator('#sample')).to_be_disabled()
            expect(page.locator('#reset')).to_be_disabled()
            expect(page.locator('#file')).to_be_disabled()
            exports_disabled(page)
            page.wait_for_timeout(100)
            assert held, 'Forecast request was not intercepted'
            with page.expect_response(lambda response: response.url.endswith('/api/forecast'), timeout=120000) as pending:
                held[0].continue_()
            assert pending.value.ok
            page.unroute('**/api/forecast', hold)
            expect(page.locator('#run')).to_be_enabled(timeout=120000)
            exports_enabled(page)
            checked('Controls and exports disabled during a real pending request')

            # Select every supplied synthetic scenario; use an actual API response.
            for key in ('monthend', 'trend', 'slowdown', 'burst', 'idle'):
                page.locator('#sample').select_option(key)
                response = run(page)
                assert response.ok, f'{key}: {response.text()}'
                expect(page.locator('#error')).to_be_hidden()
                expect(page.locator('#scenario-table tbody tr')).to_have_count(3)
            checked('All six synthetic sample flows')

            with page.expect_download() as download_event:
                page.locator('#sample-download').click()
            observed_download = download_event.value
            observed_path = report_dir / observed_download.suggested_filename
            observed_download.save_as(str(observed_path))
            observed = observed_path.read_bytes()
            assert len(list(csv.DictReader(io.StringIO(observed.decode('utf-8-sig'))))) >= 150
            checked('Observed sample CSV download')

            page.locator('#file').set_input_files({'name': 'observed-upload.csv', 'mimeType': 'text/csv', 'buffer': observed})
            expect(page.locator('#file-name')).to_have_text('observed-upload.csv')
            exports_disabled(page)
            response = run(page)
            assert response.ok, response.text()
            exports_enabled(page)
            checked('Valid CSV upload')
            page.locator('#clear-file').click()
            expect(page.locator('#file-name')).to_have_text('ファイル未選択')
            assert not page.locator('#file').input_value()
            exports_disabled(page)
            checked('CSV clear marks results stale')

            # A gap inside an otherwise valid 150-row calendar must show an error.
            beginning = date(2025, 1, 1)
            invalid = 'date,records,runtime_minutes\n' + ''.join(f'{beginning + timedelta(days=i)},100,5\n' for i in range(151) if i != 75)
            page.locator('#file').set_input_files({'name': 'missing-date.csv', 'mimeType': 'text/csv', 'buffer': invalid.encode('utf-8')})
            response = run(page)
            assert not response.ok
            expect(page.locator('#error')).to_be_visible()
            assert page.locator('#error').inner_text().strip()
            exports_disabled(page)
            expect(page.locator('#run')).to_be_enabled()
            screenshot(page, 'desktop-input-error.png')
            checked('Rejected missing-date CSV and recoverable error display')

            with page.expect_response(lambda response: response.url.endswith('/api/forecast'), timeout=120000) as pending:
                page.locator('#reset').click()
            assert pending.value.ok
            expect(page.locator('#run')).to_be_enabled(timeout=120000)
            expect(page.locator('#file-name')).to_have_text('ファイル未選択')
            expect(page.locator('#horizon')).to_have_value('7')
            expect(page.locator('#sample')).to_have_value('weekday')
            expect(page.locator('#start-time')).to_have_value('22:00')
            expect(page.locator('#deadline-time')).to_have_value('00:00')
            expect(page.locator('#buffer')).to_have_value('15')
            expect(page.locator('#stress')).to_have_value('120')
            expect(page.locator('#error')).to_be_hidden()
            exports_enabled(page)
            checked('Reset restores defaults and computes a fresh result')

            # Force a clock crossing and render full dates for all three timestamps.
            page.locator('#start-time').fill('23:30')
            page.locator('#deadline-time').fill('00:15')
            page.locator('#stress').fill('150')
            response = run(page)
            assert response.ok, response.text()
            clock_data = response.json()
            first = clock_data['forecast'][0]
            assert first['deadline_at'][:10] > first['date']
            first_row = page.locator('#forecast-table tbody tr').first.inner_text()
            assert first['deadline_at'][:10] in first_row
            assert first['latest_start_at'][:10] in first_row
            assert any(abs(row['multiplier'] - 1.5) < 1e-8 for row in clock_data['scenarios'])
            checked('Next-day deadline, full timestamp dates and custom scenario multiplier')

            page.locator('.evaluation-card summary').click()
            expect(page.locator('#evaluation-table')).to_be_visible()
            assert page.locator('#evaluation-table tbody tr').count() >= 3
            checked('Evaluation details and limitations visible')

            page.set_viewport_size({'width': 390, 'height': 844})
            expect(page.locator('#run')).to_be_visible()
            page.wait_for_timeout(200)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile body overflows'
            table = page.locator('#forecast-table').evaluate('(el) => ({width:el.scrollWidth,parent:el.parentElement.clientWidth})')
            assert table['width'] > table['parent'], 'Mobile table should scroll within its container'
            assert page.locator('.input-panel').bounding_box()['y'] < page.locator('.result-area').bounding_box()['y']
            screenshot(page, 'mobile-390.png')
            checked('390px mobile flow and contained horizontal table scrolling')

            page.locator('[data-chart="forecast"][data-metric="records"]').click()
            expect(page.locator('#forecast-chart svg')).to_have_attribute('aria-label', '件数（件）の日別グラフ')
            page.locator('#horizon').select_option('30')
            response = run(page)
            assert response.ok
            expect(page.locator('#forecast-table tbody tr')).to_have_count(30)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            screenshot(page, 'mobile-390-30days.png')
            checked('Mobile metric switch and 30-day rerun')
            assert not report['console_errors'], report['console_errors']
            checked('No browser JavaScript or console errors')
            report['passed'] = True
            context.close()
            browser.close()
    except Exception as error:
        report['failure'] = str(error)
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], flush=True)
    (report_dir / 'ui.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Report: {report_dir / 'ui.json'}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
