"""Hand-calculated core contracts; all synthetic cases use exploration seeds only."""
import unittest

import numpy as np
import pandas as pd

from forecasting.data import COLUMNS, read_csv
from forecasting.models import MODELS, predict
from forecasting.planning import daily_plan, overview, schedule
from scripts.generate_demo import SCENARIOS, generate


def history_frame(count=180):
    index = np.arange(count)
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=count),
        'records': 1000 + 10 * index,
        'runtime_minutes': 10.0 + index / 10,
    })


def csv_bytes(frame):
    return frame.to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8')


class InputTests(unittest.TestCase):
    def test_zero_idle_startup_and_decreasing_values_are_valid(self):
        frame = history_frame(150)
        frame['records'] = np.arange(149, -1, -1)
        frame['runtime_minutes'] = np.arange(149, -1, -1) / 10
        checked = read_csv(csv_bytes(frame))
        pd.testing.assert_frame_equal(checked, frame, check_dtype=False)
        frame.loc[149, 'runtime_minutes'] = 3.5
        self.assertEqual(read_csv(csv_bytes(frame)).runtime_minutes.iloc[-1], 3.5)

    def test_unsorted_csv_is_sorted_without_changing_observations(self):
        frame = history_frame(150)
        checked = read_csv(csv_bytes(frame.iloc[::-1]))
        pd.testing.assert_frame_equal(checked, frame, check_dtype=False)

    def test_rejects_missing_duplicate_malformed_and_nonfinite_rows(self):
        source = history_frame(151)
        examples = {}
        examples['missing calendar day'] = source.drop(index=40)
        duplicate = source.copy()
        duplicate.loc[40, 'date'] = duplicate.loc[39, 'date']
        examples['duplicate date'] = duplicate
        for column, value, label in [
            ('date', '2026-02-30', 'impossible date'),
            ('date', '', 'missing date'),
            ('records', -1, 'negative records'),
            ('records', 2.5, 'fractional records'),
            ('records', '1件', 'non numeric records'),
            ('runtime_minutes', -1, 'negative runtime'),
            ('runtime_minutes', np.inf, 'infinite runtime'),
            ('runtime_minutes', '', 'missing runtime'),
            ('runtime_minutes', 0, 'positive records with zero runtime'),
        ]:
            changed = source.copy().astype(object)
            changed.loc[40, column] = value
            examples[label] = changed
        examples['missing column'] = source.drop(columns='records')
        examples['future truth column'] = source.assign(actual_future_records=123)
        examples['too few days'] = source.iloc[:149]
        for label, frame in examples.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                read_csv(csv_bytes(frame))
        for content in [b'', b'\xff\xfe', b'date,records,runtime_minutes\ninvalid,1,2\n']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_csv(content)


class BaselineTests(unittest.TestCase):
    def test_weekday_uses_last_four_observed_matches(self):
        frame = history_frame()
        result = predict(frame, 30, 'weekday4')
        for row in result['forecast']:
            day = pd.Timestamp(row['date'])
            observed = frame[frame.date.dt.dayofweek == day.dayofweek].iloc[-4:]
            self.assertEqual(len(observed), 4)
            self.assertAlmostEqual(row['records'], sum(observed.records) / 4)
            self.assertAlmostEqual(row['runtime_minutes'], sum(observed.runtime_minutes) / 4)
        self.assertEqual(result['parameters']['fit_end'], '2026-06-29')

    def test_mean7_is_frozen_observed_mean_for_all_forecast_days(self):
        frame = history_frame()
        result = predict(frame, 30, 'mean7')
        expected_volume = sum(frame.records.iloc[-7:]) / 7
        expected_runtime = sum(frame.runtime_minutes.iloc[-7:]) / 7
        self.assertEqual(len(result['forecast']), 30)
        for row in result['forecast']:
            self.assertAlmostEqual(row['records'], expected_volume)
            self.assertAlmostEqual(row['runtime_minutes'], expected_runtime)

    def test_all_models_allow_zero_and_decreasing_series(self):
        for kind in ['zero', 'decreasing']:
            frame = history_frame()
            frame['records'] = 0 if kind == 'zero' else np.arange(180, 0, -1)
            frame['runtime_minutes'] = 0. if kind == 'zero' else np.arange(180, 0, -1) / 10
            original = frame.copy(deep=True)
            for model in MODELS:
                with self.subTest(kind=kind, model=model):
                    result = predict(frame, 30, model)
                    for row in result['forecast']:
                        for key in ['records', 'runtime_minutes']:
                            self.assertTrue(np.isfinite(row[key]))
                            self.assertGreaterEqual(row[key], 0)
                            if kind == 'zero':
                                self.assertEqual(row[key], 0)
            pd.testing.assert_frame_equal(frame, original)

    def test_predictor_interface_rejects_future_truth_and_metadata_columns(self):
        for column in ['actual_future_records', 'scenario', 'seed']:
            with self.subTest(column=column), self.assertRaises(ValueError):
                predict(history_frame().assign(**{column: 999}), 7, 'ridge')
        for horizon, model in [(0, 'ridge'), (31, 'ridge'), (7, 'unknown')]:
            with self.subTest(horizon=horizon, model=model), self.assertRaises(ValueError):
                predict(history_frame(), horizon, model)


class DeadlineTests(unittest.TestCase):
    def test_midnight_deadline_latest_start_and_seconds_ceil(self):
        row = daily_plan('2026-01-31', 90.001, '22:00', '00:00', 15)
        self.assertEqual(row['start_at'], '2026-01-31T22:00:00+09:00')
        self.assertEqual(row['deadline_at'], '2026-02-01T00:00:00+09:00')
        self.assertEqual(row['finish_at'], '2026-01-31T23:30:01+09:00')
        self.assertEqual(row['latest_start_at'], '2026-01-31T22:14:59+09:00')
        self.assertAlmostEqual(row['slack_minutes'], 29 + 59 / 60)
        self.assertEqual(row['status'], 'ok')

    def test_equal_clock_time_means_next_day_and_multi_day_duration(self):
        row = daily_plan('2026-01-31', 1500, '22:00', '22:00', 15)
        self.assertEqual(row['deadline_at'], '2026-02-01T22:00:00+09:00')
        self.assertEqual(row['finish_at'], '2026-02-01T23:00:00+09:00')
        self.assertEqual(row['latest_start_at'], '2026-01-31T20:45:00+09:00')
        self.assertEqual(row['slack_minutes'], -60)
        self.assertEqual(row['status'], 'late')
        long = daily_plan('2026-01-31', 3000, '22:00', '00:00', 15)
        self.assertEqual(long['finish_at'], '2026-02-03T00:00:00+09:00')
        self.assertEqual(long['latest_start_at'], '2026-01-29T21:45:00+09:00')

    def test_boundary_status_includes_exact_deadline_as_tight(self):
        for duration, status in [(0, 'ok'), (105, 'ok'), (106, 'tight'), (120, 'tight'), (120.001, 'late')]:
            with self.subTest(duration=duration):
                self.assertEqual(daily_plan('2026-01-01', duration, '22:00', '00:00', 15)['status'], status)
        row = daily_plan('2026-01-01', 60, '01:00', '03:00', 15)
        self.assertEqual(row['deadline_at'], '2026-01-01T03:00:00+09:00')

    def test_scenario_multiplier_is_applied_before_schedule_and_overview(self):
        forecasts = [
            {'date': '2026-01-01', 'records': 100, 'runtime_minutes': 90},
            {'date': '2026-01-02', 'records': 200, 'runtime_minutes': 110},
        ]
        settings = {'start_time': '22:00', 'deadline_time': '00:00', 'buffer_minutes': 15}
        rows = schedule(forecasts, settings, 1.2)
        self.assertEqual(rows[0]['runtime_minutes'], 108)
        self.assertEqual(rows[0]['status'], 'tight')
        self.assertEqual(rows[1]['runtime_minutes'], 132)
        self.assertEqual(rows[1]['status'], 'late')
        summary = overview(rows)
        self.assertEqual(summary['late_days'], 1)
        self.assertEqual(summary['tight_days'], 1)
        self.assertEqual(summary['worst_date'], '2026-01-02')
        self.assertEqual(summary['min_slack_minutes'], -12)
        self.assertEqual(forecasts[1]['runtime_minutes'], 110)


class GeneratorTests(unittest.TestCase):
    def test_exploration_cases_are_reproducible_and_keep_truth_separate(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                observed, truth, metadata = generate(scenario, seed=0)
                observed_again, truth_again, metadata_again = generate(scenario, seed=0)
                pd.testing.assert_frame_equal(observed, observed_again)
                pd.testing.assert_frame_equal(truth, truth_again)
                self.assertEqual(metadata, metadata_again)
                self.assertEqual(list(observed.columns), COLUMNS)
                self.assertEqual(list(truth.columns), COLUMNS)
                self.assertEqual((len(observed), len(truth)), (180, 30))
                self.assertLess(observed.date.max(), truth.date.min())
                self.assertEqual(truth.date.min() - observed.date.max(), pd.Timedelta(days=1))
                read_csv(csv_bytes(observed))
                self.assertTrue((truth.records >= 0).all())
                self.assertTrue((truth.runtime_minutes >= 0).all())
                self.assertEqual(metadata['seed'], 0)
                self.assertEqual(metadata['scenario'], scenario)
        observed0, _, _ = generate('weekday', seed=0)
        observed1, _, _ = generate('weekday', seed=1)
        self.assertFalse(observed0.equals(observed1))

class SettingsTests(unittest.TestCase):
    def test_defaults_and_valid_boundaries(self):
        from app import settings_from
        settings = settings_from({})
        self.assertEqual(settings, {'horizon': 7, 'start_time': '22:00', 'deadline_time': '00:00', 'buffer_minutes': 15., 'stress_pct': 120.})
        settings = settings_from({'horizon': '30', 'start_time': '00:00', 'deadline_time': '23:59', 'buffer_minutes': '0', 'stress_pct': '500'})
        self.assertEqual(settings['horizon'], 30)
        self.assertEqual(settings['buffer_minutes'], 0)
        self.assertEqual(settings['stress_pct'], 500)

    def test_invalid_and_nonfinite_settings_are_rejected(self):
        from app import settings_from
        for field, value in [
            ('horizon', '2'), ('horizon', '7.5'), ('horizon', 'NaN'),
            ('start_time', '24:00'), ('start_time', '9:00'),
            ('deadline_time', '12:60'), ('deadline_time', ''),
            ('buffer_minutes', '-1'), ('buffer_minutes', 'inf'), ('buffer_minutes', '1441'),
            ('stress_pct', '0'), ('stress_pct', '501'), ('stress_pct', 'NaN'),
        ]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                settings_from({field: value})


if __name__ == '__main__':
    unittest.main()
