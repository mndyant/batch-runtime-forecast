"""Temporal audit contracts. No final evaluation seed or saved truth is read here."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecasting import models
from forecasting.data import COLUMNS
from scripts.generate_demo import generate
from tests.test_core import history_frame


class RidgeAuditTests(unittest.TestCase):
    def test_ridge_scaling_fits_only_the_origin_training_window(self):
        frame = history_frame(180)
        origin = 148
        prefix = frame.iloc[:origin].copy()
        result = models.predict(prefix, 7, 'ridge')
        training = prefix.iloc[-120:]
        volume = training.records.to_numpy(dtype=float)
        runtime = training.runtime_minutes.to_numpy(dtype=float)
        vx = []
        tx = []
        for i in range(28, len(training)):
            day = training.date.iloc[i]
            common = [float(day.dayofweek == j) for j in range(7)] + [
                np.sin(2 * np.pi * day.day / 31), np.cos(2 * np.pi * day.day / 31),
                float(day.days_in_month - day.day < 3), float(day.day <= 3), day.toordinal() / 365,
            ]
            vx.append(common + [volume[i - 1], volume[i - 7], np.mean(volume[i - 7:i]), np.mean(volume[i - 28:i])])
            tx.append(common + [volume[i], runtime[i - 1], runtime[i - 7], np.mean(runtime[i - 7:i]), np.mean(runtime[i - 28:i])])
        for name, features, target in [('volume', vx, volume[28:]), ('runtime', tx, runtime[28:])]:
            with self.subTest(target=name):
                x = np.asarray(features)
                expected_std = x.std(axis=0)
                expected_std[expected_std < 1e-8] = 1
                np.testing.assert_allclose(result['parameters'][name]['mean'], x.mean(axis=0))
                np.testing.assert_allclose(result['parameters'][name]['std'], expected_std)
                self.assertAlmostEqual(result['parameters'][name]['center'], np.mean(target))
        self.assertEqual(result['parameters']['fit_start'], str(training.date.iloc[0].date()))
        self.assertEqual(result['parameters']['fit_end'], str(prefix.date.iloc[-1].date()))
        self.assertEqual(result['parameters']['training_rows'], 92)

    def test_recursive_features_use_predicted_volume_and_runtime_prefixes(self):
        frame, _, _ = generate('burst', seed=1)
        prefix = frame.iloc[:120].copy()
        volume_features = models.volume_features
        runtime_features = models.runtime_features
        future_volume_calls = []
        future_runtime_calls = []

        def capture_volume(day, volumes):
            if day > prefix.date.iloc[-1]:
                future_volume_calls.append((day, list(volumes)))
            return volume_features(day, volumes)

        def capture_runtime(day, volumes, times, volume):
            if day > prefix.date.iloc[-1]:
                future_runtime_calls.append((day, list(volumes), list(times), volume))
            return runtime_features(day, volumes, times, volume)

        with patch.object(models, 'volume_features', side_effect=capture_volume), patch.object(models, 'runtime_features', side_effect=capture_runtime):
            result = models.predict(prefix, 30, 'ridge')
        self.assertEqual(len(future_volume_calls), 30)
        self.assertEqual(len(future_runtime_calls), 30)
        for step, row in enumerate(result['forecast']):
            with self.subTest(step=step):
                expected_volumes = prefix.records.astype(float).tolist() + [r['records'] for r in result['forecast'][:step]]
                expected_times = prefix.runtime_minutes.astype(float).tolist() + [r['runtime_minutes'] for r in result['forecast'][:step]]
                vday, volumes = future_volume_calls[step]
                tday, runtime_volumes, times, current_volume = future_runtime_calls[step]
                self.assertEqual(str(vday.date()), row['date'])
                self.assertEqual(tday, vday)
                self.assertEqual(volumes, expected_volumes)
                self.assertEqual(runtime_volumes, expected_volumes)
                self.assertEqual(times, expected_times)
                self.assertEqual(current_volume, row['records'])
        short = models.predict(prefix, 7, 'ridge')
        self.assertEqual(short['forecast'], result['forecast'][:7])
        self.assertEqual(short['parameters'], result['parameters'])


class EvaluationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluation import backtest
        cls.evaluation = backtest

    def test_standard_origins_and_all_fold_date_boundaries(self):
        development, audit = self.evaluation.origins(180)
        self.assertEqual(list(development), [60, 67, 74, 81, 88])
        self.assertEqual(list(audit), [120, 127, 134, 141, 148])
        for count in [150, 151, 165, 179, 180, 210]:
            with self.subTest(count=count):
                frame = history_frame(count)
                development, audit = self.evaluation.origins(count)
                development_end = count - 60
                self.assertTrue(len(development) > 0)
                self.assertTrue(len(audit) > 0)
                for split, fold_origins, end in [('development', development, development_end), ('audit', audit, count)]:
                    for origin in fold_origins:
                        self.assertGreaterEqual(origin, 60)
                        self.assertLessEqual(origin + 30, end)
                        self.assertLess(frame.date.iloc[origin - 1], frame.date.iloc[origin])
                        if split == 'audit':
                            self.assertGreaterEqual(origin, development_end)

    def test_future_mutation_and_row_append_cannot_change_same_origin_fit(self):
        frame, truth, metadata = generate('slowdown', seed=2)
        changed = frame.copy(deep=True)
        changed.loc[120:, 'records'] = 900_000_000
        changed.loc[120:, 'runtime_minutes'] = 9000.
        extended = pd.concat([changed, truth], ignore_index=True)
        extended['scenario'] = metadata['scenario']
        extended['actual_future_records'] = -12345
        original = frame.copy(deep=True)
        for model in models.MODELS:
            with self.subTest(model=model):
                expected = self.evaluation.predict_at(frame, 120, 30, model)
                for alternative in [changed, extended]:
                    actual = self.evaluation.predict_at(alternative, 120, 30, model)
                    self.assertEqual(actual['forecast'], expected['forecast'])
                    self.assertEqual(actual['parameters'], expected['parameters'])
        pd.testing.assert_frame_equal(frame, original)

    def test_predict_at_passes_only_three_observed_columns(self):
        frame = history_frame().assign(seed=1000, actual_future_records=8000)
        observed_calls = []

        def spy(history, horizon, model):
            observed_calls.append((history.copy(deep=True), horizon, model))
            return {'forecast': [], 'parameters': {}}

        # The evaluator may import predict directly; patch its own binding.
        with patch.object(self.evaluation, 'predict', side_effect=spy):
            self.evaluation.predict_at(frame, 120, 7, 'ridge')
        self.assertEqual(len(observed_calls), 1)
        passed, horizon, model = observed_calls[0]
        self.assertEqual(list(passed.columns), COLUMNS)
        pd.testing.assert_frame_equal(passed.reset_index(drop=True), frame[COLUMNS].iloc[:120].reset_index(drop=True))
        self.assertEqual((horizon, model), (7, 'ridge'))

    def test_selection_is_invariant_after_fixed_development_end(self):
        frame, truth, _ = generate('trend', seed=3)
        mutated = frame.copy(deep=True)
        mutated.loc[120:, 'records'] = 999_999_999
        mutated.loc[120:, 'runtime_minutes'] = 9500.
        extended = pd.concat([mutated, truth], ignore_index=True)
        expected = self.evaluation.select_model(frame, development_end=120)
        for alternative in [mutated, extended]:
            selected = self.evaluation.select_model(alternative, development_end=120)
            self.assertEqual(selected['selected_model'], expected['selected_model'])
            self.assertEqual(selected['scores'], expected['scores'])
            self.assertEqual(selected['origins'], expected['origins'])
            self.assertEqual(selected['development_end'], 120)

class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluation.backtest import metrics
        cls.metrics = staticmethod(metrics)

    def test_hand_calculated_errors_deadline_confusion_and_active_days(self):
        actual = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=4),
            'records': [100, 0, 300, 400],
            'runtime_minutes': [130, 100, 130, 100],
        })
        prediction = [
            {'date': str(day.date()), 'records': volume, 'runtime_minutes': runtime}
            for day, volume, runtime in zip(actual.date, [110, 10, 270, 420], [140, 90, 110, 125])
        ]
        result = self.metrics(actual, prediction)
        self.assertEqual(result['days'], 4)
        self.assertEqual(result['volume_mae'], 17.5)
        self.assertEqual(result['runtime_mae'], 16.25)
        self.assertEqual(result['runtime_rmse'], 17.5)
        self.assertEqual(result['endpoint_error'], 25)
        self.assertEqual(result['endpoint_abs_error'], 25)
        self.assertEqual(result['latest_start_mae'], 16.25)
        self.assertEqual(result['completion_error_mean'], 1.25)
        self.assertEqual(result['deadline'], {'tp': 1, 'tn': 1, 'fn': 1, 'fp': 1})
        self.assertEqual(result['active_deadline'], {'tp': 1, 'tn': 0, 'fn': 1, 'fp': 1})
        self.assertEqual(result['active_days'], 3)
        self.assertEqual(result['active_volume_mae'], 20)
        self.assertAlmostEqual(result['active_runtime_mae'], 55 / 3)

    def test_timestamps_use_seconds_ceil_while_runtime_error_uses_original_minutes(self):
        actual = pd.DataFrame({'date': pd.to_datetime(['2026-01-01']), 'records': [1], 'runtime_minutes': [1.001]})
        prediction = [{'date': '2026-01-01', 'records': 1, 'runtime_minutes': 1.009}]
        result = self.metrics(actual, prediction)
        self.assertAlmostEqual(result['runtime_mae'], .008)
        self.assertEqual(result['latest_start_mae'], 0)
        self.assertEqual(result['completion_error_mean'], 0)
        prediction[0]['runtime_minutes'] = 1.02
        result = self.metrics(actual, prediction)
        self.assertAlmostEqual(result['latest_start_mae'], 1 / 60)
        self.assertAlmostEqual(result['completion_error_mean'], 1 / 60)

    def test_misaligned_dates_lengths_and_empty_inputs_are_rejected(self):
        actual = history_frame(2)
        prediction = [{'date': str(day.date()), 'records': 10, 'runtime_minutes': 1} for day in actual.date]
        for changed in [[], prediction[:1], prediction[::-1], [dict(prediction[0], date='2026-02-01'), prediction[1]]]:
            with self.subTest(prediction=changed), self.assertRaises(ValueError):
                self.metrics(actual, changed)
        with self.assertRaises(ValueError):
            self.metrics(actual.iloc[:0], [])
        zero = actual.copy()
        zero['records'] = 0
        result = self.metrics(zero, prediction)
        self.assertIsNone(result['active_volume_mae'])
        self.assertIsNone(result['active_runtime_mae'])
        self.assertEqual(result['active_days'], 0)


class BacktestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluation import backtest
        cls.evaluation = backtest

    def test_selection_tolerance_and_simplicity_priority_use_only_seven_day_folds(self):
        frame = history_frame()
        scenarios = [
            ({'weekday4': 1.13, 'mean7': 1., 'ridge': .99}, 'weekday4'),
            ({'weekday4': 1.15, 'mean7': 1., 'ridge': .99}, 'mean7'),
            ({'weekday4': 10., 'mean7': 5., 'ridge': 1.}, 'ridge'),
            ({'weekday4': 0., 'mean7': 0., 'ridge': 0.}, 'weekday4'),
        ]
        for errors, expected_model in scenarios:
            calls = []

            def prediction(history, origin, horizon, model):
                calls.append((origin, horizon, model))
                actual = history.iloc[origin:origin + horizon]
                forecast = [
                    {'date': str(day.date()), 'records': volume, 'runtime_minutes': runtime + errors[model]}
                    for day, volume, runtime in actual.itertuples(index=False, name=None)
                ]
                return {'forecast': forecast, 'parameters': {}}

            with self.subTest(errors=errors), patch.object(self.evaluation, 'predict_at', side_effect=prediction):
                selected = self.evaluation.select_model(frame, development_end=120)
                self.assertEqual(selected['selected_model'], expected_model)
                self.assertEqual(len(calls), 15)
                self.assertTrue(all(horizon == 7 and origin + 7 <= 120 for origin, horizon, _ in calls))
                for model in models.MODELS:
                    self.assertAlmostEqual(selected['scores'][model], errors[model])

    def test_backtest_retains_all_candidate_folds_and_aligned_comparison(self):
        frame, _, _ = generate('weekday', seed=4)
        result = self.evaluation.backtest(frame)
        self.assertEqual(len(result['fold_results']), 30)
        self.assertEqual(len(result['summary']), 6)
        for row in result['fold_results']:
            self.assertIn(row['model'], models.MODELS)
            self.assertIn(row['horizon'], [7, 30])
            self.assertEqual(row['days'], row['horizon'])
            self.assertLessEqual(row['origin'] + row['horizon'], len(frame))
            self.assertGreaterEqual(row['fit_ms'], 0)
            self.assertGreaterEqual(row['predict_ms'], 0)
            self.assertEqual(sum(row['deadline'].values()), row['horizon'])
        selection = self.evaluation.select_model(frame)
        self.assertEqual(result['selected_model'], selection['selected_model'])
        expected = self.evaluation.predict_at(frame, 148, 30, selection['selected_model'])['forecast']
        rows = result['selected_comparison']
        self.assertEqual(len(rows), 30)
        for offset, row in enumerate(rows):
            actual = frame.iloc[148 + offset]
            self.assertEqual(row['date'], str(actual.date.date()))
            self.assertEqual(row['date'], expected[offset]['date'])
            self.assertEqual(row['actual_records'], actual.records)
            self.assertEqual(row['actual_runtime'], actual.runtime_minutes)
            self.assertEqual(row['predicted_records'], expected[offset]['records'])
            self.assertEqual(row['predicted_runtime'], expected[offset]['runtime_minutes'])

    def test_zero_baseline_error_yields_na_improvement(self):
        frame = history_frame()
        frame['records'] = 0
        frame['runtime_minutes'] = 0.
        result = self.evaluation.backtest(frame)
        self.assertEqual(result['selected_model'], 'weekday4')
        for row in result['summary']:
            self.assertEqual(row['runtime_mae'], 0)
            self.assertIsNone(row['improvement_pct'])
        with self.assertRaises(ValueError):
            self.evaluation.origins(149)


if __name__ == '__main__':
    unittest.main()
