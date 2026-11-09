#!/usr/bin/env python3
"""CPU contracts for fresh local boundary checks and exhaustive fallback."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('table5_measure_tests_target', Path(__file__).with_name('measure.py'))
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


class BoundaryMeasurement(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = {'profile': 'paper', 'N': [64], 'batch': 65536, 'seed': 1234,
                         'range_mode': 'boundary', 'selected_phases': ['range_fp16'],
                         'range_exponents': {p: [0, 127] for p in measure.RANGES.values()}}
        self.thresholds = {'fp16': 11, 'tf32': 123, 'fp32_cufftdx': 123, 'fp16_cufftdx': 11}
        self.commands = []

    def native(self, command, **kwargs):
        self.commands.append(command)
        values = dict(zip(command[1::2], command[2::2])) if '--alpha-sweep' not in command else None
        def value(flag):
            return command[command.index(flag) + 1]
        precision = value('--precision')
        precision = {v: k for k, v in measure.NATIVE_PRECISION.items()}.get(precision, precision)
        precisions = ['fp16', 'tf32'] if precision == 'both' else [precision]
        scan = '--alpha-sweep' in command
        exponents = range(int(value('--alpha-exp-min')), int(value('--alpha-exp-max')) + 1) if scan else [0]
        rows = []
        for n in map(int, value('--N-list').split(',')):
            for p in precisions:
                for exponent in exponents:
                    rows.append(dict(zip(measure.FIELDS, [n, value('--batch'), 'gaussian', 2.0**exponent,
                        p, 'nan' if scan else '0.0001', 'nan', 'nan', int(not scan or exponent <= self.thresholds[p])])))
        measure.write_rows(Path(value('--csv')), rows)
        Path(value('--environment-json')).write_text(json.dumps({'gpu_uuid': 'test-gpu'}))
        kwargs['stdout'].write('synthetic native output\n')
        return SimpleNamespace(returncode=0)

    def collect(self, native=None):
        (self.root / 'metadata.json').write_text(json.dumps(self.metadata))
        with patch.object(measure.subprocess, 'run', side_effect=native or self.native):
            return measure.collect(self.root, self.root / 'ordinary', self.root / 'cufftdx', 10)

    def test_known_boundary_measures_four_fresh_exponents(self):
        result = self.collect()
        check = result['range_checks']['fp16']['64']
        self.assertEqual(check['window'], [10, 13])
        self.assertEqual(len(check['attempts']), 1)
        self.assertTrue(check['boundary_matches_preset'])
        self.assertEqual(len(measure.read_rows(self.root / 'range_fp16.csv')), 4)
        self.assertFalse(result['full_range_scan'])

    def test_shifted_boundary_expands_and_reports_observed_value(self):
        self.thresholds['fp16'] = 12
        result = self.collect()
        check = result['range_checks']['fp16']['64']
        self.assertEqual(check['window'], [6, 17])
        self.assertEqual(check['observed_last_finite'], 12)
        self.assertFalse(check['boundary_matches_preset'])
        self.assertEqual(len(check['attempts']), 2)
        self.assertEqual(len(measure.read_rows(self.root / 'range_fp16.csv')), 12)

    def test_missing_bracket_expands_until_found(self):
        self.thresholds['fp16'] = 24
        result = self.collect()
        check = result['range_checks']['fp16']['64']
        self.assertEqual(check['observed_last_finite'], 24)
        self.assertEqual(check['window'], [0, 29])
        self.assertEqual(len(check['attempts']), 3)

    def test_full_keeps_all_exponents_and_four_precisions(self):
        self.metadata.update(range_mode='full', selected_phases=list(measure.ACCURACY) + list(measure.RANGES))
        result = self.collect()
        self.assertTrue(result['full_range_scan'])
        self.assertEqual(result['range_checks'], {})
        for phase in measure.RANGES:
            self.assertEqual(len(measure.read_rows(self.root / (phase + '.csv'))), 128)
        dx = [c for c in self.commands if c[c.index('--precision') + 1].startswith('cufftdx_')]
        self.assertEqual(len(dx), 4)
        self.assertEqual({c[c.index('--precision') + 1] for c in dx}, {'cufftdx_fp16', 'cufftdx_fp32'})
        self.assertTrue(all(Path(c[0]).name == 'cufftdx' for c in dx))
        self.assertTrue(all(Path(c[0]).name == 'ordinary' for c in self.commands if c not in dx))

    def test_cufftdx_fp32_starts_from_tf32_hint_and_measures_fresh(self):
        self.metadata['selected_phases'] = ['range_fp32_cufftdx']
        self.thresholds['fp32_cufftdx'] = 122
        result = self.collect()
        check = result['range_checks']['fp32_cufftdx']['64']
        self.assertEqual(check['initial_window'], [122, 125])
        self.assertIsNone(check['expected_last_finite'])
        self.assertIn('tf32', check['preset_source'])
        self.assertEqual(check['observed_last_finite'], 122)
        self.assertTrue(all(Path(c[0]).name == 'cufftdx' and 'cufftdx_fp32' in c for c in self.commands))

    def test_invalid_mode_runs_nothing(self):
        self.metadata['range_mode'] = 'typo'
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(self.commands, [])

    def test_native_failure_is_not_a_policy_skip(self):
        with self.assertRaisesRegex(RuntimeError, 'native exit 7'):
            self.collect(lambda *a, **kw: SimpleNamespace(returncode=7))

    def test_incomplete_native_window_is_rejected(self):
        def incomplete(command, **kwargs):
            result = self.native(command, **kwargs)
            path = Path(command[command.index('--csv') + 1])
            measure.write_rows(path, measure.read_rows(path)[:-1])
            return result
        with self.assertRaisesRegex(ValueError, 'incomplete exponent'):
            self.collect(incomplete)

    def test_no_boundary_in_profile_fails(self):
        self.thresholds['fp16'] = 127
        with self.assertRaisesRegex(ValueError, 'no finite/nonfinite pair'):
            self.collect()

    def test_nonmonotonic_native_window_is_rejected(self):
        rows = [dict(zip(measure.FIELDS, [64, 65536, 'gaussian', 2**k, 'fp16', 'nan', 'nan', 'nan', ok]))
                for k, ok in [(10, '1'), (11, '0'), (12, '1'), (13, '0')]]
        with self.assertRaisesRegex(ValueError, 'nonmonotonic'):
            measure.boundary(rows, 64, 'fp16', [10, 13], 65536)


if __name__ == '__main__':
    unittest.main()
