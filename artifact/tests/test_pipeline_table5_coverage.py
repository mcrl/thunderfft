"""A fresh boundary recheck is sufficient only for the selected Table5 mode."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact import full_pipeline


class Table5PipelineCoverage(unittest.TestCase):
    def test_full_requires_full_profile_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'table5').mkdir()
            (output / 'table5/table.csv').write_text('synthetic output fixture\n')
            entries = [{'id': 'table5', 'required_outputs': {'paper': ['table.csv']}}]
            states = [{'evaluation': 'table5', 'status': 'complete', 'missing': [], 'range_mode': 'boundary'}]
            def check(mode):
                return full_pipeline.assembly_result(states, output, entries, 'paper', ROOT, table5_range_mode=mode)
            self.assertTrue(check('boundary')['outputs_complete'])
            self.assertEqual(len(check('boundary')['coverage_limits']), 1)
            self.assertFalse(check('full')['outputs_complete'])
            states[0]['range_mode'] = 'full'
            self.assertTrue(check('full')['outputs_complete'])
            self.assertEqual(check('full')['coverage_limits'], [])
            states[0].update(status='partial', missing=[{'item': 'range_fp16_cufftdx:N=64', 'reason': 'not measured'}])
            self.assertFalse(check('boundary')['outputs_complete'])


if __name__ == '__main__':
    unittest.main()
