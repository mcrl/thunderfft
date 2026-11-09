"""Real CPU subprocess checks of runner telemetry, deadlines and cancellation."""
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from artifact import progress, runner


class RunnerProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / 'artifact/probe'
        self.task.mkdir(parents=True)
        self.output = self.root / 'output'
        self.output.mkdir()
        self.batch = self.root / 'run'
        self.batch.mkdir()
        self.external = self.root / 'controller-progress.json'
        self.entry = {'id': 'probe', 'title': 'CPU fixture', 'implementation': 'ready',
                      'required_outputs': {'smoke': ['data.csv']}}
        self.args = SimpleNamespace(profile='smoke', build_dir=self.root / 'build', device=0,
            jobs=1, run_id='test', result_dir=self.root / 'result', timeout=5,
            cufftdx_mode='preset', cudnn_mode='fast', table5_range_mode='boundary',
            gpu='A100', progress='off')

    def script(self, body):
        source = ("import os, sys, time\nfrom pathlib import Path\n"
                  f"sys.path.insert(0, {str(Path(runner.__file__).resolve().parents[1])!r})\n"
                  "from artifact.progress import report\n" + body)
        (self.task / 'run.sh').write_text('"$THUNDERFFT_PYTHON" - <<\'PY\'\n' + source + '\nPY\n')

    def execute(self):
        with mock.patch.object(runner, 'ROOT', self.root):
            return runner.execute(self.entry, self.output, self.args)

    def test_adapter_snapshots_reach_controller_before_process_exit(self):
        self.script("report(0, 2, detail='first')\n"
                    "time.sleep(.05)\nreport(1, 2, detail='second')\n"
                    "time.sleep(1.25)\nreport(2, 2, detail='done')\n"
                    "Path(os.environ['THUNDERFFT_RESULT_DIR'], 'data.csv').write_text('fresh,1\\n')\n"
                    "print('fixture detailed output')\n")
        snapshots = []
        with mock.patch.dict(os.environ, {'THUNDERFFT_RUN_PROGRESS_FILE': str(self.external)}):
            with runner.RunProgress([self.entry], self.args, self.batch) as display:
                def sample(path):
                    display.poll(path)
                    snapshots.append(progress.read_snapshot(self.external))
                self.args._progress_callback = sample
                display.begin(self.entry)
                state = self.execute()
                display.end(state)
                display.finish(0)
        self.assertEqual(state['status'], 'completed')
        self.assertTrue(any(s['completed'] == 0 and (s['detail'] or {}).get('done') == 1 for s in snapshots))
        final = progress.read_snapshot(self.external)
        self.assertEqual((final['completed'], final['total'], final['status']), (1, 1, 'completed'))
        self.assertIsNone(final['detail'])  # A completed task must not be counted twice in the bar.
        self.assertEqual(progress.read_snapshot(self.batch / 'progress.json'), final)
        self.assertIn('fixture detailed output', (self.output / 'run.log').read_text())

    def test_optional_callback_failure_cannot_fail_or_orphan_adapter(self):
        self.script("time.sleep(.05)\nPath(os.environ['THUNDERFFT_RESULT_DIR'], 'data.csv').write_text('ok')")
        self.args._progress_callback = mock.Mock(side_effect=OSError('display unavailable'))
        state = self.execute()
        self.assertEqual(state['status'], 'completed')
        self.args._progress_callback.assert_called_once()

    def test_progress_polling_preserves_task_timeout(self):
        self.script("report(0, 2)\ntime.sleep(20)")
        self.args.timeout = 1
        self.args._progress_callback = lambda path: progress.read_snapshot(path / 'progress.json')
        state = self.execute()
        self.assertEqual((state['status'], state['exit_code']), ('failed', 124))
        self.assertLess(state['elapsed_seconds'], 5)
        self.assertEqual(progress.read_snapshot(self.output / 'progress.json')['done'], 0)

    def test_slow_optional_callback_does_not_reclassify_finished_adapter_as_timeout(self):
        self.script("Path(os.environ['THUNDERFFT_RESULT_DIR'], 'data.csv').write_text('ok')")
        self.args.timeout = 1
        self.args._progress_callback = lambda path: time.sleep(1.1)
        self.assertEqual(self.execute()['status'], 'completed')

    def test_interrupt_during_progress_stops_adapter_without_completion(self):
        self.script('time.sleep(20)')
        self.args._progress_callback = mock.Mock(side_effect=KeyboardInterrupt)
        state = self.execute()
        self.assertEqual((state['status'], state['exit_code']), ('interrupted', 130))
        self.assertFalse((self.output / 'data.csv').exists())


if __name__ == '__main__':
    unittest.main()
