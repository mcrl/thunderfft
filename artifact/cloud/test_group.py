"""Offline lifecycle checks for grouped rentals; no credentials, GPU or API."""
import argparse
from contextlib import ExitStack, redirect_stdout
import copy
import fcntl
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from artifact import collection
from artifact.cloud import config, coordinator, group, provider


class FakeClient:
    def __init__(self):
        self.rows = []
        self.events = []
        self.mutex = threading.Lock()
        self.balance = 10
        self.fail_create = None
        self.ambiguous_create = None
        self.create_count = 0

    def instances(self):
        with self.mutex:
            self.events.append(('instances', None))
            return copy.deepcopy(self.rows)

    def account_summary(self):
        return {'balance_usd': self.balance, 'observed_at': time.time()}

    def billing_preflight(self):
        return {'billing_read_verified': True}

    def search_offers(self, gpu, disk):
        self.events.append(('plan', gpu))
        return [{'id': {'A100': 10, 'H100': 20, 'B200': 30}[gpu], 'machine_id': 100,
                 'num_gpus': 1, 'gpu_name': gpu, 'gpu_ram': 81920,
                 'dph_total': .1, 'storage_cost': .01, 'inet_down_cost': .001, 'inet_up_cost': .001}]

    def create_instance(self, offer_id, **kwargs):
        self.create_count += 1
        self.events.append(('create', offer_id))
        if self.fail_create == offer_id:
            raise provider.CreateRejected('Offer became unavailable')
        row = {'id': offer_id + 1000, 'label': kwargs['label']}
        self.rows.append(row)
        if self.ambiguous_create == offer_id:
            raise provider.CreateOutcomeUnknown('Creation acknowledgement lost')
        return row['id']

    def attach_ssh_key(self, instance_id, public):
        self.events.append(('attach', instance_id))

    def destroy_instance(self, instance_id):
        with self.mutex:
            self.events.append(('destroy', instance_id))
            self.rows = [row for row in self.rows if row['id'] != instance_id]

    def billing_evidence(self, start, end, instance_id=None):
        return {'observed_charge_usd': .01, 'settlement_verified': False}


class GroupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.keys = self.root / 'keys'
        self.keys.mkdir(mode=0o700)
        self.key = self.keys / 'private'
        self.key.write_text('fake noncredential')
        self.key.chmod(0o600)
        public = self.keys / 'public'
        public.write_text('ssh-ed25519 QUJD test\n')
        base = copy.deepcopy(config.DEFAULTS)
        base.update(api_key_file=str(self.keys / 'api'), ssh_key_file=str(self.key),
                    ssh_public_key_file=str(public), budget_usd=2, startup_seconds=100,
                    execution_seconds=100, cleanup_seconds=1)
        self.configs = {gpu: copy.deepcopy(base) for gpu in ['H100', 'B200']}
        self.client = FakeClient()
        self.args = SimpleNamespace(result_dir=self.root / 'result', collection='group-test',
            profile='smoke', cloud_allow_dirty=False, jobs=2, timeout=0,
            cufftdx_mode='preset', cudnn_mode='fast', table5_range_mode='boundary')
        self.registry = {'evaluations': [{'id': 'table3'}, {'id': 'table4'}]}
        self.source_count = 0
        self.workers = []
        self.prepared = set()
        self.prep_failure = None
        self.worker_failure = None
        self.barrier = None
        self.local_called = False
        self.updates = []
        self.controller_thread = threading.get_ident()

    def source(self, root, dest, **kwargs):
        self.source_count += 1
        dest.write_bytes(b'fake safe bundle')
        return {'archive_sha256': 'fake', 'files': {}}

    def watchdog(self, folder, lock):
        state = json.loads((folder / 'state.json').read_text())
        self.assertFalse(state['create_attempted'])
        self.client.events.append(('armed', state['label']))
        # A separate file description cannot enter while the group owns it.
        with (self.keys / '.thunderfft-execution.lock').open('a') as other:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return mock.Mock(wait=mock.Mock(return_value=0))

    def wait_ssh(self, client, state, settings, folder, root, deadline, cancel_event=None):
        self.assertEqual(len([e for e in self.client.events if e[0] == 'create']), len(self.configs))
        gpu = {1020: 'H100', 1030: 'B200', 1010: 'A100'}[state['instance_id']]
        ssh = SimpleNamespace(gpu=gpu, cancel_event=cancel_event)
        ssh.run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, 'armed\n', ''))
        ssh.upload = mock.Mock()
        self.client.events.append(('ssh', gpu))
        return ssh

    def bootstrap(self, ssh, remote, deadline, **kwargs):
        if ssh.gpu == self.prep_failure:
            raise TimeoutError('offline preparation failure')
        self.prepared.add(ssh.gpu)
        self.client.events.append(('prepared', ssh.gpu))

    def worker(self, ssh, remote, settings, cancel_event=None, progress_callback=None):
        self.assertEqual(self.prepared, set(self.configs))
        self.workers.append(ssh.gpu)
        self.client.events.append(('worker', ssh.gpu))
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        if self.worker_failure == ssh.gpu:
            return 1
        return 0

    def local(self, cancel):
        self.assertEqual(self.prepared, set(self.configs))
        self.local_called = True
        self.client.events.append(('worker', 'local-A100'))
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        return 0

    def stage_update(self, gpu, patch):
        self.assertEqual(threading.get_ident(), self.controller_thread)
        self.updates.append((gpu, patch))

    def selected(self, requested, entries, parser):
        return entries

    def patches(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(group, 'VastClient', return_value=self.client))
        stack.enter_context(mock.patch.object(group, 'build_source_bundle', side_effect=self.source))
        stack.enter_context(mock.patch.object(group.subprocess, 'run', return_value=
            subprocess.CompletedProcess([], 0, 'ssh-ed25519 QUJD\n', '')))
        stack.enter_context(mock.patch.object(coordinator, 'arm_watchdog', side_effect=self.watchdog))
        stack.enter_context(mock.patch.object(coordinator, 'wait_ssh', side_effect=self.wait_ssh))
        stack.enter_context(mock.patch.object(coordinator, 'bootstrap', side_effect=self.bootstrap))
        stack.enter_context(mock.patch.object(coordinator, 'execute_worker', side_effect=self.worker))
        stack.enter_context(mock.patch.object(coordinator, 'stop_worker'))
        stack.enter_context(mock.patch.object(coordinator, 'fetch_results', return_value=self.root / 'fake-result'))
        stack.enter_context(mock.patch.object(coordinator, 'register_results', return_value=[]))
        # Billing has a documented three-second second sample. Keep real
        # lifecycle cleanup and cost computation, but bypass that wait only.
        original_cleanup = coordinator.cleanup_and_bill
        def cleanup(*args, **kwargs):
            with mock.patch.object(coordinator.time, 'sleep', return_value=None):
                return original_cleanup(*args, **kwargs)
        stack.enter_context(mock.patch.object(coordinator, 'cleanup_and_bill', side_effect=cleanup))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def run_group(self, *, local=True, ready_check=None, log=None, tick=None):
        return group.run(self.args, self.registry, self.repo, collection, self.selected,
                         argparse.ArgumentParser(), self.configs, [],
                         local_run=self.local if local else None, stage_update=self.stage_update,
                         ready_check=ready_check, log=log, tick=tick)

    def test_progress_and_logs_are_delivered_only_on_controller_thread(self):
        received, ticks = [], []
        snapshot = {'schema_version': 1, 'completed': 0, 'total': 2,
                    'evaluation': 'table3', 'status': 'running', 'updated_at': 123.0,
                    'detail': {'schema_version': 1, 'done': 36, 'total': 50,
                               'unit': 'cases', 'detail': 'cuFFTDx', 'status': 'running',
                               'updated_at': 123.0}}
        def worker(ssh, remote, settings, cancel_event=None, progress_callback=None):
            self.assertNotEqual(threading.get_ident(), self.controller_thread)
            progress_callback(snapshot)
            return self.worker(ssh, remote, settings, cancel_event=cancel_event)
        def log(message):
            self.assertEqual(threading.get_ident(), self.controller_thread)
            received.append(message)
        def tick():
            self.assertEqual(threading.get_ident(), self.controller_thread)
            ticks.append(time.monotonic())
        with self.patches(), mock.patch.object(coordinator, 'execute_worker', side_effect=worker):
            result = self.run_group(log=log, tick=tick)
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual({gpu for gpu, patch in self.updates if patch.get('progress') == snapshot},
                         {'H100', 'B200'})
        self.assertTrue(any('running cloud measurements' in line for line in received))
        self.assertTrue(ticks)
        self.assertTrue(all(second - first >= .99 for first, second in zip(ticks, ticks[1:])))
        self.assertTrue(result['cleanup_confirmed'])

    def test_optional_display_failure_cannot_skip_measurement_or_cleanup(self):
        with self.patches():
            result = self.run_group(log=mock.Mock(side_effect=OSError('terminal closed')),
                                    tick=mock.Mock(side_effect=ValueError('invalid display')))
        self.assertEqual(result['exit_code'], 0)
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(self.client.rows, [])

    def test_all_acquired_and_ready_before_three_measurements_overlap(self):
        self.barrier = threading.Barrier(3)
        with self.patches():
            result = self.run_group()
        self.assertEqual(result['exit_code'], 0)
        self.assertTrue(result['all_ready_gate_opened'])
        self.assertEqual(set(self.workers), set(self.configs))
        self.assertTrue(self.local_called)
        self.assertEqual(self.source_count, 1)
        self.assertTrue(result['cleanup_confirmed'])
        self.assertFalse(result['billing_settlement_verified'])
        self.assertEqual(self.client.rows, [])
        self.assertEqual(result['startup_retries'], 0)
        self.assertEqual(self.client.create_count, 2)
        self.assertTrue(all(stage['status'] == 'completed' for stage in result['stages']))
        self.assertAlmostEqual(result['reserved_total_usd'], sum(
            stage.get('cloud', {}).get('reserved_total_usd', 0) for stage in result['stages']))
        events = [name for name, _ in self.client.events]
        self.assertLess(max(i for i, name in enumerate(events) if name == 'create'), events.index('ssh'))
        self.assertLess(max(i for i, name in enumerate(events) if name == 'prepared'), events.index('worker'))
        self.assertEqual(json.loads((Path(result['audit_folder']) / 'execution.json').read_text()), result)

    def test_later_create_failure_cleans_first_and_starts_no_measurement(self):
        self.client.fail_create = 30
        with self.patches():
            result = self.run_group()
        self.assertNotEqual(result['exit_code'], 0)
        self.assertFalse(result['all_ready_gate_opened'])
        self.assertEqual(self.workers, [])
        self.assertFalse(self.local_called)
        self.assertEqual(self.prepared, set())
        self.assertEqual(self.client.rows, [])
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(self.client.create_count, 2)
        rejected = next(stage for stage in result['stages'] if stage['gpu'] == 'B200')
        self.assertEqual(rejected['cloud']['reserved_total_usd'], 0)

    def test_preparation_failure_cleans_all_and_starts_no_measurement(self):
        self.prep_failure = 'B200'
        with self.patches():
            result = self.run_group()
        self.assertNotEqual(result['exit_code'], 0)
        self.assertEqual(self.workers, [])
        self.assertFalse(self.local_called)
        self.assertEqual(self.client.rows, [])
        self.assertEqual(len([e for e in self.client.events if e[0] == 'destroy']), 2)

    def test_ready_check_failure_prevents_every_worker(self):
        with self.patches():
            result = self.run_group(ready_check=mock.Mock(side_effect=RuntimeError('local A100 became busy')))
        self.assertNotEqual(result['exit_code'], 0)
        self.assertFalse(result['all_ready_gate_opened'])
        self.assertEqual(self.workers, [])
        self.assertFalse(self.local_called)
        self.assertEqual(self.client.rows, [])

    def test_interrupt_at_gate_cleans_all_and_never_starts(self):
        with self.patches():
            result = self.run_group(ready_check=mock.Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(result['exit_code'], 130)
        self.assertEqual(self.workers, [])
        self.assertFalse(self.local_called)
        self.assertEqual(self.client.rows, [])
        self.assertTrue(result['cleanup_confirmed'])

    def test_combined_balance_failure_happens_before_any_creation(self):
        self.client.balance = .4  # Fits either cheap single plan, but not both.
        with self.patches(), self.assertRaisesRegex(group.PreflightUnavailable, 'shared budget or account balance') as caught:
            self.run_group()
        self.assertIsNone(caught.exception.gpu)
        self.assertEqual(self.client.create_count, 0)
        self.assertEqual(self.workers, [])

    def test_fresh_offer_failure_keeps_gpu_and_detail_without_creating_resources(self):
        original_plan = coordinator.plan
        detail = 'No matching B200 offer fits the all-in budget and available balance'
        def plan(client, settings, gpu):
            if gpu == 'B200':
                raise provider.ProviderError(detail)
            return original_plan(client, settings, gpu)
        with self.patches(), mock.patch.object(coordinator, 'plan', side_effect=plan), \
                self.assertRaises(group.PreflightUnavailable) as caught:
            self.run_group()
        self.assertEqual(caught.exception.gpu, 'B200')
        self.assertEqual(caught.exception.detail, detail)
        self.assertEqual(str(caught.exception), detail)
        self.assertEqual(self.client.create_count, 0)
        self.assertEqual(self.workers, [])
        self.assertEqual(self.source_count, 1)

    def test_held_account_lock_blocks_group_without_provider_access(self):
        with (self.keys / '.thunderfft-execution.lock').open('a') as locked:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.patches(), mock.patch.object(group, 'VastClient', side_effect=AssertionError('provider access')), \
                    self.assertRaisesRegex(ValueError, 'Another cloud execution'):
                self.run_group()

    def test_existing_artifact_rental_is_checked_once_and_blocks_creation(self):
        self.client.rows = [{'id': 999, 'label': 'thunderfft-ae-existing'}]
        with self.patches(), self.assertRaisesRegex(provider.ProviderError, 'existing ThunderFFT rental'):
            self.run_group()
        self.assertEqual(self.client.events, [('instances', None)])
        self.assertEqual(self.client.create_count, 0)

    def test_worker_failure_cancels_peer_and_local_before_cleanup(self):
        self.barrier = threading.Barrier(3)
        def worker(ssh, remote, settings, cancel_event=None, progress_callback=None):
            self.barrier.wait(timeout=3)
            if ssh.gpu == 'H100':
                return 1
            self.assertTrue(cancel_event.wait(3))
            raise InterruptedError('cancelled peer')
        def local(cancel):
            self.barrier.wait(timeout=3)
            self.assertTrue(cancel.wait(3))
            return 130
        self.local = local
        with self.patches(), mock.patch.object(coordinator, 'execute_worker', side_effect=worker):
            result = self.run_group()
        self.assertEqual(result['exit_code'], 1)
        self.assertEqual(self.client.rows, [])
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual({stage['gpu']: stage['exit_code'] for stage in result['stages']},
                         {'A100': 130, 'H100': 1, 'B200': 130})

    def test_three_cloud_gpus_share_acquisition_gate_without_local_callback(self):
        self.configs['A100'] = copy.deepcopy(self.configs['H100'])
        self.barrier = threading.Barrier(3)
        with self.patches():
            result = self.run_group(local=False)
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(set(self.workers), {'A100', 'H100', 'B200'})
        self.assertEqual(self.client.create_count, 3)
        self.assertEqual(self.client.rows, [])
        self.assertFalse(self.local_called)

    def test_gate_deadline_does_not_extend_paid_startup_allowance(self):
        def expired_deadline(ssh, remote, deadline, **kwargs):
            self.prepared.add(ssh.gpu)
            # A prepared environment may reach the common deadline just before
            # dispatch; the all-ready gate must check the remaining allowance.
            coordinator.remaining(deadline - 200)
        with self.patches(), mock.patch.object(coordinator, 'bootstrap', side_effect=expired_deadline):
            result = self.run_group()
        self.assertNotEqual(result['exit_code'], 0)
        self.assertFalse(result['all_ready_gate_opened'])
        self.assertEqual(self.workers, [])
        self.assertEqual(self.client.rows, [])

    def test_cost_audit_failure_reserves_allocations_and_still_cleans_every_instance(self):
        with self.patches(), mock.patch.object(coordinator, 'attempt_cost', side_effect=OSError('missing evidence')):
            result = self.run_group()
        self.assertEqual(result['exit_code'], 1)
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(result['reserved_total_usd'], 4)
        self.assertEqual(self.client.rows, [])
        for stage in result['stages']:
            if stage['backend'] == 'vast':
                self.assertIn('audit_errors', stage['cloud'])

    def test_listing_calls_are_serialized_across_group_threads(self):
        entered, release = threading.Event(), threading.Event()
        active = 0
        maximum = 0
        calls = 0
        mutex = threading.Lock()
        def listing():
            nonlocal active, maximum, calls
            with mutex:
                active += 1
                maximum = max(maximum, active)
                calls += 1
            entered.set()
            release.wait(2)
            with mutex:
                active -= 1
            return []
        client = group._GroupClient(SimpleNamespace(instances=listing))
        threads = [threading.Thread(target=client.instances) for _ in range(2)]
        threads[0].start()
        self.assertTrue(entered.wait(2))
        threads[1].start()
        release.set()
        for thread in threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertEqual((calls, maximum), (2, 1))

    def test_ambiguous_later_creation_is_reconciled_without_retry(self):
        self.client.ambiguous_create = 30
        with self.patches():
            result = self.run_group()
        self.assertEqual(result['exit_code'], 1)
        self.assertFalse(result['all_ready_gate_opened'])
        self.assertEqual(self.client.rows, [])
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(self.client.create_count, 2)
        stage = next(stage for stage in result['stages'] if stage['gpu'] == 'B200')
        self.assertEqual(stage['cloud']['attempts'][0]['instance_id'], 1030)
        self.assertGreater(stage['cloud']['reserved_total_usd'], 0)

    def test_interrupt_while_workers_active_cancels_all_and_preserves_costs(self):
        def worker(ssh, remote, settings, cancel_event=None, progress_callback=None):
            self.assertTrue(cancel_event.wait(3))
            raise InterruptedError('cancelled worker')
        def local(cancel):
            self.assertTrue(cancel.wait(3))
            return 130
        interrupted = False
        def update(gpu, patch):
            nonlocal interrupted
            if patch.get('status') == 'running' and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
        self.local, self.stage_update = local, update
        with self.patches(), mock.patch.object(coordinator, 'execute_worker', side_effect=worker):
            result = self.run_group()
        self.assertTrue(interrupted)
        self.assertEqual(result['exit_code'], 130)
        self.assertEqual(self.client.rows, [])
        self.assertTrue(result['cleanup_confirmed'])
        self.assertGreater(result['reserved_total_usd'], 0)

    def test_failed_stage_audit_callback_cannot_skip_cleanup_or_accounting(self):
        def update(gpu, patch):
            if patch.get('status') == 'preparing':
                raise OSError('disk full')
        self.stage_update = update
        with self.patches():
            result = self.run_group()
        self.assertEqual(result['exit_code'], 1)
        self.assertEqual(self.workers, [])
        self.assertEqual(self.client.rows, [])
        self.assertTrue(result['cleanup_confirmed'])
        self.assertGreater(result['reserved_total_usd'], 0)
        self.assertIn('callback failed', result['error']['error'])

    def test_terminal_group_audit_failure_returns_cleanup_and_cost_summary(self):
        original_write = group.write_json
        def write(path, value):
            if Path(path).name == 'execution.json' and Path(path).parent.name.startswith('group-'):
                raise OSError('disk full')
            original_write(path, value)
        with self.patches(), mock.patch.object(group, 'write_json', side_effect=write):
            result = self.run_group()
        self.assertEqual(result['exit_code'], 1)
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(self.client.rows, [])
        self.assertGreater(result['reserved_total_usd'], 0)
        self.assertEqual(result['audit_error'], 'OSError')

    def test_interrupt_after_measurement_enqueue_joins_owner_before_fallback_cleanup(self):
        original_submit = group.ThreadPoolExecutor.submit
        interrupted = False
        def submit(pool, function, *args, **kwargs):
            nonlocal interrupted
            future = original_submit(pool, function, *args, **kwargs)
            if getattr(function, '__name__', '') == 'measure' and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return future
        def worker(ssh, remote, settings, cancel_event=None, progress_callback=None):
            self.assertTrue(cancel_event.wait(3))
            raise InterruptedError('worker cancelled')
        finish_original = group._finish
        finished = []
        def finish(member, *args, **kwargs):
            finished.append(member['gpu'])
            return finish_original(member, *args, **kwargs)
        with self.patches(), mock.patch.object(group.ThreadPoolExecutor, 'submit', new=submit), \
                mock.patch.object(coordinator, 'execute_worker', side_effect=worker), \
                mock.patch.object(group, '_finish', side_effect=finish):
            result = self.run_group()
        self.assertTrue(interrupted)
        self.assertEqual(result['exit_code'], 130)
        self.assertEqual(sorted(finished), ['B200', 'H100'])
        self.assertEqual(self.client.rows, [])
        self.assertTrue(result['cleanup_confirmed'])

    def test_preparation_reports_original_failed_gpu_instead_of_cancelled_peer(self):
        def prepare(ssh, remote, deadline, **kwargs):
            if ssh.gpu == 'B200':
                raise TimeoutError('B200 preparation failed')
            self.assertTrue(ssh.cancel_event.wait(3))
            raise InterruptedError('cancelled H100 preparation')
        with self.patches(), mock.patch.object(coordinator, 'bootstrap', side_effect=prepare):
            result = self.run_group()
        self.assertEqual(result['exit_code'], 1)
        self.assertEqual(result['error']['gpu'], 'B200')
        self.assertEqual(result['error']['error_type'], 'TimeoutError')
        self.assertEqual(self.client.rows, [])


if __name__ == '__main__':
    unittest.main()
