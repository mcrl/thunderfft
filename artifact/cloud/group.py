"""One budgeted cloud group: acquire everything, prepare everything, then measure.

The account execution lock is shared with standalone runs and inherited by every
independent watchdog. Group failures never start a fresh rental/retry budget.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import uuid

from . import coordinator as cloud
from .config import ARCHS, required_dependencies
from .provider import ProviderError, VastClient
from .transport import build_source_bundle, validate_private_key
from .watchdog import write_json


class PreflightUnavailable(ProviderError):
    """Fresh group planning failed before any billable resource was created."""
    def __init__(self, gpu, detail):
        self.gpu = gpu
        self.detail = detail
        super().__init__(detail)


class _GroupClient:
    """Serialize the provider's shared pagination/rate-limit clock per group."""
    def __init__(self, client):
        self._client = client
        self._instances_lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._client, name)

    def instances(self):
        with self._instances_lock:
            return self._client.instances()


def _cancelled(cancel):
    if cancel.is_set():
        raise InterruptedError("Cloud group cancelled")


def _prepare(member, root, deadline, cancel):
    _cancelled(cancel)
    config, folder = member['config'], member['folder']
    ssh = cloud.wait_ssh(member['client'], member['state'], config, folder, root, deadline,
                         cancel_event=cancel)
    member['ssh'] = ssh
    ssh.cancel_event = cancel
    remote = member['remote']
    ssh.run(['mkdir', '-p', remote], timeout=cloud.remaining(deadline))
    ssh.run(['bash', '-c', 'command -v python3 >/dev/null || (apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends python3)'],
            timeout=cloud.remaining(deadline, 300))
    ssh.upload(folder / 'source.tar.gz', remote + '/source.tar.gz', timeout=cloud.remaining(deadline, 180))
    ssh.upload(folder / 'request.json', remote + '/request.json', timeout=cloud.remaining(deadline, 30))
    cloud.bootstrap(ssh, remote, deadline, **member['dependencies'])
    status = ssh.run(['cat', '/tmp/thunderfft-watchdog/status'],
                     timeout=cloud.remaining(deadline), check=False)
    write_json(folder / 'remote-watchdog.json', {
        'status': status.stdout.strip(), 'credential_source': 'provider container-scoped credential'})
    _cancelled(cancel)


def _finish(member, collection_module, code, *, collect):
    """Exactly one owner cleans each member; other members clean concurrently."""
    state, folder, config = member['state'], member['folder'], member['config']
    ssh, fetched = member.get('ssh'), None
    audit_errors = []

    def record(name, value):
        try:
            write_json(folder / name, value)
        except Exception as error:
            audit_errors.append(type(error).__name__)

    if ssh is not None:
        # Cancellation stops active work; bounded terminal collection/cleanup
        # still needs a usable transport after the shared event was set.
        ssh.cancel_event = None
        if collect:
            try:
                cloud.stop_worker(ssh, member['remote'])
                fetched = cloud.fetch_results(ssh, member['remote'], folder, config, state['deadline'])
            except Exception as error:
                record('download-error.json', {'error_type': type(error).__name__, 'error': str(error)[:1000]})
                code = code or 1
    try:
        clean = cloud.cleanup_and_bill(member['client'], state, folder, config['cleanup_seconds'])
    except Exception as error:
        clean = False
        state['cleanup_confirmed'] = False
        record('cleanup-error.json', {'error_type': type(error).__name__,
               'error': str(error)[:1000], 'observed_at': time.time()})
    state['finished'] = True
    record('state.json', state)
    if clean or not state['create_attempted']:
        watchdog = member.get('watchdog')
        if watchdog is not None:
            try:
                watchdog.wait(timeout=5)
            except Exception:
                pass
    else:
        code = code or 1
        message = (f"[{member['gpu']}] cleanup unresolved; watchdog remains active. "
                   f"Use --cloud-recover {folder}")
        if member.get('log') is not None:
            member['log'](message)
        else:
            print(message, flush=True)
    if fetched is not None:
        try:
            imported = cloud.register_results(member['args'], member['jobs'], fetched, collection_module)
            record('imported.json', imported)
        except Exception as error:
            record('import-error.json', {'error_type': type(error).__name__, 'error': str(error)[:1000]})
            code = code or 1
    try:
        cost = cloud.attempt_cost(member['proposal'], state, folder)
    except Exception as error:
        audit_errors.append(type(error).__name__)
        cost = {'reserved_total_usd': config['budget_usd'], 'observed_charge_usd': None,
                'billing_settlement_verified': False, 'basis': 'cost evidence unavailable; full allocation reserved'}
    record('cost.json', cost)
    if audit_errors:
        code = code or 1
    outcome = {'folder': str(folder), 'instance_id': state['instance_id'],
               'machine_id': member['proposal']['offer']['machine_id'], 'exit_code': code,
               'cleanup_confirmed': bool(clean), 'startup_retryable': False,
               'results_retrieved': fetched is not None, **cost}
    evidence = {'gpu': member['gpu'], 'budget_usd': config['budget_usd'], 'attempts': [outcome],
                'reserved_total_usd': cost['reserved_total_usd'],
                'remaining_budget_usd': max(0, config['budget_usd'] - cost['reserved_total_usd']),
                'cleanup_confirmed': bool(clean), 'billing_settlement_verified': False,
                'exit_code': code, 'audit_folder': str(folder)}
    record('execution.json', evidence)
    record('status.json', outcome)
    if audit_errors:
        evidence['audit_errors'] = audit_errors
        evidence['exit_code'] = evidence['exit_code'] or 1
    member['finished'] = True
    member['evidence'] = evidence
    return evidence


def run(args, registry, root, collection_module, select_tasks, parser, configs, proposals,
        *, local_run=None, stage_update=None, ready_check=None, log=None, tick=None):
    """Run one fixed-budget group; callbacks never receive provider credentials.

    ``configs`` maps cloud GPU names to validated configurations with disjoint
    budget allocations. Their sum is this group's ceiling. ``local_run(event)``
    cooperatively stops its subprocess on cancellation. ``ready_check()`` runs
    after every remote environment is ready and before any benchmark starts.
    ``stage_update(gpu, patch)``, ``log(message)`` and ``tick()`` are called only
    on the controller thread. ``tick`` refreshes local progress at most once
    per second while remote preparation or measurement owners are pending.
    """
    if not configs or set(configs) - set(ARCHS):
        raise ValueError('A cloud group requires supported GPU configurations')
    if local_run is not None and 'A100' in configs:
        raise ValueError('A100 cannot run both locally and in the cloud')
    configs = copy.deepcopy(configs)
    order = [gpu for gpu in ARCHS if gpu in configs]
    keys = {str(Path(value['api_key_file']).expanduser().resolve()) for value in configs.values()}
    if len(keys) != 1:
        raise ValueError('All grouped rentals must use the same account credential file')
    budget = sum(value['budget_usd'] for value in configs.values())
    if budget <= 0:
        raise ValueError('Cloud group budget must be positive')
    cancel, updates = threading.Event(), queue.Queue()
    members, stages = [], {}
    interrupted = False
    group_error = None
    callback_failed = False
    gate_opened = False
    group_started = time.monotonic()
    next_tick = group_started
    for gpu in (['A100'] if local_run else []) + order:
        stages[gpu] = {'gpu': gpu, 'backend': 'local' if gpu not in configs else 'vast',
                       'status': 'pending', 'exit_code': None, 'elapsed_seconds': 0.0}

    def update(gpu, patch):
        nonlocal group_error, callback_failed, interrupted
        stages[gpu].update(patch)
        if stage_update is not None and not callback_failed:
            try:
                stage_update(gpu, dict(patch))
            except KeyboardInterrupt:
                interrupted = True
                cancel.set()
            except Exception as error:
                callback_failed = True
                cancel.set()
                group_error = {'error_type': type(error).__name__,
                               'error': 'Stage audit callback failed; cleanup continues'}

    def emit(message):
        nonlocal log
        if log is None:
            try:
                print(message, flush=True)
            except OSError:
                pass
        else:
            try:
                log(message)
            except Exception:
                # Optional display failures cannot interrupt cleanup owners.
                log = None
                emit(message)

    def refresh():
        nonlocal next_tick, tick
        now = time.monotonic()
        if tick is not None and now >= next_tick:
            next_tick = now + 1
            try:
                tick()
            except Exception:
                tick = None

    def drain():
        while True:
            try:
                gpu, patch = updates.get_nowait()
            except queue.Empty:
                break
            if gpu is None:
                emit(patch)
            else:
                update(gpu, patch)

    def settle(futures):
        """Join every owner, including after repeated Ctrl-C during cleanup."""
        nonlocal interrupted
        heartbeat = time.monotonic() + 30
        while True:
            try:
                drain()
                refresh()
                if all(future.done() for future in futures):
                    break
                if time.monotonic() >= heartbeat:
                    emit('Cloud group progress: ' + ', '.join(
                        f"{gpu}={stage['status']}" for gpu, stage in stages.items()))
                    heartbeat = time.monotonic() + 30
                time.sleep(.1)
            except KeyboardInterrupt:
                interrupted = True
                cancel.set()
        drain()

    def join_pool(pool):
        # SIGINT can arrive after submit() enqueues a task but before its
        # Future is appended locally. Joining the executor itself also joins
        # those owners; fallback cleanup must never race their SSH/cleanup.
        nonlocal interrupted
        while True:
            try:
                pool.shutdown(wait=True)
                break
            except KeyboardInterrupt:
                interrupted = True
                cancel.set()
        drain()

    def measure(member):
        gpu = member['gpu']
        started = time.monotonic()
        code = 1
        try:
            _cancelled(cancel)
            updates.put((gpu, {'status': 'running'}))
            updates.put((None, f'[{gpu}] running cloud measurements'))
            outcome = cloud.execute_worker(member['ssh'], member['remote'], member['config'],
                cancel_event=cancel,
                progress_callback=lambda snapshot: updates.put((gpu, {'progress': snapshot})))
            code = outcome if outcome in (0, 2, 124, 130) else 1
            if code:
                cancel.set()
        except Exception as error:
            code = 130 if cancel.is_set() else 1
            cancel.set()
            write_json(member['folder'] / 'error.json', {'error_type': type(error).__name__,
                       'error': str(error)[:1000]})
        finally:
            updates.put((gpu, {'status': 'cleaning'}))
            evidence = _finish(member, collection_module, code, collect=True)
            code = evidence['exit_code']
            updates.put((gpu, {'status': 'completed' if code == 0 else 'interrupted' if code == 130 else 'failed',
                               'exit_code': code, 'elapsed_seconds': round(time.monotonic() - started, 6),
                               'cloud': evidence}))
        return code

    def measure_local():
        started = time.monotonic()
        code = 1
        try:
            _cancelled(cancel)
            updates.put(('A100', {'status': 'running'}))
            code = int(local_run(cancel))
            if code:
                cancel.set()
        except Exception:
            code = 130 if cancel.is_set() else 1
            cancel.set()
        finally:
            updates.put(('A100', {'status': 'completed' if code == 0 else 'interrupted' if code == 130 else 'failed',
                                  'exit_code': code, 'elapsed_seconds': round(time.monotonic() - started, 6)}))
        return code

    key_file = Path(next(iter(keys)))
    lock_path = key_file.parent / '.thunderfft-execution.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another cloud execution or cleanup watchdog is active for this key directory') from None
        client = _GroupClient(VastClient(key_file, repo_root=root))
        if any(str(row.get('label', '')).startswith('thunderfft-ae-') for row in client.instances()):
            raise ProviderError('An existing ThunderFFT rental needs cleanup before another execution')
        # Validate every job, SSH key pair and collection identity before rental.
        public_keys = {}
        for gpu in order:
            config = configs[gpu]
            private = validate_private_key(Path(config['ssh_key_file']), repo_root=root)
            public = Path(config['ssh_public_key_file']).read_text().strip()
            derived = subprocess.run(['ssh-keygen', '-y', '-P', '', '-f', str(private)],
                                     text=True, capture_output=True, timeout=10, check=True).stdout.strip()
            if public.split()[:2] != derived.split()[:2]:
                raise ValueError('SSH public and private key do not match')
            public_keys[gpu] = public
        folder = Path(args.result_dir).resolve() / 'cloud' / ('group-' + uuid.uuid4().hex)
        folder.mkdir(parents=True, mode=0o700)
        source_path = folder / 'source.tar.gz'
        source = build_source_bundle(root, source_path, allow_dirty=args.cloud_allow_dirty,
                                     max_bytes=min(512 * 1024**2, min(int(c['upload_gb'] * 1e9) for c in configs.values())))
        write_json(folder / 'source.json', source)
        write_json(folder / 'preflight.json', {'proposals': proposals, 'budget_usd': budget})
        fresh = {}
        for gpu in order:
            try:
                fresh[gpu] = cloud.plan(client, configs[gpu], gpu)
            except ProviderError as error:
                raise PreflightUnavailable(gpu, str(error)) from None
        total = sum(proposal['cost']['planned_total_usd'] for proposal in fresh.values())
        balance = min(proposal['account_before']['balance_usd'] for proposal in fresh.values())
        if total > min(budget, balance):
            raise PreflightUnavailable(None,
                'Fresh cloud group plans exceed the shared budget or account balance; no rentals created')
        write_json(folder / 'plan.json', {'proposals': list(fresh.values()), 'planned_total_usd': total,
                                        'budget_usd': budget, 'startup_retries': 0})
        existing = collection_module.read_manifest(collection_module.location(args.result_dir, args.collection)) if args.collection else None
        for gpu in order:
            child = copy.copy(args)
            child.gpu, child.part = gpu, gpu.lower()
            child.evaluations = ['all'] if gpu == 'A100' else ['table3', 'table4']
            child.cloud_batch = None
            entries = [e for e in select_tasks(child.evaluations, registry['evaluations'], parser) if e['id'] != 'figure2']
            jobs = cloud.make_jobs(child, entries, registry, configs[gpu], select_tasks, parser)
            if existing and any(part['part'] == child.part and part['evaluation'] in jobs[0]['evaluations'] for part in existing['parts']):
                raise ValueError('A requested collection part already exists; choose a new collection before rental')
            dependencies = required_dependencies(jobs, mathdx=configs[gpu]['mathdx'])
            label = 'thunderfft-ae-' + uuid.uuid4().hex
            member_folder = folder / label
            member_folder.mkdir(mode=0o700)
            shutil.copyfile(source_path, member_folder / 'source.tar.gz')
            write_json(member_folder / 'source.json', source)
            write_json(member_folder / 'plan.json', fresh[gpu])
            request = {'gpu': gpu, 'arch': ARCHS[gpu], 'profile': args.profile, 'jobs': jobs,
                       'build_jobs': args.jobs, 'task_timeout': args.timeout, 'collection': args.collection,
                       'execution_seconds': configs[gpu]['execution_seconds'],
                       'max_result_bytes': int(configs[gpu]['download_gb'] * 1e9), **dependencies}
            write_json(member_folder / 'request.json', request)
            members.append({'gpu': gpu, 'args': child, 'config': configs[gpu], 'jobs': jobs,
                            'dependencies': dependencies, 'folder': member_folder, 'proposal': fresh[gpu],
                            'label': label, 'remote': '/workspace/' + label, 'client': client,
                            'public': public_keys[gpu], 'finished': False,
                            'log': lambda message: updates.put((None, message))})
        with ThreadPoolExecutor(max_workers=len(members) + bool(local_run)) as pool:
            phase, active_gpu = 'acquisition', None
            try:
                # All watchdogs are armed before their own create. No benchmark
                # or remote preparation starts until every acquisition succeeded.
                for member in members:
                    _cancelled(cancel)
                    gpu, config = member['gpu'], member['config']
                    active_gpu = gpu
                    update(gpu, {'status': 'acquiring'})
                    _cancelled(cancel)
                    emit(f"[{gpu}] acquiring cloud server; all-in plan "
                         f"${member['proposal']['cost']['planned_total_usd']:.4f}, "
                         f"allocation ${config['budget_usd']:.4f}")
                    created = time.time()
                    state = {'label': member['label'], 'api_key_file': config['api_key_file'],
                             'created_at': created,
                             'deadline': created + config['startup_seconds'] + config['execution_seconds'] + config['download_seconds'],
                             'instance_id': None, 'create_attempted': False, 'cleanup_confirmed': False, 'finished': False}
                    member['state'] = state
                    write_json(member['folder'] / 'state.json', state)
                    member['watchdog'] = cloud.arm_watchdog(member['folder'], lock)

                    def validate_replacement(replacement):
                        plans = [replacement if other is member else other['proposal'] for other in members]
                        total = sum(item['cost']['planned_total_usd'] for item in plans)
                        if total > min(budget, replacement['account_before']['balance_usd']):
                            raise ProviderError('Replacement offer exceeds the shared budget or account balance')
                        write_json(folder / 'plan.json', {'proposals': plans, 'planned_total_usd': total,
                                   'budget_usd': budget, 'startup_retries': 0})

                    cloud.acquire_instance(client, config, gpu, state, member['folder'], member['proposal'],
                        cancel_event=cancel, log=emit, validate_plan=validate_replacement,
                        acquisition_deadline=min(other['state']['created_at'] + other['config']['startup_seconds']
                                                 for other in members if 'state' in other))
                    client.attach_ssh_key(state['instance_id'], member['public'])
                deadline = min(member['state']['created_at'] + member['config']['startup_seconds'] for member in members)
                phase, active_gpu = 'preparation', None
                prepare_futures = []
                for member in members:
                    update(member['gpu'], {'status': 'preparing'})
                    emit(f"[{member['gpu']}] server acquired; waiting for SSH and preparing build environment; "
                         f"audit: {member['folder']}")
                    prepare_futures.append(pool.submit(_prepare, member, root, deadline, cancel))
                # A failed preparation cancels peers immediately instead of
                # letting a slow bootstrap run until its full startup allowance.
                heartbeat = time.monotonic() + 30
                while True:
                    refresh()
                    failures = [(member, future.exception()) for member, future in zip(members, prepare_futures)
                                if future.done() and future.exception() is not None]
                    if failures:
                        cancel.set()
                        original = next(((member, error) for member, error in failures
                                         if not isinstance(error, InterruptedError)), failures[0])
                        failed, error = original
                        if group_error is None or group_error.get('error_type') == 'InterruptedError':
                            group_error = {'gpu': failed['gpu'], 'error_type': type(error).__name__,
                                           'error': f"[{failed['gpu']}] preparation failed: {str(error)[:1000]}"}
                    for member, future in zip(members, prepare_futures):
                        if future.done() and future.exception() is None and stages[member['gpu']]['status'] == 'preparing':
                            update(member['gpu'], {'status': 'ready'})
                            emit(f"[{member['gpu']}] environment ready; waiting for all GPUs")
                    if all(future.done() for future in prepare_futures):
                        break
                    if time.monotonic() >= heartbeat:
                        pending = [member['gpu'] for member, future in zip(members, prepare_futures) if not future.done()]
                        emit('Waiting for cloud preparation: ' + ', '.join(pending) +
                             f'; {max(0, int(deadline - time.time()))}s until the shared startup deadline')
                        heartbeat = time.monotonic() + 30
                    time.sleep(.1)
                if failures:
                    raise original[1]
                _cancelled(cancel)
                cloud.remaining(deadline)
                phase, active_gpu = 'final readiness check', 'A100' if local_run else None
                if ready_check is not None:
                    ready_check()
                cloud.remaining(deadline)
                _cancelled(cancel)
                gate_opened = True
                write_json(folder / 'ready.json', {'all_cloud_ready_at': time.time(),
                                                  'startup_deadline': deadline, 'gpus': order})
                emit('All required GPU environments are ready; starting measurements in parallel.')
                phase, active_gpu = 'measurement', None
                futures = [pool.submit(measure, member) for member in members]
                if local_run is not None:
                    futures.append(pool.submit(measure_local))
                settle(futures)
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                interrupted = True
                cancel.set()
            except Exception as error:
                cancel.set()
                prefix = f'[{active_gpu}] ' if active_gpu else ''
                group_error = group_error or {'error_type': type(error).__name__,
                                             'error': f'{prefix}{phase} failed: {str(error)[:1000]}'}
                try:
                    write_json(folder / 'error.json', group_error)
                except (OSError, ValueError, TypeError):
                    pass  # An unavailable audit filesystem cannot skip cleanup.
            finally:
                # Preparation may have been interrupted in the controller. Join
                # its cancelled transports before cleanup touches the same SSH.
                if 'prepare_futures' in locals():
                    settle(prepare_futures)
                if 'futures' in locals():
                    settle(futures)
                join_pool(pool)
                unfinished = [member for member in members if member.get('state') is not None and not member['finished']]
                while unfinished:
                    cleanup_pool = ThreadPoolExecutor(max_workers=len(unfinished))
                    cleanup = []
                    try:
                        for member in unfinished:
                            update(member['gpu'], {'status': 'cleaning'})
                            try:
                                cleanup.append(cleanup_pool.submit(_finish, member, collection_module,
                                                                   130 if interrupted else 1, collect=False))
                            except KeyboardInterrupt:
                                interrupted = True
                                cancel.set()
                        settle(cleanup)
                    finally:
                        join_pool(cleanup_pool)
                    for future in cleanup:
                        future.result()
                    for member in unfinished:
                        if member['finished']:
                            evidence = member['evidence']
                            update(member['gpu'], {'status': 'interrupted' if interrupted else 'failed',
                                                  'exit_code': evidence['exit_code'], 'cloud': evidence,
                                                  'elapsed_seconds': round(time.monotonic() - group_started, 6)})
                    # A signal before submit() may leave a member unsubmitted;
                    # queued owners have all joined, so only these may retry cleanup.
                    unfinished = [member for member in unfinished if not member['finished']]
                drain()
        for gpu, stage in stages.items():
            if stage['exit_code'] is None:
                update(gpu, {'status': 'not_started', 'exit_code': 130 if interrupted else 1})
        evidence = [member['evidence'] for member in members if member.get('evidence')]
        reserved = sum(item['reserved_total_usd'] for item in evidence)
        clean = all(item['cleanup_confirmed'] for item in evidence)
        codes = [stage['exit_code'] for stage in stages.values()]
        # Cancellation caused by a failed peer must not hide the initiating
        # failure behind a different stage's cooperative exit 130.
        failure_code = next((c for c in codes if c not in (0, 2, 130)),
                            2 if 2 in codes else next((c for c in codes if c), 0))
        code = 130 if interrupted else 1 if group_error or reserved > budget or not clean else failure_code
        result = {'exit_code': code, 'stages': list(stages.values()), 'reserved_total_usd': reserved,
                  'budget_usd': budget, 'cleanup_confirmed': clean, 'billing_settlement_verified': False,
                  'audit_folder': str(folder), 'all_ready_gate_opened': gate_opened, 'startup_retries': 0,
                  'elapsed_seconds': round(time.monotonic() - group_started, 6)}
        if group_error:
            result['error'] = group_error
        try:
            write_json(folder / 'execution.json', result)
        except (OSError, ValueError, TypeError) as error:
            result['audit_error'] = type(error).__name__
            result['exit_code'] = result['exit_code'] or 1
        return result
