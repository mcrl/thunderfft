"""Correctness-only preflight for the immutable paired benchmark binaries."""
import csv
import hashlib
import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from artifact.table3.measure import RAW_FIELDS

TIMING_ERROR = 'ERROR: nonpositive/nonfinite differential timing'


def timing_only_race_outcome(code, stderr, stdout):
    """Recognize the instrumented timing stop after the native R=1 gates.

    This is not a successful timing trial. The immutable native program has
    already checked repeat/e2e accuracy and executed the repeated wrappers.
    The caller must independently require a zero-hazard sanitizer summary.
    """
    lines = stderr.rstrip().splitlines()
    reader = csv.DictReader(stdout.splitlines())
    rows = list(reader)
    return (code == 1 and bool(lines) and lines[-1] == TIMING_ERROR
            and [line for line in lines if line.startswith('ERROR:')] == [TIMING_ERROR]
            and tuple(reader.fieldnames or ()) == RAW_FIELDS and not rows)


def race_command(binary, n, precision, placement):
    # Native ABI: N precision placement R trials batch. R32 gives a meaningful
    # differential interval even for a tiny batch; these timings are discarded.
    return [str(binary), str(n), precision, placement, '32', '1', '32']


def sanitizer_path():
    path = shutil.which('compute-sanitizer')
    if path:
        return path
    roots = [Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')),
             Path('/usr/local/cuda')]
    nvcc = os.environ.get('CUDACXX') or os.environ.get('NVCC') or shutil.which('nvcc')
    if nvcc:
        roots.insert(0, Path(nvcc).resolve().parents[1])
    for root in roots:
        for relative in ('bin/compute-sanitizer', 'compute-sanitizer/compute-sanitizer'):
            candidate = root / relative
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    raise RuntimeError('compute-sanitizer is required for the requested validation gates')


def sanitizer_clean(tool, text):
    if tool == 'racecheck':
        return bool(re.search(r'RACECHECK SUMMARY:\s*0 hazards displayed\s*\(0 errors, 0 warnings\)', text))
    return bool(re.search(r'ERROR SUMMARY:\s*0 errors\b', text))


def run_checked(command, stdout, stderr, timeout):
    # Stop the instrumented application together with its sanitizer on timeout.
    # The external cloud worker/watchdog retains its independent total deadline.
    with stdout.open('w') as out, stderr.open('w') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def validation_plan(validator, benchmark, output, sanitizers, sanitizer='compute-sanitizer'):
    output = Path(output)
    tasks = [{'name': 'after-full-validation', 'kind': 'validator', 'expected_rows': 500,
              'command': [str(validator), '--csv', str(output/'after-full-validation.csv')],
              'csv': 'after-full-validation.csv', 'timeout': 120}]
    if not sanitizers:
        return tasks
    for tool in ('memcheck', 'synccheck'):
        name = 'after-quick-' + tool
        tasks.append({'name': name, 'kind': tool, 'expected_rows': 88,
                      'command': [sanitizer, '--tool', tool, '--error-exitcode', '99',
                                  '--log-file', str(output/(name+'-sanitizer.log')),
                                  str(validator), '--quick', '--csv', str(output/(name+'.csv'))],
                      'csv': name+'.csv', 'timeout': 120})
    for n in (64, 128, 512):
        for precision in ('fp16', 'tf32'):
            for placement in ('register', 'shared'):
                name = f'after-racecheck-n{n}-{precision}-{placement}'
                tasks.append({'name': name, 'kind': 'racecheck',
                              'command': [sanitizer, '--tool', 'racecheck',
                                          '--error-exitcode', '99', '--racecheck-report', 'all',
                                          '--log-file', str(output/(name+'-sanitizer.log')),
                                          *race_command(benchmark, n, precision, placement)],
                              'timeout': 45})
    return tasks


def run_validation(validator, benchmark, output, *, sanitizers, save, report):
    output = Path(output)
    sanitizer = sanitizer_path() if sanitizers else None
    tasks = validation_plan(validator, benchmark, output, sanitizers, sanitizer)
    started = time.monotonic()
    evidence = {'status': 'running', 'full_expected_cases': 500,
                'validator_binary_sha256': hashlib.sha256(Path(validator).read_bytes()).hexdigest(),
                'sanitizers': 'required' if sanitizers else 'not_requested_on_this_architecture',
                'sanitizer_executable': sanitizer,
                'aggregate_timeout_seconds': 360, 'runs': []}
    evidence['validation_helper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    evidence['native_adapter_sha256'] = hashlib.sha256(
        (Path(__file__).resolve().parents[2]/'sample/bench/paper_fft.cu').read_bytes()).hexdigest()
    if evidence['native_adapter_sha256'] != '8bca2a0f3d54e75aa9ce9ad82aca47f376bf3df44cdcd8e67351f8ab6e3b4f53':
        raise RuntimeError('Native adapter changed; re-audit the post-correctness timing exception')
    save(evidence)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    def interrupt_validation(signum, frame):
        raise KeyboardInterrupt('validation interrupted by SIGTERM')
    signal.signal(signal.SIGTERM, interrupt_validation)
    try:
        for task in tasks:
            remaining = 360 - (time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('validation preflight exceeded its 360-second aggregate limit')
            name = task['name']
            report(0, 56, detail=name, status='validating')
            entry = {**task, 'status': 'running', 'started_at_unix': time.time()}
            evidence['runs'].append(entry)
            save(evidence)
            begin = time.monotonic()
            code = run_checked(task['command'], output/(name+'.stdout'),
                               output/(name+'.log'), min(task['timeout'], remaining))
            entry.update(returncode=code, elapsed_seconds=time.monotonic()-begin)
            timing_unavailable = task['kind'] == 'racecheck' and timing_only_race_outcome(
                code, (output/(name+'.log')).read_text(), (output/(name+'.stdout')).read_text())
            if code and not timing_unavailable:
                raise RuntimeError(f'{name} failed with return code {code}')
            if 'csv' in task:
                path = output/task['csv']
                rows = list(csv.DictReader(path.open())) if path.is_file() else []
                entry['rows'] = len(rows)
                if len(rows) != task['expected_rows'] or any(row.get('pass') != '1' for row in rows):
                    raise RuntimeError(f'{name}: expected {task["expected_rows"]} passing cases')
            if task['kind'] != 'validator':
                log = output/(name+'-sanitizer.log')
                entry['sanitizer_zero_summary'] = log.is_file() and sanitizer_clean(task['kind'], log.read_text())
                if not entry['sanitizer_zero_summary']:
                    raise RuntimeError(f'{name}: missing zero-error sanitizer summary')
            if task['kind'] == 'racecheck':
                # This is the real unmodified benchmark wrapper, including its
                # FP64 R=1 gate. Sanitized timings never enter paired-data.csv.
                rows = list(csv.DictReader((output/(name+'.stdout')).open()))
                if not timing_unavailable and (len(rows) != 1 or
                        (rows[0].get('inside_repeats'), rows[0].get('batch')) != ('32', '32')):
                    raise RuntimeError(f'{name}: benchmark wrapper did not complete the requested case')
                entry['rows'] = len(rows)
                entry['native_correctness_gate_completed'] = True
                entry['native_timing_status'] = ('unavailable_under_instrumentation'
                                                if timing_unavailable else 'completed_but_discarded')
                if timing_unavailable:
                    entry['native_terminal_error'] = TIMING_ERROR
                    entry['native_csv_header_verified'] = True
                    entry['note'] = ('Both R=1 FP64 correctness gates and repeated wrapper launches '
                                     'precede this exact native timing exception; no timing row is claimed.')
            entry['status'] = 'passed'
            save(evidence)
        evidence.update(status='passed', elapsed_seconds=time.monotonic()-started)
        save(evidence)
        return evidence
    except BaseException as error:
        evidence.update(status='failed', error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.monotonic()-started)
        if evidence['runs'] and evidence['runs'][-1]['status'] == 'running':
            evidence['runs'][-1]['status'] = 'failed'
        save(evidence)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
