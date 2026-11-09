#!/usr/bin/env python3
"""Run Table 5 accuracy phases and freshly remeasure finite boundaries."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
FIELDS = ('N', 'batch', 'mode', 'alpha', 'precision', 'rel_l2', 'max_abs', 'rel_linf', 'all_finite')
RANGES = {'range_fp16': 'fp16', 'range_tf32': 'tf32', 'range_fp16_cufftdx': 'fp16_cufftdx',
          'range_fp32_cufftdx': 'fp32_cufftdx'}
ACCURACY = {'accuracy_reduced': 'both', 'accuracy_fp16_cufftdx': 'cufftdx_fp16',
            'accuracy_fp32_cufftdx': 'cufftdx_fp32'}
NATIVE_PRECISION = {'fp16_cufftdx': 'cufftdx_fp16', 'fp32_cufftdx': 'cufftdx_fp32'}
# A ThunderFFT boundary of the same storage format provides a starting hint only.
HINT_FALLBACK = {'fp16_cufftdx': 'fp16', 'fp32_cufftdx': 'tf32'}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_rows(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f'{path}: invalid CSV header')
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f'{path}: malformed CSV')
        return rows


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def boundary(rows, n, precision, window, batch):
    observed = {}
    for row in rows:
        alpha = float(row['alpha'])
        if (int(row['N']) != n or int(row['batch']) != batch or row['precision'] != precision
                or row['mode'] != 'gaussian' or row['all_finite'] not in ('0', '1')
                or not math.isfinite(alpha) or alpha <= 0):
            raise ValueError('finite scan returned an invalid measurement configuration')
        exponent = round(math.log2(alpha))
        if not math.isclose(alpha, math.ldexp(1.0, exponent), rel_tol=1e-7) or exponent in observed:
            raise ValueError('finite scan has duplicate or invalid exponents')
        observed[exponent] = row['all_finite'] == '1'
    if sorted(observed) != list(range(window[0], window[1] + 1)):
        raise ValueError('finite scan has incomplete exponent coverage')
    yes = [k for k, ok in observed.items() if ok]
    no = [k for k, ok in observed.items() if not ok]
    if yes and no:
        if max(yes) >= min(no):
            raise ValueError('finite scan is nonmonotonic; cannot identify one boundary')
        return max(yes)
    return None


def starting_window(presets, precision, n, limits, metadata):
    values = presets['entries'].get(precision, {})
    source_precision = precision
    if str(n) not in values and precision in HINT_FALLBACK:
        source_precision = HINT_FALLBACK[precision]
        values = presets['entries'].get(source_precision, {})
    hint = values.get(str(n))
    if hint is None:
        return list(limits), None, 'no retained boundary; full profile interval'
    matched = (source_precision == precision and metadata['batch'] == presets['batch']
               and metadata['seed'] == presets['seed'])
    sources = [s['id'] for s in presets['sources'] if s['precision'] == source_precision]
    source = ','.join(sources) + f':{source_precision}:N={n}'
    if not matched:
        source += ' (starting hint; measurement conditions or implementation differ)'
    center = max(limits[0], min(hint, limits[1]))
    return [max(limits[0], center - 1), min(limits[1], center + 2)], hint if matched else None, source


def expand(window, limits):
    width = window[1] - window[0] + 1
    return [max(limits[0], window[0] - width), min(limits[1], window[1] + width)]


def collect(output, binary, cufftdx_binary, timeout):
    metadata_path = output / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    mode = metadata.get('range_mode', 'boundary')
    if mode not in ('boundary', 'full'):
        raise ValueError('THUNDERFFT_TABLE5_RANGE_MODE must be boundary or full')
    presets_path = HERE / 'boundary_presets.json'
    presets = json.loads(presets_path.read_text())
    metadata['boundary_presets'] = {'sha256': hashlib.sha256(presets_path.read_bytes()).hexdigest(),
                                   'data': presets}
    metadata['range_checks'] = {}
    metadata['range_coverage'] = 'boundary_recheck' if mode == 'boundary' else 'full_profile_scan'
    metadata['full_range_scan'] = mode == 'full'
    commands = metadata.setdefault('execution', {}).setdefault('commands', [])
    raw = output / 'raw'
    raw.mkdir(exist_ok=True)

    def execute(phase, sizes, extra, label):
        target = cufftdx_binary if phase.endswith('_cufftdx') else binary
        csv_path, env_path, log_path = [raw / (label + suffix) for suffix in ('.csv', '.environment.json', '.log')]
        command = [str(target.resolve()), '--N-list', ','.join(map(str, sizes)), '--batch', str(metadata['batch']),
                   '--mode', 'gaussian', '--seed', str(metadata['seed']), *extra,
                   '--csv', str(csv_path), '--environment-json', str(env_path)]
        record = {'phase': phase, 'argv': command, 'started_at': now(),
                  'csv': str(csv_path.relative_to(output)), 'environment': str(env_path.relative_to(output)),
                  'log': str(log_path.relative_to(output))}
        commands.append(record)
        write_json(metadata_path, metadata)
        with (output / 'commands.txt').open('a') as stream:
            stream.write(shlex.join(command) + '\n')
        print(f'Table 5 [{label}]', flush=True)
        try:
            with log_path.open('w') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            record['exit_code'] = process.returncode
        except subprocess.TimeoutExpired:
            record['exit_code'] = 124
            raise RuntimeError(f'{label}: timed out after {timeout}s')
        finally:
            record['finished_at'] = now()
            write_json(metadata_path, metadata)
        with (output / (phase + '.log')).open('a') as log:
            log.write(log_path.read_text())
        if record['exit_code']:
            raise RuntimeError(f'{label}: native exit {record["exit_code"]}; see {log_path}')
        environment = json.loads(env_path.read_text())
        combined_environment = output / (phase + '.environment.json')
        if combined_environment.exists() and json.loads(combined_environment.read_text()) != environment:
            raise ValueError('native GPU/library environment changed within a phase')
        write_json(combined_environment, environment)
        return read_rows(csv_path), record

    for phase in metadata['selected_phases']:
        rows = []
        if phase in ACCURACY:
            rows, _ = execute(phase, metadata['N'], ['--precision', ACCURACY[phase], '--alpha', '1',
                                                   '--reference', 'fp64'], phase)
        elif phase in RANGES:
            precision = RANGES[phase]
            native_precision = NATIVE_PRECISION.get(precision, precision)
            common = ['--precision', native_precision, '--alpha-sweep', '--fast-finite-only']
            limits = metadata['range_exponents'][precision]
            if mode == 'full':
                rows, _ = execute(phase, metadata['N'], common + ['--alpha-exp-min', str(limits[0]),
                                                                 '--alpha-exp-max', str(limits[1])], phase)
            else:
                checks = metadata['range_checks'].setdefault(precision, {})
                for n in metadata['N']:
                    window, expected, source = starting_window(presets, precision, n, limits, metadata)
                    check = {'initial_window': list(window), 'window': list(window),
                             'expected_last_finite': expected, 'preset_source': source, 'attempts': []}
                    checks[str(n)] = check
                    while True:
                        number = len(check['attempts'])
                        measured, record = execute(phase, [n], common + ['--alpha-exp-min', str(window[0]),
                                    '--alpha-exp-max', str(window[1])], f'{phase}-n{n}-attempt{number}')
                        check['attempts'].append(dict(min=window[0], max=window[1],
                            **{key: record[key] for key in ('csv', 'environment', 'log', 'exit_code')}))
                        check['window'] = list(window)
                        found = boundary(measured, n, precision, window, metadata['batch'])
                        wider = expand(window, limits)
                        shifted = expected is not None and found is not None and found != expected
                        if found is not None and not (number == 0 and shifted and wider != window):
                            check['observed_last_finite'] = found
                            check['boundary_matches_preset'] = None if expected is None else found == expected
                            rows.extend(measured)
                            write_json(metadata_path, metadata)
                            break
                        if wider == window:
                            raise ValueError(f'{phase} N={n}: no finite/nonfinite pair in {window}; '
                                             'use the paper profile or full mode to record censored coverage')
                        window = wider
                        print(f'Table 5 [{phase} N={n}]: expanding boundary window to {window}', flush=True)
        else:
            raise ValueError(f'unsupported phase: {phase}')
        write_rows(output / (phase + '.csv'), rows)
        write_json(metadata_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cufftdx-binary', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    try:
        collect(args.output.resolve(), args.binary, args.cufftdx_binary, args.timeout)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Table 5 failed: {error}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
