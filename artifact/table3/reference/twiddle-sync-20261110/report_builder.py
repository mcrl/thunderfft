#!/usr/bin/env python3
"""Build an offline Table-3-shaped report from completed paired GPU runs.

Example:
  python3 report_builder.py --input A100 RUN/paired-summary.csv \
    RUN/paired-metadata.json RUN/after-full-validation.csv --output REPORT
Use after-full-validation.csv for v2 or after-validation.csv for v1.
Repeat --input for other GPUs. The sibling paired-data.csv supplies trial
variation; --data LABEL PATH can override it. No GPU or network calls are made.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

BASELINE = '00bc845d0f3119fa405517cc601126bdbd9b3351'
NATIVE_ADAPTER_SHA256 = '8bca2a0f3d54e75aa9ce9ad82aca47f376bf3df44cdcd8e67351f8ab6e3b4f53'
TIMING_ERROR = 'ERROR: nonpositive/nonfinite differential timing'
NATIVE_FIELDS = ('N', 'precision', 'placement', 'batch', 'inside_repeats', 'trial',
                 't_R_us', 't_2R_us', 'compute_us', 'e2e_out_of_place_us',
                 'single_fft_relL2', 'shared_bytes')
SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
GROUPS = (('fp16', 'register'), ('fp16', 'shared'),
          ('tf32', 'register'), ('tf32', 'shared'))
MATRIX = {(n, p, l) for n in SIZES for p, l in GROUPS}
OUTPUT_NAMES = ('before-after.csv', 'before-after.md', 'aggregate.csv', 'report-inputs.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, description, *, positive=False):
    v = float(value)
    require(math.isfinite(v) and (v > 0 if positive else v >= 0),
            f'{description}: expected a finite {"positive" if positive else "nonnegative"} value')
    return v


def close(a, b, description):
    require(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10),
            f'{description}: inconsistent values {a} versus {b}')


def read_csv(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                f'{path}: missing or duplicate CSV headers')
        rows = list(reader)
    require(rows and all(None not in r and all(v is not None for v in r.values()) for r in rows),
            f'{path}: empty or malformed CSV')
    return rows


def file_record(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_metadata(meta, label):
    require(meta['evaluation'] == 'paired_thunderfft_optimization' and meta['status'] == 'completed',
            f'{label}: only completed paired measurements can be reported')
    require(meta['baseline_head'] == BASELINE and meta['full_paper_comparison'] is False,
            f'{label}: wrong baseline or paper-comparison scope')
    settings = meta['settings']
    require(all(settings[k] == v for k, v in {'batch': 65536, 'inside_repeats': 1000, 'trials': 3}.items()),
            f'{label}: requires batch=65536, R=1000, trials=3')
    require(set(settings['N']) == set(SIZES) and set(settings['precisions']) == {'fp16', 'tf32'}
            and set(settings['placements']) == {'register', 'shared'}, f'{label}: incomplete settings matrix')
    require(meta['completed_native_runs'] == 56 and meta['complete_paired_configurations'] == 28,
            f'{label}: incomplete paired execution')
    require(meta['passed_single_fft_gates'] == 56, f'{label}: native correctness gates incomplete')
    commands = meta['commands']
    require(len(commands) == 56 and all(c['returncode'] == 0 for c in commands),
            f'{label}: ordinary paired performance commands must all return zero')
    command_cases = {(int(c['command'][-6]), c['command'][-5], c['command'][-4], c['variant'])
                     for c in commands}
    require(command_cases == {(*case, side) for case in MATRIX for side in ('before', 'after')}
            and all(c['command'][-3:] == ['1000', '3', '65536'] for c in commands),
            f'{label}: incomplete or mismatched ordinary performance command matrix')
    identity = meta['native_identity']
    require(re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', identity['uuid']),
            f'{label}: invalid GPU UUID')
    expected_arch = {'A100': '80', 'H100': '90', 'B200': '100'}.get(identity['gpu'])
    require(expected_arch is None or settings['arch'] == expected_arch,
            f'{label}: build architecture does not match GPU family')
    versions = [meta['builds'][v]['nvcc'] for v in ('before', 'after')]
    require(all(v['returncode'] == 0 for v in versions) and versions[0]['stdout'] == versions[1]['stdout'],
            f'{label}: before/after compiler versions differ or are unavailable')
    hashes = meta['source_sha256']
    adapter = 'sample/bench/paper_fft.cu'
    require(hashes['before'][adapter] == hashes['after'][adapter], f'{label}: timing adapter changed')
    return identity


def validation_contract(saved, label, evidence_dir):
    """Accept the historical quick gate and the v2 full/sanitizer evidence."""
    if 'runs' not in saved:
        require(saved['returncode'] == 0, f'{label}: candidate validation failed')
        return saved['rows'], {'schema': 'v1', 'sanitizers': 'not_recorded', 'sanitizer_runs': 0}
    require(saved['status'] == 'passed' and saved['full_expected_cases'] == 500,
            f'{label}: v2 full validation did not pass')
    require(re.fullmatch(r'[0-9a-f]{64}', saved['validator_binary_sha256']),
            f'{label}: missing/invalid validator binary hash')
    mode = saved['sanitizers']
    require(mode in ('required', 'not_requested_on_this_architecture'),
            f'{label}: unknown sanitizer scope')
    expected = {'after-full-validation': ('validator', 500)}
    if mode == 'required':
        expected.update({'after-quick-memcheck': ('memcheck', 88),
                         'after-quick-synccheck': ('synccheck', 88)})
        expected.update({f'after-racecheck-n{n}-{p}-{l}': ('racecheck', 1)
                         for n in (64, 128, 512) for p, l in GROUPS})
    runs = {r['name']: r for r in saved['runs']}
    require(len(runs) == len(saved['runs']) and set(runs) == set(expected),
            f'{label}: missing, duplicate or unexpected required validation runs')
    unavailable, special_files = [], {}
    for name, (kind, count) in expected.items():
        run = runs[name]
        require(run['status'] == 'passed' and run['kind'] == kind,
                f'{label}: required validation run failed/incomplete: {name}')
        timing_only = kind == 'racecheck' and run['returncode'] == 1
        if timing_only:
            require(run['rows'] == 0 and run.get('native_timing_status') == 'unavailable_under_instrumentation'
                    and run.get('native_correctness_gate_completed') is True
                    and run.get('native_terminal_error') == TIMING_ERROR
                    and run.get('native_csv_header_verified') is True,
                    f'{label}: rc1 lacks exact instrumentation-only evidence: {name}')
            require(saved.get('native_adapter_sha256') == NATIVE_ADAPTER_SHA256
                    and re.fullmatch(r'[0-9a-f]{64}', saved.get('validation_helper_sha256', '')),
                    f'{label}: rc1 requires the pinned audited adapter and helper hash')
            stderr = evidence_dir / (name + '.log')
            stdout = evidence_dir / (name + '.stdout')
            sanitizer_log = evidence_dir / (name + '-sanitizer.log')
            lines = stderr.read_text().rstrip().splitlines()
            require(lines and lines[-1] == TIMING_ERROR
                    and [line for line in lines if line.startswith('ERROR:')] == [TIMING_ERROR],
                    f'{label}: rc1 is not the sole terminal timing exception: {name}')
            reader = csv.DictReader(stdout.read_text().splitlines())
            require(tuple(reader.fieldnames or ()) == NATIVE_FIELDS and not list(reader),
                    f'{label}: rc1 must have exactly the native CSV header and zero rows: {name}')
            summaries = re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards displayed\s*\((\d+) errors, (\d+) warnings\)',
                                   sanitizer_log.read_text())
            require(summaries == [('0', '0', '0')], f'{label}: rc1 has no unique zero-hazard summary: {name}')
            unavailable.append(name)
            special_files[name] = {k: file_record(path) for k, path in
                                   [('stderr', stderr), ('stdout', stdout), ('sanitizer_log', sanitizer_log)]}
        else:
            require(run['returncode'] == 0 and run['rows'] == count,
                    f'{label}: required validation run failed/incomplete: {name}')
        if kind != 'validator':
            require(run.get('sanitizer_zero_summary') is True,
                    f'{label}: sanitizer has no zero-error summary: {name}')
            command = run['command']
            require(any(command[i:i+2] == ['--tool', kind] for i in range(len(command)))
                    and any(command[i:i+2] == ['--error-exitcode', '99'] for i in range(len(command))),
                    f'{label}: sanitizer command differs: {name}')
        if kind == 'racecheck':
            parts = name.removeprefix('after-racecheck-n').split('-')
            require(run['command'][-6:] == [*parts, '32', '1', '32'],
                    f'{label}: racecheck did not exercise the requested wrapper/protocol: {name}')
    return 500, {'schema': 'v2', 'sanitizers': mode, 'sanitizer_runs': len(runs) - 1,
                 'validator_binary_sha256': saved['validator_binary_sha256'],
                 'instrumented_timing_unavailable': unavailable, 'instrumented_timing_evidence': special_files}


def load_validation(path, meta, label):
    rows = read_csv(path)
    expected_rows, scope = validation_contract(meta['candidate_validation'], label, path.parent)
    require(expected_rows == len(rows), f'{label}: validation evidence count differs')
    coverage, pattern_coverage = set(), set()
    maxima = {p: 0.0 for p, _ in GROUPS}
    by_size = {}
    for r in rows:
        require(r['pass'] == '1', f'{label}: failed candidate validation row')
        require(all(int(r[k]) == 0 for k in ('nonfinite_outputs', 'shared_guard_changed_max', 'output_guard_changed')),
                f'{label}: nonfinite output or overwritten validation guard')
        n, p = int(r['N']), r['precision']
        require(n in SIZES and p in maxima and r['direction'] in ('forward', 'inverse')
                and r['path'] in ('direct_core_guarded', 'public_global'), f'{label}: unknown validation case')
        rel = number(r['relative_l2'], f'{label} validation relative L2')
        require(rel <= .005, f'{label}: candidate relative L2 exceeds 0.005')
        coverage.add((n, p, r['direction'], r['path']))
        pattern_coverage.add((n, p, r['direction'], r['path'], r['input']))
        maxima[p] = max(maxima[p], rel)
        by_size[n, p] = max(by_size.get((n, p), 0), rel)
    expected = {(n, p, d, path) for n in SIZES for p in ('fp16', 'tf32')
                for d in ('forward', 'inverse') for path in ('direct_core_guarded', 'public_global')}
    require(coverage == expected, f'{label}: validation lacks forward/inverse direct/public cases')
    if scope['schema'] == 'v2':
        full_patterns = {'gaussian_rms1', 'offset_impulse', 'complex_tone', 'constant'}
        require(pattern_coverage == {(*case, pattern) for case in expected for pattern in full_patterns},
                f'{label}: full validation lacks the four required input patterns')
    return rows, maxima, by_size, scope


def load_trials(path, label, identity):
    if path is None:
        return None
    rows, groups = read_csv(path), {}
    require(len(rows) == 168, f'{label}: paired-data must contain 168 trial rows')
    for r in rows:
        key = int(r['N']), r['precision'], r['placement']
        require(key in MATRIX and r['variant'] in ('before', 'after'), f'{label}: unexpected trial configuration')
        require(r['gpu_uuid'] == identity['uuid'] and r['gpu'] == identity['gpu'], f'{label}: trial GPU identity differs')
        require(int(r['batch']) == 65536 and int(r['inside_repeats']) == 1000, f'{label}: trial protocol differs')
        group = groups.setdefault((*key, r['variant']), {})
        trial = int(r['trial'])
        require(trial in (0, 1, 2) and trial not in group, f'{label}: duplicate/out-of-range trial')
        parsed = {k: number(r[k], f'{label} {k}', positive=True) for k in
                  ('compute_us', 'e2e_out_of_place_us', 't_R_us', 't_2R_us')}
        # Native CSV uses nine significant digits for each separately printed time.
        require(parsed['t_2R_us'] > parsed['t_R_us'] and math.isclose(
            parsed['compute_us'], (parsed['t_2R_us'] - parsed['t_R_us']) / 1000,
            rel_tol=1e-4, abs_tol=1e-6), f'{label}: inconsistent differential latency')
        parsed['single_fft_relL2'] = number(r['single_fft_relL2'], f'{label} native relative L2')
        require(parsed['single_fft_relL2'] <= .005, f'{label}: native correctness gate failed')
        group[trial] = parsed
    require(len(groups) == 56 and all(set(g) == {0, 1, 2} for g in groups.values()),
            f'{label}: incomplete trial matrix')
    return {key: list(group.values()) for key, group in groups.items()}


def add_variation(row, before, after, metric, raw_key):
    for side, samples in (('before', before), ('after', after)):
        values = [sample[raw_key] for sample in samples]
        median = statistics.median(values)
        close(row[f'{side}_{metric}_us'], median, f'{metric}: summary versus raw median')
        row[f'{side}_{metric}_min_us'] = min(values)
        row[f'{side}_{metric}_max_us'] = max(values)
        row[f'{side}_{metric}_span_pct'] = 100 * (max(values) - min(values)) / median
    bmin, bmax = (row[f'before_{metric}_{x}_us'] for x in ('min', 'max'))
    amin, amax = (row[f'after_{metric}_{x}_us'] for x in ('min', 'max'))
    row[f'{metric}_observed_ratio_min'] = bmin / amax
    row[f'{metric}_observed_ratio_max'] = bmax / amin
    row[f'{metric}_trial_ranges_overlap'] = max(bmin, amin) <= min(bmax, amax)


def load_input(spec, data_override, threshold):
    label, summary_path, metadata_path, validation_path = spec
    require(re.fullmatch(r'[A-Za-z0-9_.-]+', label), 'Input labels must use letters, digits, dot, dash or underscore')
    paths = [Path(p).resolve() for p in (summary_path, metadata_path, validation_path)]
    summary_path, metadata_path, validation_path = paths
    meta = json.loads(metadata_path.read_text())
    identity = validate_metadata(meta, label)
    validation, errors, errors_by_size, validation_scope = load_validation(validation_path, meta, label)
    data_path = Path(data_override).resolve() if data_override else summary_path.with_name('paired-data.csv')
    if not data_path.is_file():
        require(not data_override, f'{label}: explicit trial CSV is missing')
        data_path = None
    trials = load_trials(data_path, label, identity)
    rows, seen = [], set()
    for original in read_csv(summary_path):
        key = int(original['N']), original['precision'], original['placement']
        require(key in MATRIX and key not in seen, f'{label}: unexpected/duplicate summary configuration')
        seen.add(key)
        row = {'gpu': label, 'gpu_name': identity['name'], 'gpu_uuid': identity['uuid'],
               'N': key[0], 'precision': key[1], 'placement': key[2]}
        for metric in ('compute', 'e2e'):
            for side in ('before', 'after'):
                k = f'{side}_{metric}_us'
                row[k] = number(original[k], f'{label} {k}', positive=True)
            ratio = row[f'before_{metric}_us'] / row[f'after_{metric}_us']
            close(number(original[f'{metric}_speedup'], 'speedup', positive=True), ratio, f'{label} reported speedup')
            row[f'{metric}_speedup'] = ratio
            row[f'{metric}_delta_pct'] = 100 * (1 / ratio - 1)
            row[f'{metric}_within_{threshold:g}pct_change'] = abs(row[f'{metric}_delta_pct']) <= threshold
        close(float(original['compute_delta_pct']), row['compute_delta_pct'], f'{label} reported delta')
        for side in ('before', 'after'):
            k = f'{side}_max_rel_l2'
            row[k] = number(original[k], f'{label} {k}')
            require(row[k] <= .005, f'{label}: summary native correctness gate failed')
        row['after_validation_max_rel_l2'] = errors_by_size[key[0], key[1]]
        row['trial_variation_available'] = trials is not None
        if trials:
            b, a = trials[(*key, 'before')], trials[(*key, 'after')]
            add_variation(row, b, a, 'compute', 'compute_us')
            add_variation(row, b, a, 'e2e', 'e2e_out_of_place_us')
            for side, samples in (('before', b), ('after', a)):
                close(row[f'{side}_max_rel_l2'], max(s['single_fft_relL2'] for s in samples),
                      f'{label}: summary versus raw relative L2')
        rows.append(row)
    require(seen == MATRIX, f'{label}: expected all 28 paired summary configurations')
    rows.sort(key=lambda r: (r['N'], GROUPS.index((r['precision'], r['placement']))))
    records = {name: file_record(path) for name, path in zip(('summary', 'metadata', 'validation'), paths)}
    if data_path:
        records['trials'] = file_record(data_path)
    return {'label': label, 'rows': rows, 'meta': meta, 'identity': identity,
            'validation_rows': len(validation), 'validation_maxima': errors,
            'validation_scope': validation_scope, 'inputs': records}


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def aggregates(run, threshold):
    result = []
    for group in (None, *GROUPS):
        rows = [r for r in run['rows'] if group is None or (r['precision'], r['placement']) == group]
        for metric in ('compute', 'e2e'):
            ratios = [r[f'{metric}_speedup'] for r in rows]
            spans = [r[f'{side}_{metric}_span_pct'] for r in rows for side in ('before', 'after')
                     if f'{side}_{metric}_span_pct' in r]
            result.append({'gpu': run['label'], 'group': 'all' if group is None else ' '.join(group),
                           'metric': metric, 'configurations': len(rows), 'geomean_speedup': geomean(ratios),
                           'min_speedup': min(ratios), 'max_speedup': max(ratios),
                           'latency_change_threshold_pct': threshold,
                           'within_threshold_count': sum(abs(r[f'{metric}_delta_pct']) <= threshold for r in rows),
                           'trial_ranges_overlap_count': sum(r.get(f'{metric}_trial_ranges_overlap', False) for r in rows)
                               if spans else '',
                           'max_trial_span_pct': max(spans) if spans else '',
                           'before_max_rel_l2': max(r['before_max_rel_l2'] for r in rows),
                           'after_max_rel_l2': max(r['after_max_rel_l2'] for r in rows)})
    return result


def table_lines(run, metric):
    header = ['N']
    for i, (p, l) in enumerate(GROUPS):
        if i:
            header.append('')
        title = p.upper() + ' ' + l
        header += [title + ' before (us)', 'after (us)', 'speedup']
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---:'] * len(header)) + ' |']
    by_key = {(r['N'], r['precision'], r['placement']): r for r in run['rows']}
    for n in SIZES:
        cells = [str(n)]
        for i, (p, l) in enumerate(GROUPS):
            if i:
                cells.append('')
            r = by_key[n, p, l]
            cells += [f'{r[f"before_{metric}_us"]:.4f}', f'{r[f"after_{metric}_us"]:.4f}',
                      f'{r[f"{metric}_speedup"]:.3f}x']
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def markdown(runs, aggregate, threshold):
    lines = ['# ThunderFFT Table 3 configuration comparison', '',
             'These are measured before/after results for ThunderFFT alone. They do not replace paper reference values or constitute a full paper Table 3 collection.', '',
             f'Baseline commit: `{BASELINE}`. Times are in **microseconds (us)** for the configured batch, not divided by the number of FFTs. Batch = **65,536**, inside repeats R = **1,000**, **3 timing trials**. Each cell is the median of the three native trials; speedup = before / after.', '',
             'Compute time is the differential repeated-kernel estimate `(t_2R - t_R) / R`. Register/shared placement follows the Table 3 native benchmark. Single-pass out-of-place end-to-end CUDA-event latency is reported separately: each trial averages 10 single-pass launches, then the report takes the median of three trials. It excludes host initialization, twiddle setup and host/device transfers. Repeated outputs are not accuracy evidence.', '',
             'Selected changes reuse the existing twiddle lookup tables for FP16 N128/512/1024 and TF32 N128/256, and use warp synchronization in benchmark wrappers for the complete single-warp N64/N512 cases and FP16 N128. TF32 N128 retains block synchronization because the measured compiler output is faster with that synchronization choice. The FFT stages, tensor-core operations and precision categories are unchanged. Public global wrapper barriers remain unchanged. Host lookup coefficients can differ in their last bits from device sin/cos, so bitwise-identical results are not assumed.', '',
             'Lookup loads and complex multiplication remain in each FFT core at source level, including single-pass kernels. The compiler can hoist invariant coefficient work, as it can for the baseline; this report does not claim each lookup executes on every repeated iteration.', '',
             'Configurations without a selected operation or synchronization change (FP16 N256, TF32 N1024, and all N2048/N4096 cases) serve as controls. Their observed movements are retained and are not evidence of optimization of those configurations.', '',
             f'Variation uses the observed minimum and maximum of only three trials; it is not a confidence interval. The ±{threshold:g}% latency-change count is a descriptive small-change band, not a statistical significance test. Overlapping trial ranges are reported without declaring a speedup robust. See `before-after.csv` for per-configuration spans and observed ratio bounds.', '']
    for run in runs:
        label, ident = run['label'], run['identity']
        lines += [f'## {label}: {ident["name"]}', '',
                  f'GPU UUID: `{ident["uuid"]}`; compute capability {ident["compute_capability"]}; native target SM{run["meta"]["settings"]["arch"]}. `{ident["native_versions"]}`.', '',
                  'Before and after were built and measured with the same compiler and CUDA/cuFFT versions on this GPU. Different GPUs may use different toolkit versions; their absolute latencies are not a controlled cross-GPU comparison.', '',
                  '### Core compute median', '', *table_lines(run, 'compute'), '',
                  '### Single-pass out-of-place end-to-end median', '', *table_lines(run, 'e2e'), '',
                  '### Aggregates and observed variation', '',
                  '| Group | Metric | Geomean speedup | Min–max speedup | Small-change cases | Overlapping trial ranges | Largest trial span / median |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for r in aggregate:
            if r['gpu'] != label:
                continue
            overlap = f'{r["trial_ranges_overlap_count"]}/{r["configurations"]}' if r['trial_ranges_overlap_count'] != '' else 'unavailable'
            span = f'{r["max_trial_span_pct"]:.2f}%' if r['max_trial_span_pct'] != '' else 'unavailable'
            lines.append(f'| {r["group"]} | {r["metric"]} | {r["geomean_speedup"]:.3f}x | {r["min_speedup"]:.3f}–{r["max_speedup"]:.3f}x | {r["within_threshold_count"]}/{r["configurations"]} | {overlap} | {span} |')
        lines += ['', '### Slower single-pass medians', '']
        slower = sorted((r for r in run['rows'] if r['e2e_delta_pct'] > 0),
                        key=lambda r: r['e2e_delta_pct'], reverse=True)
        if slower:
            lines += ['The following cases have a higher single-pass median after the change. These are observed regressions in this run; the three-trial ranges and small-change caveat still apply. Core-compute improvements do not imply a faster single-pass kernel for every case.', '',
                      '| N | Precision | Placement | Before (us) | After (us) | Latency change | Trial ranges overlap |',
                      '| ---: | --- | --- | ---: | ---: | ---: | --- |']
            for r in slower:
                overlap = ('yes' if r['e2e_trial_ranges_overlap'] else 'no') if 'e2e_trial_ranges_overlap' in r else 'unavailable'
                lines.append(f'| {r["N"]} | {r["precision"].upper()} | {r["placement"]} | {r["before_e2e_us"]:.4f} | {r["after_e2e_us"]:.4f} | +{r["e2e_delta_pct"]:.2f}% | {overlap} |')
        else:
            lines += ['No single-pass median is slower in this measured run; this does not establish a general performance guarantee.']
        lines += ['', '### Correctness', '',
                  f'Candidate validation: **{run["validation_rows"]} rows passed**, including both directions and direct/public APIs at all seven sizes. The native single-FFT FP64 relative-L2 gates passed for all 56 before/after executions; the limit is 0.005.', '',
                  '| Precision | Before native maximum relative L2 | After native maximum relative L2 | After validation maximum relative L2 |',
                  '| --- | ---: | ---: | ---: |']
        for p in ('fp16', 'tf32'):
            selected = [r for r in run['rows'] if r['precision'] == p]
            lines.append(f'| {p.upper()} | {max(r["before_max_rel_l2"] for r in selected):.8g} | {max(r["after_max_rel_l2"] for r in selected):.8g} | {run["validation_maxima"][p]:.8g} |')
        scope = run['validation_scope']
        if scope['sanitizers'] == 'required':
            lines += ['', 'Sanitizers: **14 required checks passed**: memcheck and synccheck each covered 88 quick validator cases, and racecheck covered 12 real benchmark wrappers (N64/128/512 × FP16/TF32 × register/shared, R32, one trial, batch32). Every sanitizer recorded an explicit zero-error summary; racecheck also required zero warnings. Instrumented timings are excluded from the performance tables.']
            unavailable = scope.get('instrumented_timing_unavailable', [])
            if unavailable:
                lines += ['', f'For **{len(unavailable)} racecheck case(s)**, the native return code remains **1**: ' + ', '.join('`' + name + '`' for name in unavailable) + '. Each stopped only at the exact differential-timing check after both R=1 FP64 gates and the repeated-wrapper launches, with zero hazards and header-only CSV output. Their instrumented timing is **unavailable**; no completed benchmark row is claimed. All other validation commands returned zero. Ordinary paired performance measurements still require return code zero and complete timing rows.']
            else:
                lines += ['', 'All required validation commands returned zero.']
        elif scope['sanitizers'] == 'not_requested_on_this_architecture':
            lines += ['', 'Sanitizers were **not requested for this GPU run**. The 500-case full numerical suite passed; this does not imply sanitizer coverage on this GPU.']
        else:
            lines += ['', 'This v1 input records a quick numerical validation gate. Sanitizer execution is **not recorded** in its metadata; no sanitizer pass is claimed.']
        if scope['schema'] == 'v2':
            lines += ['', 'Full numerical inputs: Gaussian RMS1, offset impulse, complex tone and constant; both directions and direct/public paths are covered.']
        lines += ['', 'Source and binary hashes, input paths and file digests are retained in `report-inputs.json`. The validation maxima and native maxima cover different input/batch/path sets and should not be interpreted as identical experiments.', '']
    return '\n'.join(lines)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs=4, action='append', required=True,
                        metavar=('LABEL', 'SUMMARY_CSV', 'METADATA_JSON', 'VALIDATION_CSV'))
    parser.add_argument('--data', nargs=2, action='append', default=[], metavar=('LABEL', 'PAIRED_DATA_CSV'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path,
                        default=Path(__file__).parent / 'combined-audit/source-and-sass-audit.json',
                        help='Selected-source audit whose candidate hashes must match every input')
    parser.add_argument('--small-change-pct', type=float, default=5.0)
    args = parser.parse_args(argv)
    require(math.isfinite(args.small_change_pct) and 0 <= args.small_change_pct <= 50, 'Invalid small-change band')
    labels = [r[0] for r in args.input]
    require(len(labels) == len(set(labels)), 'Duplicate input label')
    overrides = dict(args.data)
    require(len(overrides) == len(args.data) and set(overrides) <= set(labels), 'Duplicate or unknown --data label')
    output = args.output.resolve()
    require(not any((output / name).exists() for name in OUTPUT_NAMES), 'Output already contains a report; choose a fresh directory')
    runs = [load_input(spec, overrides.get(spec[0]), args.small_change_pct) for spec in args.input]
    audit_path = args.source_audit.resolve()
    audit = json.loads(audit_path.read_text())
    require(audit['git_head'] == BASELINE, 'Source audit baseline differs')
    for run in runs:
        hashes = run['meta']['source_sha256']['after']
        require(all(hashes.get(path) == digest for path, digest in audit['candidate_source_hashes'].items()),
                f'{run["label"]}: candidate source differs from the selected-source audit')
    for side in ('before', 'after'):
        hashes = [fingerprint(run['meta']['source_sha256'][side]) for run in runs]
        require(len(set(hashes)) == 1, f'{side} source snapshots differ between supplied GPU runs')
    rows = [row for run in runs for row in run['rows']]
    aggregate = [row for run in runs for row in aggregates(run, args.small_change_pct)]
    provenance = {'schema_version': 1, 'baseline_head': BASELINE, 'full_paper_comparison': False,
                  'gpu_or_network_commands_executed': False, 'small_change_pct': args.small_change_pct,
                  'source_audit': file_record(audit_path),
                  'inputs': {run['label']: run['inputs'] for run in runs},
                  'validation_scope': {run['label']: run['validation_scope'] for run in runs},
                  'measurements': {run['label']: {k: run['meta'][k] for k in
                      ('native_identity', 'settings', 'source_sha256', 'builds', 'candidate_validation', 'order')}
                      for run in runs},
                  'report_builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'before-after.csv', rows)
    write_csv(output / 'aggregate.csv', aggregate)
    (output / 'before-after.md').write_text(markdown(runs, aggregate, args.small_change_pct))
    (output / 'report-inputs.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {len(rows)} measured configurations across {len(runs)} GPU run(s) to {output}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f'Report input rejected: {error}', file=sys.stderr)
        raise SystemExit(2)
