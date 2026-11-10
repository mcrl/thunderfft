#!/usr/bin/env python3
"""Scratch same-GPU before/after protocol; creates no paper collection files."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact.progress import report
from artifact.table3.measure import RAW_FIELDS, gpu_name, gpu_uuid, parse_raw
from artifact.table3.paired_validation import run_validation

BASE_HEAD = '00bc845d0f3119fa405517cc601126bdbd9b3351'
SIZES = (64, 128, 256, 512, 1024, 2048, 4096)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows, fields):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_command(binary, n, precision, placement):
    # paper_fft.cu positional ABI: N precision placement R trials batch.
    return [str(binary), str(n), precision, placement, '1000', '3', '65536']


def source_hashes(root):
    return {str(p.relative_to(root)): sha256(p)
            for folder in ('include', 'sample/bench', 'sample/validation')
            for p in sorted((root / folder).rglob('*'))
            if p.is_file() and p.suffix in ('.h', '.cuh', '.cu', '.txt')}


def capture(command):
    p = subprocess.run(command, text=True, capture_output=True, timeout=20)
    return {'command': command, 'returncode': p.returncode,
            'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}


def build(root, build_dir, output, label, args):
    cmake = os.environ.get('THUNDERFFT_CMAKE') or shutil.which('cmake')
    nvcc = os.environ.get('CUDACXX') or os.environ.get('NVCC')
    if not nvcc and os.environ.get('CUDA_HOME'):
        nvcc = str(Path(os.environ['CUDA_HOME']) / 'bin/nvcc')
    nvcc = nvcc or shutil.which('nvcc')
    if not cmake or not nvcc:
        raise RuntimeError('CMake and nvcc are required')
    command = [cmake, '-S', str(ROOT / 'artifact/table3/paired-build'),
               '-B', str(build_dir), '-DCMAKE_BUILD_TYPE=Release',
               '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCMAKE_CUDA_FLAGS=',
               '-DTHUNDERFFT_ARCHS=' + args.arch, '-DCMAKE_CUDA_COMPILER=' + nvcc,
               '-DCUDAToolkit_ROOT=' + str(Path(nvcc).resolve().parents[1]),
               '-DTHFF_SOURCE_ROOT=' + str(root),
               '-DTHFF_BUILD_VALIDATION=' + ('ON' if label == 'after' else 'OFF')]
    targets = ['thff_paper_fft'] + (['thff_validate'] if label == 'after' else [])
    commands = [command, [cmake, '--build', str(build_dir), '--target',
                          *targets, '-j', str(args.jobs)]]
    for name, cmd in zip(('configure', 'build'), commands):
        with (output / (label + '-' + name + '.log')).open('w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=60 if name == 'configure' else 360)
    binary = build_dir / 'thff_paper_fft'
    return binary, {'commands': commands, 'binary_sha256': sha256(binary),
                    'compile_commands': json.loads((build_dir / 'compile_commands.json').read_text()),
                    'nvcc': capture([nvcc, '--version'])}


def summarize(rows):
    groups = {}
    for row in rows:
        key = row['N'], row['precision'], row['placement']
        groups.setdefault(key, {}).setdefault(row['variant'], []).append(row)
    result = []
    for (n, precision, placement), variants in sorted(
            groups.items(), key=lambda item: (int(item[0][0]), *item[0][1:])):
        if set(variants) != {'before', 'after'}:
            continue
        before, after = variants['before'], variants['after']
        b = statistics.median(float(r['compute_us']) for r in before)
        a = statistics.median(float(r['compute_us']) for r in after)
        be = statistics.median(float(r['e2e_out_of_place_us']) for r in before)
        ae = statistics.median(float(r['e2e_out_of_place_us']) for r in after)
        result.append({'N': n, 'precision': precision, 'placement': placement,
                       'before_compute_us': b, 'after_compute_us': a,
                       'compute_speedup': b/a, 'compute_delta_pct': 100*(a/b-1),
                       'before_e2e_us': be, 'after_e2e_us': ae, 'e2e_speedup': be/ae,
                       'before_max_rel_l2': max(float(r['single_fft_relL2']) for r in before),
                       'after_max_rel_l2': max(float(r['single_fft_relL2']) for r in after)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--arch', default=os.environ.get('THUNDERFFT_ARCHS', '80'))
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--sanitizers', choices=('auto', 'all', 'none'), default='auto',
                        help='auto: full correctness on all GPUs and additional sanitizers on H100')
    parser.add_argument('--build-only', action='store_true', help='compile both snapshots without any GPU command')
    args = parser.parse_args()
    if args.arch not in ('80', '90', '100') or not 1 <= args.jobs <= 32:
        parser.error('requires arch 80/90/100 and 1..32 build jobs')
    args.output = args.output.resolve()
    args.build_dir = args.build_dir.resolve()
    if not args.build_only and any((args.output/name).exists() for name in
                                   ('paired-metadata.json','paired-data.csv','paired-summary.csv')):
        raise RuntimeError('paired output already contains evidence; select a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'raw').mkdir(exist_ok=True)
    evidence = json.loads((ROOT / 'paired-baseline-source.json').read_text())
    if evidence['source_head'] != BASE_HEAD:
        raise RuntimeError('baseline source revision does not match the preserved reference')
    baseline = ROOT / 'paired-baseline'
    if any(sha256(baseline/name) != digest for name,digest in evidence['files'].items()):
        raise RuntimeError('preserved baseline source hash mismatch')
    if sha256(baseline/'sample/bench/paper_fft.cu') != sha256(ROOT/'sample/bench/paper_fft.cu'):
        raise RuntimeError('before/after timing adapter differs; preserve matched timing and input protocol')
    roots = {'before': baseline, 'after': ROOT}
    meta = {'schema_version': 1, 'evaluation': 'paired_thunderfft_optimization',
            'scope': 'Before/after ThunderFFT on one GPU; not a paper Table 3 collection part',
            'status': 'building', 'started_at': now(), 'baseline_head': BASE_HEAD,
            'settings': {'batch': 65536, 'inside_repeats': 1000, 'trials': 3,
                         'N': list(SIZES), 'precisions': ['fp16','tf32'],
                         'placements': ['register','shared'], 'arch': args.arch},
            'correctness': 'Native R=1 cuFFT FP64 gate, relL2 <= 0.005; repeated outputs are not correctness evidence',
            'order': 'Before then after for even configuration indices; after then before for odd indices',
            'source_sha256': {label:source_hashes(root) for label,root in roots.items()},
            'builds': {}, 'commands': [], 'completed_native_runs': 0, 'full_paper_comparison': False}
    meta_path = args.output / 'paired-metadata.json'
    atomic_json(meta_path, meta)
    rows = []
    fields = ('variant','gpu','gpu_uuid',*RAW_FIELDS,'raw_csv')
    started = time.monotonic()
    try:
        binaries = {}
        for label,root in roots.items():
            report(0,56,detail=label+' native build',status='building')
            binaries[label],meta['builds'][label] = build(root,args.build_dir/('paired_'+label),args.output,label,args)
            atomic_json(meta_path,meta)
        if args.build_only:
            meta.update(status='built_only',finished_at=now(),gpu_commands_run=False,elapsed_seconds=time.monotonic()-started)
            atomic_json(meta_path,meta)
            print('Both snapshots compiled; no GPU command was executed.',flush=True)
            return 0
        meta.update(status='measuring',gpu_commands_run=True)
        inventory = ['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit','--format=csv,noheader']
        meta['gpu_inventory_before'] = capture(inventory)
        def save_validation(evidence):
            meta['candidate_validation'] = evidence
            atomic_json(args.output/'after-validation-evidence.json', evidence)
            atomic_json(meta_path,meta)
        run_validation(args.build_dir/'paired_after/thff_validate', binaries['after'], args.output,
                       sanitizers=args.sanitizers == 'all' or (args.sanitizers == 'auto' and args.arch == '90'),
                       save=save_validation, report=report)
        identity = None
        jobs = [(n,p,l) for n in SIZES for p in ('fp16','tf32') for l in ('register','shared')]
        for index,(n,precision,placement) in enumerate(jobs):
            order = ('before','after') if index%2==0 else ('after','before')
            for label in order:
                stem = f'{label}-n{n}-{precision}-{placement}'
                command = native_command(binaries[label], n, precision, placement)
                raw = args.output/'raw'/(stem+'.csv')
                log = args.output/'raw'/(stem+'.log')
                report(meta['completed_native_runs'],56,detail=f'{label} N={n} {precision} {placement}')
                entry = {'variant':label,'command':command,'started_at':now()}
                meta['commands'].append(entry)
                atomic_json(meta_path,meta)
                with raw.open('w') as out,log.open('w') as err:
                    run = subprocess.run(command,stdout=out,stderr=err,timeout=180)
                entry.update(returncode=run.returncode,finished_at=now(),raw_csv=str(raw.relative_to(args.output)))
                if run.returncode:
                    raise RuntimeError(f'{stem} failed: return code {run.returncode}; see raw log')
                text = log.read_text()
                gpu,name,capability = gpu_name(text)
                uuid = gpu_uuid(text)
                versions = next(line for line in text.splitlines() if line.startswith('compiled_CUDART='))
                observed = {'gpu':gpu,'name':name,'compute_capability':capability,'uuid':uuid,'native_versions':versions}
                if not uuid:
                    raise RuntimeError('native executable did not report its GPU UUID')
                identity = identity or observed
                if observed != identity:
                    raise RuntimeError('before/after GPU identity or CUDA/cuFFT versions changed')
                meta['native_identity'] = identity
                text = raw.read_text()
                parse_raw(text,n=n,precision=precision,placement=placement,batch=65536,repeats=1000,trials=3,gpu=gpu,raw_csv=str(raw.relative_to(args.output)))
                rows.extend({'variant':label,'gpu':gpu,'gpu_uuid':uuid,**r,'raw_csv':str(raw.relative_to(args.output))} for r in csv.DictReader(text.splitlines()))
                meta['completed_native_runs'] += 1
                write_csv(args.output/'paired-data.csv',rows,fields)
                summary = summarize(rows)
                if summary:
                    write_csv(args.output/'paired-summary.csv',summary,tuple(summary[0]))
                atomic_json(meta_path,meta)
        summary = summarize(rows)
        if meta['completed_native_runs']!=56 or len(rows)!=168 or len(summary)!=28:
            raise RuntimeError('incomplete paired configuration matrix')
        meta.update(status='completed',finished_at=now(),elapsed_seconds=time.monotonic()-started,passed_single_fft_gates=56,complete_paired_configurations=28)
        meta['gpu_inventory_after'] = capture(inventory)
        atomic_json(meta_path,meta)
        report(56,56,detail='28 paired configurations retained',status='completed')
        print('Completed 28 paired configurations; see paired-summary.csv.',flush=True)
        return 0
    except BaseException as error:
        meta.update(status='failed',finished_at=now(),elapsed_seconds=time.monotonic()-started,error_type=type(error).__name__,error=str(error))
        atomic_json(meta_path,meta)
        report(meta['completed_native_runs'],56,detail=str(error),status='failed')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
