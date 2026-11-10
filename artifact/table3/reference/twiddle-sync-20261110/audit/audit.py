"""CPU-only final selected Table 3 binary audit; run only after root rebuilds."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import re
import subprocess

BASE = Path('/home/chaewon/ae-e2e-20261110/table3-opt')
OUT = Path(__file__).resolve().parent
OLD = BASE / 'combined-audit'
SOURCE = BASE / 'paired-cloud-worktree-v2'
MAIN = Path('/home/chaewon/ThunderFFT')
BUILD = BASE / 'paired-local-build'
BINARY = BUILD / 'paired_after/thff_paper_fft'
EXPECTED_HEADER = 'adf1312fd6ab85dfb85205d4217b252d630c1afe83826ad3a6d7e0b7c2b2d52e'
HEADER = 'sample/bench/baseline/thunderfft_bench.h'
CUOBJDUMP = '/usr/local/cuda-12.9/bin/cuobjdump'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_sass(path):
    chunks = re.split(r'\s*Function : (\S+)\s*', path.read_text())
    functions = {}
    for name, body in zip(chunks[1::2], chunks[2::2]):
        if 'ThunderFFT_benchmark' not in name:
            continue
        rows = []
        for line in body.splitlines():
            match = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*', line)
            if match:
                instruction = re.sub(r'\s+', ' ', match[2].strip())
                tokens = instruction.split()
                opcode = tokens[1] if tokens[0].startswith('@') else tokens[0]
                rows.append((int(match[1], 16), opcode, instruction))
        functions[name] = rows
    return functions

def parse_resources(path):
    return {name: {key: int(value) for key, value in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)', line)}
            for name, line in re.findall(r'Function (\S+):\s*\n([^\n]*)', path.read_text())
            if 'ThunderFFT_benchmark' in name}

def counts(rows):
    counter = collections.Counter(opcode.split('.')[0] for _, opcode, _ in rows)
    return {'instructions': len(rows), **{key: counter[key] for key in
            ('HMMA', 'MUFU', 'LDG', 'LDS', 'STS', 'LDL', 'STL', 'BAR', 'WARPSYNC', 'CALL', 'BRA')}}

def loop_spans(rows):
    loops = []
    for address, opcode, instruction in rows:
        target = re.search(r'0x([0-9a-f]+)', instruction) if opcode == 'BRA' else None
        if target and int(target[1], 16) < address:
            start = int(target[1], 16)
            segment = [row for row in rows if start <= row[0] <= address]
            if any(row[1].startswith('HMMA') for row in segment):
                loops.append({'start': hex(start), 'backedge': hex(address),
                              'counts': counts(segment)})
    return loops

old_equivalence = json.loads((OLD / 'paired-sass-equivalence.json').read_text())
assert all(old_equivalence['comparisons'][label]['all_identical'] for label in ('before', 'after'))
old_meta = json.loads((OLD / 'source-and-sass-audit.json').read_text())
old_paired_sha = old_equivalence['binaries']['paired_after']['sha256']
assert sha(SOURCE / HEADER) == EXPECTED_HEADER, 'Unexpected final sync helper source'
assert sha(MAIN / HEADER) == EXPECTED_HEADER, 'Main and actual paired sync helper differ'
new_binary_sha = sha(BINARY)
assert new_binary_sha != old_paired_sha, 'New paired binary has not been rebuilt yet'
assert BINARY.stat().st_mtime_ns >= (SOURCE / HEADER).stat().st_mtime_ns, 'Binary predates selected source'
# The synchronization helper alone changed since the previously audited LUTs.
for relative, previous_sha in old_meta['candidate_source_hashes'].items():
    if relative != HEADER:
        assert sha(SOURCE / relative) == previous_sha, f'Unexpected core source change: {relative}'
source_hashes = {relative: sha(SOURCE / relative) for relative in old_meta['candidate_source_hashes']}
assert all(sha(MAIN / relative) == digest for relative, digest in source_hashes.items())

for option, name in [('--dump-sass', 'candidate.sass'), ('--dump-resource-usage', 'candidate.resources.txt')]:
    with (OUT / name).open('w') as output:
        subprocess.run([CUOBJDUMP, option, str(BINARY)], stdout=output, check=True)
assert sha(BINARY) == new_binary_sha, 'Binary changed during audit'
assert all(sha(SOURCE / relative) == digest for relative, digest in source_hashes.items()), 'Source changed during audit'

old_functions = parse_sass(OLD / 'candidate.sass')
base_functions = parse_sass(OLD / 'baseline.sass')
new_functions = parse_sass(OUT / 'candidate.sass')
old_resources = parse_resources(OLD / 'candidate.resources.txt')
new_resources = parse_resources(OUT / 'candidate.resources.txt')
assert set(old_functions) == set(base_functions) == set(new_functions)
assert len(new_functions) == 56
comparison = []
for name in sorted(new_functions):
    n = int(re.search(r'Li(\d+)E', name)[1])
    precision = 'fp16' if 'I6__half' in name else 'tf32'
    intended_change = n == 128 and precision == 'tf32'
    row = {'symbol': name, 'N': n, 'precision': precision,
           'placement': 'shared' if '_smem' in name else 'register',
           'kind': 'e2e' if '_e2e' in name else 'repeat',
           'intended_sync_change': intended_change,
           'same_assembly_as_final2': new_functions[name] == old_functions[name],
           'same_static_HMMA_as_baseline': counts(new_functions[name])['HMMA'] == counts(base_functions[name])['HMMA'],
           'same_static_HMMA_as_final2': counts(new_functions[name])['HMMA'] == counts(old_functions[name])['HMMA'],
           'baseline_static_counts': counts(base_functions[name]),
           'final2_static_counts': counts(old_functions[name]),
           'final_static_counts': counts(new_functions[name]),
           'final2_resources': old_resources[name], 'final_resources': new_resources[name],
           'final2_loop_spans': loop_spans(old_functions[name]),
           'final_loop_spans': loop_spans(new_functions[name])}
    comparison.append(row)
unchanged = [row for row in comparison if not row['intended_sync_change']]
changed = [row for row in comparison if row['intended_sync_change']]
assert len(unchanged) == 52 and all(row['same_assembly_as_final2'] for row in unchanged), 'Unintended function changed'
assert len(changed) == 4
assert all(row['same_static_HMMA_as_baseline'] and row['same_static_HMMA_as_final2'] for row in comparison)
for row in changed:
    if row['kind'] == 'repeat':
        assert len(row['final_loop_spans']) == 1
        assert row['final_loop_spans'][0]['counts']['LDG'] == 0, 'N128 TF32 LUT reload remains in repeated interval'
        assert row['final_loop_spans'][0]['counts']['HMMA'] == 48

bounds = json.loads((OLD / 'lut-bounds-proof.json').read_text())
assert all(case['all_inside_N_element_table'] for case in bounds['cases'])
bounds['applies_to_final_core_sources'] = True
bounds['reason'] = 'Every LUT/core header hash is unchanged from independently reviewed source; only benchmark sync helper changed.'
(OUT / 'lut-bounds-proof.json').write_text(json.dumps(bounds, indent=2) + '\n')
git_heads = {label: subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
             for label, path in [('main', MAIN), ('paired_source', SOURCE)]}
assert git_heads['main'] == json.loads((SOURCE / 'paired-baseline-source.json').read_text())['source_head'] == '00bc845d0f3119fa405517cc601126bdbd9b3351'
report = {'git_head': git_heads['main'], 'git_head_verification': git_heads,
          'git_head_basis': 'Main working-tree base revision; selected uncommitted source is separately identified by candidate_source_hashes. Paired harness has its own committed revision.',
          'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'actual_paired_binary': str(BINARY), 'actual_paired_binary_sha256': new_binary_sha,
          'actual_paired_before_sha256': sha(BUILD / 'paired_before/thff_paper_fft'),
          'previous_paired_after_sha256': old_paired_sha,
          'previous_equivalence_evidence': str(OLD / 'paired-sass-equivalence.json'),
          'candidate_source_hashes': source_hashes,
          'selected_header_sha256': EXPECTED_HEADER,
          'architecture': 'sm_80', 'functions_audited': 56,
          'all_static_HMMA_counts_preserved': True,
          'unchanged_functions_identical_to_final2': 52,
          'n128_tf32_repeat_loop_LDG_count': 0,
          'notes': ['No GPU workloads launched by this audit.',
                    'Counts are static assembly counts; backward-branch spans are linear assembly intervals, not dynamic instruction totals.',
                    'LUT values/precision and bounds are inherited unchanged; compiler invariant handling, not source hoisting across the timing interval, removes N128 TF32 repeated loads.'],
          'changed_functions': changed}
(OUT / 'source-and-sass-audit.json').write_text(json.dumps(report, indent=2) + '\n')
(OUT / 'static-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ('actual_paired_binary_sha256', 'functions_audited', 'all_static_HMMA_counts_preserved', 'unchanged_functions_identical_to_final2', 'n128_tf32_repeat_loop_LDG_count')}, indent=2))
