#!/usr/bin/env python3
"""Copy an allowlisted completed paired run and exact sources; no GPU/network."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target, expected=None):
    digest = sha(source)
    assert expected is None or digest == expected, f'source hash mismatch: {source}'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == digest, f'refusing to replace different evidence: {target}'
    else:
        shutil.copyfile(source, target)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', required=True, type=Path)
    parser.add_argument('--label', required=True, choices=('a100', 'h100', 'b200'))
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--main-repository', required=True, type=Path)
    args = parser.parse_args()
    meta = json.loads((args.measurement/'paired-metadata.json').read_text())
    audit = json.loads((args.audit/'source-and-sass-audit.json').read_text())
    assert meta['status'] == 'completed' and meta['completed_native_runs'] == 56
    assert meta['baseline_head'] == audit['git_head'] == '00bc845d0f3119fa405517cc601126bdbd9b3351'
    assert all(meta['source_sha256']['after'][name] == digest
               for name, digest in audit['candidate_source_hashes'].items())
    if args.label == 'a100':
        assert meta['builds']['after']['binary_sha256'] == audit['actual_paired_binary_sha256']
        assert meta['builds']['before']['binary_sha256'] == audit['actual_paired_before_sha256']
    archive = args.archive
    manifest = archive/'source-manifest.json'
    snapshots = {}
    for side in ('before', 'after'):
        source = args.source/'paired-baseline' if side == 'before' else args.source
        target = archive/'protocol'/'paired-baseline' if side == 'before' else archive/'protocol'
        snapshots[side] = {}
        for relative, digest in meta['source_sha256'][side].items():
            if side == 'before':
                original = subprocess.check_output(['git', '-C', str(args.main_repository),
                    'show', meta['baseline_head'] + ':' + relative])
                assert hashlib.sha256(original).hexdigest() == digest, f'baseline differs from git: {relative}'
            snapshots[side][relative] = copy(source/relative, target/relative, digest)
    protocol_files = ['paired-baseline-source.json', 'artifact/progress.py',
        'artifact/table3/paired_measure.py', 'artifact/table3/paired_validation.py',
        'artifact/table3/measure.py', 'artifact/table3/cufftdx_presets.py',
        'artifact/table3/cufftdx_presets.json', 'artifact/table3/paired-build/CMakeLists.txt']
    protocol_hashes = {name: copy(args.source/name, archive/'protocol'/name) for name in protocol_files}
    assert protocol_hashes['artifact/table3/paired_validation.py'] == meta['candidate_validation']['validation_helper_sha256']
    manifest_data = {'baseline_head': meta['baseline_head'], 'source_sha256': snapshots,
                     'protocol_sha256': protocol_hashes,
                     'native_timing_adapter_unchanged': snapshots['before']['sample/bench/paper_fft.cu'] == snapshots['after']['sample/bench/paper_fft.cu']}
    if manifest.exists():
        assert json.loads(manifest.read_text()) == manifest_data, 'different source manifest'
    else:
        manifest.write_text(json.dumps(manifest_data, indent=2) + '\n')
    patch = []
    for relative, digest in snapshots['before'].items():
        if snapshots['after'][relative] != digest:
            patch.extend(difflib.unified_diff((archive/'protocol/paired-baseline'/relative).read_text().splitlines(True),
                (archive/'protocol'/relative).read_text().splitlines(True),
                fromfile='before/'+relative, tofile='after/'+relative))
    patch_path = archive/'selected-source.patch'
    patch_text = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n' for line in patch)
    if patch_path.exists():
        assert patch_path.read_text() == patch_text
    else:
        patch_path.write_text(patch_text)
    for name in ('source-and-sass-audit.json', 'static-comparison.json',
                 'lut-bounds-proof.json', 'candidate.resources.txt', 'audit.py'):
        copy(args.audit/name, archive/'audit'/name)
    allowed = []
    for name in ('paired-summary.csv', 'paired-data.csv', 'paired-metadata.json',
                 'after-validation-evidence.json', 'after-build.log', 'after-configure.log',
                 'before-build.log', 'before-configure.log'):
        allowed.append(args.measurement/name)
    for run in meta['candidate_validation']['runs']:
        name = run['name']
        assert name.startswith('after-') and '/' not in name
        allowed += [args.measurement/(name+'.log'), args.measurement/(name+'.stdout')]
        if run.get('csv'):
            allowed.append(args.measurement/run['csv'])
        if run['kind'] != 'validator':
            allowed.append(args.measurement/(name+'-sanitizer.log'))
    for run in meta['commands']:
        assert run['returncode'] == 0
        relative = Path(run['raw_csv'])
        assert relative.parent == Path('raw')
        allowed += [args.measurement/relative, args.measurement/relative.with_suffix('.log')]
    records = {}
    for source in sorted(set(allowed)):
        relative = source.relative_to(args.measurement)
        records[str(relative)] = copy(source, archive/args.label/relative)
    (archive/args.label/'copied-file-sha256.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'Archived {len(records)} exact evidence files for {args.label}; {len(snapshots["before"])} baseline and {len(snapshots["after"])} candidate source files')


if __name__ == '__main__':
    main()
