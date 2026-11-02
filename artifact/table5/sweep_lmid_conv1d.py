#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path('/home/chaewon/fft')
BUILD = ROOT / 'thunderfft/build'
BIN = BUILD / 'sample/conv1d/thff_conv1d_fullfft4096'

CANDIDATES = [
    (64, 1, 16, 1),
    (64, 2, 16, 1),
    (128, 1, 16, 1),
    (64, 1, 32, 1),
    (64, 1, 64, 1),
    (128, 2, 32, 1),
]


def run(cmd):
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout


def parse(out: str):
    m_l2 = re.search(r"raw/N\): L2_rel=([0-9eE+\-.]+)", out)
    m_t = re.search(r"time: ([0-9eE+\-.]+) ms", out)
    l2 = float(m_l2.group(1)) if m_l2 else None
    t = float(m_t.group(1)) if m_t else None
    return l2, t


def bench_n(n, k, b, d, device, validate, runs):
    cmd = [str(BIN), str(n), str(k), str(device), str(b), str(d), '2', str(runs), '1', str(validate), '1', '1']
    rc, out = run(cmd)
    l2, t = parse(out)
    return rc, l2, t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=9)
    ap.add_argument('--b', type=int, default=64)
    ap.add_argument('--d', type=int, default=768)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--out', type=Path, default=ROOT / 'Table5_precision/lmid_sweep_conv1d.csv')
    args = ap.parse_args()

    rows = []
    for p256, pad256, p1024, pad1024 in CANDIDATES:
        rc, out = run([
            'cmake', '-S', str(ROOT / 'thunderfft'), '-B', str(BUILD),
            f'-DTHFF_LMID256_PAD_PERIOD={p256}',
            f'-DTHFF_LMID256_PAD={pad256}',
            f'-DTHFF_LMID1024_PAD_PERIOD={p1024}',
            f'-DTHFF_LMID1024_PAD={pad1024}',
        ])
        if rc != 0:
            rows.append({'p256': p256, 'pad256': pad256, 'p1024': p1024, 'pad1024': pad1024, 'status': 'cmake_fail'})
            continue

        rc, out = run(['cmake', '--build', str(BUILD), '--target', 'thff_conv1d_fullfft4096', '-j', '8'])
        if rc != 0:
            rows.append({'p256': p256, 'pad256': pad256, 'p1024': p1024, 'pad1024': pad1024, 'status': 'build_fail'})
            continue

        rc1, l2_256, _, out1 = bench_n(256, args.k, args.b, args.d, args.device, validate=1, runs=20)
        rc2, l2_1024, _, out2 = bench_n(1024, args.k, args.b, args.d, args.device, validate=1, runs=20)
        rc3, _, t_256, out3 = bench_n(256, args.k, args.b, args.d, args.device, validate=0, runs=100)
        rc4, _, t_1024, out4 = bench_n(1024, args.k, args.b, args.d, args.device, validate=0, runs=100)

        ok = all(r == 0 for r in (rc1, rc2, rc3, rc4)) and l2_256 is not None and l2_1024 is not None and t_256 is not None and t_1024 is not None
        status = 'ok' if ok else 'run_fail'
        row = {
            'p256': p256, 'pad256': pad256, 'p1024': p1024, 'pad1024': pad1024,
            'l2_256': l2_256 if l2_256 is not None else '',
            'l2_1024': l2_1024 if l2_1024 is not None else '',
            'ms_256': t_256 if t_256 is not None else '',
            'ms_1024': t_1024 if t_1024 is not None else '',
            'status': status,
        }
        rows.append(row)
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['p256','pad256','p1024','pad1024','l2_256','l2_1024','ms_256','ms_1024','status'])
        w.writeheader()
        w.writerows(rows)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
