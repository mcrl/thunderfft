#!/usr/bin/env python3
import argparse
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/chaewon/fft')
LAYOUT = ROOT / 'thunderfft/include/thunderfft/thunderfft_layout_tuned.h'
BUILD_DIR = ROOT / 'thunderfft/build'
BIN = BUILD_DIR / 'sample/conv1d/thff_conv1d_fullfft4096'


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout


def patch_block(text: str, n: int, pad_period: int, pad: int) -> str:
    pat = re.compile(
        rf"(template <int BPB>\nstruct bench_layout<half, {n}, BPB> \{{\n)([^}}]*?)(\n\}};)",
        re.MULTILINE | re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        raise RuntimeError(f'Cannot find bench_layout<half, {n}, BPB> block')
    body = (
        f"    using L_in = layout_t<{n}, BPB, 1, {n}, {pad_period}, {pad}, false>;\n"
        f"    using L_out = layout_t<{n}, BPB, 1, {n}, {pad_period}, {pad}, false>;"
    )
    return text[: m.start()] + m.group(1) + body + m.group(3) + text[m.end() :]


def parse_metrics(out: str):
    l2 = None
    ms = None
    m_l2 = re.search(r"raw/N\): L2_rel=([0-9eE+\-.]+)", out)
    if m_l2:
        l2 = float(m_l2.group(1))
    m_ms = re.search(r"time: ([0-9eE+\-.]+) ms", out)
    if m_ms:
        ms = float(m_ms.group(1))
    return l2, ms


def bench_one(n: int, k: int, b: int, d: int, device: int):
    cmd = [
        str(BIN), str(n), str(k), str(device), str(b), str(d),
        '2', '20', '1', '1', '1', '1'
    ]
    rc, out = run(cmd)
    if rc != 0:
        return None, None, rc, out
    l2, ms = parse_metrics(out)
    return l2, ms, rc, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=9)
    ap.add_argument('--b', type=int, default=64)
    ap.add_argument('--d', type=int, default=768)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--out', type=Path, default=ROOT / 'Table5_precision/layout_sweep_conv1d.csv')
    ap.add_argument('--keep-best', action='store_true')
    args = ap.parse_args()

    original = LAYOUT.read_text()

    c256 = [(64,1), (128,1), (256,1), (64,2), (128,2), (256,2)]
    c1024 = [(16,1), (32,1), (64,1), (128,1), (256,1), (128,2)]

    rows = []
    best = None

    try:
        for (p256, pd256), (p1024, pd1024) in itertools.product(c256, c1024):
            txt = original
            txt = patch_block(txt, 256, p256, pd256)
            txt = patch_block(txt, 1024, p1024, pd1024)
            LAYOUT.write_text(txt)

            rc, bout = run([
                'cmake', '--build', str(BUILD_DIR), '--target', 'thff_conv1d_fullfft4096', '-j', '8'
            ])
            if rc != 0:
                rows.append({
                    'p256': p256, 'pad256': pd256, 'p1024': p1024, 'pad1024': pd1024,
                    'n256_l2': '', 'n256_ms': '', 'n1024_l2': '', 'n1024_ms': '',
                    'status': 'build_fail'
                })
                continue

            l2_256, ms_256, rc1, out1 = bench_one(256, args.k, args.b, args.d, args.device)
            l2_1024, ms_1024, rc2, out2 = bench_one(1024, args.k, args.b, args.d, args.device)

            status = 'ok' if (rc1 == 0 and rc2 == 0 and l2_256 is not None and l2_1024 is not None) else 'run_fail'
            row = {
                'p256': p256, 'pad256': pd256, 'p1024': p1024, 'pad1024': pd1024,
                'n256_l2': l2_256 if l2_256 is not None else '',
                'n256_ms': ms_256 if ms_256 is not None else '',
                'n1024_l2': l2_1024 if l2_1024 is not None else '',
                'n1024_ms': ms_1024 if ms_1024 is not None else '',
                'status': status,
            }
            rows.append(row)

            if status == 'ok' and l2_256 < 1e-2 and l2_1024 < 1e-2:
                score = ms_256 + ms_1024
                if best is None or score < best[0]:
                    best = (score, p256, pd256, p1024, pd1024)

            print(f"[{status}] 256=({p256},{pd256}) 1024=({p1024},{pd1024}) "
                  f"l2=({l2_256},{l2_1024}) ms=({ms_256},{ms_1024})")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'p256','pad256','p1024','pad1024','n256_l2','n256_ms','n1024_l2','n1024_ms','status'
            ])
            w.writeheader()
            w.writerows(rows)

        if args.keep_best and best is not None:
            _, p256, pd256, p1024, pd1024 = best
            txt = original
            txt = patch_block(txt, 256, p256, pd256)
            txt = patch_block(txt, 1024, p1024, pd1024)
            LAYOUT.write_text(txt)
            print(f"Kept best layout: 256=({p256},{pd256}), 1024=({p1024},{pd1024})")
        else:
            LAYOUT.write_text(original)
            print('Restored original layout file.')

        print(f'CSV written: {args.out}')

    except Exception as e:
        LAYOUT.write_text(original)
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
