#!/usr/bin/env python3
"""Legacy timing draft. Correctness and baseline semantics remain unresolved.

The evaluated runner intentionally refuses to use this script until the tasks
in README.md are complete. Historical CSVs are never overwritten by default.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def bench_cuda_ms(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(runs):
        fn()
    ed.record()
    torch.cuda.synchronize()
    return st.elapsed_time(ed) / runs


def parse_ours_ms(text):
    m = re.search(
        r"ThunderFFT full FFT-Conv \(N=4096, include_filter_fft=[01]\) time:\s*([0-9.eE+-]+)\s*ms",
        text,
    )
    if not m:
        raise RuntimeError(f"failed to parse ours output:\n{text}")
    return float(m.group(1))


def run_ours(ours_bin, n, k, device, b, d, warmup, runs, include_filter_fft, real_input, real_filter):
    cmd = [
        ours_bin,
        str(n),
        str(k),
        str(device),
        str(b),
        str(d),
        str(warmup),
        str(runs),
        str(include_filter_fft),
        "0",  # validate off for sweep speed
        str(real_input),
        str(real_filter),
    ]
    out = subprocess.check_output(cmd, text=True)
    return parse_ours_ms(out)


def main():
    script_path = Path(__file__).resolve()
    repo_root = Path(os.environ.get("THUNDERFFT_ROOT", script_path.parents[2]))
    build_dir = Path(os.environ.get("THUNDERFFT_BUILD_DIR", repo_root / "build"))
    result_dir = Path(os.environ.get("THUNDERFFT_RESULT_DIR", repo_root / "result" / "figure4"))
    default_ours_bin = build_dir / "sample" / "conv1d" / "thff_conv1d_fullfft4096"
    default_csv = result_dir / "legacy_timings.csv"

    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4096)
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--D", type=int, default=768)
    p.add_argument("--k-start", type=int, default=3)
    p.add_argument("--k-end", type=int, default=53)
    p.add_argument("--k-step", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--flash-root", type=str, required=True,
                   help="Path to external FlashFFTConv repo root containing python package `flashfftconv`")
    p.add_argument("--ours-bin", type=str, default=str(default_ours_bin))
    p.add_argument("--ours-include-filter-fft", type=int, default=1)
    p.add_argument("--ours-filter-fft-mode", type=int, choices=[1], default=1,
                   help="Legacy compatibility option; only 1 is supported because the current binary uses cuFFT for the filter FFT")
    p.add_argument("--ours-real-filter", type=int, default=1)
    p.add_argument("--csv", type=str, default=str(default_csv))
    args = p.parse_args()

    print("WARNING: legacy timing draft; correctness is disabled and baseline semantics are unresolved. See artifact/figure4/README.md.", file=sys.stderr)

    assert args.L == 4096, "this script assumes L=4096"
    flash_root = Path(args.flash_root).resolve()
    if not flash_root.exists():
        raise FileNotFoundError(f"flash root not found: {flash_root}")
    if str(flash_root) not in sys.path:
        sys.path.insert(0, str(flash_root))
    try:
        from flashfftconv import FlashFFTConv
        from flashfftconv.depthwise_1d import conv1dFunc
    except Exception as e:
        raise RuntimeError(
            f"failed to import flashfftconv from --flash-root={flash_root}. "
            f"Expected package path like: {flash_root}/flashfftconv"
        ) from e

    ours_bin = Path(args.ours_bin)
    if not ours_bin.exists():
        raise FileNotFoundError(
            f"ours binary not found: {ours_bin}\n"
            f"build it first, e.g.:\n"
            f"  cmake -S {repo_root} -B {repo_root / 'build'}\n"
            f"  cmake --build {repo_root / 'build'} --target thff_conv1d_fullfft4096 -j"
        )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    dtype = torch.float16
    dev = torch.device(f"cuda:{args.device}")
    fft = FlashFFTConv(args.L, dtype=dtype).to(dev)

    rows = []
    for k in range(args.k_start, args.k_end + 1, args.k_step):
        x = torch.randn(args.B, args.D, args.L, device=dev, dtype=dtype)
        kr = torch.randn(args.D, k, device=dev, dtype=dtype)

        # circular conv emulation for direct 1d paths
        xp = torch.cat([x[..., -(k - 1):], x], dim=-1) if k > 1 else x
        wk = kr.view(args.D, 1, k).contiguous()
        bias = torch.zeros(args.D, device=dev, dtype=dtype)

        def flashfft_real_fn():
            _ = fft(x, kr)

        def flashconv1d_real_fn():
            _ = conv1dFunc.apply(xp, kr, bias, 0, True)

        def cudnn_real_fn():
            _ = F.conv1d(xp, wk, bias=None, stride=1, padding=0, groups=args.D)

        t_flashfft = bench_cuda_ms(flashfft_real_fn, args.warmup, args.runs)
        t_flashconv1d = bench_cuda_ms(flashconv1d_real_fn, args.warmup, args.runs)
        t_cudnn = bench_cuda_ms(cudnn_real_fn, args.warmup, args.runs)
        t_ours = run_ours(
            str(ours_bin),
            args.L,
            k,
            args.device,
            args.B,
            args.D,
            args.warmup,
            args.runs,
            args.ours_include_filter_fft,
            1,
            args.ours_real_filter,
        )

        rows.append((k, t_flashfft, t_flashconv1d, t_cudnn, t_ours))
        print(
            f"k={k:4d} flashfft_real={t_flashfft:.6f}ms "
            f"flashconv1d_real={t_flashconv1d:.6f}ms cudnn_real={t_cudnn:.6f}ms "
            f"ours_complex_realinput={t_ours:.6f}ms"
        )

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "flashfft_real", "flashconv1d_real", "cudnn_real", "ours_complex_realinput"])
        w.writerows(rows)
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
