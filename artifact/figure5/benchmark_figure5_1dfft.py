#!/usr/bin/env python3
import argparse
import csv
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
        r"ThunderFFT full FFT-Conv \(N=4096, include_filter_fft=[01], filter_fft_mode=[01]\) time:\s*([0-9.]+)\s*ms",
        text,
    )
    if not m:
        raise RuntimeError(f"failed to parse ours output:\n{text}")
    return float(m.group(1))


def run_ours(ours_bin, k, device, b, d, warmup, runs, include_filter_fft, filter_fft_mode, real_input, real_filter):
    cmd = [
        ours_bin,
        str(k),
        str(device),
        str(b),
        str(d),
        str(warmup),
        str(runs),
        str(include_filter_fft),
        "0",  # validate off for sweep speed
        str(filter_fft_mode),
        str(real_input),
        str(real_filter),
    ]
    out = subprocess.check_output(cmd, text=True)
    return parse_ours_ms(out)


def main():
    script_path = Path(__file__).resolve()
    fig5_dir = script_path.parent
    repo_root = fig5_dir.parent  # <repo>/fft
    default_ours_bin = repo_root / "thunderfft" / "build" / "sample" / "conv1d" / "thff_conv1d_fullfft4096"
    default_csv = fig5_dir / "compare_fft4096_realflash_vs_ours_complex_B64_D768_K3_63_step2.csv"

    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4096)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--k-start", type=int, default=3)
    p.add_argument("--k-end", type=int, default=1023)
    p.add_argument("--k-step", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--flash-root", type=str, required=True,
                   help="Path to external FlashFFTConv repo root containing python package `flashfftconv`")
    p.add_argument("--ours-bin", type=str, default=str(default_ours_bin))
    p.add_argument("--ours-include-filter-fft", type=int, default=1)
    p.add_argument("--ours-filter-fft-mode", type=int, default=1)
    p.add_argument("--ours-real-filter", type=int, default=1)
    p.add_argument("--csv", type=str, default=str(default_csv))
    args = p.parse_args()

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
            f"  cmake -S {repo_root / 'thunderfft'} -B {repo_root / 'thunderfft' / 'build'}\n"
            f"  cmake --build {repo_root / 'thunderfft' / 'build'} --target thff_conv1d_fullfft4096 -j"
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
            k,
            args.device,
            args.B,
            args.D,
            args.warmup,
            args.runs,
            args.ours_include_filter_fft,
            args.ours_filter_fft_mode,
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
