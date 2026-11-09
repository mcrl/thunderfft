#!/usr/bin/env python3
"""Derive Table 7 Tensor Core throughput and utilization from timing CSVs."""

import math
import re


METHOD = "tensor_flops_per_second / hardware_peak_flops_per_second"
REFERENCE = {
    "gpu_family": "A100",
    "precision": "tf32",
    "hardware_peak_tflops": 156.0,
    "source": "Supplied paper Table 4: A100 hardware peak, TF32 dense",
    "paper_pdf_sha256": "02ab4855e16b721789c35b80a4b2488bc1e7849fa6da3f12f0bae861bb109fd9",
}

# The radix-4 kernels issue 3 stages * 16 MMA instructions per warp.
# The reconstructed radix-16 kernel uses a mixed radix-16/radix-4 N64 FFT:
# 4 butterflies * 16 MMA instructions + 16 butterflies * 1 MMA = 80 per warp.
# Each warp processes 16 FFTs; each mma.m16n8k8 counts 2*m*n*k.
# This is Tensor Core arithmetic including complex embedding, not 5*N*log2(N)
# useful FFT operations or scalar trigonometric/remapping instructions.
# In particular, the approximate paper model 32*N*log2(N)=12288 for radix-16
# is not the actual instruction count of the mixed-radix N64 implementation.
FLOPS_PER_MMA = 2 * 16 * 8 * 8
MMA_PER_WARP = {"register_branch": 48, "register_uniform": 48, "shared_stage_radix4": 48,
                "historical_e4aa1b0": 48, "historical_652808b": 48,
                "reconstructed_radix16": 80, "reconstructed_radix4_compute": 48}
# The two historical partial kernels retain 48 issued MMA instructions per R.
# Their counts describe hardware work, not correct FFTs: e4aa1b0 omits B setup;
# 652808b also omits inter-stage permutation. Its unrolled SASS has a two-R body
# plus a one-R tail, so the static instruction total must not be used as FLOPs/R.
# The unrecovered canonical radix16/radix4_compute IDs intentionally have no
# count. Experimental reconstructions must never provide counts for those rows.
FLOPS_PER_FFT = {variant: count * FLOPS_PER_MMA // 16
                 for variant, count in MMA_PER_WARP.items()}


def calculate(variant, batch, time_us, gpu_name, precision="tf32"):
    """Use the measured time for one batch; the native R divisor is already applied."""
    if variant not in FLOPS_PER_FFT:
        raise ValueError(f"Tensor Core operation count has not been verified for {variant}")
    if type(batch) is not int or batch <= 0 or batch % 16:
        raise ValueError("batch must be a positive multiple of 16")
    if isinstance(time_us, bool) or not isinstance(time_us, (int, float)) or not math.isfinite(time_us) or time_us <= 0:
        raise ValueError("time_us must be positive and finite")
    if precision.lower() != "tf32":
        raise ValueError("The Table 7 operation count and reference require TF32")
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise ValueError("native GPU name is required")
    count = FLOPS_PER_FFT[variant]
    total = count * batch
    throughput = total / time_us / 1e6
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("derived Tensor Core throughput must be positive and finite")
    result = {
        "tensor_flops_per_fft": count,
        "tensor_flops_per_batch": total,
        "achieved_tflops": throughput,
        "reference_peak_tflops": "",
        "utilization_pct": "",
        "utilization_metric": METHOD,
        "utilization_status": "pending_reference",
        "reference_source": "Hardware reference is not configured for this GPU",
    }
    if re.search(r"\bA100\b", gpu_name):
        peak = REFERENCE["hardware_peak_tflops"]
        result.update(reference_peak_tflops=peak, utilization_pct=throughput / peak * 100.0,
                      utilization_status="calculated", reference_source=REFERENCE["source"])
    # Do not clamp values above 100%: the recorded time and normalization must
    # remain visible so an inconsistent measurement or reference can be checked.
    return result
