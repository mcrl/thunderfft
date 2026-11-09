"""Declared cuDNN subsets for quick Figure 4/5 reproduction.

The limits precede the first 3x (Figure 5) / 10x (Figure 4) slowdown versus
the fastest measured FFT library in the retained 2026-11-09 A100 sweep.
Filters are batched in one invocation, preserving input-generation reuse.
"""

MODES = ("fast", "full")
ENVIRONMENT = "THUNDERFFT_CUDNN_MODE"
MAX_FILTER = {"figure5": 25, "figure4": 11}


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"{ENVIRONMENT} must be fast or full")
    return mode


def selected_filters(evaluation, filters, library, mode):
    validate_mode(mode)
    return tuple(size for size in filters
                 if mode == "full" or library != "cuDNN" or size <= MAX_FILTER[evaluation])


def skipped_cells(evaluation, filters, libraries, mode):
    validate_mode(mode)
    if mode == "full" or "cuDNN" not in libraries:
        return []
    limit = MAX_FILTER[evaluation]
    return [{"library": "cuDNN", "filter": size,
             "reason": f"cuDNN skipped by fast mode (filter > {limit}); use --full"}
            for size in sorted(set(filters)) if size > limit]
