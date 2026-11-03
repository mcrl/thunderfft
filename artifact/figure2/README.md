# Figure 2: FFT kernel execution time

Status: ready for the **ThunderFFT-only A100/register subset**. The cuFFTDx
FP16/FP32 series remain assigned separately; this adapter does not fabricate
them or claim to reproduce the complete comparison.

Paper p. 9, Section 6.2: A100 inputs in registers; FFT sizes
64/128/256/512/1024/2048/4096; batch 65,536. The original figure compares
cuFFTDx FP16 against ThunderFFT FP16, and cuFFTDx FP32 against ThunderFFT TF32.
This subset plots the two ThunderFFT series using exactly the fresh
[Table 3](../table3/README.md) measurements, without another GPU timing sweep.

## Run

Install the plotting dependency once in the Python environment used by the
runner:

```bash
python3 -m pip install -r artifact/figure2/requirements.txt
./run.sh figure2 --profile smoke
./run.sh figure2 --profile paper
```

The root runner schedules Table 3 first and passes that run's result directory
through `THUNDERFFT_DEPENDENCY_RESULTS`. Both profiles generate `data.csv`,
`summary.csv`, `figure.pdf`, and `metadata.json` under the Figure 2 result
directory. Smoke is a short functional measurement, not the paper workload.
Paper uses the Table 3 batch of 65,536. Plotting itself uses no GPU.

To regenerate a plot from one explicitly selected completed Table 3 result,
without rerunning its measurements:

```bash
python3 artifact/figure2/plot.py \
  --source-dir result/table3/<run_id> \
  --output-dir result/figure2/<new_plot_id> --profile paper
```

The source must have a completed `table3` `status.json`, exit code 0, and the
same profile. Missing dependency files or a source without A100/register data
return 78 (pending). Failed source rows, missing configurations, duplicate
trials, inconsistent batch/repeat counts, and nonfinite data return 65. This
figure is explicitly scoped to A100; H100/B200 measurements are not relabeled.

## Data and plotting contract

The adapter requires every one of the 7 sizes × 2 precisions and the same set
of trial IDs in every configuration. It selects `gpu=A100`,
`placement=register`, `library=ThunderFFT`; other Table 3 placements are not
plotted. Every selected row must carry a passed numerical gate and finite,
positive compute/end-to-end timings and a finite, nonnegative relative L2.
The Table 3 gate defines the error tolerance; the figure preserves that result
rather than introducing a different accuracy test.

`data.csv` preserves all selected raw Table 3 columns, then adds `source_run`,
`source_result_dir`, `source_file`, source data/status SHA-256 hashes, and
`time_ms`. `summary.csv` contains the median, minimum, and maximum of the
individual trial compute times for each of the 14 configurations. It uses
`time_us = (t_2R_us - t_R_us) / R` from Table 3, not end-to-end timing.
The explicit unit conversion is `time_ms = time_us / 1000`.
The source metadata path/hash are also retained when present. Source precision
labels are retained in raw data; `fp16`/`tf32` are normalized for the summary.

The PDF uses a logarithmic millisecond axis, median points, and min–max trial
bars. TF32 is blue and FP16 is green; the legend sits above the data area at
the upper right. These statistics are an explicit artifact convention because
the draft does not specify the original trial aggregation or error bars.
Its caption identifies the profile and the missing cuFFTDx series.

`metadata.json` records scope, source paths/hashes/run ID, plotting command,
script hash, output hashes, statistics, units, and colors. The runner's
`status.json` reports whether this adapter completed; it is not evidence of
the missing cuFFTDx comparison or a complete Figure 2 reproduction.

CPU contract checks, including deliberate invalid source data, can be run with:

```bash
python3 -m unittest discover -s artifact/figure2 -p 'test_*.py' -v
```
