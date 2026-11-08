# Figure 2: A100 register-resident FFT kernel timing

Plots kernel compute time for N=64..4096 with inputs in registers (paper p. 9):
ThunderFFT FP16 vs cuFFTDx FP16 and ThunderFFT TF32 vs cuFFTDx FP32, on an A100.
Figure 2 runs no measurements of its own; it is derived from the Table 3 A100
register results. Selecting `figure2` runs `table3` first and plots its result.

## Run

Run locally on an A100 80GB PCIe GPU.

```bash
./run.sh figure2                    # smoke (runs table3, then plots)
./run.sh figure2 --profile paper    # paper (runs table3, then plots)
```

In a collection, Figure 2 is produced only by `--assemble` from the collected
Table 3 parts (missing points stay blank):

```bash
THUNDERFFT_TABLE3_PLACEMENTS=register \
  ./run.sh table3 --profile paper --collection f2 --part register
./run.sh figure2 --assemble --collection f2 --profile paper
```

To replot from an existing completed Table 3 result without measuring again:

```bash
python3 artifact/figure2/plot.py --source-dir result/table3/<run_id> \
  --output-dir result/figure2/<new_id> --profile paper
```

## Options

Figure 2 reads no variables of its own. Table 3 variables control the source
measurement; the non-collection run needs all 7 sizes and both precisions, so
only these two are useful here:

| Environment variable | Values | Default (smoke / paper) |
|---|---|---|
| `THUNDERFFT_TABLE3_PLACEMENTS` | `register` (only placement Figure 2 uses), `shared`, or both | `register,shared` |
| `THUNDERFFT_TABLE3_LIBRARIES` | `ThunderFFT` (no cuFFTDx series) or `ThunderFFT,cuFFTDx` | both (when MathDX is configured) |

## Output

Written to `result/figure2/<run_id>/` (the Table 3 source is in `result/table3/<run_id>/`):

- `figure.pdf`: the plot (median points, min-max trial bars, log time axis)
- `summary.csv`: median/min/max time per series and size, selected cuFFTDx candidate, speedup over cuFFTDx
- `data.csv`: the Table 3 rows used for the plot
- `metadata.json`: source run, selection rule and hashes
- `status.json`, `run.log`: runner status and console log

Assembly writes `figure.pdf` (only if at least one point was collected),
`summary.csv`, `data.csv` and `metadata.json` to
`result/collections/<name>/assembled/figure2/`.
