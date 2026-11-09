# Table 2: A100 theoretical performance model

The FP16 A100 model on paper p. 3. Four rows (ideal CUDA Core FFT and Tensor
Core radix-4/8/16) are calculated from `assumptions.json` and need no GPU. The
measured cuFFTDx row comes from the Table 3 measurement for A100, FP16,
register input, N=64. Table 3's default `preset` mode remeasures a fixed
candidate selected from an earlier validated sweep. With `--cufftdx-mode all`,
the row uses the smallest median time among candidates measured in that run.
Both modes report fresh timing trials and retain the candidate identity.

## Run

The calculated rows need no GPU; the measured row runs locally on an A100 80GB
PCIe GPU (or on a rented A100 with `--backend vast`).

```sh
./run.sh table2                          # calculated rows only; no GPU
./run.sh table3 table2 --profile paper   # on an A100: remeasure the cuFFTDx preset
./run.sh table3 table2 --profile paper --cufftdx-mode all  # run the candidate sweep

# Without a local A100
./run.sh table3 table2 --backend vast --gpu A100 --profile paper
```

Table 3 is an optional dependency. When `table3` is in the same command and
completes, its result is passed to Table 2. `./run.sh table2` alone never runs
Table 3 and omits the measured row. The row is also omitted if that Table 3 run
has no A100 cuFFTDx N=64 FP16 register measurement.

To build Table 2 from Table 3 parts already in a collection (see
[Table 3](../table3/README.md#collect-per-gpu-and-assemble)), without a GPU:

```sh
./run.sh table2 --assemble --collection ae --profile paper
```

## Options

This task reads no task-specific environment variables. The Table 3 run must
use the same profile as Table 2. The smoke profile measures batch 1024, so its
measured row cannot be compared with the paper.

## Output

Written to `result/table2/<run_id>/` (with `--assemble`:
`result/collections/ae/assembled/table2/`):

- `data.csv`, `table.csv`: the Table 2 rows (identical files)
- `table.md`: readable table
- `metadata.json`: assumptions, formulas, Table 3 source, preset/all selection mode, preset hash when used, and paper comparison
- `source/table3/`: copy of the Table 3 files used for the measured row
- `status.json`, plus `run.log` for a run (not for `--assemble`)
