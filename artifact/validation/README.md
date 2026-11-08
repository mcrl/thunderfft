# Validation: FFT correctness

Checks ThunderFFT correctness on the selected GPU: seven sizes (64 to 4096),
FP16 and TF32, forward and inverse, public wrappers, direct register cores and
memory guards, against a cuFFT FP64 reference (relative L2 at most 0.005).
This is not a numbered paper result; any supported GPU can run it.

## Run

```sh
./run.sh validation                   # smoke: 88 cases + 18 host API checks
./run.sh validation --profile paper   # paper: 500 cases + 18 host API checks
```

Use `--device N` to select another local GPU.

## Options

This task reads no task-specific environment variables.

## Output

Written to `result/validation/<run_id>/`:

- `data.csv`: one row per correctness case
- `table.csv`: summary per precision, size and direction
- `validation.json`: pass/fail status, case count and maximum relative L2 error
- `metadata.json`: command, GPU inventory, source and binary hashes
- `native.log`: output of the native `thff_validate` program
- `commands.txt`, `configure.log`, `build.log`: exact command and build logs
- `status.json`, `run.log`: runner status and full task log
