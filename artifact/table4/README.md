# Table 4: MMA and unit-kernel throughput

TF32 Tensor Core throughput on paper p. 8 for A100, H100 and B200. The hardware
peak row is the paper's reported value. Ideal MMA throughput is measured by a
TF32 `m16n8k8` benchmark (`ideal_mma.cu`). Unit-kernel throughput is derived
from the Table 3 ThunderFFT N=64 TF32 register-input time, and efficiency is
unit / ideal when both come from the same GPU. The task runs only on A100,
H100 or B200.

## Run

```sh
# Ideal MMA only; the unit-kernel and efficiency cells stay blank
./run.sh table4
./run.sh table4 --profile paper

# Ideal MMA plus the Table 3 subset it needs, from the same GPU
export THUNDERFFT_TABLE3_N_LIST=64 THUNDERFFT_TABLE3_PRECISIONS=tf32 \
       THUNDERFFT_TABLE3_PLACEMENTS=register THUNDERFFT_TABLE3_LIBRARIES=ThunderFFT
./run.sh table3 table4                    # smoke, local GPU
./run.sh table3 table4 --profile paper    # paper, local GPU
./run.sh table3 table4 --backend vast --gpu H100 --profile paper
./run.sh table3 table4 --backend vast --gpu B200 --profile paper
unset THUNDERFFT_TABLE3_N_LIST THUNDERFFT_TABLE3_PRECISIONS \
      THUNDERFFT_TABLE3_PLACEMENTS THUNDERFFT_TABLE3_LIBRARIES

# Reuse a completed local Table 3 result from the same GPU
THUNDERFFT_TABLE4_SOURCE_DIR=result/table3/<run_id> ./run.sh table4 --profile paper
```

Add `--cloud-plan` to a Vast.ai command to check offer and cost without renting.
Table 4 never launches Table 3 itself. When `table3` is in the same command and
completes, its result is passed to Table 4. `THUNDERFFT_TABLE4_SOURCE_DIR` is a
local path and is not forwarded to Vast.ai, so on Vast.ai use `table3 table4`
in one command. The Table 3 result must use the same profile as Table 4.

## Collect per GPU and assemble

Collect each GPU in one command, each GPU as a different part, then assemble
without a GPU:

```sh
export THUNDERFFT_TABLE3_N_LIST=64 THUNDERFFT_TABLE3_PRECISIONS=tf32 \
       THUNDERFFT_TABLE3_PLACEMENTS=register THUNDERFFT_TABLE3_LIBRARIES=ThunderFFT
./run.sh table3 table4 --profile paper --collection t4 --part a100    # local A100
./run.sh table3 table4 --backend vast --gpu H100 --profile paper --collection t4 --part h100
./run.sh table3 table4 --backend vast --gpu B200 --profile paper --collection t4 --part b200
unset THUNDERFFT_TABLE3_N_LIST THUNDERFFT_TABLE3_PRECISIONS \
      THUNDERFFT_TABLE3_PLACEMENTS THUNDERFFT_TABLE3_LIBRARIES
./run.sh table4 --assemble --collection t4 --profile paper
```

Keep this N=64 subset in its own collection (`t4`), because Table 3 assembly
rejects overlapping parts. If Table 3 was collected with `table3 table4` per
GPU as described in [Table 3](../table3/README.md#collect-per-gpu-and-assemble),
assemble Table 4 from that collection instead:
`./run.sh table4 --assemble --collection ae --profile paper`.

## Options

| Environment variable | Values | Default (smoke / paper) |
| --- | --- | --- |
| `THUNDERFFT_TABLE4_MODE` | `measure` (build and run the ideal MMA benchmark), `derive` (CPU only, no benchmark) | `measure` |
| `THUNDERFFT_TABLE4_SOURCE_DIR` | completed Table 3 result directory; local only, not forwarded to Vast.ai | Table 3 result from the same command |
| `THUNDERFFT_TABLE4_IDEAL_DIR` | `derive` mode only: an earlier `result/table4/<run_id>/measurement` directory; not forwarded | none |
| `THUNDERFFT_TABLE4_ITERATIONS` | 1 to 8192 | 64 / 1024 |
| `THUNDERFFT_TABLE4_TRIALS` | 1 to 50, per candidate | 3 / 5 |
| `THUNDERFFT_TABLE4_WARMUPS` | 0 to 100, per candidate | 2 / 5 |
| `THUNDERFFT_TABLE4_WAVES` | 1 to 16 | 2 |
| `THUNDERFFT_TABLE4_TIMEOUT` | 1 to 1200 seconds per candidate | 120 |

The Table 3 selectors are described in [Table 3](../table3/README.md#options).

## Output

Written to `result/table4/<run_id>/` (with `--assemble`:
`result/collections/<NAME>/assembled/table4/`):

- `data.csv`: Table 3 N=64 TF32 register trials with unit-kernel TFLOPS
- `ideal.csv`: ideal MMA trials for all four launch candidates
- `table.csv`, `table.md`: per-GPU peak, ideal MMA, unit-kernel TFLOPS and efficiency; missing cells blank
- `metadata.json`: formulas, sources, coverage and missing items
- `inputs/`: copies of the Table 3 and ideal MMA inputs used
- `measurement/`: ideal MMA CSV, metadata, raw logs, build files and `disassembly.sass` (`measure` mode only)
- `configure.log`, `build.log`: benchmark build logs (`measure` mode only)
- `status.json`, plus `run.log` for a run (not for `--assemble`)
