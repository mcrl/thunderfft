# Table 3 twiddle lookup and synchronization evidence

This archive contains the final A100/H100 before/after measurements for the selected
ThunderFFT changes. The [comparison table](before-after.md) keeps every Table 3
size and placement, with separate before, after and speedup columns. It measures
ThunderFFT alone and does not replace the paper references or a complete Table 3
collection that also measures other libraries. No B200 measurement is included;
B200 performance remains unmeasured for these changes.

On the A100 80 GB PCIe, the core-compute geomean speedup over all 28 configurations
is **1.055x** (range **0.992–1.256x**). The single-pass geomean is **0.997x**:
single-pass performance is not generally improved. In particular, TF32 N128
register/shared single-pass latency increased **2.59% / 2.60%**. These regressions
and the three-trial variation are retained in the report. The final A100 run
completed on 2026-11-10; earlier candidate timing data are not inputs to this
archive's tables.

The H100 PCIe core-compute geomean is **1.072x** (range **0.981–1.318x**).
Its single-pass geomean is **1.021x**, with maximum observed three-trial spans of
**13.54% for compute and 45.46% for single-pass**. Those noisy cases are retained
without treating their medians as a robust speedup. TF32 N128 single-pass
latency increased **1.56% / 1.93%** for register/shared placement; FP16 N256
register increased **2.85%**. A100 and H100 used different CUDA toolkits, so their
absolute timings are not a controlled comparison across GPU architectures.

Configurations without a selected operation or synchronization change are
controls, not additional optimization wins. For example, unchanged FP16 N256
shared on H100 moved about 7.1% faster, but its after-run trial span was 13.54%
and before/after ranges overlap. Unchanged N2048/N4096 controls moved from
−0.81% to +1.91% in core-compute time. This archive reports those movements
without attributing them to changed FFT instructions.

## Measured change and timing boundary

- FP16 N128/512/1024 and TF32 N128/256 rotations load the existing initialized
  twiddle table. The packed complex multiplication, FFT stage order and precision
  category remain unchanged. Host table coefficients can differ in their last
  bits from device sin/cos; numerical accuracy is checked; bitwise equality is not assumed.
- Benchmark wrappers use full-warp synchronization for N64/N512 and FP16 N128.
  TF32 N128 and the remaining sizes retain block synchronization. The public
  global API wrapper's barriers are unchanged.
- At source level, lookup loads and multiplication remain inside the FFT core
  and apply to single-pass calls as well as repeated benchmarks. The compiler can
  hoist invariant coefficient work. There is no source edit moving FFT work
  across the measured repetition boundary. The SM80 static audit preserves the
  baseline HMMA counts for all 56 repeat/single-pass functions.

Both sides use batch **65,536**, **R=1,000** and **3 trials** on the same GPU and
compiler. Reported microseconds apply to the configured batch, not one FFT.
Core compute is the median differential estimate `(t_2R - t_R) / R`.
Single-pass out-of-place CUDA-event latency is reported separately; each trial
averages 10 launches, then the report takes the median of three trials. It
excludes host initialization, twiddle setup and host/device transfers. Warmups,
data generation, numerical gates and the timing adapter are unchanged.

Only three trials were taken. Trial ranges are descriptive observations, not
confidence intervals, and a small measured change is not a statistical claim.

## Correctness and the instrumentation exception

On each GPU the final candidate passed all **500 full numerical cases**: all seven sizes,
FP16/TF32, forward/inverse, direct guarded core/public API, Gaussian RMS1,
offset impulse, complex tone and constant input, including public API tail
batches. Maximum relative L2 is listed below. The native repeat and single-pass
R=1 FP64 gates also passed in every one of the **56 ordinary native executions
per GPU**, all returning zero.

| GPU | FP16 maximum relative L2 | TF32 maximum relative L2 |
| --- | ---: | ---: |
| A100 | 0.00066231711096 | 0.00264049482836 |
| H100 | 0.00066231711096 | 0.00264543569515 |

On each GPU, memcheck and synccheck each passed 88 quick validator cases. Racecheck exercised
12 actual benchmark wrappers: N64/128/512 × FP16/TF32 × register/shared, R32,
one trial and batch32. All 14 sanitizer checks per GPU have zero-error summaries;
racecheck additionally requires zero warnings. These checks cover the changed
single-warp benchmark synchronization paths, not every possible runtime input.

Three instrumented cases retain native return code **1**: A100
`after-racecheck-n64-tf32-register`, and H100
`after-racecheck-n128-tf32-register` / `after-racecheck-n512-fp16-shared`. Each stops with exactly `ERROR: nonpositive/nonfinite differential timing`.
Their zero-hazard racecheck summaries and header-only native CSVs are retained. The
pinned native adapter performs both R=1 accuracy checks and the repeated wrapper
launches before that differential-timing check. Therefore the helper records
sanitizer correctness evidence but explicitly marks its timing unavailable; it
does not invent a completed timing row. Ordinary performance runs still require
return code zero and three real timing rows. All instrumented timings are
excluded from the performance table.

The exact helper change is retained in
[overlays/validation-helper/change.patch](overlays/validation-helper/change.patch),
with its before/after source and [provenance](overlays/validation-helper/provenance.json).
The original orchestration helper was from harness commit
`9b6b4235f835b068d1a0816b82b6bc6128737b31`; its SHA256 was
`dad0bbec266c7cfac23eeb6fcadb59b34715b335a9d7868fcadce0eb54a0ba6e`.
The executed helper SHA256 is
`e1744817fe0f720c1673632bc024cdfcc5247df23f7fd055a386461892fddc94`,
also recorded by runtime metadata. This overlay changes validation handling
only: it recognizes the exact post-correctness timing stop, checks the sanitizer
summary, preserves return code 1 and records explicit timing-unavailable fields.
It does not alter the native binary, numerical limit or ordinary timing parser.
The report builder independently verifies each exception's three sidecar files.
The H100 helper was applied before the paired adapter imported it; its exact
source matches the A100 helper. The original application record had a pending
runtime confirmation, which the final metadata now resolves to the expected
helper hash; see [h100-application.json](overlays/validation-helper/h100-application.json).

H100 first measured a candidate whose synchronization wrapper hash was
`92679b5f...d0092`. That timing data is excluded. The final H100 run rebuilt the
wrapper `adf1312f...b2d52e`, reran full numerical/sanitizer validation and all
56 native commands on the same H100. The [header overlay](overlays/h100-final-header/change.patch)
and [selected provenance fields](overlays/h100-final-header/public-audit.json)
record that transition. Its baseline, core headers and native timing adapter
were unchanged. The published status describes the completed paired benchmark;
the surrounding cloud coordinator later returned an SSH polling timeout after
being resumed. The final paired program itself returned zero and its results
were retrieved before cleanup.

## Identity and archive contents

Baseline source is exactly commit
`00bc845d0f3119fa405517cc601126bdbd9b3351`. The separate harness commit identifies
the measurement scripts, not the candidate kernel revision. The
[source manifest](source-manifest.json) and [selected source patch](selected-source.patch)
identify the eight changed source files. Every archived baseline source file was
checked against `git show` of the baseline commit; every candidate source file
was checked against the completed run's metadata and selected-source audit.

| A100 artifact | SHA256 |
| --- | --- |
| Before native binary | `ae68e68e9a00479e0165239fc952487512cf82f8f2b5df3d98676a096a94b60b` |
| After native binary | `3ad33dd5a11063210b687a98e031bb97765861904df8afd53a905cd74489eb23` |
| Candidate validator | `5ef5ee5b2a19aa2fcca3e9ae7bc7b0c4587be19a420cfafb58881bb8ef19b6f5` |
| Unchanged timing adapter | `8bca2a0f3d54e75aa9ce9ad82aca47f376bf3df44cdcd8e67351f8ab6e3b4f53` |
| Final benchmark wrapper | `adf1312fd6ab85dfb85205d4217b252d630c1afe83826ad3a6d7e0b7c2b2d52e` |

| H100 artifact | SHA256 |
| --- | --- |
| Before native binary | `ea02d627400ea9d9a8c0344e530da0b37f59e3bc3143c456b1682d3eb0ff87c6` |
| After native binary | `19576cbfb29e1a0b28adb822473fb8c2d8b56b739a934e225e641f785c5a5e02` |
| Candidate validator | `432942a1bc281eadc6ca953fa6d37979eec6867626ccaebf091460200e37d518` |

- `a100/`, `h100/`: byte-preserved paired CSVs, all native logs, build metadata, full
  validation and sanitizer sidecars. `copied-file-sha256.json` records their
  original hashes. No previous candidate timings are mixed into these files.
- `protocol/`: complete measured native source inputs, `paired-baseline/`, its
  source manifest, the paired Python adapter, executed validation helper, parser
  dependencies and paired CMake build definition. CUDA binaries are identified
  by hashes rather than committed here.
- `audit/`: final source identities, LUT bounds proof, compact static instruction
  comparison and resource counts. Historical candidate assembly counts in this
  audit explain the final synchronization choice; they are not timing inputs.
  Full SASS dumps are omitted to keep this archive small. `audit.py` is retained
  host-specific audit machinery requiring the original SASS inputs and paths;
  it is not the portable report or benchmark entry point.
- `report_builder.py`, `before-after.csv`, `before-after.md`, `aggregate.csv` and
  `report-inputs.json`: strict offline report generation and its input hashes.
  `cpu-report-checks.log` records the 14 passing CPU report checks.
- `SHA256SUMS`: hashes of all published files other than itself.

Original metadata and logs retain their recorded local absolute paths to
preserve evidence hashes. Those paths are provenance, not dependencies for
reading the archive; the directory mapping above and commands below use the
archived copies. Account credentials, provider records and billing data are not
part of this archive.

## Rebuild the report without a GPU

From this directory, select a new output directory:

```sh
sha256sum -c SHA256SUMS
python3 report_builder.py \
  --input A100 a100/paired-summary.csv a100/paired-metadata.json a100/after-full-validation.csv \
  --input H100 h100/paired-summary.csv h100/paired-metadata.json h100/after-full-validation.csv \
  --source-audit audit/source-and-sass-audit.json \
  --output /tmp/thunderfft-table3-report
```

The report builder rejects incomplete runs, changed source hashes, ordinary
native failures, missing accuracy coverage and unsupported sanitizer exceptions.
It does not run GPU or network commands. Output tables can be regenerated on
another machine; `report-inputs.json` will naturally record that machine's input
locations while the original CSV/log hashes stay identical.

## Reconstruct and rerun the paired protocol

`paired_measure.py` resolves its source root from its location. Copying just the
script is insufficient: retain the complete `protocol/` directory, including
`include/`, `sample/`, `paired-baseline/`, `paired-baseline-source.json` and
`artifact/table3/paired-build/CMakeLists.txt`.

Alternatively, reconstruct `protocol/paired-baseline/` from commit `00bc845`
using exactly the 28 entries in `source-manifest.json`'s `before` map. Copy those
baseline paths to `protocol/`, apply `selected-source.patch` with `patch -p1`,
and retain the archived candidate validator files and protocol/helper files in
their recorded relative locations. Verify every file against
`source-manifest.json` before building. The ready archived `protocol/` already
has that layout. [reconstruction-proof.json](reconstruction-proof.json) records
a CPU-only reconstruction check: applying the patch to the preserved baseline
and adding the validator sources reproduced all 31 candidate source hashes.

With CMake, a compatible CUDA toolkit and Compute Sanitizer installed, choose one
visible GPU, a matching architecture (A100=80, H100=90, B200=100) and fresh output
paths. The complete paired run is a GPU workload:

```sh
python3 protocol/artifact/table3/paired_measure.py \
  --arch 80 --jobs 2 --sanitizers all \
  --build-dir /tmp/thunderfft-paired-build \
  --output /tmp/thunderfft-paired-run
```

Use `CUDA_VISIBLE_DEVICES` to select exactly one GPU and `CUDACXX` to select the
CUDA compiler when multiple toolkits are installed. `--build-only` prepares
both binaries without GPU commands. `--sanitizers all` explicitly includes the
sanitizer gates on the chosen architecture. This archived protocol exercises
ThunderFFT before/after only; use the repository's regular `run.sh` for the
complete artifact pipeline and paper-shaped figures/tables.
