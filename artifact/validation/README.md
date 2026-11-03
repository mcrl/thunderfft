# FFT correctness

This additional task checks the library itself and uses only CUDA and cuFFT
FP64 as a numerical reference. It needs no cuFFT tuning parameters, cuFFTDx
configuration or external library.

```sh
./run.sh validation                  # 88 quick cases and 18 host API checks
./run.sh validation --profile paper  # 500 full cases and 18 host API checks
```

Both modes cover seven sizes, FP16/TF32, forward/inverse, public wrappers,
direct register cores and memory guards. Full mode adds impulse, tone,
constant and public partial-batch inputs. Each case requires finite outputs,
relative L2 at most 0.005, and intact guards. This calls the existing
[`thff_validate`](../../sample/validation/README.md) suite; it is not a paper
performance measurement. It selects a logical device within inherited CUDA
visibility and does not change clocks or power settings.

Generated `data.csv`, `table.csv`, `validation.json`, `metadata.json`,
`native.log`, build logs and exact commands are saved under the task's root
`result/` directory. Source/binary hashes and compiler commands are retained.
Compute Sanitizer is a separate optional check documented in the library
validation guide; it is not silently implied by this task's success.
