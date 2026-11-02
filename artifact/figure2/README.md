# Figure 2: FFT kernel execution time

Owner: unassigned

Status: pending — depends on verified Table 3 measurements

Paper p. 9, Section 6.2: A100 inputs in registers; FFT sizes
64/128/256/512/1024/2048/4096; batch 65,536. Plot cuFFTDx FP16 against
ThunderFFT FP16, and cuFFTDx FP32 against ThunderFFT TF32. The figure uses
milliseconds and a logarithmic time axis; Table 3 reports the same data in
microseconds.

## Work to assign

- Use [Table 3](../table3/README.md)'s A100/register subset after its same-input
  correctness checks pass. Do not run a second independent timing sweep.
- Implement `run.sh` to accept or locate the verified Table 3 output, recording
  its run ID, source path, and hash. Missing input must return 78.
- Implement a deterministic plot and check all 7 sizes and 4 series are present.
  Convert units explicitly; preserve missing measurements as missing data.
- Resolve the exact timing statistic and error-bar policy with Table 3's owner;
  the draft does not specify them. Include these in the caption or metadata.

## Results and completion

Write `data.csv` and `figure.pdf` under `THUNDERFFT_RESULT_DIR`. Proposed raw
columns: `source_run,gpu,N,batch,placement,library,precision,time_us,time_ms`.
Retain source validation evidence and the plotting command. Completion requires
a plot generated only from validated fresh Table 3 measurements, with a readable
legend, units, and documented conversion. See the [runner contract](../README.md).
