# Figure 5: 2D convolution

Owner: unassigned

Status: pending

Paper p. 10, Sections 5.4/6.6: H100; cuDNN's automatically selected best
algorithm, full-image cuFFT, tiled cuFFTDx, and tiled ThunderFFT. FFT tile size is
64, filters extend through 53×53, and each tile produces `(64-f+1)` valid outputs
per dimension. All FFT method timings include filter FFT construction. The two
tiled methods use the same pipeline, differing only in the FFT kernel. Plot
runtime in milliseconds; the paper includes a zoomed view for the faster paths.

## Starting points and work to assign

- Start with `sample/conv2d/{main,thunder,cufftdx}.cu` and
  `sample/conv2d/thunder_tile64.cu`. Current `main.cu` forces N=1024 and f=3,
  comments out the cuFFTDx call, and uses cuFFT for validation rather than a full
  timed comparison. It is not the Figure 5 sweep.
- Implement the common input/filter driver, filter-size sweep, cuDNN and cuFFT
  timings, enabled cuFFTDx/ThunderFFT tiled paths, CSV export, plot, and `run.sh`.
  Record external MathDx/CuTe/cuDNN versions and build requirements.
- Confirm image dimensions, batch/channels, precision, exact filter-size list,
  direction/normalization, valid-output convention, filter orientation,
  randomness/seed, cuDNN algorithm policy, warmups/repetitions, and statistic.
  These are not fully specified in the draft. Do not infer the paper's workload
  from the sample's fixed N=1024,f=3.
- Validate every baseline with identical inputs and an agreed reference, including
  edge tiles and representative large filters, before timing. Match convolution
  or cross-correlation semantics, output cropping, and numerical tolerances.
- Include filter FFT time for all FFT paths and document allocations, plans,
  transfers, and other setup in or out of the timing interval. Keep both tiled
  pipelines equivalent so their difference isolates the FFT implementation.

## Results and completion

Write `data.csv` and `figure.pdf` under `THUNDERFFT_RESULT_DIR`. Proposed raw
columns: `gpu,image_h,image_w,batch,channels,filter,tile,library,precision,repeat,time_ms,filter_fft_included,rel_l2,passed`.
Keep baseline algorithm IDs, correctness evidence, and raw repetitions.
Completion requires all four matching and validated implementations over the
confirmed paper sweep, including filter FFT cost, and a plot generated from
fresh data. See the [runner contract](../README.md).
