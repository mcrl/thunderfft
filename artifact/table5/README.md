# Table5_precision

`thunderfft/sample/bench/main.cu`는 기존 상태로 유지하고,
정밀도/다이내믹레인지 전용 벤치는 여기 `main.cu`로 분리했습니다.

## Build

`thunderfft` 기존 빌드 디렉토리에서 전용 타겟만 빌드:

```bash
cmake -S /home/chaewon/fft/thunderfft -B /home/chaewon/fft/thunderfft/build
cmake --build /home/chaewon/fft/thunderfft/build --target benchmark_table5_precision -j
```

## Run (TF32 빠른 alpha sweep)

```bash
/home/chaewon/fft/thunderfft/build/sample/bench/benchmark_table5_precision \
  --N-list 4096 --batch 64 --mode gaussian \
  --precision tf32 --alpha-sweep --alpha-exp-min -20 --alpha-exp-max 140 \
  --fast-finite-only --seed 1234 \
  --csv /home/chaewon/fft/Table5_precision/tf32_alpha_sweep_fast.csv
```
