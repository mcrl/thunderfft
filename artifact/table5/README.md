# Table 5: precision and dynamic range

`sample/bench/main.cu`는 기존 상태로 유지하고,
정밀도/다이내믹레인지 전용 벤치는 여기 `main.cu`로 분리했습니다.

이 디렉터리의 기존 CSV는 저장소 이전 시 복사한 로컬 실행 결과입니다. 이전된 소스에서 재검증한 결과가 아닙니다. 기존 결과를 보존하려면 재실행 시 다른 `--csv` 또는 `--out` 경로를 지정하세요.

## Build

저장소 루트에서 전용 타겟만 빌드:

```bash
cmake -S . -B build
cmake --build build --target benchmark_table5_precision -j
```

## Run (TF32 빠른 alpha sweep)

```bash
./build/sample/bench/benchmark_table5_precision \
  --N-list 4096 --batch 64 --mode gaussian \
  --precision tf32 --alpha-sweep --alpha-exp-min -20 --alpha-exp-max 140 \
  --fast-finite-only --seed 1234 \
  --csv artifact/table5/tf32_alpha_sweep_fast.csv
```
