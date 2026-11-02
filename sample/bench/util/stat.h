#pragma once
#include <cstddef>
#include <string>

namespace stat {

struct RunStat {
    std::string type; // "half", "float", ...
    unsigned N;       // FFT size
    unsigned radix;   // radix
    unsigned B;       // batch size
    double max_abs_err;
    double compute_ms; // kernel-only (inside repeat)
    double e2e_ms;     // end-to-end (no inside repeat)
};

// 결과 추가/관리
void push(const RunStat &s);
void reserve(std::size_t n);
void clear();

// 조회
bool empty();
std::size_t size();

// 출력(한 번만 호출하면 됨)
void set_title(const char *title);   // 선택: 표 위에 제목 한 줄
void print_table();                  // 콘솔 표 출력
void print_csv(const char *path);    // 선택: CSV 파일 저장
void print_ndjson(const char *path); // 선택: NDJSON 라인별 저장

} // namespace stat
