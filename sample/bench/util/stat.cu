#include "stat.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
std::vector<stat::RunStat> g_stats;
const char *g_title = nullptr;

// 간단한 헤더/행 출력 유틸
void print_header() {
    if (g_title && *g_title) {
        std::printf("%s\n", g_title);
    }
    std::printf("%-10s %5s %9s | %13s | %13s | %13s\n", "type", "N",
                "B", "max_err", "compute", "e2e");
    std::printf("---------- ----- --------- | ------------- | "
                "------------- | -------------\n");
}

void print_row(const stat::RunStat &s) {
    std::printf(
        "%-10s %5u %9u | %13.6g | %10.5f ms | %10.5f ms\n",
        s.type.c_str(), s.N, s.B, s.max_abs_err, s.compute_ms, s.e2e_ms);
}
} // unnamed namespace

namespace stat {

void push(const RunStat &s) { g_stats.emplace_back(s); }

void reserve(std::size_t n) { g_stats.reserve(n); }

void clear() { g_stats.clear(); }

bool empty() { return g_stats.empty(); }

std::size_t size() { return g_stats.size(); }

void set_title(const char *title) { g_title = title; }

void print_table() {
    print_header();
    for (const auto &s : g_stats) {
        print_row(s);
    }
    std::fflush(stdout);
}

void print_csv(const char *path) {
    std::ofstream ofs(path, std::ios::out);
    if (!ofs)
        return;
    ofs << "type,N,B,max_err,compute_ms,e2e_ms\n";
    for (const auto &s : g_stats) {
        ofs << s.type << ',' << s.N << ',' << s.B << ','
            << s.max_abs_err << ',' << s.compute_ms << ',' << s.e2e_ms << '\n';
    }
}

void print_ndjson(const char *path) {
    std::ofstream ofs(path, std::ios::out);
    if (!ofs)
        return;
    for (const auto &s : g_stats) {
        ofs << "{"
            << "\"type\":\"" << s.type << "\","
            << "\"N\":" << s.N << ","
            << "\"B\":" << s.B << ","
            << "\"max_err\":" << s.max_abs_err << ","
            << "\"compute_ms\":" << s.compute_ms << ","
            << "\"e2e_ms\":" << s.e2e_ms << "}\n";
    }
}

} // namespace stat
