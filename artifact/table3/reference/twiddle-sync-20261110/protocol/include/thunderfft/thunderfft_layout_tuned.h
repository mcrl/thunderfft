#pragma once

#include "thunderfft.cuh"

namespace thunderfft {
template <typename T, int N, int BPB>
struct bench_layout {
    static_assert(sizeof(T) == 0, "bench_layout specialization missing");
};

template <int BPB>
struct bench_layout<half, 64, BPB> {
    using L_in = layout_t<64, BPB, 1, 64, 16, 1, false>;
    using L_out = layout_t<64, BPB, 1, 64, 16, 1, false>;
};

template <int BPB>
struct bench_layout<float, 64, BPB> {
    using L_in = layout_t<64, BPB, 1, 64, 16, 1, false>;
    using L_out = layout_t<64, BPB, 1, 64, 16, 1, false>;
};

template <int BPB>
struct bench_layout<half, 128, BPB> {
    using L_in = layout_t<128, BPB, 1, 128, 32, 1, false>;
    using L_out = layout_t<128, BPB, 1, 128, 16, 1, false>;
};

template <int BPB>
struct bench_layout<float, 128, BPB> {
    // using L_in = layout_t<128, BPB, 1, 128, 64, 4, true>;
    using L_in = layout_t<128, BPB, 1, 128, 32, 1, false>;
    using L_out = layout_t<128, BPB, 1, 128, 16, 1, false>;
};


template <int BPB>
struct bench_layout<half, 256, BPB> {
    using L_in = layout_t<256, BPB, 1, 256, 64, 1, false>;
    using L_out = layout_t<256, BPB, 1, 256, 256, 4, false>;
};

template <int BPB>
struct bench_layout<float, 256, BPB> {
    using L_in = layout_t<256, BPB, 1, 256, 64, 1, false>;
    using L_out = layout_t<256, BPB, 1, 256, 256, 4, false>;
};

template <int BPB>
struct bench_layout<half, 512, BPB> {
    using L_in = layout_t<512, BPB, 1, 512, 16, 1, false>;
    using L_out = layout_t<512, BPB, 1, 512, 8, 1, false>;
};

template <int BPB>
struct bench_layout<float, 512, BPB> {
    using L_in = layout_t<512, BPB, 1, 512, 16, 1, false>;
    using L_out = layout_t<512, BPB, 1, 512, 16, 1, false>;
};



template <int BPB>
struct bench_layout<half, 1024, BPB> {
    using L_in = layout_t<1024, BPB, 1, 1024, 16, 1, false>;
    using L_out = layout_t<1024, BPB, 1, 1024, 256, 2, false>;
};

template <int BPB>
struct bench_layout<float, 1024, BPB> {
    using L_in = layout_t<1024, BPB, 1, 1024, 256, 1, false>;
    using L_out = layout_t<1024, BPB, 1, 1024, 256, 4, false>;
};

template <int BPB>
struct bench_layout<half, 2048, BPB> {
    using L_in = layout_t<2048, BPB, 1, 2048, 32, 1, false>;
    using L_out = layout_t<2048, BPB, 1, 2048, 64, 1, false>;
};

template <int BPB>
struct bench_layout<float, 2048, BPB> {
    using L_in = layout_t<2048, BPB, 1, 2048, 64, 1, false>;
    using L_out = layout_t<2048, BPB, 1, 2048, 128, 1, false>;
};

template <int BPB>
struct bench_layout<half, 4096, BPB> {
    using L_in = layout_t<4096, BPB, 1, 4096, 128, 1, false>;
    using L_out = layout_t<4096, BPB, 1, 4096, 128, 1, false>;
};

template <int BPB>
struct bench_layout<float, 4096, BPB> {
    using L_in = layout_t<4096, BPB, 1, 4096, 256, 1, false>;
    using L_out = layout_t<4096, BPB, 1, 4096, 256, 1, false>;
};
} // namespace thunderfft
