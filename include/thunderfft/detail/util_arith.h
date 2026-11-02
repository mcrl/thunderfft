template <unsigned int N>
inline constexpr unsigned LOG2P_builtin = [] {
    static_assert(N && (N & (N - 1)) == 0, "N must be power of two");
    return __builtin_ctz(N); // pow2에서 log2(N)과 동일
}();

template <unsigned int Bits>
__device__ __forceinline__ int reverse_bits(int x) {
    int result = 0;
    #pragma unroll
    for (unsigned int i = 0; i < Bits; ++i) {
        result = (result << 1) | ((x >> i) & 1);
    }
    return result;
}

template <int r, int N>
__device__ __forceinline__ int reverse_bit_groups(int x) {
    int num_groups = N / r;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (r * i)) & ((1 << r) - 1);
        result |= group << (r * (num_groups - 1 - i));
    }
    for (int i = 0; i < N % r; i++) {
        int bit = (x >> (r * num_groups + i)) & 1;
        result = (result << 1) | bit;
    }
    return result;
}

__device__ __forceinline__ float2 W(int index, int N) {
    return make_float2(__cosf(-2 * PI * index / N),
                       __sinf(-2 * PI * index / N));
    // return make_float2(cos(-2 * PI * index / N),
    //                    sin(-2 * PI * index / N));
}

__device__ __forceinline__ float2 cmul(float2 a, float2 w) {
    return make_float2(a.x * w.x - a.y * w.y,  // real
                       a.y * w.x + a.x * w.y); // imag
}

// float2용 연산자
__device__ __forceinline__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// half 버전
__device__ __forceinline__ half2 cmul(half2 a, float2 w) {
    // half2 -> float2
    float ax = __half2float(__low2half(a));
    float ay = __half2float(__high2half(a));

    float rx = ax * w.x - ay * w.y;
    float ry = ay * w.x + ax * w.y;

    // float2 -> half2 (반올림)
    return __floats2half2_rn(rx, ry);
}

__device__ __forceinline__ half2 cmul(half2 a, half2 b) {
    // c = (0, 0)
    // constexpr half2 zero = __half2{0.0f, 0.0f};
    const half2 zero = __float2half2_rn(0.0f);


    // Complex multiply: a * b
    return __hcmadd(a, b, zero);
}

__device__ __forceinline__ void rotate(vec2_t<half>& val, int index, int N) {
    constexpr float pi = 3.14159265358979323846f;
    const float angle_f = -2.0f * pi * float(index) / float(N);

    half2 W = __floats2half2_rn(__cosf(angle_f), __sinf(angle_f));
    // half2 W = __floats2half2_rn(angle_f, angle_f + pi/2);
    // half2 W = __floats2half2_rn(1.5f, 2.3f);
    val = cmul(val, W);
}

__device__ __forceinline__ void rotate(vec2_t<float>& val, int index, int N) {
    constexpr float pi = 3.14159265358979323846f;
    const float angle_f = -2.0f * pi * float(index) / float(N);

    float2 W = make_float2(__cosf(angle_f), __sinf(angle_f));
    val = cmul(val, W);
}

template <typename T> inline float2 to_float2(const T &v);

template <> inline float2 to_float2<float2>(const float2 &v) { return v; }

template <> inline float2 to_float2<half2>(const half2 &v) {
    return __half22float2(v);
}

// template <typename T>
// float check_max_abs_err(const float2 *ref, const T *test, int N) {
//     float max_abs_err = 0.0f;

//     for (int i = 0; i < N; ++i) {
//         float2 tf = to_float2<T>(test[i]);
//         float dx = ref[i].x - tf.x;
//         float dy = ref[i].y - tf.y;
//         float abs_err = sqrtf(dx * dx + dy * dy);
//         if (abs_err > max_abs_err)
//             max_abs_err = abs_err;
//     }

//     return max_abs_err;
// }

template <typename T> __device__ __forceinline__ void swap_inline(T &x, T &y) {
    T tmp = x;
    x = y;
    y = tmp;
}

constexpr int pad_h(int N) {
    // switch (N) {
    //     case 64: return 4;
    //     case 256: return 16;
    //     case 1024: return 64;
    //     case 4096: return 256;
    //     default:  return -1; // power of two 아닌 경우
    // }
    if(N == 512) return N/8;
    if(N == 2048) return N/8 + N/64;
    if(N == 4096) return N/16+16;
    return N / 16;
}

__host__ __device__ constexpr int pad(int N) {
    // switch (N) {
    //     case 64: return 4;
    //     case 256: return 16;
    //     case 1024: return 64;
    //     case 4096: return 256;
    //     default:  return -1;  // not a power of two
    // }
    return N / 16;
}

template <typename T> __device__ void swap_vals(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T> __device__ void swap_thread_data(T *thread_data) {
    swap_vals(thread_data[1], thread_data[4]);
    swap_vals(thread_data[17], thread_data[20]);
    swap_vals(thread_data[2], thread_data[8]);
    swap_vals(thread_data[18], thread_data[24]);
    swap_vals(thread_data[3], thread_data[12]);
    swap_vals(thread_data[19], thread_data[28]);
    swap_vals(thread_data[6], thread_data[9]);
    swap_vals(thread_data[22], thread_data[25]);
    swap_vals(thread_data[7], thread_data[13]);
    swap_vals(thread_data[23], thread_data[29]);
    swap_vals(thread_data[11], thread_data[14]);
    swap_vals(thread_data[27], thread_data[30]);
}
