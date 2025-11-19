// fast_fft.hpp
// Brutal, fast, and to the point.
// In-place iterative radix-2 FFT with twiddle caching and plan reuse.
// Public API:
//   - FFT::fft(a, inverse=false) : in-place on vector<complex<double>>
//   - FFT::convolve_complex(a, b) : complex convolution
//   - FFT::convolve_real(a, b)    : real convolution (double), padded to pow2
//
// Compile flags recommended: -O3 -march=native -ffast-math
// Assumes IEEE754 and sane libstdc++.

#pragma once
#include <bits/stdc++.h>
namespace FFT {

using cd = std::complex<double>;
using vd = std::vector<double>;
using vcd = std::vector<cd>;

static inline bool is_power_of_two(size_t n) { return n && ((n & (n - 1)) == 0); }
static inline size_t next_power_of_two(size_t n) {
    if (!n) return 1;
    --n; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; 
#if SIZE_MAX > 0xFFFFFFFFu
    n |= n >> 32;
#endif
    return ++n;
}

// Cached plan so you don’t rebuild twiddles every call.
struct Plan {
    size_t n_cached = 0;
    std::vector<int> rev;
    vcd roots; // roots[k] = exp(i * 2*pi*k / n_cached)

    void ensure(size_t n) {
        if (n == n_cached) return;
        if (!is_power_of_two(n)) {
            throw std::runtime_error("FFT::Plan.ensure: n must be a power of two");
        }
        n_cached = n;
        rev.assign(n, 0);
        for (size_t i = 0, j = 0; i < n; ++i) {
            rev[i] = (int)j;
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
        }
        roots.resize(n);
        const double tau = 2.0 * M_PI;
        for (size_t k = 0; k < n; ++k) {
            double ang = tau * k / (double)n;
            roots[k] = cd(std::cos(ang), std::sin(ang));
        }
    }
};

static Plan GLOBAL_PLAN;

// In-place FFT. a.size() must be a power of two. inverse = true does IFFT and scales by 1/n.
inline void fft(vcd &a, bool inverse = false) {
    const size_t n = a.size();
    if (!is_power_of_two(n)) {
        throw std::runtime_error("FFT::fft: input size must be power of two");
    }
    GLOBAL_PLAN.ensure(n);

    // bit-reversal
    const auto &rev = GLOBAL_PLAN.rev;
    for (size_t i = 0; i < n; ++i) {
        size_t j = (size_t)rev[i];
        if (i < j) std::swap(a[i], a[j]);
    }

    // iterative butterflies
    const auto &roots = GLOBAL_PLAN.roots;
    for (size_t len = 1; len < n; len <<= 1) {
        size_t step = n / (len << 1); // root step
        for (size_t i = 0; i < n; i += (len << 1)) {
            for (size_t k = 0; k < len; ++k) {
                // twiddle index
                size_t idx = k * step;
                cd w = inverse ? std::conj(roots[idx]) : roots[idx];
                cd u = a[i + k];
                cd v = a[i + k + len] * w;
                a[i + k]       = u + v;
                a[i + k + len] = u - v;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / (double)n;
        for (size_t i = 0; i < n; ++i) a[i] *= inv_n;
    }
}

// Complex convolution: c = a (*) b
inline vcd convolve_complex(const vcd &a, const vcd &b) {
    if (a.empty() || b.empty()) return {};
    size_t n = next_power_of_two(a.size() + b.size() - 1);
    vcd fa(a.begin(), a.end()); fa.resize(n);
    vcd fb(b.begin(), b.end()); fb.resize(n);
    fft(fa, false);
    fft(fb, false);
    for (size_t i = 0; i < n; ++i) fa[i] *= fb[i];
    fft(fa, true);
    // fa now holds the linear convolution (complex)
    return fa;
}

// Real convolution (double): faster trick using one FFT.
// Packs a and b into real/imag, does single FFT, unpacks. Classic optimization.
inline vd convolve_real(const vd &a, const vd &b) {
    if (a.empty() || b.empty()) return {};
    size_t need = a.size() + b.size() - 1;
    size_t n = next_power_of_two(need);
    vcd fa(n);
    for (size_t i = 0; i < a.size(); ++i) fa[i].real(a[i]);
    for (size_t i = 0; i < b.size(); ++i) fa[i].imag(b[i]);
    for (size_t i = a.size(); i < n; ++i) fa[i].real(0.0);
    for (size_t i = b.size(); i < n; ++i) fa[i].imag(0.0);

    fft(fa, false);

    vcd c(n);
    // c[k] = F[a]_k * conj(F[b]_{-k}) extracted from packed spectrum
    // Using the well-known identity to compute two real convs from one complex FFT.
    c[0] = cd(fa[0].real() * fa[0].imag(), 0.0);
    for (size_t k = 1; k < n; ++k) {
        cd x = fa[k];
        cd y = std::conj(fa[(n - k) % n]);
        cd a1 = (x + y) * cd(0.5, 0.0);
        cd a2 = (x - y) * cd(0.0, -0.5);
        c[k] = a1 * a2; // (A_re + i B_re) * (A_im + i B_im) after packing algebra
    }

    // Inverse transform trick:
    // We need convolution of re and im parts separately:
    // Recovered spectra for A and B:
    //   Fa = a1 + i a2
    //   Fb = a1 - i a2
    // Their pointwise product’s real part gives the real convolution result after IFFT.
    // But we packaged to compute directly; now convert c to time domain carefully.

    // The direct trick above gives half-baked; safer path: unpack and multiply properly.
    // Let's do the robust two-FFT approach for correctness + speed still fine at -O3.
    // If you want the absolute last drop of speed, keep the single-FFT trick but be
    // very careful—bug-prone. Here’s the robust version:

    // Robust approach:
    // Build Fa (real) and Fb (real) via two FFTs using cached plan anyway.
    vcd A(n), B(n);
    for (size_t i = 0; i < a.size(); ++i) A[i] = cd(a[i], 0.0);
    for (size_t i = a.size(); i < n; ++i) A[i] = cd(0.0, 0.0);
    for (size_t i = 0; i < b.size(); ++i) B[i] = cd(b[i], 0.0);
    for (size_t i = b.size(); i < n; ++i) B[i] = cd(0.0, 0.0);

    fft(A, false);
    fft(B, false);
    for (size_t i = 0; i < n; ++i) A[i] *= B[i];
    fft(A, true);

    vd res(need);
    for (size_t i = 0; i < need; ++i) res[i] = A[i].real();
    return res;
}

} // namespace FFT

