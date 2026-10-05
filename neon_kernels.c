#include <stdint.h>
#include <stddef.h>
#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

int neon_enabled(void) { return HAVE_NEON; }

/* Fraction of raw signed-8-bit I/Q components at or beyond a near-rail level. */
float iq_near_rail_fraction(const int8_t *x, size_t count, int8_t rail) {
    if (!x || count == 0) return 0.0f;
    if (rail < 1) rail = 1;
    if (rail > 127) rail = 127;
#if HAVE_NEON
    uint64_t hot = 0;
    size_t i = 0;
    const int8x16_t hi_thr = vdupq_n_s8(rail - 1);
    const int8x16_t lo_thr = vdupq_n_s8((int8_t)(-rail + 1));
    for (; i + 16 <= count; i += 16) {
        int8x16_t v = vld1q_s8(x + i);
        uint8x16_t hi = vcgtq_s8(v, hi_thr);
        uint8x16_t lo = vcltq_s8(v, lo_thr);
        uint8x16_t mask = vorrq_u8(hi, lo);
        hot += (uint64_t)vaddlvq_u8(mask) / 255u;
    }
    for (; i < count; ++i) {
        int v = x[i];
        if (v >= rail || v <= -rail) ++hot;
    }
    return (float)hot / (float)count;
#else
    size_t hot = 0;
    for (size_t i = 0; i < count; ++i) {
        int v = x[i];
        if (v >= rail || v <= -rail) ++hot;
    }
    return (float)hot / (float)count;
#endif
}

static inline void scalar_mean_iq(const int8_t *src, size_t n, float *mr, float *mi) {
    int64_t sr = 0, si = 0;
    for (size_t i = 0; i < n; ++i) {
        sr += src[2*i + 0];
        si += src[2*i + 1];
    }
    *mr = (float)sr / (float)n;
    *mi = (float)si / (float)n;
}

void iq_batch_to_c64_hann_split(
    const int8_t *src_m, const int8_t *src_s,
    float *dst_m, float *dst_s, const float *window,
    size_t frames, size_t n
) {
    for (size_t f = 0; f < frames; ++f) {
        const int8_t *sm = src_m + f * 2 * n;
        const int8_t *ss = src_s + f * 2 * n;
        float *dm = dst_m + f * 2 * n;
        float *ds = dst_s + f * 2 * n;
        float mmr, mmi, msr, msi;
#if HAVE_NEON
        int32x4_t smr_v = vdupq_n_s32(0), smi_v = vdupq_n_s32(0);
        int32x4_t ssr_v = vdupq_n_s32(0), ssi_v = vdupq_n_s32(0);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            int8x8x2_t vm = vld2_s8(sm + 2*i);
            int8x8x2_t vs = vld2_s8(ss + 2*i);
            int16x8_t mr16 = vmovl_s8(vm.val[0]);
            int16x8_t mi16 = vmovl_s8(vm.val[1]);
            int16x8_t sr16 = vmovl_s8(vs.val[0]);
            int16x8_t si16 = vmovl_s8(vs.val[1]);
            smr_v = vaddq_s32(smr_v, vmovl_s16(vget_low_s16(mr16)));
            smr_v = vaddq_s32(smr_v, vmovl_s16(vget_high_s16(mr16)));
            smi_v = vaddq_s32(smi_v, vmovl_s16(vget_low_s16(mi16)));
            smi_v = vaddq_s32(smi_v, vmovl_s16(vget_high_s16(mi16)));
            ssr_v = vaddq_s32(ssr_v, vmovl_s16(vget_low_s16(sr16)));
            ssr_v = vaddq_s32(ssr_v, vmovl_s16(vget_high_s16(sr16)));
            ssi_v = vaddq_s32(ssi_v, vmovl_s16(vget_low_s16(si16)));
            ssi_v = vaddq_s32(ssi_v, vmovl_s16(vget_high_s16(si16)));
        }
        int32_t smr = vaddvq_s32(smr_v), smi = vaddvq_s32(smi_v);
        int32_t ssr = vaddvq_s32(ssr_v), ssi = vaddvq_s32(ssi_v);
        for (; i < n; ++i) {
            smr += sm[2*i]; smi += sm[2*i+1];
            ssr += ss[2*i]; ssi += ss[2*i+1];
        }
        mmr = (float)smr / (float)n; mmi = (float)smi / (float)n;
        msr = (float)ssr / (float)n; msi = (float)ssi / (float)n;

        const float32x4_t bmr = vdupq_n_f32(mmr), bmi = vdupq_n_f32(mmi);
        const float32x4_t bsr = vdupq_n_f32(msr), bsi = vdupq_n_f32(msi);
        i = 0;
        for (; i + 8 <= n; i += 8) {
            int8x8x2_t vm = vld2_s8(sm + 2*i);
            int8x8x2_t vs = vld2_s8(ss + 2*i);
            int16x8_t mr16 = vmovl_s8(vm.val[0]);
            int16x8_t mi16 = vmovl_s8(vm.val[1]);
            int16x8_t sr16 = vmovl_s8(vs.val[0]);
            int16x8_t si16 = vmovl_s8(vs.val[1]);

            float32x4x2_t om, os;
            float32x4_t mr0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(mr16)));
            float32x4_t mi0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(mi16)));
            float32x4_t sr0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(sr16)));
            float32x4_t si0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(si16)));
            float32x4_t mr1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(mr16)));
            float32x4_t mi1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(mi16)));
            float32x4_t sr1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(sr16)));
            float32x4_t si1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(si16)));
            float32x4_t w0 = vld1q_f32(window + i), w1 = vld1q_f32(window + i + 4);
            om.val[0] = vmulq_f32(vsubq_f32(mr0, bmr), w0);
            om.val[1] = vmulq_f32(vsubq_f32(mi0, bmi), w0);
            os.val[0] = vmulq_f32(vsubq_f32(sr0, bsr), w0);
            os.val[1] = vmulq_f32(vsubq_f32(si0, bsi), w0);
            vst2q_f32(dm + 2*i, om); vst2q_f32(ds + 2*i, os);

            om.val[0] = vmulq_f32(vsubq_f32(mr1, bmr), w1);
            om.val[1] = vmulq_f32(vsubq_f32(mi1, bmi), w1);
            os.val[0] = vmulq_f32(vsubq_f32(sr1, bsr), w1);
            os.val[1] = vmulq_f32(vsubq_f32(si1, bsi), w1);
            vst2q_f32(dm + 2*(i+4), om); vst2q_f32(ds + 2*(i+4), os);
        }
        for (; i < n; ++i) {
            float w = window[i];
            dm[2*i] = ((float)sm[2*i] - mmr) * w;
            dm[2*i+1] = ((float)sm[2*i+1] - mmi) * w;
            ds[2*i] = ((float)ss[2*i] - msr) * w;
            ds[2*i+1] = ((float)ss[2*i+1] - msi) * w;
        }
#else
        scalar_mean_iq(sm, n, &mmr, &mmi);
        scalar_mean_iq(ss, n, &msr, &msi);
        for (size_t i = 0; i < n; ++i) {
            float w = window[i];
            dm[2*i] = ((float)sm[2*i] - mmr) * w;
            dm[2*i+1] = ((float)sm[2*i+1] - mmi) * w;
            ds[2*i] = ((float)ss[2*i] - msr) * w;
            ds[2*i+1] = ((float)ss[2*i+1] - msi) * w;
        }
#endif
    }
}

void cfar_noise_circular_f32(const float *p, size_t n, size_t train, size_t guard, float *out) {
    // Circular cell-averaging CFAR noise estimate. We maintain the two
    // train-side sums separately to make the update O(1) per FFT bin.
    if (n == 0 || train == 0) return;
    const size_t span = guard + train;
    double left = 0.0, right = 0.0;
    for (size_t d = guard + 1; d <= span; ++d) {
        left += p[(n - d) % n];
        right += p[d % n];
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = (float)((left + right) / (double)(2 * train));
        // Move the window one bin forward: the oldest left sample leaves,
        // the newest right sample enters, and the near-side guard boundary shifts.
        size_t old_left = (i + n - span) % n;
        size_t new_left = (i + n - guard) % n;
        size_t old_right = (i + guard + 1) % n;
        size_t new_right = (i + span + 1) % n;
        left += (double)p[new_left] - (double)p[old_left];
        right += (double)p[new_right] - (double)p[old_right];
    }
}

void windowed_ratio_best_f32(const float *p, size_t n, float noise,
                              const size_t *widths, size_t nwidths,
                              float *best_ratio, size_t *best_width) {
    if (n == 0 || nwidths == 0 || noise <= 0.0f) return;
    for (size_t i=0; i<n; ++i) { best_ratio[i] = 0.0f; best_width[i] = 1; }
    const double inv_noise = 1.0 / (double)noise;
    for (size_t wi=0; wi<nwidths; ++wi) {
        size_t w = widths[wi];
        if (w < 2) w = 1;
        if (w > n) w = n;
        size_t half = w / 2;
        double sum = 0.0;
        // Centered circular window at center=0: indices [-half, -half+w-1].
        for (size_t j=0; j<w; ++j) sum += p[(n - half + j) % n];
        for (size_t c=0; c<n; ++c) {
            double ratio = (sum / (double)w) * inv_noise;
            double score2 = (ratio - 1.0) * (ratio - 1.0) * (double)w;
            double old_delta = (double)best_ratio[c] - 1.0;
            double old_score2 = old_delta * old_delta * (double)best_width[c];
            if (score2 > old_score2) { best_ratio[c]=(float)ratio; best_width[c]=w; }
            size_t old_idx = (c + n - half) % n;
            size_t new_idx = (c + n - half + w) % n;
            sum += (double)p[new_idx] - (double)p[old_idx];
        }
    }
}

void power_c64_f32(const float *x, float *out, size_t count_complex) {
#if HAVE_NEON
    size_t i = 0;
    for (; i + 4 <= count_complex; i += 4) {
        float32x4x2_t z = vld2q_f32(x + 2*i);
        float32x4_t p = vaddq_f32(vmulq_f32(z.val[0], z.val[0]), vmulq_f32(z.val[1], z.val[1]));
        vst1q_f32(out + i, p);
    }
    for (; i < count_complex; ++i) {
        float r = x[2*i], im = x[2*i+1];
        out[i] = r*r + im*im;
    }
#else
    for (size_t i = 0; i < count_complex; ++i) {
        float r = x[2*i], im = x[2*i+1];
        out[i] = r*r + im*im;
    }
#endif
}

void cross_sum_range_f32(
    const float *xm, const float *xs, size_t count_complex,
    float *out_re, float *out_im, float *out_pm, float *out_ps
) {
#if HAVE_NEON
    float32x4_t cre = vdupq_n_f32(0.0f), cim = vdupq_n_f32(0.0f);
    float32x4_t pm = vdupq_n_f32(0.0f), ps = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= count_complex; i += 4) {
        float32x4x2_t m = vld2q_f32(xm + 2*i);
        float32x4x2_t s = vld2q_f32(xs + 2*i);
        // conj(s) * m = (sr*mr + si*mi) + j(sr*mi? sign follows conj convention)
        cre = vaddq_f32(cre, vmlaq_f32(vmulq_f32(s.val[0], m.val[0]), s.val[1], m.val[1]));
        cim = vaddq_f32(cim, vmlsq_f32(vmulq_f32(s.val[0], m.val[1]), s.val[1], m.val[0]));
        pm = vaddq_f32(pm, vmlaq_f32(vmulq_f32(m.val[0], m.val[0]), m.val[1], m.val[1]));
        ps = vaddq_f32(ps, vmlaq_f32(vmulq_f32(s.val[0], s.val[0]), s.val[1], s.val[1]));
    }
    *out_re = vaddvq_f32(cre); *out_im = vaddvq_f32(cim);
    *out_pm = vaddvq_f32(pm); *out_ps = vaddvq_f32(ps);
    for (; i < count_complex; ++i) {
        float mr = xm[2*i], mi = xm[2*i+1];
        float sr = xs[2*i], si = xs[2*i+1];
        *out_re += sr*mr + si*mi;
        *out_im += sr*mi - si*mr;
        *out_pm += mr*mr + mi*mi;
        *out_ps += sr*sr + si*si;
    }
#else
    float re=0, im=0, pm=0, ps=0;
    for (size_t i = 0; i < count_complex; ++i) {
        float mr=xm[2*i], mi=xm[2*i+1], sr=xs[2*i], si=xs[2*i+1];
        re += sr*mr + si*mi;
        im += sr*mi - si*mr;
        pm += mr*mr + mi*mi;
        ps += sr*sr + si*si;
    }
    *out_re=re; *out_im=im; *out_pm=pm; *out_ps=ps;
#endif
}

void dft_bins_c64_root(
    const float *x, size_t n,
    const float *roots_ri, const size_t *bins,
    size_t nbins, float *out
) {
    /*
     * Exact DFT samples using a single precomputed unit-circle root table.
     * roots_ri is interleaved complex float: re_0, im_0, re_1, im_1, ...
     * For bin k, exp(-j*2*pi*k*i/N) == roots[(k*i) mod N].
     * This keeps sparse DFT independent of changing/FHSS bin sets: no trig table
     * allocation, no Python sin/cos, and only O(N) compact twiddle storage.
     */
    for (size_t b = 0; b < nbins; ++b) {
        const size_t k = bins[b] % n;
        size_t idx = 0;
#if HAVE_NEON
        float32x4_t ar = vdupq_n_f32(0.0f), ai = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            /* Four scalar phase steps are used per unrolled group.  The root
               lookup remains contiguous in x and bounded to one compact table. */
            float xr0=x[2*i],   xi0=x[2*i+1];
            size_t j0=idx;
            idx += k; if (idx >= n) idx -= n;
            float xr1=x[2*(i+1)], xi1=x[2*(i+1)+1];
            size_t j1=idx;
            idx += k; if (idx >= n) idx -= n;
            float xr2=x[2*(i+2)], xi2=x[2*(i+2)+1];
            size_t j2=idx;
            idx += k; if (idx >= n) idx -= n;
            float xr3=x[2*(i+3)], xi3=x[2*(i+3)+1];
            size_t j3=idx;
            idx += k; if (idx >= n) idx -= n;

            float32x4_t wr = {roots_ri[2*j0], roots_ri[2*j1], roots_ri[2*j2], roots_ri[2*j3]};
            float32x4_t wi = {roots_ri[2*j0+1], roots_ri[2*j1+1], roots_ri[2*j2+1], roots_ri[2*j3+1]};
            float32x4_t xv_r = {xr0,xr1,xr2,xr3};
            float32x4_t xv_i = {xi0,xi1,xi2,xi3};
            ar = vaddq_f32(ar, vsubq_f32(vmulq_f32(xv_r,wr), vmulq_f32(xv_i,wi)));
            ai = vaddq_f32(ai, vaddq_f32(vmulq_f32(xv_r,wi), vmulq_f32(xv_i,wr)));
        }
        float sr = vaddvq_f32(ar), si = vaddvq_f32(ai);
        for (; i < n; ++i) {
            float wr=roots_ri[2*idx], wi=roots_ri[2*idx+1];
            float xr=x[2*i], xi=x[2*i+1];
            sr += xr*wr - xi*wi;
            si += xr*wi + xi*wr;
            idx += k; if (idx >= n) idx -= n;
        }
        out[2*b] = sr; out[2*b+1] = si;
#else
        float sr=0.0f, si=0.0f;
        for (size_t i=0; i<n; ++i) {
            float wr=roots_ri[2*idx], wi=roots_ri[2*idx+1];
            float xr=x[2*i], xi=x[2*i+1];
            sr += xr*wr - xi*wi;
            si += xr*wi + xi*wr;
            idx += k; if (idx >= n) idx -= n;
        }
        out[2*b]=sr; out[2*b+1]=si;
#endif
    }
}

