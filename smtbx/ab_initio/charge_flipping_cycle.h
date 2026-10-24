#ifndef SMTBX_AB_INITIO_CHARGE_FLIPPING_CYCLE_H
#define SMTBX_AB_INITIO_CHARGE_FLIPPING_CYCLE_H

#include <cmath>
#include <cstddef>

/* The line transforms run four lines at a time: scitbx fftpack instantiated
   on a four-double lane type, every lane the scalar operation sequence of one
   line, so the same roundings (no FMA: SSE2/AVX have none, and the portable
   loops must not be contracted). x64 gets two SSE2 registers, the baseline;
   AVX only where the build asks for it (libtbx configure
   --msvc_arch_flag=AVX|AVX2, which defines __AVX__), never on gcc/clang,
   whose __m256d loads assume 32-byte alignment that af::shared does not give.
   The std overloads must be declared before fftpack is first included: its
   templates name std::sqrt etc. qualified.
 */
#if defined(_MSC_VER)
#define SMTBX_CF_FI __forceinline
#else
#define SMTBX_CF_FI inline __attribute__((always_inline))
#endif
#if defined(_MSC_VER) && defined(__AVX__)
#define SMTBX_CF_AVX
#include <immintrin.h>
#elif defined(_M_X64) || defined(__x86_64__)
#define SMTBX_CF_X64
#include <emmintrin.h>
#endif
namespace smtbx { namespace ab_initio { namespace cf_lanes {
#if defined(SMTBX_CF_AVX)
  // the vector is the member, so MSVC keeps it in a register; d is the view
  // the gathers and scatters use
  struct L {
    union { __m256d v; double d[4]; };
    SMTBX_CF_FI L() : v(_mm256_setzero_pd()) {}
    SMTBX_CF_FI L(__m256d x) : v(x) {}
    template <class T> SMTBX_CF_FI L(T x) : v(_mm256_set1_pd((double)x)) {}
    SMTBX_CF_FI L& operator+=(const L& b) { v = _mm256_add_pd(v, b.v); return *this; }
    SMTBX_CF_FI L& operator-=(const L& b) { v = _mm256_sub_pd(v, b.v); return *this; }
    SMTBX_CF_FI L& operator*=(const L& b) { v = _mm256_mul_pd(v, b.v); return *this; }
  };
  SMTBX_CF_FI L operator+(const L& a, const L& b) { return L(_mm256_add_pd(a.v, b.v)); }
  SMTBX_CF_FI L operator-(const L& a, const L& b) { return L(_mm256_sub_pd(a.v, b.v)); }
  SMTBX_CF_FI L operator*(const L& a, const L& b) { return L(_mm256_mul_pd(a.v, b.v)); }
  SMTBX_CF_FI L operator/(const L& a, const L& b) { return L(_mm256_div_pd(a.v, b.v)); }
  SMTBX_CF_FI L operator-(const L& a) { return L(_mm256_xor_pd(a.v, _mm256_set1_pd(-0.0))); }
  // the whole vector at once: four lane stores and then a vector load would
  // stall store forwarding on every point
  SMTBX_CF_FI L lset(double a, double b, double c, double d) { return L(_mm256_set_pd(d, c, b, a)); }
#elif defined(SMTBX_CF_X64)
  struct L {
    union { struct { __m128d a, b; }; double d[4]; };
    SMTBX_CF_FI L() : a(_mm_setzero_pd()), b(_mm_setzero_pd()) {}
    SMTBX_CF_FI L(__m128d x, __m128d y) : a(x), b(y) {}
    template <class T> SMTBX_CF_FI L(T x) : a(_mm_set1_pd((double)x)), b(a) {}
    SMTBX_CF_FI L& operator+=(const L& o) { a = _mm_add_pd(a, o.a); b = _mm_add_pd(b, o.b); return *this; }
    SMTBX_CF_FI L& operator-=(const L& o) { a = _mm_sub_pd(a, o.a); b = _mm_sub_pd(b, o.b); return *this; }
    SMTBX_CF_FI L& operator*=(const L& o) { a = _mm_mul_pd(a, o.a); b = _mm_mul_pd(b, o.b); return *this; }
  };
  SMTBX_CF_FI L operator+(const L& x, const L& y) { return L(_mm_add_pd(x.a, y.a), _mm_add_pd(x.b, y.b)); }
  SMTBX_CF_FI L operator-(const L& x, const L& y) { return L(_mm_sub_pd(x.a, y.a), _mm_sub_pd(x.b, y.b)); }
  SMTBX_CF_FI L operator*(const L& x, const L& y) { return L(_mm_mul_pd(x.a, y.a), _mm_mul_pd(x.b, y.b)); }
  SMTBX_CF_FI L operator/(const L& x, const L& y) { return L(_mm_div_pd(x.a, y.a), _mm_div_pd(x.b, y.b)); }
  SMTBX_CF_FI L operator-(const L& x) { __m128d s = _mm_set1_pd(-0.0); return L(_mm_xor_pd(x.a, s), _mm_xor_pd(x.b, s)); }
  SMTBX_CF_FI L lset(double a, double b, double c, double d) { return L(_mm_set_pd(b, a), _mm_set_pd(d, c)); }
#else
  struct L {
    double d[4];
    L() : d{0, 0, 0, 0} {}
    template <class T> L(T x) { for (int i = 0; i < 4; i++) d[i] = (double)x; }
    L& operator+=(const L& b) { for (int i = 0; i < 4; i++) d[i] += b.d[i]; return *this; }
    L& operator-=(const L& b) { for (int i = 0; i < 4; i++) d[i] -= b.d[i]; return *this; }
    L& operator*=(const L& b) { for (int i = 0; i < 4; i++) d[i] *= b.d[i]; return *this; }
  };
  inline L operator+(const L& a, const L& b) { L r; for (int i = 0; i < 4; i++) r.d[i] = a.d[i] + b.d[i]; return r; }
  inline L operator-(const L& a, const L& b) { L r; for (int i = 0; i < 4; i++) r.d[i] = a.d[i] - b.d[i]; return r; }
  inline L operator*(const L& a, const L& b) { L r; for (int i = 0; i < 4; i++) r.d[i] = a.d[i]*b.d[i]; return r; }
  inline L operator/(const L& a, const L& b) { L r; for (int i = 0; i < 4; i++) r.d[i] = a.d[i]/b.d[i]; return r; }
  inline L operator-(const L& a) { L r; for (int i = 0; i < 4; i++) r.d[i] = -a.d[i]; return r; }
  inline L lset(double a, double b, double c, double d) { L r; r.d[0] = a; r.d[1] = b; r.d[2] = c; r.d[3] = d; return r; }
#endif
  struct LC { L re, im; };
}}} // namespace smtbx::ab_initio::cf_lanes

// fftpack calls these only on twiddles and constants, equal in all lanes
namespace std {
  inline smtbx::ab_initio::cf_lanes::L sqrt(const smtbx::ab_initio::cf_lanes::L& a) { return std::sqrt(a.d[0]); }
  inline smtbx::ab_initio::cf_lanes::L cos(const smtbx::ab_initio::cf_lanes::L& a) { return std::cos(a.d[0]); }
  inline smtbx::ab_initio::cf_lanes::L sin(const smtbx::ab_initio::cf_lanes::L& a) { return std::sin(a.d[0]); }
  inline smtbx::ab_initio::cf_lanes::L atan(const smtbx::ab_initio::cf_lanes::L& a) { return std::atan(a.d[0]); }
}

#include <scitbx/fftpack/complex_to_complex.h>
#include <scitbx/fftpack/real_to_complex.h>
#include <scitbx/array_family/shared.h>
#include <scitbx/array_family/tiny.h>
#include <scitbx/array_family/accessors/c_grid_padded.h>
#include <cctbx/miller.h>
#include <smtbx/error.h>

#include <algorithm>
#include <complex>
#include <vector>
#include <cmath>

namespace smtbx { namespace ab_initio {

  namespace af = scitbx::af;

  /* One cycle of charge_flipping.weak_reflection_improved_iterator (flip,
     Fourier analysis, free-set cap, Oszlanyi-Suto transfer, synthesis) over
     one persistent padded map, with the operations of the flex / fft_map path
     in the same order so that it gives the same bits. The line transforms of
     real_to_complex_3d run only on the lines whose results are read (forward)
     or whose input is not all zero (backward). P1 data only, as the iterator
     has it.
   */
  class charge_flipping_cycle
  {
  public:
    typedef std::complex<double> cd;

    charge_flipping_cycle(af::tiny<int, 3> const& n_real,
                          af::const_ref<cctbx::miller::index<> > const& h_obs,
                          af::const_ref<double> const& f_obs,
                          af::const_ref<cctbx::miller::index<> > const& h_free,
                          double fft_scale, double volume_scale,
                          std::size_t cut, double delta_varphi)
      : n0(n_real[0]), n1(n_real[1]), n2(n_real[2]), nzc(n2/2 + 1), m2(2*nzc),
        no(f_obs.size()), nf(h_free.size()), cut_(cut),
        fft_scale_(fft_scale), vscale(volume_scale), dphi(delta_varphi),
        f000_(0), g000_(0), mean(0), sigma_(0), skew(0), have_skew(false),
        fx(n0), fy(n1), fz(n2),
        fo(f_obs.begin(), f_obs.end()),
        off(no + nf), om(no + nf), op(no + nf), cj(no + nf),
        fc(no), ff(nf), g(no + nf)
    {
      SMTBX_ASSERT(h_obs.size() == f_obs.size());
      SMTBX_ASSERT(cut <= f_obs.size());
      int ss = 2*std::max(std::max(n0, n1), nzc);
      seq.assign(ss, L(0)); scr.assign(ss, L(0));
      map.assign((std::size_t)n0*n1*m2, 0.0);
      zrow.assign(m2, 0.0);
      cap = no ? *std::max_element(fo.begin(), fo.end()) : 0;
      // trig_table[0] of structure_factors::to_map: -two_pi * 0 / t_den = -0.0
      double a0 = -(2*3.14159265358979323846)*0/12;
      tr0 = cd(std::cos(a0), std::sin(a0));
      std::vector<char> used((std::size_t)n1*nzc, 0);
      used[0] = 1;
      for (int i = 0; i < no + nf; i++) {
        cctbx::miller::index<> const& h = i < no ? h_obs[i] : h_free[i - no];
        // from_map, non-anomalous, conjugate_flag: l < 0 reads -h unconjugated
        int s = h[2] < 0 ? -1 : 1;
        cj[i] = h[2] >= 0;
        off[i] = coff(mod(s*h[0], n0), mod(s*h[1], n1), s*h[2]);
        used[(std::size_t)mod(s*h[1], n1)*nzc + s*h[2]] = 1;
        // to_map in P1, conjugate_flag: -h gets the term, +h its conjugate,
        // only inside the half-complex focus
        int a = mod(-h[2], n2), b = mod(h[2], n2);
        om[i] = a < nzc ? coff(mod(-h[0], n0), mod(-h[1], n1), a) : -1;
        op[i] = b < nzc ? coff(mod(h[0], n0), mod(h[1], n1), b) : -1;
        if (a < nzc) used[(std::size_t)mod(-h[1], n1)*nzc + a] = 1;
        if (b < nzc) used[(std::size_t)mod(h[1], n1)*nzc + b] = 1;
      }
      for (int iz = 0; iz < nzc; iz++) {
        std::vector<int> ys;
        for (int iy = 0; iy < n1; iy++) if (used[(std::size_t)iy*nzc + iz]) ys.push_back(iy);
        if (ys.size()) { zl.push_back(iz); yl.push_back(ys); }
      }
    }

    /// The padded real map of a cctbx fft_map, to carry on from
    void load(af::const_ref<double, af::c_grid_padded<3> > const& m) {
      SMTBX_ASSERT(m.accessor().all()[2] == m2 &&m.size() == map.size());
      std::copy(m.begin(), m.end(), map.begin());
    }

    /// One cycle: modify_electron_density, compute_structure_factors,
    /// transfer_phase_to_f_obs, compute_electron_density_map
    void run(double delta) {
      // flip over the whole padded buffer, flex.sum over it in memory order,
      // then z on every row, y on the planes the gather reads, x on the
      // columns it reads
      double s = 0;
      for (int ix = 0; ix < n0; ix++) {
        for (int iy0 = 0; iy0 < n1; iy0 += B) {
          int cnt = std::min<int>(B, n1 - iy0);
          for (int iy = iy0; iy < iy0 + cnt; iy++) {
            double* r = row(ix, iy);
#if defined(SMTBX_CF_X64) || defined(SMTBX_CF_AVX)
            // branchless (the signs are random, a branch mispredicts half the
            // time): set the sign bit where r < delta; m2 is even
            const __m128d dl = _mm_set1_pd(delta), sg = _mm_set1_pd(-0.0);
            for (int k = 0; k < m2; k += 2) {
              __m128d x = _mm_loadu_pd(r + k);
              _mm_storeu_pd(r + k, _mm_xor_pd(x, _mm_and_pd(_mm_cmplt_pd(x, dl), sg)));
            }
            for (int k = 0; k < m2; k++) s += r[k];
#else
            for (int k = 0; k < m2; k++) {
              double v = r[k] < delta ? -r[k] : r[k];
              r[k] = v; s += v;
            }
#endif
          }
          ztr(ix, iy0, cnt, true);
        }
        ytr(ix, true);
      }
      g000_ = s*fft_scale_;
      for (std::size_t a = 0; a < zl.size(); a++) xtr(a, true);
      cd const* c = cmap();
      for (int i = 0; i < no + nf; i++) {
        cd v = cj[i] ? std::conj(c[off[i]]) : c[off[i]];
        v *= fft_scale_;
        g[i] = v;
      }
      // unmeasured amplitudes follow the map but may not outgrow the measured
      for (int j = 0; j < nf; j++) {
        cd const& gf = g[no + j];
        double a = std::abs(gf);
        if (a > cap) a = cap;
        ff[j] = std::polar(a, std::arg(gf));
      }
      // oszlanyi_suto_phase_transfer, then phase_transfer in P1
      for (int i = 0; i < no; i++) {
        cd const& gi = g[i];
        double phase = std::atan2(gi.imag(), gi.real()), mo, p;
        if ((std::size_t)i < cut_) { mo = std::abs(gi); p = phase + dphi; }
        else { mo = fo[i]; p = phase; }
        fc[i] = mo*cd(std::cos(p), std::sin(p));
      }
      f000_ = g000_;
      synthesis();
    }

    double sigma() const { return sigma_; }

    /// maptbx.more_statistics skewness, a second pass only when asked for
    double skewness() {
      if (have_skew) return skew;
      skew = 0;
      if (sigma_ != 0) {
        double n = 0;
        for (int i = 0; i < n0; i++) for (int j = 0; j < n1; j++) {
          const double* r = row(i, j);
          for (int k = 0; k < n2; k++) {
            n += 1;
            double u = (r[k] - mean)/sigma_;
            skew += (u*u*u - skew)/n;
          }
        }
      }
      have_skew = true;
      return skew;
    }

    // copies: the arrays are overwritten by the next cycle
    af::shared<cd> f_calc() const { return af::shared<cd>(fc.data(), fc.data() + no); }
    af::shared<cd> f_calc_free() const { return af::shared<cd>(ff.data(), ff.data() + nf); }
    af::shared<cd> g_obs() const { return af::shared<cd>(g.data(), g.data() + no); }
    double f_000() const { return f000_; }
    double g_000() const { return g000_; }

  private:
    int n0, n1, n2, nzc, m2, no, nf;
    std::size_t cut_;
    double fft_scale_, vscale, dphi, cap, f000_, g000_, mean, sigma_, skew;
    bool have_skew;
    typedef cf_lanes::L L;
    enum { B = 4 };
    scitbx::fftpack::complex_to_complex<L, cf_lanes::LC> fx, fy;
    scitbx::fftpack::real_to_complex<L, cf_lanes::LC> fz;
    std::vector<double> map, fo, zrow;
    std::vector<L> seq, scr;
    // from_map offsets (complex units); to_map offsets of -h, +h (-1: outside)
    std::vector<long long> off, om, op;
    std::vector<char> cj;
    std::vector<cd> fc, ff, g;
    std::vector<int> zl;                // planes iz holding a reflection
    std::vector<std::vector<int> > yl;  // per plane: columns iy holding one
    cd tr0;

    cd* cmap() { return reinterpret_cast<cd*>(&map[0]); }
    double* row(int i0, int i1) { return &map[((std::size_t)i0*n1 + i1)*m2]; }
    static int mod(int h, int n) { h %= n; if (h < 0) h += n; return h; }
    long long coff(int i0, int i1, int i2) const {
      return ((long long)i0*n1 + i1)*nzc + i2;
    }

    // cnt <= B complex lines of n points, point j of line l at q[l] + j*st;
    // unused lanes read a zero pair
    void tr(scitbx::fftpack::complex_to_complex<L, cf_lanes::LC>& F, int n,
            int cnt, double* const* q, std::size_t st, bool fwd) {
      static const double z2[2] = {0, 0};
      const double* p[B];
      std::size_t sl[B];
      for (int l = 0; l < B; l++) { p[l] = l < cnt ? q[l] : z2; sl[l] = l < cnt ? st : 0; }
      L* s = &seq[0];
      for (int j = 0; j < n; j++) {
        const double *a = p[0] + j*sl[0], *b = p[1] + j*sl[1], *c = p[2] + j*sl[2], *d = p[3] + j*sl[3];
        s[2*j] = cf_lanes::lset(a[0], b[0], c[0], d[0]);
        s[2*j + 1] = cf_lanes::lset(a[1], b[1], c[1], d[1]);
      }
      if (fwd) F.transform(scitbx::fftpack::select_sign<scitbx::fftpack::forward_tag>(), s, &scr[0]);
      else F.transform(scitbx::fftpack::select_sign<scitbx::fftpack::backward_tag>(), s, &scr[0]);
      for (int j = 0; j < n; j++)
        for (int l = 0; l < cnt; l++) { q[l][j*st] = s[2*j].d[l]; q[l][j*st + 1] = s[2*j + 1].d[l]; }
    }

    // y lines of plane ix, B planes iz at a time
    void ytr(int ix, bool fwd) {
      double* q[B];
      for (std::size_t a = 0; a < zl.size(); a += B) {
        int cnt = (int)std::min<std::size_t>(B, zl.size() - a);
        for (int l = 0; l < cnt; l++) q[l] = row(ix, 0) + 2*zl[a + l];
        tr(fy, n1, cnt, q, m2, fwd);
      }
    }

    // x lines of plane zl[a], B columns at a time
    void xtr(std::size_t a, bool fwd) {
      double* q[B];
      for (std::size_t b = 0; b < yl[a].size(); b += B) {
        int cnt = (int)std::min<std::size_t>(B, yl[a].size() - b);
        for (int l = 0; l < cnt; l++) q[l] = row(0, yl[a][b + l]) + 2*zl[a];
        tr(fx, n0, cnt, q, (std::size_t)n1*m2, fwd);
      }
    }

    // z on rows iy0..iy0+cnt-1 of plane ix, the whole padded row in and out
    void ztr(int ix, int iy0, int cnt, bool fwd) {
      const double* p[B];
      for (int l = 0; l < B; l++) p[l] = l < cnt ? row(ix, iy0 + l) : &zrow[0];
      L* s = &seq[0];
      for (int k = 0; k < m2; k++) s[k] = cf_lanes::lset(p[0][k], p[1][k], p[2][k], p[3][k]);
      if (fwd) fz.forward(s, &scr[0]);
      else fz.backward(s, &scr[0]);
      for (int l = 0; l < cnt; l++) {
        double* r = row(ix, iy0 + l);
        for (int k = 0; k < m2; k++) r[k] = s[k].d[l];
      }
    }

    // fft_map: to_map into a zeroed map, F000, backward FFT, volume scaling,
    // then the mean and sigma of maptbx.statistics (Welford)
    void synthesis() {
      std::fill(map.begin(), map.end(), 0.0);
      cd* c = cmap();
      for (int i = 0; i < no + nf; i++) {
        cd term = (i < no ? fc[i] : ff[i - no])*tr0;
        if (om[i] >= 0) c[om[i]] += term;
        if (op[i] >= 0) c[op[i]] += std::conj(term);
      }
      c[0] = cd(f000_);
      for (std::size_t a = 0; a < zl.size(); a++) xtr(a, false);
      for (int ix = 0; ix < n0; ix++) ytr(ix, false);
      // counts as double: exact below 2^53, the same quotients as size_t
      double n = 0, m2s = 0;
      for (int i = 0; i < n0; i++) for (int j0 = 0; j0 < n1; j0 += B) {
        int cnt = std::min<int>(B, n1 - j0);
        ztr(i, j0, cnt, false);
        for (int j = j0; j < j0 + cnt; j++) {
          double* r = row(i, j);
          for (int k = 0; k < m2; k++) r[k] *= vscale;
          int k = 0;
          if (n == 0) { n = 1; mean = r[0]; k = 1; }
          for (; k < n2; k++) {
            n += 1;
            double x = r[k], delta = x - mean;
            mean += delta/n;
            m2s += delta*(x - mean);
          }
        }
      }
      sigma_ = std::sqrt(m2s/n);
      have_skew = false;
    }
  };

}} // namespace smtbx::ab_initio

#endif
