#ifndef SMTBX_AB_INITIO_CHARGE_FLIPPING_CYCLE_H
#define SMTBX_AB_INITIO_CHARGE_FLIPPING_CYCLE_H

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
      seq.assign(ss, 0.0); scr.assign(ss, 0.0);
      map.assign((std::size_t)n0*n1*m2, 0.0);
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
        for (int iy = 0; iy < n1; iy++) {
          double* r = row(ix, iy);
          for (int k = 0; k < m2; k++) {
            double v = r[k] < delta ? -r[k] : r[k];
            r[k] = v; s += v;
          }
          fz.forward(r, &scr[0]);
        }
        for (std::size_t a = 0; a < zl.size(); a++) ytr(ix, zl[a], true);
      }
      g000_ = s*fft_scale_;
      for (std::size_t a = 0; a < zl.size(); a++)
        for (std::size_t b = 0; b < yl[a].size(); b++) xtr(yl[a][b], zl[a], true);
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
    scitbx::fftpack::complex_to_complex<double> fx, fy;
    scitbx::fftpack::real_to_complex<double> fz;
    std::vector<double> map, fo, seq, scr;
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

    void ytr(int ix, int iz, bool fwd) {
      double* s = &seq[0];
      for (int iy = 0; iy < n1; iy++) { double* r = row(ix, iy) + 2*iz; s[2*iy] = r[0]; s[2*iy + 1] = r[1]; }
      if (fwd) fy.transform(scitbx::fftpack::select_sign<scitbx::fftpack::forward_tag>(), s, &scr[0]);
      else fy.transform(scitbx::fftpack::select_sign<scitbx::fftpack::backward_tag>(), s, &scr[0]);
      for (int iy = 0; iy < n1; iy++) { double* r = row(ix, iy) + 2*iz; r[0] = s[2*iy]; r[1] = s[2*iy + 1]; }
    }

    void xtr(int iy, int iz, bool fwd) {
      double* s = &seq[0];
      for (int ix = 0; ix < n0; ix++) { double* r = row(ix, iy) + 2*iz; s[2*ix] = r[0]; s[2*ix + 1] = r[1]; }
      if (fwd) fx.transform(scitbx::fftpack::select_sign<scitbx::fftpack::forward_tag>(), s, &scr[0]);
      else fx.transform(scitbx::fftpack::select_sign<scitbx::fftpack::backward_tag>(), s, &scr[0]);
      for (int ix = 0; ix < n0; ix++) { double* r = row(ix, iy) + 2*iz; r[0] = s[2*ix]; r[1] = s[2*ix + 1]; }
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
      for (std::size_t a = 0; a < zl.size(); a++) {
        for (std::size_t b = 0; b < yl[a].size(); b++) xtr(yl[a][b], zl[a], false);
        for (int ix = 0; ix < n0; ix++) ytr(ix, zl[a], false);
      }
      // counts as double: exact below 2^53, the same quotients as size_t
      double n = 0, m2s = 0;
      for (int i = 0; i < n0; i++) for (int j = 0; j < n1; j++) {
        double* r = row(i, j);
        fz.backward(r, &scr[0]);
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
      sigma_ = std::sqrt(m2s/n);
      have_skew = false;
    }
  };

}} // namespace smtbx::ab_initio

#endif
