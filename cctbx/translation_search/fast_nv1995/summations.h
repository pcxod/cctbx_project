#ifndef CCTBX_TRANSLATION_SEARCH_FAST_NV1995_SUMMATIONS_H
#define CCTBX_TRANSLATION_SEARCH_FAST_NV1995_SUMMATIONS_H

// Navaza, J. & Vernoslova, E. (1995). Acta Cryst. A51, 445-449.

#include <cctbx/sgtbx/space_group.h>
#include <cctbx/sgtbx/miller_ops.h>
#include <cctbx/maptbx/utils.h>
#include <cctbx/miller/f_calc_map.h>
#include <scitbx/array_family/versa_plain.h>
#include <boost/scoped_array.hpp>

namespace cctbx { namespace translation_search { namespace fast_nv1995_detail {

  // Class for accumulating the results of the summations.
  template <typename FloatType>
  class summation_accumulator
  {
    private:
      typedef std::complex<FloatType> complex_type;
      typedef typename miller::index<>::value_type mi_v_t;

    public:
      summation_accumulator() {}

      summation_accumulator(complex_type* begin,
                            miller::index<> const& n_real,
                            miller::index<> const& n_complex)
      :
        begin_(begin),
        n_0_(n_real[0]),
        n_1_(n_real[1]),
        n_real_2_(n_real[2]),
        n_complex_2_(n_complex[2])
      {
        CCTBX_ASSERT(n_complex[0] == n_real[0]);
        CCTBX_ASSERT(n_complex[1] == n_real[1]);
        CCTBX_ASSERT(n_complex[2] == n_real[2]/2+1);
      }

      void
      plus_000(FloatType const& cf)
      {
        begin_[0] += cf;
      }

      // Adds cf if -h is in the positive halfspace.
      // Ignores cf otherwise.
      // Manually optimized for best performance.
      void
      minus(miller::index<> const& h, complex_type const& cf)
      {
        mi_v_t
        h2 = (-h[2]) % n_real_2_;
        if (h2 < 0) h2 += n_real_2_;
        if (h2 < n_complex_2_) {
          mi_v_t h1 = (-h[1]) % n_1_;
          if (h1 < 0) h1 += n_1_;
          mi_v_t h0 = (-h[0]) % n_0_;
          if (h0 < 0) h0 += n_0_;
          begin_[(h0 * n_1_ + h1) * n_complex_2_ + h2] += cf;
        }
      }

      // Adds conj(cf) if +h is in the positive halfspace.
      // Adds      cf  if -h is in the positive halfspace.
      // Manually optimized for best performance.
      void
      plus_minus(miller::index<> const& h, complex_type const& cf)
      {
        mi_v_t
        h2 = h[2] % n_real_2_;
        if (h2 < 0) h2 += n_real_2_;
        if (h2 < n_complex_2_) {
          mi_v_t h1 = h[1] % n_1_;
          if (h1 < 0) h1 += n_1_;
          mi_v_t h0 = h[0] % n_0_;
          if (h0 < 0) h0 += n_0_;
          begin_[(h0 * n_1_ + h1) * n_complex_2_ + h2] += std::conj(cf);
        }
        h2 = (-h[2]) % n_real_2_;
        if (h2 < 0) h2 += n_real_2_;
        if (h2 < n_complex_2_) {
          mi_v_t h1 = (-h[1]) % n_1_;
          if (h1 < 0) h1 += n_1_;
          mi_v_t h0 = (-h[0]) % n_0_;
          if (h0 < 0) h0 += n_0_;
          begin_[(h0 * n_1_ + h1) * n_complex_2_ + h2] += cf;
        }
      }

      // a = (-h) mod n and b = h mod n per axis. Without f_part every
      // index of Eqs. (14) and (15) is -(hs0 - hs1 [+ hs2 - hs3]), i.e. a sum
      // a0 + b1 [+ a2 + b3] wrapped by one conditional subtract per axis
      // instead of the three modulos of minus() per term.
      void
      reduce(miller::index<> const& h, mi_v_t* a, mi_v_t* b) const
      {
        mi_v_t n[3] = {n_0_, n_1_, n_real_2_};
        for (std::size_t i = 0; i < 3; i++) {
          a[i] = (-h[i]) % n[i];
          if (a[i] < 0) a[i] += n[i];
          b[i] = h[i] % n[i];
          if (b[i] < 0) b[i] += n[i];
        }
      }

      // r = p + q wrapped into [0, n), p and q already reduced.
      void
      wrap(mi_v_t const* p, mi_v_t const* q, mi_v_t* r) const
      {
        r[0] = p[0] + q[0]; if (r[0] >= n_0_) r[0] -= n_0_;
        r[1] = p[1] + q[1]; if (r[1] >= n_1_) r[1] -= n_1_;
        r[2] = p[2] + q[2]; if (r[2] >= n_real_2_) r[2] -= n_real_2_;
      }

      // minus(h, lhs * rhs) with -h = p + q, p and q already reduced.
      void
      minus_reduced(mi_v_t const* p, mi_v_t const* q,
                    complex_type const& lhs, complex_type const& rhs)
      {
        mi_v_t h2 = p[2] + q[2];
        if (h2 >= n_real_2_) h2 -= n_real_2_;
        if (h2 < n_complex_2_) {
          mi_v_t h1 = p[1] + q[1];
          if (h1 >= n_1_) h1 -= n_1_;
          mi_v_t h0 = p[0] + q[0];
          if (h0 >= n_0_) h0 -= n_0_;
          begin_[(h0 * n_1_ + h1) * n_complex_2_ + h2] += lhs * rhs;
        }
      }

    protected:
      complex_type* begin_;
      mi_v_t n_0_, n_1_, n_real_2_, n_complex_2_;
  };

  // Precomputes f~(hs) (Navaza & Vernoslova (1995), p.447, following Eq. (14))
  // ftil = f_calc(hr) * exp(2*pi*i*ht)
  template <typename FloatType>
  void set_ftilde(sgtbx::space_group const& space_group,
                  miller::f_calc_map<FloatType> const& fc_map,
                  miller::index<> const& h,
                  miller::index<>* hs,
                  std::complex<FloatType>* fts)
  {
    for(std::size_t i=0;i<space_group.order_p();i++) {
      sgtbx::rt_mx s = space_group(i);
      hs[i] = h * s.r();
      fts[i] = fc_map[hs[i]] * std::polar(1.,
        2 * (h * s.t()) * scitbx::constants::pi / s.t().den());
    }
  }

  // lhs * conj(rhs)
  // With some compilers much faster than lhs * std::conj(rhs).
  template <typename T>
  inline
  std::complex<T>
  mul_conj(std::complex<T> const& lhs, std::complex<T> const& rhs)
  {
    return std::complex<T>(
      lhs.real() * rhs.real() + lhs.imag() * rhs.imag(),
      lhs.imag() * rhs.real() - lhs.real() * rhs.imag());
  }

  // Sum according to Eq. (15) of Navaza & Vernoslova (1995), p. 447.
  // Manually optimized for best performance.
  template <typename FloatType>
  void
  summation_eq15(
    sgtbx::space_group const& space_group,
    af::const_ref<miller::index<> > const& miller_indices,
    af::const_ref<FloatType> const& m,
    af::const_ref<std::complex<FloatType> > const& f_part,
    miller::f_calc_map<FloatType> const& p1_f_calc,
    summation_accumulator<FloatType>& sum)
  {
    CCTBX_ASSERT(m.size() == miller_indices.size());
    CCTBX_ASSERT(   f_part.size() == 0
                 || f_part.size() == miller_indices.size());
    typedef miller::index<> mi_t;
    typedef std::complex<FloatType> cx_t;
    std::size_t order_p = space_group.order_p();
    FloatType n_ltr = static_cast<FloatType>(space_group.n_ltr());
    bool have_f_part = f_part.size();
    boost::scoped_array<mi_t> hs_buffer(new mi_t[order_p]);
    mi_t* hs = hs_buffer.get();
    boost::scoped_array<cx_t> fts_buffer(new cx_t[order_p]);
    cx_t* fts = fts_buffer.get();
    if (!have_f_part) {
      typedef miller::index<>::value_type mi_v_t;
      boost::scoped_array<mi_v_t> ab_buffer(new mi_v_t[6*order_p]);
      mi_v_t* a = ab_buffer.get();
      mi_v_t* b = a + 3*order_p;
      for (std::size_t ih = 0; ih < miller_indices.size(); ih++) {
        FloatType mh = m[ih];
        set_ftilde(space_group, p1_f_calc, miller_indices[ih], hs, fts);
        for (std::size_t i = 0; i < order_p; i++) {
          sum.reduce(hs[i], a+3*i, b+3*i);
        }
        for (std::size_t is0 = 0; is0 < order_p; is0++) {
          cx_t mh_ftil0c = mh * std::conj(fts[is0]);
          for (std::size_t is1 = 0; is1 < order_p; is1++) {
            mi_v_t p[3];
            sum.wrap(a+3*is0, b+3*is1, p);
            cx_t mh_ftil0c_ftil1(mh_ftil0c * fts[is1]);
            for (std::size_t is2 = 0; is2 < order_p; is2++) {
              mi_v_t q[3];
              sum.wrap(p, a+3*is2, q);
              cx_t mh_ftil0c_ftil1_ftil2c(
                mul_conj(mh_ftil0c_ftil1, fts[is2]));
              for (std::size_t is3 = 0; is3 < order_p; is3++) {
                sum.minus_reduced(q, b+3*is3, mh_ftil0c_ftil1_ftil2c, fts[is3]);
              }
            }
          }
        }
      }
      return;
    }
    cx_t fpi(0);
    cx_t fpi_sq(0);
    cx_t two_fpi_sq_conj_fpi(0);
    cx_t four_conj_fpi_fpi(0);
    cx_t two_fpi(0);
    for (std::size_t ih = 0; ih < miller_indices.size(); ih++) {
      mi_t h = miller_indices[ih];
      FloatType mh = m[ih];
      set_ftilde(space_group, p1_f_calc, h, hs, fts);
      if (have_f_part) {
        fpi = f_part[ih] / n_ltr;
        fpi_sq = fpi * fpi;
        sum.plus_000(mh * std::norm(fpi_sq));
        two_fpi_sq_conj_fpi = 2. * fpi_sq * std::conj(fpi);
        four_conj_fpi_fpi = 4. * std::conj(fpi) * fpi;
        two_fpi = 2. * fpi;
      }
      for (std::size_t is0 = 0; is0 < order_p; is0++) {
        const mi_t& hm0 = hs[is0];
        cx_t mh_ftil0c = mh * std::conj(fts[is0]);
        if (have_f_part) {
          cx_t cf = mh_ftil0c * two_fpi_sq_conj_fpi;
          sum.plus_minus(hm0, cf);
        }
        for (std::size_t is1 = 0; is1 < order_p; is1++) {
          mi_t hm01(hm0 - hs[is1]);
          cx_t mh_ftil0c_ftil1(mh_ftil0c * fts[is1]);
          if (have_f_part) {
            cx_t cf = mh_ftil0c_ftil1 * four_conj_fpi_fpi;
            sum.minus(hm01, cf);
            mi_t hm0p1(hm0 + hs[is1]);
            cf = mul_conj(mh_ftil0c, fts[is1]) * fpi_sq;
            sum.plus_minus(hm0p1, cf);
          }
          for (std::size_t is2 = 0; is2 < order_p; is2++) {
            mi_t hm01p2(hm01 + hs[is2]);
            cx_t mh_ftil0c_ftil1_ftil2c(
              mul_conj(mh_ftil0c_ftil1, fts[is2]));
            if (have_f_part) {
              cx_t cf = mh_ftil0c_ftil1_ftil2c * two_fpi;
              sum.plus_minus(hm01p2, cf);
            }
            for (std::size_t is3 = 0; is3 < order_p; is3++) {
              mi_t hm01p23(hm01p2 - hs[is3]);
              cx_t cf = mh_ftil0c_ftil1_ftil2c * fts[is3];
              sum.minus(hm01p23, cf);
            }
          }
        }
      }
    }
  }

  // Sum according to Eq. (14) of Navaza & Vernoslova (1995), p. 447.
  template <typename FloatType>
  void
  summation_eq14(
    sgtbx::space_group const& space_group,
    af::const_ref<miller::index<> > const& miller_indices,
    af::const_ref<FloatType> const& m,
    af::const_ref<std::complex<FloatType> > const& f_part,
    miller::f_calc_map<FloatType> const& p1_f_calc,
    summation_accumulator<FloatType>& sum)
  {
    CCTBX_ASSERT(m.size() == miller_indices.size());
    CCTBX_ASSERT(   f_part.size() == 0
                 || f_part.size() == miller_indices.size());
    typedef miller::index<> mi_t;
    typedef std::complex<FloatType> cx_t;
    std::size_t order_p = space_group.order_p();
    FloatType n_ltr = static_cast<FloatType>(space_group.n_ltr());
    bool have_f_part = f_part.size();
    boost::scoped_array<mi_t> hs_buffer(new mi_t[order_p]);
    mi_t* hs = hs_buffer.get();
    boost::scoped_array<cx_t> fts_buffer(new cx_t[order_p]);
    cx_t* fts = fts_buffer.get();
    if (!have_f_part) {
      typedef miller::index<>::value_type mi_v_t;
      boost::scoped_array<mi_v_t> ab_buffer(new mi_v_t[6*order_p]);
      mi_v_t* a = ab_buffer.get();
      mi_v_t* b = a + 3*order_p;
      for (std::size_t ih = 0; ih < miller_indices.size(); ih++) {
        FloatType mh = m[ih];
        set_ftilde(space_group, p1_f_calc, miller_indices[ih], hs, fts);
        for (std::size_t i = 0; i < order_p; i++) {
          sum.reduce(hs[i], a+3*i, b+3*i);
        }
        for (std::size_t is0 = 0; is0 < order_p; is0++) {
          cx_t mh_ftil0c = mh * std::conj(fts[is0]);
          for (std::size_t is1 = 0; is1 < order_p; is1++) {
            sum.minus_reduced(a+3*is0, b+3*is1, mh_ftil0c, fts[is1]);
          }
        }
      }
      return;
    }
    cx_t fpi(0);
    for (std::size_t ih = 0; ih < miller_indices.size(); ih++) {
      mi_t h = miller_indices[ih];
      FloatType mh = m[ih];
      set_ftilde(space_group, p1_f_calc, h, hs, fts);
      if (have_f_part) {
        fpi = f_part[ih] / n_ltr;
        sum.plus_000(mh * std::norm(fpi));
      }
      for (std::size_t is0 = 0; is0 < order_p; is0++) {
        const mi_t& hm0 = hs[is0];
        cx_t mh_ftil0c = mh * std::conj(fts[is0]);
        if (have_f_part) {
          cx_t cf = mh_ftil0c * fpi;
          sum.plus_minus(hm0, cf);
        }
        for (std::size_t is1 = 0; is1 < order_p; is1++) {
          mi_t hm01(hm0 - hs[is1]);
          cx_t cf = mh_ftil0c * fts[is1];
          sum.minus(hm01, cf);
        }
      }
    }
  }

}}} // namespace cctbx::translation_search::fast_nv1995_detail

#endif // CCTBX_TRANSLATION_SEARCH_FAST_NV1995_SUMMATIONS_H
