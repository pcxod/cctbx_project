#pragma once
#include <smtbx/ED/dyn_calculator.h>
namespace smtbx { namespace ED
{
  using namespace cctbx;
  /* As in Electron Microscopy of Thin Crystals by Hirsch, Peter B., etc., et al
  (ISBN: 9780882753768)
  */
  template <typename FloatType>
  class dyn_calculator_def : public a_dyn_calculator<FloatType> {
  public:
    ED_UTIL_TYPEDEFS;
    typedef a_dyn_calculator<FloatType> parent_t;

    dyn_calculator_def(const af::shared<miller::index<> >& indices,
      const cmat_t& mat_Ug,
      cart_t const& K,
      mat3_t const& RMf,
      cart_t const& N,
      FloatType thickness)
      : parent_t(indices, mat_Ug, K, RMf, N, thickness)
    {}

    dyn_calculator_def(const af::shared<miller::index<> >& indices,
      cart_t const& K, const cart_t& N,
      FloatType thickness)
      : parent_t(indices, K, N, thickness)
    {}

    virtual af::shared<complex_t> calc_amps(size_t num, bool include_incident) {
      using namespace fast_linalg;
      const size_t n_beams = this->A.accessor().n_columns();
      af::shared<FloatType> ev(n_beams);
      // heev replaces A with column-wise eigenvectors
      lapack_int info = heev(LAPACK_ROW_MAJOR, 'V', LAPACK_UPPER, n_beams,
        this->A.begin(), n_beams, ev.begin());
      SMTBX_ASSERT(!info)(info);
      const FloatType Kl = this->K.length();
      const complex_t exp_k(0, scitbx::constants::pi * this->thickness / Kl);
      af::shared<complex_t> im(n_beams);
      // diagonal by first row of A*
      for (size_t i = 0; i < n_beams; i++) {
        im[i] = std::exp(ev[i] * exp_k) * std::conj(this->A(0, i));
      }
      af::shared<complex_t> rv(num);
      const size_t off = include_incident ? 0 : 1;
      for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < n_beams; j++) {
          rv[i] += this->A(i + off, j) * im[j];
        }
      }
      return rv;
    }

    virtual complex_t calc_amps_1(size_t idx) {
      using namespace fast_linalg;
      const size_t n_beams = this->A.accessor().n_columns();
      af::shared<FloatType> ev(n_beams);
      // heev replaces A with column-wise eigenvectors
      lapack_int info = heev(LAPACK_ROW_MAJOR, 'V', LAPACK_UPPER, n_beams,
        this->A.begin(), n_beams, ev.begin());
      SMTBX_ASSERT(!info)(info);
      const FloatType Kl = this->K.length();
      const complex_t exp_k(0, scitbx::constants::pi * this->thickness / Kl);
      complex_t res;
      for (size_t i = 0; i < n_beams; i++) {
        res += this->A(idx, i) * std::exp(ev[i] * exp_k) * std::conj(this->A(0, i));
      }
      return res;
    }

    /* to compute d(exp(tA))/dp using approach as described here
    Bernoulli 9(5), 2003, 895�919
    */
    virtual af::shared<complex_t> calc_amps_ext(
      af::shared<cmat_t> const& Ds_kin,
      bool grad_thickness,
      mat_t& D_dyn,
      size_t num)
    {
      using namespace fast_linalg;
      const size_t n_beams = this->A.accessor().n_columns();
      af::shared<FloatType> ev(n_beams);
      // heev replaces A with column-wise eigenvectors
      lapack_int info = heev(LAPACK_ROW_MAJOR, 'V', LAPACK_UPPER, n_beams,
        this->A.begin(), n_beams, ev.begin());
      SMTBX_ASSERT(!info)(info);
      cmat_t A_cjt(af::mat_grid(n_beams, n_beams));
      const FloatType Kl = this->K.length();
      const complex_t exp_k(0, scitbx::constants::pi * this->thickness / Kl),
        k_dt(0, scitbx::constants::pi / Kl);
      af::shared<complex_t> exps(n_beams), im(n_beams),
        im_dt(n_beams);
      for (size_t i = 0; i < n_beams; i++) {
        A_cjt(i, i) = std::conj(this->A(i, i));
        for (size_t j = i + 1; j < n_beams; j++) {
          A_cjt(i, j) = std::conj(this->A(j, i));
          A_cjt(j, i) = std::conj(this->A(i, j));
        }
      }
      for (size_t i = 0; i < n_beams; i++) {
        exps[i] = std::exp(ev[i] * exp_k);
        // diagonal by first row of A*
        im[i] = exps[i] * A_cjt(i, 0);
        if (grad_thickness) {
          // diagonal_dt by first row of A* (dI/dThickness)
          im_dt[i] = exps[i] * ev[i] * k_dt * A_cjt(i, 0);
        }
      }
      // !need only num rows!
      cmat_t G(af::mat_grid(n_beams, n_beams));
      for (size_t i = 0; i < n_beams; i++) {
        //G(i, i) = exps[i];
        G(i, i) = exps[i] * exp_k;
        for (size_t j = i + 1; j < n_beams; j++) {
          G(i, j) = (exps[i] - exps[j]) / (ev[i] - ev[j]);
          G(j, i) = G(i, j);
        }
      }
      // last column - dI_dT
      size_t d_T_off = Ds_kin.size();
      D_dyn.resize(af::mat_grid(num, d_T_off + (grad_thickness ? 1 : 0)));

      af::shared<complex_t> rv(num); // complex amplitudes
      for (size_t i = 1; i <= num; i++) {
        complex_t ci = 0, dt = 0;
        for (size_t j = 0; j < n_beams; j++) {
          ci += this->A(i, j) * im[j];
          if (grad_thickness) {
            dt += this->A(i, j) * im_dt[j];
          }
        }
        rv[i - 1] = ci;
        if (grad_thickness) {
          D_dyn(i - 1, d_T_off) = 2 * (ci.real() * dt.real() + ci.imag() * dt.imag());
        }
      }

      for (size_t pi = 0; pi < d_T_off; pi++) {
        cmat_t V = af::matrix_multiply(
          af::matrix_multiply(A_cjt.const_ref(), Ds_kin[pi].const_ref()).const_ref(),
          this->A.const_ref()
        );

        // Hadamard product of G x V by A* first column into dI_dP
        af::shared<complex_t> df(n_beams);
        for (size_t i = 0; i < n_beams; i++) {
          for (size_t j = 0; j < n_beams; j++) {
            df[i] += G(i, j) * V(i, j) * A_cjt(j, 0);
          }
        }
        df = af::matrix_multiply(this->A.const_ref(), df.const_ref());
        // copy result to output (dI/dp - > |CI|^2)
        for (size_t i = 0; i < num; i++) {
          complex_t dp = df[i + 1];
          D_dyn(i, pi) = 2 * (rv[i].real() * dp.real() +
            rv[i].imag() * dp.imag());
        }
      }
      return rv;

    }

    virtual complex_t calc_amps_ext_1(
      af::shared<cmat_t> const& Ds_kin,
      bool grad_thickness,
      mat_t& D_dyn,
      size_t idx)
    {
      using namespace fast_linalg;
      const size_t n_beams = this->A.accessor().n_columns();
      af::shared<FloatType> ev(n_beams);
      // heev replaces A with column-wise eigenvectors
      lapack_int info = heev(LAPACK_ROW_MAJOR, 'V', LAPACK_UPPER, n_beams,
        this->A.begin(), n_beams, ev.begin());
      SMTBX_ASSERT(!info)(info);
      cmat_t A_cjt(af::mat_grid(n_beams, n_beams));
      const FloatType Kl = this->K.length();
      const complex_t exp_k(0, scitbx::constants::pi * this->thickness / Kl),
        k_dt(0, scitbx::constants::pi / Kl);
      af::shared<complex_t> exps(n_beams), im(n_beams),
        im_dt(n_beams);
      for (size_t i = 0; i < n_beams; i++) {
        A_cjt(i, i) = std::conj(this->A(i, i));
        for (size_t j = i + 1; j < n_beams; j++) {
          A_cjt(i, j) = std::conj(this->A(j, i));
          A_cjt(j, i) = std::conj(this->A(i, j));
        }
      }
      for (size_t i = 0; i < n_beams; i++) {
        exps[i] = std::exp(ev[i] * exp_k);
        // diagonal by first row of A*
        im[i] = exps[i] * A_cjt(i, 0);
        if (grad_thickness) {
          // diagonal_dt by first row of A* (dI/dThickness)
          im_dt[i] = exps[i] * ev[i] * k_dt * A_cjt(i, 0);
        }
      }
      // !need only num rows!
      cmat_t G(af::mat_grid(n_beams, n_beams));
      for (size_t i = 0; i < n_beams; i++) {
        //G(i, i) = exps[i];
        G(i, i) = exps[i] * exp_k;
        for (size_t j = i + 1; j < n_beams; j++) {
          G(i, j) = (exps[i] - exps[j]) / (ev[i] - ev[j]);
          G(j, i) = G(i, j);
        }
      }
      // last column - dI_dT
      size_t d_T_off = Ds_kin.size();
      D_dyn.resize(af::mat_grid(1, d_T_off + (grad_thickness ? 1 : 0)));

      complex_t rv; // complex amplitudes
      {
        complex_t dt = 0;
        for (size_t j = 0; j < n_beams; j++) {
          rv += this->A(idx, j) * im[j];
          if (grad_thickness) {
            dt += this->A(idx, j) * im_dt[j];
          }
        }
        if (grad_thickness) {
          D_dyn[d_T_off] = 2 * (rv.real() * dt.real() + rv.imag() * dt.imag());
        }
      }

      for (size_t pi = 0; pi < d_T_off; pi++) {
        cmat_t V = af::matrix_multiply(
          af::matrix_multiply(A_cjt.const_ref(), Ds_kin[pi].const_ref()).const_ref(),
          this->A.const_ref());

        // Hadamard product of G x V by A* first column into dI_dP
        af::shared<complex_t> df(n_beams);
        for (size_t i = 0; i < n_beams; i++) {
          for (size_t j = 0; j < n_beams; j++) {
            df[i] += G(i, j) * V(i, j) * A_cjt(j, 0);
          }
        }
        complex_t dp;
        for (size_t j = 0; j < n_beams; j++) {
          dp += this->A(idx, j) * df[j];
        }
        // copy result to output (dI/dp - > |CI|^2)
        D_dyn(0, pi) = 2 * (rv.real() * dp.real() + rv.imag() * dp.imag());
      }
      return rv;
    }

    a_dyn_calculator<FloatType>& build() {
      const FloatType Kl = this->K.length();
      const size_t n_beams = this->indices.size() + 1; // g0+
      for (size_t i = 1; i < n_beams; i++) {
        miller::index<> h = this->indices[i - 1];
        cart_t K_g = this->K + this->RMf * cart_t(h[0], h[1], h[2]);
        FloatType K_g_l = K_g.length();
        this->A(i, i) = Kl * Kl - K_g_l * K_g_l;
        FloatType cos_theta = K_g * this->N / K_g_l;
        this->A(i, 0) *= cos_theta;
        this->A(0, i) *= cos_theta;
        for (size_t j = i; j < n_beams; j++) {
          this->A(i, j) *= cos_theta;
          this->A(j, i) *= cos_theta;
        }
      }
      return *this;
    }
  };

}}
