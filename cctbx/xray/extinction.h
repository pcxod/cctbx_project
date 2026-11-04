#ifndef CCTBX_XRAY_EXTINCTION_H
#define CCTBX_XRAY_EXTINCTION_H
#include <cctbx/uctbx.h>
#include <scitbx/array_family/tiny.h>
#include <scitbx/sparse/matrix.h>
#include <cctbx/xray/fc_correction.h>

namespace cctbx { namespace xray {

  /* shelx-like extinction correction, shelx manual 7-7
  Fc = Fc*(1+0.001*x*Fc^2*lambda^3/sin(2theta))^0.25,
  where x is the extinction coefficient
  */
  template <typename FloatType>
  struct shelx_extinction_correction : public fc_correction<FloatType> {
    typedef fc_correction<FloatType> fc_correction_t;

    shelx_extinction_correction(uctbx::unit_cell const &u_cell,
      FloatType wavelength, FloatType value)
      : u_cell(u_cell),
      wavelength(wavelength),
      value(value),
      grad_value(0), grad_fc_mod(1),
      grad_index(-1)
    {}

    virtual FloatType compute(FloatType wl,
      miller::index<> const &h,
      FloatType fc_sq,
      bool compute_grad) const
    {
      const FloatType
        p = calc_factor(wl, h, fc_sq),
        k = 1 + p * value,
        k_sqrt = std::sqrt(k),
        rv = 1 / k_sqrt;
      if (compute_grad) {
        grad_value = -fc_sq * p / (2 * k * k_sqrt);
        // Laura Midgley, Oct 2020
        //grad_fc_mod = 0.5*(rv + rv*rv*rv);
      }
      return rv;
    }
    virtual int get_grad_index() const { return grad_index; }
    virtual af::const_ref<FloatType> get_gradients() const {
      return af::const_ref<FloatType>(&grad_value, 1);
    }
    virtual FloatType get_grad_Fc_multiplier() const { return grad_fc_mod; }
    virtual boost::shared_ptr<fc_correction_t> fork() const {
      shelx_extinction_correction *cr =
        new shelx_extinction_correction(u_cell, wavelength, value);
      cr->grad = fc_correction_t::grad;
      cr->grad_index = grad_index;
      return boost::shared_ptr<fc_correction_t>(cr);
    }
    virtual size_t n_param() const { return 1; }
  protected:
    FloatType calc_factor(FloatType wl,
      const miller::index<>& h,
      FloatType fc_sq) const
    {
      FloatType l = wl == 0 ? this->wavelength : wl;
      //const FloatType sin_2t = u_cell.sin_two_theta(h, l);
      const double sin_t_sq = u_cell.d_star_sq(h) * l * l * 0.25;
      const double sin_2t = 2 * std::sqrt(std::abs(sin_t_sq * (1 - sin_t_sq)));
      return fc_sq * l * l * l * 0.001 / (sin_2t);
    }

    uctbx::unit_cell const &u_cell;
  public:
    FloatType wavelength, value;
    mutable FloatType grad_value, grad_fc_mod;
    int grad_index;
  };

  /* anisotropic form of shelx_extinction_correction, as NoSpherA2's
  `extinction shelx aniso` (Src/core/extinction.h): x is replaced by
  x(h) = sum_p a_p X_p, X in Voigt order X11 X22 X33 X12 X13 X23 in the
  Cartesian frame of uctbx (the frame of NoSpherA2's rcm, so the tensor
  transfers as is), u the unit scattering vector and
  a = (0.5(1-u1^2), 0.5(1-u2^2), 0.5(1-u3^2), -u1u2, -u1u3, -u2u3).
  X = x*I gives x(h) = x, the isotropic correction.
  ponytail: X is not constrained to the Laue symmetry, like NoSpherA2; add
  the point-group constraints when this leaves debug mode.
  */
  template <typename FloatType>
  struct shelx_aniso_extinction_correction : public fc_correction<FloatType> {
    typedef fc_correction<FloatType> fc_correction_t;

    shelx_aniso_extinction_correction(uctbx::unit_cell const &u_cell,
      FloatType wavelength, scitbx::sym_mat3<FloatType> const &values)
      : u_cell(u_cell),
      wavelength(wavelength),
      values(values),
      grad_index(-1)
    {}

    af::tiny<FloatType, 6> coefficients(miller::index<> const &h) const {
      scitbx::vec3<double> u = u_cell.reciprocal_space_vector(h);
      const double l = u.length();
      if (l > 0) u /= l;
      af::tiny<FloatType, 6> a;
      a[0] = 0.5 * (1 - u[0] * u[0]);
      a[1] = 0.5 * (1 - u[1] * u[1]);
      a[2] = 0.5 * (1 - u[2] * u[2]);
      a[3] = -u[0] * u[1];
      a[4] = -u[0] * u[2];
      a[5] = -u[1] * u[2];
      return a;
    }

    virtual FloatType compute(FloatType wl,
      miller::index<> const &h,
      FloatType fc_sq,
      bool compute_grad) const
    {
      FloatType l = wl == 0 ? this->wavelength : wl;
      const double sin_t_sq = u_cell.d_star_sq(h) * l * l * 0.25;
      const double sin_2t = 2 * std::sqrt(std::abs(sin_t_sq * (1 - sin_t_sq)));
      const FloatType p = fc_sq * l * l * l * 0.001 / sin_2t;
      const af::tiny<FloatType, 6> a = coefficients(h);
      FloatType x = 0;
      for (int i = 0; i < 6; i++) x += a[i] * values[i];
      const FloatType k = 1 + p * x;
      // a non positive definite X can drive k through zero; NoSpherA2 then
      // leaves the reflection uncorrected too
      if (k <= 0) {
        if (compute_grad) grads.fill(0);
        return 1;
      }
      const FloatType k_sqrt = std::sqrt(k);
      if (compute_grad) {
        const FloatType g = -fc_sq * p / (2 * k * k_sqrt);
        for (int i = 0; i < 6; i++) grads[i] = g * a[i];
      }
      return 1 / k_sqrt;
    }
    virtual int get_grad_index() const { return grad_index; }
    virtual af::const_ref<FloatType> get_gradients() const {
      return grads.const_ref();
    }
    virtual FloatType get_grad_Fc_multiplier() const { return 1; }
    virtual boost::shared_ptr<fc_correction_t> fork() const {
      shelx_aniso_extinction_correction *cr =
        new shelx_aniso_extinction_correction(u_cell, wavelength, values);
      cr->grad = fc_correction_t::grad;
      cr->grad_index = grad_index;
      return boost::shared_ptr<fc_correction_t>(cr);
    }
    virtual size_t n_param() const { return 6; }
  protected:
    uctbx::unit_cell const &u_cell;
  public:
    FloatType wavelength;
    scitbx::sym_mat3<FloatType> values;
    mutable af::tiny<FloatType, 6> grads;
    int grad_index;
  };

  /* shelx-like extinction correction, shelx manual
  Fc_sq = Fc_sq*(1 - g*exp(-8 * PI^2 * U * stol_sq))^2
  coefficient
  */
  template <typename FloatType>
  struct shelx_SWAT_correction : public fc_correction<FloatType> {
    typedef fc_correction<FloatType> fc_correction_t;

    shelx_SWAT_correction(uctbx::unit_cell const& u_cell,
      FloatType g, FloatType U)
      : u_cell(u_cell),
      grad_index(-1)
    {
      values[0] = g;
      values[1] = U;
    }

    // for testing
    FloatType calc(miller::index<> const& h, FloatType Fsq,
      FloatType g, FloatType U) const
    {
      FloatType stol_sq = u_cell.stol_sq(h);
      return Fsq * std::pow(1-g*std::exp(-scitbx::constants::eight_pi_sq * stol_sq *U), 2);
    }

    virtual FloatType compute(FloatType wl,
      miller::index<> const& h,
      FloatType fc_sq,
      bool compute_grad) const
    {
      using namespace scitbx::constants;
      FloatType stol_sq = u_cell.stol_sq(h);
      FloatType f = eight_pi_sq * stol_sq;
      FloatType ef = exp(-f * values[1]);
      FloatType rv = 1 - values[0] * ef;
      if (compute_grad) {
        FloatType dm = 2 * fc_sq * rv * ef;
        grads[0] = -dm;
        grads[1] = dm * values[0] * f;
        /* Testing derivatives shows consistensy with analytical expressions above
        FloatType eps = 1e-6;
        FloatType v1 = calc(h, fc_sq, values[0] - eps, values[1]);
        FloatType v2 = calc(h, fc_sq, values[0] + eps, values[1]);
        FloatType diff = (v2 - v1) / (2*eps);

        v1 = calc(h, fc_sq, values[0], values[1] - eps);
        v2 = calc(h, fc_sq, values[0], values[1] + eps);
        diff = (v2 - v1) / (2 * eps);
        v1 = 0; // allow for a breakpoint here
        */
      }
      return rv*rv;
    }

    FloatType get_g() const { return values[0]; }
    void set_g(FloatType v) { values[0] = v; }

    FloatType get_U() const { return values[1]; }
    void set_U(FloatType v) { values[1] = v; }

    virtual int get_grad_index() const { return grad_index; }
    virtual af::const_ref<FloatType> get_gradients() const {
      return grads.const_ref();
    }
    virtual FloatType get_grad_Fc_multiplier() const { return 1; }
    virtual boost::shared_ptr<fc_correction_t> fork() const {
      shelx_SWAT_correction* cr =
        new shelx_SWAT_correction(u_cell, values[0], values[1]);
      cr->grad = fc_correction_t::grad;
      cr->grad_index = grad_index;
      return boost::shared_ptr<fc_correction_t>(cr);
    }
    virtual size_t n_param() const { return 2; }
  protected:
    uctbx::unit_cell const &u_cell;
  public:
    af::tiny<FloatType, 2> values; // g, U
    mutable af::tiny<FloatType, 2> grads; // d_Fc_d_g, d_Fc_d_U;
    int grad_index;
  };
}} // namespace cctbx::xray

#endif // GUARD
