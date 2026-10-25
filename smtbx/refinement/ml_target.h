#ifndef SMTBX_REFINEMENT_ML_TARGET_H
#define SMTBX_REFINEMENT_ML_TARGET_H

/// Maximum-likelihood refinement expressed as weighted least squares.
/** The build loop passes its accumulator a computed observable, its gradient,
    an observation and a weight, per reflection. That interface carries no
    reflection index, so per-reflection distribution parameters cannot be
    looked up downstream. Instead an *effective* observation and weight are
    substituted for the measured ones, chosen so that the normal equations
    assembled are those of the likelihood.

    For the amplitude target (Rice; cctbx/xray/targets/mlf.h) the derivative
    separates as
    \f$ \partial T/\partial F_c
        = (\partial T/\partial |F_c|)\, \overline{F_c}/|F_c| \f$,
    the real scalar \f$ \partial T/\partial |F_c| \f$ being the sum of the two
    terms mlf.h names d1 and d2. With \f$ a = \alpha k \f$,
    \f$ b = \beta k^2 \f$ and \f$ X = I_1/I_0 \f$,

    \f[ w_{\mathrm{eff}} = \frac{2a^2}{\epsilon b} \quad\text{(acentric)},
        \qquad \frac{a^2}{\epsilon b} \quad\text{(centric)} \f]
    \f[ y_{\mathrm{eff}}
        = \frac{f_o}{a} X\!\left(\frac{2 a f_o |F_c|}{\epsilon b}\right)
          \quad\text{(acentric)},
        \qquad \frac{f_o}{a}\tanh\!\left(\frac{a f_o |F_c|}{\epsilon b}\right)
          \quad\text{(centric)} \f]

    satisfy \f$ w_{\mathrm{eff}}(y_{\mathrm{eff}} - |F_c|)
                = -\partial T/\partial |F_c| \f$ exactly, so the ordinary
    Gauss-Newton assembly yields the likelihood gradient. This is the reduction
    used by Refmac: the likelihood becomes least squares against an expected
    amplitude re-derived each cycle.

    \f$ w_{\mathrm{eff}} \f$ is the positive part of the curvature only. That
    keeps the normal matrix positive definite, as required by the Cholesky
    decomposition in linear_ls::solve and by the non-negative weight that
    matrix::rank_n_update and store_design_row both assume. The omitted term is
    \f$ (2 a f_o/\epsilon b)^2 X'(u) \f$, of relative size
    \f$ O(1/|F_c|^2) \f$ against the retained one.

    The observable remains \f$ u = |F_c|^2 \f$. Since
    \f$ \partial u/\partial p = 2|F_c|\, \partial |F_c|/\partial p \f$, taking

    \f[ w_u = \frac{w_{\mathrm{eff}}}{4u}, \qquad
        y_u = 2|F_c| y_{\mathrm{eff}} - u \f]

    reproduces the \f$ |F_c| \f$-space normal equations identically,

    \f[ \sum w_u \frac{\partial u}{\partial p}
                 \frac{\partial u}{\partial p}^T
        = \sum w_{\mathrm{eff}} \frac{\partial |F_c|}{\partial p}
                                \frac{\partial |F_c|}{\partial p}^T \f]
    \f[ \sum w_u (y_u - u) \frac{\partial u}{\partial p}
        = \sum w_{\mathrm{eff}} (y_{\mathrm{eff}} - |F_c|)
               \frac{\partial |F_c|}{\partial p} \f]

    leaving the observable, the gradients, the meaning of the scale factor and
    the existing bindings unchanged.

    Twinning, a non-trivial Fc correction and non-unit measured scales are
    refused by the builder; see check_maximum_likelihood() in least_squares.h.

    References:
     - Murshudov, Vagin & Dodson (1997) Acta Cryst. D53, 240-255.
     - Pannu & Read (1996) Acta Cryst. A52, 659-668.
     - Read (1990) Acta Cryst. A46, 900-912.
*/

/* mlf.h is written to be included from cctbx/xray/boost_python/targets.cpp and
   does not include its own prerequisites, hence the two headers before it.
 */
#include <cctbx/xray/targets.h>
#include <cctbx/xray/targets/common_results.h>
#include <cctbx/xray/targets/mlf.h>

#include <smtbx/error.h>
#include <scitbx/array_family/ref.h>
#include <scitbx/array_family/shared.h>
#include <scitbx/math/bessel.h>
#include <scitbx/constants.h>
#include <cmath>
#include <limits>
#include <vector>

namespace smtbx { namespace refinement { namespace least_squares {

  /// Per-reflection distribution parameters for the likelihood targets.
  /** A default-constructed instance is inactive, denoting ordinary least
      squares and leaving that path untouched. Every array is indexed by the
      position of the reflection in the observation list, so a worker needs
      only the index it already holds; all are read-only during a build and
      therefore need no per-thread copy.
   */
  template <typename FloatType>
  struct ml_data {
    /* Owning copies rather than references: instances are constructed from
       Python, where the arrays passed may be temporaries whose lifetime ends
       before the build does.
     */
    scitbx::af::shared<FloatType> alpha, beta, epsilon;
    scitbx::af::shared<bool> centric;
    /// Reflections excluded from the target sum; empty excludes none.
    /** alpha and beta are estimated on this set, so including it in the sum
        as well would make that estimate self-referential, biasing alpha up
        and beta down and hence every effective weight.
     */
    scitbx::af::shared<bool> free_flags;
    /* Activity is a flag rather than a test on alpha.size(): the arrays arrive
       as scitbx::af::const_ref, whose default constructor initialises nothing,
       so size() on a default-constructed one is indeterminate.
     */
    bool active_;
    /// Selects the intensity target over the amplitude one.
    bool intensity_;
    /* The scale the *likelihood* works at, which is not the crystallographic
       one. mlf.h forms a = alpha k, so alpha and k have to come from the same
       estimation; the builder's scale_factor is the Fo/Fc scale that R1 and
       the difference map want, and feeding that here would apply it a second
       time on top of the one alpha already carries.
     */
    FloatType scale_factor_;

    ml_data() : active_(false), intensity_(false), scale_factor_(1) {}

    ml_data(scitbx::af::const_ref<FloatType> const &alpha_,
            scitbx::af::const_ref<FloatType> const &beta_,
            scitbx::af::const_ref<FloatType> const &epsilon_,
            scitbx::af::const_ref<bool> const &centric_,
            bool intensity = false,
            scitbx::af::const_ref<bool> const &free_flags_
              = scitbx::af::const_ref<bool>(0, 0),
            FloatType scale_factor = 1)
      : alpha(alpha_.begin(), alpha_.end()),
        beta(beta_.begin(), beta_.end()),
        epsilon(epsilon_.begin(), epsilon_.end()),
        centric(centric_.begin(), centric_.end()),
        free_flags(free_flags_.begin(), free_flags_.end()),
        active_(true),
        intensity_(intensity),
        scale_factor_(scale_factor > 0 ? scale_factor : FloatType(1))
    {}

    /// Whether this is a maximum-likelihood refinement at all.
    bool active() const { return active_; }
    /// The scale alpha and beta were estimated against.
    FloatType scale_factor() const { return scale_factor_; }
    /// True for the intensity target, which convolves with the measured sigma.
    bool is_intensity() const { return intensity_; }
    /// Whether reflection i_h is held out of the target sum.
    bool is_free(std::size_t i_h) const {
      return free_flags.size() != 0 && free_flags[i_h];
    }

    /// Whether the arrays describe exactly n_reflections reflections.
    /** free_flags may also be empty, denoting that none are held out.
     */
    bool is_consistent_with(std::size_t n_reflections) const {
      if (!active()) return true;
      return alpha.size() == n_reflections
          && beta.size() == n_reflections
          && epsilon.size() == n_reflections
          && centric.size() == n_reflections
          && (free_flags.size() == 0 || free_flags.size() == n_reflections);
    }
  };

  /// The effective observation and weight for one reflection, in F^2 space.
  /** \param fo_sq the measured intensity
      \param observable the computed intensity \f$ |F_c|^2 \f$
      \param yo_eff set to the effective observation \f$ y_u \f$
      \param w_eff set to the effective weight \f$ w_u \f$
      \return false if the reflection cannot contribute, in which case the
              caller gives it zero weight. The guards match those in mlf.h, so
              a reflection refused here is one for which the target would
              return zero anyway.
   */
  /// Whether the F^2-space conversion is safe for this reflection.
  /** Both effective observations divide by the *calculated* intensity, and a
      reflection the model puts at zero makes that enormous. Shared so the two
      targets cannot disagree about which reflections they drop - the amplitude
      one was guarded first and the intensity one was not, which left MLI
      unchanged and looked like the guard having no effect at all.
   */
  template <typename FloatType>
  inline bool ml_intensity_is_usable(FloatType fo_sq, FloatType sigma_fo_sq,
                                     FloatType observable)
  {
    /* Against the larger of the measurement and its uncertainty. A purely
       relative test on fo_sq alone does not bound the weight: where Fo^2 is
       itself tiny the threshold shrinks with it and a |Fc|^2 of 1e-25 still
       clears a bar of 1e-28. That left the worst weights at 1e29 after the
       first attempt, with the 99th percentile already down at 16.7.

       sigma is the scale that does not vanish - a measured reflection has a
       real uncertainty whatever its intensity came out at - so it is what
       makes this an absolute bound in the units of the data.
     */
    FloatType const scale = std::max(fo_sq, sigma_fo_sq);
    return observable > 1e-6*scale;
  }

  template <typename FloatType>
  bool ml_effective_observation(
    FloatType fo_sq,
    FloatType sigma_fo_sq,
    FloatType observable,
    FloatType alpha,
    FloatType beta,
    FloatType scale_factor,
    FloatType epsilon,
    bool centric,
    FloatType &yo_eff,
    FloatType &w_eff)
  {
    yo_eff = observable;
    w_eff = 0;
    if (scale_factor <= 0) scale_factor = 1;
    // mlf.h's own guards, so that the two agree about which reflections count
    if (alpha <= 0 || beta <= 1e-3 || fo_sq <= 0 || observable <= 0) {
      return false;
    }
    if (epsilon <= 0) {
      return false;
    }
    FloatType const a = alpha*scale_factor;
    FloatType const b = beta*scale_factor*scale_factor;
    FloatType const eb = epsilon*b;
    FloatType const fo = std::sqrt(fo_sq);
    FloatType const modulus_fc = std::sqrt(observable);
    FloatType w_amplitude, yo_amplitude;
    if (centric) {
      w_amplitude = a*a/eb;
      yo_amplitude = (fo/a)*std::tanh(a*fo*modulus_fc/eb);
    }
    else {
      w_amplitude = 2*a*a/eb;
      yo_amplitude = (fo/a)*scitbx::math::bessel::i1_over_i0(
        2*a*fo*modulus_fc/eb);
    }
    /* Into F^2 space; see the header comment for why this is exact.

       The division is by the *calculated* intensity, and a reflection the
       model puts at zero makes it enormous: measured on a small molecule, 193 of 6819
       reflections have |Fc|^2 below 1e-6 and the smallest is 6e-31, which
       turned the largest weight into 6e29 and the objective into 2.5e22
       against least squares' 4e-3.

       Zero is the right weight there, not infinity. The conversion exists
       because du/dp = 2|Fc| d|Fc|/dp, so as |Fc| goes to zero the row of the
       design matrix vanishes too and the reflection stops saying anything
       about the parameters through this route. The product w (x) row (x) row
       stays finite - it is w_amplitude g (x) g - but w and the row are kept
       apart, and sum w yo^2, which normalises the objective, is not protected
       by that cancellation.

       The threshold is relative to what was measured, so it means "the model
       says this reflection is absent" rather than a fixed number of electrons.
       It wants review against a structure with genuine systematic absences.
     */
    if (!ml_intensity_is_usable(fo_sq, sigma_fo_sq, observable)) {
      w_eff = 0;
      yo_eff = observable;
      return false;
    }
    w_eff = w_amplitude/(4*observable);
    yo_eff = 2*modulus_fc*yo_amplitude - observable;
    return true;
  }

  /// Intensity-based maximum likelihood: the Rice model convolved with the
  /// experimental error.
  /** This is not a change of variable. Expressing the Rice distribution in
      \f$ I_o = F_o^2 \f$ rather than \f$ F_o \f$ introduces the Jacobian
      \f$ |dF_o/dI_o| = 1/(2F_o) \f$, which is independent of \f$ F_c \f$ and
      so alters no derivative with respect to the model; a target formed that
      way refines identically to the amplitude one.

      The intensity likelihood proper convolves the model distribution with the
      experimental error, which allows a negative measured intensity to be used
      rather than rejected and requires no French-Wilson conversion:

      \f[ P(I_o) = \int_0^{\infty}
                   P_{\mathrm{Rice}}(I_t \mid F_c)\,
                   N(I_o; I_t, \sigma^2)\, dI_t \f]

      with, for acentric reflections,

      \f[ P_{\mathrm{Rice}}(I_t \mid F_c)
          = \frac{1}{\epsilon b}
            \exp\!\left(-\frac{I_t + (a F_c)^2}{\epsilon b}\right)
            I_0\!\left(\frac{2 a \sqrt{I_t} F_c}{\epsilon b}\right) \f]

      and for centric ones the Woolfson distribution of F carried over to
      \f$ I_t = F^2 \f$, which brings the factor \f$ 1/(2\sqrt{I_t}) \f$:

      \f[ P(I_t \mid F_c)
          = \frac{1}{2\sqrt{I_t}}\sqrt{\frac{2}{\pi \epsilon b}}
            \exp\!\left(-\frac{I_t + (a F_c)^2}{2 \epsilon b}\right)
            \cosh\!\left(\frac{a \sqrt{I_t} F_c}{\epsilon b}\right) \f]

      The acentric density used to be applied to centric reflections too -
      which in a centrosymmetric space group is every reflection - giving a
      gradient that did not tend to the amplitude target's as sigma -> 0
      (measured: up to 80 times off, against 4e-6 for acentric ones).

      It departs from the amplitude target by an amount which depends on
      \f$ F_c \f$, grows with \f$ \sigma^2 \f$ and vanishes as
      \f$ \sigma \to 0 \f$.

      How the integral is evaluated is described at mli_quadrature.

      Reference: Pannu & Read (1996) Acta Cryst. A52, 659-668.
   */
  /// How the integral over the true intensity is evaluated.
  /** It used to be one 20-node Gauss-Hermite rule centred on the measurement,
      \f$ I_t = I_o + \sqrt{2}\,\sigma x \f$. That rule assumes the integrand
      is the measurement Gaussian times something smooth on the scale of
      sigma, and two common cases break that:

      - **Weak reflections.** The centric density has an integrable
        \f$ 1/\sqrt{I_t} \f$ singularity at zero, and both densities are
        steep there. When the measurement reaches zero, as it does for every
        near-absent reflection, no rule in \f$ I_t \f$ resolves it.
      - **A model distribution narrower than the measurement**
        (sigma comparable to or larger than the spread of
        \f$ P(I_t \mid F_c) \f$). Then the nodes, spaced on the scale of
        sigma, step over the peak.

      Measured against an adaptive Gauss-Kronrod reference on the reflections
      of a refined synthetic structure: 9 to 19 percent of reflections had
      dT/d|Fc| wrong by more than 1 percent of \f$ a^2|F_c|/\epsilon b \f$ (the
      effective observation relative to \f$ |F_c| \f$), some by factors of
      several and with the wrong sign.

      Now there are two branches, chosen per reflection:

      - **Away from zero** (the posterior mean is more than six posterior
        standard deviations above zero): Gauss-Hermite in \f$ I_t \f$, centred
        on the product of the measurement Gaussian and a Gaussian with the
        model distribution's mean \f$ a^2F_c^2 + \epsilon b \f$ and variance,
        with the ratio to that proposal carried in the weights. It follows
        whichever factor is narrower.
      - **Near zero, or a measurement more than four combined standard
        deviations from the model's mean and not much narrower than the
        model's spread:** Gauss-Legendre in
        \f$ u = \sqrt{I_t} \f$, where
        \f$ P(I_t)\,dI_t = P_F(u)\,du \f$ is smooth (Rice or Woolfson in
        amplitude). The rule spans the intersection of the two factors'
        supports (each taken to \f$ e^{-40} \f$ of its own peak), or their
        union where they do not overlap. For such an outlier the posterior
        sits in the tails of both, so it takes the union in four panels.

      With 8 Hermite and 24 Legendre nodes, the worst error in dT/d|Fc| over
      the effective-observation scale \f$ |d| + a^2|F_c|/\epsilon b \f$ was
      1.6e-4 on three refined synthetic data sets (it had been 5.5) and 1e-3
      over 2,000 random cases (sigma/Io from 0.003 to beyond 1, model spread
      from a thousandth to half of Fc^2). On 3,000 cases with models far off
      the data it was 8e-3, where the old rule failed outright. Per pass over
      the refined data it is 20-37 percent faster than the old rule, because
      most reflections take the 8-node branch. With 10 + 32 nodes the error
      falls to the level of the Bessel approximations in scitbx/math/bessel.h
      (3e-7) at about the old rule's cost.

      The old warning against trimming nodes still applies to the rule it was
      about: centred on the measurement, fewer nodes are worse. Here the
      Hermite rule is centred on the posterior and the weak cases go to the
      Legendre rule, so it does not.
   */
  namespace mli_quadrature {
    inline int n_hermite() { return 8; }

    inline double const* hermite_nodes() {
      static const double x[8] = {
        -2.9306374202572441, -1.981656756695843, -1.1571937124467802,
        -0.38118699020732211, 0.38118699020732211, 1.1571937124467802,
        1.981656756695843, 2.9306374202572441 };
      return x;
    }

    /// Weights of the Hermite rule, already divided by sqrt(pi).
    inline double const* hermite_weights() {
      static const double w[8] = {
        0.0001126145383753679, 0.009635220120788263, 0.117239907661759,
        0.37301225767907747, 0.37301225767907747, 0.117239907661759,
        0.009635220120788263, 0.0001126145383753679 };
      return w;
    }

    inline int n_legendre() { return 24; }

    inline double const* legendre_nodes() {
      static const double x[24] = {
        -0.99518721999702131, -0.97472855597130947, -0.9382745520027328,
        -0.88641552700440107, -0.82000198597390295, -0.74012419157855436,
        -0.64809365193697555, -0.54542147138883956, -0.43379350762604513,
        -0.3150426796961634, -0.19111886747361631, -0.06405689286260563,
        0.06405689286260563, 0.19111886747361631, 0.3150426796961634,
        0.43379350762604513, 0.54542147138883956, 0.64809365193697555,
        0.74012419157855436, 0.82000198597390295, 0.88641552700440107,
        0.9382745520027328, 0.97472855597130947, 0.99518721999702131 };
      return x;
    }

    inline double const* legendre_weights() {
      static const double w[24] = {
        0.01234122979998869, 0.028531388628933559, 0.044277438817419412,
        0.05929858491543636, 0.073346481411080161, 0.086190161531953205,
        0.097618652104113926, 0.10744427011596556, 0.11550566805372552,
        0.12167047292780329, 0.12583745634682825, 0.12793819534675202,
        0.12793819534675202, 0.12583745634682825, 0.12167047292780329,
        0.11550566805372552, 0.10744427011596556, 0.097618652104113926,
        0.086190161531953205, 0.073346481411080161, 0.05929858491543636,
        0.044277438817419412, 0.028531388628933559, 0.01234122979998869 };
      return w;
    }

    /// ln of P(I_t) dI_t/du and its derivative in |Fc|, at u = sqrt(I_t) > 0.
    /** Rice (acentric) or Woolfson (centric) in amplitude, with every
        constant, so that the target value is the log-likelihood proper.
     */
    template <typename FloatType>
    inline void ln_p_amplitude(FloatType u, FloatType a, FloatType modulus_fc,
                               FloatType eb, bool centric,
                               FloatType &ln_p, FloatType &d_ln_p)
    {
      FloatType const sq = u*u + a*a*modulus_fc*modulus_fc;
      if (centric) {
        // ln cosh(z) = |z| + ln(1 + exp(-2|z|)) - ln 2, safe for large z
        FloatType const z = a*u*modulus_fc/eb;
        FloatType const az = std::abs(z);
        ln_p = 0.5*std::log(2/(scitbx::constants::pi*eb)) - sq/(2*eb)
               + az + std::log1p(std::exp(-2*az)) - std::log(FloatType(2));
        d_ln_p = -a*a*modulus_fc/eb + (a*u/eb)*std::tanh(z);
      }
      else {
        FloatType const arg = 2*a*u*modulus_fc/eb;
        ln_p = std::log(2*u/eb) - sq/eb + scitbx::math::bessel::ln_of_i0(arg);
        d_ln_p = -2*a*a*modulus_fc/eb
                 + (2*a*u/eb)*scitbx::math::bessel::i1_over_i0(arg);
      }
    }
  }

  /// dT/d|Fc| for the intensity target, and whether it could be computed.
  /** The gradient is exact for the convolved target; the curvature used by the
      caller is the amplitude one, which is a positive approximation. That is
      ordinary Gauss-Newton practice: an approximate Hessian changes how fast a
      refinement converges, never what it converges to, because the gradient is
      what vanishes at the answer.

      T = -ln of the integral of P(I_t) N(I_o; I_t, sigma^2) over I_t >= 0.
   */
  template <typename FloatType>
  bool mli_target_and_d_modulus_fc(
    FloatType fo_sq,
    FloatType sigma_fo_sq,
    FloatType observable,
    FloatType alpha,
    FloatType beta,
    FloatType scale_factor,
    FloatType epsilon,
    bool centric,
    FloatType &target_value,
    FloatType &d_target)
  {
    target_value = 0;
    d_target = 0;
    if (scale_factor <= 0) scale_factor = 1;
    if (alpha <= 0 || beta <= 1e-3 || observable <= 0 || sigma_fo_sq <= 0) {
      return false;
    }
    if (epsilon <= 0) {
      return false;
    }
    FloatType const a = alpha*scale_factor;
    FloatType const b = beta*scale_factor*scale_factor;
    FloatType const eb = epsilon*b;
    FloatType const modulus_fc = std::sqrt(observable);
    FloatType const s = sigma_fo_sq;
    FloatType const two_s_sq = 2*s*s;

    // the posterior, both factors taken as Gaussians in I_t
    FloatType const mean_p = a*a*observable + eb;
    FloatType const var_p = (centric ? 2 : 1)*eb*(eb + 2*a*a*observable);
    FloatType const precision = 1/(s*s) + 1/var_p;
    FloatType const mu = (fo_sq/(s*s) + mean_p/var_p)/precision;
    FloatType const tau = 1/std::sqrt(precision);

    /* Accumulated with the largest log-term factored out: the densities
       underflow to zero for a reflection the model fits badly, and the ratio
       below is then 0/0 rather than the finite number it should be.
     */
    FloatType ln_q[96], dln_p[96];
    bool usable[96];
    FloatType ln_max = -std::numeric_limits<FloatType>::max();
    FloatType ln_prefactor;
    int n, n_w;             // nodes used, and the length of the rule repeated
    double const *w;
    /* A measurement many combined standard deviations from what the model
       expects puts the posterior in the far tails of both factors, where
       neither Gaussian picture holds; that case takes the u branch too, over
       the union of the supports and in four panels.
     */
    bool const outlier =
      std::abs(fo_sq - mean_p) >= 4*std::sqrt(s*s + var_p)
      && s*s >= 0.09*var_p;  // a narrow measurement is still the posterior
    if (mu - 6*tau > 0 && !outlier) {
      n = n_w = mli_quadrature::n_hermite();
      double const *x = mli_quadrature::hermite_nodes();
      w = mli_quadrature::hermite_weights();
      FloatType const root_two_tau = std::sqrt(FloatType(2))*tau;
      for (int i = 0; i < n; i++) {
        usable[i] = false;
        FloatType const xi = static_cast<FloatType>(x[i]);
        FloatType const it = mu + root_two_tau*xi;
        if (it <= 0) continue;
        FloatType const u = std::sqrt(it), d = it - fo_sq;
        FloatType ln_p;
        mli_quadrature::ln_p_amplitude(u, a, modulus_fc, eb, centric,
                                       ln_p, dln_p[i]);
        // P(I_t) = P_F(u)/(2u); the proposal's exp(-x^2) is divided out
        ln_q[i] = ln_p - std::log(2*u) - d*d/two_s_sq + xi*xi;
        usable[i] = true;
        if (ln_q[i] > ln_max) ln_max = ln_q[i];
      }
      // the integral is (tau/sigma) sum w exp(ln_q), w summing to one
      ln_prefactor = std::log(tau/s);
    }
    else {
      int const n_rule = mli_quadrature::n_legendre();
      double const *x = mli_quadrature::legendre_nodes();
      w = mli_quadrature::legendre_weights();
      n_w = n_rule;
      // supports in u, each factor to exp(-40) of its own peak
      FloatType const m_lo = std::sqrt(std::max(fo_sq - 9*s, FloatType(0)));
      FloatType const m_hi = std::sqrt(std::max(fo_sq + 9*s, FloatType(0)));
      FloatType const half_width =
        (centric ? FloatType(13) : FloatType(9.5))*std::sqrt(eb);
      FloatType const p_lo = std::max(a*modulus_fc - half_width, FloatType(0));
      FloatType const p_hi = a*modulus_fc + half_width;
      FloatType lo = std::max(m_lo, p_lo), hi = std::min(m_hi, p_hi);
      if (outlier || !(hi > lo)) {
        lo = std::min(m_lo, p_lo);
        hi = std::max(m_hi, p_hi);
      }
      if (!(hi > lo)) return false;
      int const panels = outlier ? 4 : 1;
      n = panels*n_rule;
      FloatType const half = (hi - lo)/(2*panels);
      for (int i = 0; i < n; i++) {
        usable[i] = false;
        FloatType const left = lo + 2*half*(i/n_rule);
        FloatType const u = left + half*(static_cast<FloatType>(x[i % n_rule]) + 1);
        if (u <= 0) continue;
        FloatType const d = u*u - fo_sq;
        FloatType ln_p;
        mli_quadrature::ln_p_amplitude(u, a, modulus_fc, eb, centric,
                                       ln_p, dln_p[i]);
        ln_q[i] = ln_p - d*d/two_s_sq;
        usable[i] = true;
        if (ln_q[i] > ln_max) ln_max = ln_q[i];
      }
      // the integral is half/(sqrt(2 pi) sigma) sum w exp(ln_q), half being
      // the half-width of one panel
      ln_prefactor = std::log(half/(std::sqrt(2*scitbx::constants::pi)*s));
    }
    if (ln_max == -std::numeric_limits<FloatType>::max()) {
      return false;
    }
    FloatType sum = 0, sum_d = 0;
    for (int i = 0; i < n; i++) {
      if (!usable[i]) continue;
      FloatType const p =
        static_cast<FloatType>(w[i % n_w])*std::exp(ln_q[i] - ln_max);
      sum += p;
      sum_d += p*dln_p[i];
    }
    if (sum <= 0) {
      return false;
    }
    /* T = -ln of the integral, so dT/d|Fc| = -(sum of P dlnP)/(sum of P).
       The factored-out maximum and the prefactor come back into the value;
       both cancel out of the ratio the derivative is.
     */
    target_value = -(ln_max + std::log(sum) + ln_prefactor);
    d_target = -sum_d/sum;
    return true;
  }

  /// The gradient alone, for callers with no use for the target value.
  template <typename FloatType>
  bool mli_d_target_d_modulus_fc(
    FloatType fo_sq,
    FloatType sigma_fo_sq,
    FloatType observable,
    FloatType alpha,
    FloatType beta,
    FloatType scale_factor,
    FloatType epsilon,
    bool centric,
    FloatType &d_target)
  {
    FloatType value_not_wanted;
    return mli_target_and_d_modulus_fc(
      fo_sq, sigma_fo_sq, observable, alpha, beta, scale_factor, epsilon,
      centric, value_not_wanted, d_target);
  }

  /// The effective observation and weight for the intensity target, in F^2.
  /** Same two-step shape as the amplitude case: the exact gradient decides the
      observation, the amplitude curvature supplies a positive weight.
   */
  template <typename FloatType>
  bool mli_effective_observation(
    FloatType fo_sq,
    FloatType sigma_fo_sq,
    FloatType observable,
    FloatType alpha,
    FloatType beta,
    FloatType scale_factor,
    FloatType epsilon,
    bool centric,
    FloatType &yo_eff,
    FloatType &w_eff)
  {
    yo_eff = observable;
    w_eff = 0;
    FloatType d_target;
    if (!mli_d_target_d_modulus_fc(fo_sq, sigma_fo_sq, observable, alpha, beta,
                                   scale_factor, epsilon, centric, d_target))
    {
      return false;
    }
    FloatType const k = (scale_factor <= 0) ? FloatType(1) : scale_factor;
    FloatType const a = alpha*k;
    FloatType const b = beta*k*k;
    FloatType const eb = epsilon*b;
    FloatType const w_amplitude = centric ? a*a/eb : 2*a*a/eb;
    FloatType const modulus_fc = std::sqrt(observable);
    // |Fc| - (dT/d|Fc|)/w, then into F^2 space exactly as above
    FloatType const yo_amplitude = modulus_fc - d_target/w_amplitude;
    // and the same near-zero guard as the amplitude case, for the same reason
    if (!ml_intensity_is_usable(fo_sq, sigma_fo_sq, observable)) {
      w_eff = 0;
      yo_eff = observable;
      return false;
    }
    w_eff = w_amplitude/(4*observable);
    yo_eff = 2*modulus_fc*yo_amplitude - observable;
    return true;
  }

  /// -ln L for one reflection, under whichever target is in force.
  /** The quantity the refinement is minimising, per reflection. Reporting goes
      through here rather than through a second implementation so that a
      log-likelihood shown to the user cannot drift away from the target being
      optimised.

      \return false for a reflection the target cannot use, whose contribution
              is then zero rather than undefined.
   */
  template <typename FloatType>
  bool ml_target_value(
    FloatType fo_sq,
    FloatType sigma_fo_sq,
    FloatType observable,
    FloatType alpha,
    FloatType beta,
    FloatType scale_factor,
    FloatType epsilon,
    bool centric,
    bool intensity,
    FloatType &value)
  {
    value = 0;
    if (intensity) {
      FloatType d_not_wanted;
      return mli_target_and_d_modulus_fc(
        fo_sq, sigma_fo_sq, observable, alpha, beta, scale_factor, epsilon,
        centric, value, d_not_wanted);
    }
    /* mlf::target_one_h returns 0 for a reflection it refuses, which is
       indistinguishable from a genuine zero, so the guards are repeated here
       to say which happened. They are the same ones, deliberately.
     */
    if (alpha <= 0 || beta <= 1e-3 || fo_sq <= 0 || observable <= 0
        || epsilon <= 0) {
      return false;
    }
    value = cctbx::xray::targets::mlf::target_one_h(
      std::sqrt(fo_sq), std::sqrt(observable), alpha, beta,
      scale_factor > 0 ? scale_factor : FloatType(1), epsilon, centric);
    return true;
  }

  /// Summed -ln L over the working and the free reflections, and their counts.
  /** The free sum is the log free likelihood, LL_free, which Tickle (2007)
      argues should replace R_free as the statistic against which weighting
      parameters are chosen: it is the same functional the refinement optimises,
      evaluated on data the refinement never saw.

      Counts accompany the sums because the two sets differ in size by an order
      of magnitude and only the mean per reflection is comparable between them.

      Reference: Tickle (2007) Acta Cryst. D63, 1274-1281.
   */
  template <typename FloatType>
  struct ml_log_likelihood_result {
    FloatType work, free;
    std::size_t n_work, n_free, n_skipped;
    ml_log_likelihood_result()
      : work(0), free(0), n_work(0), n_free(0), n_skipped(0)
    {}
    /// Mean -ln L per working reflection, or zero if there are none.
    FloatType work_per_reflection() const {
      return n_work ? work/static_cast<FloatType>(n_work) : FloatType(0);
    }
    /// Mean -ln L per free reflection, or zero if there are none.
    FloatType free_per_reflection() const {
      return n_free ? free/static_cast<FloatType>(n_free) : FloatType(0);
    }
  };

  /** \param free_only evaluate the free reflections alone, leaving the working
        sums zero. The free set is a tenth of the data, and for the intensity
        target each reflection costs a twenty-node quadrature, so a caller that
        only wants LL_free should say so: measured across eight structures,
        computing both sums cost 12-21% of an MLI cycle and the free half of it
        about a tenth of that. Ignored when no free set was given, since then
        there is nothing to restrict to.
   */
  template <typename FloatType>
  ml_log_likelihood_result<FloatType> ml_log_likelihood(
    scitbx::af::const_ref<FloatType> const &fo_sq,
    scitbx::af::const_ref<FloatType> const &sigma_fo_sq,
    scitbx::af::const_ref<FloatType> const &observables,
    ml_data<FloatType> const &ml,
    bool free_only = false)
  {
    ml_log_likelihood_result<FloatType> result;
    SMTBX_ASSERT(ml.active());
    SMTBX_ASSERT(ml.is_consistent_with(fo_sq.size()));
    SMTBX_ASSERT(sigma_fo_sq.size() == fo_sq.size());
    SMTBX_ASSERT(observables.size() == fo_sq.size());
    bool const skip_work = free_only && ml.free_flags.size() != 0;
    for (std::size_t i = 0; i < fo_sq.size(); i++) {
      if (skip_work && !ml.is_free(i)) {
        continue;
      }
      FloatType value;
      if (!ml_target_value(fo_sq[i], sigma_fo_sq[i], observables[i],
                           ml.alpha[i], ml.beta[i], ml.scale_factor(),
                           ml.epsilon[i], ml.centric[i], ml.is_intensity(),
                           value))
      {
        result.n_skipped++;
        continue;
      }
      if (ml.is_free(i)) {
        result.free += value;
        result.n_free++;
      }
      else {
        result.work += value;
        result.n_work++;
      }
    }
    return result;
  }

}}}

#endif // GUARD
