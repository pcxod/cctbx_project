#include <smtbx/boost_python/flex_fwd.h>

#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>

#include <scitbx/array_family/accessors/c_grid_padded.h>
#include <scitbx/array_family/accessors/flex_grid.h>

#include <smtbx/ab_initio/density_modification.h>
#include <cctbx/miller/phase_transfer.h>
#include <smtbx/error.h>

#include <cmath>
#include <complex>

namespace smtbx { namespace ab_initio { namespace boost_python {

  namespace af = scitbx::af;

  namespace {

    // Release the GIL for the duration of a call that touches no Python API.
    struct gil_release
    {
      gil_release() : state_(PyEval_SaveThread()) {}
      ~gil_release() { PyEval_RestoreThread(state_); }
    private:
      PyThreadState* state_;
      gil_release(gil_release const&);
      gil_release& operator=(gil_release const&);
    };

  }

  /* The Oszlanyi-Suto phase transfer, in one call.

     This was six flex operations driven from Python -- flex.arg, two slices,
     flex.abs, extend, then phase_transfer -- and Python holds the GIL between
     them, so the step could not run on more than one thread. Measured 30
     August 2026 it is 0.150 ms of a 0.78 ms charge-flipping cycle and scaled
     0.97x on four threads while the two FFT steps reached 2.56x and 3.32x,
     making it the largest remaining serial part of the cycle.

     The arithmetic is copied from `charge_flipping._array_extension`, not
     re-derived: the weak head takes its modulus from the SOURCE and its phase
     shifted by delta_varphi, the strong tail takes its modulus from f_obs and
     the source phase unchanged. Centric restrictions are not reimplemented
     here -- cctbx::miller::phase_transfer applies them, which is why it is
     called rather than inlined.
   */
  af::shared<std::complex<double> >
  oszlanyi_suto_phase_transfer(
    cctbx::sgtbx::space_group const& space_group,
    af::const_ref<cctbx::miller::index<> > const& miller_indices,
    af::const_ref<double> const& f_obs_data,
    af::const_ref<std::complex<double> > const& source_data,
    std::size_t cut,
    double delta_varphi)
  {
    SMTBX_ASSERT(f_obs_data.size() == source_data.size());
    SMTBX_ASSERT(f_obs_data.size() == miller_indices.size());
    SMTBX_ASSERT(cut <= f_obs_data.size());
    gil_release nogil;
    std::size_t n = f_obs_data.size();
    af::shared<double> moduli(n, af::init_functor_null<double>());
    af::shared<double> phases(n, af::init_functor_null<double>());
    for (std::size_t i = 0; i < n; i++) {
      std::complex<double> const& g = source_data[i];
      double phase = std::atan2(g.imag(), g.real());
      if (i < cut) {
        moduli[i] = std::abs(g);
        phases[i] = phase + delta_varphi;
      }
      else {
        moduli[i] = f_obs_data[i];
        phases[i] = phase;
      }
    }
    return cctbx::miller::phase_transfer(
      space_group, miller_indices, moduli.const_ref(), phases.const_ref(),
      false);
  }

  template<class FloatType, class AccessorType>
  struct density_modification_wrapper
  {
    typedef void (*f_t)(af::ref<FloatType, AccessorType> const&, FloatType);
    static void wrap() {
      using namespace boost::python;
      using namespace density_modification;
      def("flip_charges_in_place",
          static_cast<f_t>(flip_charges_in_place));
      def("low_density_elimination_in_place_tanaka_et_al_2001",
          static_cast<f_t>(low_density_elimination_in_place_tanaka_et_al_2001));
      def("oszlanyi_suto_phase_transfer", oszlanyi_suto_phase_transfer,
          (arg("space_group"), arg("miller_indices"), arg("f_obs_data"),
           arg("source_data"), arg("cut"), arg("delta_varphi")));
    }
  };

  void init_module() {
    density_modification_wrapper<double, af::c_grid_padded<3> >::wrap();
    density_modification_wrapper<double, af::flex_grid<> >::wrap();
  }

}}} // namespace smtbx::ab_initio::boost_python


BOOST_PYTHON_MODULE(smtbx_ab_initio_ext)
{
  smtbx::ab_initio::boost_python::init_module();
}
