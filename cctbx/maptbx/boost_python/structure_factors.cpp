#include <cctbx/boost_python/flex_fwd.h>

#include <cctbx/maptbx/structure_factors.h>
#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/shared_ptr.hpp>

namespace cctbx { namespace maptbx { namespace structure_factors {

namespace {

  /* releases the GIL while a transform runs: these constructors do most of a
  charge-flipping cycle and touch no Python API. They read const_refs into
  arrays the caller holds for the call, and each smtbx.ab_initio.multi_trial
  thread works on its own, so nothing is shared
  */
  struct gil_release
  {
    gil_release() : state_(PyEval_SaveThread()) {}
    ~gil_release() { PyEval_RestoreThread(state_); }
  private:
    PyThreadState* state_;
    gil_release(gil_release const&);
    gil_release& operator=(gil_release const&);
  };

  struct to_map_wrappers
  {
    typedef to_map<> w_t;

    // the init<> below without the GIL; miller's fft_map uses this overload
    static boost::shared_ptr<w_t>
    make(
      sgtbx::space_group const& space_group,
      bool anomalous_flag,
      af::const_ref<miller::index<> > const& miller_indices,
      af::const_ref<std::complex<double> > const& structure_factors,
      af::int3 const& n_real,
      af::flex_grid<> const& map_grid,
      bool conjugate_flag,
      bool treat_restricted)
    {
      gil_release nogil;
      return boost::shared_ptr<w_t>(new w_t(
        space_group, anomalous_flag, miller_indices, structure_factors,
        n_real, map_grid, conjugate_flag, treat_restricted));
    }

    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("structure_factors_to_map", no_init)
        .def(init<
          sgtbx::space_group const&,
          bool,
          af::const_ref<miller::index<> > const&,
          af::const_ref<std::complex<double> > const&,
          af::int3 const&,
          af::flex_grid<> const&,
          bool,
          optional<bool> >((
            arg("space_group"),
            arg("anomalous_flag"),
            arg("miller_indices"),
            arg("structure_factors"),
            arg("n_real"),
            arg("map_grid"),
            arg("conjugate_flag"),
            arg("treat_restricted")=true)))
        // after the init<> overloads: Boost.Python tries the most recently
        // registered first, so placed before them init<> wins every call
        .def("__init__", make_constructor(
          make, default_call_policies(),
          (arg("space_group"),
           arg("anomalous_flag"),
           arg("miller_indices"),
           arg("structure_factors"),
           arg("n_real"),
           arg("map_grid"),
           arg("conjugate_flag"),
           arg("treat_restricted")=true)))
        .def("complex_map", &w_t::complex_map)
      ;
    }
  };

  struct from_map_wrappers
  {
    typedef from_map<> w_t;

    // the overload structure_factors_from_map reaches by default (use_sg
    // False); the space-group form stays on init<>, the hot path skips it
    static boost::shared_ptr<w_t>
    make(
      bool anomalous_flag,
      af::const_ref<miller::index<> > const& miller_indices,
      af::const_ref<std::complex<double>,
        af::c_grid_padded<3> > const& complex_map,
      bool conjugate_flag,
      bool allow_miller_indices_outside_map)
    {
      gil_release nogil;
      return boost::shared_ptr<w_t>(new w_t(
        anomalous_flag, miller_indices, complex_map, conjugate_flag,
        allow_miller_indices_outside_map));
    }

    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("structure_factors_from_map", no_init)
        .def(init<
          uctbx::unit_cell const&,
          sgtbx::space_group_type const&,
          bool,
          double,
          af::const_ref<std::complex<double>,
            af::c_grid_padded<3> > const&,
          bool,
          optional<bool> >((
            arg("unit_cell"),
            arg("space_group_type"),
            arg("anomalous_flag"),
            arg("d_min"),
            arg("complex_map"),
            arg("conjugate_flag"),
            arg("discard_indices_affected_by_aliasing")=false)))
        .def(init<
          bool,
          af::const_ref<miller::index<> > const&,
          af::const_ref<std::complex<double>,
            af::c_grid_padded<3> > const&,
          bool,
          optional<bool> >((
            arg("anomalous_flag"),
            arg("miller_indices"),
            arg("complex_map"),
            arg("conjugate_flag"),
            arg("allow_miller_indices_outside_map")=false)))
        .def(init<
          sgtbx::space_group const&,
          bool,
          af::const_ref<miller::index<> > const&,
          af::const_ref<std::complex<double>,
          af::c_grid_padded<3> > const&,
          bool>((
            arg("space_group"),
            arg("anomalous_flag"),
            arg("miller_indices"),
            arg("complex_map"),
            arg("conjugate_flag"))))
        // after the init<> overloads: Boost.Python tries the most recently
        // registered first, so placed before them init<> wins every call
        .def("__init__", make_constructor(
          make, default_call_policies(),
          (arg("anomalous_flag"),
           arg("miller_indices"),
           arg("complex_map"),
           arg("conjugate_flag"),
           arg("allow_miller_indices_outside_map")=false)))
        .def("miller_indices", &w_t::miller_indices)
        .def("data", &w_t::data)
        .def("n_indices_affected_by_aliasing",
             &w_t::n_indices_affected_by_aliasing)
        .def("outside_map", &w_t::outside_map)
      ;
    }
  };

}} // namespace structure_factors::<anoymous>

namespace boost_python {

  void wrap_structure_factors()
  {
    structure_factors::to_map_wrappers::wrap();
    structure_factors::from_map_wrappers::wrap();
  }

}}} // namespace cctbx::maptbx::boost_python
