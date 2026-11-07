#include <cctbx/boost_python/flex_fwd.h>

#include <cctbx/maptbx/statistics.h>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/shared_ptr.hpp>

namespace cctbx { namespace maptbx { namespace boost_python {

namespace {

  // releases the GIL for a whole-map pass: charge_flipping.solving_iterator
  // asks for the skewness every iteration, outside the flipping cycle. The
  // constructors read a const_ref the caller owns and touch no Python API
  struct gil_release
  {
    gil_release() : state_(PyEval_SaveThread()) {}
    ~gil_release() { PyEval_RestoreThread(state_); }
  private:
    PyThreadState* state_;
    gil_release(gil_release const&);
    gil_release& operator=(gil_release const&);
  };

  struct statistics_wrappers
  {
    typedef statistics<> w_t;

    // the double overload, the one the map path uses
    static boost::shared_ptr<w_t>
    make(af::const_ref<double, af::flex_grid<> > const& map)
    {
      gil_release nogil;
      return boost::shared_ptr<w_t>(new w_t(map));
    }

    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t>("statistics", no_init)
        .def(init<af::const_ref<float, af::flex_grid<> > const&>())
        .def(init<af::const_ref<double, af::flex_grid<> > const&>())
        .def("min", &w_t::min)
        .def("max", &w_t::max)
        .def("mean", &w_t::mean)
        .def("mean_sq", &w_t::mean_sq)
        .def("sigma", &w_t::sigma)
        // after the init<> overloads: Boost.Python tries the most recently
        // registered first, so placed before them init<> wins every call
        .def("__init__", make_constructor(make, default_call_policies(),
                                          (arg("map"))))
      ;
    }
  };

  struct more_statistics_wrappers
  {
    typedef more_statistics<> w_t;

    static boost::shared_ptr<w_t>
    make(af::const_ref<double, af::flex_grid<> > const& map)
    {
      gil_release nogil;
      return boost::shared_ptr<w_t>(new w_t(map));
    }

    static void
    wrap()
    {
      using namespace boost::python;
      class_<w_t, bases<statistics<> > >("more_statistics", no_init)
        .def(init<af::const_ref<float, af::flex_grid<> > const&>())
        .def(init<af::const_ref<double, af::flex_grid<> > const&>())
        .def("skewness", &w_t::skewness)
        .def("kurtosis", &w_t::kurtosis)
        .def("__init__", make_constructor(make, default_call_policies(),
                                          (arg("map"))))
      ;
    }
  };

  struct even_more_statistics_wrappers
  {
    static void
    wrap()
    {
      typedef mem_iteration<> w_t;
      using namespace boost::python;
      class_<w_t>("mem_iteration", no_init)
        .def(init<af::ref<double, af::c_grid<3> > const&,
                  af::ref<double, af::c_grid<3> > const&,
                  af::ref<double, af::c_grid<3> >,
                  double,
                  af::tiny<int, 3> const&,
                  double,
                  double,
                  bool >())
        .def("tp", &w_t::tp)
        .def("scale", &w_t::scale)
        .def("z", &w_t::z)
        .def("hw", &w_t::hw)
        .def("hn", &w_t::hn)
      ;
    }
  };

} // namespace <anoymous>

  void wrap_statistics()
  {
    statistics_wrappers::wrap();
    more_statistics_wrappers::wrap();
    even_more_statistics_wrappers::wrap();
  }

}}} // namespace cctbx::maptbx::boost_python
