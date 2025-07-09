#include <boost/python/class.hpp>
#include <boost/python/implicit.hpp>

#include <scitbx/boost_python/container_conversions.h>

#include <smtbx/refinement/constraints/geometrical_hydrogens.h>

#include <sstream>

namespace smtbx { namespace refinement { namespace constraints {
namespace boost_python {

  template <int n_hydrogens, bool staggered>
  struct terminal_tetrahedral_xhn_sites_wrapper
  {
    typedef terminal_tetrahedral_xhn_sites<n_hydrogens, staggered> wt;

    static void wrap() {
      using namespace boost::python;
      std::ostringstream sname;
      if (staggered) sname << "staggered_";
      sname << "terminal_tetrahedral_xh";
      if (n_hydrogens > 1) sname << n_hydrogens;
      sname << "_site";
      if (n_hydrogens > 1) sname << "s";
      std::string name = sname.str();
      if (staggered) {
        class_<wt,
               bases<asu_parameter>,
               std::auto_ptr<wt> >(name.c_str(), no_init)
          .def(init<site_parameter *,
                    site_parameter *,
                    site_parameter *,
                    independent_scalar_parameter *,
                    af::tiny<typename wt::scatterer_type *,
                             n_hydrogens> const &>
               ((arg("pivot"), arg("pivot_neighbour"), arg("stagger_on"),
                 arg("length"),
                 arg("hydrogen"))))
          ;
      }
      else {
        class_<wt,
               bases<asu_parameter>,
               std::auto_ptr<wt> >(name.c_str(), no_init)
          .def(init<site_parameter *,
                    site_parameter *,
                    independent_scalar_parameter *,
                    independent_scalar_parameter *,
                    cart_t const &,
                    af::tiny<typename wt::scatterer_type *,
                             n_hydrogens> const &>
               ((arg("pivot"), arg("pivot_neighbour"),
                 arg("azimuth"), arg("length"),
                 arg("e_zero_azimuth"),
                 arg("hydrogen"))))
          ;
      }
      implicitly_convertible<std::auto_ptr<wt>, std::auto_ptr<parameter> >();
    }
  };

  struct angle_parameter_wrapper {
    typedef angle_parameter wt;

    static void wrap() {
      using namespace boost::python;
      class_<wt,
             bases<scalar_parameter>,
             std::auto_ptr<wt> >("angle_parameter", no_init)
        .def(init<site_parameter *,
                site_parameter *,
                site_parameter *,
                double>
           ((arg("left"), arg("center"), arg("right"), arg("value"))));
      implicitly_convertible<std::auto_ptr<wt>, std::auto_ptr<parameter> >();
    }
  };

  struct secondary_xh2_sites_wrapper
  {
    typedef secondary_xh2_sites wt;

    static void wrap() {
      using namespace boost::python;
      class_<wt,
             bases<asu_parameter>,
             std::auto_ptr<wt> >("secondary_xh2_sites", no_init)
        .def(init<site_parameter *,
                site_parameter *,
                site_parameter *,
                independent_scalar_parameter *,
                scalar_parameter *,
                wt::scatterer_type *,
                wt::scatterer_type *>
           ((arg("pivot"), arg("pivot_neighbour_0"), arg("pivot_neighbour_1"),
             arg("length"), arg("h_c_h_angle"),
             arg("hydrogen_0"), arg("hydrogen_1"))));
      implicitly_convertible<std::auto_ptr<wt>, std::auto_ptr<parameter> >();
    }
  };

  struct tertiary_xh_site_wrapper
  {
    typedef tertiary_xh_site wt;

    static void wrap() {
      using namespace boost::python;
      class_<wt,
             bases<asu_parameter>,
             std::auto_ptr<wt> >("tertiary_xh_site", no_init)
        .def(init<site_parameter *,
                  site_parameter *,
                  site_parameter *,
                  site_parameter *,
                  independent_scalar_parameter *,
                  wt::scatterer_type *>
             ((arg("pivot"), arg("pivot_neighbour_0"), arg("pivot_neighbour_1"),
               arg("pivot_neighbour_2"), arg("length"),
               arg("hydrogen"))));
      implicitly_convertible<std::auto_ptr<wt>, std::auto_ptr<parameter> >();
    }
  };

  struct secondary_planar_xh_site_wrapper
  {
    typedef secondary_planar_xh_site wt;

    static void wrap() {
      using namespace boost::python;
      class_<wt,
             bases<asu_parameter>,
             std::auto_ptr<wt> >("secondary_planar_xh_site", no_init)
        .def(init<site_parameter *,
                  site_parameter *,
                  site_parameter *,
                  independent_scalar_parameter *,
                  wt::scatterer_type *>
             ((arg("pivot"), arg("pivot_neighbour_0"), arg("pivot_neighbour_1"),
               arg("length"),
               arg("hydrogen"))));
      implicitly_convertible<std::auto_ptr<wt>, std::auto_ptr<parameter> >();
    }
  };

  struct terminal_planar_xh2_sites_wrapper
  {
    typedef terminal_planar_xh2_sites wt;

    static void wrap() {
      using namespace boost::python;
      class_<wt,
             bases<asu_parameter>,
             std::auto_ptr<wt> >("terminal_planar_xh2_sites", no_init)
        .def(init<site_parameter *,
                  site_parameter *,
                  site_parameter *,
                  independent_scalar_parameter *,
                  wt::scatterer_type *, wt::scatterer_type *>
             ((arg("pivot"), arg("pivot_neighbour"),
               arg("pivot_neighbour_substituent"), arg("length"),
               arg("hydrogen_0"), arg("hydrogen_1"))));
      implicitly_convertible<std::auto_ptr<wt>, std::auto_ptr<parameter> >();
    }
  };


  struct terminal_linear_ch_site_wrapper
  {
    typedef terminal_linear_ch_site wt;

    static void wrap() {
      using namespace boost::python;
      class_<wt,
             bases<asu_parameter>,
             std::auto_ptr<wt> >("terminal_linear_ch_site", no_init)
        .def(init<site_parameter *,
                  site_parameter *,
                  independent_scalar_parameter *,
                  wt::scatterer_type *>
             ((arg("pivot"), arg("pivot_neighbour"), arg("length"),
               arg("hydrogen"))));
      implicitly_convertible<std::auto_ptr<wt>, std::auto_ptr<parameter> >();
    }
  };

  struct polyhedral_bh_site_wrapper
  {
    typedef polyhedral_bh_site wt;

    static void wrap() {
      using namespace boost::python;
      class_<wt,
             bases<asu_parameter>,
             std::auto_ptr<wt> >("polyhedral_bh_site", no_init)
        .def(init<site_parameter *,
                  af::shared<site_parameter *> const&,
                  independent_scalar_parameter *,
                  wt::scatterer_type *>
             ((arg("pivot"), arg("pivot_neighbours"), arg("length"),
               arg("hydrogen"))));
      implicitly_convertible<std::auto_ptr<wt>, std::auto_ptr<parameter> >();
    }
  };

  void wrap_geometrical_hydrogens() {
    {
      using namespace scitbx::boost_python::container_conversions;
      tuple_mapping_fixed_size<af::tiny<asu_parameter::scatterer_type *, 1> >();
      tuple_mapping_fixed_size<af::tiny<asu_parameter::scatterer_type *, 2> >();
      tuple_mapping_fixed_size<af::tiny<asu_parameter::scatterer_type *, 3> >();
      tuple_mapping_fixed_size<af::tiny<asu_parameter::scatterer_type *, 6> >();
      tuple_mapping_variable_capacity<af::shared<site_parameter *> >();
    }
    //                                    #H  #staggered?
    terminal_tetrahedral_xhn_sites_wrapper<1, false>::wrap();
    terminal_tetrahedral_xhn_sites_wrapper<2, false>::wrap();
    terminal_tetrahedral_xhn_sites_wrapper<3, false>::wrap();
    terminal_tetrahedral_xhn_sites_wrapper<6, false>::wrap();
    terminal_tetrahedral_xhn_sites_wrapper<1, true>::wrap();
    terminal_tetrahedral_xhn_sites_wrapper<2, true>::wrap();
    terminal_tetrahedral_xhn_sites_wrapper<3, true>::wrap();
    terminal_tetrahedral_xhn_sites_wrapper<6, true>::wrap();

    angle_parameter_wrapper::wrap();
    secondary_xh2_sites_wrapper::wrap();
    tertiary_xh_site_wrapper::wrap();
    secondary_planar_xh_site_wrapper::wrap();
    terminal_planar_xh2_sites_wrapper::wrap();
    terminal_linear_ch_site_wrapper::wrap();
    polyhedral_bh_site_wrapper::wrap();
  }


}}}}
