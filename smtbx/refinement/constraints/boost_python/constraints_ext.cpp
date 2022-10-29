#include <boost/python/module.hpp>

#include <smtbx/refinement/constraints/reparametrisation.h>

namespace smtbx { namespace refinement { namespace constraints {

namespace boost_python {
  void wrap_reparametrisation();
  void wrap_geometrical_hydrogens();
  void wrap_special_position();
  void wrap_scatterer_parameters();
  void wrap_independent_scalar_parameters();
  void wrap_affine_scalar_parameter_wrapper();
  void wrap_symmetry_equivalent_site_parameter();
  void wrap_u_eq_dependent_u_iso();
  void wrap_u_iso_dependent_u_iso();
  void wrap_shared();
  void wrap_occupancy();
  void wrap_rigid();
  void wrap_direction();
  void wrap_same_group();
  void wrap_scaled_adp();

  namespace {
    void init_module() {
      wrap_reparametrisation();
      wrap_geometrical_hydrogens();
      wrap_special_position();
      wrap_scatterer_parameters();
      wrap_independent_scalar_parameters();
      wrap_affine_scalar_parameter_wrapper();
      wrap_symmetry_equivalent_site_parameter();
      wrap_u_eq_dependent_u_iso();
      wrap_u_iso_dependent_u_iso();
      wrap_shared();
      wrap_occupancy();
      wrap_rigid();
      wrap_direction();
      wrap_same_group();
      wrap_scaled_adp();
    }
  }

}}}} // end namespace smtbx::refinement::constraints::boost_python

BOOST_PYTHON_MODULE(smtbx_refinement_constraints_ext)
{
  smtbx::refinement::constraints::boost_python::init_module();
}
