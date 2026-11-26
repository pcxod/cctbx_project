#include <smtbx/refinement/constraints/symmetry_equivalent.h>

namespace smtbx { namespace refinement { namespace constraints {

  // Symmetry equivalent site

  symmetry_equivalent_site_parameter
    ::symmetry_equivalent_site_parameter(site_parameter *site,
    sgtbx::rt_mx const &op)
    : parameter(1),
    op(op)
  {
    set_arguments(site);
    /* old: a local Jacobian fixed at construction, replaced by chaining the
    original's columns in linearise because it handled only independent and
    special-position sites, not a site inside a rigid group (needs
    special_position.h, versa_matrix.h and the members in the header)
    special_position(dynamic_cast<special_position_site_parameter *>(site)),
    ...
    if (special_position == 0) {
      scitbx::mat3<double> r_t = op.r().as_double().transpose();
      local_jt = af::ref<double, af::mat_grid>(r_t.begin(), af::mat_grid(3, 3));
    }
    else {
      af::const_ref<double, af::mat_grid> gm =
        special_position->get_site_constraints().gradient_sum_matrix();
      af::const_ref<double, af::mat_grid> rm =
        af::const_ref<double, af::mat_grid>(op.r().as_double().begin(), 3, 3);
      af::shared<double> res(3 * gm.n_rows());
      local_jt = af::ref<double, af::mat_grid>(res.begin(), gm.n_rows(), 3);
      af::multiply(gm, rm, local_jt);
    }
    */
  }

  void symmetry_equivalent_site_parameter
  ::linearise(uctbx::unit_cell const &unit_cell,
              sparse_matrix_type *jacobian_transpose)
  {
    // any site parameter: independent, special position, or a proxy into a
    // rigid group (the latter is not an asu_site_parameter, so original() is 0)
    // old: site_parameter *site = original();
    site_parameter *site = dynamic_cast<site_parameter *>(argument(0));

    value = op*site->value;

    if (!jacobian_transpose) {
      return;
    }
    // x' = R x + t, so column j of this site is sum_k R(j,k) times column k
    // of the original, whatever that column depends on
    sparse_matrix_type &jt = *jacobian_transpose;
    /* old: copied the constructor's local Jacobian, see above
    if (special_position == 0) {
      jt.assign_block(local_jt, site->index(), index());
    }
    else {
      jt.assign_block(local_jt,
        special_position->independent_params().index(), index());
    }
    */
    scitbx::mat3<double> r = op.r().as_double();
    for (int j=0; j<3; j++) {
      jt.col(index() + j) = r(j, 0)*jt.col(site->index());
      jt.col(index() + j) += r(j, 1)*jt.col(site->index() + 1);
      jt.col(index() + j) += r(j, 2)*jt.col(site->index() + 2);
    }
  }

}}}
