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
  }

  void symmetry_equivalent_site_parameter
  ::linearise(uctbx::unit_cell const &unit_cell,
              sparse_matrix_type *jacobian_transpose)
  {
    // any site parameter: independent, special position, or a proxy into a
    // rigid group (the latter is not an asu_site_parameter, so original() is 0)
    site_parameter *site = dynamic_cast<site_parameter *>(argument(0));

    value = op*site->value;

    if (!jacobian_transpose) {
      return;
    }
    // x' = R x + t, so column j of this site is sum_k R(j,k) times column k
    // of the original, whatever that column depends on
    sparse_matrix_type &jt = *jacobian_transpose;
    scitbx::mat3<double> r = op.r().as_double();
    for (int j=0; j<3; j++) {
      jt.col(index() + j) = r(j, 0)*jt.col(site->index());
      jt.col(index() + j) += r(j, 1)*jt.col(site->index() + 1);
      jt.col(index() + j) += r(j, 2)*jt.col(site->index() + 2);
    }
  }

}}}
