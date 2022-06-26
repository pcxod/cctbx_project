#include <smtbx/refinement/constraints/rigid.h>
#include <scitbx/math/euler_angles.h>

namespace smtbx { namespace refinement { namespace constraints {

  index_range
  rigid_group_base
  ::component_indices_for(scatterer_type const *scatterer) const {
    for (int i=0; i < scatterers_.size(); i++)  {
      if( scatterers_[i] == scatterer )
        return index_range(index() + 3*i, 3);
    }
    return index_range();
  }

  void
  rigid_group_base
  ::write_component_annotations_for(scatterer_type const *scatterer,
                                    std::ostream &output) const
  {
    for (int i=0; i < scatterers_.size(); i++)  {
      if( scatterers_[i] == scatterer )  {
        output << scatterers_[i]->label << ".x,";
        output << scatterers_[i]->label << ".y,";
        output << scatterers_[i]->label << ".z,";
        return;
      }
    }
  }


  // pivoted rotatable...
  void
  pivoted_rotatable_group
  ::linearise(uctbx::unit_cell const &unit_cell,
              sparse_matrix_type *jacobian_transpose)
  {
    site_parameter
      *pivot = dynamic_cast<site_parameter *>(this->argument(0)),
      *pivot_neighbour = dynamic_cast<site_parameter *>(this->argument(1));
    scalar_parameter
      *azimuth = dynamic_cast<scalar_parameter *>(this->argument(2)),
      *size = dynamic_cast<scalar_parameter *>(this->argument(3));

    const cart_t
      x_p = unit_cell.orthogonalize(pivot->value),
      x_pn = unit_cell.orthogonalize(pivot_neighbour->value),
      rv = (x_p - x_pn).normalize();
    const double
      angle = azimuth->value,
      size_value = size->value,
      ca = cos(angle),
      sa = sin(angle),
      t = 1.0-ca;
    // rotation matrix
    const scitbx::mat3<double> rm(
      t*rv[0]*rv[0] + ca,       t*rv[0]*rv[1] + sa*rv[2], t*rv[0]*rv[2] - sa*rv[1],
      t*rv[0]*rv[1] - sa*rv[2], t*rv[1]*rv[1] + ca,       t*rv[1]*rv[2] + sa*rv[0],
      t*rv[0]*rv[2] + sa*rv[1], t*rv[2]*rv[1] - sa*rv[0], t*rv[2]*rv[2] + ca
    );
    // derivative of the rotation matrix by angle
    const scitbx::mat3<double> rmd(
      sa*rv[0]*rv[0] - sa,       sa*rv[0]*rv[1] + ca*rv[2], sa*rv[0]*rv[2] - ca*rv[1],
      sa*rv[0]*rv[1] - ca*rv[2], sa*rv[1]*rv[1] - sa,       sa*rv[1]*rv[2] + ca*rv[0],
      sa*rv[0]*rv[2] + ca*rv[1], sa*rv[1]*rv[2] - ca*rv[0], sa*rv[2]*rv[2] - sa
    );
    if (!crd_initialised) {
      const cart_t rot_center = unit_cell.orthogonalize(pivot->value);
      for (int i=0; i < scatterers_.size(); i++)
        co_s[i] = unit_cell.orthogonalize(scatterers_[i]->site) -
                  rot_center;
      crd_initialised = true;
    }
    // Loop over the scatterers
    for (int i=0; i < scatterers_.size(); i++) {
      // update site of i-th scatterers
      const cart_t p = co_s[i]*rm;
     fx_s[i] = unit_cell.fractionalize(size_value*p + x_p);

      // Derivatives
      if (!jacobian_transpose) continue;
      sparse_matrix_type &jt = *jacobian_transpose;
      std::size_t const j_s = this->index() + 3*i;

      // Riding
      for (int j=0; j<3; j++)
        jt.col(j_s + j) = jt.col(pivot->index() + j);

      // Rotating
      if (azimuth->is_variable()) {
        frac_t grad_f = unit_cell.fractionalize(size_value*co_s[i]*rmd);
        for (int j=0; j<3; j++)
          jt(azimuth->index(), j_s + j) = grad_f[j];
      }
      // stretching
      if (size->is_variable()) {
        frac_t grad_f = unit_cell.fractionalize(p);
        for (int j=0; j<3; j++)
          jt(size->index(), j_s + j) = grad_f[j];
      }
    }
  }

  // spherical rotatable expandable...
  void
  rotatable_expandable_group
  ::linearise(uctbx::unit_cell const &unit_cell,
              sparse_matrix_type *jacobian_transpose)
  {
    site_parameter
      *pivot = dynamic_cast<site_parameter *>(this->argument(0));
    scalar_parameter
      *size = dynamic_cast<scalar_parameter *>(argument(1));
    const scalar_parameter * angles[] = {
      dynamic_cast<scalar_parameter *>(argument(2)),
      dynamic_cast<scalar_parameter *>(argument(3)),
      dynamic_cast<scalar_parameter *>(argument(4))
    };
    scitbx::mat3<double> rmd[3];
    scitbx::mat3<double> rm = scitbx::math::euler_angles::xyz_matrix_rad(
      angles[0]->value, angles[1]->value, angles[2]->value, &rmd[0]
    );
    if (!crd_initialised) {
      const cart_t original_pivot_crd = unit_cell.orthogonalize(pivot->value);
      cart_t rotation_center = original_pivot_crd;
      for (int i=0; i < scatterers_.size(); i++)  {
        co_s[i] = unit_cell.orthogonalize(scatterers_[i]->site);
        rotation_center += co_s[i];
      }
      rotation_center = rotation_center/(scatterers_.size()+1);
      for (int i=0; i < scatterers_.size(); i++)
        co_s[i] = co_s[i] - rotation_center;
      this->shift_to_pivot = original_pivot_crd-rotation_center;
      crd_initialised = true;
    }
    const cart_t new_pivot_crd = unit_cell.orthogonalize(pivot->value);
    const double size_value = size->value;
    const cart_t shift = new_pivot_crd - size_value*shift_to_pivot*rm;
    // expansion/contraction happens from/to the center
    for (int i=0; i < scatterers_.size(); i++) {
      // update site of i-th atoms
      fx_s[i] = unit_cell.fractionalize(size_value*co_s[i]*rm + shift);

      // Derivatives
      if (!jacobian_transpose) continue;
      sparse_matrix_type &jt = *jacobian_transpose;
      std::size_t const j_s = this->index() + 3*i;

      // Riding
      for (int j=0; j<3; j++)
        jt.col(j_s + j) = jt.col(pivot->index() + j);

      const cart_t grad_vec_ = co_s[i]-shift_to_pivot;
      // Rotating
      for (int j=0; j<3; j++ )  {
        if (!angles[j]->is_variable())  continue;
        frac_t grad_f = unit_cell.fractionalize(size_value*grad_vec_*rmd[j]);
        for (int k=0; k<3; k++)
          jt(angles[j]->index(), j_s + k) = grad_f[k];
      }

      // expansion/contraction
      if (size->is_variable())  {
        frac_t grad_f = unit_cell.fractionalize(grad_vec_*rm);
        for (int j=0; j<3; j++)
          jt(size->index(), j_s + j) = grad_f[j];
      }
    }
  }

  // riding expandable...
  void
  riding_expandable_group
  ::linearise(uctbx::unit_cell const &unit_cell,
              sparse_matrix_type *jacobian_transpose)
  {
    site_parameter
      *pivot = dynamic_cast<site_parameter *>(this->argument(0));
    scalar_parameter
      *size = dynamic_cast<scalar_parameter *>(argument(1));
    const double size_value = size->value;
    const cart_t center = unit_cell.orthogonalize(pivot->value);
    if (!crd_initialised) {
      const cart_t original_pivot_crd = unit_cell.orthogonalize(pivot->value);
      for (int i=0; i < scatterers_.size(); i++)  {
        co_s[i] = unit_cell.orthogonalize(scatterers_[i]->site) -
          original_pivot_crd;
      }
      crd_initialised = true;
    }
    for (int i=0; i < scatterers_.size(); i++) {
      // update site of i-th atoms
      fx_s[i] = unit_cell.fractionalize(center + co_s[i]*size_value);

      // Derivatives
      if (!jacobian_transpose) continue;
      sparse_matrix_type &jt = *jacobian_transpose;
      std::size_t const j_s = this->index() + 3*i;

      // Riding
      for (int j=0; j<3; j++)
        jt.col(j_s + j) = jt.col(pivot->index() + j);

      // expansion/contraction
      if (size->is_variable())  {
        frac_t grad_f = unit_cell.fractionalize(co_s[i]);
        for (int j=0; j<3; j++)
          jt(size->index(), j_s + j) = grad_f[j];
      }
    }
  }
}}}
