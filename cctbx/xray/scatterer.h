#ifndef CCTBX_XRAY_SCATTERER_H
#define CCTBX_XRAY_SCATTERER_H

#include <cctbx/xray/scatterer_flags.h>
#include <cctbx/xray/scatterer_id.h>
#include <cctbx/sgtbx/site_symmetry.h>
#include <cctbx/adptbx.h>
#include <cctbx/anharmonic.h>
#include <cctbx/eltbx/tiny_pse.h>

// XXX backward compatibility 2009-12-12
#define CCTBX_XRAY_SCATTERER_ANISOTROPIC_FLAG_REMOVED

namespace cctbx {
  //! X-ray scatterer and structure factor namespace.
namespace xray {

  /*! \brief Information about an atom that is needed for a
      structure factor calculation.
   */
  /*! Constructors are provided for scatterers with isotropic
      and anisotropic displacement parameters (temperature factors).
      <p>
      The anisotropic displacement parameters have to be
      "u_star." Conversions between different conventions for
      the representation of anisotropic displacement
      parameters are provided by cctbx::adptbx.
      <p>
      One of the apply_symmetry() member functions has to be called before
      the scatterer can be used in a structure factor calculation.
   */
  template <typename FloatType=double,
            typename LabelType=std::string,
            typename ScatteringTypeType=std::string>
  class scatterer
  {
    public:
      //! Facilitates meta-programming.
      typedef FloatType float_type;
      //! Facilitates meta-programming.
      typedef LabelType label_type;
      //! Facilitates meta-programming.
      typedef ScatteringTypeType scattering_type_type;
      //
      typedef adptbx::anharmonic::GramCharlier<FloatType> anharmonic_adp_type;

      //! Default constructor. Data members are not initialized!
      scatterer() {}

      //! Initialization with isotropic displacement parameter.
      scatterer(LabelType const& label_,
                fractional<FloatType> const& site_,
                FloatType const& u_iso_,
                FloatType const& occupancy_,
                ScatteringTypeType const& scattering_type_,
                FloatType fp_,
                FloatType fdp_)
      :
        label(label_),
        scattering_type(scattering_type_),
        fp(fp_),
        fdp(fdp_),
        site(site_),
        occupancy(occupancy_),
        u_iso(u_iso_),
        u_star(-1,-1,-1,-1,-1,-1),
        flags(scatterer_flags::use_bit|scatterer_flags::use_u_iso_bit),
        multiplicity_(0),
        weight_without_occupancy_(0)
      {}

      //! Initialization with anisotropic displacement parameters.
      scatterer(LabelType const& label_,
                fractional<FloatType> const& site_,
                scitbx::sym_mat3<FloatType> const& u_star_,
                FloatType const& occupancy_,
                ScatteringTypeType const& scattering_type_,
                FloatType fp_,
                FloatType fdp_)
      :
        label(label_),
        scattering_type(scattering_type_),
        fp(fp_),
        fdp(fdp_),
        site(site_),
        occupancy(occupancy_),
        u_iso(-1),
        u_star(u_star_),
        flags(scatterer_flags::use_bit|scatterer_flags::use_u_aniso_bit),
        multiplicity_(0),
        weight_without_occupancy_(0)
      {}

      //! Direct access to label.
      LabelType label;

      //! Direct access to the scattering type.
      /*! See also: eltbx::xray_scattering::it1992,
                    eltbx::xray_scattering::wk1995
       */
      ScatteringTypeType scattering_type;

      //! Direct access to f-prime.
      /*! f-prime is the dispersive contribution to the scattering
          factor.
       */
      FloatType fp;

      //! Direct access to f-double-prime.
      /*! f-double-prime is the anomalous contribution to the scattering
          factor.
       */
      FloatType fdp;

      //! Direct access to fractional coordinates.
      /*! See also: apply_symmetry(), apply_symmetry_site()
       */
      fractional<FloatType> site;

      //! Direct access to occupancy factor.
      FloatType occupancy;

      //! Direct access to isotropic displacement parameter.
      FloatType u_iso;

      //! Direct access to anisotropic displacement parameters.
      /*! Conversions between isotropic and anisotropic displacement
          parameters are provided by cctbx::adptbx.
          <p>
          See also: apply_symmetry(), apply_symmetry_u_star()
       */
      scitbx::sym_mat3<FloatType> u_star;

      //anharmonic part of the ADP
      boost::shared_ptr<anharmonic_adp_type> anharmonic_adp;

      //! Support for refinement.
      scatterer_flags flags;

      void set_use_u(bool iso, bool aniso)
      {
        flags.set_use_u_iso(iso);
        flags.set_use_u_aniso(aniso);
        if(!iso) u_iso = -1;
        if (!aniso) {
          u_star.fill(-1);
          anharmonic_adp.reset();
        }
      }

      void set_use_u_iso_only()
      {
        set_use_u(true, false);
      }

      void set_use_u_aniso_only()
      {
        set_use_u(false, true);
      }

      //! Converts u_star to the equivalent u_iso in place.
      /*! The u_star values are reset to -1.
       */
      void
      convert_to_isotropic(
        uctbx::unit_cell const& unit_cell)
      {
        if (flags.use_u_aniso()) {
          if (!flags.use_u_iso()) u_iso = 0;
          u_iso += adptbx::u_star_as_u_iso(unit_cell, u_star);
          set_use_u(true, false);
        }
      }

      //! Converts u_iso to u_star in place.
      /*! The u_iso value is reset to -1.
       */
      void
      convert_to_anisotropic(
        uctbx::unit_cell const& unit_cell)
      {
        if (flags.use_u_iso()) {
          CCTBX_ASSERT(u_iso >= 0.0);
          if (!flags.use_u_aniso()) {
            u_star = adptbx::u_iso_as_u_star(unit_cell, u_iso);
          }
          else {
            u_star += adptbx::u_iso_as_u_star(unit_cell, u_iso);
          }
          set_use_u(false, true);
        }
      }

      //! Tests u_iso > 0 or adptbx::is_positive_definite(u_cart).
      bool
      is_positive_definite_u(
        uctbx::unit_cell const& unit_cell) const
      {
        if (flags.use_u_aniso()) {
          scitbx::sym_mat3<FloatType>
            u_cart = adptbx::u_star_as_u_cart(unit_cell, u_star);
          if (flags.use_u_iso()) {
            u_cart[0] += u_iso;
            u_cart[1] += u_iso;
            u_cart[2] += u_iso;
          }
          return adptbx::is_positive_definite(u_cart);
        }
        else if (flags.use_u_iso()) {
          return u_iso > 0;
        }
        return true;
      }

      /*! Tests u_iso >= u_cart_tolerance or
          adptbx::is_positive_definite(u_cart, u_cart_tolerance).
       */
      bool
      is_positive_definite_u(
        uctbx::unit_cell const& unit_cell,
        FloatType const& u_cart_tolerance) const
      {
        if (flags.use_u_aniso()) {
          scitbx::sym_mat3<FloatType>
            u_cart = adptbx::u_star_as_u_cart(unit_cell, u_star);
          if (flags.use_u_iso()) {
            u_cart[0] += u_iso;
            u_cart[1] += u_iso;
            u_cart[2] += u_iso;
          }
          return adptbx::is_positive_definite(u_cart, u_cart_tolerance);
        }
        else if (flags.use_u_iso()) {
          return u_iso >= -u_cart_tolerance;
        }
        return true;
      }

      bool is_anharmonic_adp() const {
        return anharmonic_adp != 0;
      }

      //! get u_iso as b_iso.
      FloatType
      b_iso(void) const
      {
        FloatType result = 0;
        result = adptbx::u_as_b(u_iso);
        return result;
      }

      //! Extracts sum of u_iso and u_star_as_u_iso (considering flags).
      FloatType
      u_iso_or_equiv(const uctbx::unit_cell* unit_cell) const
      {
        FloatType result = 0;
        if (flags.use_u_aniso()) {
          CCTBX_ASSERT(unit_cell != 0);
          result += adptbx::u_star_as_u_iso(*unit_cell, u_star);
        }
        if (flags.use_u_iso()) {
          result += u_iso;
        }
        return result;
      }

      //! Extracts sum of u_iso and u_star_as_u_cart (considering flags).
      scitbx::sym_mat3<double>
      u_cart_plus_u_iso(const uctbx::unit_cell* unit_cell) const
      {
        scitbx::sym_mat3<double>
          result = scitbx::sym_mat3<double>(0,0,0,0,0,0);
        if (flags.use_u_aniso()) {
          CCTBX_ASSERT(unit_cell != 0);
          result += adptbx::u_star_as_u_cart(*unit_cell, u_star);
        }
        if (flags.use_u_iso()) {
          for(unsigned i=0;i<3;i++) result[i] += u_iso;
        }
        return result;
      }

      //! Changes u_iso or u_star in place such that u_iso >= u_min.
      /*! In the anisotropic case the eigenvalues of u_cart are
          changed using adptbx::eigenvalue_filtering().
       */
      void
      tidy_u(
        uctbx::unit_cell const& unit_cell,
        sgtbx::site_symmetry_ops const& site_symmetry_ops,
        FloatType const& u_min,
        FloatType const& u_max,
        FloatType const& anisotropy_min)
      {
        if(flags.use_u_aniso()) {
           CCTBX_ASSERT(
             u_star != scitbx::sym_mat3<FloatType>(-1,-1,-1,-1,-1,-1));
           u_star = site_symmetry_ops.average_u_star(u_star);
           scitbx::sym_mat3<FloatType>
             u_cart = adptbx::u_star_as_u_cart(unit_cell, u_star);
           u_cart = adptbx::eigenvalue_filtering(u_cart, u_min, u_max);
           u_cart = adptbx::isotropize(u_cart, anisotropy_min);
           u_star = adptbx::u_cart_as_u_star(unit_cell, u_cart);
           u_star = site_symmetry_ops.average_u_star(u_star);
        }
        if(flags.use_u_iso()) {
           if(u_iso < u_min) u_iso = u_min;
           if(u_iso > u_max) u_iso = u_max;
        }
      }

      //! Changes u_iso or u_star in place by adding u_shift.
      /*! If u_shift is negative tidy_u() should be called
          after shift_u().
       */
      void
      shift_u(
        uctbx::unit_cell const& unit_cell,
        FloatType const& u_shift)
      {
        if (flags.use_u_aniso()) {
          u_star += adptbx::u_iso_as_u_star(unit_cell, u_shift);
        }
        else if (flags.use_u_iso()) {
          u_iso += u_shift;
        }
      }

      //! Changes occupancy in place by adding q_shift.
      void
      shift_occupancy(FloatType const& q_shift)
      {
        occupancy += q_shift;
      }

      /*! \brief Computes multiplicity(), weight_without_occupancy(),
          weight() and symmetry-averaged anisotropic displacement parameters.
       */
      /*! This member function or the other overload must be called before
          the scatterer is used in a structure factor calculation.

          See also:
            apply_symmetry_site,
            apply_symmetry_u_star,
            cctbx::sgtbx::site_symmetry,
            cctbx::sgtbx::site_symmetry::exact_site,
            cctbx::sgtbx::site_symmetry_ops::average_u_star
       */
      sgtbx::site_symmetry
      apply_symmetry(
        uctbx::unit_cell const& unit_cell,
        sgtbx::space_group const& space_group,
        FloatType const& min_distance_sym_equiv=0.5,
        FloatType const& u_star_tolerance=0,
        bool assert_min_distance_sym_equiv=true);

      /*! \brief Computes multiplicity(), weight_without_occupancy(),
          weight() and symmetry-averaged anisotropic displacement parameters.
       */
      /*! This member function or the other overload must be called before
          the scatterer is used in a structure factor calculation.

          See also:
            apply_symmetry_site,
            apply_symmetry_u_star
       */
      void
      apply_symmetry(
        sgtbx::site_symmetry_ops const& site_symmetry_ops,
        FloatType const& u_star_tolerance=0);

      //! Apply previously determined site symmetry to site.
      /*! Shorthand for: site = site_symmetry_ops.special_op() * site
       */
      void
      apply_symmetry_site(sgtbx::site_symmetry_ops const& site_symmetry_ops)
      {
        site = site_symmetry_ops.special_op() * site;
      }

      //! Apply previously determined site symmetry to u_star.
      /*! For scatterers with anisotropic displacement
          parameters, the symmetry-averaged u_star is determined
          using cctbx::sgtbx::site_symmetry_ops::average_u_star .
          If u_star_tolerance is greater than zero an
          exception is thrown if the discrepancy between
          components of u_star before and after the
          application of the site symmetry is greater than
          u_star_tolerance.
       */
      void
      apply_symmetry_u_star(
        sgtbx::site_symmetry_ops const& site_symmetry_ops,
        FloatType const& u_star_tolerance=0);

      //! Access to "multiplicity" computed by apply_symmetry().
      /** multiplicity is the number of symmetry equivalent positions
          of this scatterer in the unit cell.
       */
      int
      multiplicity() const { return multiplicity_; }

      //! Access to "weight_without_occupancy" computed by apply_symmetry().
      /*! weight_without_occupancy is defined as
              multiplicity() / space_group.order_z(),
          with space_group as passed to apply_symmetry().
          weight_without_occupancy() is used in the computation
          of structure factor derivatives.
       */
      FloatType
      weight_without_occupancy() const { return weight_without_occupancy_; }

      //! Access to "weight" computed by apply_symmetry().
      /*! The weight is defined as
              occupancy * multiplicity() / space_group.order_z(),
          with space_group as passed to apply_symmetry().
          The weight() is used in structure factor calculations.
       */
      FloatType
      weight() const { return weight_without_occupancy_ * occupancy; }

      //! Helper function for object serialization (Python pickle).
      /*! For internal use only.
       */
      void
      setstate(int multiplicity,
               FloatType weight_without_occupancy)
      {
        multiplicity_ = multiplicity;
        weight_without_occupancy_ = weight_without_occupancy;
      }

      //! Exception message. Not available in Python.
      std::string
      report_negative_u_iso(
        const char* where_file_name,
        long where_line_number) const
      {
        std::vector<char> buf(512);
        std::snprintf(&buf[0], buf.size(),
          "Negative u_iso: scatterer label=%s, u_iso=%.6g (%s, line %ld)",
          label.c_str(), u_iso, where_file_name, where_line_number);
        return std::string(&buf[0]);
      }

      //! Details for exception messages.
      std::string
      report_details(
        uctbx::unit_cell const& unit_cell,
        const char* prefix) const
      {
        std::string result;
        std::vector<char> buf_(512);
        char* buf = &buf_[0];
        size_t sz = buf_.size();
        std::snprintf(buf, sz, "%sscatterer label: %s\n",
          prefix, label.c_str()); result += buf;
        std::snprintf(buf, sz, "%sscattering type: %s\n",
          prefix, scattering_type.c_str()); result += buf;
        std::snprintf(buf, sz, "%sfractional coordinates: %.6f %.6f %.6f\n",
          prefix, site[0], site[1], site[2]); result += buf;
        cctbx::cartesian<FloatType> cart = unit_cell.orthogonalize(site);
        std::snprintf(buf, sz, "%scartesian coordinates: %.6f %.6f %.6f\n",
          prefix, cart[0], cart[1], cart[2]); result += buf;
        if (flags.use_u_iso()) {
          std::snprintf(buf, sz, "%su_iso: %.6g\n",
            prefix, u_iso); result += buf;
          std::snprintf(buf, sz, "%sb_iso: %.6g\n",
            prefix, adptbx::u_as_b(u_iso)); result += buf;
        }
        if (flags.use_u_aniso()) {
          scitbx::sym_mat3<FloatType> u = u_star;
          std::snprintf(buf, sz, "%su_star: %.6g %.6g %.6g %.6g %.6g %.6g\n",
            prefix, u[0], u[1], u[2], u[3], u[4], u[5]); result += buf;
          u = adptbx::u_star_as_u_cart(unit_cell, u_star);
          std::snprintf(buf, sz, "%su_cart: %.6g %.6g %.6g %.6g %.6g %.6g\n",
            prefix, u[0], u[1], u[2], u[3], u[4], u[5]); result += buf;
        }
        if (flags.use_u_aniso() && anharmonic_adp) {
          std::snprintf(buf, sz, "%sanharmonic ADP: order %d\n",
            prefix, anharmonic_adp->order); result += buf;
          std::snprintf(buf, sz, "%sC: %.6g %.6g %.6g %.6g %.6g %.6g\n",
            prefix, anharmonic_adp->C[0], anharmonic_adp->C[1],
            anharmonic_adp->C[2], anharmonic_adp->C[3],
            anharmonic_adp->C[4], anharmonic_adp->C[5]); result += buf;
          std::snprintf(buf, sz, "%sD: %.6g %.6g %.6g %.6g %.6g %.6g\n",
            prefix, anharmonic_adp->D[0], anharmonic_adp->D[1],
            anharmonic_adp->D[2], anharmonic_adp->D[3],
            anharmonic_adp->D[4], anharmonic_adp->D[5]); result += buf;
        }
        std::snprintf(buf, sz, "%soccupancy: %.6g\n",
          prefix, occupancy); result += buf;
        std::snprintf(buf, sz, "%sf-prime: %.6g\n",
          prefix, fp); result += buf;
        std::snprintf(buf, sz, "%sf-double-prime: %.6g",
          prefix, fdp); result += buf;
        return result;
      }

      const eltbx::tiny_pse::table& element_info() const {
        if (element_info_ == 0) {
          element_info_.reset(
            new eltbx::tiny_pse::table(scattering_type.empty() ? label : scattering_type));
          if (!element_info_->is_valid()) {
            element_info_.reset();
            CCTBX_ERROR("Could not locate given element label/symbol");
          }
        }
        return *element_info_;
      }

      int get_atomic_number() const { return element_info().atomic_number(); }
      std::string get_element_name() const { return std::string(element_info().name()); }
      float get_element_weight() const { return element_info().weight(); }
      // for use witin (-16..16)
      uint64_t get_id_2_16(short data=0, FloatType multiplier = 1) const {
        return scatterer_id_2<FloatType, 16>(
          element_info().atomic_number(), site, data, multiplier).id;
      }
      // for use within (-1..1)
      uint64_t get_id_2_1(short data = 0, FloatType multiplier = 1) const {
        return scatterer_id_2<FloatType, 1>(
          element_info().atomic_number(), site, data, multiplier).id;
      }
      // for use witin (-16..16)
      uint64_t get_id_5_16(short data = 0, FloatType multiplier = 1) const {
        return scatterer_id_5<FloatType, 16>(element_info().atomic_number(), site, data).id;
      }
      // for use within (-1..1)
      uint64_t get_id_5_1(short data = 0, FloatType multiplier = 1) const {
        return scatterer_id_5<FloatType, 1>(
          element_info().atomic_number(), site, data, multiplier).id;
      }

      template <class mask_info, uint64_t cell_m> 
      scatterer_id_base<FloatType, mask_info, cell_m>
        get_id(short data = 0, FloatType multiplier=1) const
      {
        return scatterer_id_base<FloatType, mask_info, cell_m>(
          element_info().atomic_number(), site, data, multiplier);
      }
    protected:
      int multiplicity_;
      FloatType weight_without_occupancy_;
      mutable boost::shared_ptr<eltbx::tiny_pse::table> element_info_;
  };

  template <typename FloatType,
            typename LabelType,
            typename ScatteringTypeType>
  sgtbx::site_symmetry
  scatterer<FloatType, LabelType, ScatteringTypeType>
  ::apply_symmetry(
    uctbx::unit_cell const& unit_cell,
    sgtbx::space_group const& space_group,
    FloatType const& min_distance_sym_equiv,
    FloatType const& u_star_tolerance,
    bool assert_min_distance_sym_equiv)
  {
    sgtbx::site_symmetry site_symmetry(
      unit_cell,
      space_group,
      site,
      min_distance_sym_equiv,
      assert_min_distance_sym_equiv);
    apply_symmetry(site_symmetry, u_star_tolerance);
    return site_symmetry;
  }

  template <typename FloatType,
            typename LabelType,
            typename ScatteringTypeType>
  void
  scatterer<FloatType, LabelType, ScatteringTypeType>
  ::apply_symmetry(
    sgtbx::site_symmetry_ops const& site_symmetry_ops,
    FloatType const& u_star_tolerance)
  {
    multiplicity_ = site_symmetry_ops.multiplicity();
    if (site_symmetry_ops.is_point_group_1()) {
      weight_without_occupancy_ = 1;
    }
    else {
      weight_without_occupancy_ = FloatType(1)
                                / site_symmetry_ops.matrices().size();
      apply_symmetry_site(site_symmetry_ops);
    }
    apply_symmetry_u_star(
      site_symmetry_ops,
      u_star_tolerance);
  }

  template <typename FloatType,
            typename LabelType,
            typename ScatteringTypeType>
  void
  scatterer<FloatType, LabelType, ScatteringTypeType>
  ::apply_symmetry_u_star(
    sgtbx::site_symmetry_ops const& site_symmetry_ops,
    FloatType const& u_star_tolerance)
  {
    if (flags.use_u_aniso() && !site_symmetry_ops.is_point_group_1()) {
      if (u_star_tolerance > 0.) {
        CCTBX_ASSERT(
          site_symmetry_ops.is_compatible_u_star(u_star, u_star_tolerance));
      }
      u_star = site_symmetry_ops.average_u_star(u_star);
    }
  }

}} // namespace cctbx::xray

#endif // CCTBX_XRAY_SCATTERER_H
