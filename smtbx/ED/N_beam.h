#pragma once
#include <smtbx/ED/ed_data.h>
#include <smtbx/ED/utils.h>
#include <smtbx/ED/dyn_calculator.h>

namespace smtbx { namespace ED
{
  using namespace cctbx;
  
  template <typename FloatType>
  class dyn_calculator_n_beam {
  public:
    ED_UTIL_TYPEDEFS;

    dyn_calculator_n_beam(
      size_t beam_N,
      int mat_type,
      const BeamGroup<FloatType>& beam_group,
      const cart_t& K,
      FloatType thickness, bool useSg, FloatType wght)
      : dc_f(mat_type),
      beam_group(beam_group), beam_n(beam_N),
      K(K),
      thickness(thickness),
      wght(wght),
      useSg(useSg)
    {
      strong_indices = af::select(beam_group.indices.const_ref(),
        beam_group.strong_beams.const_ref());
    }

    /* builds the potential matrix in dc; init must be called before this
    function!! */
    void build(const mat3_t &R) {
      dc->reset(A, R);
    }

    complex_t calc_amp(const mat3_t& R, size_t idx=1) {
      return dc->reset(A, R)
        .calc_amps_1(idx);
    }

    //D_dyn has one row as output
    complex_t calc_amp_ext(const mat3_t& R,
      const af::shared<cmat_t>& Ds_kin,
      bool grad_thickness,
      mat_t& D_dyn)
    {
      return dc->reset(A, R)
        .calc_amps_ext_1(Ds_kin, grad_thickness, D_dyn, 1);
    }

    // recomputes the Eigen matrix
    dyn_calculator_n_beam& init(const miller::index<> &h, FloatType angle,
      const af::shared<complex_t> & Fcs_kin, const lookup_t  &mi_lookup)
    {
      return init(h, beam_group.get_R(angle), Fcs_kin, mi_lookup);
    }

    // recomputes the Eigen matrix
    dyn_calculator_n_beam& init(const miller::index<>& h,
      const af::shared<FloatType> &angles,
      const af::shared<complex_t>& Fcs_kin, const lookup_t& mi_lookup)
    {
      af::shared<mat3_t> RMfs(af::reserve(angles.size()));
      for (size_t i = 0; i < angles.size(); i++) {
        RMfs.push_back(beam_group.get_R(angles[i]));
      }
      return init(h, RMfs, Fcs_kin, mi_lookup);
    }

    // recomputes the Eigen matrix
    dyn_calculator_n_beam& init(const miller::index<>& h,
      const mat3_t& RMf,
      const af::shared<complex_t>& Fcs_kin, const lookup_t& mi_lookup)
    {
      indices = utils<FloatType>::build_Ug_matrix_N(A, Fcs_kin, mi_lookup,
        strong_indices, K, h, RMf, beam_n, useSg, wght);
      dc = dc_f.make(indices, K, beam_group.get_N(), thickness);
      return *this;
    }

    // recomputes the Eigen matrix
    dyn_calculator_n_beam& init(const miller::index<>& h,
      const af::shared<mat3_t>& RMfs,
      const af::shared<complex_t>& Fcs_kin, const lookup_t& mi_lookup)
    {
      indices = utils<FloatType>::build_Ug_matrix_N_ext(A, Fcs_kin, mi_lookup,
        strong_indices, K, h, RMfs, beam_n, wght);
      dc = dc_f.make(indices, K, beam_group.get_N(), thickness);
      return *this;
    }
    // indices selected for the Ug matrix - intialised by 'build'
    af::shared<miller::index<> > indices;
    const cmat_t& get_matrix() const {
      return A;
    }
    a_dyn_calculator<FloatType>&  get_dc() const {
      return *dc;
    }
  protected:
    dyn_calculator_factory<FloatType> dc_f;
    const BeamGroup<FloatType>& beam_group;
    af::shared<miller::index<> > strong_indices;
    boost::shared_ptr<a_dyn_calculator<FloatType> > dc;
    size_t beam_n;
    cmat_t A;
    cart_t K;
    FloatType thickness, wght;
    bool useSg;
  };

}}
