#pragma once
#ifndef SMTBX_REFINEMENT_LEAST_SQUARES_TWINNING_H
#define SMTBX_REFINEMENT_LEAST_SQUARES_TWINNING_H

/// Crystallographic least-squares

#include <scitbx/sparse/matrix.h>

#include <cctbx/xray/observations.h>

#include <smtbx/error.h>
#include <smtbx/refinement/least_squares_fc.h>


namespace smtbx {
  namespace refinement {
    namespace least_squares {
      template <typename FloatType>
      struct MaskData {
        typedef std::complex<FloatType> complex_type;

        MaskData(af::const_ref<complex_type> const& f_mask)
          : f_mask(f_mask)
        {}

        MaskData(cctbx::xray::observations<FloatType> const& reflections,
          sgtbx::space_group const& space_group,
          bool anomalous_flag,
          af::const_ref<miller::index<> > const& indices,
          af::const_ref<complex_type> const& f_mask)
          : f_mask(f_mask)
        {
          mi_lookup = miller::lookup_utils::lookup_tensor<FloatType>(
            indices, space_group, anomalous_flag);
        }

        complex_type find(miller::index<> const& h) const {
          long index = mi_lookup.find_hkl(h);
          SMTBX_ASSERT(index >= 0 && index < f_mask.size())(h.as_string());
          return f_mask[index];
        }

        int size() const {
          return f_mask.size();
        }

        const complex_type& get(int i) const {
          return f_mask[i];
        }

      private:
        af::const_ref<complex_type> f_mask;
        miller::lookup_utils::lookup_tensor<FloatType> mi_lookup;
      };

      template <typename FloatType>
      class twinning_processor {
      public:
        typedef typename cctbx::xray::observations<FloatType>::iterator itr_t;
        typedef typename cctbx::xray::twin_fraction<FloatType> twf_t;
        typedef typename cctbx::xray::observations<FloatType>::index_twin_component twc_t;
        typedef std::complex<FloatType> complex_type;
      protected:
        void update_component_grads(const twf_t* fraction,
          af::shared<FloatType>& gradients,
          FloatType obs) const
        {
          if (fraction == 0) {
            const scitbx::af::shared<twin_component<FloatType>*>& tcs =
              reflections.merohedral_components();
            for (size_t i = 0; i < tcs.size(); i++) {
              if (tcs[i]->grad) {
                SMTBX_ASSERT(tcs[i]->grad_index >= 0 && tcs[i]->grad_index < gradients.size());
                gradients[tcs[i]->grad_index] -= obs;
              }
            }
            const scitbx::af::shared<twin_fraction<FloatType>*>& twf =
              reflections.twin_fractions();
            for (size_t i = 0; i < twf.size(); i++) {
              if (twf[i]->grad) {
                SMTBX_ASSERT(twf[i]->grad_index >= 0 && twf[i]->grad_index < gradients.size());
                gradients[twf[i]->grad_index] -= obs;
              }
            }
          }
          else if (fraction->grad) {
            SMTBX_ASSERT(fraction->grad_index >= 0 && fraction->grad_index < gradients.size());
            gradients[fraction->grad_index] += obs;
          }
        }
      public:
        
        twinning_processor(
          cctbx::xray::observations<FloatType> const& reflections,
          MaskData<FloatType> const& f_mask_data,
          bool compute_grad,
          scitbx::sparse::matrix<FloatType> const&
            jacobian_transpose_matching_grad_fc)
          : reflections(reflections),
          f_mask_data(f_mask_data),
          compute_grad(compute_grad),
          jacobian_transpose_matching_grad_fc(jacobian_transpose_matching_grad_fc)
        {}

        FloatType process(int i_h,
          f_calc_function_base<FloatType>& f_calc_function,
          af::shared<FloatType>& gradients) const
        {
          FloatType obs = f_calc_function.get_observable();
          if (reflections.has_twin_components()) {
            const twf_t* measured_fraction = reflections.fraction(i_h);
            itr_t itr = reflections.iterate(i_h);
            FloatType obs_scale = reflections.scale(i_h);
            if (compute_grad) {
              gradients *= obs_scale;
              update_component_grads(measured_fraction, gradients, obs);
            }
            obs *= obs_scale;
            while (itr.has_next()) {
              twc_t twc = itr.next();
              boost::optional<complex_type> f_mask = boost::none;
              if (f_mask_data.size() > 0) {
                f_mask = f_mask_data.find(twc.h);
              }
              f_calc_function.compute(twc.h, f_mask, measured_fraction, compute_grad);
              obs += twc.scale() * f_calc_function.get_observable();
              if (compute_grad) {
                if (f_calc_function.raw_gradients()) {
                  af::shared<FloatType> tmp_gradients =
                    jacobian_transpose_matching_grad_fc * f_calc_function.get_grad_observable();
                  gradients += twc.scale() * tmp_gradients;
                }
                else {
                  af::shared<FloatType> tmp_gradients(
                    f_calc_function.get_grad_observable().begin(),
                    f_calc_function.get_grad_observable().end());
                  gradients += twc.scale() * tmp_gradients;
                }
                update_component_grads(twc.fraction, gradients, f_calc_function.get_observable());
              }
            }
          }
          return obs;
        }
      private:
        cctbx::xray::observations<FloatType> const& reflections;
        MaskData<FloatType> const& f_mask_data;
        bool compute_grad;
        scitbx::sparse::matrix<FloatType> const&
          jacobian_transpose_matching_grad_fc;
      };
    }
  }
}


#endif // GUARD
