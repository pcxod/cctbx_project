#ifndef SMTBX_REFINEMENT_LEAST_SQUARES_FC_ED_H
#define SMTBX_REFINEMENT_LEAST_SQUARES_FC_ED_H

#include <cctbx/xray/thickness.h>

#include <smtbx/refinement/least_squares_fc.h>
#include <smtbx/ED/ed_data.h>
#include <smtbx/ED/integrator.h>
#include <smtbx/ED/kinematic.h>
#include <smtbx/refinement/constraints/reparametrisation.h>

namespace smtbx {
  namespace refinement {
    namespace least_squares {
      using namespace smtbx::ED;
      using namespace refinement::constraints;

      template <typename FloatType>
      struct ed_shared_data {
        ED_UTIL_TYPEDEFS;
        typedef f_calc_function_base<FloatType> f_calc_function_base_t;
        typedef boost::shared_ptr< f_calc_function_base_t> f_calc_function_base_ptr_t;
        typedef builder_base<FloatType> data_t;
        typedef af::shared<const BeamInfo<FloatType>*> beam_at;
        typedef std::pair<int, af::shared<const BeamInfo<FloatType>*> > beam_me;
        typedef boost::shared_ptr<lookup_t> lookup_ptr_t;
        typedef cctbx::xray::fc_correction<FloatType> fc_correction_t;
        typedef boost::shared_ptr< fc_correction_t> fc_correction_ptr_t;

        ed_shared_data(const scitbx::sparse::matrix<FloatType>&
            Jt_matching_grad_fc,
          f_calc_function_base_t& f_calc_function,
          sgtbx::space_group const& space_group,
          bool anomalous_flag,
          af::shared<BeamGroup<FloatType> > beam_groups,
          cctbx::xray::thickness<FloatType> const& thickness,
          const RefinementParams<FloatType>& params,
          bool compute_grad,
          bool do_build = true)
          : Jt_matching_grad_fc(Jt_matching_grad_fc),
          f_calc_function(f_calc_function),
          space_group(space_group),
          params(params),
          Kl(params.getKl()),
          Fc2Ug(params.getFc2Ug()),
          mat_type(params.getMatrixType()),
          beam_groups(beam_groups),
          thickness(thickness),
          compute_grad(compute_grad),
          thread_n(params.getThreadN())
        {
          K = beam_groups[0].geometry->Kl_as_K(Kl);
          // build lookups for each beam_group + collect all indices and they diffs
          af::shared<miller::index<> > all_indices;
          size_t offset = 0;
          // treat equivalents independently inside the beam_groups
          sgtbx::space_group P1("P 1");
          for (size_t i = 0; i < beam_groups.size(); i++) {
            BeamGroup<FloatType>& beam_group = beam_groups[i];
            beam_groups_map.insert(std::make_pair(beam_group.id, &beam_group));
            beam_group.offset = offset;
            offset += beam_group.strong_measured_beams.size();
            // need only strong beams here??
            for (size_t hi = 0; hi < beam_group.indices.size(); hi++) {
              all_indices.push_back(beam_group.indices[hi]);
              for (size_t hj = hi + 1; hj < beam_group.indices.size(); hj++) {
                all_indices.push_back(beam_group.indices[hi] - beam_group.indices[hj]);
                all_indices.push_back(beam_group.indices[hj] - beam_group.indices[hi]);
              }
            }
            lookup_ptr_t mi_l(new lookup_t(
              af::select(beam_group.indices.const_ref(),
                beam_group.strong_measured_beams.const_ref()).const_ref(),
              P1,
              true));
            beam_group_lookups.insert(std::make_pair(beam_groups[i].id, mi_l));
          }
          beam_n = offset;
          // a tricky way of getting unique only...
          mi_lookup = lookup_t(
            all_indices.const_ref(),
            P1,
            anomalous_flag);
          indices = mi_lookup.get_unique();
          mi_lookup = lookup_t(
            indices.const_ref(),
            P1,
            anomalous_flag);
          if (do_build) {
            build();
          }
        }

        void build_Ug_matrix(const BeamGroup<FloatType>& beam_group,
          cmat_t &Ugs, af::shared<cmat_t> &Ds_kin)
        {
          af::shared<miller::index<> >
            strong_indices = af::select(beam_group.indices.const_ref(),
              beam_group.strong_beams.const_ref());
          utils<FloatType>::build_Ug_matrix(
            Ugs, Fcs_kin,
            mi_lookup,
            strong_indices
          );
          if (compute_grad && design_matrix_kin.accessor().n_columns() > 0) {
            utils<FloatType>::build_D_matrices(
              mi_lookup,
              strong_indices,
              design_matrix_kin,
              Ds_kin
            );
          }
        }

        void do_build_kin_mt() {
          if (thread_n < 0) {
            thread_n = builder_base<FloatType>::get_available_threads();
          }
          build_kin_mt(thread_n, Jt_matching_grad_fc, Fc2Ug, f_calc_function,
            indices, Fcs_kin, design_matrix_kin, compute_grad);
        }

        void process_beam_groups_mt() {
          if (thread_n < 0) {
            thread_n = builder_base<FloatType>::get_available_threads();
          }

          BeamGroup<FloatType> f0 = beam_groups[0];

          boost::thread_group pool;
          typedef beam_group_integrator<FloatType> integrator_t;
          typedef typename boost::shared_ptr<integrator_t> beam_group_integrator_ptr_t;
          size_t to = 0,
            n_param = design_matrix.accessor().n_columns();

          dyn_calculator_factory<FloatType> dc_f(mat_type);

          for (size_t fi = 0; fi < beam_groups.size(); fi += thread_n) {
            size_t t_end = std::min(thread_n, (int)(beam_groups.size() - fi));
            if (t_end == 0) {
              break;
            }
            std::vector<beam_group_integrator_ptr_t> accumulators;
            for (int thread_idx = 0; thread_idx < t_end; thread_idx++) {
              cmat_t Ugs;
              af::shared<FloatType> angles =
                beam_groups[to].get_int_angles(K, params.getIntSpan(),
                  params.getIntStep(),
                  params.getIntPoints(),
                  !params.isAngleInt());
              af::shared<cmat_t> Ds_kin;
              build_Ug_matrix(beam_groups[to], Ugs, Ds_kin);
              beam_group_integrator_ptr_t pf(
                new integrator_t(
                  new beam_group_processor<FloatType>(
                    dc_f,
                    beam_groups[to],
                    Ugs, K, thickness,
                    Ds_kin,
                    compute_grad),
                angles
              ));
              accumulators.push_back(pf);
              pool.create_thread(boost::ref(*pf));
              to++;
            }
            pool.join_all();
            for (int thread_idx = 0; thread_idx < t_end; thread_idx++) {
              integrator_t& p = *accumulators[thread_idx];
              if (p.exception_) {
                throw* p.exception_.get();
              }
              const BeamGroup<FloatType>& fi = p.processor->beam_group;
              //std::copy(p.Is.begin(), p.Is.end(), &Is[fi.offset]);
              for (size_t i = 0; i < p.beam_n; i++) {
                Is[fi.offset + i] = p.Is[i];
                if (compute_grad) {
                  for (size_t j = 0; j < n_param; j++) {
                    design_matrix(fi.offset + i, j) = p.D_dyn(i, j);
                  }
                }
              }
            }
          }
        }

        void build() {
          Is.resize(beam_n);
          Fcs.resize(beam_n);
          if (Fcs_kin.size() != indices.size()) {
            if (compute_grad) {
              size_t cn = Jt_matching_grad_fc.n_rows() - (thickness.grad ? 1 : 0);
              if (cn > 0) {
                design_matrix_kin.resize(
                  af::mat_grid(indices.size(), cn));
              }
            }
            Fcs_kin.resize(indices.size());
            do_build_kin_mt();
            // expand uniq Fc to beam_group indices
            size_t offset = 0;
            for (size_t i = 0; i < beam_groups.size(); i++) {
              const af::shared<miller::index<> >& fidx = beam_groups[i].indices;
              size_t measured = beam_groups[i].strong_measured_beams.size();
              for (size_t i = 0; i < measured; i++) {
                long idx = mi_lookup.find_hkl(fidx[i]);
                Fcs[offset + i] = Fcs_kin[idx];
              }
              offset += measured;
            }
          }
          if (compute_grad) {
            design_matrix.resize(af::mat_grid(indices.size(),
              Jt_matching_grad_fc.n_rows()));
          }
          process_beam_groups_mt();
        }


        /* computes the position of the given miller index of the given
        beam_group in the uniform arrays
        */
        size_t find_hkl(int beam_group_id, miller::index<> const& h) const {
          typename std::map<int, lookup_ptr_t>::const_iterator i =
            beam_group_lookups.find(beam_group_id);
          SMTBX_ASSERT(i != beam_group_lookups.end());
          typename std::map<int, BeamGroup<FloatType>*>::const_iterator fi =
            beam_groups_map.find(beam_group_id);
          long hi = i->second->find_hkl(h);
          SMTBX_ASSERT(hi >= 0);
          return hi + fi->second->offset;
        }

        const scitbx::sparse::matrix<FloatType>& Jt_matching_grad_fc;
        f_calc_function_base_t& f_calc_function;
        sgtbx::space_group const& space_group;
        af::shared<miller::index<> > indices;
        RefinementParams<FloatType> params;
        FloatType Kl, Fc2Ug;
        int mat_type;
        cart_t K;
        size_t beam_n;
        /* to lookup an index in particular beam_group, have to keep a copy of the
        indices
        */
        typename std::map<int, lookup_ptr_t> beam_group_lookups;
        typename std::map<int, BeamGroup<FloatType>*> beam_groups_map;
        af::shared<BeamGroup<FloatType> > beam_groups;
        cctbx::xray::thickness<FloatType> const& thickness;
        bool compute_grad;
        // newly-calculated, aligned by beam_groups
        af::shared<complex_t> Fcs, Fcs_kin;
        af::shared<FloatType> Is;
        // 
        mat_t design_matrix;
        cmat_t design_matrix_kin;
        lookup_t mi_lookup;
        int thread_n;
      };

      template <typename FloatType>
      class f_calc_function_ed : public f_calc_function_base<FloatType> {
      public:
        typedef f_calc_function_base<FloatType> f_calc_function_base_t;
        typedef ed_shared_data<FloatType> data_t;
        typedef scitbx::vec3<FloatType> cart_t;

        f_calc_function_ed(ed_shared_data<FloatType> const& data)
          : data(data),
          index(~0)
        {}
        f_calc_function_ed(f_calc_function_ed const& other)
          : data(other.data),
          index(~0)
        {}

        virtual void compute(
          miller::index<> const& h,
          boost::optional<std::complex<FloatType> > const& f_mask = boost::none,
          twin_fraction<FloatType> const* fraction = 0,
          bool compute_grad = true)
        {
          SMTBX_ASSERT(fraction->tag >= 0);
          index = data.find_hkl(fraction->tag, h);
          this->h = h;
        }

        virtual boost::shared_ptr<f_calc_function_base_t> fork() const {
          return boost::shared_ptr<f_calc_function_base_t>(
            new f_calc_function_ed(*this));
        }

        virtual std::complex<FloatType> get_f_calc() const {
          return data.Fcs[index];
        }
        virtual af::const_ref<std::complex<FloatType> > get_grad_f_calc() const {
          SMTBX_NOT_IMPLEMENTED();
          throw 1;
        }

        virtual FloatType get_observable() const {
          return data.Is[index];
        }

        virtual af::const_ref<FloatType> get_grad_observable() const {
          typedef af::versa_plain<FloatType> one_dim_type;
          typedef typename one_dim_type::accessor_type one_dim_accessor_type;
          one_dim_accessor_type a(data.design_matrix.accessor().n_columns());
          return af::const_ref<FloatType>(&data.design_matrix(index, 0), a);
        }

        virtual bool raw_gradients() const { return false; }
      private:
        ed_shared_data<FloatType> const& data;
        miller::index<> h;
        mutable af::shared<FloatType> grads;
        size_t index;
      };

    }
  }
}

#endif // GUARD
