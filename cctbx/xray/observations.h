#ifndef CCTBX_XRAY_OBSERVATIONS
#define CCTBX_XRAY_OBSERVATIONS
#include <scitbx/array_family/shared.h>
#include <cctbx/miller/lookup_utils.h>
#include <cctbx/import_scitbx_af.h>
#include <cctbx/miller/match_indices.h>
#include <cctbx/xray/twin_component.h>

namespace cctbx { namespace xray {

  /* This class is to provide access to non-twinned, merohedrally twinned and
  multi-dataset (HKLF 5) observations in a unified way. Some notes:
    Merohedral components: always positive, for the case of racemic twinning
  both - normal and inversion components have to be provided. All merohedral
  components have to be given explicitly - i.e. if there is only one matrix
  which generates several components, all matrix products have to be given.
    The observations object keeps pointers to external twin_fraction and
  twin_component objects and therefore their life-time must be guaranteed to
  be at least as life-time of observation object if is being used. This can be
  easily done on the Python side by attaching the twin_fraction/component
  objects to an instance of the observations object.
    update_prime_fraction function must be called before scales or iterator
  methods can be used.

    Auxiliary objects provide HKLF5 specific filtering by resolution, intensity
  and elimination of systematic absences which is needed because most of the
  implemented operations work only on merged datasets.
  */
  template <typename FloatType>
  class observations {
  public:
    // iterator result
    struct index_twin_component {
      miller::index<> h;
      twin_fraction<FloatType> const* fraction;
      FloatType scale_;
      index_twin_component(miller::index<> const& h_,
        twin_fraction<FloatType> const* fraction,
        FloatType scale_)
        : h(h_),
        fraction(fraction),
        scale_(scale_)
      {}
      FloatType scale() const {
        return fraction == 0 ? scale_ : scale_*fraction->value;
      }
    };
    // HKLF 5 requires special filtering
    struct filter {
      uctbx::unit_cell unit_cell;
      sgtbx::space_group space_group;
      miller::lookup_utils::lookup_tensor<FloatType> omit_map;
      FloatType res_d_min, res_d_max, min_i_o_sig;
      bool res_filter;
      filter(uctbx::unit_cell const& unit_cell_,
        sgtbx::space_group const& space_group_, bool anomalous_flag,
        scitbx::af::const_ref<miller::index<> > const& omit_indices,
        FloatType res_d_min_, FloatType res_d_max_, FloatType min_i_o_sig_)
        : unit_cell(unit_cell_),
        space_group(space_group_),
        omit_map(omit_indices, space_group_, anomalous_flag),
        res_d_min(res_d_min_),
        res_d_max(res_d_max_),
        min_i_o_sig(min_i_o_sig_),
        res_filter(res_d_min_ > 0 || res_d_max_ > 0)
      {}

      bool is_to_omit(miller::index<> const& h,
        FloatType f_sq, FloatType sig) const
      {
        if (res_filter) {
          FloatType d = unit_cell.d(h);
          if (d <= res_d_min || (res_d_max > 0 && d >= res_d_max)) {
            return true;
          }
        }
        if (min_i_o_sig > 0 && f_sq < min_i_o_sig*sig) {
          return true;
        }
        return omit_map.find_hkl(h) >= 0;
      }
    };

    // observations filtering result
    struct filter_result {
      scitbx::af::shared<bool> selection;
      int omitted_count, sys_abs_count;
      filter_result(int sz)
        : selection(sz), omitted_count(0), sys_abs_count(0)
      {}
    };

    // iterator for twin components
    struct iterator {
      const int h_index;
      int current;
      observations const& parent;
      // copy constructor
//      iterator(iterator& i)
//        : h_index(i.h_index), current(i.current), parent(i.parent)
//      {}

      iterator(observations const &parent_, int h_index_)
        : parent(parent_), h_index(h_index_), current(-1)
      {}

      bool has_next() const {
        return parent.index_components_[h_index].size() > current + 1;
      }

      index_twin_component next() {
        CCTBX_ASSERT(has_next());
        local_twin_component const& ltw =
          parent.index_components_[h_index][++current];
        if (ltw.fraction == 0) {
          return index_twin_component(
            ltw.h,
            0, // refers to the prime scale
            parent.prime_fraction_
          );
        }
        return index_twin_component(
          ltw.h,
          ltw.fraction,
          1
        );
      }
      void reset() {
        current = -1;
      }
    };
  protected:
    struct local_twin_component {
      miller::index<> h;
      twin_fraction<FloatType> *fraction;
      local_twin_component(miller::index<> const& h,
        twin_fraction<FloatType>* fraction)
        : h(h), fraction(fraction)
      {}
    };

    scitbx::af::shared<miller::index<> > indices_;
    scitbx::af::shared<FloatType> data_, sigmas_;
    scitbx::af::shared<scitbx::af::shared<local_twin_component> >
      index_components_;
    scitbx::af::shared<twin_component<FloatType>*> merohedral_components_;
    scitbx::af::shared<twin_fraction<FloatType>*> twin_fractions_;
    scitbx::af::shared<int> measured_scale_indices_;
    mutable FloatType prime_fraction_;
    size_t total_data_cnt;

    miller::index<> generate(scitbx::mat3<FloatType> const& tl,
      miller::index<> const& h) const
    {
      return miller::index<>(
        scitbx::math::iround(tl[0]*h[0]+tl[3]*h[1]+tl[6]*h[2]),
        scitbx::math::iround(tl[1]*h[0]+tl[4]*h[1]+tl[7]*h[2]),
        scitbx::math::iround(tl[2]*h[0]+tl[5]*h[1]+tl[8]*h[2]));
    }

    void build_indices_twin_components(
      scitbx::af::shared<miller::index<> > const& indices,
      scitbx::af::shared<FloatType> const& data,
      scitbx::af::shared<FloatType> const& sigmas,
      scitbx::af::shared<int> const& scale_indices)
    {
      CCTBX_ASSERT(indices.size()==data.size());
      CCTBX_ASSERT(indices.size()==sigmas.size());
      CCTBX_ASSERT(indices.size()==scale_indices.size());
      index_components_.reserve(indices.size()); //better more than less
      measured_scale_indices_.reserve(indices.size());
      indices_.reserve(index_components_.size());
      data_.reserve(index_components_.size());
      sigmas_.reserve(index_components_.size());
      if (indices.size() != 0) {
        index_components_.push_back(scitbx::af::shared<local_twin_component>());
      }
      int index = 0;
      for (int i=0; i<indices.size(); i++) {
        if (scale_indices[i] < 0) {
          int s_ind = -scale_indices[i]-1;
          CCTBX_ASSERT(s_ind >= 0 && s_ind <= twin_fractions_.size());
          index_components_[index].push_back(
            local_twin_component(indices[i], s_ind == 0 ? 0 : twin_fractions_[s_ind-1]));
        }
        else {
          int s_ind = scale_indices[i];
          CCTBX_ASSERT(!(s_ind < 1 || s_ind > twin_fractions_.size()+1));
          measured_scale_indices_.push_back(s_ind);
          indices_.push_back(indices[i]);
          data_.push_back(data[i]);
          sigmas_.push_back(sigmas[i]);
          index++;
          index_components_.push_back(
            scitbx::af::shared<local_twin_component>());
        }
      }
    }
    void process_merohedral_components(sgtbx::space_group const& space_group,
      scitbx::af::shared<twin_component<FloatType>*> const& cmps)
    {
      CCTBX_ASSERT(indices_.size() == data_.size());
      CCTBX_ASSERT(indices_.size() == sigmas_.size());
      if (index_components_.size() != 0) {
        CCTBX_ASSERT(measured_scale_indices_.size() == indices_.size());
      }
      if (cmps.size() == 0) {
        return;
      }
      scitbx::af::shared<scitbx::mat3<FloatType> > merohedral_laws_;
      for (int i = 0; i < cmps.size(); i++) {
        merohedral_components_.push_back(cmps[i]);
        merohedral_laws_.push_back(cmps[i]->twin_law
          .as_floating_point(scitbx::type_holder<FloatType>()));
      }
      index_components_.reserve(indices_.size());
      measured_scale_indices_.reserve(indices_.size());
      for (int i = 0; i < indices_.size(); i++) {
        measured_scale_indices_.push_back(0);
        index_components_.push_back(
          scitbx::af::shared<local_twin_component>());

        for (int mci = 0; mci < merohedral_laws_.size(); mci++) {
          miller::index<> mi = generate(merohedral_laws_[mci], indices_[i]);
          if (!space_group.is_sys_absent(mi)) {
            index_components_[i].push_back(
              local_twin_component(mi, cmps[mci]));
          }
        }
      }
    }

  public:

    // hklf 4 + optional merohedral twinning
    observations(sgtbx::space_group const& space_group,
      scitbx::af::shared<miller::index<> > const& indices,
      scitbx::af::shared<FloatType> const& data,
      scitbx::af::shared<FloatType> const& sigmas,
      scitbx::af::shared<twin_component<FloatType>*> const&
        merohedral_components)
      : indices_(indices),
        data_(data),
        sigmas_(sigmas),
        prime_fraction_(1),
        total_data_cnt(data.size())
    {
      process_merohedral_components(space_group, merohedral_components);
    }

    // hklf 5
    observations(scitbx::af::shared<miller::index<> > const& indices,
      scitbx::af::shared<FloatType> const& data,
      scitbx::af::shared<FloatType> const& sigmas,
      scitbx::af::shared<int> const& scale_indices,
      scitbx::af::shared<twin_fraction<FloatType>*> const&
        twin_fractions)
      : twin_fractions_(twin_fractions),
      total_data_cnt(data.size())
    {
      build_indices_twin_components(indices, data, sigmas, scale_indices);
      update_prime_fraction();
    }

    /* customised copy constructor - with possible external twin fractions
    and twin components, used for the Flack parameter refinement, when the
    inversion twin law needs to be added
    */
    observations(const observations& obs,
      sgtbx::space_group const& space_group,
      scitbx::af::shared<twin_fraction<FloatType>*> const&
        twin_fractions,
      scitbx::af::shared<twin_component<FloatType>*> const&
        merohedral_components)
      : indices_(obs.indices_),
        data_(obs.data_),
        sigmas_(obs.sigmas_),
        measured_scale_indices_(obs.measured_scale_indices_),
        twin_fractions_(twin_fractions),
        total_data_cnt(obs.total_data_cnt)
    {
      CCTBX_ASSERT(twin_fractions.size()==obs.twin_fractions_.size());
      CCTBX_ASSERT(!(twin_fractions.size() != 0 && merohedral_components.size() != 0));
      process_merohedral_components(space_group, merohedral_components);
      update_prime_fraction();
    }

    bool has_twin_components() const {
      return index_components_.size() != 0;
    }

    iterator iterate(int i) const { return iterator(*this, i); }

    /* must be called before using scale(index) or iterator */
    void update_prime_fraction() const {
      FloatType sum=0;
      for (int i = 0; i < twin_fractions_.size(); i++) {
        sum += twin_fractions_[i]->value;
      }
      for (int i = 0; i < merohedral_components_.size(); i++) {
        sum += merohedral_components_[i]->value;
      }
      prime_fraction_ = 1 - sum;
    }

    // returns number of measured reflections
    int size() const { return indices_.size(); }

    // returns inex of a measured reflection
    miller::index<> const& index(int i) const { return indices_[i]; }

    // returns intensity of a measured reflection
    FloatType fo_sq(int i) const { return data_[i]; }

    // returns sigma of a measured reflection
    FloatType sig(int i) const { return sigmas_[i]; }

    // returns scale of a measured reflection
    FloatType scale(int i) const {
      FloatType rv = (measured_scale_indices_.size() == 0 ||
                      measured_scale_indices_[i] < 2) ? prime_fraction_
                      : twin_fractions_[measured_scale_indices_[i]-2]->value;
      return rv;
    }
    const twin_fraction<FloatType>* fraction(int i) const {
      return (measured_scale_indices_.size() == 0 ||
        measured_scale_indices_[i] < 2) ? 0
        : twin_fractions_[measured_scale_indices_[i] - 2];
    }

    observations detwin(
      sgtbx::space_group const& space_group,
      bool anomalous_flag,
      scitbx::af::const_ref<miller::index<> > const& fo_sq_indices,
      scitbx::af::const_ref<FloatType> const& fc_sqs,
      scitbx::af::shared<twin_component<FloatType>*> const&
        merohedral_components, bool complete) const
    {
      observations res = detwin(space_group, anomalous_flag, fo_sq_indices, fc_sqs, complete);
      res.process_merohedral_components(space_group, merohedral_components);
      return res;
    }

    observations detwin(
      sgtbx::space_group const& space_group,
      bool anomalous_flag,
      scitbx::af::const_ref<miller::index<> > const& fo_sq_indices,
      scitbx::af::const_ref<FloatType> const& fc_sqs, bool complete) const
    {
      if (!has_twin_components()) {
        return *this;
      }
      CCTBX_ASSERT(fo_sq_indices.size()==fc_sqs.size());
      miller::lookup_utils::lookup_tensor<FloatType>
        twin_map(fo_sq_indices, space_group, anomalous_flag);
      update_prime_fraction();
      const size_t sz = complete ? total_data_cnt : data_.size();
      scitbx::af::shared<FloatType> i_dtw((af::reserve(sz)));
      scitbx::af::shared<FloatType> s_dtw((af::reserve(sz)));
      scitbx::af::shared<miller::index<> > idx_dtw;
      if (complete) {
        idx_dtw.reserve(total_data_cnt);
      }
      for (int i = 0; i < data_.size(); i++) {
        long hi = twin_map.find_hkl(indices_[i]);
        CCTBX_ASSERT(hi >= 0);
        if (complete) {
          idx_dtw.push_back(indices_[i]);
        }
        FloatType fc_sq_prime = fc_sqs[hi];
        FloatType fc_sq_prime_scaled = fc_sq_prime * scale(i);
        FloatType twin_contrib = 0;
        iterator itr = iterate(i);
        while (itr.has_next()) {
          index_twin_component tw = itr.next();
          long ti = twin_map.find_hkl(tw.h);
          CCTBX_ASSERT(ti >= 0);
          twin_contrib += tw.scale() * fc_sqs[ti];
        }
        FloatType Fc_total = fc_sq_prime_scaled + twin_contrib;
        FloatType p_scale = (complete ? fc_sq_prime_scaled : fc_sq_prime) / Fc_total;
        i_dtw.push_back(data_[i] * p_scale);
        s_dtw.push_back(sigmas_[i] * p_scale);

        if (complete) {
          itr.reset();
          while (itr.has_next()) {
            index_twin_component tw = itr.next();
            long ti = twin_map.find_hkl(tw.h);
            FloatType t_scale = tw.scale() * fc_sqs[ti] / Fc_total;
            idx_dtw.push_back(tw.h);
            i_dtw.push_back(data_[i] * t_scale);
            s_dtw.push_back(sigmas_[i] * t_scale);
          }
        }
      }
      return observations(space_group, complete ? idx_dtw : indices_,
        i_dtw, s_dtw,
          scitbx::af::shared<twin_component<FloatType>*>());
    }

    /* Instead of de-twinning that create a Fo_sq matching Fc_sq, this procedure
    twins the Fc_sq set to match the Fo_sq set
    (similar to how this happens in the refinement)
    */
    observations twin(
      sgtbx::space_group const& space_group,
      bool anomalous_flag,
      scitbx::af::const_ref<miller::index<> > const& fo_sq_indices,
      scitbx::af::const_ref<FloatType> const& fc_sqs) const
    {
      if (!has_twin_components()) {
        return *this;
      }
      CCTBX_ASSERT(fo_sq_indices.size() == fc_sqs.size());
      miller::lookup_utils::lookup_tensor<FloatType>
        twin_map(fo_sq_indices, space_group, anomalous_flag);
      update_prime_fraction();
      scitbx::af::shared<FloatType> i_tw(data_.size());
      scitbx::af::shared<FloatType> s_tw(data_.size());
      for (int i = 0; i < data_.size(); i++) {
        long hi = twin_map.find_hkl(indices_[i]);
        CCTBX_ASSERT(hi >= 0);
        iterator itr = iterate(i);
        FloatType fc_sq = fc_sqs[hi] * scale(i);
        while (itr.has_next()) {
          index_twin_component twc = itr.next();
          hi = twin_map.find_hkl(twc.h);
          CCTBX_ASSERT(hi >= 0);
          fc_sq += twc.scale() * fc_sqs[hi];
        }
        i_tw[i] = fc_sq;
        s_tw[i] = sigmas_[i];
      }
      return observations(space_group, indices_, i_tw, s_tw,
        scitbx::af::shared<twin_component<FloatType>*>());
    }

    scitbx::af::shared<FloatType> const& data() const { return data_; }

    scitbx::af::shared<FloatType> const& sigmas() const { return sigmas_; }

    scitbx::af::shared<miller::index<> > const& indices() const { return indices_; }

    scitbx::af::shared<int> const& measured_scale_indices() const {
      return measured_scale_indices_;
    }

    scitbx::af::shared<twin_fraction<FloatType>*> const& twin_fractions() const {
      return twin_fractions_;
    }

    scitbx::af::shared<twin_component<FloatType>*> const& merohedral_components() const {
      return merohedral_components_;
    }

    // special filtering for HKLF 5
    static filter_result filter_data(
      scitbx::af::const_ref<miller::index<> > const& indices,
      scitbx::af::const_ref<FloatType> const& data,
      scitbx::af::const_ref<FloatType> const& sigmas,
      scitbx::af::const_ref<int> const& scale_indices,
      filter const& filter_)
    {
      CCTBX_ASSERT(indices.size()==data.size());
      CCTBX_ASSERT(indices.size()==sigmas.size());
      if (scale_indices.size() != 0) {
        CCTBX_ASSERT(scale_indices.size() == indices.size());
      }
      filter_result res(indices.size());
      for (int i=0; i<indices.size(); i++) {
        if (!(res.selection[i] = !filter_.is_to_omit(indices[i], data[i], sigmas[i]))) {
          if (scale_indices.size() != 0 && scale_indices[i] > 0) {
            res.omitted_count++;
          }
        }
      }
      if (scale_indices.size() != 0) {
        for (int i=indices.size()-1; i>=0; i--) {
          if (!res.selection[i]) {
            continue;
          }
          if (scale_indices[i] > 0) {
            bool remove = filter_.space_group.is_sys_absent(indices[i]);
            if (!remove) {
              int j = i;
              while (--j >= 0 && scale_indices[j] < 0) {
                if (filter_.space_group.is_sys_absent(indices[j]) && res.selection[j]) {
                  res.selection[j] = false;
                }
              }
            }
            else {
              res.sys_abs_count++;
              res.selection[i] = false;
              int j = i;
              while (--j >= 0 && scale_indices[j] < 0) {
                res.selection[j] = false;
              }
            }
          }
        }
      }
      return res;
    }
  }; //class observations
}} //namespace cctbx::xray
#endif // CCTBX_XRAY_OBSERVATIONS
