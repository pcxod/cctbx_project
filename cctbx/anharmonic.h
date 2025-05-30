#ifndef CCTBX_GRAM_CHARLIER_H
#define CCTBX_GRAM_CHARLIER_H

#include <vector>
#include <scitbx/matrix/tensors.h>
#include <scitbx/array_family/shared.h>
#include <cctbx/miller.h>
#include <scitbx/constants.h>

namespace cctbx { namespace adptbx { namespace anharmonic {
  using namespace scitbx::matrix::tensors;

  template <typename FloatType=double>
  struct GramCharlier {
    tensor_rank_3<FloatType> C;
    tensor_rank_4<FloatType> D;
    int order;

    int get_order() { return order; };

    static FloatType c_factor() {
      static const FloatType c_factor_ =
        -scitbx::constants::pi_sq * scitbx::constants::pi * 4 / 3;
      return c_factor_;
    }
    static FloatType d_factor() {
      static const FloatType d_factor_ =
        scitbx::constants::pi_sq * scitbx::constants::pi_sq * 2 / 3;
      return d_factor_;
    }
    static const std::complex<FloatType>& compl_c_factor() {
      static const std::complex<FloatType> compl_c_factor_ =
        std::complex<FloatType>(0, c_factor());
      return compl_c_factor_;
    }
    GramCharlier(int order) {
      CCTBX_ASSERT(order > 2 && order < 5);
      order = order;
    }

    GramCharlier(const af::shared<FloatType> &Cijk)
      : C(Cijk),order(3)
    {}

    GramCharlier(const af::shared<FloatType>& Cijk,
      const af::shared<FloatType>& Dijkl)
      : C(Cijk), D(Dijkl), order(4)
    {}

    std::complex<FloatType> calculate(const miller::index<> &h) const {
      using namespace scitbx::constants;
      FloatType c = C.sum_up(h);
      if (order == 3) {
        return std::complex<FloatType>(1, c * c_factor());
      }
      FloatType  d = D.sum_up(h);
      return std::complex<FloatType>(1 + d * d_factor(), c * c_factor());
    }

    af::shared<std::complex<FloatType> >
      gradient_coefficients(const miller::index<>& h) const
    {
      using namespace scitbx::constants;
      af::shared<std::complex<FloatType> > r(param_count());
      if (order >= 3) {
        af::shared<FloatType> c = C.gradient_coefficients(h);
        for (size_t i = 0; i < 10; i++) {
          r[i] = c[i] * compl_c_factor();
        }
      }
      if (order >= 4) {
        af::shared<FloatType> d = D.gradient_coefficients(h);
        std::complex<FloatType>* r_ = &r[10];
        for (size_t i = 0; i < 15; i++) {
          r_[i] = d[i] * d_factor();
        }
      }
      return r;
    }
    
    size_t param_count() const {
      return order == 3 ? 10 : 25;
    }

    void gradient_coefficients_in_place(
        const miller::index<>& h,
        std::vector<std::complex<FloatType> >& rv) const
    {
      if (order >= 3){
        C.gradient_coefficients(h, rv, 0);
        for (size_t i = 0; i < 10; i++) {
          rv[i] *= compl_c_factor();
        }
      }
      if(order >= 4){
        D.gradient_coefficients(h, rv, 10);
        for (size_t i = 10; i < 25; i++) {
          rv[i] *= d_factor();
        }
      }
    }

    af::shared<FloatType> data() const {
      af::shared<FloatType> rv(param_count());
      if (order >= 3) {
        for (size_t i = 0; i < 10; i++) {
          rv[i] = C[i];
        }
      }
      if (order >= 4) {
        for (size_t i = 0; i < 15; i++) {
          rv[i + 10] = D[i];
        }
      }
      return rv;
    }

  };

}}} // end of cctbx::adptbx::anharmonic
#endif
