#ifndef CCTBX_XRAY_TWIN_COMPONENT_H
#define CCTBX_XRAY_TWIN_COMPONENT_H

#include <cctbx/sgtbx/rot_mx.h>
#include <scitbx/array_family/shared.h>

namespace cctbx { namespace xray {

  template <typename FloatType>
  struct twin_fraction {
    FloatType value;
    bool grad;
    int grad_index;
    /* a generic attribute that allows to pass information about this particular
    fraction. In the case of ED will keep index to the associated zone/frame
    index
    */
    int tag;

    twin_fraction(const twin_fraction& tf)
      : value(tf.value), grad(tf.grad), grad_index(tf.grad_index),
      tag(tf.tag)
    {}
    twin_fraction(FloatType value_, bool grad_)
      : value(value_), grad(grad_), grad_index(-1), tag(-1)
    {}
    twin_fraction(FloatType value_, int tag, bool grad_)
      : value(value_), grad(grad_), grad_index(-1), tag(tag)
    {}
    twin_fraction deep_copy() { return twin_fraction(*this); }
  };

  template <typename FloatType>
  class twin_component : public twin_fraction<FloatType> {
  public:
    // copy constructor
    twin_component(twin_component const& tc)
      : twin_fraction<FloatType>(tc), twin_law(tc.twin_law)
    {}

    twin_component(sgtbx::rot_mx const &twin_law_,
      FloatType fraction, bool grad_fraction)
      : twin_fraction<FloatType>(fraction, grad_fraction),
        twin_law(twin_law_)
    {}
    twin_component deep_copy() { return twin_component(*this); }
    sgtbx::rot_mx twin_law;
  };

  template <typename FloatType>
  FloatType sum_twin_fractions(
    af::shared<twin_component<FloatType> *> twin_components)
  {
    FloatType result = 0.;
    for (std::size_t i_twin=0; i_twin<twin_components.size(); i_twin++) {
      result += twin_components[i_twin]->value;
    }
    return result;
  }

  template <typename FloatType>
  void set_grad_twin_fraction(
    af::shared<twin_component<FloatType> *> twin_components,
    bool grad_twin_fraction=true)
  {
    for (std::size_t i_twin=0; i_twin<twin_components.size(); i_twin++) {
      twin_components[i_twin]->grad = grad_twin_fraction;
    }
  }

}} // namespace cctbx::xray

#endif // GUARD
