from __future__ import absolute_import, division, print_function
import cctbx.array_family.flex # import dependency

import boost_adaptbx.boost.python as bp
ext = bp.import_ext("cctbx_dmtbx_ext")
from cctbx_dmtbx_ext import *

@bp.inject_into(weighted_triplet_phase_relation)
class _():

  def format(self, miller_indices, ih=None):
    l = [miller_indices[self.ik()],
         self.friedel_flag_k(),
         miller_indices[self.ihmk()],
         self.friedel_flag_hmk(),
         self.ht_sum(),
         self.weight()]
    if (ih is not None):
      l.insert(0, miller_indices[ih])
    return " ".join([str(item).replace(" ", "") for item in l])

def triplet_generator(miller_set,
                      amplitudes=None, max_relations_per_reflection=0,
                      sigma_2_only=False, discard_weights=False):
  return ext.triplet_generator(
    miller_set.space_group(), miller_set.indices(),
    amplitudes, max_relations_per_reflection,
    sigma_2_only, discard_weights)

@bp.inject_into(ext.triplet_generator)
class _():

  def apply_tangent_formula(self, amplitudes, phases_rad,
                                  selection_fixed=None,
                                  use_fixed_only=False,
                                  reuse_results=False,
                                  sum_epsilon=1.e-10):
    return self.raw_apply_tangent_formula(
      amplitudes, phases_rad,
      selection_fixed, use_fixed_only, reuse_results, sum_epsilon)
