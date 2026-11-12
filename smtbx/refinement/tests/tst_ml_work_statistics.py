"""Under a likelihood target with free flags, R1, wR2, the goodness of fit and
the scale describe the working reflections only; r_free() the free ones. Least
squares is unchanged.
"""
from __future__ import absolute_import, division, print_function

import random

from cctbx.array_family import flex
from cctbx.development import random_structure
from scitbx.lstbx import normal_eqns
from smtbx.refinement import least_squares, constraints, sigma_a
import smtbx.utils

def observations():
  xs = random_structure.xray_structure(
    space_group_symbol="P 21 21 21",
    elements=["C"]*14 + ["O"]*3, u_iso=0.045, random_u_iso=True)
  fo_sq = xs.structure_factors(d_min=1.0).f_calc().norm()
  # perturbed, so that the working and the free set have different R
  fo_sq = fo_sq.customized_copy(
    data=fo_sq.data()*flex.double(
      [random.uniform(0.9, 1.1) for _ in range(fo_sq.size())]),
    sigmas=flex.double([max(0.5, 0.04*v) for v in fo_sq.data()]))
  fo_sq.set_observation_type_xray_intensity()
  free = fo_sq.generate_r_free_flags(fraction=0.1, max_free=None).data()
  return xs, fo_sq, free

def build(xs, fo_sq, free, target):
  rep = constraints.reparametrisation(
    structure=xs.deep_copy_scatterers(), constraints=[],
    connectivity_table=smtbx.utils.connectivity_table(xs))
  engine = (normal_eqns.non_linear_ls_with_fixed_scale_factor_BLAS_2 if target
            else normal_eqns.non_linear_ls_with_separable_scale_factor_BLAS_2)
  klass = least_squares.crystallographic_ls_class(
    non_linear_ls_with_separable_scale_factor=engine,
    n_parameters=rep.n_independents, may_parallelise=False)
  kwds = dict(weighting_scheme=least_squares.unit_weighting())
  if target:
    kwds.update(ml_target=target, ml_free_flags=free,
                ml_free_reflections_per_bin=40)
  ls = klass(fo_sq.as_xray_observations(), rep, **kwds)
  ls.build_up(objective_only=True)
  return ls

def exercise_work_statistics():
  if not sigma_a.is_available():
    print("\tskipped: the alpha/beta estimator is not available here")
    return
  xs, fo_sq, free = observations()
  ls = build(xs, fo_sq, free, None)
  assert ls.r1_factor()[1] == fo_sq.size()
  for target in ("mlf", "mli"):
    ml = build(xs, fo_sq, free, target)
    r1, n = ml.r1_factor()
    assert n == free.count(False), (target, n, free.count(False))
    rf, nf = ml.r_free()
    assert nf == free.count(True)
    # R1 by hand over the working set, at the same scale
    k = ml.scale_factor()
    fo = fo_sq.select(~free).f_sq_as_f()
    fc = ml.fc_sq.select(~free).f_sq_as_f()
    r1_hand = fo.r1_factor(fc, scale_factor=k**0.5, assume_index_matching=True)
    assert abs(r1 - r1_hand) < 1e-12, (r1, r1_hand)
    print(target, "R1(work) %.5f n=%d  R_free %.5f n=%d  wR2 %.4f"
          % (r1, n, rf, nf, ml.wR2()))

def run():
  random.seed(3)
  flex.set_random_seed(3)
  exercise_work_statistics()
  print('OK')

if __name__ == '__main__':
  run()
