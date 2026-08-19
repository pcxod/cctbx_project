"""The F(000) level of the solvent region, and what may be done to it.

F(000) is not measured, so the difference map has zero mean over the cell and
the region integrates to Q(1 - <w>) instead of Q. van der Sluis and Spek divide
that factor out. Holding the density non-negative keeps the same balance but
makes it implicit, so it is solved instead:

    Q = integral of max(w*(rho + Q/V), 0)

These check that the solve reproduces the published formula when nothing is
clamped, that it really is a fixed point when something is, and that limiting
the data used to find the solvent removes high angle content from f_mask
without touching the low angle content that carries the solvent.
"""
from __future__ import absolute_import, division, print_function

from cctbx import crystal, uctbx, xray
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from scitbx.math import approx_equal_relatively
from smtbx import masks


def structure_and_data(d_min=1.1):
  """Solvent that is in the data and not in the model - see tst_boundary_smearing."""
  cs = crystal.symmetry(unit_cell=uctbx.unit_cell((22, 22, 22, 90, 90, 90)),
                        space_group_symbol="P1")
  def blob(centre, n, element, radius=0.10):
    out = flex.xray_scatterer()
    for i in range(n):
      f = ((i*37 % 17)/17. - 0.5, (i*53 % 19)/19. - 0.5, (i*71 % 23)/23. - 0.5)
      site = tuple(centre[k] + 2*radius*f[k] for k in range(3))
      out.append(xray.scatterer(label="%s%d" % (element, i),
                                site=site, u=0.02, scattering_type=element))
    return out
  model = blob((0.28, 0.28, 0.28), 24, "C")
  everything = model.deep_copy()
  everything.extend(blob((0.76, 0.76, 0.76), 10, "O", radius=0.06))
  xs_model = xray.structure(crystal_symmetry=cs, scatterers=model)
  xs_full = xray.structure(crystal_symmetry=cs, scatterers=everything)
  fc = xs_full.structure_factors(d_min=d_min, algorithm="direct").f_calc()
  fo_sq = fc.as_intensity_array().customized_copy(
    sigmas=flex.double(fc.size(), 1.))
  return xs_model, fo_sq


def make(xs, fo_sq, **kw):
  mk = masks.mask(xs.deep_copy_scatterers(), fo_sq)
  mk.compute(solvent_radius=1.2, shrink_truncation_radius=1.2,
             resolution_factor=1/3.)
  for k, v in kw.items():
    assert hasattr(mk, k), k     # a typo here would silently test the default
    setattr(mk, k, v)
  mk.structure_factors()
  return mk


def exercise_level_solve():
  xs, fo_sq = structure_and_data()

  plain = make(xs, fo_sq)
  assert plain.n_voids() > 0, "no void, so there is no level to solve for"
  assert flex.max(flex.abs(plain.f_mask().data())) > 1e-6

  # unclamped, the solve must be the published formula and nothing else
  weight = plain.solvent_weight_map()
  mean_w = flex.sum(weight)/weight.size()
  raw = plain.diff_map.real_map_unpadded()*weight
  integral = flex.sum(raw)*plain.fft_scale
  assert approx_equal(plain.f_000_s, integral/(1 - mean_w), eps=1e-6), \
      (plain.f_000_s, integral/(1 - mean_w))
  print("\tvdS&S reproduced: %.3f e from a %.3f e integral over %.1f%% of the cell"
        % (plain.f_000_s, integral, 100*mean_w))

  # clamped, the answer must satisfy its own equation
  clamped = make(xs, fo_sq, void_positivity=True)
  weight = clamped.solvent_weight_map()
  v_cell = xs.unit_cell().volume()
  raw = clamped.diff_map.real_map_unpadded()*weight
  trial = raw + weight*(clamped.f_000_s/v_cell)
  trial.set_selected(trial < 0, 0)
  assert approx_equal_relatively(flex.sum(trial)*clamped.fft_scale,
                                 clamped.f_000_s, 1e-6),       (flex.sum(trial)*clamped.fft_scale, clamped.f_000_s)
  print("\tfixed point holds: Q = %.3f e" % clamped.f_000_s)

  # max(x, 0) >= x pointwise, so clamping can never lower the count
  assert clamped.f_000_s >= plain.f_000_s - 1e-6, \
      (clamped.f_000_s, plain.f_000_s)

  # and the per void counts must add up to what was reported
  assert approx_equal_relatively(sum(plain.electron_counts_per_void()),
                                 plain.f_000_s, 1e-9)


def exercise_band_limit():
  """Solvent is a low angle signal; the high angle map is the model's error."""
  xs, fo_sq = structure_and_data()
  full = make(xs, fo_sq)
  cut = make(xs, fo_sq, solvent_d_min=3.)
  assert full.f_mask().indices().all_eq(cut.f_mask().indices()), \
      "f_mask must still cover every reflection, only its source is limited"

  d = full.f_mask().d_spacings().data()
  lo, hi = d > 4., d < 2.
  def mean_abs(a, sel):
    return flex.mean(flex.abs(a.f_mask().data().select(sel)))
  print("\tmean |f_mask|  low angle %.3f -> %.3f, high angle %.3f -> %.3f"
        % (mean_abs(full, lo), mean_abs(cut, lo),
           mean_abs(full, hi), mean_abs(cut, hi)))
  assert mean_abs(cut, hi) < mean_abs(full, hi), \
      "band limiting did not remove high angle content"
  # the point of doing it this way rather than damping f_mask: the low angle
  # solvent, which is the part that was measured, survives
  assert mean_abs(cut, lo) > 0.5*mean_abs(full, lo), \
      "band limiting ate the low angle solvent it was supposed to keep"


def exercise_inward_taper():
  """The inward taper folds the outward half of the smear away."""
  xs, fo_sq = structure_and_data()
  mk = masks.mask(xs.deep_copy_scatterers(), fo_sq)
  mk.compute(solvent_radius=1.2, shrink_truncation_radius=1.2,
             resolution_factor=1/3.)
  hard = mk.solvent_weight_map()
  mk.boundary_smearing = 0.5
  mk.boundary_smearing_inward = True
  inward = mk.solvent_weight_map()
  mk.boundary_smearing_inward = False
  symmetric = mk.solvent_weight_map()
  outside = (hard < 0.5).iselection()
  leak_in = flex.sum(inward.select(outside))
  leak_sym = flex.sum(symmetric.select(outside))
  # Not asserted: that the inward taper is strictly contained. A smoothed
  # indicator is above 1/2 just outside a concave boundary, because most of
  # the neighbourhood there is inside the region, and folding at 1/2 cannot
  # remove that. The region here wraps a small blob and is about as concave
  # as a region gets, so this is the worst case rather than the typical one;
  # what has to hold is that the fold removes most of the outward half.
  assert leak_in < 0.25*leak_sym, (leak_in, leak_sym)
  assert leak_sym > 1e-3, "the symmetric taper should straddle the boundary"
  # a symmetric kernel is normalised, so it moves weight about rather than
  # creating it; the inward fold throws the outer half away
  assert flex.sum(inward) < flex.sum(symmetric)
  print("\tregion weight: hard %.0f, symmetric %.0f, inward %.0f grid points"
        % (flex.sum(hard), flex.sum(symmetric), flex.sum(inward)))
  print("	weight outside the region: symmetric %.1f, inward %.1f grid points"
        % (leak_sym, leak_in))


def run():
  exercise_level_solve()
  exercise_band_limit()
  exercise_inward_taper()
  print("OK")


if __name__ == '__main__':
  run()
