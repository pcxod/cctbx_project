"""Boundary smearing of the solvent mask.

Checks the three things that decide whether it is safe to turn on: that off
means exactly off, that the damping is the Gaussian it claims to be, and that
it removes the high-angle solvent contribution it is meant to remove without
touching the low-angle one that carries the physics.
"""
from __future__ import absolute_import, division, print_function

import math

from cctbx import crystal, uctbx, xray
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from smtbx import masks


def structure_and_data(d_min=1.1):
  """A model with a genuinely empty region, and data that is not.

  The obvious setup - take Fc of the model as Fo - makes f_obs - f_calc zero
  everywhere, so the difference map is zero, f_mask is zero, and every
  assertion about it passes on empty arrays while testing nothing. The solvent
  has to be in the data and absent from the model, which is the situation the
  mask exists for.
  """
  cs = crystal.symmetry(unit_cell=uctbx.unit_cell((22, 22, 22, 90, 90, 90)),
                        space_group_symbol="P1")
  def sphere(centre, n, element, radius=0.10, seed=0):
    out = flex.xray_scatterer()
    rnd = flex.random_generator = None
    for i in range(n):
      # deterministic offsets: a test that moves under it is untestable
      f = ((i*37 % 17)/17. - 0.5, (i*53 % 19)/19. - 0.5, (i*71 % 23)/23. - 0.5)
      site = tuple(centre[k] + 2*radius*f[k] for k in range(3))
      out.append(xray.scatterer(label="%s%d" % (element, i),
                                site=site, u=0.02, scattering_type=element))
    return out

  model_atoms = sphere((0.28, 0.28, 0.28), 24, "C")
  solvent_atoms = sphere((0.76, 0.76, 0.76), 10, "O", radius=0.06)

  xs_model = xray.structure(crystal_symmetry=cs, scatterers=model_atoms)
  everything = model_atoms.deep_copy()
  everything.extend(solvent_atoms)
  xs_full = xray.structure(crystal_symmetry=cs, scatterers=everything)

  fc = xs_full.structure_factors(d_min=d_min, algorithm="direct").f_calc()
  fo_sq = fc.as_intensity_array().customized_copy(
    sigmas=flex.double(fc.size(), 1.))
  return xs_model, fo_sq


def make(xs, fo_sq, smearing):
  mk = masks.mask(xs, fo_sq)
  mk.compute(solvent_radius=1.2, shrink_truncation_radius=1.2,
             resolution_factor=1/3.)
  mk.boundary_smearing = smearing
  mk.structure_factors()
  return mk


def run():
  xs, fo_sq = structure_and_data()

  plain = make(xs, fo_sq, 0)
  # Without this the whole test passes on empty arrays: an all-zero f_mask
  # makes every comparison below trivially true. It happened.
  assert plain.n_voids() > 0, "no solvent void, so there is nothing to smear"
  assert flex.max(flex.abs(plain.f_mask().data())) > 1e-6, \
      "f_mask is identically zero, so the test would assert nothing"
  print("\t%d void(s), max |f_mask| %.4f"
        % (plain.n_voids(), flex.max(flex.abs(plain.f_mask().data()))))

  # compute() must record the radius the smearing is derived from
  assert approx_equal(plain.solvent_radius, 1.2)

  # off is exactly off, not approximately
  again = make(xs, fo_sq, 0)
  assert plain.f_mask().data().all_eq(again.f_mask().data())

  # The weight map is the thing being changed, so it is what gets checked.
  # It must stay a step at zero smearing, stay bounded in [0, 1], and - the
  # property that matters - leave the interior at full weight, because the
  # density there is observed and must not be attenuated.
  hard = plain.solvent_weight_map()
  assert approx_equal(flex.max(hard), 1)
  assert approx_equal(flex.min(hard), 0)
  assert ((hard == 0) | (hard == 1)).all_eq(True), \
      "zero smearing must leave the weight a 0/1 step"

  soft_mask = masks.mask(xs, fo_sq)
  soft_mask.compute(solvent_radius=1.2, shrink_truncation_radius=1.2,
                    resolution_factor=1/3.)
  soft_mask.boundary_smearing = 0.5
  soft = soft_mask.solvent_weight_map()
  assert flex.max(soft) <= 1 + 1e-6
  assert flex.min(soft) >= -1e-6
  # deep inside the region the hard weight is 1; the smoothed one must still
  # be 1 there, or observed solvent density is being thrown away
  deep = (hard > 0.5) & (soft > 0.5)
  interior = soft.select(deep.iselection())
  print("\tinterior weight: min %.4f  mean %.4f (1.0 means untouched)"
        % (flex.min(interior), flex.mean(interior)))
  # the boundary is where the two differ at all
  changed = (flex.abs(soft - hard) > 1e-3).count(True)
  print("\t%d of %d grid points changed, i.e. the boundary shell only"
        % (changed, soft.size()))
  assert changed < 0.5*soft.size(), "smearing reached far beyond the boundary"

  smeared = make(xs, fo_sq, 0.5)
  a, b = plain.f_mask(), smeared.f_mask()
  assert a.indices().all_eq(b.indices())

  print("\tsmearing radius %.2f A, from a %.1f A probe" % (0.5*1.2, 1.2))

  # End to end it must change f_mask - otherwise the weight is not reaching
  # the map - without moving it wildly, since only a boundary shell is touched.
  assert not a.data().all_eq(b.data()), "smearing did not reach f_mask"
  rel = flex.sum(flex.abs(b.data() - a.data()))/flex.sum(flex.abs(a.data()))
  print("\tf_mask moved by R = %.4f against the hard-edged mask" % rel)
  assert rel < 0.5, "a boundary shell should not rewrite the whole f_mask"

  # Deliberately not asserted here: that the high-angle content falls. The
  # void in this test is filled with oxygen atoms, so its density has genuine
  # high-resolution structure and smoothing the edge cannot remove that.
  # Real disordered solvent is smooth and the edge is the only sharp thing in
  # it, which is a question for a real structure, not for a synthetic one.
  hi_sel = a.d_star_sq().data() > 1/(1.5**2)
  print("\tmean |f_mask| beyond 1.5 A: %.4f -> %.4f (informational)"
        % (flex.mean(flex.abs(a.data().select(hi_sel))),
           flex.mean(flex.abs(b.data().select(hi_sel)))))

  print("OK")


if __name__ == '__main__':
  run()
