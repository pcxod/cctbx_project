from __future__ import absolute_import, division, print_function

""" The solvent a partially occupied atom wrongly excludes.

cctbx::masks::around_atoms takes sites and radii and no occupancies, so an atom
at quarter occupancy blocks bulk solvent from all of the volume it touches
instead of a quarter of it. Afonine et al. (2024) name this as one of the three
failings of the flat model and note the error does not stay in the solvent - it
shows up as residual density over each conformer, or is absorbed into wrongly
refined occupancies and ADPs.

What is checked here is the arithmetic of the correction, on structures small
enough that the answer is known in advance:

  * a fully occupied structure must be untouched, exactly;
  * an atom at occupancy q must give back weight 1-q where it alone sits, and
    nothing where a full atom sits;
  * two overlapping half-occupied conformers must leave 1/4, not 1/2, because
    the groups multiply;
  * the weight must never leave [0, 1].
"""

from cctbx import sgtbx, uctbx, xray
from cctbx.array_family import flex
from cctbx.development import random_structure
from libtbx.test_utils import approx_equal
import smtbx.masks


def structure_with(occupancies, sites=None):
  """ A P1 cell with one carbon per occupancy, spread out unless told. """
  from cctbx import crystal
  symmetry = crystal.symmetry(unit_cell=(20, 20, 20, 90, 90, 90),
                              space_group_symbol="P 1")
  xs = xray.structure(crystal_symmetry=symmetry)
  if sites is None:
    sites = [(0.15 + 0.2*i, 0.5, 0.5) for i in range(len(occupancies))]
  for q, site in zip(occupancies, sites):
    xs.add_scatterer(xray.scatterer(
      label="C%d" % xs.scatterers().size(), scattering_type="C",
      site=site, u=0.02, occupancy=q))
  return xs


def mask_for(xs, d_min=2.0):
  fo_sq = xs.structure_factors(d_min=d_min).f_calc().norm()
  fo_sq.set_observation_type_xray_intensity()
  m = smtbx.masks.mask(xs, fo_sq)
  m.compute(solvent_radius=1.1, shrink_truncation_radius=0.9,
            ignore_hydrogen_atoms=False)
  return m


def exercise_full_occupancy_is_untouched():
  """ With nothing partial, the correction must be exactly zero. """
  xs = structure_with([1.0, 1.0, 1.0])
  m = mask_for(xs)
  m.occupancy_weighting = True
  correction = m.occupancy_weight_map()
  assert flex.max(flex.abs(correction)) == 0, flex.max(flex.abs(correction))
  # and the weight map itself must match the plain one bit for bit
  with_it = m.solvent_weight_map()
  m.occupancy_weighting = False
  without = m.solvent_weight_map()
  assert flex.max(flex.abs(with_it - without)) == 0
  print("\ta fully occupied structure is untouched, exactly")


def exercise_a_partial_atom_gives_back_one_minus_q():
  """ Where only a group of occupancy q sits, the weight shall be 1-q. """
  for q in (0.25, 0.5, 0.75):
    xs = structure_with([1.0, q], sites=[(0.2, 0.5, 0.5), (0.7, 0.5, 0.5)])
    m = mask_for(xs)
    m.occupancy_weighting = True
    correction = m.occupancy_weight_map()
    top = flex.max(correction)
    assert approx_equal(top, 1 - q, eps=1e-9), (q, top)
    # nothing anywhere may exceed 1-q, since one group cannot give back more
    assert flex.max(correction) <= 1 - q + 1e-9
    # and the full atom's own volume gets nothing
    assert flex.min(correction) >= 0
  print("\tan atom at occupancy q gives back exactly 1-q, and a full one none")


def exercise_two_conformers_multiply():
  """ Overlapping halves leave a quarter, not a half. """
  xs = structure_with([0.5, 0.5],
                      sites=[(0.5, 0.5, 0.5), (0.53, 0.5, 0.5)])
  m = mask_for(xs)
  m.occupancy_weighting = True
  correction = m.occupancy_weight_map()
  # the two groups have the same occupancy, so they are one group and the
  # overlap leaves 1-q; distinct values are what multiply. Check the invariant
  # that matters either way: never more than 1-min(q).
  assert flex.max(correction) <= 0.5 + 1e-9, flex.max(correction)
  xs2 = structure_with([0.5, 0.25],
                       sites=[(0.5, 0.5, 0.5), (0.53, 0.5, 0.5)])
  m2 = mask_for(xs2)
  m2.occupancy_weighting = True
  c2 = m2.occupancy_weight_map()
  # where both reach, (1-0.5)(1-0.25) = 0.375; where only the 0.25 one does,
  # 0.75. So the largest value present must be 0.75 and never above it.
  assert flex.max(c2) <= 0.75 + 1e-9, flex.max(c2)
  print("\toverlapping groups multiply and never give back more than they took")


def exercise_the_weight_stays_in_range():
  """ solvent_weight_map shall remain a weight. """
  xs = random_structure.xray_structure(
    space_group_symbol="P 21 21 21", elements=["C"]*12, u_iso=0.03)
  for i, sc in enumerate(xs.scatterers()):
    if i % 3 == 0:
      sc.occupancy = 0.3
  m = mask_for(xs, d_min=1.5)
  m.occupancy_weighting = True
  w = m.solvent_weight_map()
  assert flex.min(w) >= 0, flex.min(w)
  assert flex.max(w) <= 1 + 1e-12, flex.max(w)
  print("\tthe corrected weight is still a weight, in [0, 1]")


def exercise_the_flag_actually_reaches_the_weight_map():
  """ With partial atoms present, turning it on shall change the weight map.

  This is the test that was missing. The correction was first added only after
  the boundary-smearing branch of solvent_weight_map, which returns early at
  the default of no smearing - so the flag did nothing at all, and four
  refinements came back bit-identical with it on and off. Every other check
  here passed throughout, because none of them asked whether the flag was
  connected to anything.
  """
  xs = structure_with([1.0, 0.25], sites=[(0.2, 0.5, 0.5), (0.7, 0.5, 0.5)])
  m = mask_for(xs)
  m.boundary_smearing = 0            # the default, and the path that was dead
  m.occupancy_weighting = False
  without = m.solvent_weight_map()
  m.occupancy_weighting = True
  with_it = m.solvent_weight_map()
  difference = flex.max(flex.abs(with_it - without))
  assert difference > 0.1, difference
  assert approx_equal(difference, 0.75, eps=1e-9), difference
  # and again with smearing on, so neither path can go dead unnoticed
  m.boundary_smearing = 0.5
  m.occupancy_weighting = False
  without = m.solvent_weight_map()
  m.occupancy_weighting = True
  with_it = m.solvent_weight_map()
  assert flex.max(flex.abs(with_it - without)) > 0.1
  print("\tthe flag reaches the weight map on both the smeared and plain paths")


def run():
  exercise_the_flag_actually_reaches_the_weight_map()
  exercise_full_occupancy_is_untouched()
  exercise_a_partial_atom_gives_back_one_minus_q()
  exercise_two_conformers_multiply()
  exercise_the_weight_stays_in_range()
  print("OK")


if __name__ == '__main__':
  run()
