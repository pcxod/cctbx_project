from __future__ import absolute_import, division, print_function

""" What the caller hands the mask: hydrogen atoms and F_calc.

  * ignore_hydrogen_atoms used to be accepted and ignored. With it set, the
    region must be exactly the region of the structure with its H deleted,
    and a partially occupied H must give nothing back through the occupancy
    correction either.
  * structure_factors(f_calc=...) is how a caller with aspherical form factors
    (Olex2 with a NoSpherA2 table) gets its own F_calc into the mask. Handed
    the very F_calc the mask would compute, it must reproduce the internal
    path bit for bit, in whatever order the reflections come.
  * x_h_length draws the region with every H at one distance from its parent
    (or one per parent element, as Olex2's neutron.dist gives them), the
    parent found through symmetry and within its own PART, so that IAM and
    aspherical models of one structure get the same region.
"""

from cctbx import crystal, xray
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
import smtbx.masks


def structure(atoms):
  """ A P1 cell holding (type, site, occupancy) atoms. """
  symmetry = crystal.symmetry(unit_cell=(16, 16, 16, 90, 90, 90),
                              space_group_symbol="P 1")
  xs = xray.structure(crystal_symmetry=symmetry)
  for i, (t, site, q) in enumerate(atoms):
    xs.add_scatterer(xray.scatterer(label="%s%d" % (t, i), scattering_type=t,
                                    site=site, u=0.02, occupancy=q))
  return xs


def methyl(q_h=1.0):
  c = (0.3, 0.5, 0.5)
  return [("C", c, 1.0), ("C", (0.39, 0.5, 0.5), 1.0),
          ("H", (0.3, 0.565, 0.5), q_h), ("H", (0.3, 0.47, 0.556), q_h),
          ("H", (0.3, 0.47, 0.444), q_h)]


def mask_for(xs, fo_sq=None, **kwds):
  if fo_sq is None:
    fo_sq = xs.structure_factors(d_min=1.5).f_calc().norm()
    fo_sq.set_observation_type_xray_intensity()
  m = smtbx.masks.mask(xs, fo_sq)
  m.compute(solvent_radius=1.2, shrink_truncation_radius=1.2, **kwds)
  return m


def exercise_ignore_hydrogen_atoms():
  xs = structure(methyl())
  heavy = xs.select(~xs.hd_selection())
  fo_sq = xs.structure_factors(d_min=1.5).f_calc().norm()
  fo_sq.set_observation_type_xray_intensity()
  with_h = mask_for(xs, fo_sq, ignore_hydrogen_atoms=False)
  without_h = mask_for(xs, fo_sq, ignore_hydrogen_atoms=True)
  deleted = mask_for(heavy, fo_sq, ignore_hydrogen_atoms=False)
  assert without_h.mask.data.all_eq(deleted.mask.data)
  assert without_h.solvent_accessible_volume > with_h.solvent_accessible_volume, (
    without_h.solvent_accessible_volume, with_h.solvent_accessible_volume)
  print("\tignore_hydrogen_atoms draws the region of the H-deleted structure"
        " (%.1f against %.1f A^3 with H)" % (
          without_h.solvent_accessible_volume, with_h.solvent_accessible_volume))


def exercise_partial_h_gives_nothing_back_when_ignored():
  xs = structure(methyl(q_h=0.5))
  m = mask_for(xs, ignore_hydrogen_atoms=True)
  m.occupancy_weighting = True
  assert flex.max(flex.abs(m.occupancy_weight_map())) == 0
  # and with H in, the same partial H does give some back
  m = mask_for(xs, ignore_hydrogen_atoms=False)
  m.occupancy_weighting = True
  assert flex.max(m.occupancy_weight_map()) > 0
  print("\ta partial H takes no part in the occupancy correction once ignored")


def exercise_f_calc_handed_in():
  model = structure([("C", (0.25, 0.5, 0.5), 1.0), ("C", (0.34, 0.5, 0.5), 1.0),
                     ("O", (0.3, 0.6, 0.5), 1.0)])
  # the data hold solvent the model does not: two O in the void
  solvent = structure([("O", (0.7, 0.3, 0.3), 1.0), ("O", (0.75, 0.7, 0.7), 1.0)])
  data = model.concatenate(solvent)
  fo_sq = data.structure_factors(d_min=1.5).f_calc().norm()
  fo_sq.set_observation_type_xray_intensity()

  inside = mask_for(model, fo_sq)
  f_inside = inside.structure_factors()
  assert inside.n_voids() > 0 and inside.f_000_s > 0

  given = mask_for(model, fo_sq)
  f_calc = fo_sq.set().structure_factors_from_scatterers(
    model, algorithm="direct").f_calc()
  f_given = given.structure_factors(f_calc=f_calc)
  assert f_given.indices().all_eq(f_inside.indices())
  assert flex.max(flex.abs(f_given.data() - f_inside.data())) == 0
  assert given.f_000_s == inside.f_000_s

  # the same F_calc in reverse order is put back into the mask's order
  reverse = flex.size_t(range(f_calc.size()-1, -1, -1))
  shuffled = mask_for(model, fo_sq)
  f_shuffled = shuffled.structure_factors(f_calc=f_calc.select(reverse))
  assert flex.max(flex.abs(f_shuffled.data() - f_inside.data())) == 0
  print("\ta handed-in F_calc reproduces the internal path bit for bit"
        " (%.1f electrons), in any order" % inside.f_000_s)


def exercise_common_length_moves_h_only():
  xs = structure(methyl())
  moved, without_parent = smtbx.masks.hydrogens_at_common_length(xs, 1.09)
  assert without_parent == []
  uc = xs.unit_cell()
  parent = xs.scatterers()[0].site
  xp = uc.orthogonalize(parent)
  for before, after in zip(xs.scatterers(), moved.scatterers()):
    if before.scattering_type == "H":
      assert approx_equal(uc.distance(after.site, parent), 1.09, eps=1e-9)
      # along its own bond: the direction from the parent is unchanged
      b = [x - p for x, p in zip(uc.orthogonalize(before.site), xp)]
      a = [x - p for x, p in zip(uc.orthogonalize(after.site), xp)]
      nb = sum(x*x for x in b)**0.5
      assert approx_equal([x/nb for x in b], [x/1.09 for x in a], eps=1e-9)
    else:
      assert before.site == after.site
  print("\tH moved to 1.09 A along their own bonds, heavy atoms untouched")


def exercise_lengths_by_parent_element():
  """ neutron.dist style: C-H 1.09, O-H 0.983, and no entry for N. """
  xs = structure([("C", (0.3, 0.5, 0.5), 1.0), ("H", (0.3, 0.565, 0.5), 1.0),
                  ("O", (0.6, 0.5, 0.5), 1.0), ("H", (0.6, 0.56, 0.5), 1.0),
                  ("N", (0.5, 0.2, 0.5), 1.0), ("H", (0.5, 0.26, 0.5), 1.0)])
  moved, without_parent = smtbx.masks.hydrogens_at_common_length(
    xs, {'c': 1.09, 'O': 0.983})
  uc = xs.unit_cell()
  s, m = xs.scatterers(), moved.scatterers()
  assert approx_equal(uc.distance(m[1].site, s[0].site), 1.09, eps=1e-9)
  assert approx_equal(uc.distance(m[3].site, s[2].site), 0.983, eps=1e-9)
  assert m[5].site == s[5].site and without_parent == ["H5"]
  print("\tlengths by parent element; an element without one keeps its H")


def exercise_parent_across_symmetry():
  """ The H's only parent is the inversion mate of the C at x = -0.05. """
  symmetry = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90),
                              space_group_symbol="P -1")
  xs = xray.structure(crystal_symmetry=symmetry)
  xs.add_scatterer(xray.scatterer("C1", site=(0.05, 0.5, 0.5), u=0.02))
  xs.add_scatterer(xray.scatterer("H1", site=(0.85, 0.5, 0.5), u=0.02))
  moved, without_parent = smtbx.masks.hydrogens_at_common_length(xs, 1.09)
  assert without_parent == []
  assert approx_equal(moved.scatterers()[1].site, (0.95 - 0.109, 0.5, 0.5), eps=1e-9)
  print("\ta parent found only through a symmetry operation")


def exercise_parent_from_own_part():
  """ PART 2 H, nearer to the PART 1 C than to its own PART 2 C. """
  symmetry = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90),
                              space_group_symbol="P 1")
  xs = xray.structure(crystal_symmetry=symmetry)
  for label, site, part in (("C1", (0.5, 0.5, 0.5), 1), ("C2", (0.59, 0.5, 0.5), 2),
                            ("H2", (0.5, 0.585, 0.5), 2)):
    sc = xray.scatterer(label, site=site, u=0.02, occupancy=0.5)
    sc.set_part(part)
    xs.add_scatterer(sc)
  uc = xs.unit_cell()
  sites = [sc.site for sc in xs.scatterers()]
  assert uc.distance(sites[2], sites[0]) < uc.distance(sites[2], sites[1]) < 1.3
  moved, without_parent = smtbx.masks.hydrogens_at_common_length(xs, 1.09)
  assert without_parent == []
  assert approx_equal(uc.distance(moved.scatterers()[2].site, sites[1]), 1.09, eps=1e-9)
  print("\tan alternate conformer's H keeps its own conformer's parent")


def exercise_region_drawn_from_common_length():
  xs = structure(methyl())
  fo_sq = xs.structure_factors(d_min=1.5).f_calc().norm()
  fo_sq.set_observation_type_xray_intensity()
  moved, _ = smtbx.masks.hydrogens_at_common_length(xs, 1.09)
  option = mask_for(xs, fo_sq, x_h_length=1.09)
  by_hand = mask_for(moved, fo_sq)
  own = mask_for(xs, fo_sq)
  assert option.mask.data.all_eq(by_hand.mask.data)
  assert not option.mask.data.all_eq(own.mask.data)
  assert option.solvent_accessible_volume < own.solvent_accessible_volume
  # and the occupancy correction is drawn around the same sites
  option.occupancy_weighting = by_hand.occupancy_weighting = True
  assert option.solvent_weight_map().all_eq(by_hand.solvent_weight_map())
  print("\tx_h_length=1.09 draws the region of the H-moved structure"
        " (%.1f against %.1f A^3 with the model's own H)" % (
          option.solvent_accessible_volume, own.solvent_accessible_volume))


def run():
  exercise_ignore_hydrogen_atoms()
  exercise_partial_h_gives_nothing_back_when_ignored()
  exercise_f_calc_handed_in()
  exercise_common_length_moves_h_only()
  exercise_lengths_by_parent_element()
  exercise_parent_across_symmetry()
  exercise_parent_from_own_part()
  exercise_region_drawn_from_common_length()
  print("OK")


if __name__ == '__main__':
  run()
