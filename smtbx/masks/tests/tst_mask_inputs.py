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
"""

from cctbx import crystal, xray
from cctbx.array_family import flex
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


def run():
  exercise_ignore_hydrogen_atoms()
  exercise_partial_h_gives_nothing_back_when_ignored()
  exercise_f_calc_handed_in()
  print("OK")


if __name__ == '__main__':
  run()
