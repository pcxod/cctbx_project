"""Boundary smearing of the solvent mask: off is exactly off, the weight stays
in [0, 1] and changes only the boundary shell, and it reaches f_mask.
"""
from __future__ import absolute_import, division, print_function


from cctbx import crystal, uctbx, xray
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from smtbx import masks


def structure_and_data(d_min=1.1):
  """A model with an empty region, and data with solvent in it. Fo from the
  model alone would make f_mask zero and every assertion vacuous.
  """
  cs = crystal.symmetry(unit_cell=uctbx.unit_cell((22, 22, 22, 90, 90, 90)),
                        space_group_symbol="P1")
  def sphere(centre, n, element, radius=0.10):
    out = flex.xray_scatterer()
    for i in range(n):
      # deterministic offsets
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
  # an all-zero f_mask would make every comparison below trivially true
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

  # the weight: a 0/1 step at zero smearing, in [0, 1] otherwise
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

  # it must reach f_mask, but only move it as much as a boundary shell can
  assert not a.data().all_eq(b.data()), "smearing did not reach f_mask"
  rel = flex.sum(flex.abs(b.data() - a.data()))/flex.sum(flex.abs(a.data()))
  print("\tf_mask moved by R = %.4f against the hard-edged mask" % rel)
  assert rel < 0.5, "a boundary shell should not rewrite the whole f_mask"

  # high-angle content is not asserted to fall: this void holds O atoms, whose
  # density is genuinely sharp
  hi_sel = a.d_star_sq().data() > 1/(1.5**2)
  print("\tmean |f_mask| beyond 1.5 A: %.4f -> %.4f (informational)"
        % (flex.mean(flex.abs(a.data().select(hi_sel))),
           flex.mean(flex.abs(b.data().select(hi_sel)))))

  print("OK")


if __name__ == '__main__':
  run()
