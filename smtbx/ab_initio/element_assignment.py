""" Which element is each peak? From integrated density, with no model.

A solution gives unlabelled maxima. Calling them all carbon -- which is what
this pipeline did until now -- produces a .res a user must retype by hand, and
is the difference between "a structure appeared" and "a structure I can refine".

**Integrated density, not peak height.** The obvious quantity is the height of
the maximum, and it is the wrong one: height depends on how tight the atom is,
so a heavy atom with a large displacement parameter can peak lower than a light
one that is well ordered. Summing the density in a small sphere is insensitive
to that, because spreading an atom out moves density around within the sphere
rather than out of it.

**The scale comes from the structure itself.** Integrated densities are on an
arbitrary scale -- they depend on the data, the map, and the resolution -- so
they cannot be turned into atomic numbers without a reference. Carbon supplies
one: in almost any organic structure there are pairs of peaks separated by a
typical C-C distance, and setting their mean to Z = 6 fixes the scale using
only the geometry that is already there. No formula from the user, no
assumption about what is present.

That last point matters given what was measured about user-supplied formulas
elsewhere in this project: every realistic corruption of a formula made
normalisation *worse* than using none at all, so a method that needs no formula
is preferable on evidence, not only on convenience.

**What this deliberately does not do.** It does not try to be clever about
elements that are close in atomic number. C, N and O differ by one electron
each and at ordinary resolution their integrated densities overlap; the honest
output is a best guess plus the information that the guess is marginal, which
`assign` reports as a confidence. Pretending otherwise is how an automatic
assignment quietly produces a wrong formula that a user then refines for an
hour.
"""
from __future__ import absolute_import, division, print_function

from libtbx import group_args

# Integration radius. Large enough to capture an atom's density, small enough
# that neighbours at a bonding distance (>= 1.2 A) do not overlap much.
DEFAULT_RADIUS = 0.7

# A C-C single bond, generously bracketed. Used only to identify which peak
# pairs set the scale, so it does not need to be tight -- it needs to exclude
# non-bonded contacts (>= 2.2 A) and fused-ring short contacts.
CC_MIN, CC_MAX = 1.25, 1.65

# Integrated density is **not** proportional to Z. A heavy atom keeps a larger
# share of its electrons inside the integration sphere, while a light atom's
# valence density spills outside it, so a scale calibrated on carbon over-reads
# everything heavier: sulfur first measured 24.6 against a true Z of 16 and was
# assigned chromium, which is why sulfur and chlorine scored exactly zero.
#
# The relation is not a clean power law either -- the ratio d/Z climbs from
# 0.97 at carbon to about 1.4 at chlorine and then flattens -- so rather than
# fit a functional form, this is the measured curve itself: median density
# (with the carbon scale applied) against true Z, over 3049 atoms of 150
# structures whose elements are known. `calibrate_density_z.py` regenerates it.
#
# Iodine was measured at 488 against a true Z of 53 on five atoms and is
# deliberately excluded: five atoms in a handful of structures is not a
# calibration, and one absurd anchor would distort every heavy assignment.
# Above bromine the curve is extrapolated, and `assign` marks those calls
# marginal for that reason.
DENSITY_CURVE = (
  (6, 5.83), (7, 7.69), (8, 9.02), (9, 10.00),
  (15, 20.30), (16, 22.23), (17, 24.29), (35, 47.28),
)


def expected_density(z):
  """ Density (in carbon-scaled units) an atom of atomic number z should show.

  Linear interpolation between measured anchors, linear extrapolation on the
  slope of the last segment beyond them.
  """
  pts = DENSITY_CURVE
  if z <= pts[0][0]:
    z0, d0 = pts[0]
    z1, d1 = pts[1]
    return d0 + (z - z0)*(d1 - d0)/(z1 - z0)
  for (z0, d0), (z1, d1) in zip(pts, pts[1:]):
    if z0 <= z <= z1:
      return d0 + (z - z0)*(d1 - d0)/(z1 - z0)
  (z0, d0), (z1, d1) = pts[-2], pts[-1]
  return d1 + (z - z1)*(d1 - d0)/(z1 - z0)


# Elements this will consider, with atomic numbers. Restricted to what turns up
# in small-molecule work: offering the whole periodic table would let noise pick
# an implausible heavy element on a single bad peak.
COMMON_ELEMENTS = (
  ("C", 6), ("N", 7), ("O", 8), ("F", 9), ("Na", 11), ("Mg", 12), ("Al", 13),
  ("Si", 14), ("P", 15), ("S", 16), ("Cl", 17), ("K", 19), ("Ca", 20),
  ("Cr", 24), ("Mn", 25), ("Fe", 26), ("Co", 27), ("Ni", 28), ("Cu", 29),
  ("Zn", 30), ("As", 33), ("Se", 34), ("Br", 35), ("Ru", 44), ("Pd", 46),
  ("Ag", 47), ("Cd", 48), ("Sn", 50), ("Sb", 51), ("Te", 52), ("I", 53),
  ("Pt", 78), ("Au", 79), ("Hg", 80), ("Pb", 82),
)


def integrated_densities(fft_map, sites_frac, radius=DEFAULT_RADIUS):
  """ Density summed in a sphere of `radius` about each site. """
  from cctbx import maptbx

  return maptbx.average_densities(
    unit_cell=fft_map.unit_cell(),
    data=fft_map.real_map_unpadded(),
    sites_frac=sites_frac,
    radius=float(radius))


def carbon_scale(unit_cell, sites_frac, densities, space_group=None):
  """ Density per electron, from peaks a C-C distance apart. Or None.

  Returns None rather than guessing when no bonded pair is found -- an
  inorganic or heavily-substituted structure may genuinely have none, and a
  fabricated scale would mislabel every atom in it.

  **Distances are minimum-image, and symmetry images count.** `unit_cell.distance`
  measures between the fractional coordinates exactly as written, so a genuine
  C-C bond whose two atoms sit either side of a cell boundary reads as most of
  a cell and is missed, while two atoms with nothing between them can read as
  1.4 A apart and be counted as a bonded pair. Both directions corrupt the
  scale, and the scale divides the density of *every* atom in the structure --
  so a calibration set contaminated this way walks the whole model up or down
  the periodic table.

  `space_group` is optional only so existing callers keep working; pass it
  whenever it is known, because a bonded neighbour is often a symmetry image
  rather than another atom of the asymmetric unit.
  """
  from cctbx.array_family import flex

  n = sites_frac.size()
  if n < 2:
    return None
  ops = list(space_group) if space_group is not None else [None]
  paired = flex.double()
  for i in range(n):
    for j in range(i + 1, n):
      best = None
      for op in ops:
        other = sites_frac[j] if op is None else op*sites_frac[j]
        diff = [other[k] - sites_frac[i][k] for k in range(3)]
        diff = [x - round(x) for x in diff]
        d = unit_cell.length(diff)
        if best is None or d < best:
          best = d
      if best is not None and CC_MIN <= best <= CC_MAX:
        paired.append(densities[i])
        paired.append(densities[j])
  if paired.size() < 4:
    return None
  mean = flex.mean(paired)
  if mean <= 0:
    return None
  return mean/6.0


def assign(unit_cell, sites_frac, densities, elements=None,
           marginal_fraction=0.25, scale=None, space_group=None):
  """ An element per site, with a flag where the call is close.

  `elements` restricts the candidates -- pass the user's expected element list
  when there is one, since knowing that only C, N, O and S are possible removes
  most of the ambiguity for free. With no list, COMMON_ELEMENTS is used.

  `scale` overrides the density-per-electron that would otherwise be derived
  from these same sites. Pass one whenever the site list contains peaks you do
  not yet trust. The scale is a *calibration*, so every site's element depends
  on it: `carbon_scale` averages the peaks a C-C distance apart, and a site
  sitting on weak residual density inside that window pulls the mean down,
  which raises `density/scale` for every atom in the structure and walks the
  whole model up the periodic table. Sites added from a difference map are
  exactly that case -- weak by construction, and placed at a bonding distance
  -- so they must be typed against the scale set by the atoms already trusted,
  not against one they helped compute.

  Each assignment carries `marginal`: true when the second-best element is
  within `marginal_fraction` of an electron of the best. Those are the
  assignments a user should look at, and C/N/O will populate them heavily,
  which is the honest outcome rather than a defect.
  """
  if scale is None:
    scale = carbon_scale(unit_cell, sites_frac, densities, space_group)
  table = [(s, z) for s, z in COMMON_ELEMENTS
           if elements is None or s in elements]
  if not table:
    table = list(COMMON_ELEMENTS)

  out = []
  for i in range(sites_frac.size()):
    if scale is None or scale <= 0:
      # No scale means no basis for a claim. Carbon is the convention for an
      # unassigned peak, and it is reported as such rather than as a result.
      out.append(group_args(element="C", z_estimate=None, marginal=True,
                            reason="no C-C pair to set the scale"))
      continue
    # Compare in *density* space, not in Z space. The candidates' expected
    # densities come from the measured curve, so the non-linearity is handled
    # where it belongs rather than by fudging the scale.
    observed = densities[i]/scale
    ranked = sorted(table, key=lambda sz: abs(expected_density(sz[1])
                                              - observed))
    best, second = ranked[0], (ranked[1] if len(ranked) > 1 else ranked[0])
    d_best = abs(expected_density(best[1]) - observed)
    d_second = abs(expected_density(second[1]) - observed)
    # Marginal when the runner-up is nearly as good, judged relative to the
    # gap between the two candidates rather than as an absolute -- one electron
    # of difference means something quite different at carbon and at bromine.
    span = max(1e-6, abs(expected_density(second[1])
                         - expected_density(best[1])))
    marginal = (d_second - d_best) < marginal_fraction*span
    # Beyond the calibrated range the curve is extrapolated, so say so.
    extrapolated = best[1] > DENSITY_CURVE[-1][0]
    out.append(group_args(
      element=best[0], z_estimate=observed,
      marginal=bool(marginal or extrapolated),
      reason=("density %.1f, nearest %s (expects %.1f)%s%s"
              % (observed, best[0], expected_density(best[1]),
                 "" if not marginal else ", %s nearly as close" % second[0],
                 "" if not extrapolated else ", beyond the calibrated range"))))
  return group_args(assignments=out, scale=scale)


def assign_from_solution(f_calc, max_peaks=None, elements=None,
                         radius=DEFAULT_RADIUS):
  """ Peaks and their elements straight from a solved f_calc. """
  from cctbx import maptbx

  fft_map = f_calc.fft_map(symmetry_flags=maptbx.use_space_group_symmetry)
  fft_map.apply_volume_scaling()
  expected = 1.3*f_calc.unit_cell().volume()/18.6/len(f_calc.space_group())
  peaks = fft_map.peak_search(
    parameters=maptbx.peak_search_parameters(
      min_distance_sym_equiv=1.0,
      max_clusters=int(max_peaks or expected)),
    verify_symmetry=False).all()
  sites = peaks.sites()
  if sites.size() == 0:
    return group_args(sites=sites, assignments=[], scale=None)
  densities = integrated_densities(fft_map, sites, radius=radius)
  result = assign(f_calc.unit_cell(), sites, densities, elements=elements)
  return group_args(sites=sites, assignments=result.assignments,
                    scale=result.scale, densities=densities)
