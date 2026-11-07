""" Which element is each peak? From integrated density, with no model.

Integrated density rather than peak height: height depends on how tight the
atom is, while spreading an atom out moves density within a small sphere rather
than out of it.

Integrated densities are on an arbitrary scale. Carbon supplies the reference:
peak pairs a typical C-C distance apart are set to a mean Z of 6, using only
the geometry already there and no user formula.

C, N and O overlap at ordinary resolution; `assign` gives a best guess and
flags it as marginal rather than pretending to certainty.
"""
from __future__ import absolute_import, division, print_function

from libtbx import group_args

# captures an atom's density with little overlap from neighbours >= 1.2 A away
DEFAULT_RADIUS = 0.7

# a C-C single bond, loosely bracketed: it only has to exclude non-bonded and
# fused-ring short contacts
CC_MIN, CC_MAX = 1.25, 1.65
# P-O, S-O, Si-O and M=O also fall in that window and would inflate the scale;
# a density ratio beyond this (C-F reads 1.7, Si-O 2.1) is not two light atoms
CC_MAX_RATIO = 1.9

# Integrated density is not proportional to Z: a heavy atom keeps more of its
# electrons inside the sphere, so a carbon scale over-reads everything heavier,
# and d/Z rises to about 1.4 at chlorine then flattens. Hence the measured
# curve itself, median carbon-scaled density against true Z over structures of
# known composition. Iodine is left out (too few atoms to calibrate); above
# bromine the curve is extrapolated and `assign` marks those calls marginal.
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


# small-molecule elements only: the whole periodic table would let noise pick
# an implausible heavy element on a single bad peak
COMMON_ELEMENTS = (
  ("C", 6), ("N", 7), ("O", 8), ("F", 9), ("Na", 11), ("Mg", 12), ("Al", 13),
  ("Si", 14), ("P", 15), ("S", 16), ("Cl", 17), ("K", 19), ("Ca", 20),
  ("Cr", 24), ("Mn", 25), ("Fe", 26), ("Co", 27), ("Ni", 28), ("Cu", 29),
  ("Zn", 30), ("As", 33), ("Se", 34), ("Br", 35), ("Ru", 44), ("Pd", 46),
  ("Ag", 47), ("Cd", 48), ("Sn", 50), ("Sb", 51), ("Te", 52), ("I", 53),
  ("Pt", 78), ("Au", 79), ("Hg", 80), ("Pb", 82),
)


def integrated_densities(fft_map, sites_frac, radius=DEFAULT_RADIUS):
  """ Density summed in a sphere of `radius` (one, or one per site) about
  each site. """
  from cctbx import maptbx
  from cctbx.array_family import flex

  data = fft_map.real_map_unpadded()
  if hasattr(radius, '__len__'):
    return [maptbx.average_densities(
      unit_cell=fft_map.unit_cell(), data=data,
      sites_frac=flex.vec3_double([s]), radius=float(r))[0]
      for s, r in zip(sites_frac, radius)]
  return maptbx.average_densities(
    unit_cell=fft_map.unit_cell(),
    data=data,
    sites_frac=sites_frac,
    radius=float(radius))


def site_radii(xs, lo=DEFAULT_RADIUS, hi=1.2, light_z=10):
  """ A sphere radius per site for an Fo/Fc electron-count read: half the
  distance to the nearest modelled neighbour, within [lo, hi], for sites
  heavier than `light_z`; `lo` for the rest.

  A wrongly typed heavy site refines to the U that mimics the true atom, so
  inside a small sphere the densities agree and the deficit only shows where
  the model's broader atom trails off. Light sites keep `lo`: a neighbour at
  1.2-1.5 A not yet modelled would bleed into a wider sphere.
  """
  from cctbx import crystal
  from cctbx.eltbx import tiny_pse

  n = xs.scatterers().size()
  d = [2*hi]*n
  am = xs.asu_mappings(buffer_thickness=2*hi)
  for p in crystal.neighbors_fast_pair_generator(am, distance_cutoff=2*hi):
    d[p.i_seq] = min(d[p.i_seq], p.dist_sq**0.5)
    d[p.j_seq] = min(d[p.j_seq], p.dist_sq**0.5)
  out = []
  for s, di in zip(xs.scatterers(), d):
    try:
      z = tiny_pse.table(s.scattering_type.strip().capitalize()).atomic_number()
    except (RuntimeError, ValueError):
      z = 0
    out.append(min(hi, max(lo, 0.5*di)) if z > light_z else lo)
  return out


def carbon_scale(unit_cell, sites_frac, densities, space_group=None):
  """ Density per electron, from peaks a C-C distance apart. Or None.

  None when no bonded pair is found, since a fabricated scale would mislabel
  every atom. Distances are minimum-image over the symmetry operators of
  `space_group`; pass it whenever known, as a bonded neighbour is often a
  symmetry image.
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
      lo, hi = sorted((densities[i], densities[j]))
      if best is not None and CC_MIN <= best <= CC_MAX and hi <= CC_MAX_RATIO*lo:
        paired.append(densities[i])
        paired.append(densities[j])
  if paired.size() < 4:
    return None
  mean = flex.mean(paired)
  if mean <= 0:
    return None
  return mean/6.0


def formula_scale(densities, formula_zs, trust=0.75):
  """ Density per electron from the declared formula, by rank. Or None.

  For a crystal without a C-C pair. Peaks sorted by density are set against
  the formula's atoms sorted by Z at the same quantile, so only the shape of
  the composition matters, not the counts typed; the weakest quarter of the
  peaks (the noise) is left out, and the median ratio is robust to one
  missing heavy atom.
  """
  n, m = densities.size(), len(formula_zs)
  if n < 2 or m < 1:
    return None
  d = sorted(densities, reverse=True)
  z = sorted(formula_zs, reverse=True)
  r = sorted(d[i]/expected_density(z[int((i + 0.5)*m/n)])
             for i in range(max(2, int(trust*n))))
  r = r[len(r)//2]
  return r if r > 0 else None

def assign(unit_cell, sites_frac, densities, elements=None,
           marginal_fraction=0.25, scale=None, space_group=None,
           formula_zs=None):
  """ An element per site, with a flag where the call is close.

  `elements` restricts the candidates (default COMMON_ELEMENTS).

  `scale` overrides the density per electron derived from these sites; pass
  one whenever the list holds peaks not yet trusted. A weak site inside the
  C-C window pulls the carbon scale down and walks every atom up the periodic
  table, so difference-map sites must be typed against the scale of the
  trusted atoms. `formula_zs` (one Z per declared non-H atom) is the fallback
  without a C-C pair, and the only path when the formula has no carbon.

  Each assignment carries `marginal`: the runner-up is within
  `marginal_fraction` of the gap between the two candidates.
  """
  scale_from = "given"
  # A formula without carbon has no C-C pair; whatever falls in the window
  # (Si-O 1.6, B-B 1.7) would set a scale off by a factor and every atom with it.
  if scale is None and not (formula_zs and 6 not in formula_zs):
    scale = carbon_scale(unit_cell, sites_frac, densities, space_group)
    scale_from = "C-C pairs"
  if scale is None and formula_zs:
    scale = formula_scale(densities, formula_zs)
    scale_from = "formula ranks"
  # the user's list is the table, whatever is in it: Ge, Ti or Ga typed in
  # the formula can not come out as Zn because a table left them out
  if elements:
    from cctbx.eltbx import tiny_pse
    table = []
    for e in elements:
      try:
        table.append((e, tiny_pse.table(e).atomic_number()))
      except (RuntimeError, ValueError):
        pass
  if not elements or not table:
    table = list(COMMON_ELEMENTS)

  out = []
  for i in range(sites_frac.size()):
    if scale is None or scale <= 0:
      # no scale, no claim: carbon by convention, flagged
      out.append(group_args(element="C", z_estimate=None, marginal=True,
                            reason="no C-C pair to set the scale"))
      continue
    # compared in density space, where the curve carries the non-linearity
    observed = densities[i]/scale
    ranked = sorted(table, key=lambda sz: abs(expected_density(sz[1])
                                              - observed))
    best, second = ranked[0], (ranked[1] if len(ranked) > 1 else ranked[0])
    d_best = abs(expected_density(best[1]) - observed)
    d_second = abs(expected_density(second[1]) - observed)
    # relative to the gap between the two candidates: one electron means
    # something different at carbon and at bromine
    span = max(1e-6, abs(expected_density(second[1])
                         - expected_density(best[1])))
    marginal = (d_second - d_best) < marginal_fraction*span
    extrapolated = best[1] > DENSITY_CURVE[-1][0]
    out.append(group_args(
      element=best[0], z_estimate=observed,
      marginal=bool(marginal or extrapolated),
      reason=("density %.1f, nearest %s (expects %.1f)%s%s"
              % (observed, best[0], expected_density(best[1]),
                 "" if not marginal else ", %s nearly as close" % second[0],
                 "" if not extrapolated else ", beyond the calibrated range"))))
  return group_args(assignments=out, scale=scale, scale_from=scale_from)


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
