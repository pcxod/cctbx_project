""" Move peaks together into connected molecules, as `compaq -a` does.

A solution gives peaks in one asymmetric unit, scattered: an atom near the edge
has its bonded neighbours in a *different* symmetry image, several angstroms
away in Cartesian space even though the two are chemically bonded. That is
harmless for refinement, which knows the symmetry, and fatal for anything that
reads local geometry.

**Why this is a precondition, not tidying.** The element-assignment model
describes each atom by its environment within a 2.5 A cutoff. An atom whose
neighbours sit in another image has an almost empty environment, so its
descriptor is not merely noisy but meaningless -- and the atoms affected are
exactly those at fragment boundaries, which is a systematic subset rather than
a random one. Feeding un-assembled peaks to a geometry-based method would not
make it look bad, it would make it look random, and nothing could be concluded
from the comparison.

**The algorithm** is the shortest-distance-matrix assembly used for this in
crystallography:

  1. compute the shortest distance between every pair of unique atoms, over all
     symmetry operations and lattice translations, remembering which operation
     achieved it
  2. mark one atom as placed
  3. find the shortest distance joining a placed atom to an unplaced one
  4. move the unplaced atom by the operation that brings it closest, mark it
     placed
  5. repeat from 3 until nothing is left

It builds outward from the closest contacts, so covalent bonds are consumed
before non-bonded ones and molecules come out whole. No covalent radii and no
element assignment are needed -- which matters here, because assembly has to
happen *before* the elements are known.
"""
from __future__ import absolute_import, division, print_function

from libtbx import group_args

# Fractional lattice shifts tried when looking for the nearest image. The
# rounded difference is the right answer for an orthogonal cell but not always
# for an oblique one, so the neighbours are checked too.
_SHIFTS = tuple((i, j, k)
                for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))


def _nearest_image(unit_cell, target_frac, moving_frac, space_group):
  """ (distance, transformed site) for the image of `moving` nearest `target`.

  Every symmetry operation is tried, each followed by the lattice translation
  that brings the result closest. Returns the best over all of them.
  """
  best_d, best_site = None, None
  for op in space_group.all_ops():
    site = op * moving_frac
    base = [site[i] - round(site[i] - target_frac[i]) for i in range(3)]
    for shift in _SHIFTS:
      candidate = (base[0] + shift[0], base[1] + shift[1], base[2] + shift[2])
      d = unit_cell.distance(target_frac, candidate)
      if best_d is None or d < best_d:
        best_d, best_site = d, candidate
  return best_d, best_site


def assemble(unit_cell, space_group, sites_frac, max_sites=200):
  """ Sites moved so that the structure is connected. Returns a group_args
  with `sites` (fractional, possibly outside 0..1) and `max_gap`.

  `max_gap` is the longest distance the assembly had to bridge. A small value
  means everything joined through bonding contacts; a large one means the
  structure is genuinely in separate pieces -- a salt, a solvate, or a bad
  solution -- and the caller may want to know rather than be told nothing.

  `max_sites` bounds the O(n^2) pair search. Peak lists longer than this are
  truncated to the strongest, because assembling several hundred noise peaks
  costs more than it informs.
  """
  from scitbx.array_family import flex

  n = min(sites_frac.size(), max_sites)
  if n == 0:
    return group_args(sites=sites_frac, max_gap=0.0, n_assembled=0)

  placed = [None]*n
  placed[0] = tuple(sites_frac[0])
  remaining = set(range(1, n))
  max_gap = 0.0

  while remaining:
    # The shortest link from anything placed to anything not yet placed.
    best = None
    for j in remaining:
      moving = tuple(sites_frac[j])
      for i in range(n):
        if placed[i] is None:
          continue
        d, site = _nearest_image(unit_cell, placed[i], moving, space_group)
        if best is None or d < best[0]:
          best = (d, j, site)
      # Bonded contacts are ~1.2-1.6 A; nothing will beat that, so stop
      # scanning once such a link is found. Without this the search is
      # O(n^2) per placement and O(n^3) overall, which is minutes for a
      # 200-peak list rather than seconds.
      if best is not None and best[0] < 1.2:
        break
    d, j, site = best
    placed[j] = site
    remaining.discard(j)
    max_gap = max(max_gap, d)

  out = flex.vec3_double(len(placed))
  for i, site in enumerate(placed):
    out[i] = site
  return group_args(sites=out, max_gap=max_gap, n_assembled=n)


def as_xyz(unit_cell, sites_frac, elements, title="assembled"):
  """ XYZ text in Cartesian angstroms, which is what geometry tools read.

  Fractional coordinates are meaningless to a method that works on local
  geometry, and the conversion has to happen after assembly -- converting
  first would just give Cartesian coordinates of a scattered structure.
  """
  lines = ["%d" % sites_frac.size(), title]
  for element, site in zip(elements, sites_frac):
    x, y, z = unit_cell.orthogonalize(site)
    lines.append("%-3s %12.6f %12.6f %12.6f" % (element, x, y, z))
  return "\n".join(lines) + "\n"


def assembled_xyz(f_calc, sites_frac, elements, title="assembled"):
  """ Convenience: assemble a peak list and render it as XYZ. """
  result = assemble(f_calc.unit_cell(), f_calc.space_group(), sites_frac)
  elements = list(elements)[:result.sites.size()]
  while len(elements) < result.sites.size():
    elements.append("C")
  return group_args(
    xyz=as_xyz(f_calc.unit_cell(), result.sites, elements, title=title),
    sites=result.sites, max_gap=result.max_gap)
