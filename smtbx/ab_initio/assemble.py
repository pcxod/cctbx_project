""" Move peaks together into connected molecules, as `compaq -a` does.

Peaks of one asymmetric unit can have their bonded neighbours in another
symmetry image; anything reading local geometry (the element-assignment
descriptors) then sees an empty environment at every fragment boundary.

Shortest-distance assembly: starting from one placed atom, repeatedly move the
unplaced atom with the shortest contact to any placed one (over all symmetry
operations and lattice translations) into that image. Closest contacts go
first, so molecules come out whole without covalent radii or elements, which
are not known yet at this stage.
"""
from __future__ import absolute_import, division, print_function

from libtbx import group_args

# lattice shifts tried around the rounded difference, which is not always the
# nearest image in an oblique cell
_SHIFTS = tuple((i, j, k)
                for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))


def _nearest_image(unit_cell, target_frac, moving_frac, space_group):
  """ (distance, transformed site) for the image of `moving` nearest `target`,
  over all symmetry operations and lattice translations. """
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

  `max_gap` is the longest distance the assembly had to bridge; a large one
  means separate pieces (a salt, a solvate or a bad solution). `max_sites`
  bounds the O(n^2) pair search: only the first (strongest) peaks are used.
  """
  import numpy as np
  from scitbx.array_family import flex

  n = min(sites_frac.size(), max_sites)
  if n == 0:
    return group_args(sites=sites_frac, max_gap=0.0, n_assembled=0)

  placed = [None]*n
  placed[0] = tuple(sites_frac[0])
  remaining = set(range(1, n))
  max_gap = 0.0
  # _nearest_image for the newest placed atom against every unplaced one in
  # one array pass; cm/ci/cs hold, per unplaced j, the nearest placed image,
  # ties to the lowest i, so the pick matches a pairwise _nearest_image loop
  # (op, then shift, then i, then j in set order)
  ops = space_group.all_ops()
  img = np.array([[op * tuple(sites_frac[j]) for op in ops] for j in range(n)])
  sh = np.array(_SHIFTS, float)
  o = unit_cell.orthogonalization_matrix()
  cm, ci, cs = np.full(n, np.inf), np.full(n, n), np.zeros((n, 3))
  p = 0
  while remaining:
    js = np.array(sorted(remaining))
    t = np.array(placed[p])
    cand = (img[js] - np.round(img[js] - t))[:, :, None, :] + sh
    dv = t - cand
    x = o[0]*dv[..., 0] + o[1]*dv[..., 1] + o[2]*dv[..., 2]
    y = o[3]*dv[..., 0] + o[4]*dv[..., 1] + o[5]*dv[..., 2]
    z = o[6]*dv[..., 0] + o[7]*dv[..., 1] + o[8]*dv[..., 2]
    d = np.sqrt(x*x + y*y + z*z).reshape(len(js), -1)
    k = d.argmin(axis=1)
    d = d[np.arange(len(js)), k]
    up = (d < cm[js]) | ((d == cm[js]) & (p < ci[js]))
    cm[js[up]], ci[js[up]] = d[up], p
    cs[js[up]] = cand.reshape(len(js), -1, 3)[np.arange(len(js)), k][up]
    # the scan in set order stops at the first contact below 1.2 A (bonded)
    order = list(remaining)
    m = cm[order]
    s = np.flatnonzero(m < 1.2)
    j = order[int(m[:s[0] + 1].argmin() if len(s) else m.argmin())]
    placed[j] = tuple(float(v) for v in cs[j])
    remaining.discard(j)
    max_gap = max(max_gap, float(cm[j]))
    p = j

  out = flex.vec3_double(len(placed))
  for i, site in enumerate(placed):
    out[i] = site
  return group_args(sites=out, max_gap=max_gap, n_assembled=n)


def as_xyz(unit_cell, sites_frac, elements, title="assembled"):
  """ XYZ text in Cartesian angstroms, of assembled sites. """
  lines = ["%d" % sites_frac.size(), title]
  for element, site in zip(elements, sites_frac):
    x, y, z = unit_cell.orthogonalize(site)
    lines.append("%-3s %12.6f %12.6f %12.6f" % (element, x, y, z))
  return "\n".join(lines) + "\n"
