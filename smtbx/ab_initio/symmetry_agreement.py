""" How well does a P1 solution obey a candidate symmetry operator?

An analogue of SHELXT's alpha. A P1 charge-flipping solution has an arbitrary
origin, so comparing the map with its image under a candidate operator measures
the origin offset unless the origin is searched. For an operator (R, t),

    rho_op(x) = rho(R x + t)

and the agreement is the maximum Pearson correlation between rho and rho_op
over the allowed relative shifts, all of which one FFT pair gives at once:

    CC(s) = IFFT( conj(FFT(rho')) * FFT(rho_op') )

with both maps mean-subtracted. The score is the mean of that per-operator best
correlation over the group's operators: a number in [0, 1] to be weighted
against the other evidence, never a test that removes a candidate.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

# nearest-neighbour resampling on this grid suffices: the agreement is a broad
# function of shift
DEFAULT_GRID = 48


def _map_array(f_calc, grid=DEFAULT_GRID):
  """ The P1 density as a plain (n, n, n) numpy array. """
  from cctbx import maptbx

  fft_map = f_calc.fft_map(
    grid_step=None,
    symmetry_flags=maptbx.use_space_group_symmetry,
    d_min=max(0.8, f_calc.d_min()))
  fft_map.apply_volume_scaling()
  rho = fft_map.real_map_unpadded().as_numpy_array()
  # resampled to a cube, so every candidate is compared on the same grid
  n = rho.shape
  i = (np.arange(grid)*n[0]//grid) % n[0]
  j = (np.arange(grid)*n[1]//grid) % n[1]
  k = (np.arange(grid)*n[2]//grid) % n[2]
  out = rho[np.ix_(i, j, k)]
  return out


def _apply_operator(rho, rotation, translation):
  """ rho_op(x) = rho(R x + t), on the grid, by index lookup.

  The operator acts on grid indices directly; the map is periodic, so wrapping
  is a modulo.
  """
  n = rho.shape[0]
  r = np.array(rotation, float).reshape(3, 3)
  t = np.array(translation, float)

  # component by component on purpose: `r.dot(frac)` on a (3, 3) x (3, n**3)
  # pair hard-crashes the cctbx-linked interpreter (exit 127, no traceback)
  ii, jj, kk = np.indices((n, n, n))
  fi, fj, fk = ii/float(n), jj/float(n), kk/float(n)
  gi = np.rint((r[0, 0]*fi + r[0, 1]*fj + r[0, 2]*fk + t[0])*n).astype(np.int64) % n
  gj = np.rint((r[1, 0]*fi + r[1, 1]*fj + r[1, 2]*fk + t[1])*n).astype(np.int64) % n
  gk = np.rint((r[2, 0]*fi + r[2, 1]*fj + r[2, 2]*fk + t[2])*n).astype(np.int64) % n
  return rho[gi, gj, gk]


def _null_space_integers(matrix, limit=3):
  """ Primitive integer vectors u with `matrix^T u == 0`.

  By brute force over a small integer box: operator entries are in {-1, 0, 1}
  and the null spaces are spanned by short lattice vectors, so this is exact,
  where a floating-point solve and rounding would not be.
  """
  mt = np.array(matrix, float).T
  out = []
  rng = range(-limit, limit + 1)
  for a in rng:
    for b in rng:
      for c in rng:
        if (a, b, c) == (0, 0, 0):
          continue
        u = np.array([a, b, c], float)
        if np.abs(mt.dot(u)).max() > 1e-9:
          continue
        g = _gcd3(abs(a), abs(b), abs(c))
        p = (a//g, b//g, c//g)
        if p not in out and tuple(-x for x in p) not in out:
          out.append(p)
  return out


def _gcd3(a, b, c):
  from math import gcd
  return max(1, gcd(gcd(int(a), int(b)), int(c)))


def _allowed_shift_mask(rotation, n):
  """ Grid shifts that correspond to a genuine change of origin.

  Changing the origin by x0 sends an operator's translation to

      t  ->  t - (I - R) x0

  so the reachable translations are `t + range(I - R)`; a component outside
  that range is intrinsic (screw or glide part). Searching all shifts would
  turn every screw axis into a rotation and every glide into a mirror, so the
  search is restricted to `range(I - R)`.

  A shift s = k/n is in `range(I - R) + Z^3` exactly when `u . s` is an integer
  for every primitive integer u spanning the null space of `(I - R)^T`, which on
  the grid is the integer condition `(u . k) % n == 0` -- no tolerance needed.
  """
  m = np.eye(3) - np.array(rotation, float).reshape(3, 3)
  nulls = _null_space_integers(m)
  ki, kj, kk = np.indices((n, n, n))
  mask = np.ones((n, n, n), dtype=bool)
  for (a, b, c) in nulls:
    mask &= ((a*ki + b*kj + c*kk) % n) == 0
  return mask


def _best_correlation(a, b, mask=None):
  """ Highest Pearson correlation between `a` and `b` over allowed shifts. """
  a = a - a.mean()
  b = b - b.mean()
  denominator = np.sqrt((a*a).sum()*(b*b).sum())
  if denominator <= 0:
    return 0.0
  cc = np.fft.ifftn(np.conj(np.fft.fftn(a))*np.fft.fftn(b)).real
  if mask is not None:
    if not mask.any():
      return 0.0
    cc = np.where(mask, cc, -np.inf)
  return float(cc.max()/denominator)


def operator_scores(f_calc_p1, space_group, grid=DEFAULT_GRID, rho=None):
  """ [(operator string, best correlation)] for every non-identity operator. """
  if rho is None:
    rho = _map_array(f_calc_p1, grid=grid)
  out = []
  for op in space_group:
    r = op.r().as_double()
    t = op.t().as_double()
    if op.r().is_unit_mx() and max(abs(x) for x in t) < 1e-9:
      continue
    mask = _allowed_shift_mask(r, rho.shape[0])
    out.append((str(op),
                _best_correlation(rho, _apply_operator(rho, r, t), mask=mask)))
  return out


def agreement_variants(f_calc_p1, space_group, grid=DEFAULT_GRID, rho=None):
  """ Several ways of reducing per-operator scores to one number.

  The mean is not scale-free across candidates: centring translations are
  satisfied almost perfectly in a centred lattice, so it favours groups with
  many easy operators. `min` is the weakest operator (a supergroup adding one
  false operator is caught by it); `rotation` is the mean over operators that
  rotate, dropping pure lattice translations.
  """
  scores = operator_scores(f_calc_p1, space_group, grid=grid, rho=rho)
  if not scores:
    return dict(mean=0.0, min=0.0, rotation=0.0, n_ops=0)
  values = [s for _, s in scores]
  rotational = []
  for op, value in _rotational_flags(space_group, scores):
    if op:
      rotational.append(value)
  return dict(mean=float(np.mean(values)), min=float(np.min(values)),
              rotation=float(np.mean(rotational)) if rotational else 0.0,
              n_ops=len(values))


def _rotational_flags(space_group, scores):
  """ [(is a genuine rotation, score)] aligned with `operator_scores`. """
  out, i = [], 0
  for op in space_group:
    t = op.t().as_double()
    if op.r().is_unit_mx() and max(abs(x) for x in t) < 1e-9:
      continue
    if i < len(scores):
      out.append((not op.r().is_unit_mx(), scores[i][1]))
    i += 1
  return out


def agreement(f_calc_p1, space_group, grid=DEFAULT_GRID, rho=None):
  """ Mean best-correlation over the group's operators. 0.0 if there are none.

  The mean rather than the minimum, so one poorly satisfied operator costs the
  candidate proportionately instead of eliminating it.
  """
  scores = operator_scores(f_calc_p1, space_group, grid=grid, rho=rho)
  if not scores:
    return 0.0
  return float(np.mean([s for _, s in scores]))


def agreement_for_candidates(f_calc_p1, space_groups, grid=DEFAULT_GRID):
  """ {str(space group): agreement}, sharing one map across all candidates.

  The map is the expensive part and does not depend on the candidate. """
  rho = _map_array(f_calc_p1, grid=grid)
  return dict((str(sg), agreement(None, sg, grid=grid, rho=rho))
              for sg in space_groups)
