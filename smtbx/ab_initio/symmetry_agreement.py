""" How well does a P1 solution obey a candidate symmetry operator?

This is our analogue of SHELXT's alpha, and the reason the existing `phi`
feature is inert.

A charge-flipping solution in P1 has an **arbitrary origin**. Applying a
candidate symmetry operator to that map and comparing it with the original
therefore compares two maps that are offset by an unknown vector, and the
comparison measures the offset rather than the symmetry. Measured over 140
structures, phase agreement as currently computed was the one feature out of
five that bought nothing at all in cross-validation -- while alpha is the main
ranking signal in the program we are losing to.

**So search the origin instead of guessing it.** For an operator (R, t):

    rho_op(x) = rho(R x + t)

and the agreement is the *maximum* correlation between rho and rho_op over all
relative shifts. That maximum is a cross-correlation, so one FFT pair gives the
value at every shift at once:

    CC(s) = IFFT( conj(FFT(rho')) * FFT(rho_op') )

with both maps mean-subtracted, normalised to a Pearson coefficient. No phase
convention, no origin bookkeeping, no analytic special cases for the shifts a
particular group allows -- the two earlier attempts at this failed on exactly
that bookkeeping.

The score returned is the mean over the group's operators of that per-operator
best correlation. A true operator scores high because the density really does
repeat under it; a false one scores low however the origin is chosen.

**Graded, never a test.** This returns a number in [0, 1] to be weighted
against the other evidence, not a threshold that removes a candidate. A wrong
answer here should be outvotable -- see the standing rule on hard tests, and
note that the last phase-based *filter* refuted the true group in 6,367 of
10,235 COD structures.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

# Nearest-neighbour resampling on a grid this size is accurate enough for a
# correlation and keeps one structure's whole candidate list inside a second.
# The agreement is a broad function of shift, not a sharp one.
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
  # Resample to a cube so every candidate is compared on identical footing and
  # the FFT sizes stay predictable.
  out = np.zeros((grid, grid, grid))
  n = rho.shape
  i = (np.arange(grid)*n[0]//grid) % n[0]
  j = (np.arange(grid)*n[1]//grid) % n[1]
  k = (np.arange(grid)*n[2]//grid) % n[2]
  out = rho[np.ix_(i, j, k)]
  return out


def _apply_operator(rho, rotation, translation):
  """ rho_op(x) = rho(R x + t), on the grid, by index lookup.

  Fractional coordinates map to grid indices by multiplying by the grid size,
  so the operator acts on indices directly and wrapping is just a modulo -- the
  map is periodic, which is the whole point.
  """
  n = rho.shape[0]
  r = np.array(rotation, float).reshape(3, 3)
  t = np.array(translation, float)

  # **Written out component by component on purpose.** The obvious form,
  # `r.dot(frac)` on a (3, 3) x (3, n**3) pair, hard-crashes this cctbx-linked
  # interpreter -- process gone, no traceback, exit 127 -- while matrix-vector
  # `dot` in the same environment is fine. Rather than depend on which BLAS
  # path a given shape takes, do the nine multiplies explicitly: it is the same
  # arithmetic, it is not measurably slower at this size, and it cannot take
  # the process down.
  ii, jj, kk = np.indices((n, n, n))
  fi, fj, fk = ii/float(n), jj/float(n), kk/float(n)
  gi = np.rint((r[0, 0]*fi + r[0, 1]*fj + r[0, 2]*fk + t[0])*n).astype(np.int64) % n
  gj = np.rint((r[1, 0]*fi + r[1, 1]*fj + r[1, 2]*fk + t[1])*n).astype(np.int64) % n
  gk = np.rint((r[2, 0]*fi + r[2, 1]*fj + r[2, 2]*fk + t[2])*n).astype(np.int64) % n
  return rho[gi, gj, gk]


def _null_space_integers(matrix, limit=3):
  """ Primitive integer vectors u with `matrix^T u == 0`.

  Found by brute force over a small integer box. Symmetry operators have
  entries in {-1, 0, 1} and the null spaces are spanned by short lattice
  vectors, so this is exact where it matters and cheap; solving it in floating
  point and rounding is what makes this kind of code fragile.
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

  **This is the difference between measuring symmetry and measuring nothing.**
  Changing the origin by x0 sends an operator's translation to

      t  ->  t - (I - R) x0

  so the translations reachable from t are exactly `t + range(I - R)`, and a
  component of t outside that range is *intrinsic*: the screw part of a screw
  axis, the glide part of a glide plane. Maximising the correlation over all
  shifts, as the first version of this did, quietly grants every operator an
  arbitrary translation -- which turns every screw axis into a rotation and
  every glide plane into a mirror. Measured on `AHA`, `P 1 21 1` and `P 1 2 1`
  both scored exactly 1.0000, indistinguishable.

  That distinction is not a detail here: **15 of 26 losses against SHELXT were
  glide-versus-mirror**. So the search is restricted to `range(I - R)`.

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

  The mean alone is **not scale-free across candidates**, and measurement shows
  it: on 105 measured structures the mean put the truth in the top three 0.705
  of the time -- close to the module's 0.743 -- but first only 0.410. It favours
  groups with many easy operators. A centring translation is satisfied almost
  perfectly whenever the lattice really is centred, so an I- or F-group averages
  up on operators that discriminate nothing.

  So three reductions, to be judged against each other rather than assumed:

    mean      every operator counts equally -- the original
    min       the weakest operator. A true group must satisfy *all* of its
              symmetry, so the worst one is the honest summary and a supergroup
              that adds one false operator is caught by it
    rotation  mean over operators that actually rotate, dropping pure lattice
              translations, which are a property of the cell rather than of
              the candidate's point symmetry

None of these removes a candidate; all three are numbers to be weighted.
  """
  scores = operator_scores(f_calc_p1, space_group, grid=grid, rho=rho)
  if not scores:
    return dict(mean=0.0, min=0.0, rotation=0.0, n_ops=0)
  values = [s for _, s in scores]
  rotational = [s for name, s in scores
                if not name.replace(" ", "").startswith(("x,y,z",))]
  # A pure lattice translation has the identity rotation part; identify it from
  # the operator string rather than re-deriving it, since `operator_scores`
  # already dropped the identity itself.
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

  The mean rather than the minimum: a single operator that the solution happens
  to satisfy poorly should cost the candidate proportionately, not eliminate it.
  A group whose operators are *all* satisfied is the one we want first.
  """
  scores = operator_scores(f_calc_p1, space_group, grid=grid, rho=rho)
  if not scores:
    return 0.0
  return float(np.mean([s for _, s in scores]))


def agreement_for_candidates(f_calc_p1, space_groups, grid=DEFAULT_GRID):
  """ {str(space group): agreement}, sharing one map across all candidates.

  The map is the expensive part and it does not depend on the candidate, so
  building it once per structure rather than once per candidate is the
  difference between this being usable in the pipeline and not.
  """
  rho = _map_array(f_calc_p1, grid=grid)
  return dict((str(sg), agreement(None, sg, grid=grid, rho=rho))
              for sg in space_groups)
