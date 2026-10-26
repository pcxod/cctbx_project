"""(3+1)D modulated structures: superspace symmetry, satellite structure factors and their refinement.

A first, Python/numpy implementation. The superspace group is built from the basic (average-structure) space
group and the modulation wave vector q, so the average structure from a 3D solution (FLINT, SHELXT, ...) is the
starting model directly; ``superspace_group.from_ops`` takes the operators of an msCIF verbatim instead, and
``change_of_basis`` carries a model between settings. ``smtbx.modulation.mscif`` reads JANA's msCIF models.

Conventions (fractional throughout; reciprocal vectors are row vectors, h' = h R):

  satellite      S = h + m q
  atom           average position xb, modulation argument x4 = q.xb + t, actual position xb + u(x4)
  displacement   u(x4) = sum_n A_n sin(2 pi n x4) + B_n cos(2 pi n x4)          (fractional components)
  occupancy      p(x4) = p0 + sum_n Ps_n sin(2 pi n x4) + Pc_n cos(2 pi n x4)
  ADP            U*(x4) = U*0 + sum_n Us_n sin(2 pi n x4) + Uc_n cos(2 pi n x4)  (U* as cctbx u_star)
  operator       (R, t | eps, M, tau):  x -> R x + t,  x4 -> eps x4 + M.x + tau,  with q R = eps q + M
  F(h, m)        sum_ops sum_atoms w f(|S|) exp(2 pi i (h.t + m tau + h'.xb)) I,   h' = h R + m M, m' = eps m
                 I = integral_0^1 p(x4) exp(-2 pi^2 S' U*(x4) S'^T) exp(2 pi i (S'.u(x4) + m' x4)) dx4,  S' = S R
I is evaluated by uniform quadrature, exponentially convergent for the periodic integrand; f = f0 + f' + i f'';
w = site multiplicity / group order for atoms on special positions (1 otherwise).
"""
from __future__ import absolute_import, division, print_function

import concurrent.futures
import math
import os
from fractions import Fraction

import numpy as np

from cctbx import sgtbx, uctbx
from cctbx.array_family import flex
from cctbx.eltbx import xray_scattering

TWO_PI = 2 * math.pi
TWO_PI_SQ = 2 * math.pi ** 2


class IncompatibleQ(ValueError):
  """q is not mapped onto +-q (modulo the reciprocal lattice) by an operation of the basic space group."""


def read_hklm(path):
  """(h (N,3), m (N,), I, sigma) from an .hklm file: h k l m I sigma [batch], fixed 4I4,2F8 or space separated,
  ended by 0 0 0 0; lines that are neither are headers before the data and skipped (and reported) after it (the
  rules of Olex2's THklFile::LoadHklmFromStrings)."""
  rows, bad = [], []
  with open(path, errors="replace") as f:
    for n, line in enumerate(f, 1):
      line = line.rstrip("\r\n")
      try:
        if len(line) < 32:
          raise ValueError
        v = [int(line[i:i + 4]) for i in (0, 4, 8, 12)] + [float(line[16:24]), float(line[24:32])]
      except ValueError:
        t = line.split()
        if not t:
          continue
        try:
          v = [int(x) for x in t[:4]] + [float(t[4]), float(t[5])]
        except (ValueError, IndexError):
          if rows:   # before the data a header; after it skipped, not the end of the file
            bad.append((n, line))
          continue
      if not any(v[:4]):
        break
      rows.append(v)
  if bad:
    print("hklm: skipped %d unreadable line(s) in %s, the first is line %d: %r" % (len(bad), path, *bad[0]))
  a = np.array(rows, dtype=float).reshape(-1, 6)
  return a[:, :3].astype(int), a[:, 3].astype(int), a[:, 4], a[:, 5]


def _pack(h, m):
  h = np.asarray(h, dtype=np.int64)
  m = np.asarray(m, dtype=np.int64)
  return (((h[:, 0] + 512) * 1024 + (h[:, 1] + 512)) * 1024 + (h[:, 2] + 512)) * 64 + (m + 32)


def _orbit_keys(ssg, h, m):
  """One key per orbit of (h, m) under the superspace point group, (h, m) R = (h R + m M, eps m), and Friedel."""
  keys = None
  for op in ssg.ops:
    hp = np.rint(h.dot(op.r) + np.outer(m, op.m)).astype(np.int64)
    mp = (op.eps * m).astype(np.int64)
    for s in (1, -1):
      k = _pack(s * hp, s * mp)
      keys = k if keys is None else np.minimum(keys, k)
  return keys


def _sys_absent(ssg, h, m):
  out = np.zeros(len(m), dtype=bool)
  for op in ssg.ops:
    fixed = np.all(np.abs(h.dot(op.r) + np.outer(m, op.m) - h) < 1e-6, axis=1) & (op.eps * m == m)
    ph = (h.dot(op.t) + m * op.tau) % 1.0
    out |= fixed & (ph > 1e-6) & (ph < 1 - 1e-6)
  return out


def reflection_statistics(ssg, unit_cell, h, m, i_obs, sigma):
  """Per satellite order |m| (0: the main reflections) and for all satellites together ('sat'): measured and unique
  counts, I/sigma = sum <I> / sum sigma(<I>) over the merged unique set (as the HOS's 1/Rsigma), R_int over the
  orbits measured more than once, d_min, and completeness against every (h, m) of that order the superspace group
  allows within the order's own d_min. Merging uses the superspace point group and Friedel's law."""
  h = np.asarray(h, dtype=float)
  m = np.asarray(m, dtype=int)
  i_obs, sigma = np.asarray(i_obs, dtype=float), np.asarray(sigma, dtype=float)
  g6 = unit_cell.reciprocal_metrical_matrix()
  g = np.array([[g6[0], g6[3], g6[4]], [g6[3], g6[1], g6[5]], [g6[4], g6[5], g6[2]]])
  s_len = lambda hh, mm: np.sqrt(np.einsum("ni,ij,nj->n", hh + np.outer(mm, ssg.q), g, hh + np.outer(mm, ssg.q)))
  abc = np.array(unit_cell.parameters()[:3])

  def one(sel, orders):
    if not sel.any():
      return None
    hh, mm, ii, ss = h[sel], m[sel], i_obs[sel], sigma[sel]
    keep = ~_sys_absent(ssg, hh, mm)
    hh, mm, ii, ss = hh[keep], mm[keep], ii[keep], ss[keep]
    keys = _orbit_keys(ssg, hh, mm)
    uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    # Olex2's RefMerger::ShelxMerger, so the satellite numbers compare with the HOS's main-reflection ones
    ss = np.where(ss != 0, ss, 0.001)
    w = np.where(ii > 3 * ss, ii / ss ** 2, 3 / ss)
    mean = np.bincount(inv, w * ii) / np.bincount(inv, w)
    dev = np.abs(ii - mean[inv])
    sig_m = 1 / np.sqrt(np.bincount(inv, 1 / ss ** 2))
    multi_u = cnt > 1
    sig_int = np.where(multi_u, np.bincount(inv, dev) / (cnt * np.sqrt(np.maximum(cnt - 1, 1))), 0)
    sig_m = np.where(multi_u, np.maximum(sig_m, sig_int), sig_m)
    multi = cnt[inv] > 1
    r_int = (float(np.sum(dev[multi]) / np.sum(ii[multi])) if multi.any() and np.sum(ii[multi]) > 0 else None)
    s_max = float(s_len(hh, mm).max())
    exp_keys = []   # one set over +k and -k: an orbit holds both signs (eps = -1 operations, Friedel)
    for k in orders:
      for mk in ((k, -k) if k else (0,)):
        lim = np.ceil(s_max * abc + abs(mk) * np.abs(ssg.q)).astype(int) + 1
        grid = np.stack(np.meshgrid(*[np.arange(-n, n + 1) for n in lim], indexing="ij"), -1).reshape(-1, 3).astype(float)
        mg = np.full(len(grid), mk)
        inside = (s_len(grid, mg) <= s_max * (1 + 1e-9)) & ~(np.all(grid == 0, axis=1) & (mg == 0))
        grid, mg = grid[inside], mg[inside]
        allowed = ~_sys_absent(ssg, grid, mg)
        exp_keys.append(_orbit_keys(ssg, grid[allowed], mg[allowed]))
    expected = len(np.unique(np.concatenate(exp_keys)))
    return {"measured": int(sel.sum()), "unique": int(len(uniq)), "i_over_sigma": float(mean.sum() / max(sig_m.sum(), 1e-30)),
            "r_int": r_int, "d_min": 1.0 / s_max if s_max > 0 else None,
            "completeness": float(len(uniq)) / expected if expected else None}

  orders = sorted(set(np.abs(m)))
  out = {k: one(np.abs(m) == k, [k]) for k in orders}
  sat = [k for k in orders if k]
  if sat:
    out["sat"] = one(np.abs(m) > 0, sat)
  return out


def write_main_hkl(hklm_path, out_path):
  """The main reflections (m = 0) of an .hklm as an HKLF 4 file, for programs that read three indices."""
  h, m, i_obs, sig = read_hklm(hklm_path)
  f8 = lambda x: ("%8.2f" % x) if len("%8.2f" % x) <= 8 else ("%8.0f" % x)[:8]
  with open(out_path, "w") as f:
    for (a, b, c), ii, ss in zip(h[m == 0], i_obs[m == 0], sig[m == 0]):
      f.write("%4d%4d%4d%s%s\n" % (a, b, c, f8(ii), f8(ss)))
    f.write("   0   0   0    0.00    0.00\n")
  return int(np.sum(m == 0))


# ---------------------------------------------------------------------------
# superspace symmetry
# ---------------------------------------------------------------------------

class superspace_operator(object):
  """(R, t | eps, M, tau); R integer 3x3, t fractional, M integer row vector with q R = eps q + M."""

  def __init__(self, r, t, eps, m, tau=0.0, xyz=""):
    self.r = np.asarray(r, dtype=float)
    self.t = np.asarray(t, dtype=float)
    self.eps = int(eps)
    self.m = np.asarray(m, dtype=float)
    self.tau = float(tau)
    self.xyz = xyz

  def as_4x4(self):
    w = np.zeros((4, 4))
    w[:3, :3] = self.r
    w[3, :3] = self.m
    w[3, 3] = self.eps
    return w

  def as_xyz4(self):
    """x1..x4 symbol, e.g. '-x1,x2+1/2,-x3,-x4+1/2'."""
    rows = []
    w = self.as_4x4()
    tr = list(self.t) + [self.tau]
    for i in range(4):
      terms = []
      for j in range(4):
        c = int(round(w[i, j]))
        if c:
          terms.append(("-" if c < 0 else "+") + ("" if abs(c) == 1 else str(abs(c))) + "x%d" % (j + 1))
      s = "".join(terms).lstrip("+")
      f = _fraction(tr[i])
      rows.append(s + (("+" + f) if f else ""))
    return ",".join(rows)


def _fraction(x):
  x = x % 1.0
  for d in (2, 3, 4, 6, 8, 12):
    n = round(x * d)
    if abs(x * d - n) < 1e-6 and n % d:
      g = math.gcd(int(n), d)
      return "%d/%d" % (n // g, d // g)
  return "" if abs(x) < 1e-6 or abs(x - 1) < 1e-6 else "%.4f" % x


def _op_key(r, t):
  return tuple(int(round(v)) for v in r.ravel()) + tuple(int(round((v % 1.0) * 24)) % 24 for v in t)


def parse_xyz4(s):
  """A superspace operator from its msCIF symbol, e.g. 'x1+1/2,-x2,x3-x4+1/2,-x4'."""
  rows = s.replace(" ", "").lower().split(",")
  if len(rows) != 4:
    raise ValueError("not a (3+1)D operator: %r" % s)
  w, tr = np.zeros((4, 4)), np.zeros(4)
  for i, row in enumerate(rows):
    for term in row.replace("-", "+-").split("+"):
      if not term:
        continue
      if "x" in term:
        coef, var = term.split("x")
        coef = coef.rstrip("*")
        w[i, int(var) - 1] += -1 if coef == "-" else (1 if coef == "" else float(Fraction(coef)))
      else:
        tr[i] += float(Fraction(term))
  if np.any(w[:3, 3]) or abs(abs(w[3, 3]) - 1) > 1e-9:
    raise ValueError("not a (3+1)D operator: %r" % s)
  return superspace_operator(w[:3, :3], tr[:3], w[3, 3], w[3, :3], tr[3] % 1.0, s)


def _rt_mx(r, t):
  tt = np.asarray(t) * 12
  if not np.allclose(tt, np.round(tt), atol=1e-6):
    raise ValueError("translation %s not on the 1/12 grid" % (t,))
  return sgtbx.rt_mx(sgtbx.rot_mx([int(round(v)) for v in np.ravel(r)], 1),
                     sgtbx.tr_vec([int(round(v)) for v in tt], 12))


def eps_and_m(space_group, q, tol=1e-5):
  """[(op, eps, M)] for every operation, or IncompatibleQ naming the first operation q does not fit."""
  q = np.asarray(q, dtype=float)
  out = []
  for op in space_group.all_ops():
    r = np.array(op.r().as_double()).reshape(3, 3)
    qr = q.dot(r)
    for eps in (1, -1):
      m = qr - eps * q
      if np.allclose(m, np.round(m), atol=tol):
        out.append((op, eps, np.round(m)))
        break
    else:
      raise IncompatibleQ("q = %s is not invariant (up to sign) under %s" % (tuple(q), op.as_xyz()))
  return out


def compatible_subgroup(space_group, q, tol=1e-5):
  """The largest subgroup of space_group under which q is invariant up to sign: the basic group a q allows."""
  q = np.asarray(q, dtype=float)
  keep = sgtbx.space_group()
  for op in space_group.all_ops():
    r = np.array(op.r().as_double()).reshape(3, 3)
    qr = q.dot(r)
    if any(np.allclose(qr - e * q, np.round(qr - e * q), atol=tol) for e in (1, -1)):
      keep.expand_smx(op)
  return keep


class superspace_group(object):
  """The superspace group generated by the basic space group's operations with x4 translations tau.

  taus maps generator xyz strings to tau (default 0). Moving the x4 origin by d changes tau by -2d on every
  eps = -1 operation at once, so it can zero tau on one eps = -1 generator only; the others, and every
  eps = +1 generator, carry real information. The x4 parts of all elements are closed from the generators,
  and an inconsistent choice raises ValueError.
  """

  def __init__(self, space_group, q, taus=None, tol=1e-5, _gens=None, _em=None):
    self.basic = space_group
    self.q = np.asarray(q, dtype=float)
    self.taus = dict(taus or {})
    # _gens, _em: what candidate_superspace_groups has worked out once for all its candidates
    em = _em if _em is not None else {op.as_xyz(): (op, e, m) for op, e, m in eps_and_m(space_group, q, tol)}
    self.generators = []
    for g in (_gens if _gens is not None else _generators(space_group)):
      op, e, m = em[g.as_xyz()]
      r = np.array(op.r().as_double()).reshape(3, 3)
      self.generators.append(superspace_operator(r, op.t().as_double(), e, m, self.taus.get(g.as_xyz(), 0.0),
                                                 g.as_xyz()))
    self.ops = _close(self.generators)
    if len(self.ops) != space_group.order_z():
      raise ValueError("closure gave %d operations for a group of order %d"
                       % (len(self.ops), space_group.order_z()))

  @classmethod
  def from_ops(cls, symbols, q, tol=1e-5):
    """The group of a list of msCIF operator symbols (closed under multiplication; centrings included)."""
    self = cls.__new__(cls)
    self.q = np.asarray(q, dtype=float)
    self.generators = [parse_xyz4(s) for s in symbols]
    for op in self.generators:
      if not np.allclose(self.q.dot(op.r) - op.eps * self.q, op.m, atol=1e-3):
        raise IncompatibleQ("q R != eps q + M for %s" % op.xyz)
    self.basic = sgtbx.space_group()
    for op in self.generators:
      self.basic.expand_smx(_rt_mx(op.r, op.t))
    self.ops = _close(self.generators)
    if len(self.ops) != self.basic.order_z():
      raise ValueError("closure gave %d operations for a group of order %d"
                       % (len(self.ops), self.basic.order_z()))
    self.taus = {}
    return self

  def label(self):
    """Basic group symbol, q and the operations with eps = +1 and a non-zero tau (these make satellite absences)."""
    sym = self.basic.info().symbol_and_number()
    q = "(%s)" % " ".join("%.4g" % v for v in self.q)
    taus = " ".join(op.as_xyz4() for op in self.ops if op.eps == 1 and _fraction(op.tau))
    return "%s q=%s%s" % (sym, q, ("  " + taus) if taus else "")

  def symbol_ops(self):
    return [op.as_xyz4() for op in self.ops]

  def is_sys_absent(self, h, m):
    """True when (h, m) is a systematic absence of this superspace group."""
    h = np.asarray(h, dtype=float)
    for op in self.ops:
      hp = h.dot(op.r) + m * op.m
      if np.allclose(hp, h) and op.eps * m == m:
        ph = (h.dot(op.t) + m * op.tau) % 1.0
        if 1e-6 < ph < 1 - 1e-6:
          return True
    return False


def _generators(space_group):
  """A small generating set of space_group's operations (centring translations included)."""
  gens = []
  closure = sgtbx.space_group()
  for op in space_group.all_ops():
    if closure.order_z() == space_group.order_z():
      break
    test = sgtbx.space_group(closure)
    test.expand_smx(op)
    if test.order_z() > closure.order_z():
      gens.append(op)
      closure = test
  return gens


def _close(generators, tol=1e-6):
  ident = superspace_operator(np.eye(3), np.zeros(3), 1, np.zeros(3), 0.0, "x,y,z")
  found = {_op_key(ident.r, ident.t): ident}
  frontier = [ident]
  while frontier:
    nxt = []
    for e in frontier:
      for g in generators:
        r = g.r.dot(e.r)
        t = (g.r.dot(e.t) + g.t) % 1.0
        eps = g.eps * e.eps
        m = g.eps * e.m + g.m.dot(e.r)
        tau = (g.eps * e.tau + g.tau + g.m.dot(e.t)) % 1.0
        k = _op_key(r, t)
        if k in found:
          d = (found[k].tau - tau) % 1.0
          if tol < d < 1 - tol:
            raise ValueError("inconsistent x4 translations: tau %.4f and %.4f" % (found[k].tau, tau))
          continue
        op = superspace_operator(r, t, eps, m, tau, "")
        found[k] = op
        nxt.append(op)
    frontier = nxt
  return list(found.values())


def candidate_superspace_groups(space_group, q, tau_values=(0.0, 0.5)):
  """Every consistent assignment of tau in tau_values to the generators, as superspace_group objects.

  The first eps = -1 generator keeps tau = 0 (it fixes the x4 origin); all other generators are free.
  shortcut: tau_values (0, 1/2) misses the 1/3, 1/4, 1/6 of 3-, 4- and 6-fold axes with eps = +1; pass them.
  """
  gens = _generators(space_group)
  em3 = {op.as_xyz(): (op, e, m) for op, e, m in eps_and_m(space_group, q)}
  em = {k: v[1] for k, v in em3.items()}
  first_inv = next((g.as_xyz() for g in gens if em[g.as_xyz()] == -1), None)
  free = [g.as_xyz() for g in gens if g.as_xyz() != first_inv]
  out = []
  for k in range(len(tau_values) ** len(free)):
    taus, kk = {}, k
    for f in free:
      taus[f] = tau_values[kk % len(tau_values)]
      kk //= len(tau_values)
    try:
      out.append(superspace_group(space_group, q, taus, _gens=gens, _em=em3))
    except ValueError:
      continue
  return out


def rank_by_absences(groups, h, m, i_obs, sig, n_sigma=3.0):
  """[(violations, n_absent_predicted, group)] best first: violations = predicted absences with I > n_sigma sigma.

  Ties go to more predicted absences (the more specific group), then to fewer non-zero tau.
  """
  h = np.asarray(h, dtype=float).reshape(-1, 3)
  m = np.asarray(m, dtype=float).ravel()
  strong = np.asarray(i_obs, dtype=float) > n_sigma * np.asarray(sig, dtype=float)
  fixed = {}     # point part (R, M, eps) -> which (h, m) it leaves alone: shared by every candidate
  ranked = []
  for g in groups:
    absent = _absent_mask(g, h, m, fixed)
    ranked.append((int(np.sum(absent & strong)), -int(np.sum(absent)), sum(1 for op in g.generators if op.tau), g))
  ranked.sort(key=lambda x: x[:3])
  return [(v, -n, g) for v, n, _, g in ranked]


def _absent_mask(g, h, m, fixed=None):
  """superspace_group.is_sys_absent for every (h, m) at once, in the same arithmetic: (h, m) fixed by an operation's
  point part, (h R + m M, eps m) = (h, m), with h.t + m tau not an integer. fixed caches the first test by point
  part across groups that differ in t and tau only."""
  out = np.zeros(len(m), dtype=bool)
  for op in g.ops:
    key = (op.r.tobytes(), op.m.tobytes(), op.eps)
    f = None if fixed is None else fixed.get(key)
    if f is None:
      hp = h.dot(op.r) + np.outer(m, op.m)
      # np.allclose(hp, h) and eps m == m, row by row
      f = np.all(np.abs(hp - h) <= 1e-8 + 1e-5 * np.abs(h), axis=1) & (op.eps * m == m)
      if fixed is not None:
        fixed[key] = f
    if not f.any():
      continue
    hs, ms = h[f], m[f]
    ph = ((hs[:, 0] * op.t[0] + hs[:, 1] * op.t[1]) + hs[:, 2] * op.t[2] + ms * op.tau) % 1.0
    hit = (ph > 1e-6) & (ph < 1 - 1e-6)
    out[np.flatnonzero(f)[hit]] = True
  return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

XHARM_MODE = "sin_first"


def xharm_basis(order, x):
  """JANA's x-harmonic of the given order (1, 2, ...) at the crenel-local coordinate x = (x4 - c) / w in
  [-1/2, 1/2]: sin(2 pi k x), cos(2 pi k x) alternating, k = (order + 1) // 2 (XHARM_MODE 'sin_first')."""
  k = (order + 1) // 2
  first = np.sin if XHARM_MODE == "sin_first" else np.cos
  second = np.cos if XHARM_MODE == "sin_first" else np.sin
  return (first if order % 2 else second)(TWO_PI * k * x)


def _waves(v, n, width):
  return np.zeros((n, width)) if v is None else np.asarray(v, dtype=float).reshape(-1, width)


class modulated_atom(object):
  """An atom with displacive (a, b: sin and cos, fractional), occupational (occ_sin, occ_cos) and ADP
  (u_sin, u_cos: u_star units) Fourier waves; u_star None means isotropic u_iso.

  crenel (c, w): the atom exists for x4 in [c - w/2, c + w/2] only, with occupancy / w there (the listed
  occupancy is the average); legendre (L, 3): displacements sum_n legendre[n-1] P_n(xi), xi = 2 (x4 - c) / w
  in [-1, 1], inside the crenel (JANA's choice there); sawtooth (a, c, w): u = a 2 (x4 - c) / w.
  """

  def __init__(self, label, scattering_type, site, u_iso, occupancy=1.0, n_harmonics=1, a=None, b=None,
               u_star=None, occ_sin=None, occ_cos=None, u_sin=None, u_cos=None, site_weight=1.0,
               crenel=None, legendre=None, sawtooth=None, xharm=None, u_xharm=None):
    self.crenel = None if crenel is None else (float(crenel[0]), float(crenel[1]))
    self.legendre = _waves(legendre, 0, 3)
    self.xharm = _waves(xharm, 0, 3)
    self.u_xharm = _waves(u_xharm, 0, 6)
    self.sawtooth = None if sawtooth is None else (np.asarray(sawtooth[0], dtype=float), float(sawtooth[1]),
                                                   float(sawtooth[2]))
    self.label = label
    self.scattering_type = scattering_type
    self.site = np.asarray(site, dtype=float)
    self.u_iso = float(u_iso)
    self.u_star = None if u_star is None else np.asarray(u_star, dtype=float).reshape(6)
    self.occupancy = float(occupancy)
    self.a = _waves(a, n_harmonics, 3)
    self.b = _waves(b, n_harmonics, 3)
    self.occ_sin = _waves(occ_sin, 0, 1).ravel()
    self.occ_cos = _waves(occ_cos, len(self.occ_sin), 1).ravel()
    self.u_sin = _waves(u_sin, 0, 6)
    self.u_cos = _waves(u_cos, len(self.u_sin), 6)
    self.site_weight = float(site_weight)

  @property
  def n_harmonics(self):
    return self.a.shape[0]

  def displacement(self, x4):
    """u(x4) for an array of x4, shape (len(x4), 3), fractional."""
    x4 = np.atleast_1d(np.asarray(x4, dtype=float))
    u = np.zeros((x4.size, 3))
    for n in range(self.n_harmonics):
      arg = TWO_PI * (n + 1) * x4
      u += np.outer(np.sin(arg), self.a[n]) + np.outer(np.cos(arg), self.b[n])
    if len(self.legendre):
      from numpy.polynomial import legendre as L
      c, w = self.crenel
      xi = 2 * ((x4 - c + 0.5) % 1.0 - 0.5) / w
      for n in range(len(self.legendre)):
        u += np.outer(L.legval(xi, [0] * (n + 1) + [1]), self.legendre[n])
    if self.sawtooth is not None:
      a, c, w = self.sawtooth
      u += np.outer(2 * ((x4 - c + 0.5) % 1.0 - 0.5) / w, a)
    for n in range(len(self.xharm)):
      u += np.outer(xharm_basis(n + 1, self._local(x4)), self.xharm[n])
    return u

  def _local(self, x4):
    c, w = self.crenel
    return ((x4 - c + 0.5) % 1.0 - 0.5) / w

  def u_xharm_at(self, x4):
    """ADP x-harmonic sum at x4, (len(x4), 6) u_star, or None."""
    if not len(self.u_xharm):
      return None
    return sum(np.outer(xharm_basis(n + 1, self._local(x4)), self.u_xharm[n]) for n in range(len(self.u_xharm)))

  def nodes(self, n_x4):
    """x4 quadrature nodes and weights of this atom's x4 integral: uniform over [0, 1), or Gauss-Legendre
    inside the crenel (the integrand is smooth there and zero outside)."""
    if self.crenel is None:
      return (np.arange(n_x4) + 0.5) / n_x4, np.full(n_x4, 1.0 / n_x4)
    c, w = self.crenel
    xi, wt = np.polynomial.legendre.leggauss(n_x4)
    return c + 0.5 * w * xi, 0.5 * w * wt

  def occupancy_at(self, x4):
    p = np.full(len(x4), self.occupancy / (self.crenel[1] if self.crenel else 1.0))
    for n in range(len(self.occ_sin)):
      p += self.occ_sin[n] * np.sin(TWO_PI * (n + 1) * x4) + self.occ_cos[n] * np.cos(TWO_PI * (n + 1) * x4)
    return p


class modulated_structure(object):
  """dispersion: {scattering type: (f', f'')}."""

  def __init__(self, unit_cell, ssg, atoms, scale=1.0, dispersion=None):
    self.unit_cell = unit_cell
    self.ssg = ssg
    self.atoms = list(atoms)
    self.scale = float(scale)
    self.dispersion = dict(dispersion or {})

  @property
  def q(self):
    return self.ssg.q

  def positions_at(self, t, amplitude_scale=1.0):
    """Actual fractional positions of the asymmetric-unit atoms in the section t."""
    return np.array([at.site + amplitude_scale * at.displacement(self.q.dot(at.site) + t)[0]
                     for at in self.atoms])


# ---------------------------------------------------------------------------
# structure factors and gradients
# ---------------------------------------------------------------------------

def _form_factors(types, stol, radiation="xray"):
  """f0(sin(theta)/lambda) per scattering type: X-rays from the WK1995 gaussians (IT1992 fallback), electrons from
  ITC Vol. C 2011 Table 4.3.2.2, cctbx's "electron" table (in A, as Olex2's ED refinement uses it)."""
  from cctbx.eltbx.e_scattering import ito_vol_c_2011_table_4_3_2_2_entry_as_gaussian as electron
  out = {}
  st = flex.double(np.ascontiguousarray(stol, dtype=float).ravel())
  for t in set(types):
    g = None
    for label in (t, t.rstrip("+-0123456789")):
      if radiation == "electron":
        try:
          g = electron(label, exact=False)
        except Exception:
          g = None
      else:
        for table in (xray_scattering.wk1995, xray_scattering.it1992):
          try:
            g = table(label, True).fetch()
            break
          except Exception:
            continue
      if g is not None:
        break
    if g is None:
      raise ValueError("no form factor for %r" % t)
    out[t] = g.at_stol(st).as_numpy_array().reshape(np.shape(stol))
  return out


TABLE_SCALE = 100000   # NoSpherA2 -hkl_scale: table rows are round(TABLE_SCALE * S), S = h + m q
RADIATIONS = ("xray", "electron")   # kinematical in both; f' and f'' apply to X-rays only


_ID_DECODE = 16.0 / 2147483647   # scatterer_id_big: fractional coordinate = int32 * 16 / INT32_MAX


def _id_from_hex(token):
  """(x, y, z, data, Z) of a scatterer id printed as two uint64, (x | y << 32) and (z | data << 32 | Z << 48)."""
  a, b = int(token[:16], 16), int(token[16:32], 16)
  s32 = lambda v: (v & 0xffffffff) - ((v & 0x80000000) << 1)
  data = (b >> 32) & 0xffff
  return (s32(a) * _ID_DECODE, s32(a >> 32) * _ID_DECODE, s32(b) * _ID_DECODE, data - ((data & 0x8000) << 1),
          (b >> 48) & 0xff)


def read_tsc(path):
  """(scatterers, indices (N, 3) int, f (N, n) complex, AD flag) of a NoSpherA2 text .tsc or binary .tscb.
  scatterers: (n, 5) array of scatterer ids (x, y, z, data = PART, Z), or a list of labels for a label table."""
  raw = open(path, "rb").read()
  if raw.startswith((b"TITLE:", b"SCATTERERS:", b"SCATTERER_IDS:")):
    head, _, body = raw.decode().partition("DATA:")
    scat, ad = [], False
    for line in head.splitlines():
      if line.startswith("SCATTERER_IDS:"):
        scat = np.array([_id_from_hex(t) for t in line.split()[1:]])
      elif line.startswith("SCATTERERS:"):
        scat = line.split()[1:]
      elif line.startswith("AD:"):
        ad = "TRUE" in line.upper()
    v = np.array(body.replace(",", " ").split(), dtype=float).reshape(-1, 3 + 2 * len(scat))
    return scat, np.rint(v[:, :3]).astype(np.int64), v[:, 3::2] + 1j * v[:, 4::2], ad
  i4 = lambda o: int(np.frombuffer(raw, "<i4", 1, o)[0])
  o = 4 + i4(0)
  n = i4(o)
  if b"SCATTERER_IDS" in raw[4:o]:
    rec = np.frombuffer(raw, np.dtype([("x", "<i4", 3), ("data", "<i2"), ("z", "u1"), ("res", "u1")]), n, o + 4)
    scat = np.column_stack([rec["x"] * _ID_DECODE, rec["data"], rec["z"]])
    o += 4 + 16 * n
  else:   # labels: n is the byte size of the space-separated payload
    scat = raw[o + 4:o + 4 + n].decode().split()
    o += 4 + n
  rec = np.frombuffer(raw, np.dtype([("h", "<i4", 3), ("f", "<c16", len(scat))]), i4(o), o + 4)
  return scat, rec["h"].astype(np.int64), rec["f"].copy(), False


class form_factor_table(object):
  """Aspherical f_j(S) from a NoSpherA2 table computed with -hkl_scale `scale` on calculator.table_indices: the
  density moves rigidly with its atom along x4. Columns are matched to atoms by scatterer id. A row missing for S
  is taken from -S as f(S)* (real density), unless the table carries f', f'' (AD: TRUE)."""

  def __init__(self, path, scale=TABLE_SCALE):
    self.path, self.scale = path, scale
    self.ids, idx, self.f, self.ad = read_tsc(path)
    if not isinstance(self.ids, np.ndarray):
      raise ValueError("%s names its scatterers by label; satellite tables are matched by scatterer id" % path)
    self.row = {tuple(k): i for i, k in enumerate(idx.tolist())}

  def columns(self, xs, tolerance=0.5):
    """The column of each atom: the id of the same element nearest the average site modulo lattice translations,
    within `tolerance` A (refinement moves the sites off the ids); the PART of an atom carrying `part` decides
    between ids that are both that close."""
    from cctbx.eltbx import tiny_pse
    o = np.array(xs.unit_cell.orthogonalization_matrix()).reshape(3, 3)
    cols = []
    for at in xs.atoms:
      z = tiny_pse.table(at.scattering_type.rstrip("+-0123456789")).atomic_number()
      d = np.asarray(at.site) - self.ids[:, :3]
      dist = np.where(self.ids[:, 4] == z, np.linalg.norm((d - np.rint(d)).dot(o.T), axis=1), np.inf)
      near = dist <= tolerance
      if not near.any():
        raise ValueError("%s has no scatterer id for %s within %.1f A: recompute it for this model"
                         % (self.path, at.label, tolerance))
      if near.sum() > 1 and getattr(at, "part", None) is not None and (near & (self.ids[:, 3] == at.part)).any():
        dist[self.ids[:, 3] != at.part] = np.inf
      cols.append(int(np.argmin(dist)))
    if len(set(cols)) < len(cols):
      raise ValueError("%s: two atoms take the same scatterer id" % self.path)
    return cols

  def lookup(self, s, cols):
    """(N, len(cols)) complex form factors at the fractional reciprocal vectors s (N, 3), columns from columns()."""
    rows, conj = np.empty(len(s), dtype=int), np.zeros(len(s), dtype=bool)
    for i, k in enumerate(np.rint(self.scale * np.asarray(s)).astype(np.int64).tolist()):
      r = self.row.get(tuple(k))
      if r is None and not self.ad:
        r, conj[i] = self.row.get((-k[0], -k[1], -k[2])), True
      if r is None:
        raise ValueError("%s has no row for S = %s: recompute it for this q and reflection list"
                         % (self.path, np.round(np.array(k) / float(self.scale), 5)))
      rows[i] = r
    f = self.f[np.ix_(rows, cols)]
    f[conj] = np.conj(f[conj])
    return f


def _sum_waves(sin_c, cos_c, sin_t, cos_t):
  """sum_n s_n sin_t[n] + c_n cos_t[n] for coefficient arrays (n, ...) and node tables (n, K) -> (K, ...)."""
  n = len(sin_c)
  return np.tensordot(sin_t[:n], sin_c, axes=(0, 0)) + np.tensordot(cos_t[:n], cos_c, axes=(0, 0))


class calculator(object):
  """|F|^2 and gradients for a reflection list (h: (N,3) int, m: (N,) int) and a modulated_structure.

  Parameter blocks, in order, for each atom (each present when its flag is set): site (3); ADP (u_iso 1 or
  u_star 6); occupancy (1); displacive waves A, B (3 per harmonic each); occupancy waves sin, cos (1 per
  harmonic each); ADP waves sin, cos (6 per harmonic each); crenel (c, w); Legendre displacements (3 per
  order); displacive x-harmonics (3 per order); ADP x-harmonics (6 per order). Then the scale. flags: 'site',
  'adp', 'occupancy', 'modulation', 'occ_waves', 'adp_waves', 'crenel', 'legendre', 'xharm', 'u_xharm',
  'scale'; all True except 'occupancy' and the last four, which are asked for.

  The crenel gradients are those of the x4 quadrature as computed: its nodes sit at fixed local coordinates
  (x4 - c) / w, so c and w move the nodes and scale the weights, and the Legendre and x-harmonic terms, which
  depend on the local coordinate alone, do not change with them.
  table: a form_factor_table replacing the WK1995 f0 by aspherical f_j(S R) for each operation R.
  radiation: "xray" or "electron" (kinematical electron diffraction, ITC electron f0 and no f', f'').
  """

  def __init__(self, h, m, n_x4=32, table=None, threads=None, radiation="xray"):
    assert radiation in RADIATIONS, radiation
    self.radiation = radiation
    self.h = np.asarray(h, dtype=float).reshape(-1, 3)
    self.m = np.asarray(m, dtype=float).ravel()
    self.n_x4 = n_x4
    self.table, self._table_f = table, {}
    self.threads = threads   # None: one per CPU, capped by the atom count and a 2 GB working set
    # what depends on the reflections, q, the operations, the cell and the dispersion only, kept across the
    # compute() calls of a refinement; rebuilt when any of those changes (_invariant_key)
    self._invariants = (None, None)
    self._share_classes = True   # False: every operation evaluates its own x4 sums (the reference path)

  def table_indices(self, ssg):
    """The rows a NoSpherA2 table needs for these reflections (-hkl_scale TABLE_SCALE): round(TABLE_SCALE S R)
    for every S = h + m q and operation R, one of each +- pair. They depend on q and the point group only."""
    s = self.h + np.outer(self.m, ssg.q)
    keys = np.vstack([np.rint(TABLE_SCALE * s.dot(op.r)) for op in ssg.ops]).astype(np.int64)
    keys = keys[np.any(keys != 0, axis=1)]
    first = keys[np.arange(len(keys)), np.argmax(keys != 0, axis=1)]
    return np.unique(keys * np.sign(first)[:, None], axis=0)

  @staticmethod
  def _invariant_key(xs):
    ops = tuple((op.r.tobytes(), op.t.tobytes(), op.m.tobytes(), op.eps, op.tau) for op in xs.ssg.ops)
    return (tuple(xs.q), ops, tuple(xs.unit_cell.parameters()), tuple(sorted(xs.dispersion.items())),
            tuple(sorted(set(a.scattering_type for a in xs.atoms))))

  def _key(self, xs):
    return self._invariant_key(xs) + (self._share_classes, self.radiation)

  def _per_operation(self, xs):
    """s, |s|^2, the form factors and, per operation, the reflection arrays atom_sum reads: unchanged as long as
    _invariant_key is, so a refinement builds them once."""
    key = self._key(xs)
    if self._invariants[0] == key:
      return self._invariants[1]
    q = xs.q
    s_vec = self.h + np.outer(self.m, q)
    g6 = xs.unit_cell.reciprocal_metrical_matrix()
    g = np.array([[g6[0], g6[3], g6[4]], [g6[3], g6[1], g6[5]], [g6[4], g6[5], g6[2]]])
    s_sq = np.einsum("ni,ij,nj->n", s_vec, g, s_vec)
    ff = None
    if self.table is None:
      f0 = _form_factors([a.scattering_type for a in xs.atoms], np.sqrt(s_sq) / 2, self.radiation)
      dispersion = xs.dispersion if self.radiation == "xray" else {}
      ff = {t: f + complex(*dispersion.get(t, (0.0, 0.0))) for t, f in f0.items()}
    ops = []
    for op in xs.ssg.ops:
      sp = s_vec.dot(op.r)
      q6 = np.column_stack([sp[:, 0] ** 2, sp[:, 1] ** 2, sp[:, 2] ** 2,
                            2 * sp[:, 0] * sp[:, 1], 2 * sp[:, 0] * sp[:, 2], 2 * sp[:, 1] * sp[:, 2]])
      # hp, mp, sp, ph0, q6, and the two factors the gradients multiply by: 2 pi i sp and -2 pi^2 q6
      ops.append((self.h.dot(op.r) + np.outer(self.m, op.m), op.eps * self.m, sp, self.h.dot(op.t) + self.m * op.tau,
                  q6, (1j * TWO_PI) * sp, -TWO_PI_SQ * q6))
    # e(x4) = exp(2 pi i (S R.u(x4) + eps m x4)) depends on an operation only through (R, eps, M): operations
    # that differ in t and tau alone (centring, screws, glides) have the same e, and (-R, -eps, -M) the complex
    # conjugate. classes[k] = (the first operation with that e or its conjugate, conjugate or not)
    first, classes = {}, []
    for k, op in enumerate(xs.ssg.ops):
      key_p = (op.r.tobytes(), op.eps, op.m.tobytes())
      key_m = ((-op.r).tobytes(), -op.eps, (-op.m + 0.0).tobytes())
      if not self._share_classes:
        classes.append((k, False))
      elif key_p in first:
        classes.append((first[key_p], False))
      elif key_m in first:
        classes.append((first[key_m], True))
      else:
        first[key_p] = k
        classes.append((k, False))
    value = (s_vec, s_sq, ff, ops, classes)
    self._invariants = (key, value)
    return value

  def compute(self, xs, flags=None, gradients=False):
    flags = _flags(flags)
    q = xs.q
    s_vec, s_sq, ff, ops, classes = self._per_operation(xs)
    tab = None
    if self.table is not None:
      # by the rotations, not op.xyz: every operation but the generators has an empty xyz
      key = (tuple(q), tuple(op.r.tobytes() for op in xs.ssg.ops), tuple(a.label for a in xs.atoms))
      if key not in self._table_f:
        cols = self.table.columns(xs)
        self._table_f = {key: [self.table.lookup(s_vec.dot(op.r), cols) for op in xs.ssg.ops]}
      disp = 0.0 if self.table.ad or self.radiation == "electron" else np.array(
        [complex(*xs.dispersion.get(a.scattering_type, (0.0, 0.0))) for a in xs.atoms])
      tab = [f + disp for f in self._table_f[key]]
    nref = self.h.shape[0]
    f_calc = np.zeros(nref, dtype=complex)
    grads = []
    if gradients:
      for at in xs.atoms:
        grads.append({"site": np.zeros((nref, 3), complex), "adp": np.zeros((nref, 1 if at.u_star is None else 6), complex),
                      "occ": np.zeros(nref, complex),
                      "a": np.zeros((nref, at.n_harmonics, 3), complex), "b": np.zeros((nref, at.n_harmonics, 3), complex),
                      "os": np.zeros((nref, len(at.occ_sin)), complex), "oc": np.zeros((nref, len(at.occ_sin)), complex),
                      "us": np.zeros((nref, len(at.u_sin), 6), complex), "uc": np.zeros((nref, len(at.u_sin), 6), complex),
                      "cren": np.zeros((nref, 2 if at.crenel else 0), complex),
                      "leg": np.zeros((nref, len(at.legendre), 3), complex),
                      "xh": np.zeros((nref, len(at.xharm), 3), complex),
                      "ux": np.zeros((nref, len(at.u_xharm), 6), complex)})
    # per atom: x4 nodes and weights, sin/cos tables, u, p(x4), ADP-wave sum at the nodes
    pre, extra = [], []
    for at in xs.atoms:
      x4, wt = at.nodes(self.n_x4)
      n_max = max(1, at.n_harmonics, len(at.occ_sin), len(at.u_sin))
      sin_t = np.array([np.sin(TWO_PI * (n + 1) * x4) for n in range(n_max)])
      cos_t = np.array([np.cos(TWO_PI * (n + 1) * x4) for n in range(n_max)])
      uw = _sum_waves(at.u_sin, at.u_cos, sin_t, cos_t) if len(at.u_sin) else None
      ux = at.u_xharm_at(x4)
      if ux is not None:
        uw = ux if uw is None else uw + ux
      pre.append((x4, wt, sin_t, cos_t, at.displacement(x4), at.occupancy_at(x4), uw,
                  1.0 / at.crenel[1] if at.crenel else 1.0))
      extra.append(_local_tables(at, x4, sin_t, cos_t) if gradients else None)

    def atom_sum(ia):
      """One atom's share of F over every operation; it touches only its own gradient blocks."""
      at = xs.atoms[ia]
      x4, wt, sin_t, cos_t, disp, p, uw, dp_docc = pre[ia]
      pw = wt * p      # node weights go into the K-vectors, not onto the (N, K) block
      ex = extra[ia]
      out = np.zeros(nref, dtype=complex)
      th, e = np.empty((nref, len(x4))), np.empty((nref, len(x4)), dtype=complex)
      sums = {}

      def x4_sums(k):
        """Every sum over the x4 nodes an operation needs, from its e: the structure factor's and the gradients'."""
        hp, mp, sp, ph0, q6, isp, dw6 = ops[k]
        np.dot(sp, disp.T, out=th)
        np.add(th, np.outer(mp, x4), out=th)       # in place: th and e are the enclosing function's buffers
        np.multiply(th, TWO_PI, out=th)
        np.cos(th, out=e.real)   # 1.5x faster than exp(1j th) on (17754, 32), the same bits
        np.sin(th, out=e.imag)
        if uw is not None:
          np.multiply(e, np.exp(dw6.dot(uw.T)), out=e)
        r = {"pw": e.dot(pw)}
        if gradients:
          r["wt"] = e.dot(wt)
          r["ps"], r["pc"] = e.dot((pw * sin_t).T), e.dot((pw * cos_t).T)          # (N, n_max)
          if len(at.occ_sin):
            r["es"], r["ec"] = e.dot((wt * sin_t).T), e.dot((wt * cos_t).T)
          if len(at.legendre):
            r["pl"] = e.dot((pw * ex["leg"]).T)
          if len(at.xharm):
            r["px"] = e.dot((pw * ex["xh"]).T)
          if len(at.u_xharm):
            r["pu"] = e.dot((pw * ex["ux"]).T)
          if at.crenel:
            # d/dx4 of the integrand at a node, its local coordinate held: the phase through the Fourier and
            # sawtooth displacements and m x4, the Debye-Waller factor through the ADP waves, and p
            rate = (1j * TWO_PI) * (sp.dot(ex["du"].T) + mp[:, None])
            if ex["duw"] is not None:
              rate -= TWO_PI_SQ * q6.dot(ex["duw"].T)
            de = e * rate
            loc, dp = ex["loc"], ex["dp"]
            r["c0"] = de.dot(pw) + e.dot(wt * dp)
            r["c1"] = e.dot(wt * ex["p_waves"] * dp_docc) + de.dot(pw * loc) + e.dot(wt * dp * loc)
        return r

      # per class: the sums of base over its operations with the class's own e (plus) and with its conjugate
      # (minus). Within a class sp, hp and 2 pi i sp change sign with the conjugate, q6 and the ADP factors do not,
      # so every term is (B+ X - B- X*) or (B+ X + B- X*) times the class leader's factor.
      b_plus, b_minus = {}, {}
      for iop, (hp, mp, sp, ph0, q6, isp, dw6) in enumerate(ops):
        k, conj = classes[iop]
        if at.u_star is None:
          t0 = np.exp(-TWO_PI_SQ * at.u_iso * s_sq)
        else:
          t0 = np.exp(dw6.dot(at.u_star))
        f_at = ff[at.scattering_type] if tab is None else tab[iop][:, ia]
        base = at.site_weight * f_at * t0 * np.exp(1j * TWO_PI * (ph0 + hp.dot(at.site)))
        acc = b_minus if conj else b_plus
        acc[k] = base if k not in acc else acc[k] + base
      for k in sorted(set(b_plus) | set(b_minus)):
        if k not in sums:
          sums[k] = x4_sums(k)
        S = sums[k]
        hp, mp, sp, ph0, q6, isp, dw6 = ops[k]
        bp = b_plus.get(k)
        bm = b_minus.get(k)

        def odd(key, cols=None):
          """B+ X - B- X*: the terms whose factor changes sign with the conjugate (sp, hp)."""
          x = S[key] if cols is None else S[key][:, :cols]
          bpv = 0 if bp is None else (bp[:, None] if x.ndim == 2 else bp)
          bmv = 0 if bm is None else (bm[:, None] if x.ndim == 2 else bm)
          return bpv * x - bmv * np.conj(x)

        def even(key, cols=None):
          """B+ X + B- X*: the terms whose factor does not (q6, the ADP factors, occupancies)."""
          x = S[key] if cols is None else S[key][:, :cols]
          bpv = 0 if bp is None else (bp[:, None] if x.ndim == 2 else bp)
          bmv = 0 if bm is None else (bm[:, None] if x.ndim == 2 else bm)
          return bpv * x + bmv * np.conj(x)
        c_even = even("pw")
        out += c_even
        if gradients:
          gr = grads[ia]
          gr["site"] += (1j * TWO_PI) * odd("pw")[:, None] * hp
          if at.u_star is None:
            gr["adp"][:, 0] += c_even * (-TWO_PI_SQ * s_sq)
          else:
            gr["adp"] += c_even[:, None] * dw6
          gr["occ"] += even("wt") * dp_docc
          nh, nu = at.n_harmonics, len(at.u_sin)
          # every harmonic at once
          gr["a"] += odd("ps", nh)[:, :, None] * isp[:, None, :]
          gr["b"] += odd("pc", nh)[:, :, None] * isp[:, None, :]
          if len(at.occ_sin):
            no = len(at.occ_sin)
            gr["os"] += even("es", no)
            gr["oc"] += even("ec", no)
          if nu:
            gr["us"] += even("ps", nu)[:, :, None] * dw6[:, None, :]
            gr["uc"] += even("pc", nu)[:, :, None] * dw6[:, None, :]
          if len(at.legendre):
            gr["leg"] += odd("pl")[:, :, None] * isp[:, None, :]
          if len(at.xharm):
            gr["xh"] += odd("px")[:, :, None] * isp[:, None, :]
          if len(at.u_xharm):
            gr["ux"] += even("pu")[:, :, None] * dw6[:, None, :]
          if at.crenel:
            gr["cren"][:, 0] += even("c0")
            gr["cren"][:, 1] += even("c1")
      return out

    # numpy's ufuncs and dots release the GIL, so atoms run in threads; summed in atom order, so the result does
    # not depend on the thread count. About 40 bytes per (reflection, node) per thread are live at once.
    n_threads = max(1, min(self.threads or os.cpu_count() or 1, len(xs.atoms),
                           int(2e9 // (40 * nref * max(len(p[0]) for p in pre)))))
    if n_threads == 1:
      for ia in range(len(xs.atoms)):
        f_calc += atom_sum(ia)
    else:
      with concurrent.futures.ThreadPoolExecutor(n_threads) as ex:
        for c in ex.map(atom_sum, range(len(xs.atoms))):
          f_calc += c
    f_sq = np.abs(f_calc) ** 2
    if not gradients:
      return f_calc, xs.scale * f_sq, None
    cols = []
    for gr in grads:
      for key, flag in (("site", "site"), ("adp", "adp"), ("occ", "occupancy"), ("a", "modulation"),
                        ("b", "modulation"), ("os", "occ_waves"), ("oc", "occ_waves"), ("us", "adp_waves"),
                        ("uc", "adp_waves"), ("cren", "crenel"), ("leg", "legendre"), ("xh", "xharm"),
                        ("ux", "u_xharm")):
        if flags[flag]:
          cols.append(gr[key].reshape(nref, -1))
    # d|F|^2/dp = 2 Re(F* dF/dp), block by block into one real array: no (N, n_params) complex temporary
    d_fsq = np.empty((nref, sum(c.shape[1] for c in cols) + int(flags["scale"])))
    fr, fi, j = 2 * xs.scale * f_calc.real[:, None], 2 * xs.scale * f_calc.imag[:, None], 0
    for c in cols:
      d_fsq[:, j:j + c.shape[1]] = fr * c.real + fi * c.imag
      j += c.shape[1]
    if flags["scale"]:
      d_fsq[:, j] = f_sq
    return f_calc, xs.scale * f_sq, d_fsq


def _local_tables(at, x4, sin_t, cos_t):
  """Per node, for the gradients: the Legendre polynomials and x-harmonics of the local coordinate; for a
  crenel atom also that coordinate, and the x4 derivatives of the displacement (Fourier and sawtooth parts),
  of the occupancy waves and of the ADP waves, and the occupancy-wave part of p."""
  out = {"leg": None, "xh": None, "ux": None}
  if len(at.legendre):
    from numpy.polynomial import legendre as L
    xi = 2 * at._local(x4)
    out["leg"] = np.array([L.legval(xi, [0] * (n + 1) + [1]) for n in range(len(at.legendre))])
  if len(at.xharm):
    out["xh"] = np.array([xharm_basis(n + 1, at._local(x4)) for n in range(len(at.xharm))])
  if len(at.u_xharm):
    out["ux"] = np.array([xharm_basis(n + 1, at._local(x4)) for n in range(len(at.u_xharm))])
  if at.crenel:
    out["loc"] = at._local(x4)
    du = np.zeros((len(x4), 3))
    for n in range(at.n_harmonics):
      k = TWO_PI * (n + 1)
      du += k * (np.outer(cos_t[n], at.a[n]) - np.outer(sin_t[n], at.b[n]))
    if at.sawtooth is not None:
      a, _, w_s = at.sawtooth
      du += np.outer(np.ones(len(x4)), 2 * a / w_s)
    out["du"] = du
    dp, pwv = np.zeros(len(x4)), np.zeros(len(x4))
    for n in range(len(at.occ_sin)):
      k = TWO_PI * (n + 1)
      dp += k * (at.occ_sin[n] * cos_t[n] - at.occ_cos[n] * sin_t[n])
      pwv += at.occ_sin[n] * sin_t[n] + at.occ_cos[n] * cos_t[n]
    out["dp"], out["p_waves"] = dp, pwv
    duw = None
    for n in range(len(at.u_sin)):
      k = TWO_PI * (n + 1)
      t = k * (np.outer(cos_t[n], at.u_sin[n]) - np.outer(sin_t[n], at.u_cos[n]))
      duw = t if duw is None else duw + t
    out["duw"] = duw
  return out


def _flags(flags):
  out = {"site": True, "adp": True, "occupancy": False, "modulation": True, "occ_waves": True,
         "adp_waves": True, "crenel": False, "legendre": False, "xharm": False, "u_xharm": False,
         "scale": True}
  for k, v in (flags or {}).items():
    out["adp" if k == "u_iso" else k] = v
  return out


def _u6_map(r):
  """6x6 matrix of U* -> R U* R^T on cctbx u_star vectors (11, 22, 33, 12, 13, 23)."""
  idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
  out = np.zeros((6, 6))
  for k, (i, j) in enumerate(idx):
    u = np.zeros((3, 3))
    u[i, j] = u[j, i] = 1.0
    v = r.dot(u).dot(r.T)
    out[:, k] = [v[a, b] for a, b in idx]
  return out


def site_symmetry_ops(xs, at, tol=1e-3):
  """The superspace operations mapping the atom's average position onto itself (modulo the lattice)."""
  out = []
  for op in xs.ssg.ops:
    d = op.r.dot(at.site) + op.t - at.site
    if np.allclose(d, np.round(d), atol=tol):
      out.append(op)
  return out


def _atom_map(at, op, flags):
  """Linear map of the atom's parameter vector onto that of its image under a site-symmetry operation.

  For each harmonic n the image of (A, B) is (R (eps c A + s B), R (-eps s A + c B)) with c, s = cos, sin of
  2 pi n (M.x + tau): the image's u(x4) is R u(eps (x4 - M.x - tau)). Occupancy waves mix the same way with
  R -> 1, ADP waves with R -> the U* map. A crenel's centre takes eps, its width 1; Legendre terms of order n
  eps^n R, x-harmonics eps R (sine) or R (cosine), with the U* map for the ADP ones.
  """
  off, i = {}, 0
  for name, _, get, _ in _blocks(at, flags):
    off[name] = (i, len(get()))
    i += len(get())
  t = np.eye(i)
  r6 = _u6_map(op.r)
  if "site" in off:
    s = off["site"][0]
    t[s:s + 3, s:s + 3] = op.r
  if "adp" in off and at.u_star is not None:
    s = off["adp"][0]
    t[s:s + 6, s:s + 6] = r6
  delta = op.m.dot(at.site) + op.tau

  def pair(na, nb, width, rk):
    if na not in off or nb not in off:
      return
    (sa, la), (sb, _) = off[na], off[nb]
    for n in range(la // width):
      c, s = math.cos(TWO_PI * (n + 1) * delta), math.sin(TWO_PI * (n + 1) * delta)
      ia, ib = slice(sa + n * width, sa + (n + 1) * width), slice(sb + n * width, sb + (n + 1) * width)
      t[ia, ia], t[ia, ib] = op.eps * c * rk, s * rk
      t[ib, ia], t[ib, ib] = -op.eps * s * rk, c * rk
  pair("a", "b", 3, op.r)
  pair("occ_sin", "occ_cos", 1, np.eye(1))
  pair("u_sin", "u_cos", 6, r6)
  # The crenel and the functions on it map like the Fourier waves, u'(x4) = R u(eps (x4 - delta)): the interval
  # [c - w/2, c + w/2] goes to centre eps c + delta, width w, so the local coordinate (x4 - c) / w turns into eps
  # times the image's. A Legendre term of order n takes eps^n (P_n is even or odd), an x-harmonic eps if it is a
  # sine and 1 if a cosine; displacements with R, ADPs with the U* map. delta only decides whether a model is
  # symmetric at all; the free directions come from the linear part.
  if "crenel" in off:
    sb, lb = off["crenel"]
    if lb:
      t[sb, sb], t[sb + 1, sb + 1] = op.eps, 1.0

  def by_order(name, width, rk, sign):
    if name not in off:
      return
    sb, lb = off[name]
    for n in range(lb // width):
      i = slice(sb + n * width, sb + (n + 1) * width)
      t[i, i] = sign(n + 1) * rk
  by_order("legendre", 3, op.r, lambda order: op.eps ** order)
  sine = lambda order: (order % 2 == 1) == (XHARM_MODE == "sin_first")
  by_order("xharm", 3, op.r, lambda order: op.eps if sine(order) else 1.0)
  by_order("u_xharm", 6, r6, lambda order: op.eps if sine(order) else 1.0)
  return t


def _reduction_blocks(xs, flags):
  """The diagonal blocks of reduction(): one per atom, then the scale."""
  blocks = []
  for at in xs.atoms:
    n = sum(len(get()) for _, _, get, _ in _blocks(at, flags))
    rows = [_atom_map(at, op, flags) - np.eye(n) for op in site_symmetry_ops(xs, at)]
    a = np.vstack(rows) if rows else np.zeros((1, n))
    _, s, vt = np.linalg.svd(a)
    rank = int(np.sum(s > 1e-6 * max(1.0, s.max() if s.size else 1.0)))
    blocks.append(vt[rank:].T)
  if flags["scale"]:
    blocks.append(np.eye(1))
  return blocks


def _block_diag(blocks):
  z = np.zeros((sum(b.shape[0] for b in blocks), sum(b.shape[1] for b in blocks)))
  i = j = 0
  for b in blocks:
    z[i:i + b.shape[0], j:j + b.shape[1]] = b
    i, j = i + b.shape[0], j + b.shape[1]
  return z


def _times_blocks(a, blocks):
  """a Z for the block-diagonal Z, block by block: the dense (N, n) x (n, n_free) product with a Z that is mostly
  zeros (the identity for atoms in general positions) was the largest serial cost of an LM cycle."""
  out = np.empty((a.shape[0], sum(b.shape[1] for b in blocks)))
  i = j = 0
  for b in blocks:
    out[:, j:j + b.shape[1]] = a[:, i:i + b.shape[0]].dot(b)
    i, j = i + b.shape[0], j + b.shape[1]
  return out


def reduction(xs, flags=None):
  """Z (n_params, n_free), orthonormal columns: the parameter directions superspace site symmetry allows;
  refine moves p = p0 + Z xi."""
  return _block_diag(_reduction_blocks(xs, _flags(flags)))


def _blocks(at, flags):
  """(name, flag, getter, setter) of an atom's parameter blocks in calculator order."""
  def adp_get():
    return [at.u_iso] if at.u_star is None else list(at.u_star)

  def adp_set(v):
    if at.u_star is None:
      at.u_iso = float(v[0])
    else:
      at.u_star = np.array(v)

  def arr(name):
    return (lambda: list(getattr(at, name).ravel()),
            lambda v, name=name: setattr(at, name, np.array(v).reshape(getattr(at, name).shape)))
  out = [("site", "site", lambda: list(at.site), lambda v: setattr(at, "site", np.array(v))),
         ("adp", "adp", adp_get, adp_set),
         ("occupancy", "occupancy", lambda: [at.occupancy], lambda v: setattr(at, "occupancy", float(v[0])))]
  for flag, name in (("modulation", "a"), ("modulation", "b"), ("occ_waves", "occ_sin"), ("occ_waves", "occ_cos"),
                     ("adp_waves", "u_sin"), ("adp_waves", "u_cos")):
    out.append((name, flag) + arr(name))
  out.append(("crenel", "crenel", lambda: list(at.crenel) if at.crenel else [],
              lambda v: setattr(at, "crenel", (float(v[0]), float(v[1])) if len(v) else at.crenel)))
  for flag, name in (("legendre", "legendre"), ("xharm", "xharm"), ("u_xharm", "u_xharm")):
    out.append((name, flag) + arr(name))
  return [b for b in out if flags[b[1]]]


def get_parameters(xs, flags):
  flags = _flags(flags)
  p = []
  for at in xs.atoms:
    for _, _, get, _ in _blocks(at, flags):
      p.extend(get())
  if flags["scale"]:
    p.append(xs.scale)
  return np.array(p)


def set_parameters(xs, flags, p):
  flags = _flags(flags)
  i = 0
  for at in xs.atoms:
    for _, _, get, put in _blocks(at, flags):
      k = len(get())
      put(p[i:i + k])
      i += k
  if flags["scale"]:
    xs.scale = float(p[i])


# ---------------------------------------------------------------------------
# refinement
# ---------------------------------------------------------------------------

def r_factors(f_obs_sq, f_calc_sq, m):
  """R1 on F for all data and per satellite order |m|."""
  fo = np.sqrt(np.maximum(f_obs_sq, 0))
  fc = np.sqrt(np.maximum(f_calc_sq, 0))
  out = {"all": float(np.sum(np.abs(fo - fc)) / max(np.sum(fo), 1e-30))}
  for k in sorted(set(np.abs(np.asarray(m)).astype(int))):
    sel = np.abs(m) == k
    out["m=%d" % k] = float(np.sum(np.abs(fo[sel] - fc[sel])) / max(np.sum(fo[sel]), 1e-30))
  return out


def seed_amplitudes(xs, size_angstrom=0.01, seed=0):
  """Give every all-zero displacive wave small random amplitudes: at A = B = 0 every satellite F is 0, so the
  F^2 gradient and the normal matrix block of the waves vanish and least squares never leaves zero."""
  rng = np.random.RandomState(seed)
  abc = np.array(xs.unit_cell.parameters()[:3])
  for at in xs.atoms:
    if not at.a.any() and not at.b.any():
      at.a = rng.uniform(-1, 1, at.a.shape) * size_angstrom / abc
      at.b = rng.uniform(-1, 1, at.b.shape) * size_angstrom / abc


class _average_link(object):
  """refine()'s side of an external parametrisation of the average structure (see refine)."""

  def __init__(self, xs, flags, average):
    self.xs, self.flags, self.average = xs, flags, average
    self.avg_idx, self.wave_idx, self.ab = [], [], {}
    i = 0
    for k, at in enumerate(xs.atoms):
      for name, _, get, _ in _blocks(at, flags):
        n = len(get())
        (self.avg_idx if name in ("site", "adp") else self.wave_idx).extend(range(i, i + n))
        if name in ("a", "b"):
          self.ab[(k, name)] = i
        i += n
    self.n_p = i + (1 if flags["scale"] else 0)
    self.na = np.asarray(average.jacobian()).shape[1]
    self.zb = _reduction_blocks(xs, dict(flags, site=False, adp=False, scale=False))
    self.zw = _block_diag(self.zb)
    self.follow = {}                # follower: (parent, cos, sin) of 2 pi n dx4, one row per harmonic
    for k, (kp, dx4) in (getattr(average, "follow", None) or {}).items():
      if flags["modulation"] and xs.atoms[k].n_harmonics == xs.atoms[kp].n_harmonics:
        arg = TWO_PI * np.arange(1, xs.atoms[k].n_harmonics + 1) * dx4
        self.follow[k] = (kp, np.cos(arg)[:, None], np.sin(arg)[:, None])

  def sync(self):
    """u_f(x4) = u_p(x4 - dx4): A_f = c A_p + s B_p, B_f = -s A_p + c B_p."""
    for k, (kp, c, s) in self.follow.items():
      p = self.xs.atoms[kp]
      self.xs.atoms[k].a, self.xs.atoms[k].b = c * p.a + s * p.b, -s * p.a + c * p.b

  def _fold(self, d):
    """The follower's wave gradient goes to its parent's (chain rule of sync); its own columns to zero."""
    for k, (kp, c, s) in self.follow.items():
      nh = len(c)
      sl = lambda kk, name: slice(self.ab[(kk, name)], self.ab[(kk, name)] + 3 * nh)
      da = d[:, sl(k, "a")].reshape(-1, nh, 3)
      db = d[:, sl(k, "b")].reshape(-1, nh, 3)
      d[:, sl(kp, "a")] += (c * da - s * db).reshape(len(d), -1)
      d[:, sl(kp, "b")] += (s * da + c * db).reshape(len(d), -1)
      d[:, sl(k, "a")] = 0
      d[:, sl(k, "b")] = 0
    return d

  def jacobian(self, d):
    """Columns: the average's independent parameters, the free wave directions, the scale."""
    d = self._fold(d)
    cols = [d[:, self.avg_idx].dot(np.asarray(self.average.jacobian())), _times_blocks(d[:, self.wave_idx], self.zb)]
    if self.flags["scale"]:
      cols.append(d[:, -1:])
    return np.hstack(cols)

  def restraints(self, n_free):
    """(design padded to every free parameter, deltas, weights) at the current model, or None."""
    r = self.average.restraints()
    if r is None or not len(r[1]):
      return None
    design = np.asarray(r[0])
    return (np.hstack([design, np.zeros((len(design), n_free - design.shape[1]))]),
            np.asarray(r[1], dtype=float), np.asarray(r[2], dtype=float))

  def apply(self, p0, x):
    step = np.zeros(self.n_p)
    step[self.wave_idx] = self.zw.dot(x[self.na:self.na + self.zw.shape[1]])
    if self.flags["scale"]:
      step[-1] = x[-1]
    set_parameters(self.xs, self.flags, p0 + step)
    self.average.shift(x[:self.na])
    self.sync()

  def undo(self, p0, x):
    self.average.shift(-x[:self.na])
    set_parameters(self.xs, self.flags, p0)


def refine(xs, h, m, f_obs_sq, sigmas, flags=None, cycles=10, n_x4=32, damping=1e-3, log=None, verbose=True,
           table=None, radiation="xray", average=None):
  """Levenberg-Marquardt on F^2 with weights 1/sigma^2. Returns per-cycle dicts (wR2, R1 by |m|, max shift).
  table: a form_factor_table for aspherical form factors (IAM without); radiation as for calculator.

  The x4 origin of a wave is free unless an eps = -1 operation fixes it; the damping keeps that direction
  (a rotation of every (A_n, B_n) pair together) from running away.

  average: the average structure's own parametrisation, so the constraints and restraints of the 3D model hold
  while the waves refine (Olex2 passes the one olex2.refine builds: AFIX, EXYZ, EADP, rigid groups, riding U,
  fixed parameters, special positions). Sites and ADPs then move only through it. It provides
    jacobian()   d(site and ADP of every atom, atom order, U* when anisotropic)/d(its n independent parameters),
                 shape (rows, n), at the current model
    shift(x)     moves its independent parameters by x and writes the resulting sites and ADPs into xs.atoms
    restraints() None or (design (n_r, n), deltas, weights) at the current model, design = d delta/dx
    follow       {atom index: (parent atom index, dx4)}: the atom's waves are the parent's moved along x4 by
                 dx4 = q.(x_atom - x_parent), so both are displaced alike in every section t (riding hydrogens)
  Restraints are weighted by the GooF^2 of the starting model, as in smtbx.refinement.least_squares.
  ponytail: restraints act on the average structure only, not section by section along t.
  """
  flags = _flags(flags)
  link = None
  if average is not None:
    flags.update(site=True, adp=True, occupancy=False)   # occupancies stay with the average's own refinement
    link = _average_link(xs, flags, average)
  else:
    zb = _reduction_blocks(xs, flags)
    z = _block_diag(zb)             # superspace site symmetry: the free directions
  if flags["modulation"]:
    p_before = get_parameters(xs, flags)
    seed_amplitudes(xs)
    p = get_parameters(xs, flags)
    if link is None:
      set_parameters(xs, flags, p_before + z.dot(z.T.dot(p - p_before)))
    else:
      wi = link.wave_idx
      p[wi] = p_before[wi] + link.zw.dot(link.zw.T.dot(p[wi] - p_before[wi]))
      set_parameters(xs, flags, p)
      link.sync()
  f_obs_sq = np.asarray(f_obs_sq, dtype=float)
  calc = calculator(h, m, n_x4, table, radiation=radiation)
  w = 1.0 / np.maximum(np.asarray(sigmas, dtype=float), 1e-12) ** 2
  sw = np.sqrt(w)
  history = []
  lam = damping
  chi = None    # from the first gradient pass: the same F^2 a separate pass before the loop would give
  norm = None
  for cycle in range(cycles):
    _, fc_sq, d = calc.compute(xs, flags, gradients=True)
    if chi is None:
      chi = float(np.sum(w * (f_obs_sq - fc_sq) ** 2))   # + the weighted restraints once there are any
    r = sw * (f_obs_sq - fc_sq)
    j = _times_blocks(sw[:, None] * d, zb) if link is None else link.jacobian(sw[:, None] * d)
    jtj = j.T.dot(j)
    jtr = j.T.dot(r)
    rest = None if link is None else link.restraints(j.shape[1])
    if rest is not None:
      if norm is None:
        norm = chi / max(len(r) - j.shape[1], 1)
        chi += norm * float(np.sum(rest[2] * rest[1] ** 2))
      rw = norm * rest[2]
      jtj += rest[0].T.dot(rw[:, None] * rest[0])
      jtr -= rest[0].T.dot(rw * rest[1])
    p0 = get_parameters(xs, flags)
    diag = np.diag(jtj).copy()
    diag[diag <= 1e-12 * max(diag.max(), 1e-300)] = max(diag.max(), 1.0)
    accepted, step = False, np.zeros_like(p0)
    for _ in range(16):
      a_mat = jtj + lam * np.diag(diag)
      try:
        x = np.linalg.solve(a_mat, jtr)
      except np.linalg.LinAlgError:
        x = np.linalg.lstsq(a_mat, jtr, rcond=None)[0]
      if link is None:
        set_parameters(xs, flags, p0 + z.dot(x))
      else:
        link.apply(p0, x)
      step = get_parameters(xs, flags) - p0
      with np.errstate(over="ignore", invalid="ignore"):
        _, fc_new, _ = calc.compute(xs)
      chi_new = float(np.sum(w * (f_obs_sq - fc_new) ** 2))
      if rest is not None:
        r_new = link.restraints(j.shape[1])
        chi_new += norm * float(np.sum(r_new[2] * r_new[1] ** 2))
      if np.isfinite(chi_new) and chi_new <= chi:
        chi, lam, accepted = chi_new, max(lam / 3, 1e-9), True
        break
      if link is not None:
        link.undo(p0, x)
      lam *= 5
    if not accepted:
      set_parameters(xs, flags, p0)
      _, fc_new, _ = calc.compute(xs)
    wr2 = math.sqrt(float(np.sum(w * (f_obs_sq - fc_new) ** 2)) / float(np.sum(w * f_obs_sq ** 2)))
    rec = {"cycle": cycle + 1, "wR2": wr2, "R1": r_factors(f_obs_sq, fc_new, m),
           "max_shift": float(np.max(np.abs(step))) if accepted and step.size else 0.0, "lambda": lam}
    history.append(rec)
    if verbose:
      line = "  cycle %2d  wR2 %.4f  R1 %.4f  %s  max shift %.2e" % (
        rec["cycle"], wr2, rec["R1"]["all"],
        "  ".join("R1(%s) %.4f" % (k, v) for k, v in rec["R1"].items() if k != "all"), rec["max_shift"])
      print(line, file=log) if log is not None else print(line)
    if not accepted or rec["max_shift"] < 1e-8:
      break
  return history


# ---------------------------------------------------------------------------
# de Wolff sections of the superspace Fourier map
# ---------------------------------------------------------------------------

def de_wolff_section(xs, h, m, f_obs_sq=None, label=None, axis=0, half_width=1.0, n_x=81, n_x4=80,
                     kind="obs", n_x4_quad=32, follow=False, table=None, radiation="xray"):
  """rho(x_axis, x4) through the average position of atom `label`, the other two coordinates fixed there
  (follow=True: they follow the atom's string instead, so its own peak stays on the section).

  rho(x, x4) = sum_H F(H) exp(-2 pi i (h.x + m x4)) over the data expanded by the superspace point group and
  Friedel's law; an atom's string sits at (xb + u(x4b), x4b + q.u). kind: 'obs' (|Fo| with calculated phases),
  'calc' or 'diff' (|Fo| - |Fc|). Returns (offsets along the axis in Angstrom, x4 values, rho (n_x4, n_x),
  the model's displacement along the axis in Angstrom on the same x4 values).
  """
  at = [a for a in xs.atoms if a.label == label][0]
  h, m = np.asarray(h), np.asarray(m)
  keys = {}
  for k in range(len(m)):
    for iop, op in enumerate(xs.ssg.ops):
      hp = np.rint(h[k].dot(op.r) + m[k] * op.m).astype(int)
      for sgn in (1, -1):
        keys.setdefault(tuple(sgn * hp) + (int(sgn * op.eps * m[k]),), (k, iop, sgn))
  hh = np.array([key[:3] for key in keys])
  mm = np.array([key[3] for key in keys])
  src, iop, sgn = np.array(list(keys.values())).T
  # only the measured reflections and their Friedel mates are computed (Friedel's law fails with f''); every image
  # follows from F(H g) = exp(-2 pi i H.T_g) F(H)
  f2, _, _ = calculator(np.vstack([h, -h]), np.concatenate([m, -m]), n_x4_quad, table,
                        radiation=radiation).compute(xs)
  t = np.array([op.t for op in xs.ssg.ops])
  tau = np.array([op.tau for op in xs.ssg.ops])
  ht = np.einsum("ij,ij->i", h[src], t[iop]) + m[src] * tau[iop]
  fc = np.where(sgn > 0, f2[src] * np.exp(-2j * math.pi * ht), f2[len(m) + src] * np.exp(2j * math.pi * ht))
  afc = np.abs(fc)
  phase = np.where(afc > 1e-6 * afc.max(), fc / np.maximum(afc, 1e-30), 0)
  if kind == "calc" or f_obs_sq is None:
    f = fc
  else:
    fo = np.sqrt(np.maximum(np.asarray(f_obs_sq, dtype=float)[src], 0) / xs.scale)
    f = (fo if kind == "obs" else fo - afc) * phase
  length = xs.unit_cell.parameters()[axis]
  s = np.linspace(-half_width, half_width, n_x) / length
  x4 = np.arange(n_x4) / float(n_x4)
  x4b = x4.copy()
  for _ in range(3):   # the string point at superspace x4 has x4b + q.u(x4b) = x4
    x4b = x4 - at.displacement(x4b).dot(xs.q)
  u = at.displacement(x4b)
  b = np.exp(-2j * math.pi * mm[:, None] * x4[None, :]) * f[:, None]
  # the point (site + offset along the axis) splits into a row factor and an axis factor shared by every row
  if follow:
    perp = np.ones(3)
    perp[axis] = 0
    b *= np.exp(-2j * math.pi * hh.dot((at.site + perp * u).T))
  else:
    b *= np.exp(-2j * math.pi * hh.dot(at.site))[:, None]
  a = np.exp(-2j * math.pi * hh[:, axis][:, None] * s[None, :])
  rho = np.real(b.T.dot(a))
  return s * length, x4, rho / xs.unit_cell.volume(), u[:, axis] * length


# ---------------------------------------------------------------------------
# settings: carry a model and its data between FLINT's setting and the superspace setting
# ---------------------------------------------------------------------------

class change_of_basis(object):
  """Superspace origin (p, d4) and basis P: x' = P^-1 (x - p), x4' = x4 - d4; q' = q P, h' = h P, A' = P^-1 A,
  U*' = P^-1 U* P^-T.

  A 3D origin shift alone (d4 = 0) leaves u(x4) unchanged and moves tau by M.p; d4 rotates the (sin, cos)
  pair of harmonic n of every wave by 2 pi n d4 and moves tau by (eps - 1) d4. Apply one object to cell, q,
  reflections, atoms and the superspace group together; inverse() goes back.
  """

  def __init__(self, p_matrix=None, origin_shift=None, d4=0.0):
    self.p = np.eye(3) if p_matrix is None else np.asarray(p_matrix, dtype=float)
    self.p_inv = np.linalg.inv(self.p)
    self.shift = np.zeros(3) if origin_shift is None else np.asarray(origin_shift, dtype=float)
    self.d4 = float(d4)

  def inverse(self):
    return change_of_basis(self.p_inv, -self.p_inv.dot(self.shift), -self.d4)

  def cb_op(self):
    """The sgtbx change_of_basis_op of the 3D part (new = c * old)."""
    t = -self.p_inv.dot(self.shift)
    if not (np.allclose(t * 144, np.round(t * 144), atol=1e-6)
            and np.allclose(self.p_inv * 12, np.round(self.p_inv * 12), atol=1e-6)):
      raise ValueError("origin shift / basis not on the sgtbx grid (1/144, 1/12): %s, %s" % (t, self.p_inv))
    return sgtbx.change_of_basis_op(sgtbx.rt_mx(
      sgtbx.rot_mx([int(round(v * 12)) for v in self.p_inv.ravel()], 12),
      sgtbx.tr_vec([int(round(v * 144)) for v in t], 144)))

  def superspace_group(self, ssg):
    """ssg in the new setting: R' = P^-1 R P, t' = P^-1 (t + (R - 1) p), M' = M P, tau' = tau + M.p + (eps-1) d4."""
    taus = {}
    for op in ssg.ops:
      r = self.p_inv.dot(op.r).dot(self.p)
      t = self.p_inv.dot(op.t + (op.r - np.eye(3)).dot(self.shift))
      taus[_op_key(r, t)] = op.tau + op.m.dot(self.shift) + (op.eps - 1) * self.d4
    basic = ssg.basic.change_basis(self.cb_op())
    gens = {}
    for g in _generators(basic):
      key = _op_key(np.array(g.r().as_double()).reshape(3, 3), np.array(g.t().as_double()))
      gens[g.as_xyz()] = taus[key]
    return superspace_group(basic, self.q(ssg.q), gens)

  def q(self, q):
    return np.asarray(q, dtype=float).dot(self.p)

  def hkl(self, h):
    return np.rint(np.asarray(h, dtype=float).dot(self.p)).astype(int)

  def unit_cell(self, uc):
    g6 = uc.metrical_matrix()
    g = np.array([[g6[0], g6[3], g6[4]], [g6[3], g6[1], g6[5]], [g6[4], g6[5], g6[2]]])
    gn = self.p.T.dot(g).dot(self.p)
    return uctbx.unit_cell(metrical_matrix=(gn[0, 0], gn[1, 1], gn[2, 2], gn[0, 1], gn[0, 2], gn[1, 2]))

  def _u_star(self, u):
    m = np.array([[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]])
    m = self.p_inv.dot(m).dot(self.p_inv.T)
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]])

  def _rotate(self, s, c):
    """(sin, cos) coefficients of harmonics n = 1.. after the x4 shift d4."""
    s, c = np.array(s, dtype=float), np.array(c, dtype=float)
    for n in range(len(s)):
      phi = TWO_PI * (n + 1) * self.d4
      cs, sn = math.cos(phi), math.sin(phi)
      s[n], c[n] = cs * s[n] - sn * c[n], sn * s[n] + cs * c[n]
    return s, c

  def atom(self, at):
    a, b = self._rotate(at.a, at.b)
    os_, oc = self._rotate(at.occ_sin, at.occ_cos)
    us, uc = self._rotate(at.u_sin, at.u_cos)
    saw = None if at.sawtooth is None else (self.p_inv.dot(at.sawtooth[0]), at.sawtooth[1] - self.d4, at.sawtooth[2])
    return modulated_atom(at.label, at.scattering_type, self.p_inv.dot(at.site - self.shift), at.u_iso,
                          at.occupancy, at.n_harmonics, [self.p_inv.dot(v) for v in a], [self.p_inv.dot(v) for v in b],
                          None if at.u_star is None else self._u_star(at.u_star), os_, oc,
                          [self._u_star(v) for v in us], [self._u_star(v) for v in uc], at.site_weight,
                          None if at.crenel is None else (at.crenel[0] - self.d4, at.crenel[1]),
                          [self.p_inv.dot(v) for v in at.legendre], saw)
