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
import threading
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


def _parse_hklm_line(line):
  """One .hklm line by the rules below: [h, k, l, m, I, sigma], or None for a line that is neither format."""
  try:
    if len(line) < 32:
      raise ValueError
    return [int(line[i:i + 4]) for i in (0, 4, 8, 12)] + [float(line[16:24]), float(line[24:32])]
  except ValueError:
    t = line.split()
    if not t:
      return None
    try:
      return [int(x) for x in t[:4]] + [float(t[4]), float(t[5])]
    except (ValueError, IndexError):
      return None


def _read_hklm_lines(lines, path):
  """The reference parser, line by line."""
  rows, bad = [], []
  for n, line in enumerate(lines, 1):
    v = _parse_hklm_line(line)
    if v is None:
      if line.split() and rows:   # before the data a header; after it skipped, not the end of the file
        bad.append((n, line))
      continue
    if not any(v[:4]):
      break
    rows.append(v)
  if bad:
    print("hklm: skipped %d unreadable line(s) in %s, the first is line %d: %r" % (len(bad), path, *bad[0]))
  return np.array(rows, dtype=float).reshape(-1, 6)


_HKLM_FIELDS = np.dtype([("h", "S4"), ("k", "S4"), ("l", "S4"), ("m", "S4"), ("i", "S8"), ("s", "S8")])


def _read_hklm_fast(lines):
  """The same rows in one numpy conversion, for the regular file: header lines, then fixed 4I4,2F8 lines (anything
  after column 32 ignored, as the reference does), ended by 0 0 0 0 or the end of the file. None as soon as
  anything is irregular - a short, blank, non-ASCII or unconvertible line in the data - so that the caller falls
  back on the reference parser for the whole file."""
  first = None
  for n, line in enumerate(lines):
    v = _parse_hklm_line(line)
    if v is not None:
      if len(line) < 32 or not any(v[:4]):
        return None
      first = n
      break
  if first is None:
    return np.zeros((0, 6))
  block = lines[first:]
  while block and not block[-1].strip():
    block = block[:-1]
  try:
    raw = np.array([l[:32] for l in block]).astype("S32")
  except UnicodeEncodeError:
    return None
  if np.any(np.char.str_len(raw) < 32):
    return None
  rec = raw.view(_HKLM_FIELDS).ravel()
  try:
    hklm = np.column_stack([rec[k].astype(np.int64) for k in ("h", "k", "l", "m")])
    vals = np.column_stack([rec[k].astype(np.float64) for k in ("i", "s")])
  except ValueError:
    return None
  zero = np.flatnonzero(~hklm.any(axis=1))
  stop = zero[0] if len(zero) else len(block)
  return np.column_stack([hklm[:stop].astype(float), vals[:stop]])


def read_hklm(path):
  """(h (N,3), m (N,), I, sigma) from an .hklm file: h k l m I sigma [batch], fixed 4I4,2F8 or space separated,
  ended by 0 0 0 0; lines that are neither are headers before the data and skipped (and reported) after it (the
  rules of Olex2's THklFile::LoadHklmFromStrings). A regular fixed-format file is converted in one go, anything
  else line by line; both give the same arrays."""
  a = _read_hklm_uniform(path)
  if a is None:
    with open(path, errors="replace") as f:
      lines = f.read().split("\n")
    a = _read_hklm_fast(lines)
    if a is None:
      a = _read_hklm_lines(lines, path)
  return a[:, :3].astype(int), a[:, 3].astype(int), a[:, 4], a[:, 5]


def _read_hklm_uniform(path):
  """_read_hklm_fast for the common file whose data lines all have one length, as fixed-size records straight
  from the bytes, with no work per line in Python. None for anything else (non-ASCII, CR line ends, lines of
  different lengths): the caller then goes through the text."""
  with open(path, "rb") as f:
    data = f.read()
  if b"\r" in data or not data.isascii():
    return None
  # the header: the lines before the first one the reference parser takes as data
  pos = 0
  while pos < len(data):
    nl = data.find(b"\n", pos)
    nl = len(data) if nl < 0 else nl
    v = _parse_hklm_line(data[pos:nl].decode("ascii"))
    if v is not None:
      if nl - pos < 32 or not any(v[:4]):
        return None
      break
    pos = nl + 1
  else:
    return np.zeros((0, 6))
  block = data[pos:].rstrip(b" \n")    # trailing blank lines, as the reference skips them
  width = block.find(b"\n")
  if width < 0:
    width = len(block)
  if width < 32 or (len(block) + 1) % (width + 1):
    return None
  n = (len(block) + 1) // (width + 1)
  buf = np.frombuffer(block + b"\n", dtype=np.uint8).reshape(n, width + 1)
  if np.any(buf[:, width] != 10) or np.any(buf[:, :width] == 10):
    return None
  rec = np.ascontiguousarray(buf[:, :32]).view(_HKLM_FIELDS).ravel()
  try:
    hklm = np.column_stack([rec[k].astype(np.int64) for k in ("h", "k", "l", "m")])
    vals = np.column_stack([rec[k].astype(np.float64) for k in ("i", "s")])
  except ValueError:
    return None
  zero = np.flatnonzero(~hklm.any(axis=1))
  stop = zero[0] if len(zero) else n
  return np.column_stack([hklm[:stop].astype(float), vals[:stop]])


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


def xharm_orthonormal(n, quadrature=64):
  """The orthonormalised x-harmonics g_1..g_n over the crenel interval, x in [-1/2, 1/2] with weight 1: Gram-Schmidt
  of (1, f_1, ..., f_n), f the plain x-harmonics of xharm_basis (the same treatment JANA gives the Legendre
  functions), the constant dropped. Returns (d, T) with g_j(x) = d_j + sum_i T[j, i] f_i(x), T lower triangular.
  A wave written in the g basis, u(x) = sum_j c_j g_j(x), is the plain wave with coefficients T^T c plus the
  constant d . c (moved into the average site or ADP); xharm_to_plain does that change of basis.
  The plain sin/cos(2 pi k x) are mutually orthogonal and orthogonal to 1 on this interval, so with XHARM_MODE
  either way d = 0 and T = sqrt(2) I: the difference between the two conventions is the normalisation alone."""
  xg, wg = np.polynomial.legendre.leggauss(quadrature)
  x, w = 0.5 * xg, 0.5 * wg
  f = np.array([np.ones_like(x)] + [xharm_basis(k + 1, x) for k in range(n)])
  coef = np.zeros((n + 1, n + 1))   # g_j = sum_i coef[j, i] f_i, f_0 = 1
  for j in range(n + 1):
    c = np.zeros(n + 1)
    c[j] = 1.0
    for i in range(j):
      gi = coef[i].dot(f)
      c -= np.sum(w * f[j] * gi) * coef[i]
    c /= math.sqrt(np.sum(w * c.dot(f) ** 2))
    coef[j] = c
  return coef[1:, 0], coef[1:, 1:]


def xharm_to_plain(coeffs):
  """Coefficients (n, width) of the orthonormalised x-harmonics -> (plain coefficients (n, width), the constant
  (width,) that the wave adds to the average)."""
  coeffs = np.asarray(coeffs, dtype=float)
  d, t = xharm_orthonormal(len(coeffs))
  return t.T.dot(coeffs), d.dot(coeffs)


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
    # commensurate: (x4 of the first section point, N) - the x4 integral becomes the mean over the N points
    # x4_j = start + j / N the atom takes in the supercell (commensurate_section); None: incommensurate
    self.section = None

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
    inside the crenel (the integrand is smooth there and zero outside). With a commensurate section: the N section
    points, weight 1/N each (0 outside a crenel), whatever n_x4."""
    if self.section is not None:
      start, n = self.section
      x4 = start + np.arange(n) / float(n)
      wt = np.full(n, 1.0 / n)
      if self.crenel is not None:
        wt[np.abs(self._local(x4)) > 0.5 + 1e-12] = 0.0
      return x4, wt
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
  """dispersion: {scattering type: (f', f'')}. extinction: SHELX's EXTI x, applied when it is not 0 (or is refined),
  which needs the wavelength (Angstrom): Fc^2 = scale |F|^2 (1 + 0.001 x |F|^2 lambda^3 / sin 2 theta)^(-1/2)."""

  def __init__(self, unit_cell, ssg, atoms, scale=1.0, dispersion=None, extinction=0.0, wavelength=None):
    self.unit_cell = unit_cell
    self.ssg = ssg
    self.atoms = list(atoms)
    self.scale = float(scale)
    self.dispersion = dict(dispersion or {})
    self.extinction = float(extinction)
    self.wavelength = wavelength

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


def auto_n_x4(xs, h, m, headroom=1.0):
  """n_x4 for the reflections (h, m) and the waves of xs: |m|max + n_h (e z / 2 + 6), a multiple of 4, at least 12,
  with z = 2 pi max |S| . (|A| + |B|) + 2 pi^2 max |q6| . (|U_sin| + |U_cos|) over atoms and reflections (A, B
  the largest displacive amplitudes of each component over the harmonics, U the ADP waves, q6 the six products of
  S that multiply U*) and n_h the largest harmonic order of any wave. The integrand of the x4 integral then has no
  harmonic the uniform rule cannot resolve above a Bessel tail J_k (I_k for the Debye-Waller factor) of
  k > e z / 2 + 6, which measured against n_x4 = 128 keeps every block family within 1e-10 (tst_modulation).
  headroom scales z, so that a refinement whose amplitudes grow keeps the rule it started with for longer.
  Returns (n_x4, z)."""
  s = np.abs(np.asarray(h, dtype=float) + np.outer(np.asarray(m, dtype=float), xs.q))
  q6 = np.column_stack([s[:, 0] ** 2, s[:, 1] ** 2, s[:, 2] ** 2,
                        2 * s[:, 0] * s[:, 1], 2 * s[:, 0] * s[:, 2], 2 * s[:, 1] * s[:, 2]])
  z, nh = 0.0, 1
  for at in xs.atoms:
    nh = max(nh, at.n_harmonics, len(at.occ_sin), len(at.u_sin))
    za = 0.0
    if at.n_harmonics:
      w = np.abs(at.a).max(axis=0) + np.abs(at.b).max(axis=0)
      za = TWO_PI * float(np.max(s.dot(w)))
    if len(at.u_sin):
      u = np.abs(at.u_sin).max(axis=0) + np.abs(at.u_cos).max(axis=0)
      za += TWO_PI_SQ * float(np.max(q6.dot(u)))
    z = max(z, za)
  n = int(np.max(np.abs(m))) + nh * (math.e * z * headroom / 2 + 6) if len(m) else 12
  return max(12, int(4 * math.ceil(n / 4.0))), z


def commensurate_section(xs, t0, n):
  """The commensurate structure of section t0 with N = n points per atom (q = r / n): every atom expanded into its
  images under all superspace operations (so the result has the identity alone), each image with the waves the
  operation gives it (_atom_map; crenel centre eps c + M.x + tau) and section (t0 + q.x', n): its x4 integral is
  then the mean over the n values x4 = t0 + q.(x' + L) takes over the supercell translations L. The structure
  factor of a reflection (h, m) is the supercell's F(H), H = h + m q, divided by n and times exp(2 pi i m t0)
  (test_commensurate_section). Gradients of crenel centres and widths are not defined for a finite sum (the
  points do not move with them); sawtooth atoms are not handled."""
  import copy
  every = _flags({k: True for k in ("site", "adp", "occupancy", "modulation", "occ_waves", "adp_waves", "crenel",
                                    "legendre", "xharm", "u_xharm")})
  identity = [op for op in xs.ssg.ops if np.allclose(op.as_4x4(), np.eye(4)) and not np.any(op.t) and not op.tau]
  ssg = copy.copy(xs.ssg)
  ssg.ops = identity[:1] or [superspace_operator(np.eye(3), np.zeros(3), 1, np.zeros(3))]
  atoms = []
  for at in xs.atoms:
    if at.sawtooth is not None:
      raise NotImplementedError("commensurate_section: sawtooth on %s" % at.label)
    blocks = _blocks(at, every)
    p = np.concatenate([np.atleast_1d(np.asarray(get(), dtype=float)) for _, _, get, _ in blocks]) if blocks else np.zeros(0)
    for k, op in enumerate(xs.ssg.ops):
      img = copy.deepcopy(at)
      img.label = "%s_%d" % (at.label, k)
      v, i = _atom_map(at, op, every).dot(p), 0
      for _, _, get, put in _blocks(img, every):
        m = len(get())
        put(v[i:i + m])
        i += m
      img.site = op.r.dot(at.site) + op.t
      if at.crenel:
        img.crenel = (op.eps * at.crenel[0] + op.m.dot(at.site) + op.tau, at.crenel[1])
      img.section = (t0 + xs.q.dot(img.site), int(n))
      atoms.append(img)
  out = modulated_structure(xs.unit_cell, ssg, atoms, xs.scale, xs.dispersion)
  out.extinction, out.wavelength = getattr(xs, "extinction", 0.0), getattr(xs, "wavelength", None)
  return out


def twin_indices(q, t, h, m):
  """(h', m') of the twin domain with matrix t (3x3, acting on the row vector h): (h + m q) t = h' + m' q, which
  needs q t = s q + G with s = +-1 and G integral; then h' = h t + m G, m' = s m."""
  q = np.asarray(q, dtype=float)
  t = np.asarray(t, dtype=float)
  qt = q.dot(t)
  for s in (1, -1):
    g = qt - s * q
    if np.allclose(g, np.round(g), atol=1e-6):
      g = np.round(g)
      hp = np.rint(np.asarray(h, dtype=float).dot(t) + np.outer(m, g)).astype(int)
      return hp, s * np.asarray(m)
  raise IncompatibleQ("twin matrix %s does not map q = %s onto +-q" % (t.tolist(), q.tolist()))


class twinned_calculator(object):
  """F^2 of a twinned crystal: sum_k f_k F^2(domain k's indices), the domain indices from twin_indices, one
  calculator per domain (each keeps its own invariants and slices). twins: [(matrix, fraction)], the first
  domain the reference (identity for JANA files). The fractions are parameters of the twin, not of xs: with
  refine(..., twins=...) the last k - 1 of them are refined after the scale (the first is 1 - their sum)."""

  def __init__(self, h, m, q, twins, **kw):
    self.h, self.m = np.asarray(h), np.asarray(m)
    self.mats = [np.asarray(t, dtype=float) for t, _ in twins]
    self.fractions = np.array([f for _, f in twins], dtype=float)
    self.calcs = []
    for t in self.mats:
      hk, mk = twin_indices(q, t, self.h, self.m)
      self.calcs.append(calculator(hk, mk, **kw))

  def compute(self, xs, flags=None, gradients=False):
    """(F of the reference domain, sum_k f_k F_k^2, gradients): the gradients of the model's parameters summed
    with the fractions, then dF^2/df_k = F_k^2 - F_1^2 for k = 2..n (f_1 = 1 - sum of the others)."""
    parts = [c.compute(xs, flags, gradients) for c in self.calcs]
    fsq = sum(f * p[1] for f, p in zip(self.fractions, parts))
    if not gradients:
      return parts[0][0], fsq, None
    d = sum(f * p[2] for f, p in zip(self.fractions, parts))
    dfrac = np.column_stack([p[1] - parts[0][1] for p in parts[1:]]) if len(parts) > 1 else np.zeros((len(fsq), 0))
    return parts[0][0], fsq, np.hstack([d, dfrac])


def commensurate_compute(calc, xs, t0, n, flags=None, gradients=False, step=1e-6):
  """(F, F^2, dF^2/dp) of the commensurate structure commensurate_section(xs, t0, n) for calc's reflections, the
  gradients with respect to the parameters of xs. The images' gradients are folded back (_commensurate_fold);
  a site also moves its images' wave phases and section points (the finite sum has no fixed nodes), which the
  folded site columns lack: that part is added as a central difference over the site, image sites held, atom by
  atom on one-atom structures (six F passes in all, step 1e-6 in fractional units). Crenel borders are not
  differentiated (a section point crossing one is a jump)."""
  flags = _flags(flags)
  xc = commensurate_section(xs, t0, n)
  if not gradients:
    return calc.compute(xc, flags)[:2] + (None,)
  f, fsq, d = calc.compute(xc, flags, gradients=True)
  d = _commensurate_fold(xs, flags)(d)
  if flags["site"]:
    sub = calculator(calc.h, calc.m, calc.n_x4, calc.table, threads=calc.threads, radiation=calc.radiation)
    col = 0
    for at in xs.atoms:
      blocks = _blocks(at, flags)
      if blocks and blocks[0][0] == "site":
        for i in range(3):
          fd = []
          for sgn in (1, -1):
            one = modulated_structure(xs.unit_cell, xs.ssg, [at], 1.0, xs.dispersion)
            moved = commensurate_section(one, t0, n)
            shifted = at.site.copy()
            shifted[i] += sgn * step
            at_s = _atom_with_site(at, shifted)
            ref = commensurate_section(modulated_structure(xs.unit_cell, xs.ssg, [at_s], 1.0, xs.dispersion), t0, n)
            for a, b in zip(ref.atoms, moved.atoms):
              a.site = b.site          # image sites held: their motion is in the folded columns already
            fd.append(sub.compute(ref)[0])
          dfi = (fd[0] - fd[1]) / (2 * step)
          d[:, col + i] += 2 * xs.scale * (f.real * dfi.real + f.imag * dfi.imag)
      col += sum(len(get()) for _, _, get, _ in blocks)
  return f, fsq, d


def _atom_with_site(at, site):
  import copy
  b = copy.deepcopy(at)
  b.site = np.asarray(site, dtype=float)
  return b


def _commensurate_fold(xs, flags):
  """d -> d Z: gradients with respect to the parameters of commensurate_section(xs, ...)'s images (atom by atom,
  each atom's images in operation order, then scale and extinction) folded onto the parameters of xs. An image's
  parameters are T_k p (+ constants: translations, crenel shift) with T_k = _atom_map(atom, op_k), so dF/dp =
  sum_k dF/dp_k T_k. T_k depends on the site (the phases of the wave pairs): call it again after every update."""
  flags = _flags(flags)
  maps, i_img, i_org = [], 0, 0
  for at in xs.atoms:
    n = sum(len(get()) for _, _, get, _ in _blocks(at, flags))
    for op in xs.ssg.ops:
      maps.append((i_img, i_org, n, _atom_map(at, op, flags)))
      i_img += n
    i_org += n
  n_tail = int(flags["scale"]) + int(flags["extinction"])

  def fold(d):
    out = np.zeros((d.shape[0], i_org + n_tail))
    for ii, io, n, t in maps:
      if n:
        out[:, io:io + n] += d[:, ii:ii + n].dot(t)
    if n_tail:
      out[:, i_org:] = d[:, i_img:i_img + n_tail]
    return out
  return fold


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
    # "auto": auto_n_x4 decides at the first compute(), and again whenever the amplitudes outgrow the z it was set
    # for (see _resolve_n_x4); a number is used as given
    self.auto = n_x4 == "auto"
    self.n_x4 = 32 if self.auto else n_x4
    self._auto_z = None
    self.table, self._table_f = table, {}
    self.threads = threads   # None: one per CPU, capped by the atom count and a 2 GB working set
    # what depends on the reflections, q, the operations, the cell and the dispersion only, kept across the
    # compute() calls of a refinement; rebuilt when any of those changes (_invariant_key)
    self._invariants = (None, None)
    self._share_classes = True   # False: every operation evaluates its own x4 sums (the reference path)
    # False: the numpy x4 sums (the reference path, and the only one without the extension)
    self.use_compiled = USE_COMPILED and _ext is not None
    self._mode = None            # None: pool_mode decides; "atoms" or "reflections" forces it
    self._sub = {}               # reflection-slice calculators, by number of slices (_chunks)
    self._tables = {"lock": threading.Lock()}   # the per-atom x4 tables, shared with the slices (_atom_tables)
    # buffers lent by _normal_equations for one pass over its slices (_take); None: every array is new
    self._workspace = None

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
    # per operation, stacked (operation, reflection, ...) so that the compiled sums take them whole; ops[k] holds
    # views of them
    n_op, nref = len(xs.ssg.ops), len(self.m)
    st = {"hp": np.empty((n_op, nref, 3)), "ph0": np.empty((n_op, nref)), "dw6": np.empty((n_op, nref, 6)),
          "sp": np.empty((n_op, nref, 3)), "mp": np.empty((n_op, nref))}
    ops = []
    for k, op in enumerate(xs.ssg.ops):
      sp = st["sp"][k]
      sp[...] = s_vec.dot(op.r)
      q6 = np.column_stack([sp[:, 0] ** 2, sp[:, 1] ** 2, sp[:, 2] ** 2,
                            2 * sp[:, 0] * sp[:, 1], 2 * sp[:, 0] * sp[:, 2], 2 * sp[:, 1] * sp[:, 2]])
      st["hp"][k] = self.h.dot(op.r) + np.outer(self.m, op.m)
      st["mp"][k] = op.eps * self.m
      st["ph0"][k] = self.h.dot(op.t) + self.m * op.tau
      st["dw6"][k] = -TWO_PI_SQ * q6
      # hp, mp, sp, ph0, q6, and the two factors the gradients multiply by: 2 pi i sp and -2 pi^2 q6
      ops.append((st["hp"][k], st["mp"][k], sp, st["ph0"][k], q6, (1j * TWO_PI) * sp, st["dw6"][k]))
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
    st["s_sq"] = s_sq
    st["cls"] = np.array([c[0] for c in classes], dtype=np.int64)
    st["conj"] = np.array([int(c[1]) for c in classes], dtype=np.int64)
    value = (s_vec, s_sq, ff, ops, classes, st)
    self._invariants = (key, value)
    return value

  # scaling of the atom pool saturates near four threads (memory bandwidth)
  default_threads = 4

  def _threads(self):
    return max(1, self.threads or min(self.default_threads, os.cpu_count() or 1))

  def _chunks(self, n_chunks, threads=1):
    """Calculators over consecutive slices of the reflections, made once per (number of slices, threads) and kept:
    each keeps its own invariants under the same key as this one (its reflections are fixed, the rest is in the
    key). threads 1: one slice per thread of the pool; more: slices of the normal equations, each with the atom
    pool inside."""
    key = (n_chunks, threads)
    if key not in self._sub:
      bounds = np.linspace(0, len(self.m), n_chunks + 1).astype(int)
      subs = []
      for a, b in zip(bounds[:-1], bounds[1:]):
        c = calculator(self.h[a:b], self.m[a:b], self.n_x4, self.table, threads=threads, radiation=self.radiation)
        c._share_classes, c._mode = self._share_classes, "atoms"
        c._tables = self._tables
        c.use_compiled = self.use_compiled
        subs.append(c)
      self._sub[key] = subs
    return self._sub[key]

  def pool_mode(self, xs):
    """'atoms' or 'reflections' for the thread pool: atoms when there are enough of them to keep every thread busy
    (twice the threads), else slices of the reflections - a few atoms with many satellites."""
    if self._mode is not None:
      return self._mode
    t = self._threads()
    if t == 1 or len(xs.atoms) >= 2 * t or len(self.m) < 2000 * t:
      return "atoms"
    return "reflections"

  @staticmethod
  def _atom_key(xs, n_x4):
    """Everything the per-atom x4 tables depend on: every wave and crenel of every atom, the average occupancy,
    n_x4, and whether the ADP is isotropic (the compiled column layout). Not the site or the ADP values (they do not
    enter the tables)."""
    parts = [n_x4, XHARM_MODE]
    for at in xs.atoms:
      parts.append((at.a.tobytes(), at.b.tobytes(), at.occ_sin.tobytes(), at.occ_cos.tobytes(),
                    at.u_sin.tobytes(), at.u_cos.tobytes(), at.legendre.tobytes(), at.xharm.tobytes(),
                    at.u_xharm.tobytes(), at.crenel, at.occupancy, at.u_star is None, at.section,
                    None if at.sawtooth is None else (at.sawtooth[0].tobytes(),) + tuple(at.sawtooth[1:])))
    return tuple(parts)

  def _atom_tables(self, xs, gradients):
    """Per atom: the x4 nodes and weights, sin/cos tables, u, p(x4), the ADP-wave sum at the nodes (pre) and
    the gradients' local tables (extra), and for the compiled sums their node arrays and column specifications
    (comp, _compiled_atom). They do not depend on the reflections, so the slices of a refinement
    share one set (self._tables, one dict for a calculator and all its slices), rebuilt when _atom_key changes -
    after every parameter update - and kept while only sites, ADPs or the scale move."""
    key = self._atom_key(xs, self.n_x4)
    t = self._tables
    with t["lock"]:
      if t.get("key") == key and (t["gradients"] or not gradients):
        return t["value"]
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
      # the compiled sums' node arrays and column specifications, from the same tables
      comp = [_compiled_atom(at, pre[ia], extra[ia], gradients) for ia, at in enumerate(xs.atoms)] \
          if _ext is not None else None
      t.update(key=key, gradients=gradients, value=(pre, extra, comp))
      return pre, extra, comp

  def _resolve_n_x4(self, xs):
    """With n_x4 = "auto": the rule's n_x4, set with 25 % headroom on z, and set again (slices dropped, so that
    they take the new value) only when the current z exceeds the one it was set for. Within a refinement step the
    value therefore does not change between the gradient pass and its trial passes unless the waves have grown."""
    if not self.auto:
      return
    z = auto_n_x4(xs, self.h, self.m)[1]
    if self._auto_z is None or z > self._auto_z:
      self.n_x4, _ = auto_n_x4(xs, self.h, self.m, headroom=1.25)
      self._auto_z = 1.25 * z
      self._sub = {}

  def compute(self, xs, flags=None, gradients=False):
    self._resolve_n_x4(xs)
    if self.pool_mode(xs) == "reflections":
      subs = self._chunks(self._threads())
      for c in subs:
        c._share_classes, c.use_compiled = self._share_classes, self.use_compiled
      with concurrent.futures.ThreadPoolExecutor(len(subs)) as ex:
        parts = list(ex.map(lambda c: c.compute(xs, flags, gradients), subs))
      f = np.concatenate([p[0] for p in parts])
      fsq = np.concatenate([p[1] for p in parts])
      return f, fsq, (np.vstack([p[2] for p in parts]) if gradients else None)
    return self._compute_rows(xs, flags, gradients)

  def _compute_rows(self, xs, flags=None, gradients=False):
    flags = _flags(flags)
    q = xs.q
    s_vec, s_sq, ff, ops, classes, st = self._per_operation(xs)
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
    pre, extra, comp = self._atom_tables(xs, gradients)
    compiled = self.use_compiled and comp is not None
    if gradients and not compiled:     # the numpy path's blocks (the compiled one makes its own below)
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
    if compiled and gradients:
      # one (N, columns) block per atom, the gradient dict entries views of it; all atoms' blocks in one flat
      # buffer, reused from slice to slice when _normal_equations lends a workspace (a fresh ~0.3 GB per slice
      # costs page faults that serialise the atom threads: 1.5-2.3 s per large slice instead of 1.2-1.4)
      grads = []
      widths = [nref * comp[ia]["ncol"] for ia in range(len(xs.atoms))]
      flat = _take(self._workspace, "g", (sum(widths),), complex, zero=True)
      o = 0
      for ia, at in enumerate(xs.atoms):
        g = flat[o:o + widths[ia]].reshape(nref, comp[ia]["ncol"])
        o += widths[ia]
        gr = {}
        for key, (off, shape) in comp[ia]["offsets"].items():
          width = int(np.prod(shape)) if shape else 1
          gr[key] = g[:, off] if not shape else g[:, off:off + width].reshape((nref,) + shape)
        gr["_all"] = g
        grads.append(gr)
    if compiled:
      ops_c = (st["hp"], st["ph0"], st["dw6"], st["sp"], st["mp"], st["s_sq"], st["cls"], st["conj"])

      def atom_sum_compiled(ia):
        at, c = xs.atoms[ia], comp[ia]
        if tab is None:
          f = ff[at.scattering_type][None, :]
        else:
          f = np.array([tab[iop][:, ia] for iop in range(len(ops))])
        f = np.ascontiguousarray(f, dtype=complex)
        out = np.zeros(nref, complex)
        u_star = np.zeros(0) if at.u_star is None else np.ascontiguousarray(at.u_star, dtype=float)
        _ext.atom_sums(ops_c + (f,),
                       (np.ascontiguousarray(at.site, dtype=float), at.u_iso, u_star, at.site_weight, c["x4"],
                        c["disp"], c["uw"], c["w"], c["du"], c["duw"], c["w2"], c["specs"], c["spec_c"]),
                       (out, grads[ia]["_all"] if gradients else None))
        return out

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
          # a real factor on the real and imaginary parts: not a complex product (numpy would promote it)
          dw = np.exp(dw6.dot(uw.T))
          e.real *= dw
          e.imag *= dw
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
    n_threads = max(1, min(self._threads(), len(xs.atoms),
                           int(2e9 // (40 * nref * max(len(p[0]) for p in pre)))))
    if compiled:
      atom_sum = atom_sum_compiled      # the GIL is released for a whole atom; no (N, K) buffers
    if n_threads == 1:
      for ia in range(len(xs.atoms)):
        f_calc += atom_sum(ia)
    else:
      with concurrent.futures.ThreadPoolExecutor(n_threads) as ex:
        for c in ex.map(atom_sum, range(len(xs.atoms))):
          f_calc += c
    f_sq = np.abs(f_calc) ** 2
    if not gradients:
      return _finish(xs, flags, f_calc, f_sq, None, s_sq)
    if compiled:
      # per atom, the refined columns of its one block straight into d|F|^2/dp
      flag_of = {"site": "site", "adp": "adp", "occ": "occupancy", "a": "modulation", "b": "modulation",
                 "os": "occ_waves", "oc": "occ_waves", "us": "adp_waves", "uc": "adp_waves", "cren": "crenel",
                 "leg": "legendre", "xh": "xharm", "ux": "u_xharm"}
      idx = []
      for ia in range(len(xs.atoms)):
        sel = []
        for key, (off, shape) in comp[ia]["offsets"].items():
          if flags[flag_of[key]]:
            sel.extend(range(off, off + (int(np.prod(shape)) if shape else 1)))
        idx.append(sel)
      d_fsq = _take(self._workspace, "d", (nref, sum(len(i) for i in idx) + int(flags["scale"])
                                           + int(flags["extinction"])), float)
      fr, fi = 2 * xs.scale * f_calc.real[:, None], 2 * xs.scale * f_calc.imag[:, None]
      starts = np.concatenate([[0], np.cumsum([len(i) for i in idx])]).astype(int)

      def assemble(ia):
        """One atom's refined columns of d|F|^2/dp = 2 scale Re(F* dF/dp); atoms write disjoint columns."""
        g, sel, j = grads[ia]["_all"], idx[ia], starts[ia]
        if len(sel) == g.shape[1]:
          d_fsq[:, j:j + len(sel)] = fr * g.real + fi * g.imag
        elif sel:
          d_fsq[:, j:j + len(sel)] = fr * g.real[:, sel] + fi * g.imag[:, sel]
      # in the pool as well: on the large slices this assembly was a quarter of a 4-thread gradient pass
      if n_threads == 1:
        for ia in range(len(xs.atoms)):
          assemble(ia)
      else:
        with concurrent.futures.ThreadPoolExecutor(n_threads) as ex:
          list(ex.map(assemble, range(len(xs.atoms))))
      j = int(starts[-1])
      if flags["scale"]:
        d_fsq[:, j] = f_sq
      return _finish(xs, flags, f_calc, f_sq, d_fsq, s_sq)
    cols = []
    for gr in grads:
      for key, flag in (("site", "site"), ("adp", "adp"), ("occ", "occupancy"), ("a", "modulation"),
                        ("b", "modulation"), ("os", "occ_waves"), ("oc", "occ_waves"), ("us", "adp_waves"),
                        ("uc", "adp_waves"), ("cren", "crenel"), ("leg", "legendre"), ("xh", "xharm"),
                        ("ux", "u_xharm")):
        if flags[flag]:
          cols.append(gr[key].reshape(nref, -1))
    # d|F|^2/dp = 2 Re(F* dF/dp), block by block into one real array: no (N, n_params) complex temporary
    d_fsq = np.empty((nref, sum(c.shape[1] for c in cols) + int(flags["scale"]) + int(flags["extinction"])))
    fr, fi, j = 2 * xs.scale * f_calc.real[:, None], 2 * xs.scale * f_calc.imag[:, None], 0
    for c in cols:
      d_fsq[:, j:j + c.shape[1]] = fr * c.real + fi * c.imag
      j += c.shape[1]
    if flags["scale"]:
      d_fsq[:, j] = f_sq
    return _finish(xs, flags, f_calc, f_sq, d_fsq, s_sq)


def _take(ws, name, shape, dtype, zero=False):
  """An array of this shape: a view of the workspace's buffer `name` (grown when too small) when there is a
  workspace, else a new one. The workspace belongs to one _normal_equations pass, which uses every slice's
  arrays before it computes the next, and is dropped at its end: nothing is kept between passes."""
  n = int(np.prod(shape))
  if ws is None:
    return np.zeros(shape, dtype) if zero else np.empty(shape, dtype)
  buf = ws.get(name)
  if buf is None or buf.size < n or buf.dtype != np.dtype(dtype):
    buf = ws[name] = np.empty(n, dtype)
  out = buf[:n].reshape(shape)
  if zero:
    out.fill(0)
  return out


def _finish(xs, flags, f_calc, f_sq, d_fsq, s_sq):
  """(F, Fc^2, dFc^2/dp) from F, |F|^2 and d(scale |F|^2)/dp (the scale column, if any, holding |F|^2): with
  xs.extinction or flags["extinction"], SHELX's EXTI correction y = (1 + a x |F|^2)^(-1/2), a = 0.001 lambda^3 /
  sin 2 theta, so that Fc^2 = scale |F|^2 y; the parameter columns take the factor d(|F|^2 y)/d|F|^2 =
  y^3 (1 + a x |F|^2 / 2), the scale column |F|^2 y, the extinction column (the last one, allocated by the
  caller) dFc^2/dx = -scale a |F|^4 y^3 / 2. Without extinction everything passes unchanged."""
  ext = flags is not None and _flags(flags)["extinction"]
  if not ext and xs.extinction == 0.0:
    return f_calc, xs.scale * f_sq, d_fsq
  if not xs.wavelength:
    raise ValueError("extinction needs modulated_structure.wavelength")
  lam = float(xs.wavelength)
  sin_t = np.minimum(lam * np.sqrt(s_sq) / 2, 1.0)
  a = 0.001 * lam ** 3 / np.maximum(2 * sin_t * np.sqrt(1 - sin_t ** 2), 1e-12)
  u = 1 + a * xs.extinction * f_sq
  y = 1 / np.sqrt(u)
  if d_fsq is not None:
    fl = _flags(flags)
    n_tail = int(fl["scale"]) + int(ext)
    n_par = d_fsq.shape[1] - n_tail
    d_fsq[:, :n_par] *= (y ** 3 * (1 + 0.5 * a * xs.extinction * f_sq))[:, None]
    if fl["scale"]:
      d_fsq[:, n_par] = f_sq * y
    if ext:
      d_fsq[:, -1] = -0.5 * xs.scale * a * f_sq ** 2 * y ** 3
  return f_calc, xs.scale * f_sq * y, d_fsq


# the compiled per-class x4 sums (smtbx_modulation_ext.atom_sums); None when the extension is not built, and then
# the numpy path below is used. calculator.use_compiled = False forces numpy.
USE_COMPILED = True     # the default of calculator.use_compiled
try:
  import boost_adaptbx.boost.python as _bp
  _ext = _bp.import_ext("smtbx_modulation_ext")
except ImportError:
  _ext = None

# the gradient blocks of one atom in the order calculator._compute_rows lists them, (key, width per atom)
def _grad_layout(at):
  nh, no, nu = at.n_harmonics, len(at.occ_sin), len(at.u_sin)
  return [("site", (3,)), ("adp", (1 if at.u_star is None else 6,)), ("occ", ()), ("a", (nh, 3)), ("b", (nh, 3)),
          ("os", (no,)), ("oc", (no,)), ("us", (nu, 6)), ("uc", (nu, 6)), ("cren", (2 if at.crenel else 0,)),
          ("leg", (len(at.legendre), 3)), ("xh", (len(at.xharm), 3)), ("ux", (len(at.u_xharm), 6))]


def _compiled_atom(at, pre, ex, gradients):
  """What atom_sums needs of one atom besides the site and ADP: the node arrays, the weight columns w (and w2 for
  the x4 derivative of the integrand, crenel atoms) whose x4 sums with e the numpy path forms one by one in
  x4_sums, and the specifications that turn the sums into gradient columns as atom_sum's odd/even terms do:
  rows (source, column, kind, first output column), kinds 0 c E, 1 c O hp, 2 c O sp (2 pi i sp), 3 c E dw6,
  4 c E s^2."""
  x4, wt, sin_t, cos_t, disp, p, uw, dp_docc = pre
  pw = wt * p
  cols, cols2, specs, cvals = [pw], [], [], []
  K = len(x4)
  # node arrays component-major, (3, K) and (6, K), so that the node loops read them contiguously
  out = {"x4": np.ascontiguousarray(x4), "disp": np.ascontiguousarray(disp.T),
         "uw": np.zeros(0) if uw is None else np.ascontiguousarray(uw.T), "du": np.zeros(0), "duw": np.zeros(0)}
  if gradients:
    off, offs = 0, {}
    for key, shape in _grad_layout(at):
      offs[key] = (off, shape)
      off += int(np.prod(shape)) if shape else 1
    out["ncol"], out["offsets"] = off, offs
    two_pi_i = 1j * TWO_PI

    def col(v, second=False):
      lst = cols2 if second else cols
      lst.append(v)
      return (1 if second else 0, len(lst) - 1)

    def spec(src, kind, dest, c):
      specs.append((src[0], src[1], kind, dest))
      cvals.append(c)
    c_pw, c_wt = (0, 0), col(wt)
    spec(c_pw, 1, offs["site"][0], two_pi_i)
    if at.u_star is None:
      spec(c_pw, 4, offs["adp"][0], -TWO_PI_SQ)
    else:
      spec(c_pw, 3, offs["adp"][0], 1.0)
    spec(c_wt, 0, offs["occ"][0], dp_docc)
    nh, no, nu = at.n_harmonics, len(at.occ_sin), len(at.u_sin)
    for n in range(max(nh, nu)):
      c_s, c_c = col(pw * sin_t[n]), col(pw * cos_t[n])
      if n < nh:
        spec(c_s, 2, offs["a"][0] + 3 * n, two_pi_i)
        spec(c_c, 2, offs["b"][0] + 3 * n, two_pi_i)
      if n < nu:
        spec(c_s, 3, offs["us"][0] + 6 * n, 1.0)
        spec(c_c, 3, offs["uc"][0] + 6 * n, 1.0)
    for n in range(no):
      spec(col(wt * sin_t[n]), 0, offs["os"][0] + n, 1.0)
      spec(col(wt * cos_t[n]), 0, offs["oc"][0] + n, 1.0)
    for n in range(len(at.legendre)):
      spec(col(pw * ex["leg"][n]), 2, offs["leg"][0] + 3 * n, two_pi_i)
    for n in range(len(at.xharm)):
      spec(col(pw * ex["xh"][n]), 2, offs["xh"][0] + 3 * n, two_pi_i)
    for n in range(len(at.u_xharm)):
      spec(col(pw * ex["ux"][n]), 3, offs["ux"][0] + 6 * n, 1.0)
    if at.crenel:
      loc, dp, c0 = ex["loc"], ex["dp"], offs["cren"][0]
      spec(col(pw, True), 0, c0, 1.0)
      spec(col(wt * dp), 0, c0, 1.0)
      spec(col(wt * ex["p_waves"] * dp_docc), 0, c0 + 1, 1.0)
      spec(col(pw * loc, True), 0, c0 + 1, 1.0)
      spec(col(wt * dp * loc), 0, c0 + 1, 1.0)
      out["du"] = np.ascontiguousarray(ex["du"].T)
      if ex["duw"] is not None:
        out["duw"] = np.ascontiguousarray(ex["duw"].T)
  # (K, M), the columns zero-padded to a multiple of 8: atom_sums' node sums take eight columns at a time
  pad = lambda c: np.ascontiguousarray(np.vstack([np.array(c), np.zeros((-len(c) % 8, K))]).T)
  out["w"] = pad(cols)
  out["w2"] = pad(cols2) if cols2 else np.zeros(0)
  out["specs"] = np.array(specs, dtype=np.int64).reshape(-1, 4)
  out["spec_c"] = np.array(cvals, dtype=complex)
  assert out["w"].shape == (K, len(cols) + (-len(cols) % 8))
  return out


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
         "scale": True, "extinction": False}
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
  if flags["extinction"]:
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
  zeros (the identity for atoms in general positions) was the largest serial cost of an LM cycle. Identity
  blocks are copied, and when Z is the identity a itself is returned (the caller must not modify it then)."""
  ident = [b.shape[0] == b.shape[1] and _is_identity(b) for b in blocks]
  if all(ident) and a.shape[1] == sum(b.shape[0] for b in blocks):
    return a      # Z = I (every atom on a general position): a itself, no copy
  out = np.empty((a.shape[0], sum(b.shape[1] for b in blocks)))
  i = j = 0
  for b, is_i in zip(blocks, ident):
    if is_i:
      out[:, j:j + b.shape[1]] = a[:, i:i + b.shape[0]]     # the same numbers as the product, without a dgemm
    else:
      out[:, j:j + b.shape[1]] = a[:, i:i + b.shape[0]].dot(b)
    i, j = i + b.shape[0], j + b.shape[1]
  return out


def _is_identity(b):
  n = b.shape[0]
  return bool(np.all(b.ravel()[::n + 1] == 1.0)) and np.count_nonzero(b) == n


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
  if flags["extinction"]:
    p.append(xs.extinction)
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
    i += 1
  if flags["extinction"]:
    xs.extinction = float(p[i])


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
    self.n_tail = int(flags["scale"]) + int(flags["extinction"])     # scale, extinction: after the atoms
    self.n_p = i + self.n_tail
    self.na = np.asarray(average.jacobian()).shape[1]
    self.zb = _reduction_blocks(xs, dict(flags, site=False, adp=False, scale=False, extinction=False))
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
    """Columns: the average's independent parameters, the free wave directions, the scale and extinction."""
    d = self._fold(d)
    cols = [d[:, self.avg_idx].dot(np.asarray(self.average.jacobian())), _times_blocks(d[:, self.wave_idx], self.zb)]
    if self.n_tail:
      cols.append(d[:, -self.n_tail:])
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
    if self.n_tail:
      step[-self.n_tail:] = x[-self.n_tail:]
    set_parameters(self.xs, self.flags, p0 + step)
    self.average.shift(x[:self.na])
    self.sync()

  def undo(self, p0, x):
    self.average.shift(-x[:self.na])
    set_parameters(self.xs, self.flags, p0)


def _blocks_times(blocks, x):
  """Z x for the block-diagonal Z of _reduction_blocks, block by block (no dense Z)."""
  out, j = [], 0
  for b in blocks:
    out.append(b.dot(x[j:j + b.shape[1]]))
    j += b.shape[1]
  return np.concatenate(out)


def _lm_factor_solve(buf, jtj_diag, damp, rhs):
  """(J^T J + damp) x = rhs, J^T J held in the upper triangle of buf (Fortran order) and its diagonal in jtj_diag.
  The lower triangle is refilled from the upper one with J^T J + damp on the diagonal and Cholesky-factorised in
  place (LAPACK potrf, lower), so the upper triangle - J^T J - survives for the next damping and no second n x n
  matrix is ever made. None without scipy or when the matrix is not positive definite (the caller falls back)."""
  try:
    from scipy.linalg import lapack
  except ImportError:
    return None
  _mirror_upper(buf)
  buf[np.diag_indices_from(buf)] = jtj_diag + damp
  c, info = lapack.dpotrf(buf, lower=1, clean=0, overwrite_a=1)
  if info != 0:
    return None
  x, info = lapack.dpotrs(c, rhs, lower=1)
  return x if info == 0 else None


def _slices(calc, n, max_rows):
  """The calculators of the slices _normal_equations uses: [calc] when one slice holds everything."""
  n_chunks = max(1, -(-n // max(1, max_rows)))
  return [calc] if n_chunks == 1 else calc._chunks(n_chunks, calc._threads())


def _f_sq(calc, xs, max_rows):
  """F^2 through the same slices as the normal equations, so that the full set of reflections never has its own
  per-operation arrays next to the slices' ones."""
  calc._resolve_n_x4(xs)
  return np.concatenate([c.compute(xs)[1] for c in _slices(calc, len(calc.m), max_rows)])


RAISE_BLAS_THREADS = True    # False: _blas_threads leaves the process's BLAS thread count alone


class _blas_threads(object):
  """with _blas_threads(n): OpenBLAS (the library numpy and scipy call) runs at least n threads inside the block
  (a higher count is left alone), its own count again after it. The libtbx dispatchers export
  OPENBLAS_NUM_THREADS=1 (libtbx/env_config.py), so under libtbx.python every dsyrk / potrf of a refinement ran
  single-threaded; the J^T J build and the damped solve run while the atom pool is idle and take the
  calculator's thread count instead. threadpoolctl when it is installed, else the OpenBLAS libraries
  _openblas_libs finds (loaded ones on Linux, Windows and macOS, and the wheels' copies) through their
  {,scipy_}openblas_{get,set}_num_threads{,64_} symbols; otherwise nothing changes."""

  def __init__(self, n):
    self.n, self.undo = n, None

  def __enter__(self):
    if not self.n or not RAISE_BLAS_THREADS:
      return self
    try:
      from threadpoolctl import threadpool_info, threadpool_limits
      blas = [i["num_threads"] for i in threadpool_info() if i.get("user_api") == "blas"]
      if blas and min(blas) < self.n:
        self.undo = threadpool_limits(limits=self.n, user_api="blas")
      return self
    except ImportError:
      pass
    for lib in _openblas_libs():
      try:
        old = lib.openblas_get_num_threads()
        if old < self.n:
          lib.openblas_set_num_threads(int(self.n))
          self.undo = (self.undo or []) + [(lib, old)]
      except AttributeError:
        continue
    return self

  def __exit__(self, *exc):
    if self.undo is None:
      return False
    if isinstance(self.undo, list):
      for lib, old in self.undo:
        lib.openblas_set_num_threads(old)
    else:
      self.undo.restore_original_limits()
    return False


_openblas_cache = []

# the thread-count entry points under the names the OpenBLAS builds export: plain (distributions, older wheels),
# "scipy_" + "64_" (scipy-openblas64, numpy >= 2 wheels: ILP64 symbol suffix) and "scipy_" alone
# (scipy-openblas32, scipy wheels)
_OPENBLAS_NAMES = [(p + "openblas_get_num_threads" + s, p + "openblas_set_num_threads" + s)
                   for p in ("", "scipy_") for s in ("", "64_", "_64")]


class _openblas_lib(object):
  """One loaded OpenBLAS: openblas_get_num_threads() / openblas_set_num_threads(n) whatever the exported names."""

  def __init__(self, lib, get_name, set_name, path=""):
    self.lib, self.path = lib, path
    self._get, self._set = getattr(lib, get_name), getattr(lib, set_name)

  def openblas_get_num_threads(self):
    return int(self._get())

  def openblas_set_num_threads(self, n):
    self._set(int(n))


def _openblas_wrap(lib, path=""):
  """An _openblas_lib for a ctypes library, or None when it exports no known pair of names."""
  for get_name, set_name in _OPENBLAS_NAMES:
    if hasattr(lib, get_name) and hasattr(lib, set_name):
      return _openblas_lib(lib, get_name, set_name, path)
  return None


def _loaded_library_paths():
  """The shared libraries loaded in this process: /proc/self/maps (Linux), EnumProcessModules (Windows),
  the dyld image list (macOS); [] where none of them works."""
  import sys
  import ctypes
  paths = []
  try:
    if sys.platform.startswith("linux"):
      with open("/proc/self/maps") as f:
        paths = [line.split()[-1] for line in f if "/" in line]
    elif sys.platform == "win32":
      from ctypes import wintypes
      psapi, k32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
      k32.GetCurrentProcess.restype = wintypes.HANDLE
      psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD)]
      k32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
      mods = (wintypes.HMODULE * 4096)()
      need = wintypes.DWORD()
      if psapi.EnumProcessModules(k32.GetCurrentProcess(), mods, ctypes.sizeof(mods), ctypes.byref(need)):
        buf = ctypes.create_unicode_buffer(32768)
        for k in range(min(need.value // ctypes.sizeof(wintypes.HMODULE), len(mods))):
          if k32.GetModuleFileNameW(mods[k], buf, 32768):
            paths.append(buf.value)
    elif sys.platform == "darwin":
      libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
      libc._dyld_image_count.restype = ctypes.c_uint32
      libc._dyld_get_image_name.restype = ctypes.c_char_p
      libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
      paths = [libc._dyld_get_image_name(k).decode("utf-8", "replace") for k in range(libc._dyld_image_count())]
  except Exception:
    paths = []
  return paths


def _openblas_wheel_paths():
  """OpenBLAS libraries shipped inside the numpy and scipy wheels: numpy.libs / scipy.libs (Windows, Linux
  manylinux), numpy/.libs, numpy/.dylibs and scipy/.dylibs (macOS), numpy/_core next to the extension. Whether
  one is loaded does not matter for Windows, where loading a DLL already in the process returns that one, and
  numpy and scipy are imported here."""
  import glob
  out = []
  for name in ("numpy", "scipy"):
    try:
      m = __import__(name)
    except ImportError:
      continue
    d = os.path.dirname(m.__file__)
    for sub in (os.path.join(os.path.dirname(d), name + ".libs"), os.path.join(d, ".libs"),
                os.path.join(d, ".dylibs"), os.path.join(d, "_core"), os.path.join(d, "core")):
      out += glob.glob(os.path.join(sub, "*openblas*"))
  return out


def _openblas_load(path):
  import ctypes
  mode = getattr(os, "RTLD_NOLOAD", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
  try:
    return ctypes.CDLL(path, mode=mode) if mode else ctypes.CDLL(path)
  except OSError:
    return None


def _openblas_libs():
  """The OpenBLAS libraries of this process, found once per process (the list never changes while it runs):
  every loaded library whose file name holds "openblas", then the wheels' copies (_openblas_wheel_paths); each
  wrapped under the names it exports (_OPENBLAS_NAMES), one entry per distinct library file."""
  if not _openblas_cache:
    seen, libs = set(), []
    for p in _loaded_library_paths() + _openblas_wheel_paths():
      base = os.path.basename(p).lower()
      if "openblas" not in base or not (".so" in base or base.endswith(".dll") or base.endswith(".dylib")):
        continue
      key = os.path.normcase(os.path.realpath(p))
      if key in seen:
        continue
      seen.add(key)
      lib = _openblas_load(p)
      w = None if lib is None else _openblas_wrap(lib, p)
      if w is not None:
        libs.append(w)
    _openblas_cache.append(libs)
  return _openblas_cache[0]


def _normal_equations(calc, xs, flags, f_obs_sq, sw, project, max_rows):
  """F^2 and J^T J, J^T r of the weighted residuals r = sw (Fo^2 - Fc^2), J = project(sw dFc^2/dp), with the
  reflections taken max_rows at a time: the design matrix (reflections x parameters) is never held whole.
  One slice is exactly the dense product. The J^T J updates run with the calculator's thread count in BLAS
  (_blas_threads): the atom pool is idle then."""
  calc._resolve_n_x4(xs)
  subs = _slices(calc, len(f_obs_sq), max_rows)
  n_chunks = len(subs)
  try:
    from scipy.linalg.blas import dsyrk
  except ImportError:
    dsyrk = None
  fc_parts, jtj, jtr, a = [], None, None, 0
  ws = {}       # buffers the slices reuse in turn: each slice's d is used up before the next one is computed
  for c in subs:
    b = a + len(c.m)
    c._workspace = ws
    try:
      _, fc_sq, d = c.compute(xs, flags, gradients=True)
    finally:
      c._workspace = None
    d *= sw[a:b, None]      # in place: d is this slice's own
    j = project(d)
    del d
    r = sw[a:b] * (f_obs_sq[a:b] - fc_sq)
    with _blas_threads(calc._threads()):
      if n_chunks == 1:
        jtj = j.T.dot(j)
      elif dsyrk is not None:
        # in place, upper triangle: no second n x n matrix per slice
        if jtj is None:
          jtj = np.zeros((j.shape[1], j.shape[1]), order="F")
        jtj = dsyrk(1.0, j, beta=1.0, c=jtj, trans=1, lower=0, overwrite_c=1)
      else:
        jtj = j.T.dot(j) if jtj is None else jtj + j.T.dot(j)
    jtr = j.T.dot(r) if jtr is None else jtr + j.T.dot(r)     # a matrix-vector product: BLAS as the process has it
    fc_parts.append(fc_sq)
    a = b
  if n_chunks > 1 and dsyrk is not None:
    _mirror_upper(jtj)
  return np.concatenate(fc_parts), jtj, jtr


def _mirror_upper(a, block=1024):
  """The lower triangle of a square matrix from its upper one, in place and a block of rows at a time: no n x n
  temporaries."""
  for k in range(0, a.shape[0], block):
    a[k:k + block, :k] = a[:k, k:k + block].T
    blk = a[k:k + block, k:k + block]
    low = np.tril_indices(blk.shape[0], -1)
    blk[low] = blk.T[low]
  return a


def _rows_per_slice(xs, flags, max_memory_mb):
  """How many reflections a slice of the normal equations may hold. Per reflection: 16 bytes for every column of
  the atoms' complex gradient blocks (all of an atom's blocks, refined or not), 8 for every refined parameter's
  real dF^2/dp and 8 for its projection, 5 % on top for the node buffers and block temporaries. The compiled path
  keeps the gradient blocks and dF^2/dp in the pass's workspace while the projection is made, so all three are
  alive at once (33 bytes per parameter on the large test problem)."""
  n_p = len(get_parameters(xs, flags))
  n_c = sum(3 + (1 if at.u_star is None else 6) + 1 + 6 * at.n_harmonics + 2 * len(at.occ_sin) + 12 * len(at.u_sin)
            + (2 if at.crenel else 0) + 3 * len(at.legendre) + 3 * len(at.xharm) + 6 * len(at.u_xharm)
            for at in xs.atoms)
  return max(256, int(max_memory_mb * 2 ** 20 // (1.05 * (16 * n_c + 16 * max(n_p, 1)))))


def refine(xs, h, m, f_obs_sq, sigmas, flags=None, cycles=10, n_x4="auto", damping=1e-3, log=None, verbose=True,
           table=None, radiation="xray", average=None, max_memory_mb=512, commensurate=None, twins=None):
  """Levenberg-Marquardt on F^2 with weights 1/sigma^2. Returns per-cycle dicts (wR2, R1 by |m|, max shift).
  table: a form_factor_table for aspherical form factors (IAM without); radiation as for calculator.
  commensurate: (t0, N) refines the commensurate structure of that section (commensurate_section): F^2 and
  gradients of the expanded images, the gradients folded back onto the parameters of xs (_commensurate_fold).
  Crenel centres and widths, and extinction, cannot be refined then (a finite sum has no gradient with respect
  to the crenel; the site correction of commensurate_compute does not carry the extinction factor). One slice.
  twins: [(3x3 matrix, mass fraction)] (mscif.read's data["twins"]): F^2 = sum_k f_k F_k^2 (twinned_calculator),
  the fractions refined under sum f_k = 1 (the first is 1 - the others); the list is updated in place. One
  slice; not with average or commensurate.
  n_x4: x4 nodes per atom, or "auto" (the default: auto_n_x4, within 1e-10 of 128 nodes, raised during the
  refinement if the amplitudes grow); 32 was the default before.

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

  max_memory_mb: what one slice of the normal equations may take; the reflections are taken in as many slices as
  that needs (_rows_per_slice), J^T J and J^T r summed over them. A problem that fits uses one slice.
  ponytail: restraints act on the average structure only, not section by section along t.
  """
  flags = _flags(flags)
  link = None
  if average is not None:
    flags.update(site=True, adp=True, occupancy=False)   # occupancies stay with the average's own refinement
    link = _average_link(xs, flags, average)
  else:
    zb = _reduction_blocks(xs, flags)   # superspace site symmetry: the free directions, block by block
  if flags["modulation"]:
    p_before = get_parameters(xs, flags)
    seed_amplitudes(xs)
    p = get_parameters(xs, flags)
    if link is None:
      set_parameters(xs, flags, p_before + _blocks_times(zb, _times_blocks((p - p_before)[None, :], zb)[0]))
    else:
      wi = link.wave_idx
      p[wi] = p_before[wi] + link.zw.dot(link.zw.T.dot(p[wi] - p_before[wi]))
      set_parameters(xs, flags, p)
      link.sync()
  f_obs_sq = np.asarray(f_obs_sq, dtype=float)
  calc = calculator(h, m, n_x4, table, radiation=radiation)
  if commensurate is not None:
    if flags["crenel"] or flags["extinction"]:
      raise ValueError("refine: crenel centres and widths, and extinction, cannot be refined in a commensurate "
                       "section")
    t0, n_sec = commensurate
    model = lambda: commensurate_section(xs, t0, n_sec)
  else:
    model = lambda: xs
  tc, nt = None, 0
  if twins is not None:
    if average is not None or commensurate is not None:
      raise NotImplementedError("refine: twins with average or commensurate")
    tc = twinned_calculator(h, m, xs.q, twins, n_x4=n_x4, table=table, radiation=radiation)
    nt = len(twins) - 1
    zb = zb + [np.eye(nt)] if nt else zb

  def get_p():
    p = get_parameters(xs, flags)
    return np.concatenate([p, tc.fractions[1:]]) if nt else p

  def set_p(p):
    if nt:
      tc.fractions[1:] = p[-nt:]
      tc.fractions[0] = 1 - np.sum(tc.fractions[1:])
      p = p[:-nt]
    set_parameters(xs, flags, p)

  def f_sq_now():
    return tc.compute(xs)[1] if tc is not None else _f_sq(calc, model(), max_rows)
  w = 1.0 / np.maximum(np.asarray(sigmas, dtype=float), 1e-12) ** 2
  sw = np.sqrt(w)
  history = []
  lam = damping
  chi = None    # from the first gradient pass: the same F^2 a separate pass before the loop would give
  norm = None
  project0 = (lambda d: _times_blocks(d, zb)) if link is None else link.jacobian
  max_rows = _rows_per_slice(model(), flags, max_memory_mb)
  for cycle in range(cycles):
    if commensurate is None and tc is None:
      fc_sq, jtj, jtr = _normal_equations(calc, xs, flags, f_obs_sq, sw, project0, max_rows)
    else:
      # one dense slice: commensurate structures are small, and commensurate_compute needs all rows' F at once
      if tc is not None:
        _, fc_sq, d = tc.compute(xs, flags, gradients=True)
      else:
        _, fc_sq, d = commensurate_compute(calc, xs, t0, n_sec, flags, gradients=True)
      j = project0(sw[:, None] * d)
      del d
      with _blas_threads(calc._threads()):
        jtj, jtr = j.T.dot(j), j.T.dot(sw * (f_obs_sq - fc_sq))
      del j
    if chi is None:
      chi = float(np.sum(w * (f_obs_sq - fc_sq) ** 2))   # + the weighted restraints once there are any
    r = sw * (f_obs_sq - fc_sq)
    n_free = jtj.shape[0]
    rest = None if link is None else link.restraints(n_free)
    if rest is not None:
      if norm is None:
        norm = chi / max(len(r) - n_free, 1)
        chi += norm * float(np.sum(rest[2] * rest[1] ** 2))
      rw = norm * rest[2]
      jtj += rest[0].T.dot(rw[:, None] * rest[0])
      jtr -= rest[0].T.dot(rw * rest[1])
    p0 = get_p()
    jtj_diag = np.diag(jtj).copy()
    diag = jtj_diag.copy()
    diag[diag <= 1e-12 * max(diag.max(), 1e-300)] = max(diag.max(), 1.0)
    accepted, step = False, np.zeros_like(p0)
    # J^T J stays in the upper triangle, every damping is factorised in the lower one: one n x n matrix in all.
    # J^T J is symmetric, so a C-ordered one is its own Fortran-ordered transpose (no copy)
    if not jtj.flags.f_contiguous:
      jtj = jtj.T
    for _ in range(16):
      with _blas_threads(calc._threads()):
        x = _lm_factor_solve(jtj, jtj_diag, lam * diag, jtr)
      if x is None:   # no scipy, or not positive definite: a full copy, rebuilt from the upper triangle
        a_mat = _mirror_upper(np.array(jtj))
        a_mat[np.diag_indices_from(a_mat)] = jtj_diag + lam * diag
        try:
          x = np.linalg.solve(a_mat, jtr)
        except np.linalg.LinAlgError:
          x = np.linalg.lstsq(a_mat, jtr, rcond=None)[0]
        del a_mat
      if link is None:
        set_p(p0 + _blocks_times(zb, x))
      else:
        link.apply(p0, x)
      step = get_p() - p0
      with np.errstate(over="ignore", invalid="ignore"):
        fc_new = f_sq_now()
      chi_new = float(np.sum(w * (f_obs_sq - fc_new) ** 2))
      if rest is not None:
        r_new = link.restraints(n_free)
        chi_new += norm * float(np.sum(r_new[2] * r_new[1] ** 2))
      if np.isfinite(chi_new) and chi_new <= chi:
        chi, lam, accepted = chi_new, max(lam / 3, 1e-9), True
        break
      if link is not None:
        link.undo(p0, x)
      lam *= 5
    if not accepted:
      set_p(p0)
      fc_new = f_sq_now()
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
  if tc is not None:
    twins[:] = [(t, float(f)) for (t, _), f in zip(twins, tc.fractions)]
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
