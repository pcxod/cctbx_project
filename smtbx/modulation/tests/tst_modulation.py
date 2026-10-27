"""smtbx.modulation: superspace closure, F against cctbx and the Bessel form, gradients, settings, absences, recovery."""
from __future__ import absolute_import, division, print_function

import math

import os
import numpy as np

from cctbx import crystal, miller, sgtbx, uctbx, xray
from cctbx.array_family import flex
from smtbx import modulation as mod


def p21c():
  return sgtbx.space_group_info("P 1 21/c 1").group()


def model(n_harmonics=1, amp=0.0, seed=1, ssg=None):
  rng = np.random.RandomState(seed)
  uc = uctbx.unit_cell((7.1, 5.3, 9.2, 90, 101.3, 90))
  ssg = ssg or mod.superspace_group(p21c(), (0.31, 0.0, 0.23))
  atoms = []
  for i, (t, xyz) in enumerate([("Cu", (0.12, 0.21, 0.33)), ("O", (0.31, 0.07, 0.15)),
                                ("N", (0.43, 0.38, 0.27)), ("C", (0.21, 0.44, 0.08)), ("C", (0.05, 0.12, 0.41))]):
    a = amp * rng.uniform(-1, 1, (n_harmonics, 3))
    b = amp * rng.uniform(-1, 1, (n_harmonics, 3))
    atoms.append(mod.modulated_atom("%s%d" % (t, i), t, xyz, 0.012 + 0.002 * i, 1.0, n_harmonics, a, b))
  return mod.modulated_structure(uc, ssg, atoms, scale=1.0)


def reflections(xs, d_min=0.9, m_max=1):
  """Half sphere of (h, m) with |h + m q| < 1/d_min, no absences removed."""
  cs = crystal.symmetry(xs.unit_cell, "P -1")
  h0 = np.array(miller.build_set(cs, anomalous_flag=False, d_min=d_min * 0.8).indices())
  g6 = xs.unit_cell.reciprocal_metrical_matrix()
  g = np.array([[g6[0], g6[3], g6[4]], [g6[3], g6[1], g6[5]], [g6[4], g6[5], g6[2]]])
  hs, ms = [], []
  for m in range(-m_max, m_max + 1):
    s = h0 + m * xs.q
    keep = np.einsum("ni,ij,nj->n", s, g, s) < 1 / d_min ** 2
    hs.append(h0[keep])
    ms.append(np.full(keep.sum(), m))
  return np.concatenate(hs), np.concatenate(ms)


def test_superspace_closure():
  q = (0.3, 0.0, 0.2)
  g = mod.superspace_group(p21c(), q)
  assert len(g.ops) == 4
  eps = sorted(op.eps for op in g.ops)
  assert eps == [-1, -1, 1, 1], eps
  try:
    mod.superspace_group(p21c(), (0.3, 0.1, 0.2))
  except mod.IncompatibleQ:
    pass
  else:
    raise AssertionError("q off the 2-fold's eigenspaces must be refused")
  assert mod.compatible_subgroup(p21c(), (0.3, 0.1, 0.2)).order_z() == 2   # only the inversion survives
  cands = mod.candidate_superspace_groups(p21c(), q)
  labels = sorted(c.label() for c in cands)
  assert len(cands) == 2, labels                  # P2_1/c(a0g)0 and (a0g)s: glide tau 0 or 1/2
  print("  closure: 4 ops, eps %s; candidates %s; incompatible q refused" % (eps, labels))


def test_main_reflections_match_cctbx():
  xs = model(amp=0.0)
  h, m = reflections(xs)
  sel = m == 0
  f, _, _ = mod.calculator(h[sel], m[sel]).compute(xs)
  sc = flex.xray_scatterer([xray.scatterer(a.label, tuple(a.site), a.u_iso, a.occupancy, a.scattering_type)
                            for a in xs.atoms])
  xrs = xray.structure(crystal_symmetry=crystal.symmetry(xs.unit_cell, space_group=xs.ssg.basic), scatterers=sc)
  xrs.scattering_type_registry(table="wk1995")
  ms = miller.set(xrs.crystal_symmetry(), flex.miller_index([tuple(int(v) for v in r) for r in h[sel]]),
                  anomalous_flag=False)
  fc = ms.structure_factors_from_scatterers(xrs, algorithm="direct").f_calc().data().as_numpy_array()
  err = np.max(np.abs(f - fc)) / np.max(np.abs(fc))
  assert err < 1e-6, err
  fsat, _, _ = mod.calculator(h[~sel], m[~sel]).compute(xs)
  assert np.max(np.abs(fsat)) < 1e-9 * np.max(np.abs(fc))
  print("  zero amplitudes: main reflections = cctbx direct to %.1e, satellites 0" % err)


def test_bessel_single_harmonic():
  from scitbx.math import bessel_J
  uc = uctbx.unit_cell((6, 7, 8, 90, 90, 90))
  ssg = mod.superspace_group(sgtbx.space_group(), (0.27, 0.0, 0.0))
  a = np.array([[0.03, -0.01, 0.02]])
  at = mod.modulated_atom("Fe", "Fe", (0.1, 0.2, 0.3), 0.0, 1.0, 1, a, np.zeros((1, 3)))
  xs = mod.modulated_structure(uc, ssg, [at])
  h = np.array([[1, 2, 3], [2, 0, 1], [3, 1, 2], [0, 2, 4]])
  worst = 0.0
  for m in (-2, -1, 0, 1, 2):
    f, _, _ = mod.calculator(h, np.full(len(h), m), n_x4=64).compute(xs)
    s = h + m * ssg.q
    stol = np.sqrt((s ** 2 / np.array([36., 49., 64.])).sum(axis=1)) / 2
    ff = mod._form_factors(["Fe"], stol)["Fe"]
    jn = np.array([bessel_J(-m, 2 * math.pi * v) for v in s.dot(a[0])])
    ref = ff * np.exp(2j * math.pi * h.dot(at.site)) * jn
    worst = max(worst, np.max(np.abs(f - ref)) / np.max(ff))
  assert worst < 1e-10, worst
  print("  one harmonic in P1 = f exp(2 pi i h.x) J_-m(2 pi S.A), m = -2..2, to %.1e" % worst)


def test_gradients_finite_difference():
  xs = model(n_harmonics=2, amp=0.02)
  h, m = reflections(xs, d_min=1.4)
  flags = {"site": True, "u_iso": True, "modulation": True, "scale": True}
  calc = mod.calculator(h, m)
  _, _, d = calc.compute(xs, flags, gradients=True)
  p0 = mod.get_parameters(xs, flags)
  worst = 0.0
  for k in range(len(p0)):
    step = 1e-6
    p = p0.copy(); p[k] += step
    mod.set_parameters(xs, flags, p)
    _, fp, _ = calc.compute(xs)
    p[k] -= 2 * step
    mod.set_parameters(xs, flags, p)
    _, fm, _ = calc.compute(xs)
    mod.set_parameters(xs, flags, p0)
    num = (fp - fm) / (2 * step)
    scale = max(np.max(np.abs(num)), np.max(np.abs(d[:, k])), 1e-12)
    worst = max(worst, np.max(np.abs(num - d[:, k])) / scale)
  assert worst < 1e-5, worst
  print("  analytic gradients = finite differences, %d parameters, worst relative %.1e" % (len(p0), worst))


def test_gradients_aniso_occ_adp_crenel():
  """Gradients of the u_star, occupancy, occupational-wave and ADP-wave blocks, with a crenel atom carrying
  Legendre terms and f', f'' on, against central differences."""
  from cctbx import adptbx
  xs = model(n_harmonics=1, amp=0.02)
  rng = np.random.RandomState(5)
  for i, at in enumerate(xs.atoms):
    at.u_star = np.array(adptbx.u_iso_as_u_star(xs.unit_cell, at.u_iso)) * (1 + 0.2 * rng.uniform(-1, 1, 6) * [1, 1, 1, 0, 0, 0])
    at.u_star[3:] = 0.0002 * rng.uniform(-1, 1, 3)
    if i % 2 == 0:
      at.occ_sin, at.occ_cos = 0.1 * rng.uniform(-1, 1, 1), 0.1 * rng.uniform(-1, 1, 1)
    at.u_sin, at.u_cos = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (1, 6)), at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (1, 6))
  crenel_atom = xs.atoms[3]
  crenel_atom.crenel = (0.3, 0.6)
  crenel_atom.legendre = 0.01 * rng.uniform(-1, 1, (2, 3))
  xs.dispersion = {"Cu": (0.3, 1.2), "O": (0.01, 0.006)}
  h, m = reflections(xs, d_min=1.5)
  flags = {"site": True, "adp": True, "occupancy": True, "modulation": True, "occ_waves": True,
           "adp_waves": True, "scale": True}
  calc = mod.calculator(h, m)
  _, _, d = calc.compute(xs, flags, gradients=True)
  p0 = mod.get_parameters(xs, flags)
  worst = 0.0
  for k in range(len(p0)):
    step = 1e-6 * max(1.0, abs(p0[k]))
    p = p0.copy(); p[k] += step
    mod.set_parameters(xs, flags, p)
    _, fp, _ = calc.compute(xs)
    p[k] -= 2 * step
    mod.set_parameters(xs, flags, p)
    _, fm, _ = calc.compute(xs)
    mod.set_parameters(xs, flags, p0)
    num = (fp - fm) / (2 * step)
    scale = max(np.max(np.abs(num)), np.max(np.abs(d[:, k])), 1e-12)
    worst = max(worst, np.max(np.abs(num - d[:, k])) / scale)
  assert worst < 1e-5, worst
  print("  u_star, occupancy, occupancy/ADP waves, crenel + Legendre, f'/f'': %d gradients = differences to %.1e"
        % (len(p0), worst))


def test_mscif_against_jana():
  """F^2 against JANA's own F^2 calc for COD 2108430 (C2/c(0b0)s0: occupational + ADP waves, Uaniso, f''),
  when the sample is on this machine."""
  import os
  from smtbx.modulation import mscif
  root = r"E:\olex2-samples\cod_2108430\2108430"
  if not os.path.exists(root + ".cif"):
    print("  msCIF vs JANA: sample not here, skipped")
    return
  xs, d = mscif.read(root + ".cif", root + ".fcf")
  _, fc, _ = mod.calculator(d["h"], d["m"], 64).compute(xs)
  fj = d["f_sq_calc"]
  k = np.sum(fc * fj) / np.sum(fc ** 2)
  r = np.sum(np.abs(k * fc - fj)) / np.sum(fj)
  assert r < 0.003, r
  print("  COD 2108430 against JANA's F^2 calc: R %.4f over %d hklm" % (r, len(fj)))


def test_hklm_files():
  """read_hklm takes fixed 4I4,2F8 and free columns with a header line and stops at 0 0 0 0; write_main_hkl keeps
  m = 0 only."""
  import os, tempfile
  d = tempfile.mkdtemp()
  fixed, free, main = (os.path.join(d, n) for n in ("a.hklm", "b.hklm", "c.hkl"))
  rows = [(1, 0, 0, 0, 123.45, 4.5), (1, 0, 0, 1, 12.3, 1.1), (-2, 3, -4, -1, 99999.5, 2.0), (0, 0, 0, 1, 7.0, 0.5)]
  with open(fixed, "w") as f:
    f.write("# header\n" + "".join("%4d%4d%4d%4d%8.2f%8.2f   1\n" % r for r in rows) + "   0   0   0   0    0.00    0.00\n")
  with open(free, "w") as f:
    f.write("".join("%d %d %d %d %g %g\n" % r for r in rows) + "0 0 0 0 0 0\nnot data\n")
  for p in (fixed, free):
    h, m, i, s = mod.read_hklm(p)
    assert h.tolist() == [list(r[:3]) for r in rows] and m.tolist() == [r[3] for r in rows], p
    assert np.allclose(i, [r[4] for r in rows], atol=0.01), (p, i)
  assert mod.write_main_hkl(fixed, main) == 1
  assert open(main).read().splitlines()[0] == "   1   0   0  123.45    4.50"
  print("  hklm files: fixed and free columns, header line, (0,0,0,1) kept as a satellite, m = 0 export")


def test_reflection_statistics():
  """Split statistics: a complete set measured twice through symmetry equivalents has completeness 1 and R_int 0
  for m = 0 and |m| = 1; perturbed duplicates give the R_int of the perturbation; half the orbits, half the
  completeness; predicted absences do not count."""
  cands = mod.candidate_superspace_groups(p21c(), (0.31, 0.0, 0.23))
  ssg = [c for c in cands if any(op.eps == 1 and op.tau for op in c.ops)][0]
  xs = model(amp=0.02, ssg=ssg)
  h, m = reflections(xs, d_min=1.0)
  keep = ~mod._sys_absent(ssg, h.astype(float), m)
  keys = mod._orbit_keys(ssg, h.astype(float), m)
  _, first = np.unique(keys, return_index=True)
  first = first[keep[first]]
  hu, mu = h[first], m[first]
  i_u = 100.0 + np.arange(len(mu)) % 7
  op = [o for o in ssg.ops if o.eps == -1][0]                      # a symmetry-equivalent second measurement
  h2 = np.rint(hu.dot(op.r) + np.outer(mu, op.m)).astype(int)
  m2 = op.eps * mu
  hh, mm = np.vstack([hu, h2]), np.concatenate([mu, m2])
  st = mod.reflection_statistics(ssg, xs.unit_cell, hh, mm, np.concatenate([i_u, i_u]), np.ones(2 * len(mu)))
  for k in (0, 1):
    assert abs(st[k]["completeness"] - 1) < 1e-9 and st[k]["r_int"] < 1e-12, (k, st[k])
  st2 = mod.reflection_statistics(ssg, xs.unit_cell, hh, mm, np.concatenate([i_u * 0.9, i_u * 1.1]), np.ones(2 * len(mu)))
  assert abs(st2["sat"]["r_int"] - 0.1) < 1e-9, st2["sat"]
  half = (np.arange(len(mu)) % 2) == 0
  st3 = mod.reflection_statistics(ssg, xs.unit_cell, hu[half], mu[half], i_u[half], np.ones(half.sum()))
  assert abs(st3[1]["completeness"] - 0.5) < 0.02, st3[1]
  print("  split statistics: complete set 100 %% / R_int 0 for m=0 and |m|=1 (%d + %d unique), +-10 %% duplicates "
        "R_int 0.100, every other orbit %.3f" % (st[0]["unique"], st[1]["unique"], st3[1]["completeness"]))


def test_change_of_basis_invariant():
  xs = model(n_harmonics=2, amp=0.02)
  h, m = reflections(xs, d_min=1.2)
  _, fsq0, _ = mod.calculator(h, m).compute(xs)
  worst = 0.0
  # x4 shift; 3D origin shift; a <-> c swap (b -> -b keeps the hand) with an origin shift and an x4 shift
  for p, shift, d4 in ((None, None, 0.13), (None, (0.25, 0.125, 0.5), 0.0),
                       ([[0, 0, 1], [0, -1, 0], [1, 0, 0]], (0.5, 0.0, 0.25), 0.07)):
    cob = mod.change_of_basis(p, shift, d4)
    xs2 = mod.modulated_structure(cob.unit_cell(xs.unit_cell), cob.superspace_group(xs.ssg),
                                  [cob.atom(a) for a in xs.atoms])
    _, fsq1, _ = mod.calculator(cob.hkl(h), m).compute(xs2)
    worst = max(worst, np.max(np.abs(fsq0 - fsq1)) / fsq0.max())
    back = cob.inverse()
    a0 = back.atom(cob.atom(xs.atoms[0]))
    assert np.allclose(a0.site, xs.atoms[0].site) and np.allclose(a0.a, xs.atoms[0].a)
  assert worst < 1e-8, worst
  print("  x4 shift, origin shift and a<->c swap leave |F|^2 unchanged to %.1e; inverse() round-trips" % worst)


def test_rank_by_absences():
  cands = mod.candidate_superspace_groups(p21c(), (0.31, 0.0, 0.23))
  truth_ssg = [c for c in cands if any(op.eps == 1 and op.tau for op in c.ops)][0]
  truth = model(amp=0.02, ssg=truth_ssg)
  h, m = reflections(truth, d_min=0.9)
  _, fsq, _ = mod.calculator(h, m).compute(truth)
  sig = 0.01 * fsq + 1e-3 * fsq.max()
  ranked = mod.rank_by_absences(cands, h, m, fsq, sig)
  assert ranked[0][2].label() == truth_ssg.label(), [(v, n, g.label()) for v, n, g in ranked]
  print("  absences pick %s (violations %d, predicted absences %d); runner-up violations %d"
        % (truth_ssg.label(), ranked[0][0], ranked[0][1], ranked[1][0]))


def test_recovery():
  truth = model(amp=0.012, seed=7)
  h, m = reflections(truth, d_min=0.8)
  _, fsq, _ = mod.calculator(h, m).compute(truth)
  rng = np.random.RandomState(3)
  sig = 0.02 * fsq + 2e-5 * fsq.max()
  fobs = fsq + rng.normal(0, 1, fsq.size) * sig
  start = model(amp=0.0)
  for at, tr in zip(start.atoms, truth.atoms):
    at.site = tr.site + rng.normal(0, 0.003, 3)
  start.scale = 0.9
  hist = mod.refine(start, h, m, fobs, sig, cycles=30, verbose=False)
  r = hist[-1]["R1"]
  floor = mod.r_factors(fobs, fsq, m)                 # the true model against the same noisy data
  # an x4 origin shift of 1/2 (all first harmonics change sign) is the same structure here
  err = min(max(max(np.max(np.abs(s * a.a - t.a)), np.max(np.abs(s * a.b - t.b)))
                for a, t in zip(start.atoms, truth.atoms)) for s in (1, -1))
  print("  recovery from zero amplitudes, %d cycles: R1(m=1) %.4f (true model %.4f), R1(m=0) %.4f (%.4f);"
        " worst |dA,dB| %.1e of amplitudes up to %.1e"
        % (len(hist), r["m=1"], floor["m=1"], r["m=0"], floor["m=0"], err, 0.012))
  assert r["m=1"] < floor["m=1"] * 1.02 + 1e-3, (r, floor)
  assert err < 0.003, err


class _riding_average(object):
  """refine(average=): atoms 0-3 free (site, u_iso); atom 4 rides on atom 3 (site + delta, 1.2 U, waves moved by
  q.delta along x4); one restraint holds atom 1's x at `target` (sigma 1e-4)."""

  def __init__(self, xs, delta, target):
    self.xs, self.delta, self.target = xs, np.asarray(delta), target
    self.x = np.concatenate([np.append(xs.atoms[i].site, xs.atoms[i].u_iso) for i in range(4)])
    self.follow = {4: (3, float(xs.q.dot(self.delta)))}
    self.shift(np.zeros(16))

  def shift(self, x):
    self.x = self.x + x
    for i in range(4):
      self.xs.atoms[i].site, self.xs.atoms[i].u_iso = self.x[4 * i:4 * i + 3].copy(), self.x[4 * i + 3]
    self.xs.atoms[4].site, self.xs.atoms[4].u_iso = self.x[12:15] + self.delta, 1.2 * self.x[15]

  def jacobian(self):
    jac = np.vstack([np.eye(16), np.zeros((4, 16))])
    jac[16:19, 12:15], jac[19, 15] = np.eye(3), 1.2
    return jac

  def restraints(self):
    design = np.zeros((1, 16))
    design[0, 4] = -1.0
    return design, [self.target - self.x[4]], [1e8]


def test_average_constraints_and_restraints():
  delta = (0.10, 0.12, 0.09)
  truth = model(amp=0.012, seed=7)
  ride = _riding_average(truth, delta, truth.atoms[1].site[0])
  mod._average_link(truth, mod._flags(None), ride).sync()   # the truth obeys the constraints too
  h, m = reflections(truth, d_min=0.8)
  _, fsq, _ = mod.calculator(h, m).compute(truth)
  rng = np.random.RandomState(3)
  sig = 0.02 * fsq + 2e-5 * fsq.max()
  fobs = fsq + rng.normal(0, 1, fsq.size) * sig
  start = model(amp=0.0)
  for at, tr in zip(start.atoms, truth.atoms):
    at.site = tr.site + rng.normal(0, 0.003, 3)
  start.scale = 0.9
  target = truth.atoms[1].site[0] + 0.002        # off the data's optimum: only the restraint puts it here
  avg = _riding_average(start, delta, target)
  hist = mod.refine(start, h, m, fobs, sig, cycles=30, verbose=False, average=avg)
  r, floor = hist[-1]["R1"], mod.r_factors(fobs, fsq, m)
  a3, a4 = start.atoms[3], start.atoms[4]
  c, s = math.cos(2 * math.pi * avg.follow[4][1]), math.sin(2 * math.pi * avg.follow[4][1])
  assert np.allclose(a4.site - a3.site, delta) and abs(a4.u_iso - 1.2 * a3.u_iso) < 1e-12
  assert np.allclose(a4.a, c * a3.a + s * a3.b) and np.allclose(a4.b, -s * a3.a + c * a3.b)
  assert abs(start.atoms[1].site[0] - target) < 2e-4, (start.atoms[1].site[0], target)
  assert r["m=1"] < floor["m=1"] * 1.05 + 2e-3, (r, floor)
  print("  riding atom keeps site, U and waves on its parent; restrained x off by %.1e; R1(m=1) %.4f (true %.4f)"
        % (abs(start.atoms[1].site[0] - target), r["m=1"], floor["m=1"]))


def _write_tsc(path, ids, idx, f, binary):
  """A NoSpherA2 id table (ids: xray.scatterer_id_big): tscb (header size, header, id count, 16-byte ids, row count,
  rows) or text."""
  if binary:
    head = b"SCATTERER_IDS"
    rec = np.zeros(len(idx), np.dtype([("h", "<i4", 3), ("f", "<c16", len(ids))]))
    rec["h"], rec["f"] = idx, f
    with open(path, "wb") as o:
      o.write(np.int32(len(head)).tobytes() + head + np.int32(len(ids)).tobytes() + b"".join(i.to_bytes() for i in ids)
              + np.int32(len(idx)).tobytes() + rec.tobytes())
  else:
    with open(path, "w") as o:
      o.write("TITLE: t\nSYMM: expanded\nSCATTERER_IDS: %s\nDATA:\n" % " ".join(i.to_hex_string() for i in ids))
      for k, row in zip(idx, f):
        o.write("%d %d %d %s\n" % (k[0], k[1], k[2], " ".join("%.17e,%.17e" % (v.real, v.imag) for v in row)))


def test_aspherical_table():
  """A table of f0(|S|) exp(2 pi i S.d), a density offset by d from its nucleus (d perpendicular to q, so the wave
  argument q.x stays), reproduces F and gradients of the IAM structure with the sites moved by d: f_j(S R) per
  operation, the +-S folding and the scaled rows, from a binary and a text table whose scatterer ids sit 0.002
  off the sites (refinement moves sites off their ids). f', f'' stay at the nucleus, so they are checked with
  d = 0."""
  import os, tempfile
  from cctbx.eltbx import tiny_pse
  tmp = tempfile.mkdtemp()
  worst = 0.0
  for shift, disp in ((0.01, {}), (0.0, {"Cu": (0.3, 1.2)})):
    xs = model(n_harmonics=1, amp=0.02)
    xs.dispersion = disp
    h, m = reflections(xs, d_min=1.2)
    d = dict((a.label, np.array([0.0, shift * (k + 1), 0.0])) for k, a in enumerate(xs.atoms))
    shifted = model(n_harmonics=1, amp=0.02)
    shifted.dispersion = disp
    for a in shifted.atoms:
      a.site = a.site + d[a.label]
    f_ref, _, d_ref = mod.calculator(h, m).compute(shifted, gradients=True)
    rows = mod.calculator(h, m).table_indices(xs.ssg)
    s = rows / float(mod.TABLE_SCALE)
    g6 = xs.unit_cell.reciprocal_metrical_matrix()
    g = np.array([[g6[0], g6[3], g6[4]], [g6[3], g6[1], g6[5]], [g6[4], g6[5], g6[2]]])
    f0 = mod._form_factors([a.scattering_type for a in xs.atoms], np.sqrt(np.einsum("ni,ij,nj->n", s, g, s)) / 2)
    f = np.array([f0[a.scattering_type] * np.exp(2j * math.pi * s.dot(d[a.label])) for a in xs.atoms]).T
    folded = len(rows) < len(np.unique(np.vstack([np.rint(mod.TABLE_SCALE * (h + np.outer(m, xs.q)).dot(op.r))
                                                  for op in xs.ssg.ops]), axis=0))
    assert folded
    ids = [xray.scatterer_id_big(a.site[0] + 0.002, a.site[1] - 0.002, a.site[2] + 0.002, 0,
                                 tiny_pse.table(a.scattering_type).atomic_number(), 0) for a in xs.atoms]
    assert np.allclose(mod._id_from_hex(ids[1].to_hex_string())[:3], ids[1].get_crd(), atol=1e-9)
    for binary in (True, False):
      path = os.path.join(tmp, "t.tscm")
      _write_tsc(path, ids, rows, f, binary)
      table = mod.form_factor_table(path)
      f_tab, _, d_tab = mod.calculator(h, m, table=table).compute(xs, gradients=True)
      worst = max(worst, np.max(np.abs(f_tab - f_ref)) / np.max(np.abs(f_ref)),
                  np.max(np.abs(d_tab - d_ref)) / np.max(np.abs(d_ref)))
  assert worst < 1e-9, worst
  try:
    table.lookup(np.array([[0.5, 0.5, 0.5]]), [0])
  except ValueError:
    pass
  else:
    raise AssertionError("a row the table lacks must be refused")
  with open(path, "w") as o:
    o.write("TITLE: t\nSCATTERERS: Cu0\nDATA:\n1 0 0 1.0,0.0\n")
  try:
    mod.form_factor_table(path)
  except ValueError:
    pass
  else:
    raise AssertionError("a label table must be refused")
  print("  aspherical table (%d rows, +-S folded): F and %d gradients = IAM with moved sites to %.1e, tscb and text"
        % (len(rows), d_ref.shape[1], worst))


def test_electron():
  """Kinematical electron diffraction: the main reflections of an unmodulated model equal cctbx's direct structure
  factors with its "electron" table (f' and f'' are not applied to electrons); gradients = central differences."""
  xs = model(amp=0.0)
  xs.dispersion = {"Cu": (0.3, 1.2)}
  h, m = reflections(xs)
  sel = m == 0
  f, _, _ = mod.calculator(h[sel], m[sel], radiation="electron").compute(xs)
  sc = flex.xray_scatterer([xray.scatterer(a.label, tuple(a.site), a.u_iso, a.occupancy, a.scattering_type)
                            for a in xs.atoms])
  xrs = xray.structure(crystal_symmetry=crystal.symmetry(xs.unit_cell, space_group=xs.ssg.basic), scatterers=sc)
  xrs.scattering_type_registry(table="electron")
  ms = miller.set(xrs.crystal_symmetry(), flex.miller_index([tuple(int(v) for v in r) for r in h[sel]]),
                  anomalous_flag=False)
  fc = ms.structure_factors_from_scatterers(xrs, algorithm="direct").f_calc().data().as_numpy_array()
  err = np.max(np.abs(f - fc)) / np.max(np.abs(fc))
  assert err < 1e-6, err
  xs = model(n_harmonics=1, amp=0.02)
  h, m = reflections(xs, d_min=1.6)
  flags = {"site": True, "u_iso": True, "modulation": True, "scale": True}
  calc = mod.calculator(h, m, radiation="electron")
  _, _, d = calc.compute(xs, flags, gradients=True)
  p0 = mod.get_parameters(xs, flags)
  worst = 0.0
  for k in range(len(p0)):
    p = p0.copy(); p[k] += 1e-6
    mod.set_parameters(xs, flags, p)
    _, fp, _ = calc.compute(xs)
    p[k] -= 2e-6
    mod.set_parameters(xs, flags, p)
    _, fm, _ = calc.compute(xs)
    mod.set_parameters(xs, flags, p0)
    num = (fp - fm) / 2e-6
    worst = max(worst, np.max(np.abs(num - d[:, k])) / max(np.max(np.abs(num)), np.max(np.abs(d[:, k])), 1e-12))
  assert worst < 1e-5, worst
  print("  electrons: main reflections = cctbx direct (electron table) to %.1e, f'' ignored; %d gradients = "
        "differences to %.1e" % (err, len(p0), worst))


def test_gradients_all_wave_blocks():
  """Every wave block against central differences, parameter by parameter, relative error <= 1e-6: occupancy
  and ADP waves; crenel c and w on atoms carrying Fourier displacements, occupancy and ADP waves, Legendre
  terms, x-harmonics and (one) a sawtooth; Legendre and displacive and ADP x-harmonic coefficients."""
  from cctbx import adptbx
  xs = model(n_harmonics=2, amp=0.02)
  rng = np.random.RandomState(7)
  for i, at in enumerate(xs.atoms):
    at.u_star = np.array(adptbx.u_iso_as_u_star(xs.unit_cell, at.u_iso))
    at.occ_sin, at.occ_cos = 0.1 * rng.uniform(-1, 1, 2), 0.1 * rng.uniform(-1, 1, 2)
    at.u_sin = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (2, 6))
    at.u_cos = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (2, 6))
  for i, (c, w) in ((2, (0.3, 0.6)), (3, (0.71, 0.45))):
    at = xs.atoms[i]
    at.crenel = (c, w)
    at.legendre = 0.01 * rng.uniform(-1, 1, (3, 3))
    at.xharm = 0.01 * rng.uniform(-1, 1, (3, 3))
    at.u_xharm = at.u_star[None, :] * 0.05 * rng.uniform(-1, 1, (2, 6))
  xs.atoms[3].sawtooth = (0.01 * rng.uniform(-1, 1, 3), 0.65, 0.5)
  xs.dispersion = {"Cu": (0.3, 1.2)}
  h, m = reflections(xs, d_min=1.5)
  flags = {"site": False, "adp": False, "modulation": False, "occ_waves": True, "adp_waves": True,
           "crenel": True, "legendre": True, "xharm": True, "u_xharm": True, "scale": False}
  names = []
  for at in xs.atoms:
    for name, _, get, _ in mod._blocks(at, mod._flags(flags)):
      names += [(name, at.label)] * len(get())
  calc = mod.calculator(h, m)
  _, _, d = calc.compute(xs, flags, gradients=True)
  p0 = mod.get_parameters(xs, flags)
  assert len(p0) == len(names) == d.shape[1], (len(p0), len(names), d.shape)
  worst = {}
  for k in range(len(p0)):
    # large enough that rounding stays below 1e-7 for coefficients of 0.01 and less
    step = 1e-5 * max(1e-2, abs(p0[k]))
    p = p0.copy(); p[k] += step
    mod.set_parameters(xs, flags, p)
    _, fp, _ = calc.compute(xs)
    p[k] -= 2 * step
    mod.set_parameters(xs, flags, p)
    _, fm, _ = calc.compute(xs)
    mod.set_parameters(xs, flags, p0)
    num = (fp - fm) / (2 * step)
    scale = max(np.max(np.abs(num)), np.max(np.abs(d[:, k])), 1e-300)
    err = np.max(np.abs(num - d[:, k])) / scale
    block = names[k][0]
    if names[k][0] == "crenel":
      block = "crenel c" if (k == 0 or names[k - 1] != names[k]) else "crenel w"
    worst[block] = max(worst.get(block, 0.0), err)
  for block in ("occ_sin", "occ_cos", "u_sin", "u_cos", "crenel c", "crenel w", "legendre", "xharm", "u_xharm"):
    assert worst[block] <= 1e-6, (block, worst[block])
  print("  wave gradients = central differences, %d parameters: %s" % (len(p0), ", ".join(
    "%s %.1e" % (b, e) for b, e in sorted(worst.items()))))


def _image_mismatch(at, op, n_x4=41):
  """How far the atom's image under a site-symmetry operation is from the atom itself, evaluated through the
  atom's own functions - not through the parameter map: crenel interval, displacement R u(eps (x4 - delta))
  and ADP x-harmonics inside the crenel."""
  delta = op.m.dot(at.site) + op.tau
  c, w = at.crenel
  dc = (op.eps * c + delta - c + 0.5) % 1.0 - 0.5
  x4 = c + 0.45 * w * np.linspace(-1, 1, n_x4)
  src = op.eps * (x4 - delta)
  worst = abs(dc)
  worst = max(worst, np.max(np.abs(at.displacement(src).dot(op.r.T) - at.displacement(x4))))
  if len(at.u_xharm):
    worst = max(worst, np.max(np.abs(at.u_xharm_at(src).dot(mod._u6_map(op.r).T) - at.u_xharm_at(x4))))
  return worst


def test_crenel_blocks_under_site_symmetry():
  """Crenel, Legendre and x-harmonic blocks of atoms on special positions, constrained by the superspace site
  symmetry rather than held: on a mirror (eps = +1) and a two-fold axis (eps = -1) of P2/m, and on a centre
  (eps = -1) of P2_1/c. Moving along the free directions keeps every atom equal to its images (checked through
  the atom's own functions, independently of the map); a step along a removed direction does not; and the
  gradients along the free directions are the central differences."""
  q = (0.31, 0.0, 0.23)
  uc = uctbx.unit_cell((7.1, 5.3, 9.2, 90, 101.3, 90))
  rng = np.random.RandomState(9)
  cases = [("P 1 2/m 1", [("Cu", (0.12, 0.0, 0.33)), ("O", (0.0, 0.21, 0.5)), ("N", (0.43, 0.38, 0.27))]),
           ("P 1 21/c 1", [("Cu", (0.0, 0.0, 0.0)), ("O", (0.31, 0.07, 0.15))])]
  flags = {"site": True, "adp": True, "modulation": True, "occ_waves": True, "adp_waves": False,
           "crenel": True, "legendre": True, "xharm": True, "u_xharm": True, "scale": False}
  table = {}
  n_free_new = 0
  for symbol, rows in cases:
    ssg = mod.superspace_group(sgtbx.space_group_info(symbol).group(), q)
    atoms = []
    for i, (t, xyz) in enumerate(rows):
      at = mod.modulated_atom("%s%d" % (t, i), t, xyz, 0.012, 1.0, 1,
                              0.01 * rng.uniform(-1, 1, (1, 3)), 0.01 * rng.uniform(-1, 1, (1, 3)),
                              occ_sin=[0.05], occ_cos=[0.03])
      at.crenel = (0.3, 0.6)
      at.legendre = 0.01 * rng.uniform(-1, 1, (3, 3))
      at.xharm = 0.01 * rng.uniform(-1, 1, (2, 3))
      at.u_xharm = 0.001 * rng.uniform(-1, 1, (2, 6))
      atoms.append(at)
    xs = mod.modulated_structure(uc, ssg, atoms)
    # make the model symmetric: the free part of every parameter, and a crenel centre an eps = -1 operation
    # pins (eps c + delta = c mod 1) put where it must be
    z = mod.reduction(xs, flags)
    p = mod.get_parameters(xs, flags)
    mod.set_parameters(xs, flags, z.dot(z.T.dot(p)))
    special = []
    for at in xs.atoms:
      ops = [op for op in mod.site_symmetry_ops(xs, at) if not np.allclose(op.r, np.eye(3)) or op.eps != 1]
      for op in ops:
        if op.eps == -1:
          at.crenel = (((op.m.dot(at.site) + op.tau) / 2) % 0.5, at.crenel[1])
      at.crenel = (at.crenel[0] or 0.3, at.crenel[1] or 0.6) if not ops else at.crenel
      if ops:
        special.append((at, ops))
    for at, ops in special:
      for op in ops:
        assert _image_mismatch(at, op) < 1e-12, (symbol, at.label, _image_mismatch(at, op))
    z = mod.reduction(xs, flags)
    p0 = mod.get_parameters(xs, flags)
    names = []
    for at in xs.atoms:
      for name, _, get, _ in mod._blocks(at, mod._flags(flags)):
        names += [(name, at.label, k) for k in range(len(get()))]
    new_blocks = ("crenel", "legendre", "xharm", "u_xharm")
    # free directions keep the symmetry, removed ones break it
    removed = np.eye(len(p0)) - z.dot(z.T)
    for at, ops in special:
      rows_at = [i for i, nm in enumerate(names) if nm[1] == at.label and nm[0] in new_blocks]
      free = z[rows_at]
      n_free_new += int(np.sum(np.abs(free).sum(axis=0) > 1e-12))
      for k in range(z.shape[1]):
        if np.abs(z[rows_at, k]).max() < 1e-12:
          continue
        mod.set_parameters(xs, flags, p0 + 0.05 * z[:, k])
        assert max(_image_mismatch(at, op) for op in ops) < 1e-10, (symbol, at.label, k)
      bad = [i for i in rows_at if np.abs(removed[:, i]).max() > 1e-6]
      assert bad, (symbol, at.label, "nothing removed")
      v = removed[:, bad[0]] / np.linalg.norm(removed[:, bad[0]])
      mod.set_parameters(xs, flags, p0 + 0.05 * v)
      assert max(_image_mismatch(at, op) for op in ops) > 1e-4, (symbol, at.label, names[bad[0]])
      mod.set_parameters(xs, flags, p0)
    # gradients along the free directions
    h, m = reflections(xs, d_min=1.5)
    calc = mod.calculator(h, m)
    _, _, d = calc.compute(xs, flags, gradients=True)
    for k in range(z.shape[1]):
      top = names[int(np.argmax(np.abs(z[:, k])))]
      block = top[0]
      if block == "crenel":
        block = "crenel c" if top[2] == 0 else "crenel w"
      if block not in ("crenel c", "crenel w", "legendre", "xharm", "u_xharm"):
        continue
      special_atom = any(top[1] == at.label for at, _ in special)
      step = 1e-6   # the ADP x-harmonics curve the Debye-Waller factor hard: 1e-5 leaves 2e-5 truncation
      mod.set_parameters(xs, flags, p0 + step * z[:, k]); _, fp, _ = calc.compute(xs)
      mod.set_parameters(xs, flags, p0 - step * z[:, k]); _, fm, _ = calc.compute(xs)
      mod.set_parameters(xs, flags, p0)
      num, ana = (fp - fm) / (2 * step), d.dot(z[:, k])
      err = np.max(np.abs(num - ana)) / max(np.max(np.abs(num)), np.max(np.abs(ana)), 1e-300)
      key = (symbol, block, "special" if special_atom else "general")
      table[key] = max(table.get(key, 0.0), err)
  for key, err in table.items():
    assert err <= 1e-6, (key, err)
  assert n_free_new > 0
  print("  crenel/Legendre/x-harmonic blocks under site symmetry: images = atoms to 1e-10 along the free"
        " directions, broken along removed ones; gradients along the free directions:")
  for key in sorted(table):
    print("    %-11s %-9s %-8s %.1e" % (key + (table[key],)))


def _rank_by_absences_reference(groups, h, m, i_obs, sig, n_sigma=3.0):
  """rank_by_absences as it was: is_sys_absent reflection by reflection."""
  ranked = []
  for g in groups:
    viol = n_abs = 0
    for hh, mm, ii, ss in zip(h, m, i_obs, sig):
      if g.is_sys_absent(hh, mm):
        n_abs += 1
        if ii > n_sigma * ss:
          viol += 1
    ranked.append((viol, -n_abs, sum(1 for op in g.generators if op.tau), g))
  ranked.sort(key=lambda x: x[:3])
  return [(v, -n, g) for v, n, _, g in ranked]


def test_rank_by_absences_equivalence():
  """The vectorised ranking (point-part tests shared between candidates) against the reflection-by-reflection
  one: the same order, violations and predicted absences, for the candidates of several groups and q, with
  intensities that leave some absences violated."""
  rng = np.random.RandomState(12)
  cases = [("P 1 21/c 1", (0.31, 0.0, 0.23), (7.1, 5.3, 9.2, 90, 101.3, 90), (0.0, 0.5)),
           ("C 1 2/c 1", (0.31, 0.0, 0.23), (14.2, 10.6, 18.4, 90, 101.3, 90), (0.0, 0.5)),
           ("P n m a", (0.0, 0.0, 0.29), (8.2, 9.6, 11.4, 90, 90, 90), (0.0, 0.5)),
           ("P 4/m m m", (0.0, 0.0, 0.29), (6.2, 6.2, 9.4, 90, 90, 90), (0.0, 0.25, 0.5, 0.75)),
           ("P 63/m", (0.0, 0.0, 0.29), (6.2, 6.2, 9.4, 90, 90, 120), (0.0, 1 / 3., 2 / 3., 0.5))]
  n_groups = 0
  for symbol, q, cell, taus in cases:
    sg = sgtbx.space_group_info(symbol).group()
    groups = mod.candidate_superspace_groups(sg, q, taus)
    xs = mod.modulated_structure(uctbx.unit_cell(cell), groups[0], [])
    h, m = reflections(xs, d_min=1.5, m_max=2)
    i_obs = rng.exponential(100, len(m)) * (rng.uniform(size=len(m)) > 0.3)
    sig = np.full(len(m), 5.0)
    a = _rank_by_absences_reference(groups, h, m, i_obs, sig)
    b = mod.rank_by_absences(groups, h, m, i_obs, sig)
    assert [(v, n, g.label()) for v, n, g in a] == [(v, n, g.label()) for v, n, g in b], symbol
    n_groups += len(groups)
  print("  ranking by absences, vectorised = reflection by reflection: %d candidates of %d groups" % (n_groups, len(cases)))


def test_calculator_equivalence():
  """The x4 sums shared between operations with the same (R, eps, M) or its negative, and the reflection
  arrays kept between calls, against every operation on its own and a fresh calculator: F and every gradient
  block to 1e-12, in centric and acentric groups with all block families on; then the kept arrays rebuilt when
  q, the cell or the dispersion change."""
  from cctbx import adptbx
  rng = np.random.RandomState(13)
  worst = 0.0
  for symbol, q, cell in (("C 1 2/c 1", (0.31, 0.0, 0.23), (14.2, 10.6, 18.4, 90, 101.3, 90)),
                          ("C m c m", (0.0, 0.0, 0.29), (18.2, 20.6, 16.4, 90, 90, 90)),
                          ("P 21 21 21", (0.0, 0.0, 0.29), (8.2, 9.6, 11.4, 90, 90, 90))):
    uc = uctbx.unit_cell(cell)
    ssg = mod.superspace_group(sgtbx.space_group_info(symbol).group(), q)
    atoms = []
    for i, t in enumerate(["Cu", "O", "N", "C", "C", "S"]):
      at = mod.modulated_atom("%s%d" % (t, i), t, rng.uniform(0.02, 0.48, 3), 0.012, 1.0, 2,
                              0.01 * rng.uniform(-1, 1, (2, 3)), 0.01 * rng.uniform(-1, 1, (2, 3)))
      at.u_star = np.array(adptbx.u_iso_as_u_star(uc, at.u_iso))
      at.occ_sin, at.occ_cos = 0.05 * rng.uniform(-1, 1, 2), 0.05 * rng.uniform(-1, 1, 2)
      at.u_sin = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (2, 6))
      at.u_cos = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (2, 6))
      if i % 3 == 0:
        at.crenel = (0.3 + 0.05 * i, 0.6)
        at.legendre = 0.01 * rng.uniform(-1, 1, (2, 3))
        at.xharm = 0.01 * rng.uniform(-1, 1, (2, 3))
        at.u_xharm = at.u_star[None, :] * 0.05 * rng.uniform(-1, 1, (1, 6))
      atoms.append(at)
    xs = mod.modulated_structure(uc, ssg, atoms)
    xs.dispersion = {"Cu": (0.3, 1.2)}
    h, m = reflections(xs, d_min=1.6, m_max=2)
    flags = {"site": True, "adp": True, "modulation": True, "occ_waves": True, "adp_waves": True,
             "crenel": True, "legendre": True, "xharm": True, "u_xharm": True, "scale": True}
    ref = mod.calculator(h, m)
    ref._share_classes = False
    fr, sr, dr = ref.compute(xs, flags, gradients=True)
    calc = mod.calculator(h, m)
    for k in range(2):        # the second call reads the kept arrays
      f, sq, d = calc.compute(xs, flags, gradients=True)
      worst = max(worst, np.max(np.abs(f - fr)) / np.max(np.abs(fr)), np.max(np.abs(d - dr)) / np.max(np.abs(dr)))
    # a new q, cell and dispersion must not be served the kept arrays
    for change in ("q", "cell", "dispersion"):
      if change == "q":
        xs.ssg = mod.superspace_group(xs.ssg.basic, np.array(q) * 1.01 if q[0] else (0.0, 0.0, 0.31))
      elif change == "cell":
        xs.unit_cell = uctbx.unit_cell([v * (1.01 if i < 3 else 1) for i, v in enumerate(cell)])
      else:
        xs.dispersion = {"Cu": (0.1, 2.0)}
      f1 = calc.compute(xs)[0]
      f2 = mod.calculator(h, m).compute(xs)[0]
      assert np.array_equal(f1, f2), (symbol, change)
  assert worst < 1e-12, worst
  print("  shared x4 sums and kept arrays = every operation on its own: F and gradients to %.1e; rebuilt on a new"
        " q, cell or dispersion" % worst)


def _read_hklm_reference(path):
  """read_hklm as it was: line by line."""
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
          if rows:
            bad.append((n, line))
          continue
      if not any(v[:4]):
        break
      rows.append(v)
  a = np.array(rows, dtype=float).reshape(-1, 6)
  return a[:, :3].astype(int), a[:, 3].astype(int), a[:, 4], a[:, 5]


def test_read_hklm_equivalence():
  """The numpy paths of read_hklm (fixed-size records; fixed-width fields of the text) against the line-by-line
  reader, on regular files and on every irregularity that must send it back to the line-by-line one."""
  import os, tempfile, io, contextlib
  rng = np.random.RandomState(14)
  def rows(n):
    hh = rng.randint(-99, 100, (n, 3)); mm = rng.randint(-3, 4, n); ii = rng.uniform(-50, 99999, n)
    return ["%4d%4d%4d%4d%8.2f%8.2f" % (a, b, c, k, x, 1 + abs(x) ** 0.5) for (a, b, c), k, x in zip(hh, mm, ii)]
  end = "   0   0   0   0    0.00    0.00"
  cases = [(rows(3000) + [end], "\n"), (["TITLE x", "# comment"] + [r + "   1" for r in rows(800)] + [end, ""], "\r\n"),
           (rows(800) + ["", "  "], "\n"), (rows(300) + [end, "end of data", "1 2"], "\n"),
           (rows(300) + [""] + rows(300), "\n"), (rows(300) + ["# c"] + rows(300), "\n"),
           (["1 2 3 0 10.5 1.0", "-4 5 6 1 99.25 2.5", "0 0 0 0 0 0"], "\n"),
           (rows(300) + ["1 2 3 1 55.5 2.0"] + rows(300), "\n"), (["-100-100-100   1 1234.56   12.34"] * 50 + rows(50), "\n"),
           ([u"T\u00eftle \u2713"] + rows(300), "\n"), (rows(300) + ["   1   2   3   0   12.0"] + rows(10), "\n")]
  d = tempfile.mkdtemp()
  for k, (lines, nl) in enumerate(cases):
    p = os.path.join(d, "c%d.hklm" % k)
    with open(p, "w", newline="", encoding="utf-8") as f:
      f.write(nl.join(lines) + nl)
    with contextlib.redirect_stdout(io.StringIO()):
      a, b = _read_hklm_reference(p), mod.read_hklm(p)
    assert all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in zip(a, b)), k
  print("  read_hklm, numpy paths = line by line: %d files, regular and irregular" % len(cases))


def rng_matrix(n, seed=15):
  return np.random.RandomState(seed).uniform(-1, 1, (n, n))


def test_normal_equations_and_pool_equivalence():
  """J^T J and J^T r summed over slices of the reflections against the dense product (one slice), and a refine
  cycle with a budget that forces slices against one that does not: shifts to 1e-10, F to 1e-12. The pool by
  slices of the reflections against the pool by atoms: F and gradients to 1e-14 (bit for bit only with one BLAS
  thread; OpenBLAS threading / kernel choice reorders the sums, seen on an i9-13950HX)."""
  from cctbx import adptbx
  xs = model(n_harmonics=2, amp=0.02)
  for at in xs.atoms:
    at.u_star = np.array(adptbx.u_iso_as_u_star(xs.unit_cell, at.u_iso))
    at.occ_sin, at.occ_cos = np.array([0.05]), np.array([-0.03])
  xs.atoms[2].crenel = (0.3, 0.6)
  h, m = reflections(xs, d_min=1.0, m_max=2)
  flags = {"site": True, "adp": True, "modulation": True, "occ_waves": True, "crenel": True, "scale": True}
  fl = mod._flags(flags)
  fsq = mod.calculator(h, m).compute(xs)[1]
  sig = 0.02 * fsq + 1e-5 * fsq.max()
  sw = 1 / sig
  zb = mod._reduction_blocks(xs, fl)
  project = lambda d: mod._times_blocks(d, zb)
  calc = mod.calculator(h, m)
  f1, a1, b1 = mod._normal_equations(calc, xs, fl, fsq * 1.01, sw, project, len(m))
  f7, a7, b7 = mod._normal_equations(calc, xs, fl, fsq * 1.01, sw, project, len(m) // 7 + 1)
  rel = lambda x, y: np.max(np.abs(x - y)) / np.max(np.abs(y))
  e_ne = max(rel(a7, a1), rel(b7, b1))
  e_f17 = rel(f7, f1)
  assert e_f17 < 1e-14 and e_ne < 1e-12, (e_f17, e_ne)
  # the in-place mirror of the upper triangle, with blocks smaller than the matrix
  u = np.triu(rng_matrix(53))
  for blk in (1, 7, 53, 1024):
    full = mod._mirror_upper(np.array(u, order="F"), block=blk)
    assert np.array_equal(full, np.triu(u) + np.triu(u, 1).T), blk
  import copy
  p0 = mod.get_parameters(xs, fl)
  out = []
  for mb in (1e9, 0.5):
    x = copy.deepcopy(xs)
    mod.refine(x, h, m, fsq * 1.01, sig, flags=flags, cycles=1, verbose=False, max_memory_mb=mb)
    out.append((mod.get_parameters(x, fl) - p0, mod.calculator(h, m).compute(x)[0]))
  e_shift = np.max(np.abs(out[1][0] - out[0][0])) / np.max(np.abs(out[0][0]))
  e_f = np.max(np.abs(out[1][1] - out[0][1])) / np.max(np.abs(out[0][1]))
  assert e_shift < 1e-10 and e_f < 1e-12, (e_shift, e_f)
  by_atoms, by_rows = mod.calculator(h, m, threads=4), mod.calculator(h, m, threads=4)
  by_atoms._mode, by_rows._mode = "atoms", "reflections"
  fa, _, da = by_atoms.compute(xs, flags, gradients=True)
  fb, _, db = by_rows.compute(xs, flags, gradients=True)
  e_pool = max(rel(fb, fa), rel(db, da))
  assert e_pool < 1e-14, e_pool
  print("  normal equations by slices = dense: J^T J, J^T r %.1e, shifts %.1e, F %.1e; pool by reflections ="
        " by atoms %.1e" % (e_ne, e_shift, e_f, e_pool))


def _medium_like(seed=16):
  """A small stand-in for the medium problem: C2/c, two harmonics, occupancy and ADP waves, a crenel atom with
  Legendre and x-harmonic terms."""
  from cctbx import adptbx
  rng = np.random.RandomState(seed)
  uc = uctbx.unit_cell((14.2, 10.6, 18.4, 90, 101.3, 90))
  ssg = mod.superspace_group(sgtbx.space_group_info("C 1 2/c 1").group(), (0.31, 0.0, 0.23))
  atoms = []
  for i, t in enumerate(["Cu", "O", "N", "C", "C", "S"]):
    at = mod.modulated_atom("%s%d" % (t, i), t, rng.uniform(0.02, 0.48, 3), 0.012, 1.0, 2,
                            0.01 * rng.uniform(-1, 1, (2, 3)), 0.01 * rng.uniform(-1, 1, (2, 3)))
    at.u_star = np.array(adptbx.u_iso_as_u_star(uc, at.u_iso))
    at.occ_sin, at.occ_cos = 0.05 * rng.uniform(-1, 1, 2), 0.05 * rng.uniform(-1, 1, 2)
    at.u_sin = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (2, 6))
    at.u_cos = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (2, 6))
    if i == 3:
      at.crenel = (0.3, 0.6)
      at.legendre = 0.01 * rng.uniform(-1, 1, (2, 3))
      at.xharm = 0.01 * rng.uniform(-1, 1, (2, 3))
      at.u_xharm = at.u_star[None, :] * 0.05 * rng.uniform(-1, 1, (1, 6))
    atoms.append(at)
  xs = mod.modulated_structure(uc, ssg, atoms)
  flags = {"site": True, "adp": True, "modulation": True, "occ_waves": True, "adp_waves": True,
           "crenel": True, "legendre": True, "xharm": True, "u_xharm": True, "scale": True}
  return xs, flags


def test_shared_tables_equivalence():
  """The per-atom x4 tables built once and shared by every slice against each slice building its own (a fresh
  calculator per slice): F^2 and J^T r bit for bit, J^T J to rounding (syrk against gemm), small and medium-like."""
  for xs, flags in ((model(n_harmonics=2, amp=0.02), {"site": True, "u_iso": True, "modulation": True, "scale": True}),
                    _medium_like()):
    fl = mod._flags(flags)
    h, m = reflections(xs, d_min=1.4, m_max=2)
    fo = mod.calculator(h, m).compute(xs)[1] * 1.01
    sw = np.full(len(m), 0.3)
    zb = mod._reduction_blocks(xs, fl)
    project = lambda d: mod._times_blocks(d, zb)
    shared = mod._normal_equations(mod.calculator(h, m), xs, fl, fo, sw, project, len(m) // 5 + 1)
    # the reference: every slice a calculator of its own, so nothing is shared
    bounds = np.linspace(0, len(m), 6).astype(int)
    jtj = jtr = None
    fcs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
      _, fc, d = mod.calculator(h[a:b], m[a:b]).compute(xs, flags, gradients=True)
      j = project(sw[a:b, None] * d)
      r = sw[a:b] * (fo[a:b] - fc)
      jtj = j.T.dot(j) if jtj is None else jtj + j.T.dot(j)
      jtr = j.T.dot(r) if jtr is None else jtr + j.T.dot(r)
      fcs.append(fc)
    assert np.array_equal(shared[0], np.concatenate(fcs)) and np.array_equal(shared[2], jtr)
    assert np.max(np.abs(shared[1] - jtj)) <= 1e-13 * np.max(np.abs(jtj))     # syrk against gemm: rounding
  print("  per-atom x4 tables shared by the slices = built by each slice: F^2 and J^T r bit for bit")


def _lm_solve_reference(jtj, jtr, lam, diag):
  """The damped solve as it was: J^T J + lam D formed in a second matrix and solved by LU."""
  return np.linalg.solve(jtj + lam * np.diag(diag), jtr)


def _cond_from_cholesky(a):
  """1-norm condition number of the symmetric positive definite a, estimated by LAPACK dpocon from its Cholesky
  factor."""
  from scipy.linalg import lapack
  c, info = lapack.dpotrf(np.array(a, order="F"), lower=1, clean=1)
  assert info == 0
  rcond, info = lapack.dpocon(c, np.max(np.sum(np.abs(a), axis=0)), uplo="L")
  assert info == 0
  return 1.0 / rcond


def test_lm_factor_solve_equivalence():
  """The damped normal equations solved by Cholesky in the lower triangle of the one matrix that holds J^T J in
  its upper triangle, against forming J^T J + lam D apart and solving it by LU: several dampings in a row on the
  same buffer (J^T J must survive each), small and medium-like. Two backward-stable solves of the same system
  differ by at most a small multiple of cond(A) eps, so the bound is 10 cond(A) eps, A equilibrated by its
  diagonal and cond(A) estimated from its Cholesky factor (dpocon) - not a fixed number, which the conditioning of the medium-like matrix (and the
  platform's rounding) can exceed. A wrong solve (the damping off by 1 %) must fail that bound. And None for a
  matrix that is not positive definite (refine then solves a full copy by LU or least squares)."""
  eps = np.finfo(float).eps
  worst_ratio, conds, wrong = 0.0, [], np.inf
  for xs, flags in ((model(n_harmonics=2, amp=0.02), {"site": True, "u_iso": True, "modulation": True, "scale": True}),
                    _medium_like()):
    fl = mod._flags(flags)
    h, m = reflections(xs, d_min=1.4, m_max=2)
    fo = mod.calculator(h, m).compute(xs)[1] * 1.01
    sw = np.full(len(m), 0.3)
    zb = mod._reduction_blocks(xs, fl)
    _, jtj, jtr = mod._normal_equations(mod.calculator(h, m), xs, fl, fo, sw, lambda d: mod._times_blocks(d, zb), len(m))
    jtj0 = jtj.copy()
    diag = np.diag(jtj).copy()
    buf = np.asfortranarray(jtj)
    cmax = 0.0
    for lam in (1e-3, 5e-3, 2.5e-2, 1e-4):
      x = mod._lm_factor_solve(buf, diag, lam * diag, jtr)
      y = _lm_solve_reference(jtj0, jtr, lam, diag)
      # in the equilibrated variables D^1/2 x (both solvers are insensitive to that diagonal scaling, so it is
      # the condition number of D^-1/2 A D^-1/2 that bounds them)
      a = jtj0 + lam * np.diag(diag)
      sc = 1 / np.sqrt(np.diag(a))
      cond = _cond_from_cholesky(sc[:, None] * a * sc[None, :])
      cmax = max(cmax, cond)
      bound = 10 * cond * eps
      rel = lambda u: np.max(np.abs((u - y) / sc)) / np.max(np.abs(y / sc))
      worst_ratio = max(worst_ratio, rel(x) / bound)
      wrong = min(wrong, rel(_lm_solve_reference(jtj0, jtr, 1.01 * lam, diag)) / bound)
    conds.append(cmax)
    assert np.array_equal(np.triu(buf, 1), np.triu(jtj0, 1))      # J^T J kept in the upper triangle
  assert worst_ratio < 1, worst_ratio
  assert wrong > 1, wrong
  bad = np.asfortranarray(np.diag([1.0, -1.0, 2.0]))
  assert mod._lm_factor_solve(bad, np.diag(bad).copy(), np.zeros(3), np.ones(3)) is None
  print("  damped solve in one matrix (Cholesky in the lower triangle) = LU on a second matrix within 10 cond eps"
        " (cond small %.1e, medium-like %.1e)" % tuple(conds))


def test_auto_n_x4():
  """n_x4 = "auto" against n_x4 = 128 for every block family and a large displacive amplitude at |m| <= 3: F^2 to
  1e-10; and the rule set again when the amplitudes outgrow the z it was set for."""
  rng = np.random.RandomState(17)
  worst = 0.0
  chosen = []
  for kind in ("displacive", "occupancy", "adp", "crenel", "xharm", "large amplitude"):
    xs = model(n_harmonics=2, amp=0.03 if kind == "large amplitude" else 0.01)
    for at in xs.atoms:
      if kind == "occupancy":
        at.occ_sin, at.occ_cos = 0.05 * rng.uniform(-1, 1, 2), 0.05 * rng.uniform(-1, 1, 2)
      elif kind == "adp":
        at.u_sin, at.u_cos = 0.001 * rng.uniform(-1, 1, (2, 6)), 0.001 * rng.uniform(-1, 1, (2, 6))
      elif kind in ("crenel", "xharm"):
        at.crenel = (0.3, 0.6)
        if kind == "xharm":
          at.xharm, at.legendre = 0.01 * np.ones((2, 3)), 0.01 * np.ones((2, 3))
    h, m = reflections(xs, d_min=0.9, m_max=3)
    ref = mod.calculator(h, m, n_x4=128).compute(xs)[1]
    c = mod.calculator(h, m, n_x4="auto")
    f = c.compute(xs)[1]
    worst = max(worst, np.max(np.abs(f - ref)) / ref.max())
    chosen.append("%s %d" % (kind, c.n_x4))
  assert worst <= 1e-10, worst
  # amplitudes grown past the z the rule was set for: a larger n_x4, and the slices remade with it
  n0 = c.n_x4
  old = c._chunks(3)
  for at in xs.atoms:
    at.a, at.b = at.a * 3, at.b * 3
  c.compute(xs)
  subs = [s for v in c._sub.values() for s in v]
  assert c.n_x4 > n0 and all(s.n_x4 == c.n_x4 for s in subs) and not any(s in subs for s in old), (n0, c.n_x4)
  print("  n_x4 = auto against 128: %.1e (%s); raised from %d to %d as the amplitudes grew"
        % (worst, ", ".join(chosen), n0, c.n_x4))


def _all_blocks_model():
  """The model of test_gradients_all_wave_blocks: anisotropic ADPs, occupancy and ADP waves on every atom, two
  crenel atoms with Legendre, x-harmonic and ADP x-harmonic terms, one sawtooth, dispersion."""
  from cctbx import adptbx
  xs = model(n_harmonics=2, amp=0.02)
  rng = np.random.RandomState(7)
  for at in xs.atoms:
    at.u_star = np.array(adptbx.u_iso_as_u_star(xs.unit_cell, at.u_iso))
    at.occ_sin, at.occ_cos = 0.1 * rng.uniform(-1, 1, 2), 0.1 * rng.uniform(-1, 1, 2)
    at.u_sin = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (2, 6))
    at.u_cos = at.u_star[None, :] * 0.1 * rng.uniform(-1, 1, (2, 6))
  for i, (c, w) in ((2, (0.3, 0.6)), (3, (0.71, 0.45))):
    at = xs.atoms[i]
    at.crenel = (c, w)
    at.legendre = 0.01 * rng.uniform(-1, 1, (3, 3))
    at.xharm = 0.01 * rng.uniform(-1, 1, (3, 3))
    at.u_xharm = at.u_star[None, :] * 0.05 * rng.uniform(-1, 1, (2, 6))
  xs.atoms[3].sawtooth = (0.01 * rng.uniform(-1, 1, 3), 0.65, 0.5)
  xs.dispersion = {"Cu": (0.3, 1.2)}
  return xs


def test_compiled_x4_sums_equivalence():
  """The compiled per-class x4 sums (smtbx_modulation_ext) against the numpy reference path (use_compiled=False):
  F, F^2 and every gradient column to 1e-12 relative, with and without gradients, on the small model (isotropic),
  the all-blocks model (anisotropic, occupancy/ADP waves, crenels, Legendre, x-harmonics, sawtooth, dispersion),
  the medium-like model, electron scattering, refined and fixed blocks mixed, and through the atom and the
  reflection pools."""
  if mod._ext is None:
    print("  compiled x4 sums: smtbx_modulation_ext not built, skipped")
    return
  every = {"site": True, "adp": True, "u_iso": True, "occupancy": True, "modulation": True, "occ_waves": True,
           "adp_waves": True, "crenel": True, "legendre": True, "xharm": True, "u_xharm": True, "scale": True}
  some = {"site": False, "adp": True, "modulation": True, "occ_waves": False, "crenel": True, "scale": False}
  cases = [("small", model(n_harmonics=2, amp=0.02), every, "xray", None),
           ("all blocks", _all_blocks_model(), every, "xray", None),
           ("all blocks, some refined", _all_blocks_model(), some, "xray", None),
           ("medium-like", _medium_like()[0], every, "xray", None),
           ("electrons", model(n_harmonics=2, amp=0.02), every, "electron", None),
           ("small, reflection pool", model(n_harmonics=2, amp=0.02), every, "xray", "reflections")]
  worst = 0.0
  for name, xs, flags, radiation, mode in cases:
    h, m = reflections(xs, d_min=1.2, m_max=2)
    res = []
    for compiled in (False, True):
      c = mod.calculator(h, m, radiation=radiation, threads=4 if mode else None)
      c.use_compiled, c._mode = compiled, mode
      f, fsq, d = c.compute(xs, flags, gradients=True)
      f0, fsq0, _ = c.compute(xs)
      res.append((f, fsq, d, f0, fsq0))
    for a, b in zip(res[0], res[1]):
      worst = max(worst, np.max(np.abs(a - b)) / np.max(np.abs(a)))
  assert worst <= 1e-12, worst
  print("  compiled x4 sums = numpy x4 sums: F, F^2 and gradients %.1e (%d models and modes)" % (worst, len(cases)))


def test_compiled_central_differences():
  """The central-difference gradient checks again, on the compiled path."""
  if mod._ext is None:
    return
  print("  compiled x4 sums, central differences:")
  mod.USE_COMPILED = True
  try:
    test_gradients_finite_difference()
    test_gradients_aniso_occ_adp_crenel()
    test_gradients_all_wave_blocks()
  finally:
    mod.USE_COMPILED = False


def test_mscif_write_round_trip():
  """mscif.write -> mscif.read: every parameter of every block family (sites, Uiso/Uaniso, occupancies and site
  multiplicities, displacive/occupancy/ADP Fourier waves, crenels, sawtooth, Legendre, x-harmonics, ADP
  x-harmonics, f'/f'') and F^2calc of the refln loop back to 1e-12, on the small, the all-blocks and the
  medium-like model."""
  import tempfile
  from smtbx.modulation import mscif
  worst = 0.0
  for xs in (model(n_harmonics=2, amp=0.02), _all_blocks_model(), _medium_like()[0]):
    h, m = reflections(xs, d_min=1.5)
    fsq = mod.calculator(h, m).compute(xs)[1]
    path = os.path.join(tempfile.mkdtemp(), "model.cif")
    mscif.write(xs, path, h, m, f_sq_meas=1.01 * fsq, f_sq_sigma=0.02 * fsq + 1)
    ys, data = mscif.read(path)
    assert [a.label for a in ys.atoms] == [a.label for a in xs.atoms]
    assert np.allclose(ys.q, xs.q, rtol=0, atol=1e-14) and len(ys.ssg.ops) == len(xs.ssg.ops)
    for a, b in zip(xs.atoms, ys.atoms):
      assert (a.crenel is None) == (b.crenel is None) and (a.sawtooth is None) == (b.sawtooth is None)
      assert a.scattering_type == b.scattering_type and (a.u_star is None) == (b.u_star is None)
      pairs = [(a.site, b.site), ([a.u_iso, a.occupancy, a.site_weight], [b.u_iso, b.occupancy, b.site_weight])]
      for k in ("a", "b", "occ_sin", "occ_cos", "u_sin", "u_cos", "legendre", "xharm", "u_xharm"):
        assert getattr(a, k).shape == getattr(b, k).shape, k
        pairs.append((getattr(a, k), getattr(b, k)))
      if a.u_star is not None:
        pairs.append((a.u_star, b.u_star))
      if a.crenel:
        pairs.append((a.crenel, b.crenel))
      if a.sawtooth is not None:
        pairs.append((np.r_[a.sawtooth[0], a.sawtooth[1:]], np.r_[b.sawtooth[0], b.sawtooth[1:]]))
      for x, y in pairs:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.size:
          worst = max(worst, np.max(np.abs(x - y)) / max(np.max(np.abs(x)), 1e-3))
    assert ys.dispersion == xs.dispersion
    f2 = mod.calculator(data["h"], data["m"]).compute(ys)[1]
    worst = max(worst, np.max(np.abs(f2 - fsq)) / fsq.max(), np.max(np.abs(data["f_sq_calc"] - fsq)) / fsq.max(),
                np.max(np.abs(data["f_sq_meas"] - 1.01 * fsq)) / fsq.max())
  assert worst <= 1e-12, worst
  print("  msCIF write -> read: every block family and F^2calc back to %.1e" % worst)


def test_xharm_orthonormal_basis():
  """The orthonormalised x-harmonics (xharm_orthonormal) against an independent orthonormalisation (QR of the
  sampled functions): the same functions to 1e-12, orthonormal over the crenel interval; and an msCIF read with
  xharm_basis="orthonormal" gives the displacement field sum_j c_j g_j(x) of its coefficients c to 1e-12 (the
  change to the plain basis), displacive and ADP x-harmonics."""
  import tempfile
  from smtbx.modulation import mscif
  n = 6
  xg, wg = np.polynomial.legendre.leggauss(80)
  x, w = 0.5 * xg, 0.5 * wg
  f = np.array([np.ones_like(x)] + [mod.xharm_basis(k + 1, x) for k in range(n)]).T
  q, r = np.linalg.qr(np.sqrt(w)[:, None] * f)
  g_ref = (q * np.sign(np.diag(r))) / np.sqrt(w)[:, None]          # g_0 = 1, then g_1..g_n
  d, t = mod.xharm_orthonormal(n)
  g = d[None, :] + f[:, 1:].dot(t.T)
  e_basis = np.max(np.abs(g - g_ref[:, 1:]))
  e_ortho = np.max(np.abs((w[:, None] * g).T.dot(g) - np.eye(n)))
  xs = _all_blocks_model()
  rng = np.random.RandomState(23)
  coeffs = {}
  for at in xs.atoms:
    if at.crenel:
      at.xharm = 0.01 * rng.uniform(-1, 1, (n, 3))
      at.u_xharm = at.u_star[None, :] * 0.05 * rng.uniform(-1, 1, (n, 6))
      coeffs[at.label] = (at.xharm.copy(), at.u_xharm.copy())
  path = os.path.join(tempfile.mkdtemp(), "ortho.cif")
  mscif.write(xs, path)
  ys, _ = mscif.read(path, xharm_basis="orthonormal")
  e_field = 0.0
  for a, b in zip(xs.atoms, ys.atoms):
    if a.label not in coeffs:
      continue
    x4 = a.crenel[0] + a.crenel[1] * x
    c, cu = coeffs[a.label]
    # the wave in the orthonormal basis, evaluated directly, on top of everything else the atom carries
    a.xharm = np.zeros_like(c)
    want = a.displacement(x4) + g_ref[:, 1:].dot(c)
    e_field = max(e_field, np.max(np.abs(b.displacement(x4) - want)) / np.max(np.abs(g_ref[:, 1:].dot(c))))
    a.u_xharm = np.zeros_like(cu)
    e_field = max(e_field, np.max(np.abs(b.u_xharm_at(x4) - g_ref[:, 1:].dot(cu))) / np.max(np.abs(cu)))
  assert e_basis < 1e-12 and e_ortho < 1e-12 and e_field < 1e-12, (e_basis, e_ortho, e_field)
  print("  x-harmonics orthonormalised over the crenel = QR to %.1e (orthonormal to %.1e, = sqrt(2) x plain); "
        "orthonormal-basis msCIF = its displacement field to %.1e" % (e_basis, e_ortho, e_field))


def test_extinction():
  """SHELX-style EXTI on the modulated F^2: Fc^2 = scale |F|^2 (1 + 0.001 x |F|^2 lambda^3 / sin 2 theta)^(-1/2).
  Every gradient (sites, ADPs, waves, scale, x) against central differences, numpy and compiled paths, with a
  correction of up to ~40 %; x = 0 gives the plain F^2 bit for bit; and a refinement started from x = 0 recovers
  x from data made with it."""
  import copy
  xs = model(n_harmonics=2, amp=0.02)
  xs.wavelength = 0.71073
  h, m = reflections(xs, d_min=1.2, m_max=2)
  flags = {"site": True, "u_iso": True, "modulation": True, "scale": True, "extinction": True}
  plain = mod.calculator(h, m).compute(xs)[1]
  # x so that the strongest reflection is reduced by ~40 % (the correction factor 1/sqrt(1 + a x F^2))
  s = h + np.outer(m, xs.q)
  g6 = xs.unit_cell.reciprocal_metrical_matrix()
  gm = np.array([[g6[0], g6[3], g6[4]], [g6[3], g6[1], g6[5]], [g6[4], g6[5], g6[2]]])
  sin_t = 0.71073 * np.sqrt(np.einsum("ni,ij,nj->n", s, gm, s)) / 2
  a = 0.001 * 0.71073 ** 3 / (2 * sin_t * np.sqrt(1 - sin_t ** 2))
  xs.extinction = 1.8 / np.max(a * plain)
  worst = {}
  for compiled in ((False, True) if mod._ext is not None else (False,)):
    calc = mod.calculator(h, m)
    calc.use_compiled = compiled
    _, fsq, d = calc.compute(xs, flags, gradients=True)
    p0 = mod.get_parameters(xs, flags)
    assert d.shape[1] == len(p0) and p0[-1] == xs.extinction
    for k in range(len(p0)):
      step = 1e-5 * max(1e-2, abs(p0[k]))
      p = p0.copy(); p[k] += step
      mod.set_parameters(xs, flags, p)
      fp = calc.compute(xs)[1]
      p[k] -= 2 * step
      mod.set_parameters(xs, flags, p)
      fm = calc.compute(xs)[1]
      mod.set_parameters(xs, flags, p0)
      num = (fp - fm) / (2 * step)
      name = "extinction" if k == len(p0) - 1 else ("scale" if k == len(p0) - 2 else "atoms")
      e = np.max(np.abs(num - d[:, k])) / max(np.max(np.abs(num)), 1e-8)
      worst[name] = max(worst.get(name, 0.0), e)
  assert max(worst.values()) < 1e-6, worst
  i = int(np.argmax(plain))
  shrink = 1 - fsq[i] / (xs.scale * plain[i])
  x_true = xs.extinction
  ys = copy.deepcopy(xs)
  ys.extinction = 0.0
  assert np.array_equal(mod.calculator(h, m).compute(ys)[1], plain)
  data = mod.calculator(h, m).compute(xs)[1]
  sig = 0.01 * data + 1e-4 * data.max()
  mod.refine(ys, h, m, data, sig, flags={"site": False, "u_iso": False, "modulation": False, "scale": True,
                                         "extinction": True}, cycles=10, verbose=False)
  assert abs(ys.extinction - x_true) < 1e-4 * x_true, (ys.extinction, x_true)
  print("  extinction (EXTI, strongest reflection -%.0f %%): gradients = central differences, %s; x = 0 gives "
        "F^2 unchanged; refined from 0 back to x to %.1e" % (100 * shrink, ", ".join(
          "%s %.1e" % kv for kv in sorted(worst.items())), abs(ys.extinction - x_true) / x_true))


def test_xharm_against_jana():
  """COD 2108551 and 2108550 (crenels with JANA x-harmonics, Legendre terms, ADP x-harmonics) against JANA's
  F^2calc of their refln loops, the x-harmonics read as plain and as orthonormalised, when the samples are on this
  machine (E:\\olex2-samples\\cod_<id>\\<id>.cif and .hkl, as COD distributes them). Only m = 0 is asserted
  (0.6 % and 1.1 % with the plain basis); the satellite figures are reported."""
  from smtbx.modulation import mscif
  found = False
  for cid in ("2108551", "2108550"):
    root = os.path.join("E:\\olex2-samples", "cod_" + cid, cid)
    if not os.path.exists(root + ".cif"):
      continue
    found = True
    line = []
    for basis in ("plain", "orthonormal"):
      xs, d = mscif.read(root + ".cif", root + ".hkl", xharm_basis=basis)
      fc = mod.calculator(d["h"], d["m"], 64).compute(xs)[1]
      fj, m = d["f_sq_calc"], d["m"]
      k = np.sum(fc[m == 0] * fj[m == 0]) / np.sum(fc[m == 0] ** 2)
      r = [np.sum(np.abs(k * fc[s] - fj[s])) / np.sum(fj[s]) for s in (m == 0, m != 0)]
      if basis == "plain":
        assert r[0] < 0.015, (cid, r)
      line.append("%s: R(m=0) %.3f, R(|m|=1) %.3f" % (basis, r[0], r[1]))
    print("  COD %s against JANA's F^2calc, x-harmonics %s" % (cid, "; ".join(line)))
  if not found:
    print("  x-harmonics vs JANA: samples not here, skipped")


def test_commensurate_section():
  """commensurate_section (the x4 integral as the mean over the N section points of every atom image) against
  cctbx direct on the explicit supercell (q = 2/3 b*, N = 3: every image at r + L and its displaced position and
  occupancy at x4 = t0 + q.(r + L)): F(h, m) = F_supercell(H) / N exp(2 pi i m t0) to 1e-12, |m| <= 2, for
  P-1 and P2_1/c, t0 = 0 and t0 != 0, displacive and occupancy waves."""
  from cctbx import uctbx
  worst = 0.0
  for group, t0 in (("P -1", 0.0), ("P -1", 0.137), ("P 1 21/c 1", 0.07)):
    q, n = (0.0, 2.0 / 3, 0.0), 3
    xs = model(n_harmonics=2, amp=0.02, ssg=mod.superspace_group(sgtbx.space_group_info(group).group(), q))
    rng = np.random.RandomState(4)
    for at in xs.atoms:
      at.occ_sin, at.occ_cos = 0.05 * rng.uniform(-1, 1, 1), 0.05 * rng.uniform(-1, 1, 1)
    xc = mod.commensurate_section(xs, t0, n)
    h, m = reflections(xs, d_min=1.2, m_max=2)
    f = mod.calculator(h, m).compute(xc)[0]
    p = list(xs.unit_cell.parameters())
    p[1] *= n
    sc = flex.xray_scatterer()
    for at in xc.atoms:
      for L in range(n):
        r = at.site + np.array([0.0, L, 0.0])
        x4 = np.array([t0 + xs.q.dot(r)])
        pos = (r + at.displacement(x4)[0]) / np.array([1.0, n, 1.0])
        sc.append(xray.scatterer("%s_%d" % (at.label, L), tuple(pos), at.u_iso,
                                 at.occupancy_at(x4)[0] * at.site_weight, at.scattering_type))
    xrs = xray.structure(crystal_symmetry=crystal.symmetry(uctbx.unit_cell(p), "P 1"), scatterers=sc)
    xrs.scattering_type_registry(table="wk1995")
    hh = np.rint((h + np.outer(m, xs.q)) * np.array([1, n, 1])).astype(int)
    ms = miller.set(xrs.crystal_symmetry(), flex.miller_index([tuple(v) for v in hh.tolist()]), anomalous_flag=False)
    fc = ms.structure_factors_from_scatterers(xrs, algorithm="direct").f_calc().data().as_numpy_array()
    ref = fc / n * np.exp(2j * np.pi * m * t0)
    worst = max(worst, np.max(np.abs(f - ref)) / np.max(np.abs(ref)))
  assert worst < 1e-12, worst
  print("  commensurate section (q = 2/3 b*, N = 3) = cctbx direct on the supercell: %.1e" % worst)


def test_slice_workspace_equivalence():
  """The compiled path with the buffers _normal_equations lends its slices (gradient blocks and dF^2/dp reused from
  slice to slice) against every slice computed on its own with fresh arrays: F^2 bit for bit, J^T r and J^T J to
  1e-14 relative, small and medium-like; a slice's own compute() outside a pass gets new arrays (no workspace)."""
  if mod._ext is None:
    print("  slice workspace: smtbx_modulation_ext not built, skipped")
    return
  worst = 0.0
  for xs, flags in ((model(n_harmonics=2, amp=0.02), {"site": True, "u_iso": True, "modulation": True, "scale": True}),
                    _medium_like()):
    fl = mod._flags(flags)
    h, m = reflections(xs, d_min=1.4, m_max=2)
    fo = mod.calculator(h, m).compute(xs)[1] * 1.01
    sw = np.full(len(m), 0.3)
    zb = mod._reduction_blocks(xs, fl)
    project = lambda d: mod._times_blocks(d, zb)
    calc = mod.calculator(h, m)
    calc.use_compiled = True
    fc, jtj, jtr = mod._normal_equations(calc, xs, fl, fo, sw, project, len(m) // 4 + 1)
    subs = mod._slices(calc, len(m), len(m) // 4 + 1)
    assert all(c._workspace is None for c in subs)
    rj = rr = None
    fcs, a = [], 0
    for c in subs:
      b = a + len(c.m)
      _, f, d = c.compute(xs, flags, gradients=True)
      j = project(sw[a:b, None] * d)
      r = sw[a:b] * (fo[a:b] - f)
      rj = j.T.dot(j) if rj is None else rj + j.T.dot(j)
      rr = j.T.dot(r) if rr is None else rr + j.T.dot(r)
      fcs.append(f)
      a = b
    assert np.array_equal(fc, np.concatenate(fcs))
    worst = max(worst, np.max(np.abs(jtj - rj)) / np.max(np.abs(rj)), np.max(np.abs(jtr - rr)) / np.max(np.abs(rr)))
  assert worst <= 1e-14, worst
  print("  slices with the pass's reused buffers = slices with fresh arrays: F^2 bit for bit, J^T J, J^T r %.1e" % worst)


def test_blas_threads_equivalence():
  """The J^T J build and the damped solve with BLAS raised to the calculator's thread count (_blas_threads)
  against BLAS as the process has it: J^T J and J^T r to 1e-14 relative, small and medium-like, sliced; and the
  process's BLAS thread count restored afterwards."""
  libs = mod._openblas_libs()
  before = [lib.openblas_get_num_threads() for lib in libs]
  raise_was, mod.RAISE_BLAS_THREADS = mod.RAISE_BLAS_THREADS, True
  worst = 0.0
  for xs, flags in ((model(n_harmonics=2, amp=0.02), {"site": True, "u_iso": True, "modulation": True, "scale": True}),
                    _medium_like()):
    fl = mod._flags(flags)
    h, m = reflections(xs, d_min=1.4, m_max=2)
    fo = mod.calculator(h, m).compute(xs)[1] * 1.01
    sw = np.full(len(m), 0.3)
    zb = mod._reduction_blocks(xs, fl)
    out = []
    for threads in (1, 4):
      out.append(mod._normal_equations(mod.calculator(h, m, threads=threads), xs, fl, fo, sw,
                                       lambda d: mod._times_blocks(d, zb), len(m) // 3 + 1))
    assert np.array_equal(out[0][0], out[1][0])
    worst = max(worst, np.max(np.abs(out[1][1] - out[0][1])) / np.max(np.abs(out[0][1])),
                np.max(np.abs(out[1][2] - out[0][2])) / np.max(np.abs(out[0][2])))
  mod.RAISE_BLAS_THREADS = raise_was
  assert worst <= 1e-14, worst
  assert [lib.openblas_get_num_threads() for lib in libs] == before
  print("  J^T J with BLAS raised to the calculator's threads = BLAS as set: %.1e (%d OpenBLAS librar%s found)"
        % (worst, len(libs), "y" if len(libs) == 1 else "ies"))


def test_commensurate_gradients_and_refine():
  """commensurate_compute: the folded image gradients plus the site part (wave phases and section points that
  move with a site) against central differences of the commensurate F^2 with respect to the parameters of the
  superspace model, P2_1/c, q = 2/3 b*, t0 = 0.07: <= 1e-6; and refine(commensurate=...) from shifted sites and
  scaled amplitudes back to the model that made the data."""
  import copy
  from cctbx import sgtbx
  xs = model(n_harmonics=2, amp=0.02, ssg=mod.superspace_group(sgtbx.space_group_info("P 1 21/c 1").group(),
                                                              (0.0, 2.0 / 3, 0.0)))
  rng = np.random.RandomState(4)
  for at in xs.atoms:
    at.occ_sin, at.occ_cos = 0.05 * rng.uniform(-1, 1, 1), 0.05 * rng.uniform(-1, 1, 1)
  flags = {"site": True, "u_iso": True, "modulation": True, "occ_waves": True, "scale": True}
  h, m = reflections(xs, d_min=1.2, m_max=2)
  t0 = 0.07
  calc = mod.calculator(h, m)
  d = mod.commensurate_compute(calc, xs, t0, 3, flags, gradients=True)[2]
  p0 = mod.get_parameters(xs, flags)
  assert d.shape[1] == len(p0)
  worst = 0.0
  for k in range(len(p0)):
    st = 1e-5 * max(1e-2, abs(p0[k]))
    p = p0.copy(); p[k] += st
    mod.set_parameters(xs, flags, p)
    fp = mod.commensurate_compute(calc, xs, t0, 3)[1]
    p[k] -= 2 * st
    mod.set_parameters(xs, flags, p)
    fm = mod.commensurate_compute(calc, xs, t0, 3)[1]
    mod.set_parameters(xs, flags, p0)
    num = (fp - fm) / (2 * st)
    worst = max(worst, np.max(np.abs(num - d[:, k])) / max(np.max(np.abs(num)), 1e-8))
  assert worst < 1e-6, worst
  fl = {"site": True, "u_iso": True, "modulation": True, "scale": True}
  xs1 = model(n_harmonics=1, amp=0.02, ssg=xs.ssg)
  h1, m1 = reflections(xs1, d_min=1.0, m_max=1)
  data = mod.commensurate_compute(mod.calculator(h1, m1), xs1, t0, 3)[1]
  ys = copy.deepcopy(xs1)
  for a in ys.atoms:
    a.a, a.b, a.site = 0.8 * a.a, 1.2 * a.b, a.site + 0.003
  mod.refine(ys, h1, m1, data, 0.01 * data + 1e-3 * data.max(), flags=fl, cycles=15, verbose=False,
             commensurate=(t0, 3))
  err = np.max(np.abs(mod.get_parameters(ys, fl) - mod.get_parameters(xs1, fl)))
  assert err < 1e-8, err
  print("  commensurate gradients = central differences: %.1e (%d parameters); refined back to the model: %.1e"
        % (worst, len(p0), err))


def run():
  # the tests below run on the numpy reference path, so that their output does not move with the compiled one's
  # rounding; test_compiled_* compare the compiled path with it and repeat the central differences on it
  mod.USE_COMPILED = False
  mod.RAISE_BLAS_THREADS = False   # likewise BLAS's thread count; test_blas_threads_equivalence raises it
  test_superspace_closure()
  test_main_reflections_match_cctbx()
  test_bessel_single_harmonic()
  test_gradients_finite_difference()
  test_calculator_equivalence()
  test_normal_equations_and_pool_equivalence()
  test_shared_tables_equivalence()
  test_lm_factor_solve_equivalence()
  test_auto_n_x4()
  test_compiled_x4_sums_equivalence()
  test_compiled_central_differences()
  test_mscif_write_round_trip()
  test_xharm_orthonormal_basis()
  test_xharm_against_jana()
  test_extinction()
  test_commensurate_section()
  test_slice_workspace_equivalence()
  test_blas_threads_equivalence()
  test_commensurate_gradients_and_refine()
  test_gradients_aniso_occ_adp_crenel()
  test_gradients_all_wave_blocks()
  test_crenel_blocks_under_site_symmetry()
  test_mscif_against_jana()
  test_hklm_files()
  test_read_hklm_equivalence()
  test_reflection_statistics()
  test_change_of_basis_invariant()
  test_rank_by_absences()
  test_rank_by_absences_equivalence()
  test_recovery()
  test_average_constraints_and_restraints()
  test_aspherical_table()
  test_electron()
  print("OK")


if __name__ == "__main__":
  run()
