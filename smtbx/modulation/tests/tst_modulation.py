"""smtbx.modulation: superspace closure, F against cctbx and the Bessel form, gradients, settings, absences, recovery."""
from __future__ import absolute_import, division, print_function

import math

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


def run():
  test_superspace_closure()
  test_main_reflections_match_cctbx()
  test_bessel_single_harmonic()
  test_gradients_finite_difference()
  test_gradients_aniso_occ_adp_crenel()
  test_gradients_all_wave_blocks()
  test_mscif_against_jana()
  test_hklm_files()
  test_reflection_statistics()
  test_change_of_basis_invariant()
  test_rank_by_absences()
  test_recovery()
  test_average_constraints_and_restraints()
  test_aspherical_table()
  test_electron()
  print("OK")


if __name__ == "__main__":
  run()
