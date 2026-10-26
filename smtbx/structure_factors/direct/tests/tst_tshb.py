""" .tshb densities (Gaussians times real spherical harmonics) as form factors.

What is checked:
- the analytic transform against a numerical Fourier integral of the density,
  with the harmonics evaluated independently (scipy's complex harmonics);
- symmetry equivalents, and the whole structure factor against a sum written
  out by hand;
- that spherical atoms written as .tshb give the spherical structure factors
  and gradients, with and without atoms missing from the file;
- that a least-squares build with them is the spherical one;
- version 4 ids, merging and errors; atoms moved onto a symmetry image
  (rotated shells), non-integer Slater powers, disorder gradients, and that
  damaged files are refused rather than crash.
"""
from __future__ import absolute_import, division, print_function

import math, os, random, tempfile
from cctbx import crystal, sgtbx, uctbx, xray, adptbx
from cctbx.array_family import flex
from cctbx.development import random_structure
import smtbx.structure_factors.direct as direct
from smtbx.structure_factors.direct import tshb

build = direct.ext.table_based_scatterer_contribution.build
build_with_fallback = direct.ext.table_based_scatterer_contribution.build_with_fallback


def real_ylm(l, m, x, y, z):
  """ Real orthonormal harmonics from scipy's complex ones, the convention of
  harmonic_expansion.h: sqrt2 (-1)^m Re Y_l^m (m > 0), sqrt2 (-1)^m Im Y_l^|m|
  (m < 0); scipy's Y_l^m carries the Condon-Shortley phase. """
  import numpy
  try:
    from scipy.special import sph_harm_y        # scipy >= 1.15
    sph_harm = lambda m, l, phi, theta: sph_harm_y(l, m, theta, phi)
  except ImportError:
    from scipy.special import sph_harm
  theta = numpy.arccos(numpy.clip(z, -1, 1))      # polar
  phi = numpy.arctan2(y, x)                       # azimuth
  if m == 0:
    return sph_harm(0, l, phi, theta).real
  y_lm = sph_harm(abs(m), l, phi, theta)
  sign = (-1)**abs(m)
  return math.sqrt(2)*sign*(y_lm.real if m > 0 else y_lm.imag)


def random_atom(label, lmax, rng):
  shells = []
  for l in range(lmax + 1):
    for alpha in (1.5, 4.0):
      shells.append(tshb.shell(l, alpha, [rng.uniform(-1, 1) for _ in range(2*l + 1)]))
  return tshb.atom(label, shells)


def exercise_transform_against_numerical_integral():
  import numpy
  rng = random.Random(1)
  a = random_atom("X1", 3, rng)
  path = os.path.join(tempfile.mkdtemp(), "one.tshb")
  tshb.write(path, [a])
  cs = crystal.symmetry((20, 20, 20, 90, 90, 90), "P1")
  xs = xray.structure(crystal_symmetry=cs, scatterers=flex.xray_scatterer(
    [xray.scatterer("X1", (0, 0, 0), u=0, scattering_type="C")]))
  contribution = build(cs.unit_cell(), xs.scatterers(), path,
                       cs.space_group(), True)
  # the density on a grid, Fourier integrated numerically
  step, L = 0.05, 4.0
  g = numpy.arange(-L, L + step/2, step)
  X, Y, Z = numpy.meshgrid(g, g, g, indexing="ij")
  R = numpy.sqrt(X*X + Y*Y + Z*Z)
  Rs = numpy.where(R == 0, 1, R)
  rho = numpy.zeros_like(R)
  for s in a.shells:
    radial = R**s.l*numpy.exp(-s.alpha*R*R)
    for m in range(-s.l, s.l + 1):
      rho += s.c[m + s.l]*radial*real_ylm(s.l, m, X/Rs, Y/Rs, Z/Rs)
  worst = 0
  for h in [(1, 0, 0), (0, 2, 3), (-3, 1, 5), (7, -4, 2), (10, 6, -9)]:
    s = numpy.array(h)/20.
    numeric = (rho*numpy.exp(2j*math.pi*(s[0]*X + s[1]*Y + s[2]*Z))).sum()*step**3
    analytic = contribution.get_full(0, h)[0]
    worst = max(worst, abs(numeric - analytic)/max(1, abs(numeric)))
  assert worst < 1e-6, worst
  print("\ttransform against a numerical integral: %.1e" % worst)


def exercise_symmetry_and_structure_factor():
  rng = random.Random(2)
  cs = crystal.symmetry((7.1, 9.3, 11.2, 90, 103.4, 90), "P 1 21/c 1")
  sites = [(0.13, 0.27, 0.41), (0.62, 0.08, 0.77)]
  xs = xray.structure(crystal_symmetry=cs, scatterers=flex.xray_scatterer(
    [xray.scatterer("X%d" % (i + 1), site, u=0, scattering_type="C")
     for i, site in enumerate(sites)]))
  path = os.path.join(tempfile.mkdtemp(), "two.tshb")
  tshb.write(path, [random_atom("X1", 3, rng), random_atom("X2", 2, rng)])
  sg = cs.space_group()
  c_sg = build(cs.unit_cell(), xs.scatterers(), path, sg, True)
  c_p1 = build(cs.unit_cell(), xs.scatterers(), path, sgtbx.space_group(), True)
  fc = direct.f_calc_modulus_squared(xs, scatterer_contribution=c_sg)
  worst_ff = worst_fc = 0
  for h in [(1, 2, 3), (-2, 0, 4), (3, -5, 1), (0, 1, 7), (4, 3, -6)]:
    by_hand = 0
    for j, sc in enumerate(xs.scatterers()):
      ff = c_sg.get_full(j, h)
      for i, op in enumerate(sg.smx()):
        # h R as a row vector
        hr = [sum(h[k]*op.r().num()[3*k + c] for k in range(3)) for c in range(3)]
        worst_ff = max(worst_ff, abs(ff[i] - c_p1.get_full(j, tuple(hr))[0]))
        x = op*sc.site
        phase = 2*math.pi*sum(h[k]*x[k] for k in range(3))
        by_hand += ff[i]*complex(math.cos(phase), math.sin(phase))
        if sg.is_centric():
          # the inverted copy: -R, and the form factor at -h R
          xi = [-v for v in op*sc.site]
          ffi = c_p1.get_full(j, tuple(-v for v in hr))[0]
          phase = 2*math.pi*sum(h[k]*xi[k] for k in range(3))
          by_hand += ffi*complex(math.cos(phase), math.sin(phase))
    fc.linearise(h)
    worst_fc = max(worst_fc, abs(fc.f_calc - by_hand)/abs(by_hand))
  assert worst_ff < 1e-12, worst_ff
  assert worst_fc < 1e-10, worst_fc
  print("\tsymmetry equivalents %.1e, structure factor against the hand sum %.1e"
        % (worst_ff, worst_fc))


def exercise_electron_diffraction():
  # "ED: true" with a version 3 Z: NoSpherA2's convert_to_ED,
  # 0.023934 (Z - f)/stl^2, on every symmetry equivalent
  rng = random.Random(3)
  cs = crystal.symmetry((7.1, 9.3, 11.2, 90, 103.4, 90), "P 1 21/c 1")
  xs = xray.structure(crystal_symmetry=cs, scatterers=flex.xray_scatterer(
    [xray.scatterer("X1", (0.13, 0.27, 0.41), u=0, scattering_type="C")]))
  a = random_atom("X1", 3, rng)._replace(slaters=[(0.3, 1, 2.5)], z=6)
  d = tempfile.mkdtemp()
  px, pe = os.path.join(d, "x.tshb"), os.path.join(d, "e.tshb")
  tshb.write(px, [a])
  tshb.write(pe, [a], header={"ED": "true"})
  assert tshb.read(pe)[1][0].z == 6
  cx = build(cs.unit_cell(), xs.scatterers(), px, cs.space_group(), True)
  ce = build(cs.unit_cell(), xs.scatterers(), pe, cs.space_group(), True)
  worst = 0
  for h in [(1, 2, 3), (-2, 0, 4), (3, -5, 1), (0, 1, 7)]:
    stl2 = cs.unit_cell().stol(h)**2
    fx, fe = list(cx.get_full(0, h)), list(ce.get_full(0, h))
    for x, e in zip(fx, fe):
      ref = complex(0.023934*(6 - x.real)/stl2, -0.023934*x.imag/stl2)
      worst = max(worst, abs(e - ref)/abs(ref))
  assert worst < 1e-13, worst
  print("\telectron diffraction (Mott-Bethe) %.1e" % worst)


def exercise_cache():
  """ The form factors of every equivalent and Friedel mate of a reflection come
  from one cached orbit: per atom either its n_smx values or its folded
  coefficients, whichever is smaller. Checked against P1, where nothing is
  shared, for both kinds, X-ray and ED, filled and hit, and with no cache. """
  rng = random.Random(4)
  d = tempfile.mkdtemp()
  # X1 lmax 3 (16 coefficients), X2 lmax 1 (4) with a Slater record
  atoms = [random_atom("X1", 3, rng)._replace(z=6),
           random_atom("X2", 1, rng)._replace(slaters=[(0.3, 1, 2.5)], z=8)]
  px, pe = os.path.join(d, "x.tshb"), os.path.join(d, "e.tshb")
  tshb.write(px, atoms)
  tshb.write(pe, atoms, header={"ED": "true"})
  worst = 0
  for symbol in ("P 1 21/c 1", "P 21 21 21", "R -3 :H", "I 41/a :2",
                 "P 63/m m c", "F m -3 m"):
    sgi = sgtbx.space_group_info(symbol)
    cs = crystal.symmetry(sgi.any_compatible_unit_cell(volume=1500), space_group_info=sgi)
    sg = cs.space_group()
    xs = xray.structure(crystal_symmetry=cs, scatterers=flex.xray_scatterer(
      [xray.scatterer(a.label, (0.11, 0.23, 0.37), u=0, scattering_type="C") for a in atoms]))
    p1 = sgtbx.space_group()
    ref_x = build(cs.unit_cell(), xs.scatterers(), px, p1, True)
    ref_e = build(cs.unit_cell(), xs.scatterers(), pe, p1, True)
    orbit = set()
    for h in [(1, 2, 3), (4, -1, 2), (0, 3, 5), (2, 2, 7)]:
      for op in sg.smx():
        r = op.r().num()
        hr = tuple(sum(h[k]*r[3*k + c] for k in range(3)) for c in range(3))
        orbit.add(hr); orbit.add(tuple(-v for v in hr))
    orbit = sorted(orbit)
    rng.shuffle(orbit)
    for budget in ("0", None):
      if budget is None: os.environ.pop("SMTBX_TSHB_CACHE_MB", None)
      else: os.environ["SMTBX_TSHB_CACHE_MB"] = budget
      cx = build(cs.unit_cell(), xs.scatterers(), px, sg, True)
      ce = build(cs.unit_cell(), xs.scatterers(), pe, sg, True)
      os.environ.pop("SMTBX_TSHB_CACHE_MB", None)
      for h in orbit + orbit[::-1]:     # the second pass hits
        stl2 = cs.unit_cell().stol(h)**2
        for j in range(len(atoms)):
          fx, fe = cx.get_full(j, h), ce.get_full(j, h)
          for i, op in enumerate(sg.smx()):
            r = op.r().num()
            hr = tuple(sum(h[k]*r[3*k + c] for k in range(3)) for c in range(3))
            x = ref_x.get_full(j, hr)[0]
            worst = max(worst, abs(fx[i] - x)/max(1, abs(x)))
            e = complex(0.023934*(atoms[j].z - x.real)/stl2, -0.023934*x.imag/stl2)
            worst = max(worst, abs(fe[i] - e)/max(1, abs(e)),
                        abs(fe[i] - ref_e.get_full(j, hr)[0])/max(1, abs(e)))
  assert worst < 1e-12, worst
  print("\tcached orbits against P1, six space groups, X-ray and ED: %.1e" % worst)


def spherical_setup():
  random.seed(3)
  flex.set_random_seed(3)
  xs = random_structure.xray_structure(sgtbx.space_group_info("P 21/c"),
    elements=["C"]*8 + ["N"]*2 + ["O"]*2, random_u_iso=True)
  for i, sc in enumerate(xs.scatterers()):
    sc.label = "%s%d" % (sc.scattering_type, i + 1)
    sc.flags.set_use_u_iso(False); sc.flags.set_use_u_aniso(True)
    sc.u_star = adptbx.u_cart_as_u_star(xs.unit_cell(),
      (0.02, 0.03, 0.025, 0.003, -0.002, 0.001))
    sc.flags.set_grad_site(True); sc.flags.set_grad_u_aniso(True)
  xs.scattering_type_registry(table="it1992")
  return xs


def write_spherical(xs, path, skip=()):
  reg = xs.scattering_type_registry()
  tshb.write(path, [tshb.spherical_atom(sc.label, reg.gaussian(sc.scattering_type))
                    for i, sc in enumerate(xs.scatterers()) if i not in skip])


def exercise_spherical_equivalence():
  xs = spherical_setup()
  path = os.path.join(tempfile.mkdtemp(), "sph.tshb")
  spherical = direct.f_calc_modulus_squared(xs)
  for skip, label in (((), "all atoms in the file"), ((3, 7), "two atoms missing")):
    write_spherical(xs, path, skip)
    if skip:
      c = build_with_fallback(xs.unit_cell(), xs.scatterers(), path,
        xs.space_group(), True, xs.scattering_type_registry(),
        allow_spherical_fallback=True)
      assert list(c.scatterers_not_in_table()) == list(skip)
    else:
      c = build(xs.unit_cell(), xs.scatterers(), path, xs.space_group(), True)
    table = direct.f_calc_modulus_squared(xs, scatterer_contribution=c)
    worst_f = worst_g = 0
    for h in xs.structure_factors(d_min=0.8).f_calc().indices()[:400]:
      spherical.linearise(h); table.linearise(h)
      worst_f = max(worst_f, abs(spherical.f_calc - table.f_calc)/max(1, abs(spherical.f_calc)))
      g1, g2 = spherical.grad_f_calc, table.grad_f_calc
      worst_g = max(worst_g, max(abs(a - b) for a, b in zip(g1, g2))/max(1, max(abs(a) for a in g1)))
    assert worst_f < 1e-8 and worst_g < 1e-8, (worst_f, worst_g)
    print("\tspherical atoms as .tshb (%s): Fc %.1e, gradients %.1e" % (label, worst_f, worst_g))


def exercise_least_squares_build():
  from smtbx.refinement import least_squares, constraints
  import smtbx.utils
  xs = spherical_setup()
  path = os.path.join(tempfile.mkdtemp(), "ls.tshb")
  write_spherical(xs, path)
  fo_sq = xs.structure_factors(d_min=0.9).f_calc().norm()
  fo_sq = fo_sq.customized_copy(sigmas=flex.double(fo_sq.size(), 1.))
  fo_sq.set_observation_type_xray_intensity()
  def normal_matrix(table):
    rep = constraints.reparametrisation(structure=xs, constraints=[],
      connectivity_table=smtbx.utils.connectivity_table(xs))
    ls = least_squares.crystallographic_ls(fo_sq.as_xray_observations(), rep,
      weighting_scheme=least_squares.unit_weighting(), may_parallelise=False)
    if table:
      c = build(xs.unit_cell(), xs.scatterers(), path, xs.space_group(), True)
      ls.one_h_linearisation = least_squares.f_calc_function_default(
        direct.f_calc_modulus_squared(xs, scatterer_contribution=c))
    ls.build_up()
    return ls.normal_matrix_packed_u().deep_copy()
  a, b = normal_matrix(False), normal_matrix(True)
  d = flex.max(flex.abs(a - b))/flex.max(flex.abs(a))
  assert d < 1e-8, d
  print("\tleast-squares normal matrix, spherical against .tshb: %.1e" % d)


def exercise_spherical_records():
  """ Version 2: Gaussians, Slater terms and form factor terms (what NoSpherA2
  writes for ECP cores), against radial integrals done numerically. """
  from scipy.integrate import quad
  gaussians = [(2.5, 3.0), (-0.4, 0.7)]
  slaters = [(30.0, 0, 9.0), (4.0, 2, 3.5), (-1.5, 1, 2.2)]
  exponentials = [(10.0, 0.125)]
  path = os.path.join(tempfile.mkdtemp(), "core.tshb")
  tshb.write(path, [tshb.atom("X1", [], gaussians, slaters, exponentials)])
  assert tshb.read(path)[1][0].slaters[1] == (4.0, 2.0, 3.5)
  cs = crystal.symmetry((20, 20, 20, 90, 90, 90), "P1")
  xs = xray.structure(crystal_symmetry=cs, scatterers=flex.xray_scatterer(
    [xray.scatterer("X1", (0, 0, 0), u=0, scattering_type="C")]))
  c = build(cs.unit_cell(), xs.scatterers(), path, cs.space_group(), True)
  def rho(r):
    return (sum(g*math.exp(-a*r*r) for g, a in gaussians)
            + sum(cc*r**n*math.exp(-z*r) for cc, n, z in slaters))
  worst = 0
  for h in [(0, 0, 0), (1, 0, 0), (3, 4, 0), (9, 7, 5), (20, 15, 11)]:
    s = math.sqrt(sum(x*x for x in h))/20.
    k = 2*math.pi*s
    if k == 0:
      f = quad(lambda r: 4*math.pi*r*r*rho(r), 0, 60, limit=400)[0]
    else:
      f = quad(lambda r: 4*math.pi*r*r*rho(r)*math.sin(k*r)/(k*r), 0, 60, limit=400)[0]
    f += sum(a*math.exp(-b*s) for a, b in exponentials)
    v = c.get_full(0, h)[0]
    assert abs(v.imag) < 1e-12
    worst = max(worst, abs(v.real - f)/max(1, abs(f)))
  assert worst < 1e-8, worst
  print("\tGaussian, Slater and form factor records against radial integrals: %.1e" % worst)

def disordered_setup():
  """ P2_1/c, two PART 0 atoms and one disordered site whose two parts share
  the label C3 - which only an id tells apart. """
  cs = crystal.symmetry((7.1, 9.3, 11.2, 90, 103.4, 90), "P 1 21/c 1")
  spec = [("O1", "O", (0.13, 0.27, 0.41), 0, 1.0),
          ("N2", "N", (0.31, 0.12, 0.18), 0, 1.0),
          ("C3", "C", (0.62, 0.08, 0.77), 1, 0.6),
          ("C3", "C", (0.66, 0.11, 0.73), 2, 0.4)]
  scs = []
  for label, t, site, part, occ in spec:
    sc = xray.scatterer(label, site, u=0, occupancy=occ, scattering_type=t)
    sc.set_part(part)
    scs.append(sc)
  return xray.structure(crystal_symmetry=cs, scatterers=flex.xray_scatterer(scs))


def keyed(xs, j, record):
  sc = xs.scatterers()[j]
  return record._replace(label=sc.label, key=tshb.scatterer_id(sc), part=sc.get_part())


def form_factors_of(record, cs, h):
  """ What a lone label-keyed record gives, in P1 - the reference. """
  path = os.path.join(tempfile.mkdtemp(), "one.tshb")
  tshb.write(path, [record._replace(label="X", key=None, part=0)])
  one = xray.structure(crystal_symmetry=crystal.symmetry(cs.unit_cell(), "P1"),
    scatterers=flex.xray_scatterer([xray.scatterer("X", (0, 0, 0), u=0, scattering_type="C")]))
  return build(one.unit_cell(), one.scatterers(), path, one.space_group(), True).get_full(0, h)[0]


def exercise_version_4_ids_and_merge():
  """ Per-part files keyed by scatterer id, merged with += (first wins), read
  back onto a model whose atoms have since moved: every atom gets its own
  record, the PART 0 atoms the first part's, the two C3 theirs. """
  rng = random.Random(7)
  xs = disordered_setup()
  d = tempfile.mkdtemp()
  # each part's density: its own PART 0 atoms (different, to show which is
  # kept) and its own C3
  part1 = tshb.tshb_block([keyed(xs, j, random_atom("x", 2, rng)._replace(z=z))
                           for j, z in ((0, 8), (1, 7), (2, 6))],
                          header={"COVERAGE": "partial", "PARTS": "0 1"})
  part2 = tshb.tshb_block([keyed(xs, j, random_atom("x", 2, rng)._replace(z=z))
                           for j, z in ((0, 8), (1, 7), (3, 6))],
                          header={"COVERAGE": "partial", "PARTS": "0 2"})
  part1.write(os.path.join(d, "p1.tshb"))
  part2.write(os.path.join(d, "p2.tshb"))
  joint = tshb.tshb_block.read(os.path.join(d, "p1.tshb"))
  joint += tshb.tshb_block.read(os.path.join(d, "p2.tshb"))
  joint.header["COVERAGE"] = "complete"
  joint.header["PARTS"] = "0 1 2"
  assert len(joint) == 4, len(joint)
  same = lambda a, b: (a.label, bytes(a.key), a.part, a.z, [(x.l, x.alpha, list(x.c)) for x in a.shells]) == \
                      (b.label, bytes(b.key), b.part, b.z, [(x.l, x.alpha, list(x.c)) for x in b.shells])
  assert same(joint.records[0], part1.records[0])    # first wins
  assert not same(joint.records[0], part2.records[0])
  path = os.path.join(d, "joint.tshb")
  joint.write(path)
  header, records = tshb.read(path)
  assert header["COVERAGE"] == "complete"
  assert [r.part for r in records] == [0, 0, 1, 2]
  assert all(r.key == tshb.scatterer_id(xs.scatterers()[j])
             for j, r in enumerate(records))
  # the model refines on: every atom moves by ~0.03 A, so no id matches as it
  # is and all four go by the position the id carries
  moved = xs.deep_copy_scatterers()
  for sc in moved.scatterers():
    sc.site = tuple(x + 0.003 for x in sc.site)
  expected = [part1.records[0], part1.records[1], part1.records[2], part2.records[2]]
  worst = 0
  for structure in (xs, moved):
    c = build_with_fallback(structure.unit_cell(), structure.scatterers(), path,
      structure.space_group(), True, structure.scattering_type_registry())
    assert len(c.scatterers_not_in_table()) == 0
    assert len(c.records_not_in_structure()) == 0
    for h in [(1, 2, 3), (-2, 0, 4), (3, -5, 1)]:
      for j, rec in enumerate(expected):
        worst = max(worst, abs(c.get_full(j, h)[0] - form_factors_of(rec, xs, h)))
  assert worst < 1e-12, worst
  # Python merges like NoSpherA2: the + operator too, and a second += is a no-op
  again = joint + part2
  assert len(again) == 4 and all(same(a, b) for a, b in zip(again.records, joint.records))
  print("\tversion 4: per-part files keyed by id, merged by +=, first wins,"
        " matched by id and after a move by position: %.1e" % worst)


def exercise_version_4_errors():
  """ A key twice, two records on one atom, an atom with no record, a record
  with no atom. """
  rng = random.Random(8)
  xs = disordered_setup()
  d = tempfile.mkdtemp()
  recs = [keyed(xs, j, random_atom("x", 1, rng)) for j in range(4)]
  def expect_error(path, words, **kw):
    try:
      build_with_fallback(xs.unit_cell(), xs.scatterers(), path, xs.space_group(),
        True, xs.scattering_type_registry(), **kw)
    except RuntimeError as e:
      for w in words:
        assert w in str(e), (w, str(e))
      return str(e)
    raise AssertionError("no error for %s" % path)
  # the same id twice
  p = os.path.join(d, "dup.tshb")
  tshb.write(p, recs + [recs[2]])
  expect_error(p, ["twice", "C3"])
  try:
    tshb.tshb_block.read(p)
    raise AssertionError("tshb_block.read accepted a repeated key")
  except ValueError:
    pass
  # two ids resolving to one atom: O1's, and O1's shifted 0.01 in x
  near = xs.scatterers()[0].customized_copy(
    site=tuple(x + d_x for x, d_x in zip(xs.scatterers()[0].site, (0.01, 0, 0))))
  p = os.path.join(d, "two.tshb")
  tshb.write(p, recs + [recs[0]._replace(key=tshb.scatterer_id(near))])
  expect_error(p, ["resolve to the atom O1"])
  # labels and ids mixed in one file
  p = os.path.join(d, "mixed.tshb")
  tshb.write(p, recs + [recs[0]._replace(label="X9", key=None)])
  expect_error(p, ["mixes label and scatterer id keys"])
  # the second part's C3 missing: refused, named with its PART
  p = os.path.join(d, "short.tshb")
  tshb.write(p, recs[:3])
  msg = expect_error(p, ["no density for 1 of 4 atoms", "C3(PART 2)"])
  # ... unless the caller asks for the spherical stand-in
  c = build_with_fallback(xs.unit_cell(), xs.scatterers(), p, xs.space_group(),
    True, xs.scattering_type_registry(), allow_spherical_fallback=True)
  assert list(c.scatterers_not_in_table()) == [3]
  # a record for an atom the model no longer has: kept for the caller to report
  gone = xray.scatterer("S9", (0.9, 0.9, 0.9), u=0, scattering_type="S")
  p = os.path.join(d, "extra.tshb")
  tshb.write(p, recs + [random_atom("S9", 0, rng)._replace(key=bytes(gone.get_id_big(0).to_bytes()))])
  c = build_with_fallback(xs.unit_cell(), xs.scatterers(), p, xs.space_group(),
    True, xs.scattering_type_registry())
  extra = list(c.records_not_in_structure())
  assert len(extra) == 1 and extra[0].startswith("S9 (id "), extra
  print("\tversion 4 errors: repeated key, two records on one atom, mixed keys, unmatched record"
        " reported, missing atom refused: '...%s'" % msg[msg.index("has no density"):])


def exercise_spherical_only_atom_cached():
  """ A Thakkar-filled atom: Slater and Gaussian records, no shells. Under the
  orbit cache it folds to the single a_00, for X-ray and ED; the ED value
  uses its Z. Cached, uncached and the analytic value agree. """
  slaters = [(30.0, 0, 9.0), (4.0, 2, 3.5)]
  gaussians = [(2.5, 3.0)]
  rec = tshb.atom("X1", [], gaussians, slaters, z=6)
  other = random_atom("X2", 3, random.Random(9))._replace(z=8)
  d = tempfile.mkdtemp()
  px, pe = os.path.join(d, "x.tshb"), os.path.join(d, "e.tshb")
  tshb.write(px, [rec, other])
  tshb.write(pe, [rec, other], header={"ED": "true"})
  def f_x(s):
    from scipy.integrate import quad
    rho = lambda r: (sum(c*math.exp(-a*r*r) for c, a in gaussians)
                     + sum(c*r**n*math.exp(-z*r) for c, n, z in slaters))
    k = 2*math.pi*s
    return quad(lambda r: 4*math.pi*r*r*rho(r)*math.sin(k*r)/(k*r), 0, 60, limit=400)[0]
  worst = 0
  for symbol in ("P 1 21/c 1", "F m -3 m"):
    sgi = sgtbx.space_group_info(symbol)
    cs = crystal.symmetry(sgi.any_compatible_unit_cell(volume=1500), space_group_info=sgi)
    xs = xray.structure(crystal_symmetry=cs, scatterers=flex.xray_scatterer(
      [xray.scatterer(a, (0.11, 0.23, 0.37), u=0, scattering_type="C") for a in ("X1", "X2")]))
    for budget in ("0", None):
      if budget is None: os.environ.pop("SMTBX_TSHB_CACHE_MB", None)
      else: os.environ["SMTBX_TSHB_CACHE_MB"] = budget
      cx = build(cs.unit_cell(), xs.scatterers(), px, cs.space_group(), True)
      ce = build(cs.unit_cell(), xs.scatterers(), pe, cs.space_group(), True)
      os.environ.pop("SMTBX_TSHB_CACHE_MB", None)
      for h in [(1, 1, 1), (2, 0, 0), (3, 1, 1), (1, 1, 1), (2, 0, 0)]:   # repeats hit
        s = math.sqrt(cs.unit_cell().d_star_sq(h))
        stl2 = (s/2)**2
        fx0 = f_x(s)
        fe0 = 0.023934*(6 - fx0)/stl2
        for v in cx.get_full(0, h):
          worst = max(worst, abs(v - fx0)/max(1, abs(fx0)))
        for v in ce.get_full(0, h):
          worst = max(worst, abs(v - fe0)/max(1, abs(fe0)))
  assert worst < 1e-10, worst
  print("\tspherical-only atom, cached and uncached, X-ray and ED (its Z): %.1e" % worst)


def exercise_symmetry_moves():
  """ Atoms moved onto a symmetry image after the .tshb was made (move,
  compaq, inv): found there by id and rotated. The crystal is the same, so Fc
  must not change; inv -f of a whole acentric structure turns Fc into its
  conjugate. Two-fold operations are their own inverse and cannot tell the
  rotation from its transpose, so 3-, 4- and -4-fold ones are there too, in
  oblique cells. """
  rng = random.Random(11)
  def structure(cs, sites):
    # not snapped onto a special position: the atom near a centre stays put
    return xray.structure(special_position_settings=crystal.special_position_settings(
      cs, min_distance_sym_equiv=0.1), scatterers=flex.xray_scatterer(
      [xray.scatterer(l, site, u=0.02, scattering_type=t) for l, t, site in sites]))
  def fc_of(xs, path, h_list):
    c = build_with_fallback(xs.unit_cell(), xs.scatterers(), path, xs.space_group(),
      True, xs.scattering_type_registry())
    f = direct.f_calc_modulus_squared(xs, scatterer_contribution=c)
    out = []
    for h in h_list:
      f.linearise(h); out.append(f.f_calc)
    return c, out
  sites = [("O1", "O", (0.13, 0.27, 0.41)), ("N2", "N", (0.31, 0.12, 0.18)),
           ("C3", "C", (0.62, 0.08, 0.77))]
  h_list = [(1, 2, 3), (-2, 0, 4), (3, -5, 1), (0, 1, 7), (4, 3, -6), (-5, 2, 2)]
  cases = [
    # 2_1 screw with a lattice translation, the inversion, the c-glide
    ((7.1, 9.3, 11.2, 90, 103.4, 90), "P 1 21/c 1",
     ["-x+1,y+1/2,-z+1/2", "-x,-y,-z", "x,-y+1/2,z+1/2+1"]),
    # 3_1 and its square, hexagonal axes
    ((8.3, 8.3, 12.1, 90, 90, 120), "P 31",
     ["-y,x-y,z+1/3", "-x+y,-x,z+2/3", "-y+1,x-y,z+1/3"]),
    # 4-fold and -4-fold (improper, not an involution)
    ((9.1, 9.1, 6.7, 90, 90, 90), "P 4/n :2",
     ["-y+1/2,x,z", "y+1/2,-x,-z+1", "-x,-y,-z"]),
    # the cube's body diagonal
    ((10.3, 10.3, 10.3, 90, 90, 90), "P 21 3",
     ["z,x,y", "y+1/2,-z+1/2,-x", "x,y,z"]),
  ]
  # an atom 0.22 A from a centre, put on its image 0.44 A away: within the
  # tolerance of where it was, but the image is nearer and must win
  near_centre = [("O1", "O", (0.02, 0.01, 0.015)), ("N2", "N", (0.31, 0.12, 0.18)),
                 ("C3", "C", (0.62, 0.08, 0.77))]
  cases.append(((7.1, 9.3, 11.2, 80, 103.4, 95), "P -1", ["-x,-y,-z", "x,y,z", "x,y,z"],
                near_centre))
  worst = 0
  for case in cases:
    cell, sg, moves = case[:3]
    cs = crystal.symmetry(cell, sg)
    xs = structure(cs, case[3] if len(case) > 3 else sites)
    path = os.path.join(tempfile.mkdtemp(), "before.tshb")
    tshb.write(path, [keyed(xs, j, random_atom("x", 4, rng)._replace(z=8 - j))
                      for j in range(3)])
    moved = xs.deep_copy_scatterers()
    for sc, op in zip(moved.scatterers(), moves):
      sc.site = sgtbx.rt_mx(op)*sc.site
    _, f0 = fc_of(xs, path, h_list)
    c, f1 = fc_of(moved, path, h_list)
    n_moved = len([op for op in moves if not sgtbx.rt_mx(op).is_unit_mx()])
    assert len(c.records_moved_by_symmetry()) == n_moved, (sg, list(c.records_moved_by_symmetry()))
    for a, b in zip(f0, f1):
      worst = max(worst, abs(a - b)/abs(a))
  # inv -f in P2_12_12_1 and in P3_1, which becomes P3_2 (no centre: the
  # inversion composed with the group's operations): the enantiomer, Fc its
  # conjugate. The two H of a CH2 swap places, as the riding constraint
  # makes them after an improper operation: densities follow the sites.
  with_h = sites + [("H4", "H", (0.70, 0.15, 0.80)), ("H5", "H", (0.55, 0.02, 0.83))]
  for cell, sg in (((7.1, 9.3, 11.2, 90, 90, 90), "P 21 21 21"),
                   ((8.3, 8.3, 12.1, 90, 90, 120), "P 31")):
    cs = crystal.symmetry(cell, sg)
    xs = structure(cs, with_h)
    path = os.path.join(tempfile.mkdtemp(), "acentric.tshb")
    tshb.write(path, [keyed(xs, j, random_atom("x", 3, rng)._replace(z=(8, 7, 6, 1, 1)[j]))
                      for j in range(5)])
    other_hand = cs.space_group_info().change_hand()
    assert str(other_hand.type().change_of_hand_op()) == "-a,-b,-c"
    inverted_sites = [(l, t, tuple(-x for x in site)) for l, t, site in with_h]
    inverted_sites[3], inverted_sites[4] = (
      ("H4",) + inverted_sites[4][1:], ("H5",) + inverted_sites[3][1:])
    inverted = structure(crystal.symmetry(cell, space_group_info=other_hand),
                         inverted_sites)
    _, f0 = fc_of(xs, path, h_list)
    c, f1 = fc_of(inverted, path, h_list)
    assert len(c.records_moved_by_symmetry()) == 5
    for a, b in zip(f0, f1):
      worst = max(worst, abs(a.conjugate() - b)/abs(a))
  assert worst < 1e-8, worst
  print("\tsymmetry moves (2_1, -1, c, 3_1, 4, -4, body-diagonal 3, image nearer"
        " than the site; inv -f, CH2 hydrogens swapped): Fc unchanged %.1e" % worst)


def exercise_fractional_slater_n():
  """ A Slater term with n = 1.5 (and -0.5), against the radial integral. """
  from scipy.integrate import quad
  slaters = [(3.0, 1.5, 2.7), (0.8, -0.5, 4.1), (1.2, 2, 3.3)]
  path = os.path.join(tempfile.mkdtemp(), "half.tshb")
  tshb.write(path, [tshb.atom("X1", [], (), slaters)])
  cs = crystal.symmetry((20, 20, 20, 90, 90, 90), "P1")
  xs = xray.structure(crystal_symmetry=cs, scatterers=flex.xray_scatterer(
    [xray.scatterer("X1", (0, 0, 0), u=0, scattering_type="C")]))
  c = build(cs.unit_cell(), xs.scatterers(), path, cs.space_group(), True)
  rho = lambda r: sum(cc*r**n*math.exp(-z*r) for cc, n, z in slaters)
  worst = 0
  for h in [(0, 0, 0), (1, 0, 0), (3, 4, 0), (9, 7, 5), (20, 15, 11)]:
    s = math.sqrt(sum(x*x for x in h))/20.
    k = 2*math.pi*s
    if k == 0:
      f = quad(lambda r: 4*math.pi*r*r*rho(r), 0, 60, limit=400)[0]
    else:
      f = quad(lambda r: 4*math.pi*r*r*rho(r)*math.sin(k*r)/(k*r), 0, 60, limit=400)[0]
    worst = max(worst, abs(c.get_full(0, h)[0] - f)/max(1, abs(f)))
  assert worst < 1e-8, worst
  print("\tSlater n = 1.5 and -0.5 against the radial integral: %.1e" % worst)


def exercise_disorder_gradients():
  """ PART 1/2 sharing one occupancy variable, with positions and ADPs, all
  off one id-keyed .tshb: analytic gradients against finite differences, and
  the normal matrix against the same densities written as a table. """
  from cctbx import adptbx
  from smtbx.refinement import least_squares, constraints
  from smtbx.refinement.constraints import occupancy
  import smtbx.utils
  rng = random.Random(12)
  xs = disordered_setup()
  for sc in xs.scatterers():
    sc.flags.set_use_u_iso(False); sc.flags.set_use_u_aniso(True)
    sc.u_star = adptbx.u_cart_as_u_star(xs.unit_cell(), (0.02, 0.025, 0.03, 0.002, -0.001, 0.003))
    sc.flags.set_grad_site(True); sc.flags.set_grad_u_aniso(True)
    sc.flags.set_grad_occupancy(sc.get_part() != 0)
  d = tempfile.mkdtemp()
  records = [keyed(xs, j, random_atom("x", 3, rng)._replace(z=z)) for j, z in enumerate((8, 7, 6, 6))]
  path = os.path.join(d, "dis.tshb")
  tshb.write(path, records)
  c = build_with_fallback(xs.unit_cell(), xs.scatterers(), path, xs.space_group(),
    True, xs.scattering_type_registry())
  # finite differences of |Fc|^2 against grad_observable, parameter by parameter
  def observable(structure, h):
    f = direct.f_calc_modulus_squared(structure, scatterer_contribution=c)
    f.linearise(h)
    return f.observable, list(f.grad_observable)
  worst = 0
  for h in [(1, 2, 3), (-2, 1, 4), (3, -3, 2)]:
    y0, grad = observable(xs, h)
    q = 0
    for j, sc in enumerate(xs.scatterers()):
      params = [("site", k, 1e-6) for k in range(3)] + [("u_star", k, 1e-7) for k in range(6)]
      if sc.flags.grad_occupancy(): params.append(("occupancy", 0, 1e-6))
      for kind, k, eps in params:
        vals = []
        for sign in (1, -1):
          t = xs.deep_copy_scatterers()
          s2 = t.scatterers()[j]
          if kind == "site":
            v = list(s2.site); v[k] += sign*eps; s2.site = tuple(v)
          elif kind == "u_star":
            v = list(s2.u_star); v[k] += sign*eps; s2.u_star = tuple(v)
          else:
            s2.occupancy += sign*eps
          vals.append(observable(t, h)[0])
        fd = (vals[0] - vals[1])/(2*eps)
        worst = max(worst, abs(fd - grad[q])/max(1, abs(grad[q])))
        q += 1
    assert q == len(grad), (q, len(grad))
  assert worst < 1e-5, worst
  # the same densities as a table, keyed by the same ids
  fo = xs.structure_factors(d_min=1.0).f_calc()
  table = os.path.join(d, "dis.tsc")
  ids = [sc.get_id_big(sc.get_part()).to_hex_string() for sc in xs.scatterers()]
  rows = set()
  for h in fo.indices():
    for op in xs.space_group().smx():
      r = op.r().num()
      hr = tuple(sum(h[k]*r[3*k + cc] for k in range(3)) for cc in range(3))
      rows.add(hr); rows.add(tuple(-v for v in hr))
  with open(table, "w") as out:
    out.write("TITLE: disorder\nSYMM: expanded\nSCATTERER_IDS: %s\nDATA:\n" % " ".join(ids))
    for hr in sorted(rows):
      vals = [c.get_full(j, hr)[0] for j in range(4)]
      out.write("%d %d %d %s\n" % (hr + (" ".join("%.17g,%.17g" % (v.real, v.imag) for v in vals),)))
  fo_sq = fo.norm()
  fo_sq = fo_sq.customized_copy(sigmas=flex.double(fo_sq.size(), 1.))
  fo_sq.set_observation_type_xray_intensity()
  def normal_matrix(file_name):
    rep = constraints.reparametrisation(structure=xs,
      constraints=[occupancy.dependent_occupancy([(2, 1.0)], [(3, 1.0)])],
      connectivity_table=smtbx.utils.connectivity_table(xs))
    ls = least_squares.crystallographic_ls(fo_sq.as_xray_observations(), rep,
      weighting_scheme=least_squares.unit_weighting(), may_parallelise=False)
    contribution = build_with_fallback(xs.unit_cell(), xs.scatterers(), file_name,
      xs.space_group(), True, xs.scattering_type_registry())
    ls.one_h_linearisation = least_squares.f_calc_function_default(
      direct.f_calc_modulus_squared(xs, scatterer_contribution=contribution))
    ls.build_up()
    return ls.n_parameters, ls.normal_matrix_packed_u().deep_copy()
  n_a, a = normal_matrix(path)
  n_b, b = normal_matrix(table)
  assert n_a == n_b
  dm = flex.max(flex.abs(a - b))/flex.max(flex.abs(a))
  assert dm < 1e-12, dm
  print("\tdisorder, shared occupancy + sites + ADPs: gradients against finite"
        " differences %.1e; normal matrix (%d parameters) against the table %.1e"
        % (worst, n_a, dm))


def exercise_corrupt_files():
  """ Truncated at every byte, bytes flipped at random, absurd counts: every
  version reads to a Python exception, never a crash or a huge allocation. """
  import struct
  rng = random.Random(13)
  xs = disordered_setup()
  d = tempfile.mkdtemp()
  base = [keyed(xs, j, random_atom("x", 2, rng)._replace(
            slaters=[(0.3, 1, 2.5)], gaussians=[(0.2, 3.0)], exponentials=[(0.1, 0.5)], z=6))
          for j in range(4)]
  files = {}
  for version in (1, 2, 3, 4):
    if version == 1:
      recs = [r._replace(key=None, part=0, slaters=(), gaussians=(), exponentials=(), z=0) for r in base]
    elif version == 2:
      recs = [r._replace(key=None, part=0, z=0) for r in base]
    elif version == 3:
      recs = [r._replace(key=None, part=0) for r in base]
    else:
      recs = base
    if version < 4:     # label-keyed: the two C3 need distinct labels
      recs = [r._replace(label="A%d" % j) for j, r in enumerate(recs)]
    p = os.path.join(d, "v%d.tshb" % version)
    tshb.write(p, recs)
    files[version] = open(p, "rb").read()
  bad = os.path.join(d, "bad.tshb")
  def attempt(data):
    with open(bad, "wb") as f:
      f.write(data)
    try:
      build_with_fallback(xs.unit_cell(), xs.scatterers(), bad, xs.space_group(),
        True, xs.scattering_type_registry(), allow_spherical_fallback=True)
      return 0
    except (RuntimeError, ValueError, IndexError, OverflowError) as e:
      return 1
  n_cases = n_errors = n_truncations = 0
  for version, data in files.items():
    for cut in range(len(data)):            # every truncation
      n_errors += attempt(data[:cut]); n_cases += 1; n_truncations += 1
    for t in range(300):                    # random byte flips
      b = bytearray(data)
      for k in range(rng.randint(1, 4)):
        b[rng.randrange(len(b))] = rng.randrange(256)
      attempt(bytes(b)); n_cases += 1
    # absurd counts: every u32 field in turn set to 0xffffffff and 0x7fffffff
    for pos in range(4, len(data) - 3):
      for v in (0xffffffff, 0x7fffffff):
        b = bytearray(data); b[pos:pos + 4] = struct.pack("<I", v)
        attempt(bytes(b)); n_cases += 1
    # NaN and infinity over every 8 bytes in turn
    for pos in range(8, len(data) - 7):
      for v in (float("nan"), float("inf")):
        b = bytearray(data); b[pos:pos + 8] = struct.pack("<d", v)
        attempt(bytes(b)); n_cases += 1
  assert n_errors == n_truncations, (n_errors, n_truncations)
  # NaN or infinity in exactly one number of a record: refused, whichever
  n_non_finite = 0
  record = base[0]
  shells = list(record.shells)
  for v in (float("nan"), float("inf"), -float("inf")):
    variants = [
      dict(shells=[shells[0]._replace(alpha=v)] + shells[1:]),
      dict(shells=[shells[0]] + [shells[1]._replace(c=[v] + list(shells[1].c[1:]))] + shells[2:]),
      dict(gaussians=[(v, 3.0)]), dict(gaussians=[(0.2, v)]),
      dict(slaters=[(v, 1, 2.5)]), dict(slaters=[(0.3, v, 2.5)]), dict(slaters=[(0.3, 1, v)]),
      dict(exponentials=[(v, 0.5)]), dict(exponentials=[(0.1, v)])]
    for change in variants:
      tshb.write(bad, [record._replace(**change)] + base[1:])
      assert attempt(open(bad, "rb").read()[:]) == 1, change
      n_non_finite += 1
  print("\tcorrupt files, versions 1-4: %d cases, no crash; all %d truncations"
        " and %d NaN/inf numbers refused" % (n_cases, n_truncations, n_non_finite))


def exercise_uncovered_atom_before_d_star_sq():
  """ get/get_full from Python on an atom the file misses, before
  at_d_star_sq: an error, not a read of an empty array (it segfaulted). """
  rng = random.Random(15)
  xs = disordered_setup()
  p = os.path.join(tempfile.mkdtemp(), "part.tshb")
  tshb.write(p, [keyed(xs, j, random_atom("x", 1, rng)._replace(z=6)) for j in range(3)])
  c = build_with_fallback(xs.unit_cell(), xs.scatterers(), p, xs.space_group(),
    True, xs.scattering_type_registry(), allow_spherical_fallback=True)
  assert list(c.scatterers_not_in_table()) == [3]
  for call in (c.get_full, c.get):
    try:
      call(3, (1, 2, 3))
    except RuntimeError as e:
      assert "at_d_star_sq" in str(e)
    else:
      raise AssertionError("no error before at_d_star_sq")
  c.get_full(0, (1, 2, 3))              # a covered atom needs nothing
  c.at_d_star_sq(xs.unit_cell().d_star_sq((1, 2, 3)))
  f = c.get_full(3, (1, 2, 3))[0]
  assert f.real > 0 and f.imag == 0
  print("\tuncovered atom before at_d_star_sq: refused, not a crash")


def run():
  exercise_transform_against_numerical_integral()
  exercise_spherical_records()
  exercise_symmetry_and_structure_factor()
  exercise_electron_diffraction()
  exercise_cache()
  exercise_spherical_only_atom_cached()
  exercise_version_4_ids_and_merge()
  exercise_version_4_errors()
  exercise_symmetry_moves()
  exercise_fractional_slater_n()
  exercise_disorder_gradients()
  exercise_corrupt_files()
  exercise_uncovered_atom_before_d_star_sq()
  exercise_spherical_equivalence()
  exercise_least_squares_build()
  print("OK")


if __name__ == '__main__':
  run()
