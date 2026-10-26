""" .tshb densities (Gaussians times real spherical harmonics) as form factors.

What is checked:
- the analytic transform against a numerical Fourier integral of the density,
  with the harmonics evaluated independently (scipy's complex harmonics);
- symmetry equivalents, and the whole structure factor against a sum written
  out by hand;
- that spherical atoms written as .tshb give the spherical structure factors
  and gradients, with and without atoms missing from the file;
- that a least-squares build with them is the spherical one.
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
        xs.space_group(), True, xs.scattering_type_registry())
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


def run():
  exercise_transform_against_numerical_integral()
  exercise_spherical_records()
  exercise_symmetry_and_structure_factor()
  exercise_electron_diffraction()
  exercise_cache()
  exercise_spherical_equivalence()
  exercise_least_squares_build()
  print("OK")


if __name__ == '__main__':
  run()
