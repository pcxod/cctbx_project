""" .tshb: atomic densities as Gaussians times real spherical harmonics.

The file format, and the form factors read from it, are described in
smtbx/structure_factors/direct/harmonic_expansion.h. This module writes and
reads it, so that a density from any source - NoSpherA2's RI fit, a SALTED
prediction, a multipole model - can be handed to olex2.refine in place of a
.tsc/.tscb table:

  from smtbx.structure_factors.direct import tshb
  tshb.write("x.tshb", [tshb.atom("C1", [tshb.shell(l, alpha, c), ...]), ...])

Units are Angstrom and electrons per cubic Angstrom, and the frame is the
Cartesian frame of the unit cell (cctbx's orthogonalisation). from_bohr()
converts shells written in atomic units, which is how NoSpherA2 holds them.
"""
from __future__ import absolute_import, division, print_function

import struct
from collections import namedtuple

shell = namedtuple("shell", "l alpha c")      # c: 2l+1 values, m = -l..l
# spherical extras (version 2): Gaussians c exp(-a r^2) as (c, a); Slater terms
# c r^n exp(-zeta r) as (c, n, zeta); form factor terms a exp(-b s) as (a, b)
# z (version 3): the nuclear charge, which Mott-Bethe needs with header {"ED": "true"}
atom = namedtuple("atom", "label shells gaussians slaters exponentials z")
atom.__new__.__defaults__ = ((), (), (), 0)

BOHR = 0.529177210903     # Angstrom


def from_bohr(l, alpha, c):
  """ A shell n exp(-alpha r^2) r^l sum_m c_m Y_lm in bohr and e/bohr^3,
  re-expressed in Angstrom and e/A^3. Fold any normalisation n into c first.
  """
  scale = 1/BOHR**(l + 3)
  return shell(l, alpha/BOHR**2, [x*scale for x in c])


def write(file_name, atoms, header=None):
  """ atoms: a sequence of atom(label, [shell(l, alpha, c), ...]).
  header: extra "key: value" lines, a dict; "AD: false" is honoured by the
  reader as in a .tsc.
  """
  lines = ["FRAME: cartesian", "UNITS: angstrom"]
  for k, v in (header or {}).items():
    lines.append("%s: %s" % (k, v))
  text = ("\n".join(lines)).encode("utf-8")
  version = 3 if any(a.z for a in atoms) else \
    2 if any(a.gaussians or a.slaters or a.exponentials for a in atoms) else 1
  with open(file_name, "wb") as f:
    f.write(b"TSHB")
    f.write(struct.pack("<II", version, len(text)))
    f.write(text)
    f.write(struct.pack("<I", len(atoms)))
    for a in atoms:
      label = a.label.encode("utf-8")
      f.write(struct.pack("<I", len(label)))
      f.write(label)
      f.write(struct.pack("<I", len(a.shells)))
      for s in a.shells:
        assert len(s.c) == 2*s.l + 1, (a.label, s.l, len(s.c))
        f.write(struct.pack("<id", s.l, s.alpha))
        f.write(struct.pack("<%dd" % len(s.c), *s.c))
      if version >= 2:
        f.write(struct.pack("<I", len(a.gaussians)))
        for c, alpha in a.gaussians:
          f.write(struct.pack("<dd", c, alpha))
        f.write(struct.pack("<I", len(a.slaters)))
        for c, n, zeta in a.slaters:
          f.write(struct.pack("<ddd", c, n, zeta))
        f.write(struct.pack("<I", len(a.exponentials)))
        for a_, b in a.exponentials:
          f.write(struct.pack("<dd", a_, b))
      if version >= 3:
        f.write(struct.pack("<i", a.z))


def read(file_name):
  """ (header dict, [atom, ...]) """
  with open(file_name, "rb") as f:
    data = f.read()
  assert data[:4] == b"TSHB", "not a .tshb file"
  version, n = struct.unpack_from("<II", data, 4)
  assert version in (1, 2, 3), version
  pos = 12
  header = {}
  for line in data[pos:pos + n].decode("utf-8").split("\n"):
    if ":" in line:
      k, v = line.split(":", 1)
      header[k.strip()] = v.strip()
  pos += n
  n_atoms, = struct.unpack_from("<I", data, pos); pos += 4
  atoms = []
  for i in range(n_atoms):
    n, = struct.unpack_from("<I", data, pos); pos += 4
    label = data[pos:pos + n].decode("utf-8"); pos += n
    n_shells, = struct.unpack_from("<I", data, pos); pos += 4
    shells = []
    for j in range(n_shells):
      l, alpha = struct.unpack_from("<id", data, pos); pos += 12
      c = list(struct.unpack_from("<%dd" % (2*l + 1), data, pos))
      pos += 8*(2*l + 1)
      shells.append(shell(l, alpha, c))
    extras = [[], [], []]
    if version >= 2:
      for kind, width in ((0, 2), (1, 3), (2, 2)):
        n, = struct.unpack_from("<I", data, pos); pos += 4
        for k in range(n):
          extras[kind].append(struct.unpack_from("<%dd" % width, data, pos))
          pos += 8*width
    z = 0
    if version >= 3:
      z, = struct.unpack_from("<i", data, pos); pos += 4
    atoms.append(atom(label, shells, *extras, z=z))
  return header, atoms


def spherical_atom(label, gaussian):
  """ The density of a spherical form factor given as a cctbx gaussian
  sum a_k exp(-b_k stol^2) + c, as l = 0 shells.

  a exp(-b s^2/4) is the transform of an l = 0 shell of exponent 4 pi^2/b; the
  constant c, a point charge, becomes a very narrow one. Used to check the
  format against the spherical form factors, and as the fallback a writer can
  put in for an atom it has no density for.
  """
  import math
  y00 = 1/math.sqrt(4*math.pi)
  shells = []
  terms = list(zip(gaussian.array_of_a(), gaussian.array_of_b()))
  if gaussian.c() != 0:
    terms.append((gaussian.c(), 1e-10))
  for a, b in terms:
    alpha = 4*math.pi**2/b
    shells.append(shell(0, alpha, [a/(y00*(math.pi/alpha)**1.5)]))
  return atom(label, shells)
