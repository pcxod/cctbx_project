""" .tshb: atomic densities as Gaussians times real spherical harmonics.

The file format, and the form factors read from it, are described in
smtbx/structure_factors/direct/harmonic_expansion.h. This module writes and
reads it, so that a density from any source - NoSpherA2's RI fit, a SALTED
prediction, a multipole model - can be handed to olex2.refine in place of a
.tsc/.tscb table:

  from smtbx.structure_factors.direct import tshb
  tshb.write("x.tshb", [tshb.atom("C1", [tshb.shell(l, alpha, c), ...]), ...])

Files are merged the way NoSpherA2 merges tables: a tshb_block collects
records keyed like the columns of a .tscb - by scatterer id, or by label when
a record has none - and append and += keep the first record of a key and drop
later ones (the "first wins" of NoSpherA2's tsc_block). So the PART 0 atoms
every per-part density holds come from the first part, as in the merged table:

  joint = tshb.tshb_block.read("part1.tshb")
  joint += tshb.tshb_block.read("part2.tshb")
  joint.write("joint.tshb")

Units are Angstrom and electrons per cubic Angstrom, and the frame is the
Cartesian frame of the unit cell (cctbx's orthogonalisation). from_bohr()
converts shells written in atomic units, which is how NoSpherA2 holds them.
"""
from __future__ import absolute_import, division, print_function

import struct
from collections import namedtuple

shell = namedtuple("shell", "l alpha c")      # c: 2l+1 values, m = -l..l
# spherical extras (version 2): Gaussians c exp(-a r^2) as (c, a); Slater terms
# c r^n exp(-zeta r) as (c, n, zeta), n real, -2 < n <= 60; form factor terms
# a exp(-b s) as (a, b)
# z (version 3): the nuclear charge, which Mott-Bethe needs with header {"ED": "true"}
# key (version 4): None, keyed by label; or the 16 bytes of the scatterer id,
# cctbx.xray scatterer.get_id_big(part).to_bytes() - see scatterer_id()
# part (version 4): the PART number
atom = namedtuple("atom", "label shells gaussians slaters exponentials z key part")
atom.__new__.__defaults__ = ((), (), (), 0, None, 0)

KEY_LABEL, KEY_ID = 0, 1


def scatterer_id(scatterer):
  """ The id a .tscb and a version 4 .tshb key this scatterer by: element,
  PART and coordinate, as the 16 bytes of cctbx's scatterer_id_big. """
  return bytes(scatterer.get_id_big(scatterer.get_part()).to_bytes())


def record_key(record):
  """ What a record is matched and merged by: its id, or failing that its
  label, case-insensitively, as the readers match labels. """
  if record.key is not None:
    return ("id", bytes(record.key))
  return ("label", record.label.upper())

BOHR = 0.529177210903     # Angstrom


def from_bohr(l, alpha, c):
  """ A shell n exp(-alpha r^2) r^l sum_m c_m Y_lm in bohr and e/bohr^3,
  re-expressed in Angstrom and e/A^3. Fold any normalisation n into c first.
  """
  scale = 1/BOHR**(l + 3)
  return shell(l, alpha/BOHR**2, [x*scale for x in c])


def write(file_name, atoms, header=None, version_override=None):
  """ atoms: a sequence of atom(label, [shell(l, alpha, c), ...]).
  header: extra "key: value" lines, a dict; "AD: false" is honoured by the
  reader as in a .tsc.
  """
  lines = ["FRAME: cartesian", "UNITS: angstrom"]
  for k, v in (header or {}).items():
    lines.append("%s: %s" % (k, v))
  text = ("\n".join(lines)).encode("utf-8")
  version = 4 if any(a.key is not None or a.part for a in atoms) else \
    3 if any(a.z for a in atoms) else \
    2 if any(a.gaussians or a.slaters or a.exponentials for a in atoms) else 1
  if version_override is not None:
    assert version_override >= version, (version_override, version)
    version = version_override
  with open(file_name, "wb") as f:
    f.write(b"TSHB")
    f.write(struct.pack("<II", version, len(text)))
    f.write(text)
    f.write(struct.pack("<I", len(atoms)))
    for a in atoms:
      label = a.label.encode("utf-8")
      f.write(struct.pack("<I", len(label)))
      f.write(label)
      if version >= 4:
        if a.key is None:
          f.write(struct.pack("<B", KEY_LABEL))
        else:
          key = bytes(a.key)
          assert len(key) == 16, (a.label, len(key))
          f.write(struct.pack("<B", KEY_ID))
          f.write(key)
        f.write(struct.pack("<i", a.part))
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
  assert version in (1, 2, 3, 4), version
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
    key, part = None, 0
    if version >= 4:
      tag, = struct.unpack_from("<B", data, pos); pos += 1
      assert tag in (KEY_LABEL, KEY_ID), (label, tag)
      if tag == KEY_ID:
        key = data[pos:pos + 16]; pos += 16
      part, = struct.unpack_from("<i", data, pos); pos += 4
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
    atoms.append(atom(label, shells, *extras, z=z, key=key, part=part))
  return header, atoms


class tshb_block(object):
  """ The records of a .tshb, merged as NoSpherA2 merges tables.

  One record per key (record_key: the scatterer id, else the label). append
  keeps the first record of a key and refuses later ones, so += of per-part
  files takes the PART 0 atoms from the first part, as tsc_block does. The
  header of the first block is kept; += adds keys it lacks. write() writes
  version 4.
  """

  def __init__(self, records=(), header=None):
    self.header = dict(header or {})
    self.records = []
    self._keys = {}
    for r in records:
      self.append(r)

  def append(self, record):
    """ Add a record unless its key is already held; True if added. """
    k = record_key(record)
    if k in self._keys:
      return False
    self._keys[k] = len(self.records)
    self.records.append(record)
    return True

  def __iadd__(self, other):
    records = other.records if isinstance(other, tshb_block) else other
    for r in records:
      self.append(r)
    if isinstance(other, tshb_block):
      for k, v in other.header.items():
        self.header.setdefault(k, v)
    return self

  def __add__(self, other):
    result = tshb_block(self.records, self.header)
    result += other
    return result

  def __len__(self):
    return len(self.records)

  def __iter__(self):
    return iter(self.records)

  def __contains__(self, record_or_key):
    k = record_or_key if isinstance(record_or_key, tuple) and \
      record_or_key[:1] in (("id",), ("label",)) else record_key(record_or_key)
    return k in self._keys

  def write(self, file_name):
    header = dict(self.header)
    for k in ("FRAME", "UNITS"):
      header.pop(k, None)      # write() puts these first
    write(file_name, self.records, header=header, version_override=4)

  @staticmethod
  def read(file_name):
    """ A file holds a key once; a repeat is an error here as in the reader. """
    header, records = read(file_name)
    for k in ("FRAME", "UNITS"):
      header.pop(k, None)
    block = tshb_block(header=header)
    for r in records:
      if not block.append(r):
        raise ValueError("%s holds the key of %s twice" % (file_name, r.label))
    return block


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
