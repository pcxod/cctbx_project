"""Read a (3+1)D model and its hklm data from an msCIF (JANA2006/Jana2020 output) into smtbx.modulation objects.

Supported: average positions, Uiso/Uaniso, occupancy, displacive, occupational and ADP Fourier waves (sin/cos or
modulus/phase form), f'/f'', the superspace operators as listed, refln loops with one m index (in the CIF or a
separate fcf). Crenel, sawtooth, rigid groups and (3+d)D with d > 1 raise NotImplementedError.
"""
from __future__ import absolute_import, division, print_function

import math

import numpy as np

from cctbx import adptbx, sgtbx, uctbx
from smtbx import modulation as mod


def _num(v):
  v = str(v)
  if v in ("?", "."):
    return None
  return float(v.split("(")[0])


def _col(block, key, default=None):
  if key not in block:
    return default
  v = block[key]
  return [v] if isinstance(v, str) else list(v)


def _block_with(model, key):
  for block in model.values():
    if key in block:
      return block
  return None


def _harmonics(block, q):
  """wave vector seq_id -> harmonic order n (negative: the wave vector is -n q)."""
  ids = _col(block, "_atom_site_Fourier_wave_vector_seq_id", [])
  coeff = _col(block, "_jana_atom_site_fourier_wave_vector_q1_coeff")
  if coeff:
    return {i: int(_num(c)) for i, c in zip(ids, coeff)}
  xyz = [_col(block, "_atom_site_Fourier_wave_vector_" + a, ["0"] * len(ids)) for a in "xyz"]
  if ids and any(a in block for a in ("_atom_site_Fourier_wave_vector_x", "_atom_site_Fourier_wave_vector_z")):
    # a component JANA leaves out (2104860 has no _y column) is 0
    return {i: int(round(np.dot([_num(xyz[j][k]) or 0.0 for j in range(3)], q) / q.dot(q)))
            for k, i in enumerate(ids)}
  return {"1": 1}


def _sin_cos(block, prefix, n):
  """(sin, cos) columns of a Fourier loop, from either the sin/cos or the modulus/phase form; sin flips for n < 0."""
  s = _col(block, prefix + "_param_sin")
  c = _col(block, prefix + "_param_cos")
  if s is None:
    mod_, ph = _col(block, prefix + "_param_modulus"), _col(block, prefix + "_param_phase")
    # |A| cos(2 pi (k r + phi)) = |A| cos(2 pi phi) cos - |A| sin(2 pi phi) sin
    c = [_num(a) * math.cos(2 * math.pi * _num(p)) for a, p in zip(mod_, ph)]
    s = [-_num(a) * math.sin(2 * math.pi * _num(p)) for a, p in zip(mod_, ph)]
  else:
    s, c = [_num(v) or 0.0 for v in s], [_num(v) or 0.0 for v in c]
  return [math.copysign(1, k) * v for v, k in zip(s, n)], c


def read(cif_path, fcf_path=None):
  """(modulated_structure, data) with data = dict(h, m, f_sq_meas, f_sq_sigma, f_sq_calc, extinction)."""
  import iotbx.cif
  model = iotbx.cif.reader(file_path=cif_path).model()
  block = _block_with(model, "_atom_site_label")
  if block is None:
    raise ValueError("no atoms in %s" % cif_path)
  if [v for v in _col(block, "_atom_site_rot_Fourier_atom_site_label", []) if v not in ("?", ".")]:
    raise NotImplementedError("%s uses rigid-group modulation" % cif_path)
  if len(set(_col(block, "_atom_site_subsystem_code", ["1"]))) > 1:
    raise NotImplementedError("%s is a composite structure (several subsystems)" % cif_path)
  cell = [_num(block["_cell_length_" + k]) for k in "abc"] + [_num(block["_cell_angle_" + k])
                                                             for k in ("alpha", "beta", "gamma")]
  uc = uctbx.unit_cell(cell)
  qx = _col(block, "_cell_wave_vector_x")
  if qx is None:
    raise ValueError("no wave vector in %s" % cif_path)
  if len(qx) > 1:
    raise NotImplementedError("(3+%d)D: only one wave vector is supported" % len(qx))
  q = np.array([_num(_col(block, "_cell_wave_vector_" + a)[0]) for a in "xyz"])
  symbols = _col(block, "_space_group_symop_ssg_operation_algebraic")
  if not symbols:
    raise ValueError("no superspace operators in %s" % cif_path)
  ssg = mod.superspace_group.from_ops(symbols, q)
  order = ssg.basic.order_z()
  harm = _harmonics(block, q)

  labels = _col(block, "_atom_site_label")
  types = _col(block, "_atom_site_type_symbol")
  xyz = [[_num(v) for v in _col(block, "_atom_site_fract_" + a)] for a in "xyz"]
  uiso = [_num(v) or 0.0 for v in _col(block, "_atom_site_U_iso_or_equiv", ["0"] * len(labels))]
  occ = [_num(v) if _num(v) is not None else 1.0 for v in _col(block, "_atom_site_occupancy", ["1"] * len(labels))]
  mult = _col(block, "_atom_site_site_symmetry_multiplicity")
  if mult is None or any(_num(v) is None for v in mult):
    xyz0 = [[_num(v) for v in _col(block, "_atom_site_fract_" + a)] for a in "xyz"]
    mult = [sgtbx.site_symmetry(uc, ssg.basic, (xyz0[0][k], xyz0[1][k], xyz0[2][k]), 0.1).multiplicity()
            for k in range(len(labels))]
  factor = [_num(v) / order for v in mult]
  # JANA exports occupancies either chemical (2108430: Sb1 on a 2-fold, occupancy 1) or with the site factor
  # mult/order folded in (2108042: every atom <= its factor, those on special positions exactly at it)
  if (all(o <= f + 1e-3 for o, f in zip(occ, factor))
      and any(f < 1 - 1e-3 and abs(o - f) < 1e-3 for o, f in zip(occ, factor))):
    occ = [o / f for o, f in zip(occ, factor)]
  aniso = {}
  if "_atom_site_aniso_label" in block:
    for k, lab in enumerate(_col(block, "_atom_site_aniso_label")):
      u = [_num(_col(block, "_atom_site_aniso_U_" + e)[k]) for e in ("11", "22", "33", "12", "13", "23")]
      aniso[lab] = adptbx.u_cif_as_u_star(uc, u)
  waves = {lab: {"d": {}, "o": {}, "u": {}} for lab in labels}
  if "_atom_site_displace_Fourier_atom_site_label" in block:
    p = "_atom_site_displace_Fourier"
    ns = [harm.get(i, 0) for i in _col(block, p + "_wave_vector_seq_id")]
    s, c = _sin_cos(block, p, ns)
    for lab, ax, n, sv, cv in zip(_col(block, p + "_atom_site_label"), _col(block, p + "_axis"), ns, s, c):
      if n == 0 or lab not in waves:   # JANA's placeholder rows of '?'
        continue
      d = waves[lab]["d"].setdefault(abs(n), [np.zeros(3), np.zeros(3)])
      k = {"x": 0, "y": 1, "z": 2, "1": 0, "2": 1, "3": 2}[ax.lower()]
      d[0][k] += sv
      d[1][k] += cv
  if "_atom_site_occ_Fourier_atom_site_label" in block:
    p = "_atom_site_occ_Fourier"
    # JANA: p(x4) = ai (a0 + sum), _atom_site_occupancy = ai a0, a0 listed per atom in this JANA loop; the
    # coefficients are therefore scaled by occupancy / a0 (2108430: a0 = 1, 1560711: a0 = occupancy)
    a0 = dict(zip(_col(block, "_jana_atom_site_occ_fourier_absolute_site_label", []),
                  [_num(v) for v in _col(block, "_jana_atom_site_occ_fourier_absolute", [])]))
    occ_of = dict(zip(labels, occ))
    ns = [harm.get(i, 0) for i in _col(block, p + "_wave_vector_seq_id")]
    s, c = _sin_cos(block, p, ns)
    for lab, n, sv, cv in zip(_col(block, p + "_atom_site_label"), ns, s, c):
      if n == 0 or lab not in waves:
        continue
      f = occ_of[lab] / a0[lab] if a0.get(lab) else 1.0
      waves[lab]["o"][abs(n)] = (f * sv, f * cv)
  if "_atom_site_U_Fourier_atom_site_label" in block:
    p = "_atom_site_U_Fourier"
    ns = [harm.get(i, 0) for i in _col(block, p + "_wave_vector_seq_id")]
    s, c = _sin_cos(block, p, ns)
    idx = {"u11": 0, "u22": 1, "u33": 2, "u12": 3, "u13": 4, "u23": 5}
    for lab, el, n, sv, cv in zip(_col(block, p + "_atom_site_label"), _col(block, p + "_tens_elem"), ns, s, c):
      if n == 0 or lab not in waves:
        continue
      d = waves[lab]["u"].setdefault(abs(n), [np.zeros(6), np.zeros(6)])
      d[0][idx[el.lower()]] += sv
      d[1][idx[el.lower()]] += cv

  # special functions; JANA writes empty loops as rows of '?'
  crenel, sawtooth, legendre = {}, {}, {}
  p = "_atom_site_occ_special_func"
  for lab, c, w in zip(_col(block, p + "_atom_site_label", []), _col(block, p + "_crenel_c", []),
                       _col(block, p + "_crenel_w", [])):
    if _num(c) is not None and _num(w) is not None:
      crenel[lab] = (_num(c), _num(w))
  p = "_atom_site_displace_special_func"
  for row in zip(*[_col(block, p + k, []) for k in ("_atom_site_label", "_sawtooth_ax", "_sawtooth_ay",
                                                     "_sawtooth_az", "_sawtooth_c", "_sawtooth_w")]):
    if _num(row[4]) is not None:
      sawtooth[row[0]] = (np.array([_num(v) or 0.0 for v in row[1:4]]), _num(row[4]), _num(row[5]))
  p = "_jana_atom_site_displace_legendre"
  for lab, ax, order, coeff in zip(_col(block, p + "_atom_site_label", []), _col(block, p + "_axis", []),
                                   _col(block, p + "_param_order", []), _col(block, p + "_param_coeff", [])):
    if _num(coeff) is None:
      continue
    d = legendre.setdefault(lab, {})
    d.setdefault(int(_num(order)), np.zeros(3))[{"x": 0, "y": 1, "z": 2}[ax.lower()]] += _num(coeff)
  xharm, u_xharm = {}, {}
  p = "_jana_atom_site_displace_xharm"
  for lab, ax, order, coeff in zip(_col(block, p + "_site_label", []), _col(block, p + "_axis", []),
                                   _col(block, p + "_param_order", []), _col(block, p + "_param_coeff", [])):
    if _num(coeff) is not None:
      xharm.setdefault(lab, {}).setdefault(int(_num(order)), np.zeros(3))[{"x": 0, "y": 1, "z": 2}[ax.lower()]] += _num(coeff)
  p = "_jana_atom_site_u_xharm"
  idx6 = {"u11": 0, "u22": 1, "u33": 2, "u12": 3, "u13": 4, "u23": 5}
  for lab, el, order, coeff in zip(_col(block, p + "_atom_site_label", []), _col(block, p + "_tens_elem", []),
                                   _col(block, p + "_param_order", []), _col(block, p + "_param_coeff", [])):
    if _num(coeff) is not None:
      u_xharm.setdefault(lab, {}).setdefault(int(_num(order)), np.zeros(6))[idx6[el.lower()]] += _num(coeff)
  for lab in list(legendre) + list(xharm) + list(u_xharm):
    if lab not in crenel:
      raise NotImplementedError("%s: Legendre/x-harmonic functions without a crenel on %s" % (cif_path, lab))
  for lab, (a_, c_, w_) in sawtooth.items():
    if lab not in crenel:
      crenel[lab] = (c_, w_)   # a sawtooth is only defined over its own interval

  def as_star(u_cif):
    return np.array(adptbx.u_cif_as_u_star(uc, tuple(u_cif)))

  atoms = []
  for k, lab in enumerate(labels):
    site = np.array([xyz[0][k], xyz[1][k], xyz[2][k]])
    weight = factor[k]
    p0 = occ[k]
    if lab in crenel and abs(p0 - crenel[lab][1]) > 2e-3:
      # crenel: the listed occupancy is the average (2108432: occupancy = width) or, as here, the value
      # inside the crenel (2108042: Al2 + Si2 share one crenel and sum to 1 there); modulated_atom wants the average
      p0 *= crenel[lab][1]
    w = waves[lab]
    nd = max(w["d"]) if w["d"] else 1
    a, b = np.zeros((nd, 3)), np.zeros((nd, 3))
    for n, (sv, cv) in w["d"].items():
      a[n - 1], b[n - 1] = sv, cv
    no = max(w["o"]) if w["o"] else 0
    os_, oc = np.zeros(no), np.zeros(no)
    for n, (sv, cv) in w["o"].items():
      os_[n - 1], oc[n - 1] = sv, cv
    nu = max(w["u"]) if w["u"] else 0
    us, uc_ = np.zeros((nu, 6)), np.zeros((nu, 6))
    for n, (sv, cv) in w["u"].items():
      us[n - 1], uc_[n - 1] = as_star(sv), as_star(cv)
    u_star = aniso.get(lab)
    if u_star is None and nu:
      u_star = adptbx.u_iso_as_u_star(uc, uiso[k])
    def by_order(d, width, conv=lambda v: v):
      if not d:
        return None
      out = np.zeros((max(d), width))
      for n, v in d.items():
        out[n - 1] = conv(v)
      return out
    if lab in u_xharm and u_star is None:
      u_star = adptbx.u_iso_as_u_star(uc, uiso[k])
    atoms.append(mod.modulated_atom(lab, types[k], site, uiso[k], p0, nd, a, b, u_star, os_, oc, us, uc_,
                                    weight, crenel.get(lab), by_order(legendre.get(lab), 3), sawtooth.get(lab),
                                    by_order(xharm.get(lab), 3), by_order(u_xharm.get(lab), 6, as_star)))
  dispersion = {}
  if "_atom_type_symbol" in block and "_atom_type_scat_dispersion_real" in block:
    for t, fp, fdp in zip(_col(block, "_atom_type_symbol"), _col(block, "_atom_type_scat_dispersion_real"),
                          _col(block, "_atom_type_scat_dispersion_imag")):
      dispersion[t] = (_num(fp) or 0.0, _num(fdp) or 0.0)
  xs = mod.modulated_structure(uc, ssg, atoms, 1.0, dispersion)

  rb = _block_with(model, "_refln_index_h")
  if rb is None and fcf_path:
    rb = _block_with(iotbx.cif.reader(file_path=fcf_path).model(), "_refln_index_h")
  data = None
  if rb is not None:
    mkey = "_refln_index_m_1" if "_refln_index_m_1" in rb else "_refln_index_m"
    if "_refln_index_m_2" in rb:
      raise NotImplementedError("(3+2)D reflections")
    col = lambda k: np.array([_num(v) if _num(v) is not None else np.nan for v in _col(rb, k)]) if k in rb else None
    data = {"h": np.column_stack([col("_refln_index_" + a) for a in "hkl"]).astype(int),
            "m": col(mkey).astype(int), "f_sq_meas": col("_refln_F_squared_meas"),
            "f_sq_sigma": col("_refln_F_squared_sigma"), "f_sq_calc": col("_refln_F_squared_calc")}
  ext = _col(block, "_refine_ls_extinction_coef")
  if data is not None:
    data["extinction"] = (_col(block, "_refine_ls_extinction_method", ["?"])[0], ext[0] if ext else None)
  return xs, data
