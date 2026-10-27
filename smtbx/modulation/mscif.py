"""Read a (3+1)D model and its hklm data from an msCIF (JANA2006/Jana2020 output) into smtbx.modulation objects.

Supported: average positions, Uiso/Uaniso, occupancy, displacive, occupational and ADP Fourier waves (sin/cos or
modulus/phase form), f'/f'', the superspace operators as listed, refln loops with one m index (in the CIF or a
separate fcf), crenels, sawtooths, Legendre and x-harmonic functions. Rigid groups, composites and (3+d)D with
d > 1 raise NotImplementedError. write() writes a model back in the same form.
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


def _cif_model(path):
  """The CIF model of a file: iotbx's default engine, or ucif where the xcif extension is not built."""
  import iotbx.cif
  try:
    return iotbx.cif.reader(file_path=path).model()
  except ImportError:
    return iotbx.cif.reader(file_path=path, engine="ucif").model()


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


def read(cif_path, fcf_path=None, xharm_basis="plain"):
  """(modulated_structure, data) with data = dict(h, m, f_sq_meas, f_sq_sigma, f_sq_calc, extinction,
  commensurate): commensurate (t0, N) for a JANA commensurate refinement (q then exactly r / N), else None.
  xharm_basis: how the x-harmonic coefficients (_jana_atom_site_displace_xharm, _jana_atom_site_u_xharm) are
  meant: "plain", sin/cos(2 pi k x) as in xharm_basis, or "orthonormal", Gram-Schmidt orthonormalised over the
  crenel interval (smtbx.modulation.xharm_orthonormal); those are changed to the plain basis on reading (the
  constant part, zero for these functions, into the average site / ADP). Which one a JANA file needs is not
  marked by any item this reader knows; the comparison against JANA's F^2calc decides."""
  if xharm_basis not in ("plain", "orthonormal"):
    raise ValueError("xharm_basis: 'plain' or 'orthonormal', not %r" % (xharm_basis,))
  model = _cif_model(cif_path)
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
  # commensurate (JANA _jana_cell_commen_*): the t0 section and N from the supercell, q made exactly r / N
  commensurate = None
  t_sec = _col(block, "_jana_cell_commen_t_section_1")
  if t_sec and _num(t_sec[0]) is not None:
    sm = np.array([[_num(_col(block, "_jana_cell_commen_supercell_matrix_%d_%d" % (i, j), ["0"])[0]) or 0.0
                    for j in (1, 2, 3)] for i in (1, 2, 3)])
    n_sec = int(round(abs(np.linalg.det(sm)))) if sm.any() else 0
    if n_sec > 0:
      q = np.round(q * n_sec) / n_sec
      commensurate = (_num(t_sec[0]), n_sec)
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
    xh, uxh = by_order(xharm.get(lab), 3), by_order(u_xharm.get(lab), 6, as_star)
    if xharm_basis == "orthonormal":
      if xh is not None:
        xh, const = mod.xharm_to_plain(xh)
        site = site + const
      if uxh is not None:
        uxh, const = mod.xharm_to_plain(uxh)
        u_star = np.asarray(u_star, dtype=float) + const
    atoms.append(mod.modulated_atom(lab, types[k], site, uiso[k], p0, nd, a, b, u_star, os_, oc, us, uc_,
                                    weight, crenel.get(lab), by_order(legendre.get(lab), 3), sawtooth.get(lab),
                                    xh, uxh))
  dispersion = {}
  if "_atom_type_symbol" in block and "_atom_type_scat_dispersion_real" in block:
    for t, fp, fdp in zip(_col(block, "_atom_type_symbol"), _col(block, "_atom_type_scat_dispersion_real"),
                          _col(block, "_atom_type_scat_dispersion_imag")):
      dispersion[t] = (_num(fp) or 0.0, _num(fdp) or 0.0)
  xs = mod.modulated_structure(uc, ssg, atoms, 1.0, dispersion)

  rb = _block_with(model, "_refln_index_h")
  if rb is None and fcf_path:
    rb = _block_with(_cif_model(fcf_path), "_refln_index_h")
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
    # electrons by name, or by a wavelength no X-ray or neutron source has (0.0197-0.0370 A at 100-300 kV)
    probe = " ".join(_col(block, k, [""])[0] for k in ("_diffrn_radiation_probe", "_diffrn_radiation_type")).lower()
    wl = _num(_col(block, "_diffrn_radiation_wavelength", ["?"])[0])
    data["radiation"] = "electron" if "electron" in probe or (wl is not None and wl < 0.1) else "xray"
    data["wavelength"] = wl
  if data is not None:
    data["commensurate"] = commensurate     # (t0, N): mod.commensurate_section(xs, t0, N) gives JANA's model
  return xs, data


def _g(v):
  return "%.15g" % v


def _loop(lines, items, rows):
  if not rows:
    return
  lines.append("loop_")
  lines.extend(" " + it for it in items)
  for r in rows:
    lines.append(" " + " ".join(str(v) for v in r))
  lines.append("")


def write(xs, path, h=None, m=None, f_sq_calc=None, f_sq_meas=None, f_sq_sigma=None, name="smtbx_modulation",
          table=None, radiation="xray"):
  """A refined model as a JANA-style msCIF that read() takes back: cell, q, the superspace operators, atom sites
  (multiplicity from site_weight), Uiso and Uaniso, the displacive, occupancy and ADP Fourier loops, crenels,
  sawtooths, Legendre and x-harmonic (plain basis) loops, f'/f'', and with h, m the refln loop with F^2calc
  (computed here when not given, with xs.scale; table/radiation as for the calculator) and F^2meas/sigma if given.
  Numbers are written with 15 significant digits. Crenel occupancies are written as the value inside the crenel,
  as JANA does."""
  uc = xs.unit_cell
  order = xs.ssg.basic.order_z()
  L = ["data_%s" % name, ""]
  for k, v in zip(("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"), uc.parameters()):
    L.append("_cell_%s %s" % (k, _g(v)))
  L.append("")
  _loop(L, ["_cell_wave_vector_seq_id", "_cell_wave_vector_x", "_cell_wave_vector_y", "_cell_wave_vector_z"],
        [[1] + [_g(v) for v in xs.q]])
  _loop(L, ["_space_group_symop_ssg_id", "_space_group_symop_ssg_operation_algebraic"],
        [[i + 1, op.as_xyz4()] for i, op in enumerate(xs.ssg.ops)])
  rows = []
  for at in xs.atoms:
    occ = at.occupancy
    if at.crenel:
      w = at.crenel[1]
      occ = occ / w if abs(occ / w - w) > 2e-3 else occ   # read() takes a value equal to w as the average
    rows.append([at.label, at.scattering_type] + [_g(v) for v in at.site] +
                [_g(at.u_iso), _g(occ), "%d" % int(round(at.site_weight * order))])
  _loop(L, ["_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
            "_atom_site_fract_z", "_atom_site_U_iso_or_equiv", "_atom_site_occupancy",
            "_atom_site_site_symmetry_multiplicity"], rows)
  as_cif = lambda u: adptbx.u_star_as_u_cif(uc, tuple(float(x) for x in u))
  _loop(L, ["_atom_site_aniso_label"] + ["_atom_site_aniso_U_" + e for e in ("11", "22", "33", "12", "13", "23")],
        [[at.label] + [_g(v) for v in as_cif(at.u_star)] for at in xs.atoms if at.u_star is not None])
  n_max = max([at.n_harmonics for at in xs.atoms] + [len(at.occ_sin) for at in xs.atoms] +
              [len(at.u_sin) for at in xs.atoms] + [0])
  _loop(L, ["_atom_site_Fourier_wave_vector_seq_id", "_jana_atom_site_fourier_wave_vector_q1_coeff"],
        [[n, n] for n in range(1, n_max + 1)])
  disp, occf, uf = [], [], []
  elems = ("U11", "U22", "U33", "U12", "U13", "U23")
  for at in xs.atoms:
    for n in range(at.n_harmonics):
      for k, ax in enumerate("xyz"):
        disp.append([at.label, ax, n + 1, _g(at.a[n][k]), _g(at.b[n][k])])
    for n in range(len(at.occ_sin)):
      occf.append([at.label, n + 1, _g(at.occ_sin[n]), _g(at.occ_cos[n])])
    for n in range(len(at.u_sin)):
      s, c = as_cif(at.u_sin[n]), as_cif(at.u_cos[n])
      for k, e in enumerate(elems):
        uf.append([at.label, e, n + 1, _g(s[k]), _g(c[k])])
  p = "_atom_site_displace_Fourier"
  _loop(L, [p + "_atom_site_label", p + "_axis", p + "_wave_vector_seq_id", p + "_param_sin", p + "_param_cos"], disp)
  p = "_atom_site_occ_Fourier"
  _loop(L, [p + "_atom_site_label", p + "_wave_vector_seq_id", p + "_param_sin", p + "_param_cos"], occf)
  p = "_atom_site_U_Fourier"
  _loop(L, [p + "_atom_site_label", p + "_tens_elem", p + "_wave_vector_seq_id", p + "_param_sin", p + "_param_cos"],
        uf)
  p = "_atom_site_occ_special_func"
  _loop(L, [p + "_atom_site_label", p + "_crenel_c", p + "_crenel_w"],
        [[at.label, _g(at.crenel[0]), _g(at.crenel[1])] for at in xs.atoms if at.crenel])
  p = "_atom_site_displace_special_func"
  _loop(L, [p + k for k in ("_atom_site_label", "_sawtooth_ax", "_sawtooth_ay", "_sawtooth_az", "_sawtooth_c",
                            "_sawtooth_w")],
        [[at.label] + [_g(v) for v in at.sawtooth[0]] + [_g(at.sawtooth[1]), _g(at.sawtooth[2])]
         for at in xs.atoms if at.sawtooth is not None])
  leg, xh, uxh = [], [], []
  for at in xs.atoms:
    for n in range(len(at.legendre)):
      for k, ax in enumerate("xyz"):
        leg.append([at.label, ax, n + 1, _g(at.legendre[n][k])])
    for n in range(len(at.xharm)):
      for k, ax in enumerate("xyz"):
        xh.append([at.label, ax, n + 1, _g(at.xharm[n][k])])
    for n in range(len(at.u_xharm)):
      u = as_cif(at.u_xharm[n])
      for k, e in enumerate(elems):
        uxh.append([at.label, e, n + 1, _g(u[k])])
  p = "_jana_atom_site_displace_legendre"
  _loop(L, [p + "_atom_site_label", p + "_axis", p + "_param_order", p + "_param_coeff"], leg)
  p = "_jana_atom_site_displace_xharm"
  _loop(L, [p + "_site_label", p + "_axis", p + "_param_order", p + "_param_coeff"], xh)
  p = "_jana_atom_site_u_xharm"
  _loop(L, [p + "_atom_site_label", p + "_tens_elem", p + "_param_order", p + "_param_coeff"], uxh)
  if xs.dispersion:
    _loop(L, ["_atom_type_symbol", "_atom_type_scat_dispersion_real", "_atom_type_scat_dispersion_imag"],
          [[t, _g(fp), _g(fdp)] for t, (fp, fdp) in sorted(xs.dispersion.items())])
  if h is not None:
    h, m = np.asarray(h).reshape(-1, 3), np.asarray(m).ravel()
    if f_sq_calc is None:
      f_sq_calc = mod.calculator(h, m, table=table, radiation=radiation).compute(xs)[1]
    items = ["_refln_index_h", "_refln_index_k", "_refln_index_l", "_refln_index_m_1", "_refln_F_squared_calc"]
    cols = [f_sq_calc]
    if f_sq_meas is not None:
      items += ["_refln_F_squared_meas", "_refln_F_squared_sigma"]
      cols += [f_sq_meas, f_sq_sigma]
    _loop(L, items, [["%d" % v for v in hh] + ["%d" % mm] + [_g(c[i]) for c in cols]
                     for i, (hh, mm) in enumerate(zip(h.tolist(), m.tolist()))])
  with open(path, "w") as f:
    f.write("\n".join(L) + "\n")
