""" Finish a partial solution the way a crystallographer does.

  sites, calls, added = model_completion.complete(f_obs, placed, sites, calls)

The missing atoms of a partial solution are usually not below the peak cut
but absent from the map, so the model is finished rather than searched
harder: refine, compute Fo-Fc, and add an atom where the difference map
shows one.

  * A new atom must be chemically possible: a peak closer than `MIN_BOND` to
    an existing atom is a ripple, one further than `MAX_BOND` from everything
    is more likely noise.
  * The loop stops when the strongest residual peak falls below `SIGMA_CUT`
    sigma.
  * Only additions are pruned; the original sites are kept unless `SWAP`.
"""
from __future__ import absolute_import, division, print_function

import os

# residual-peak cut in robust sigma; lower adds more atoms and more spurious ones
SIGMA_CUT = float(os.environ.get("COMPLETION_SIGMA", "7.0"))
MIN_BOND = float(os.environ.get("COMPLETION_MIN_BOND", "1.0"))

# Element-aware floor on how close an addition may sit:
# COV_FACTOR*(r_cov(neighbour) + r_cov(C)), floored at MIN_BOND. The 1.2-1.6 A
# shell is mostly ripple beside a heavy atom and mostly real bonds in an
# organic; with carbon's sum of 1.52 A the floor leaves organics untouched and
# only rises next to large atoms. 0 = off.
COV_FACTOR = float(os.environ.get("COMPLETION_COV_FACTOR", "0"))

# Batched numpy nearest-image distances in `candidates` instead of one
# `min_distance` call per pair. Never a 2-D `.dot` there: numpy routes it
# through OpenBLAS, and loading that DLL after this cctbx build's aborts the
# process (0xC06D007F); `einsum` avoids BLAS.
FAST_DIST = os.environ.get("COMPLETION_FAST_DIST", "0") != "0"


def _fast_ctx(uc, sg, sites):
  """ Precompute the metric tensor, symmetry ops and site array once. """
  import numpy as np
  from smtbx.ab_initio.assemble import _SHIFTS

  g = uc.metrical_matrix()
  G = np.array([[g[0], g[3], g[4]], [g[3], g[1], g[5]], [g[4], g[5], g[2]]])
  SH = np.array(_SHIFTS, dtype=float)
  OPS = [(np.array(o.as_double_array()[:9]).reshape(3, 3),
          np.array(o.as_double_array()[9:])) for o in sg.all_ops()]
  return G, SH, OPS, np.asarray(list(sites))


def _fast_dists(target, ctx):
  """ Nearest-image distance from `target` to every site, as an array. """
  import numpy as np

  G, SH, OPS, S = ctx
  t = np.asarray(target)
  best = None
  for R, T in OPS:
    s = np.einsum("ni,ji->nj", S, R) + T          # never .dot -- see above
    base = s - np.round(s - t)
    dv = base[:, None, :] + SH[None, :, :] - t
    d2 = np.einsum("nsi,ij,nsj->ns", dv, G, dv).min(axis=1)
    best = d2 if best is None else np.minimum(best, d2)
  return np.sqrt(best)
_COV_CACHE = {}


def min_bond_to(symbol):
  """ How close an addition may come to an atom of this element.

  MIN_BOND for an element unknown to the table, never a radius of zero.
  """
  if COV_FACTOR <= 0:
    return MIN_BOND
  key = (symbol or "C").strip().capitalize()
  if key in _COV_CACHE:
    return _COV_CACHE[key]
  try:
    from cctbx.eltbx import covalent_radii
    r = covalent_radii.table(key).radius()
    c = covalent_radii.table("C").radius()
    out = max(MIN_BOND, COV_FACTOR*(r + c))
  except Exception:
    out = MIN_BOND
  _COV_CACHE[key] = out
  return out
MAX_BOND = float(os.environ.get("COMPLETION_MAX_BOND", "2.6"))
MAX_ROUNDS = int(os.environ.get("COMPLETION_ROUNDS", "5"))
# Peak cut scaled by how full the model is: `cut*max(ADAPT_FLOOR,
# occupancy**ADAPT)`, occupancy = model size over one atom per ADAPT_VOLUME
# A^3 (the median of real compositions). A short model admits weaker residual
# peaks; recomputed every round, so the leniency closes as the model fills.
# A weight, not a switch. 0 = off.
ADAPT = float(os.environ.get("COMPLETION_ADAPT", "0"))
ADAPT_VOLUME = float(os.environ.get("COMPLETION_ADAPT_VOLUME", "17.2"))
ADAPT_FLOOR = float(os.environ.get("COMPLETION_ADAPT_FLOOR", "0.5"))
# what the last structure ran with, so "off" and "ran, changed nothing" differ
LAST_ADAPT_WANT = 0
LAST_ADAPT_CUT = 0.0
_ADAPT_ANNOUNCED = False
MAX_ADD_PER_ROUND = 8
# starting u_iso before the structure's own median is known
DEFAULT_U = float(os.environ.get("COMPLETION_DEFAULT_U", "0.05"))
# per-atom prune thresholds relative to the structure's own refined atoms;
# only used with BUDGET_MODE off
PRUNE_CYCLES = int(os.environ.get("COMPLETION_PRUNE_CYCLES", "3"))
U_RATIO = float(os.environ.get("COMPLETION_U_RATIO", "3.0"))
MAX_SHIFT = float(os.environ.get("COMPLETION_MAX_SHIFT", "0.7"))
MIN_OCC = float(os.environ.get("COMPLETION_MIN_OCC", "0.3"))
# weighted contest (`_by_budget`) instead of the per-atom thresholds above;
# shift and occupancy do not separate real from spurious additions
BUDGET_MODE = os.environ.get("COMPLETION_BUDGET", "1") != "0"
BUDGET_BETA = float(os.environ.get("COMPLETION_BETA", "1.0"))
# A^3 per non-H atom for the completion ceiling; tighter than the typical 17
# because a surviving addition goes straight into the model
BUDGET_VOLUME = float(os.environ.get("COMPLETION_VOLUME", "13.0"))
# Every addition is kept while
#     log(U_MAX) - log(uratio) + SIGMA_W*log(sigma/SIGMA_REF) > 0
# with uratio = u_iso over the originals' median: a strong peak buys tolerance
# for a worse ADP. SIGMA_W = 0 because peak strength has not been shown to
# help here.
U_MAX = float(os.environ.get("COMPLETION_UMAX", "2.0"))
SIGMA_W = float(os.environ.get("COMPLETION_SIGMA_W", "0.0"))
SIGMA_REF = float(os.environ.get("COMPLETION_SIGMA_REF", "3.0"))

# The bar moves with how full the model is: the score above gains
#     PRUNE_ADAPT*min(log(PRUNE_ADAPT_CEIL), -log(occupancy))
# i.e. U_MAX becomes U_MAX*occupancy**-PRUNE_ADAPT, so a sparse model is not
# stripped of correct atoms. Recomputed as the model fills; the ceiling stops
# a two-atom model from keeping everything the map offered.
PRUNE_ADAPT = float(os.environ.get("COMPLETION_PRUNE_ADAPT", "2.0"))
PRUNE_ADAPT_CEIL = float(os.environ.get("COMPLETION_PRUNE_ADAPT_CEIL", "4.0"))

# Originals compete with the additions for the same cell budget, so an
# original that fits nothing can lose its slot to a better-supported atom.
SWAP = os.environ.get("COMPLETION_SWAP", "0") != "0"
# never fewer atoms than this after a swap contest: thin rather than empty
SWAP_FLOOR = int(os.environ.get("COMPLETION_SWAP_FLOOR", "4"))


def build_model(placed, sites, calls, u_values=None):
  """ An xray.structure at `sites`, typed by `calls`, started at `u_values`.

  Additions are seeded at the structure's median u_iso, so refinement spends
  its cycles on the position, and a large final u_iso is refinement pushing
  it there, which is what the prune reads.
  """
  from cctbx import xray

  model = xray.structure(placed.crystal_symmetry().special_position_settings())
  for i, s in enumerate(sites):
    el = calls[i] if calls and i < len(calls) else "C"
    u = DEFAULT_U
    if u_values is not None and i < len(u_values) and u_values[i] > 0:
      u = u_values[i]
    model.add_scatterer(xray.scatterer(label="%s%d" % (el, i), site=s,
                                       u=u, scattering_type=el))
  return model


def refined_u_values(refined, n):
  """ Refined u_iso per site, and the median over the sound ones.

  Non-positive and absent values are replaced by the median rather than kept,
  so a single failed atom cannot seed the next round with nonsense.
  """
  import numpy as np

  scat = refined.scatterers()
  raw = []
  for i in range(min(n, scat.size())):
    u = scat[i].u_iso
    raw.append(float(u) if u is not None and u > 0 else 0.0)
  sound = [u for u in raw if u > 0]
  if len(sound) < 4:
    return None, None
  med = float(np.median(sound))
  return [u if u > 0 else med for u in raw], med


def difference_map(f_obs, placed, sites, calls, u_values=None):
  """ Fo - Fc with model phases, for the model at `sites` typed by `calls`. """
  from cctbx import maptbx
  from cctbx.array_family import flex
  from smtbx.ab_initio import dual_space

  model = build_model(placed, sites, calls, u_values)
  f_here = f_obs.customized_copy(
    space_group_info=placed.space_group_info()).merge_equivalents().array()
  refined = dual_space.refine_model(f_here, model, cycles=3)
  if refined is None:
    return None, None
  fc = refined.structure_factors(d_min=f_here.d_min(),
                                 algorithm="direct").f_calc()
  fo, fc = f_here.common_sets(fc)
  diff = fo.customized_copy(
    data=fo.data() - flex.abs(fc.data())).phase_transfer(fc)
  fft = diff.fft_map(symmetry_flags=maptbx.use_space_group_symmetry,
                     resolution_factor=0.5)
  fft.apply_volume_scaling()
  return fft, refined


def _p1():
  from cctbx import sgtbx
  if SG_P1_CACHE[0] is None:
    SG_P1_CACHE[0] = sgtbx.space_group_info("P1").group()
  return SG_P1_CACHE[0]


def min_distance(uc, sg, a, b):
  """ Shortest distance from `a` to any symmetry image of `b`, in Angstrom.

  `unit_cell.distance` does not wrap across cell boundaries, and rounding the
  fractional difference is only right for an orthogonal cell: in an oblique
  cell the nearest image can be a neighbouring lattice translation, which
  `assemble._nearest_image` checks.
  """
  from smtbx.ab_initio import assemble

  best = 1e9
  for op in sg:
    d, _site = assemble._nearest_image(uc, a, op*b, _p1())
    if d < best:
      best = d
  return best


def budget_atoms(placed):
  """ How many non-H atoms completion may leave in the model. """
  return max(8, int(placed.unit_cell().volume()
                    / max(placed.space_group().order_z(), 1)/BUDGET_VOLUME))


def expected_atoms(placed):
  """ How many non-H atoms a cell this size usually holds.

  The typical value (ADAPT_VOLUME), used as evidence that a model is short;
  `budget_atoms` is the ceiling completion may not exceed.
  """
  return max(1, int(placed.unit_cell().volume()
                    / max(placed.space_group().order_z(), 1)/ADAPT_VOLUME))


def map_sigma(real):
  """ A noise level the missing atoms cannot inflate.

  The standard deviation is dominated by the missing atoms themselves, so a
  missing heavy atom can read below the cut. The median absolute deviation is
  not, and 1.4826*MAD equals the sd for Gaussian noise. Large maps are
  sampled on a stride.
  """
  import numpy as np

  a = real.as_1d().as_numpy_array()
  if a.size > 400000:
    a = a[::max(1, a.size//400000)]
  med = float(np.median(a))
  mad = float(np.median(np.abs(a - med)))
  robust = 1.4826*mad
  if robust <= 0:
    # more than half the points identical
    return float(real.sample_standard_deviation())
  return robust


def candidates(fft, placed, sites, sigma_cut=SIGMA_CUT, calls=None):
  """ Residual maxima that could be atoms: strong, and at a bonding distance.

  Returns [(site, height_in_sigma)], strongest first.
  """
  from cctbx import maptbx
  from cctbx.array_family import flex

  real = fft.real_map_unpadded()
  sigma = map_sigma(real)
  if sigma <= 0:
    return []
  # `.all()` asserts without a cluster cap; sized from the cell
  n_expect = expected_atoms(placed)
  peak_list = fft.peak_search(
    parameters=maptbx.peak_search_parameters(
      peak_search_level=1, interpolate=True,
      min_distance_sym_equiv=MIN_BOND),
    verify_symmetry=False).all(max_clusters=2*n_expect)
  uc = placed.unit_cell()
  sg = placed.space_group()

  ctx = _fast_ctx(uc, sg, sites) if (FAST_DIST and sites.size()) else None
  out = []
  for site, height in zip(peak_list.sites(), peak_list.heights()):
    if height < sigma_cut*sigma:
      continue
    # nearest existing atom over symmetry images; the floor is that atom's own
    if ctx is not None:
      ds = _fast_dists(site, ctx)
      k = int(ds.argmin())
      best = float(ds[k])
      best_floor = min_bond_to(calls[k] if calls and k < len(calls) else "C")
    else:
      best, best_floor = 1e9, MIN_BOND
      for k, s in enumerate(sites):
        dd = min_distance(uc, sg, site, s)
        if dd < best:
          best = dd
          best_floor = min_bond_to(calls[k] if calls and k < len(calls) else "C")
    if best < best_floor or best > MAX_BOND:
      continue
    # carried on: an addition close to a heavy atom integrates part of it and
    # reads heavier than it is
    out.append((site, height/sigma, best))
  out.sort(key=lambda t: -t[1])
  return out


# Per-atom evidence from the last prune, for calibration, written even when
# nothing is dropped: (index, u_ratio, shift, occupancy, peak_sigma,
# nearest_atom_distance).
LAST_EVIDENCE = []
# P1 for `_nearest_image`, which then does only the lattice search (the ops
# are applied by the caller); built once
SG_P1_CACHE = [None]
# additions the cell cap removed although their own evidence supported them
LAST_CAPPED = 0
# Surviving originals. With swap an original can be dropped, and a caller
# that assumes its first n atoms are the trusted ones would fit the carbon
# scale on additions.
LAST_N_ORIGINAL = 0


def prune_added(f_obs, placed, sites, calls, n_original,
                u_values=None, mode="on", strengths=None, budget=None,
                dists=None, orig_strengths=None, swap=False):
  """ Refine once, then drop added atoms the refinement will not support.

  An atom placed on noise inflates its u_iso to absorb the absent density
  and drifts; a real one keeps a u_iso like its neighbours'. Judged relative
  to the median of the original atoms, since absolute u_iso varies by an
  order of magnitude between structures. Only additions can be removed,
  unless `swap`.
  """
  import numpy as np
  from cctbx.array_family import flex
  from smtbx.ab_initio import dual_space

  # with swap the originals contest each other even when nothing was added
  if sites.size() <= n_original and not swap:
    return sites, calls, 0

  model = build_model(placed, sites, calls, u_values)
  f_here = f_obs.customized_copy(
    space_group_info=placed.space_group_info()).merge_equivalents().array()
  refined = dual_space.refine_model(f_here, model, cycles=PRUNE_CYCLES)
  if refined is None or refined.scatterers().size() != sites.size():
    return sites, calls, 0

  scat = refined.scatterers()
  uc = placed.unit_cell()
  orig_u = [scat[i].u_iso for i in range(n_original)
            if scat[i].u_iso is not None and scat[i].u_iso > 0]
  if len(orig_u) < 4:
    return sites, calls, 0
  ref_u = float(np.median(orig_u))

  del LAST_EVIDENCE[:]
  # without peak strengths for the originals the contest would be unfair, so
  # swap turns itself off
  swap = bool(swap and orig_strengths is not None
              and len(orig_strengths) >= n_original)
  keep_sites, keep_calls, dropped = flex.vec3_double(), [], 0
  for i in range(sites.size()):
    if i < n_original and not swap:
      keep_sites.append(sites[i])
      keep_calls.append(calls[i] if calls and i < len(calls) else "C")
      continue
    if i < n_original:
      u = scat[i].u_iso if scat[i].u_iso is not None else 0.0
      d = [x - y for x, y in zip(scat[i].site, sites[i])]
      LAST_EVIDENCE.append(
        (i, u/ref_u if ref_u > 0 else 0.0,
         uc.length([x - round(x) for x in d]), scat[i].occupancy,
         orig_strengths[i], -1.0))
      keep_sites.append(sites[i])
      keep_calls.append(calls[i] if calls and i < len(calls) else "C")
      continue
    u = scat[i].u_iso if scat[i].u_iso is not None else 0.0
    # wrapped: a refining atom can cross a cell boundary
    d = [x - y for x, y in zip(scat[i].site, sites[i])]
    shift = uc.length([x - round(x) for x in d])
    occ = scat[i].occupancy
    k = i - n_original
    sig = strengths[k] if strengths and k < len(strengths) else -1.0
    dst = dists[k] if dists and k < len(dists) else -1.0
    LAST_EVIDENCE.append((i, u/ref_u if ref_u > 0 else 0.0, shift, occ, sig,
                          dst))
    drop = (u > U_RATIO*ref_u or u <= 0.0 or shift > MAX_SHIFT
            or occ < MIN_OCC)
    if BUDGET_MODE:
      drop = False        # decided below, by the contest rather than per atom
    # `mark` keeps everything and only records the evidence
    if drop and mode != "mark":
      dropped += 1
      continue
    keep_sites.append(sites[i])
    keep_calls.append(calls[i] if calls and i < len(calls) else "C")

  if BUDGET_MODE and mode != "mark":
    # `_by_budget` reads strengths[k] for the k-th contestant, so with swap the
    # originals' strengths come first, in site order
    contest = (list(orig_strengths[:n_original]) + list(strengths or [])
               if swap else strengths)
    keep_sites, keep_calls, dropped = _by_budget(
      sites, calls, 0 if swap else n_original, contest, budget,
      floor=SWAP_FLOOR if swap else 0, n_expect=expected_atoms(placed))
    global LAST_N_ORIGINAL
    LAST_N_ORIGINAL = (_count_originals(sites, keep_sites, n_original)
                       if swap else n_original)
    if os.environ.get("COMPLETION_DEBUG"):
      # requested != active when the originals' peak heights did not arrive
      print("SWAPDIAG requested=%d active=%d orig=%d added=%d kept=%d "
            "budget=%s room_binds=%d"
            % (int(SWAP), int(swap), n_original, sites.size() - n_original,
               keep_sites.size(), budget,
               int((budget or 0) < sites.size())))
  return keep_sites, keep_calls, dropped


def _count_originals(sites, keep_sites, n_original):
  """ How many of the first `n_original` sites survived the contest.

  By coordinate, since `_by_budget` rebuilds the list and indices shift.
  """
  originals = set(tuple(sites[i]) for i in range(n_original))
  return sum(1 for s in keep_sites if tuple(s) in originals)


def _by_budget(sites, calls, n_original, strengths, budget, floor=0,
               n_expect=None):
  """ Keep the additions the cell has room for, best evidence first.

  No single quantity vetoes an atom: peak strength (sigma) and the u_iso
  ratio to the originals' median (a real atom refines near 1, a spurious one
  inflates) are combined in log space, so a strong peak with a mediocre ADP
  can outvote a weak peak with a good one. At most `budget - n_original`
  additions are kept, the room the cell leaves.
  """
  import numpy as np
  from cctbx.array_family import flex

  n_add = sites.size() - n_original
  if n_add <= 0:
    return sites, list(calls), 0
  room = max(0, (budget or 0) - n_original)

  score, by_i = [], dict((e[0], e) for e in LAST_EVIDENCE)
  for k in range(n_add):
    i = n_original + k
    sig = strengths[k] if strengths and k < len(strengths) else SIGMA_REF
    ev = by_i.get(i)
    uratio = ev[1] if ev else 1.0
    # positive = the evidence supports the atom
    score.append(np.log(U_MAX) - np.log(max(uratio, 1e-3))
                 + SIGMA_W*np.log(max(sig, 1e-3)/SIGMA_REF))

  # kept: positive evidence and inside the room the cell leaves
  ranked = sorted(range(n_add), key=lambda k: -score[k])
  if PRUNE_ADAPT > 0 and n_expect:
    # Best evidence first, so a sparse model's leniency goes to its strongest
    # candidates and closes as it fills. With swap n_original is 0 and the
    # originals are counted inside `keep`.
    keep = set()
    for k in ranked[:room]:
      occ = (n_original + len(keep))/float(n_expect)
      bonus = 0.0
      if occ < 1.0:
        bonus = PRUNE_ADAPT*min(np.log(PRUNE_ADAPT_CEIL),
                                -np.log(max(occ, 1e-3)))
      if score[k] + bonus > 0.0:
        keep.add(k)
  else:
    keep = set(k for k in ranked[:room] if score[k] > 0.0)
  # swap only (floor is 0 otherwise): the contest may not empty the model
  if floor and len(keep) < floor:
    keep = set(ranked[:min(floor, n_add)])
  # which limit did the work: cell cap against own evidence
  global LAST_CAPPED
  LAST_CAPPED = sum(1 for k in ranked[room:] if score[k] > 0.0)
  out_sites, out_calls = flex.vec3_double(), []
  # re-key the evidence rows to the surviving atoms' new indices
  by_index = dict((e[0], e) for e in LAST_EVIDENCE)
  rekeyed = []
  for i in range(sites.size()):
    if i >= n_original and (i - n_original) not in keep:
      continue
    if i >= n_original and i in by_index:
      rekeyed.append((out_sites.size(),) + tuple(by_index[i][1:]))
    out_sites.append(sites[i])
    out_calls.append(calls[i] if calls and i < len(calls) else "C")
  LAST_EVIDENCE[:] = rekeyed
  return out_sites, out_calls, n_add - len(keep)


def complete(f_obs, placed, sites, calls, rounds=MAX_ROUNDS,
             sigma_cut=SIGMA_CUT, orig_strengths=None):
  """ Add atoms where the difference map says one is missing.

  Returns (sites, calls, n_added). The sites are extended in place order, so a
  caller's pairing against them stays valid for the original entries.
  """
  from cctbx.array_family import flex

  sites = flex.vec3_double(sites)
  calls = list(calls) if calls else ["C"]*sites.size()
  n_original = sites.size()
  # reset before anything can fail, so no stale value from the last structure
  global LAST_N_ORIGINAL, LAST_ADAPT_WANT, LAST_ADAPT_CUT
  LAST_N_ORIGINAL = n_original
  LAST_ADAPT_WANT, LAST_ADAPT_CUT = 0, 0.0
  global _ADAPT_ANNOUNCED
  if not _ADAPT_ANNOUNCED:
    _ADAPT_ANNOUNCED = True
    print("ADAPTDIAG requested=%g active=%d volume=%.1f floor=%.2f"
          % (ADAPT, 1 if ADAPT > 0 else 0, ADAPT_VOLUME, ADAPT_FLOOR))
  added = 0
  u_values = None
  strengths, dists = [], []
  for _round in range(rounds):
    fft, _refined = difference_map(f_obs, placed, sites, calls, u_values)
    if fft is None:
      break
    # refined ADPs carry into the next round; additions start at their median
    med = None
    if _refined is not None and _refined.scatterers().size() == sites.size():
      u_values, med = refined_u_values(_refined, sites.size())
    cut = sigma_cut
    if ADAPT > 0:
      want = expected_atoms(placed)
      occ = sites.size()/float(want) if want else 1.0
      if occ < 1.0:
        cut = sigma_cut*max(ADAPT_FLOOR, occ**ADAPT)
      if _round == 0:
        LAST_ADAPT_WANT, LAST_ADAPT_CUT = want, cut
    found = candidates(fft, placed, sites, cut, calls)
    if not found:
      break
    for site, _sig, _dst in found[:MAX_ADD_PER_ROUND]:
      sites.append(site)
      strengths.append(_sig)
      dists.append(_dst)
      if u_values is not None:
        u_values.append(med)
      # carbon; the geometry aid retypes everything afterwards
      calls.append("C")
      added += 1
    if len(found) <= 0:
      break

  # refine once more and drop the additions the refinement will not support,
  # before anything is typed; with swap even when nothing was added
  mode = os.environ.get("COMPLETION_PRUNE", "1")
  if (added or (SWAP and orig_strengths is not None)) and mode != "0":
    n_before = sites.size()
    sites, calls, dropped = prune_added(f_obs, placed, sites, calls,
                                        n_original, u_values,
                                        mode=("mark" if mode == "mark"
                                              else "on"),
                                        strengths=strengths,
                                        budget=budget_atoms(placed),
                                        dists=dists,
                                        orig_strengths=orig_strengths,
                                        swap=SWAP)
    # describes the model handed back; with swap a drop can hit an original
    added = max(0, sites.size() - (n_before - added))
  return sites, calls, added


# The staged heavy-atom build.

STAGE_ROUNDS = int(os.environ.get("STAGE_ROUNDS", "12"))
# Peak cut for the staged path only: light atoms in a map from a heavy-atom
# model do not clear SIGMA_CUT. 0 = use SIGMA_CUT.
STAGE_SIGMA = float(os.environ.get("STAGE_SIGMA", "0")) or None


def align3(fo, cf, fc):
  """ Put three Miller arrays on one common index set.

  `common_sets` pairs two at a time and each pairing can shrink the set, and
  a misaligned subtraction is silent, so it is iterated, then asserted.
  """
  for _ in range(4):
    fo, cf = fo.common_sets(cf)
    fo, fc = fo.common_sets(fc)
    cf, fc = cf.common_sets(fc)
  assert fo.indices().size() == cf.indices().size() == fc.indices().size()
  assert fo.indices().all_eq(cf.indices())
  assert fo.indices().all_eq(fc.indices())
  return fo, cf, fc


def complete_cf_phase(f_obs, placed, seed_sites, calls, n_peaks, rounds=None):
  """ Staged build on the charge-flipping phases: |Fo| phi_CF - Fc(heavy).

  Series-termination ripple is the truncated transform of the heavy atom, so
  subtracting Fc(heavy) at the same cutoff with the phases left alone cancels
  it term by term; phases from a partial model would bias the map toward what
  was placed, worst in acentric groups. `placed` carries the charge-flipping
  phases.
  """
  from cctbx import maptbx
  from cctbx.array_family import flex

  if rounds is None:
    rounds = STAGE_ROUNDS

  sites = flex.vec3_double(seed_sites)
  calls = list(calls)
  f_here = f_obs.customized_copy(
    space_group_info=placed.space_group_info()).merge_equivalents().array()

  for _round in range(rounds):
    model = build_model(placed, sites, calls, None)
    fc = model.structure_factors(d_min=f_here.d_min(),
                                 algorithm="direct").f_calc()
    fo, cf, fc = align3(f_here, placed, fc)
    fo_cf = fo.phase_transfer(cf)
    diff = fo_cf.customized_copy(data=fo_cf.data() - fc.data())
    fft = diff.fft_map(symmetry_flags=maptbx.use_space_group_symmetry,
                       resolution_factor=0.5)
    fft.apply_volume_scaling()
    found = candidates(fft, placed, sites, STAGE_SIGMA or SIGMA_CUT, calls)
    if not found:
      break
    added = 0
    for site, _sig, _dst in found[:MAX_ADD_PER_ROUND]:
      if sites.size() >= n_peaks:
        break
      sites.append(site)
      calls.append("C")
      added += 1
    if added == 0:
      break
  return sites, calls


def heavy_seed(sites, heights, n_seed):
  """ The `n_seed` tallest peaks: the blind seed the staged build starts from.

  `n_seed` comes from the user's expected formula.
  """
  from cctbx.array_family import flex

  n_seed = max(1, min(int(n_seed), sites.size()))
  order = sorted(range(sites.size()), key=lambda i: -heights[i])
  return flex.vec3_double([sites[i] for i in sorted(order[:n_seed])])


if __name__ == "__main__":
  print(__doc__)
  print("sigma cut %.1f, bond window %.1f-%.1f A, max %d rounds"
        % (SIGMA_CUT, MIN_BOND, MAX_BOND, MAX_ROUNDS))
