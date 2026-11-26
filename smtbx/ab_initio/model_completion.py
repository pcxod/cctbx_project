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
  * Only additions are pruned; the original sites are kept.
"""
from __future__ import absolute_import, division, print_function

import os

# residual-peak cut in robust sigma; lower adds more atoms and more spurious ones
SIGMA_CUT = float(os.environ.get("COMPLETION_SIGMA", "7.0"))
# a peak nearer an atom than this is ripple: below the shortest non-H bond
MIN_BOND = float(os.environ.get("COMPLETION_MIN_BOND", "1.0"))
# a peak farther than this from every atom is more likely noise than a bonded
# atom; long enough for most metal-donor bonds
MAX_BOND = float(os.environ.get("COMPLETION_MAX_BOND", "2.6"))
# map-and-add rounds; the loop also stops when no peak clears the cut
MAX_ROUNDS = int(os.environ.get("COMPLETION_ROUNDS", "5"))
# A^3 per non-H atom of a typical model (the median of real compositions);
# `expected_atoms` reads it as the size a model should reach
ADAPT_VOLUME = float(os.environ.get("COMPLETION_ADAPT_VOLUME", "17.2"))
# strongest first, so the map is re-phased before a much-changed model adds
# more
MAX_ADD_PER_ROUND = 8
# starting u_iso before the structure's own median is known
DEFAULT_U = float(os.environ.get("COMPLETION_DEFAULT_U", "0.05"))
# refinement cycles before the prune reads the additions' ADPs
PRUNE_CYCLES = int(os.environ.get("COMPLETION_PRUNE_CYCLES", "3"))
# A^3 per non-H atom for the completion ceiling; tighter than the typical 17
# because a surviving addition goes straight into the model
BUDGET_VOLUME = float(os.environ.get("COMPLETION_VOLUME", "13.0"))
# Every addition is kept while
#     log(U_MAX) - log(uratio) > 0
# with uratio = u_iso over the originals' median: shift, occupancy and peak
# height do not separate real from spurious additions, the ADP does.
U_MAX = float(os.environ.get("COMPLETION_UMAX", "2.0"))

# The bar moves with how full the model is: the score above gains
#     PRUNE_ADAPT*min(log(PRUNE_ADAPT_CEIL), -log(occupancy))
# i.e. U_MAX becomes U_MAX*occupancy**-PRUNE_ADAPT, so a sparse model is not
# stripped of correct atoms. Recomputed as the model fills; the ceiling stops
# a two-atom model from keeping everything the map offered.
PRUNE_ADAPT = float(os.environ.get("COMPLETION_PRUNE_ADAPT", "2.0"))
PRUNE_ADAPT_CEIL = float(os.environ.get("COMPLETION_PRUNE_ADAPT_CEIL", "4.0"))


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

  Returns [(site, height_in_sigma, nearest_atom_distance)], strongest first.
  `calls` is accepted for the callers' signature and not read.
  """
  from cctbx import maptbx

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

  out = []
  for site, height in zip(peak_list.sites(), peak_list.heights()):
    if height < sigma_cut*sigma:
      continue
    # nearest existing atom over symmetry images
    best = 1e9
    for s in sites:
      best = min(best, min_distance(uc, sg, site, s))
    if best < MIN_BOND or best > MAX_BOND:
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


def prune_added(f_obs, placed, sites, calls, n_original,
                u_values=None, strengths=None, budget=None, dists=None):
  """ Refine once, then drop added atoms the refinement will not support.

  An atom placed on noise inflates its u_iso to absorb the absent density
  and drifts; a real one keeps a u_iso like its neighbours'. Judged relative
  to the median of the original atoms, since absolute u_iso varies by an
  order of magnitude between structures. Only additions can be removed.
  """
  import numpy as np
  from smtbx.ab_initio import dual_space

  if sites.size() <= n_original:
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
  for i in range(n_original, sites.size()):
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

  return _by_budget(sites, calls, n_original, budget,
                    n_expect=expected_atoms(placed))


def _by_budget(sites, calls, n_original, budget, n_expect=None):
  """ Keep the additions the cell has room for, best evidence first.

  The score is the u_iso ratio to the originals' median (a real atom refines
  near 1, a spurious one inflates), in log space. At most
  `budget - n_original` additions are kept, the room the cell leaves.
  """
  import numpy as np
  from cctbx.array_family import flex

  n_add = sites.size() - n_original
  if n_add <= 0:
    return sites, list(calls), 0
  room = max(0, (budget or 0) - n_original)

  score, by_i = [], dict((e[0], e) for e in LAST_EVIDENCE)
  for k in range(n_add):
    ev = by_i.get(n_original + k)
    uratio = ev[1] if ev else 1.0
    # positive = the evidence supports the atom
    score.append(np.log(U_MAX) - np.log(max(uratio, 1e-3)))

  # kept: positive evidence and inside the room the cell leaves
  ranked = sorted(range(n_add), key=lambda k: -score[k])
  if PRUNE_ADAPT > 0 and n_expect:
    # Best evidence first, so a sparse model's leniency goes to its strongest
    # candidates and closes as it fills.
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
             sigma_cut=SIGMA_CUT):
  """ Add atoms where the difference map says one is missing.

  Returns (sites, calls, n_added). The sites are extended in place order, so a
  caller's pairing against them stays valid for the original entries.
  """
  from cctbx.array_family import flex

  sites = flex.vec3_double(sites)
  calls = list(calls) if calls else ["C"]*sites.size()
  n_original = sites.size()
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
    found = candidates(fft, placed, sites, sigma_cut, calls)
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

  # refine once more and drop the additions the refinement will not support,
  # before anything is typed
  if added:
    sites, calls, dropped = prune_added(f_obs, placed, sites, calls,
                                        n_original, u_values,
                                        strengths=strengths,
                                        budget=budget_atoms(placed),
                                        dists=dists)
    added -= dropped
  return sites, calls, added
