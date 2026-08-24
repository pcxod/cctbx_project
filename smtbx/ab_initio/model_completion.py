""" Finish a partial solution the way a crystallographer does.

  import model_completion
  sites, calls, added = model_completion.complete(f_obs, placed, sites, calls)

Completeness, not typing, is what blocks "the whole structure found and
correct". With the space group right, per-atom completeness is 0.8989 but only
60.3% of structures have every atom: the average structure is missing about a
tenth of itself, spread thinly, rather than a few structures failing badly.

That is not a search problem. Raising the peak cap from 1.3x to 3x the heavy
atom count moves atoms-found from 0.744 to 0.828 and completes **no additional
structure**, because the extra maxima land in structures that were already
incomplete. The missing atoms are not below the cut; 128 of 140 unmatched atoms
had nothing within 1.2 A at all.

So the model has to be *finished* rather than searched harder: refine what is
there, look at what the refinement cannot explain, and put an atom where the
difference map says one is missing. That is the Fo-Fc loop, and the machinery
already exists -- `make_refine_rows` builds exactly this map to compute the v5
physics features, and currently uses it only to *type* atoms already found.

Rules that keep it honest:

  * **A new atom must be chemically possible.** A peak closer than
    `MIN_BOND` to an existing atom is a ripple, not an atom; one further than
    `MAX_BOND` from everything is floating in solvent and is more likely noise.
  * **Stop when nothing significant remains.** The loop ends when the strongest
    residual peak falls below `SIGMA_CUT` sigma, not after a fixed count.
  * **Never remove.** Completion only adds; discarding is `cleanup_refine`'s
    job and mixing the two would make each impossible to measure.
"""
from __future__ import absolute_import, division, print_function

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# **Default 7.0 as of 22 August 2026**, raised from 3.0. A three-point ladder
# measured in one session against an identical baseline at 6.0: sigma 7 gives
# GOAL +0.0042 (95% CI [+0.0014, +0.0068], excludes zero) while sigma 8 gives
# +0.0031 with a CI spanning zero -- an interior optimum, with 8 overshooting.
# The trade is visible in the clauses: at 7 the count gain (+0.0081) exceeds the
# recall loss (-0.0065); at 8 the recall damage (-0.0123) swamps it.
SIGMA_CUT = float(os.environ.get("COMPLETION_SIGMA", "7.0"))
# Tunable, because the completion loop is where the remaining headroom is:
# it already lifts complete structures 0.6685 -> 0.7894 at sigma 3.0, and the
# goal is 0.90 and beyond. Lowering sigma adds more atoms; the risk is spurious
# ones, which cost correctness rather than completeness, so both halves have to
# be watched together.
MIN_BOND = float(os.environ.get("COMPLETION_MIN_BOND", "1.0"))
MAX_BOND = float(os.environ.get("COMPLETION_MAX_BOND", "2.6"))
MAX_ROUNDS = int(os.environ.get("COMPLETION_ROUNDS", "5"))
# **The proposal cut, adapted per structure to how full the model already is.**
# Measured on `phi1`, 18 August, over 2,296 correctly-solved structures: the
# model size divided by the volume budget separates the structures that end up
# short of atoms from the ones that end up right at **AUC 0.8998**, and it is a
# runtime quantity -- cell volume and the current model, never the deposited
# count. The healthy value is 0.233 of a budget set at 4.0 A^3/atom, i.e. one
# atom per 17.2 A^3, which reproduces the 17.3 measured independently as the
# median of the true composition.
#
# 202 of those 2,296 (8.8%) end at a median 0.750 of the reference size and
# score **zero** on the recall clause -- they cannot reach 90% recall with 75%
# of the atoms. A single global cut cannot serve both them and the 1,950 that
# are already right, which is what the cap ladder in
# `Goal-Metric-And-Where-The-Loss-Is` exhausted as a lever.
#
# So the cut is scaled by how short the model is, continuously and with a
# floor: `cut * occupancy**ADAPT`. A structure at 0.75 of its expected size
# sees 0.75x the cut at ADAPT=1 and admits weaker residual peaks; one already
# at its expected size is untouched. **It is a weight, not a test** -- an AUC
# of 0.90 is one structure in ten misjudged, and a switch would hand those the
# full effect. The loop recomputes it every round, so it relaxes only until the
# model fills and then closes itself.
ADAPT = float(os.environ.get("COMPLETION_ADAPT", "0"))
ADAPT_VOLUME = float(os.environ.get("COMPLETION_ADAPT_VOLUME", "17.2"))
ADAPT_FLOOR = float(os.environ.get("COMPLETION_ADAPT_FLOOR", "0.5"))
# Set per structure so the harness can stamp what actually happened; a mode
# that silently disables itself and a mode that ran and changed nothing must
# not look the same from the output.
LAST_ADAPT_WANT = 0
LAST_ADAPT_CUT = 0.0
_ADAPT_ANNOUNCED = False
MAX_ADD_PER_ROUND = 8
# Starting u_iso when the structure has not yet told us its own scale.
DEFAULT_U = float(os.environ.get("COMPLETION_DEFAULT_U", "0.05"))
# Pruning thresholds, all relative to the structure's own refined atoms.
PRUNE_CYCLES = int(os.environ.get("COMPLETION_PRUNE_CYCLES", "3"))
U_RATIO = float(os.environ.get("COMPLETION_U_RATIO", "3.0"))
MAX_SHIFT = float(os.environ.get("COMPLETION_MAX_SHIFT", "0.7"))
MIN_OCC = float(os.environ.get("COMPLETION_MIN_OCC", "0.3"))
# Weighted contest instead of per-atom thresholds. Measured: shift and
# occupancy carry no information at all (AUC 0.584 and a constant 1.000), so
# two of the three original criteria were dead and the third was doing all the
# work as a hard cut.
BUDGET_MODE = os.environ.get("COMPLETION_BUDGET", "1") != "0"
BUDGET_BETA = float(os.environ.get("COMPLETION_BETA", "1.0"))
# **The peak cap and the completion budget want different volumes.** 13 A^3 per
# non-H atom deliberately overshoots the measured median of 17.0, which is right
# for picking candidate peaks -- a spare candidate is cheap and a missing atom
# is fatal. It is wrong here: an addition that survives goes straight into the
# model, so overshooting by 31% buys 31% more noise. Separate knob, same rule,
# and it still reads only the cell.
BUDGET_VOLUME = float(os.environ.get("COMPLETION_VOLUME", "13.0"))
# **The ADP evidence has to apply to every addition, not only where the cell
# budget binds.** Ranking additions and keeping what the cell has room for
# leaves most structures untouched, because the volume rule overshoots: 27% of
# additions were removed and precision did not move (0.1325 -> 0.1268), since
# the structures the budget never reached kept everything.
#
# Measured over 17,883 additions, u_iso/median separates real from spurious at
# AUC 0.7757 -- `uratio > 2` removes 58.6% of the spurious atoms for 10.1% of
# the real ones. So that is the operating point, expressed as a weight rather
# than a verdict: a peak that stood well above the noise buys tolerance for a
# worse ADP, and can outvote the ADP entirely if it is strong enough.
#
#     keep while   log(UMAX) - log(uratio) + SIGMA_W*log(sigma/SIGMA_REF) > 0
#
# SIGMA_W defaults to 0 -- peak strength has *not* been shown to carry
# information here, and the one arm that weighted it did not improve precision.
# It stays available and off rather than assumed.
U_MAX = float(os.environ.get("COMPLETION_UMAX", "2.0"))
SIGMA_W = float(os.environ.get("COMPLETION_SIGMA_W", "0.0"))
SIGMA_REF = float(os.environ.get("COMPLETION_SIGMA_REF", "3.0"))

# **Occupancy-aware pruning.** Measured 23 August on the 4,041 solution-bad
# structures: `COMPLETION_PRUNE=0` is worth +0.0339 GOAL on them (28 of 826
# rescued, CI [+0.0206,+0.0484]) and -- against the prediction registered
# before the run -- the *count* clause improves more than recall does (+0.0400
# against +0.0116). The reason is in the stage trace: these models enter
# cleanup at recall 0.500 and leave at 0.194, so the prune is not trimming
# surplus, it is deleting correct atoms from a model that was already sparse.
#
# Turning the prune off globally is not the answer -- only that population was
# measured, and a global off switch is the shape `weighted-evidence-not-cutoffs`
# rejects. So the *bar* moves with how full the model already is, in the same
# log space and the same units as the evidence it is weighed against:
#
#     score = log(U_MAX) - log(uratio) + SIGMA_W*log(sigma/SIGMA_REF)
#             + PRUNE_ADAPT * min(log(CEIL), -log(occupancy))
#
# which is exactly `U_MAX * occupancy**-PRUNE_ADAPT` -- a half-full model
# tolerates twice the ADP inflation at PRUNE_ADAPT=1, a full one is untouched.
# The ceiling stops a two-atom model from keeping everything the map offered.
#
# **It is recomputed as the model fills**, so leniency closes itself: this is
# why the same hook failed on *acceptance* (`ADAPT_FLOOR=1.0` rescued 0 of 303
# -- it fires only once the model is already full, too late to prevent
# over-addition), and why it should work here, where the model is sparse at
# exactly the moment the prune runs.
#
# **On by default since 24 August 2026.** The A/B it was waiting for has run:
# two arms over the same 3,000-structure corpus, same slots, same session,
# differing in this one variable (verified by diffing the SETTINGS stamps).
#
#   solution-bad (n=1,500)   GOAL 0.0380 vs 0.0173 = +0.0207
#                            95% CI [+0.0140, +0.0280], 31 gained, 0 lost
#   healthy      (n=1,500)   GOAL +0.0007, CI [+0.0000, +0.0020], spans zero
#                            1 gained, 0 lost -- inert, which is the condition
#                            for shipping this rather than disabling the prune
#   null control             `sg` +0.0000 with ZERO structures changed, both
#                            populations
#
# Count improves nearly ten times as much as recall (+0.0240 against +0.0027),
# which is the mechanism the stage trace predicted: the prune was deleting
# correct atoms from models that were already too small, not trimming surplus.
#
# Only 1.0 has been measured against 0. The exponent is not tuned.
PRUNE_ADAPT = float(os.environ.get("COMPLETION_PRUNE_ADAPT", "1.0"))
PRUNE_ADAPT_CEIL = float(os.environ.get("COMPLETION_PRUNE_ADAPT_CEIL", "4.0"))

# **Whether an original site may lose its place to a better one.**
# `prune_added` exempts the originals deliberately -- they survived cleanup, and
# re-litigating them there would confuse two decisions. Measured on `v2b6`
# (2,234 structures) that exemption is what caps the model: 9.1% of the original
# atoms pair with nothing, against 28.6% of the additions, and because the two
# groups are similar in size they contribute the same 4,200-atom share of the
# error. An original that pairs with nothing occupies a slot in the cell budget
# that a real atom could have had, so with the count constrained -- which is
# what Florian asked for on 18 August -- the exemption is no longer free.
#
# On, every atom competes for the same slots on the same weighted evidence, so
# nothing is removed by a test of its own: an original is dropped only because
# some other atom made a better case for the room. Off by default until the A/B
# lands.
SWAP = os.environ.get("COMPLETION_SWAP", "0") != "0"
# A floor under the contest. Charge flipping needs a few atoms to mean
# anything, and a structure whose evidence is uniformly poor should come back
# thin rather than empty.
SWAP_FLOOR = int(os.environ.get("COMPLETION_SWAP_FLOOR", "4"))


def build_model(placed, sites, calls, u_values=None):
  """ An xray.structure at `sites`, typed by `calls`, started at `u_values`.

  Florian, 15 Aug: *"can we also add atoms with the median uiso already to have
  less work on that end?"* -- yes. Every atom used to start at a flat 0.05,
  which is wrong for both ends of the corpus: a tightly packed inorganic
  refines near 0.01 and a loose organic near 0.08. An added atom seeded at the
  structure's own median starts where its neighbours already are, so the
  refinement spends its cycles on the position instead of walking the ADP
  across an order of magnitude.

  It also sharpens the prune. Judging an atom by `u_iso > 3 x median` only
  means something if the atom did not *begin* far from the median -- seeded at
  the median, a large final u_iso is refinement pushing it there, which is the
  signal we actually want.
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

  **`unit_cell.distance` does not wrap.** It measures between the fractional
  coordinates exactly as given, so an atom at (0.95, 0.5, 0.5) and one at
  (0.05, 0.5, 0.5) -- one Angstrom apart across the cell boundary in a 10 A
  cell -- are reported as 9 A apart. Every pair that straddles a boundary was
  judged on a fictitious distance.

  That matters here more than anywhere: the bond window is the rule that makes
  a new atom chemically possible at all. With unwrapped distances a genuine
  bonded neighbour reads as far away and its peak is rejected, while a peak
  with nothing near it can read as bonded and be accepted. Both directions were
  happening, which is a large part of why only about one addition in eight was
  real.
  """
  # `smtbx.ab_initio.assemble` already solves this correctly for the descriptor
  # path -- it is `compaq -a` -- so the same routine is used here rather than a
  # second, subtly different one. Rounding the fractional difference is the
  # right answer only for an orthogonal cell; in an oblique cell the nearest
  # image can be a neighbouring lattice translation, which `_nearest_image`
  # checks and a bare `round` does not. Much of the COD is monoclinic or
  # triclinic, so that distinction is not academic.
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

  **There used to be a second definition of this name above `budget_atoms`,
  at 13.0 A^3.** Python kept the later one, so every caller has always had
  17.2 and every measured number in the vault was produced with it; the 13.0
  version was dead code that would have silently changed behaviour had anyone
  reordered the file. Removed 23 Aug 2026 on Florian's decision. The
  overshoot-for-peak-search argument it carried belongs to `MAX_PEAK_VOLUME`
  reasoning, not here -- this constant is the population's *typical* value.

  Distinct from `budget_atoms`, which is a *ceiling* completion may not exceed.
  This is the population's typical value, used as evidence that a model is
  short -- so it is set from the measured median (17.2 A^3/atom) rather than
  from the deliberately generous budget constant.
  """
  return max(1, int(placed.unit_cell().volume()
                    / max(placed.space_group().order_z(), 1)/ADAPT_VOLUME))


def map_sigma(real):
  """ A noise level the missing atoms cannot inflate.

  `sample_standard_deviation` is dominated by the very features completion is
  looking for. Measured on three omission tests: with a molybdenum deleted the
  map's sd is 10.28 against a robust 3.14, so the strongest residual peak --
  the molybdenum itself -- reads as 2.0 sigma and falls under a 3.0 cut. With a
  nitrogen and a carbon deleted the sd is inflated 6x and a **14 sigma** peak
  reads as 2.3. Both structures recovered nothing at all, and the signal was
  there the whole time.

  The median absolute deviation is not moved by a handful of large residuals,
  which is exactly the property wanted: the threshold should measure the noise,
  not the thing being detected. Scaled by 1.4826 so it matches the standard
  deviation for genuinely Gaussian noise and existing sigma cuts keep their
  meaning.

  Sampled on a stride for maps too big to sort -- a noise level needs three
  significant figures, not every grid point.
  """
  import numpy as np

  a = real.as_1d().as_numpy_array()
  if a.size > 400000:
    a = a[::max(1, a.size//400000)]
  med = float(np.median(a))
  mad = float(np.median(np.abs(a - med)))
  robust = 1.4826*mad
  if robust <= 0:
    # Degenerate map (more than half the points identical). Fall back rather
    # than divide by zero, and let the caller's cut do what it can.
    return float(real.sample_standard_deviation())
  return robust


def candidates(fft, placed, sites, sigma_cut=SIGMA_CUT):
  """ Residual maxima that could be atoms: strong, and at a bonding distance.

  Returns [(site, height_in_sigma)], strongest first.
  """
  from cctbx import maptbx
  from cctbx.array_family import flex

  real = fft.real_map_unpadded()
  sigma = map_sigma(real)
  if sigma <= 0:
    return []
  # `.all()` needs a cluster cap -- without one it asserts rather than
  # defaulting, which is how the first version failed on every structure.
  # Sized from the cell, not from the deposited atom count: the whole point is
  # to find atoms nobody has told us about.
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
    # Distance to the nearest existing atom, over symmetry images.
    best = 1e9
    for s in sites:
      dd = min_distance(uc, sg, site, s)
      if dd < best:
        best = dd
    if best < MIN_BOND or best > MAX_BOND:
      continue
    # `best` is the distance to the nearest existing atom, and it is worth
    # carrying: MIN_BOND is 1.0 A while no real bond is shorter than about 1.2,
    # and the density integration radius is 0.7 A, so an addition at 1.0 A from
    # a heavy atom has its sphere reaching to 0.3 A of that atom's centre and
    # reads as something far heavier than it is.
    out.append((site, height/sigma, best))
  out.sort(key=lambda t: -t[1])
  return out


# Per-added-atom evidence from the last prune, for calibration. Each entry is
# (index, u_ratio, shift, occupancy, peak_sigma, nearest_atom_distance).
# Written even when nothing is dropped, so
# a run can measure what a threshold *would* have done against the deposited
# truth instead of guessing the threshold first.
LAST_EVIDENCE = []
# `_nearest_image` loops over the ops of whatever group it is handed; the ops
# are already applied here, so it is given P1 and used purely for the lattice
# search. Built once because constructing a space group per call is not free.
SG_P1_CACHE = [None]
# Additions the cell cap removed *despite* their own evidence supporting them.
LAST_CAPPED = 0
# **How many of the surviving atoms are originals.** Without swap this is
# always the count the caller passed in, so nothing reads it; with swap an
# original can be dropped, and a caller that still assumes its first `n` atoms
# are the trusted ones would fit the carbon scale on additions. Written by
# every prune so the caller never has to infer it.
LAST_N_ORIGINAL = 0


def prune_added(f_obs, placed, sites, calls, n_original,
                u_values=None, mode="on", strengths=None, budget=None,
                dists=None, orig_strengths=None, swap=False):
  """ Refine once, then drop added atoms the refinement will not support.

  Florian, 15 Aug: *"will the spurious atoms not yield very jumpy ADPs and
  positions in a first refinement cycle? Can we detect them like that and
  remove them before we type any atoms?"* -- yes, and that is what a refinement
  is for. An atom placed on noise has nothing holding it: its u_iso inflates to
  absorb the absent density and it drifts from where the map put it. A real
  atom stays put with a u_iso like its neighbours'.

  The test is **relative to the structure's own atoms**, not an absolute
  window. A tightly-held inorganic refines everything near 0.01 and a loosely
  packed organic near 0.08, so a fixed cut would prune the second structure
  entirely and the first not at all -- the same reason the v5 physics features
  carry neighbour contrasts rather than raw u_iso.

  Only added atoms are eligible for removal. The originals survived cleanup
  already, and re-litigating them here would confuse two decisions.
  """
  import numpy as np
  from cctbx.array_family import flex
  from smtbx.ab_initio import dual_space

  # **Swap has work to do even when nothing was added.** The contest is then
  # between the originals alone. Returning here on `added == 0` would silently
  # confine the mode to structures completion had something to say about, which
  # are not a random subset of the corpus.
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
  # **Swap mode needs the originals measured, not assumed.** Their evidence is
  # computed by the same refinement in the same units; without
  # `orig_strengths` they would enter the contest with no peak-strength term
  # while every addition carries one, and would lose the tie on a missing
  # column rather than on the evidence. So the mode turns itself off rather
  # than run an unfair contest.
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
    # Wrapped too: a refining atom can cross a cell boundary, and an
    # unwrapped shift would then read as most of a cell rather than a fraction
    # of an Angstrom.
    d = [x - y for x, y in zip(scat[i].site, sites[i])]
    shift = uc.length([x - round(x) for x in d])
    occ = scat[i].occupancy
    k = i - n_original
    sig = strengths[k] if strengths and k < len(strengths) else -1.0
    # Peak strength travels with the ADP so the weighting between them can be
    # *fitted* on paired data rather than assumed -- BUDGET_BETA is currently
    # 1.0 by choice, not by measurement.
    dst = dists[k] if dists and k < len(dists) else -1.0
    LAST_EVIDENCE.append((i, u/ref_u if ref_u > 0 else 0.0, shift, occ, sig,
                          dst))
    drop = (u > U_RATIO*ref_u or u <= 0.0 or shift > MAX_SHIFT
            or occ < MIN_OCC)
    if BUDGET_MODE:
      drop = False        # decided below, by the contest rather than per atom
    # `mark` keeps everything and only records the evidence. The thresholds
    # above were chosen from physical intuition and, measured, removed nearly
    # every addition -- 1 structure in 250 kept any, against roughly three in
    # four before pruning. That is not a filter, it is an off switch, so the
    # cut has to be fitted to what the evidence actually separates.
    if drop and mode != "mark":
      dropped += 1
      continue
    keep_sites.append(sites[i])
    keep_calls.append(calls[i] if calls and i < len(calls) else "C")

  if BUDGET_MODE and mode != "mark":
    # **One list, in site order.** `_by_budget` reads `strengths[k]` for the
    # k-th *contestant*, so entering the originals as contestants without
    # prefixing their strengths would score atom k with atom k's peak from the
    # wrong group -- every original judged on some addition's peak height. The
    # combined list keeps index k meaning the same atom in both arrays.
    contest = (list(orig_strengths[:n_original]) + list(strengths or [])
               if swap else strengths)
    keep_sites, keep_calls, dropped = _by_budget(
      sites, calls, 0 if swap else n_original, contest, budget,
      floor=SWAP_FLOOR if swap else 0, n_expect=expected_atoms(placed))
    global LAST_N_ORIGINAL
    LAST_N_ORIGINAL = (_count_originals(sites, keep_sites, n_original)
                       if swap else n_original)
    if os.environ.get("COMPLETION_DEBUG"):
      # Printed whether or not the mode is on, and whether or not it dropped
      # anything: `requested` says what was asked for, `active` what actually
      # ran after the fairness guard, and the two differ exactly when the peak
      # heights did not arrive. Without this line an arm with no strengths
      # reads as an arm where the evidence supported every atom.
      print("SWAPDIAG requested=%d active=%d orig=%d added=%d kept=%d "
            "budget=%s room_binds=%d"
            % (int(SWAP), int(swap), n_original, sites.size() - n_original,
               keep_sites.size(), budget,
               int((budget or 0) < sites.size())))
  return keep_sites, keep_calls, dropped


def _count_originals(sites, keep_sites, n_original):
  """ How many of the first `n_original` sites survived the contest.

  Compared by value rather than by index: `_by_budget` rebuilds the list, so
  the surviving originals keep their order but not their positions, and the
  only thing that identifies them afterwards is the coordinate itself.
  """
  originals = set(tuple(sites[i]) for i in range(n_original))
  return sum(1 for s in keep_sites if tuple(s) in originals)


def _by_budget(sites, calls, n_original, strengths, budget, floor=0,
               n_expect=None):
  """ Keep the additions the cell has room for, best evidence first.

  Florian's standing rule is that noisy data gets weighted evidence, never a
  black-and-white cutoff -- *"a filter that removes the truth caps everything
  built on top"*. So no single quantity vetoes an atom. Two pieces of evidence
  are combined and the additions compete:

    peak strength   how far above the noise the residual maximum stood, which
                    is the map's own statement that something is there.
    u_iso ratio     measured AUC 0.775 over 878 additions -- a real recovery
                    refines to the structure's median (1.00), a spurious one
                    inflates to about 3.5x. Strong, and not decisive alone:
                    at `uratio > 3` a hard cut would still throw away 8.3% of
                    the genuine atoms.

  A strong peak with a mediocre ADP therefore outvotes a weak peak with a good
  one, which is what neither criterion could do by itself.

  The number kept is set by the **cell**, not by a score threshold: the
  asymmetric unit has room for `budget` non-H atoms, the model already holds
  `n_original`, so the difference is what completion may spend. That is the
  same volume rule the peak cap uses, and it reads nothing from the deposited
  model.
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
    # Positive means the evidence supports the atom. Everything is in log
    # space, so both terms are ratios and neither can dominate by its units.
    score.append(np.log(U_MAX) - np.log(max(uratio, 1e-3))
                 + SIGMA_W*np.log(max(sig, 1e-3)/SIGMA_REF))

  # Two independent reasons to drop an atom, and an atom must survive both:
  # its own evidence must be positive, and it must place inside whatever room
  # the cell leaves. Neither is applied as a veto on a single raw quantity.
  ranked = sorted(range(n_add), key=lambda k: -score[k])
  if PRUNE_ADAPT > 0 and n_expect:
    # **The bar moves with how full the model is, and it moves as it fills.**
    # Walked best-evidence-first, so the leniency a sparse model is given is
    # spent on its strongest candidates and has closed again by the time the
    # model reaches its expected size. `n_original + len(keep)` is the model
    # size either way: in swap mode the originals are contestants and
    # `n_original` is passed as 0, so they are counted inside `keep` instead.
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
  # **A floor, only in swap mode.** With the originals in the contest the
  # evidence cut can in principle empty a structure, and a model of two atoms
  # is not a smaller answer but a different question -- every metric
  # downstream divides by it. `floor` is 0 in the shipped path, where the
  # originals are never at risk and this line cannot fire.
  if floor and len(keep) < floor:
    keep = set(ranked[:min(floor, n_add)])
  # **Say which limit did the work.** The evidence cut alone costs 9.1% of the
  # genuine additions (658 of 7,217 measured), yet the arm running both limits
  # kept barely half of them -- so the cell cap is dropping atoms whose own
  # evidence supported them. Counted rather than inferred, because the two
  # limits are easy to confuse and only one of them has ever been shown to buy
  # anything.
  global LAST_CAPPED
  LAST_CAPPED = sum(1 for k in ranked[room:] if score[k] > 0.0)
  out_sites, out_calls = flex.vec3_double(), []
  # **Re-key the evidence to the surviving atoms.** LAST_EVIDENCE is written
  # against pre-prune indices; once atoms are removed the caller's index `i` no
  # longer selects the row that describes it, so every ADDEV line would carry
  # another atom's ADP. The metrics are unaffected -- they never read the
  # evidence -- but the calibration data would be quietly scrambled, which is
  # worse than missing.
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
  # Set before anything can fail. A structure that returns early would
  # otherwise leave the previous structure's count standing, which is the kind
  # of stale global that reads as a plausible number.
  # Reset per structure. A stale global here would report the previous
  # structure's expected size next to this one's model, which is the failure
  # mode `LAST_N_ORIGINAL` is commented for two lines up.
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
    # Carry this round's refined ADPs into the next one, and seed whatever we
    # add at the median of them.
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
    found = candidates(fft, placed, sites, cut)
    if not found:
      break
    for site, _sig, _dst in found[:MAX_ADD_PER_ROUND]:
      sites.append(site)
      strengths.append(_sig)
      dists.append(_dst)
      if u_values is not None:
        u_values.append(med)
      # Typed as carbon: the geometry aid retypes everything afterwards, and
      # guessing here would put an element in the model that the difference
      # map cannot justify.
      calls.append("C")
      added += 1
    if len(found) <= 0:
      break

  # Refine once more and drop the additions the refinement will not support.
  # Done here, before anything is typed, so a spurious atom never reaches the
  # classifier -- and so the atom count the caller sees is the one that
  # survived physics rather than the one the map suggested.
  mode = os.environ.get("COMPLETION_PRUNE", "1")
  # **Swap mode runs the prune even when nothing was added.** The contest is
  # then between the originals alone, which is the arm that says whether the
  # count clause is met by removing atoms or only by adding them; gating it on
  # `added` would silently skip every structure completion had nothing to say
  # about, and those are not a random subset.
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
    # `added` is what the caller reports and what the count metric reads, so it
    # has to describe the model handed back. In swap mode a drop can fall on an
    # original, and subtracting those from `added` alone would report a model
    # larger than it is.
    added = max(0, sites.size() - (n_before - added))
  return sites, calls, added


if __name__ == "__main__":
  print(__doc__)
  print("sigma cut %.1f, bond window %.1f-%.1f A, max %d rounds"
        % (SIGMA_CUT, MIN_BOND, MAX_BOND, MAX_ROUNDS))
