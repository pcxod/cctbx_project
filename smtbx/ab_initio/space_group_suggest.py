""" Space-group suggestions from a P1 charge-flipping solution.

Produces a short ranked list of candidate space groups rather
than one answer. The measurements behind every choice here are recorded in
`notes/obsidian/` in the working repository; the summary is:

- Solving in P1 and recovering the symmetry afterwards is the best blind
  strategy. It solves more entries than any alternative (1938 of 1986 on a
  uniform Crystallography Open Database sample), is about 5x faster than
  solving in an assumed group, and beats a plausible-but-wrong assumed group
  by 13 percentage points. A wrong assumption is the expensive case.
- Offering the best few *solutions* is worthless -- the solution is not what
  fails. Offering the best few *space groups* is worth 5.6 points. Hence this
  module.
- **Fit-based figures of merit cannot rank space groups.** The charge-flipping
  solution is a P1 solution, so imposing symmetry can only degrade the fit, and
  correlation peak height and R1 both prefer the lowest symmetry regardless of
  the truth. They are deliberately not used for ranking here.

The evidence sources below are disjoint, and each is used only for what it can
actually decide:

  systematic absences  screws and glides -- and, crucially, a *hard filter*:
                       a group predicting reflections absent that are
                       observed strong is refuted outright
  <|E^2-1|> statistics centrosymmetry, which no map-folding argument can
                       decide (see the note in `symmetry_agreement` below)
  cell metric          which groups are geometrically possible at all
  frequency prior      a tie-break, and the single strongest ranker measured

None substitutes for another.
"""
from __future__ import absolute_import, division, print_function

from libtbx import group_args

# Space-group frequency among small-molecule structures, most common first, by
# International Tables number. Counted from the 1986-entry uniform sample of
# the COD used throughout this work: 14 (P2(1)/c) 792, 2 (P-1) 440, 15 (C2/c)
# 166, 19 (P2(1)2(1)2(1)) 130, 4 (P2(1)) 91, 61 (Pbca) 70 ...
#
# This is a genuinely strong ranker -- it beat every physical figure of merit
# tested, including triplet phase consistency -- and that is exactly why it is
# used only to break ties here. A prior this concentrated will confidently name
# P2(1)/c for a structure that is not P2(1)/c, and the structures where a user
# most needs help are precisely the unusual ones.
FREQUENCY_ORDER = (14, 2, 15, 19, 4, 61, 33, 62, 9, 29, 60, 5, 88, 7, 56, 13,
                   43, 1, 12, 11, 18, 148, 96, 92, 41, 20, 36, 31, 86, 114)

# <|E^2-1|> is ~0.968 for a centric distribution and ~0.736 for an acentric
# one; the midpoint is the classical dividing line and, measured blind on the
# COD sample, classifies centrosymmetry correctly 92.0% of the time.
CENTRIC_THRESHOLD = 0.858

# The symmetry agreement factor for an inversion centre, below which the
# structure is taken to be centrosymmetric. The factor is normalised to be
# about unity for random phases, so a value below it means the phases are
# genuinely consistent with a centre rather than accidentally so.
#
# **0.30, and 0.60 was tried and reverted.** As a classifier 0.30 is plainly
# wrong for these maps: it calls almost everything acentric, 0.238 accuracy
# over 2,818 structures, worse than guessing the commoner class. Sweeping
# against known centricity puts the optimum at 0.60 (0.794 alone), and
# letting alpha_0 decide inside +-0.02 of the <|E^2-1|> cut lifts centricity
# classification from 0.894 to 0.909.
#
# Both changes made the actual product **worse**: on 92 frozen real structures
# top1 fell 0.870 -> 0.804 and top3 0.891 -> 0.859, and the two arms scored
# identically, so all of the damage came from the threshold.
#
# The reason is that `alpha_0`'s job in this module is not to classify. It
# feeds `disputed`, and `disputed` *suppresses centricity as a ranking key*.
# At 0.30 it disagreed with the statistic constantly, so centricity was
# usually excluded from the sort; at 0.60 it agrees, centricity is admitted,
# and an 89%-accurate key does more harm than good ahead of the absence
# evidence. The miscalibrated threshold was working as a safety valve.
#
# So: a better classifier made a worse product. If centricity is ever to be
# used as a ranking key it has to earn that on the end-to-end number, not on
# its own accuracy -- which is what `ALPHA_0_DECIDES_WITHIN` is left in place
# to test.
ALPHA_0_CENTRIC_THRESHOLD = 0.30

# How close <|E^2-1|> must be to its own cut before alpha_0 may overrule it.
# Zero disables it. Kept because the mechanism is sound and only the coupling
# to `disputed` spoiled it; re-test it against the end-to-end number, not
# against centricity accuracy.
ALPHA_0_DECIDES_WITHIN = 0.0

# Fewest predicted absences a group must have before the absence test is
# allowed an opinion about it. A 2(1) screw predicts only the odd axial
# reflections -- a couple of dozen at most -- and a group with none at all
# (P222, Pmmm) predicts nothing to check.
MIN_ABSENCES_TO_JUDGE = 10

# A coverage margin below this means the group's predicted absences are no more
# missing than the file is incomplete anyway -- it explains nothing. Set low
# because coverage refutation is the weaker claim of the two: absent-from-file
# is only evidence if the file is one that deletes absences, and that is
# inferred rather than known. On the COD dev split the true group's margin
# averaged 0.84, so nothing near the truth is at risk from this cut.
MIN_COVERAGE_MARGIN = 0.05

# Decimals the absence ratio is rounded to before candidates are compared on
# it.
#
# **Two, and one was tried and refuted.** The argument for coarsening was
# specific: C2/c predicts strictly more absences than C2/m -- the c-glide class
# on top of the C-centring -- so it explains everything C2/m explains and more,
# yet lost to it three times on real data because its glide class leaked to
# 0.01 while C2/m's pure centring class sat at 0.00, and the bucket is
# consulted before the count. Glide-versus-mirror was five of ten failures.
#
# Rounding to one decimal to collapse that distinction cost more than it
# bought: on 92 frozen real structures top1 fell 0.870 -> 0.837 and top3
# 0.891 -> 0.870. The fine bucket is discriminating correctly far more often
# than it misfires on the glide case, so the glide failure needs a fix that
# targets it rather than one that blunts the ranking everywhere.
#
# Overridable only so that A/B can be re-run; not a user setting.
ABSENCE_BUCKET_DECIMALS = 2


def _absence_bucket_decimals():
  import os

  try:
    return int(os.environ.get("SMTBX_ABSENCE_BUCKET_DECIMALS",
                              ABSENCE_BUCKET_DECIMALS))
  except ValueError:
    return ABSENCE_BUCKET_DECIMALS

# Refute a group when its predicted absences carry more than this fraction of
# the mean intensity of everything else. Measured on real samples the two
# populations are far apart: a true c-glide class came in around 0.08 and a
# true 2(1) screw around 0.05, while glide planes that are not there sat at
# 0.9 and above. A quarter is comfortably between them, so nothing near the
# boundary has to be adjudicated.
DEFAULT_MAX_ABSENCE_RATIO = 0.25

# How many leading candidates enter the pairwise run-off. The contest is
# O(k^2) absence tests over the observed list, so this is the one knob that
# decides its cost; eight covers every case seen where the truth was in the
# shortlist at all, and a candidate the file-level evidence has already ranked
# tenth is not in contention.
PAIRWISE_RUNOFF_SIZE = 8


def e_squared_minus_one(f_obs):
  """ <|E^2-1|>, computed in P1 so no space-group knowledge leaks in.

  Deliberately normalised in P1 and rescaled to <E^2> = 1 by hand: using the
  candidate group's own symmetry would make the statistic depend on the answer
  it is being used to choose.
  """
  from cctbx import crystal, sgtbx
  from cctbx.array_family import flex

  try:
    p1 = f_obs.customized_copy(
      crystal_symmetry=crystal.symmetry(
        unit_cell=f_obs.unit_cell(),
        space_group_info=sgtbx.space_group_info("P 1")))
    p1.setup_binner_counting_sorted(reflections_per_bin=200)
    norm = p1.amplitude_quasi_normalisations()
    sel = norm.data() > 0
    e = p1.data().select(sel)/norm.data().select(sel)
    if e.size() < 20:
      return None
    e_sq = flex.pow2(e)
    mean = flex.mean(e_sq)
    if mean <= 0:
      return None
    return flex.mean(flex.abs(e_sq/mean - 1.0))
  except Exception:
    return None


def absence_ratio(f_obs_p1, space_group):
  """ How strong this group's predicted absences are, relative to everything
  else. Returns (ratio of mean intensities, n_predicted_absent).

  A ratio near zero means the class really is absent and the group explains
  why those reflections are missing; a ratio near one means the reflections are
  as strong as any other and the group is refuted.

  **Relative strength, not significance.** The obvious test -- count how many
  predicted absences exceed I/sigma = 3 -- is wrong, and wrong in a way that
  gets worse as the data improve. On a real P2(1)/c structure measured with Cu
  radiation the c-glide class averaged |F| = 2.24 against 13.51 for its
  neighbours, unmistakably absent, and yet 61% of it still cleared I/sigma = 3
  because the sigmas are small. Significance asks "is this reflection non-zero",
  which multiple scattering and glide leakage answer yes to on any good data
  set. The question that matters is "is this class weak compared to the rest",
  which is scale-free and needs no sigmas at all.

  Intensities rather than amplitudes because absence is a statement about
  intensity, and squaring separates the populations further: the same structure
  gives 0.28 in |F| and about 0.08 in I.

  Only refutation is claimed, never confirmation. A merged file has already had
  the *true* group's absences removed, so "this class is missing from the file"
  cannot be told from "it was never measured" -- which is why the ratio is
  taken over reflections that are present, and a group predicting none of them
  gets no credit rather than a perfect score.
  """
  from cctbx.array_family import flex

  indices = f_obs_p1.indices()
  absent = flex.bool(space_group.is_sys_absent(indices))
  n_absent = absent.count(True)
  if n_absent == 0:
    return 0.0, 0

  intensities = flex.pow2(f_obs_p1.data())
  mean_absent = flex.mean(intensities.select(absent))
  rest = intensities.select(~absent)
  if rest.size() == 0:
    return 0.0, n_absent
  mean_rest = flex.mean(rest)
  if mean_rest <= 0:
    return 0.0, n_absent
  return mean_absent/mean_rest, n_absent


# Observations per unique reflection, above which a file is taken to have
# measured its absences rather than merged them away. Redundancy 1.0 means one
# observation each: the systematically absent classes are simply not in the
# file and nothing can be said about their strength. Florian's corpus runs
# 1.75-28x; COD is merged.
UNMERGED_REDUNDANCY = 1.5


def _discriminating_ranking(f_obs=None):
  """ Whether the discriminating-absence rank applies to *this file*.

  **Measured on both sides, and the answer differs.** The rule fixes real
  unmerged data -- sg_top3 +0.078 over 64 structures with deposited phases, and
  eight known glide-versus-mirror failures rank first -- and it *costs* merged
  data: on 10,235 COD entries, top1 -0.019 and top3 -0.027, gaining 111 and
  losing 304. That is the second time an idea in this area has been refuted on
  merged COD, and it is not a coincidence:

    On merged data the true group's absences were deleted, so it has nothing
    left to claim and its discriminating set is empty. Ranking on what a group
    uniquely claims therefore demotes exactly the right answer.

    On unmerged data the true group's absences were measured at noise, so it
    is the one making a claim, while a wrong higher-symmetry group predicts
    nothing extra and cannot be refuted at all.

  So this is not a constant to be tuned. It depends on whether the file merges
  its absences away, which is a property of the data and is measurable before
  any ranking happens.

  `SMTBX_DISCRIMINATING_ABSENCES` overrides for A/B: 0 never, 1 always, unset
  or `auto` decides per file.
  """
  import os

  override = os.environ.get("SMTBX_DISCRIMINATING_ABSENCES", "auto")
  if override == "0":
    return False
  if override == "1":
    return True
  if f_obs is None:
    return False
  try:
    unique = f_obs.merge_equivalents().array().size()
    if unique <= 0:
      return False
    return (f_obs.size()/unique) >= UNMERGED_REDUNDANCY
  except Exception:
    # Cannot tell, so do not change the behaviour that is measured best on the
    # larger corpus.
    return False


# Worst phase disagreement a candidate's rotations may show before it is
# refuted. cctbx's own `structure_factor_symmetry` uses 0.25 to accept an
# operator and 0.5 to reject one, and the measured separation is wide enough
# that the exact value between them does not matter: on five real structures
# where we picked the wrong point group the truth scored 0.0000 and the wrong
# answer 0.71 to 1.11.
MAX_POINT_GROUP_DISAGREEMENT = 0.5

# **The same statistic, as a weight instead of a veto.** `phi_point_group` is
# the evidence behind the refutation above, and as a cutoff it was a disaster:
# it refuted the true group in 6,367 of 10,235 COD structures, which is why the
# gate is off by default. Used the way Florian's standing rule asks for --
# graded, able to be outvoted, never removing a candidate -- the same numbers
# are worth 3.3 points of top-1 space group.
#
# Measured 18 August 2026 on the module's own feature tables, threshold fitted
# on `split_dev` only:
#
#   split      coverage order   + phi weight   gain    fires on
#   dev        0.8954           0.9302         +0.0348   0.236
#   confirm    0.8998           0.9292         +0.0293   0.237
#   sealed     0.8968           0.9294         +0.0327   0.230   (35,922)
#
# The weight is the candidate's own margin over the runner-up, so a phi that
# cannot separate the leaders contributes nothing and one that separates them
# sharply can outvote the absence order. `PHI_MARGIN_SCALE` sets how loudly it
# may argue; the result is flat between 0.08 and 0.32, so it is not a fitted
# threshold in disguise. Applied *before* the discriminating run-off, because
# that run-off is a targeted glide-versus-mirror fix and should keep the last
# word among the leaders.
import os
PHI_RERANK = os.environ.get("SG_PHI_RERANK", "0") != "0"
PHI_MARGIN_SCALE = float(os.environ.get("SG_PHI_MARGIN_SCALE", "0.16"))


# How many of the module's own leaders alpha may not touch. Two, from
# measurement: keeping one regressed COD top3 on every held-out split, keeping
# three is the unchanged module.
ALPHA_KEEP_LEADERS = 2

# How many candidates below the leaders alpha is computed for. Each one costs
# an FFT pair per symmetry operator, and computing it for a full candidate list
# made six structures take over ten minutes -- unacceptable when the target is
# SHELXT's speed. Measured over the window: 4, 6, 8, 12 and unlimited all give
# COD top3 in 0.9775-0.9805 and measured top3 in 0.839-0.881, against baselines
# of 0.9726 and 0.7458. The spread on the measured corpus is four structures out
# of 118 -- noise -- so this is set to a cheap value inside the flat region
# rather than to whichever number scored highest.
ALPHA_WINDOW = 6


def _alpha_shortlist():
  """ Whether alpha orders the shortlist below the module's leaders.

  Off by default. `SMTBX_ALPHA_SHORTLIST=1` enables it, so the A/B is one code
  version under two environments rather than two versions of the file -- the
  same discipline as `SMTBX_COVERAGE_REFUTES`, and the reason the point-group
  filter's regression was attributable when it appeared.

  An earlier attempt put alpha *inside* the sort key as a last tiebreak. It
  measured -0.015 top1 and -0.038 solved on the measured corpus: too low in the
  key to help and still able to disturb orders. Replaced, not kept behind a
  second switch.
  """
  import os
  return os.environ.get("SMTBX_ALPHA_SHORTLIST", "0") not in ("", "0", "false",
                                                              "False")


def _coverage_refutes():
  """ Whether the coverage test may remove a candidate outright.

  Off by default from 7 August 2026: it demotes instead. Set
  `SMTBX_COVERAGE_REFUTES=1` for the old behaviour, which is what every
  measurement before that date was taken under.
  """
  import os

  return os.environ.get("SMTBX_COVERAGE_REFUTES", "0") == "1"


def _point_group_compute():
  """ Compute the phase agreement without letting it refute anything. """
  import os

  return os.environ.get("SMTBX_POINT_GROUP_COMPUTE", "0") == "1"


def _point_group_filter():
  """ Whether the phase-based point-group refutation is on. **Off by default.**

  **Refuted on COD, catastrophically, and the demonstration that motivated it
  was n=5.** On five real structures where we had picked the wrong point group
  the truth scored phi_sym 0.0000 and the wrong answer 0.71-1.11, which looked
  decisive. On 10,235 merged COD entries with deposited phases the same filter
  took sg_top1 from 0.9314 to 0.3145: it gained 53 and **refuted the true group
  in 6,367 structures**.

  So the factor is not measuring what those five cases suggested. It is not yet
  understood why -- candidates for the difference include the array layout the
  factor is handed (merged-then-expanded rather than a genuine P1 solution),
  the origin, and a threshold that has no business being a constant. Until that
  is understood this must stay off: a filter that discards the right answer two
  times in three is worse than no filter.

  `SMTBX_POINT_GROUP_PHASES=1` re-enables it for further investigation.
  """
  import os

  return os.environ.get("SMTBX_POINT_GROUP_PHASES", "0") == "1"


def point_group_agreement(f_calc_in_p1, space_group, cache=None):
  """ Worst phase disagreement over a candidate's distinct rotations.

  **The one kind of evidence this module had no access to.** Everything else
  here judges a candidate by its systematic absences -- a few hundred
  reflections -- or by intensity statistics. This asks whether the *phases* of
  the P1 solution actually satisfy the candidate's rotations, over every
  reflection, which is what SHELXT ranks on ([[Sheldrick-2015-SHELXT]] calls it
  alpha) and what VLD avoids needing by working in the correct group from the
  start.

  Measured with deposited phases on the five real structures where we chose the
  wrong point group:

      DB_0099_auto      P 21 21 21  0.0000   we picked P m m m    0.8998
      Co_salen_py_bad   C m c 21    0.0000             C 2 2 21   0.7136
      DB_0153_1         A b a 2     0.0000             A m m m    1.1136
      DB_0112_auto      C 1 c 1     0.0000             C 1 2 1    0.7709
      DB_0081_auto      I 1 a 1     0.0000             I 1 2 1    0.9950

  **Rotations only, deliberately.** The factor is insensitive to the intrinsic
  translation: a c-glide `x,-y,z+1/2` and a mirror `x,-y,z` both score 0.0000 on
  a structure that has the glide, because they share a rotation. So this cannot
  decide glide versus mirror and must not be asked to -- that is the absence
  test's job, and the two are complementary rather than competing. Feeding the
  full operator instead would produce a confident wrong answer on exactly the
  decision we are worst at.

  `cache` maps a rotation to its factor across candidates in one call; the
  ninety candidates of a Laue class share very few distinct rotations, so
  without it the same handful is recomputed dozens of times.

  Returns None when it cannot be computed, which is treated as no evidence.
  """
  from cctbx import sgtbx

  if f_calc_in_p1 is None:
    return None
  if cache is None:
    cache = {}
  worst = 0.0
  try:
    for op in space_group:
      r = op.r()
      if r.is_unit_mx():
        continue
      key = tuple(r.num())
      if key not in cache:
        cache[key] = f_calc_in_p1.symmetry_agreement_factor(
          sgtbx.change_of_basis_op(sgtbx.rt_mx(r)),
          assert_is_similar_symmetry=False)
      worst = max(worst, cache[key])
  except Exception:
    return None
  return worst


def pairwise_absence_winner(f_obs_p1, group_a, group_b, max_ratio=None):
  """ Which of two candidates the reflections where they *differ* support.

  Taking the intersection over every candidate does not work: the list contains
  P1, which predicts no absences, so the shared set is empty and each candidate
  is scored over its whole prediction again -- diluted exactly as before. The
  comparison that means something is pairwise, because two candidates from one
  Laue class differ by a specific set of reflections and nothing else.

      A_only = absent(A) and not absent(B)     what A claims and B does not
      B_only = absent(B) and not absent(A)

  A wins when its own claim is weak; it loses when that claim is strong, which
  is a direct refutation rather than a preference. When only one side makes a
  claim the answer is decided by that claim alone -- which is the glide-versus-
  mirror case, where the mirror predicts nothing extra and so can neither be
  supported nor refuted by the data.

  Returns +1 if A is favoured, -1 if B is, 0 if the reflections cannot say.
  """
  from cctbx.array_family import flex

  if max_ratio is None:
    max_ratio = DEFAULT_MAX_ABSENCE_RATIO
  indices = f_obs_p1.indices()
  a = flex.bool(group_a.is_sys_absent(indices))
  b = flex.bool(group_b.is_sys_absent(indices))
  a_only, b_only = a & ~b, b & ~a
  n_a, n_b = a_only.count(True), b_only.count(True)
  if n_a < MIN_ABSENCES_TO_JUDGE and n_b < MIN_ABSENCES_TO_JUDGE:
    return 0

  intensities = flex.pow2(f_obs_p1.data())
  def weakness(mask, n, other):
    """ Mean intensity of the disputed class against the undisputed rest. """
    if n < MIN_ABSENCES_TO_JUDGE:
      return None
    rest = intensities.select(~(a | b))
    if rest.size() == 0:
      return None
    mean_rest = flex.mean(rest)
    if mean_rest <= 0:
      return None
    return flex.mean(intensities.select(mask))/mean_rest

  wa, wb = weakness(a_only, n_a, b_only), weakness(b_only, n_b, a_only)
  # A claim that is present and strong refutes the group making it. That is the
  # only direction in which absences can speak with confidence, and it is the
  # rule the whole module is built on.
  if wa is not None and wa > max_ratio:
    return -1
  if wb is not None and wb > max_ratio:
    return +1
  # Both claims weak, or only one side claims anything: prefer the group that
  # explains more of the pattern, which is the one still making a claim.
  if wa is not None and wb is None:
    return +1
  if wb is not None and wa is None:
    return -1
  return 0


def copeland_scores(f_obs_p1, entries, max_ratio=None):
  """ How many pairwise contests each candidate wins. Order by this.

  A comparator built from `pairwise_absence_winner` need not be transitive --
  three candidates can beat each other in a cycle when the disputed classes
  overlap -- and handing a non-transitive comparator to `sort` gives an order
  that depends on the input order, which is the kind of instability that shows
  up as unexplained run-to-run variation. Counting wins (a Copeland score) is
  well defined whether or not the relation is transitive.
  """
  wins = [0]*len(entries)
  for i in range(len(entries)):
    for j in range(i + 1, len(entries)):
      verdict = pairwise_absence_winner(f_obs_p1,
                                        entries[i].space_group_info.group(),
                                        entries[j].space_group_info.group(),
                                        max_ratio=max_ratio)
      if verdict > 0:
        wins[i] += 1
      elif verdict < 0:
        wins[j] += 1
  return wins


def shared_absences(f_obs_p1, candidates):
  """ The predicted absences every candidate agrees on. They decide nothing.

  Candidates drawn from one Laue class have **nested** absence sets: they share
  the lattice centring and differ only in the screw axes and glide planes that
  distinguish them. C2/c predicts the C-centring class *and* the c-glide class;
  C2/m predicts only the centring. The centring part is thousands of
  reflections and is genuinely absent for both, so scoring a candidate over its
  whole set buries the handful of reflections that actually decide the question
  under evidence that applies equally to its rival.

  Measured on `DB_0148_1` (true C 1 2/c 1), with deposited phases so this is not
  map noise: C2/c's 10,428 predicted absences are ~99% centring, its c-glide
  class is diluted about eighty to one, and C2/m -- which predicts nothing
  beyond the centring, so has nothing present to measure -- ranked first with
  the truth sixth.

  Returns a flex.bool over `f_obs_p1.indices()`, true where every candidate
  predicts absence. Empty when there is one candidate or none.
  """
  from cctbx.array_family import flex

  indices = f_obs_p1.indices()
  if len(candidates) < 2:
    return flex.bool(indices.size(), False)
  shared = None
  for sgi in candidates:
    absent = flex.bool(sgi.group().is_sys_absent(indices))
    shared = absent if shared is None else (shared & absent)
  return shared if shared is not None else flex.bool(indices.size(), False)


def discriminating_absence_ratio(f_obs_p1, space_group, shared):
  """ `absence_ratio` over this group's *own* absences only.

  Same statistic, same scale-free reasoning -- mean intensity of the class
  against the mean of everything else -- but restricted to the reflections this
  group declares absent and its rivals do not. That is the evidence a
  crystallographer actually looks at when choosing between a glide and a
  mirror.

  A group whose predictions are entirely shared gets `(0.0, 0)`: no
  discriminating evidence, therefore no claim. That is the same principle
  `absence_ratio` already states -- "a group predicting none of them gets no
  credit rather than a perfect score" -- applied relative to the competitors
  rather than relative to the file.
  """
  from cctbx.array_family import flex

  indices = f_obs_p1.indices()
  absent = flex.bool(space_group.is_sys_absent(indices))
  own = absent & ~shared
  n_own = own.count(True)
  if n_own == 0:
    return 0.0, 0

  intensities = flex.pow2(f_obs_p1.data())
  # The comparison population is everything this group does not call absent.
  # Deliberately *not* "everything outside the discriminating set": the shared
  # absences are genuinely missing for every candidate, and letting them into
  # the denominator would drag the mean down and make every ratio look small.
  rest = intensities.select(~absent)
  if rest.size() == 0:
    return 0.0, n_own
  mean_rest = flex.mean(rest)
  if mean_rest <= 0:
    return 0.0, n_own
  return flex.mean(intensities.select(own))/mean_rest, n_own


def completeness_reference(f_obs_p1):
  """ The reflection list this cell *could* have, and which of it is present.

  Returned once and reused for every candidate, because building it is a
  `miller.build_set` and a set membership test over the whole sphere -- cheap
  once, wasteful ninety times.

  Returns (complete indices, present flags, baseline missing fraction), or None
  if no usable sphere can be built.
  """
  import os

  from cctbx import miller
  from cctbx.array_family import flex

  # A benchmark switch, not a user setting. Returning None here disables the
  # coverage test everywhere -- every candidate falls back to the intensity
  # test -- which is how the paired A/B isolates what coverage contributed from
  # everything else that changed since the baseline run.
  if os.environ.get("SMTBX_ABSENCE_COVERAGE") == "0":
    return None

  try:
    d_min = f_obs_p1.d_min()
    complete = miller.build_set(
      crystal_symmetry=f_obs_p1.crystal_symmetry().cell_equivalent_p1(),
      anomalous_flag=False, d_min=d_min).indices()
  except Exception:
    return None
  if complete.size() == 0:
    return None
  seen = set(f_obs_p1.indices())
  present = flex.bool([h in seen for h in complete])
  baseline = 1.0 - present.count(True)/complete.size()
  return complete, present, baseline


def absence_coverage(reference, space_group):
  """ How much of this group's predicted-absent class is *missing* from the
  file, in excess of how incomplete the file is generally.

  Returns (margin, n_predicted_absent) with margin in -1..1, or (None, 0).

  **This is the test for merged data**, where `absence_ratio` has nothing to
  work with: CIF deposition deletes the systematically absent reflections, so
  the true group predicts absences that are not in the file at all. Measured
  over 1,193 COD entries, `absence_ratio` could judge the true group for 0.6%
  of them; this can judge 92.5%.

  Being deleted is exactly what an absence looks like in a merged file, and a
  crystallographer reading one reasons the same way -- they notice which
  classes are gone rather than measuring the intensity of nothing. The excess
  over the general incompleteness is what matters: a file missing 30% of
  everything makes any group look good on raw coverage.

  **It is a filter, not a discriminator.** Many groups have every predicted
  absence missing and tie at the same margin -- a mean tie of 8.4 candidates
  out of 83 on the COD dev split. It almost never eliminates the truth (99.2%),
  which is what makes it safe to apply first, but the ranking still has to
  happen among the survivors.
  """
  complete, present, baseline = reference
  absent = space_group.is_sys_absent(complete)
  n_absent = absent.count(True)
  if n_absent == 0:
    return None, 0
  n_present_absent = (absent & present).count(True)
  return (1.0 - n_present_absent/n_absent) - baseline, n_absent


def _settings_compatible_with(unit_cell, laue_number=None, numbers=None,
                              relative_length_tolerance=0.01):
  """ Conventional space-group *settings* the cell admits, not just groups.

  Iterating settings rather than the 230 group numbers is essential, not
  cosmetic. P2(1)/n and P2(1)/c are the same group (number 14) in different
  settings, and they differ in exactly which reflections are systematically
  absent -- so for a given cell only one of them can be right, and offering
  only the standard setting would be wrong for a large fraction of real
  structures. P2(1)/n alone accounts for 8133 entries in a 534,824-entry
  Crystallography Open Database census, against 14420 for P2(1)/c.

  Choosing between settings is left to the absence test, which is precisely the
  evidence that distinguishes them.
  """
  from cctbx import sgtbx

  out, seen = [], set()
  for symbols in sgtbx.space_group_symbol_iterator():
    hall = symbols.hall()
    if hall in seen:
      continue
    try:
      group = sgtbx.space_group(hall)
      if not group.is_compatible_unit_cell(
          unit_cell, relative_length_tolerance=relative_length_tolerance):
        continue
      sgi = sgtbx.space_group_info(group=group)
      if numbers is not None and sgi.type().number() not in numbers:
        continue
      if laue_number is not None:
        laue = sgtbx.space_group_info(
          group=group.build_derived_laue_group()).type().number()
        if laue != laue_number:
          continue
    except Exception:
      continue
    seen.add(hall)
    out.append(sgi)
  return out


def r_weak(f_obs, f_calc, weak_fraction=0.10):
  """ Are the reflections observed weak also calculated weak? Small is good.

  The mean of E_c^2 over the `weak_fraction` of unique reflections with the
  smallest observed normalised structure factors, **including systematically
  absent ones**.

  This fills the gap left by every other agreement statistic available here.
  Correlation coefficients and R1 are computed over the *strong* data, and were
  measured to prefer the lowest-symmetry candidate regardless of the truth,
  because a P1 solution fits P1 data by construction -- imposing symmetry can
  only make the fit worse. This asks the opposite question: the crystal says
  these reflections are near-extinct, so a correct phase set must also predict
  them near-extinct, and a wrong space group has no reason to. The weak
  reflections thus decide between candidates even though they contribute
  nothing to the phasing itself.

  Including the systematic absences makes this a graded form of the absence
  test: a group whose predicted absences are observed strong fails here
  continuously rather than only at a threshold.

  Returns None when it cannot be computed rather than a neutral value, so a
  caller cannot silently rank on a number that was never measured.
  """
  from cctbx.array_family import flex

  try:
    fo, fc = f_obs.common_sets(f_calc.as_amplitude_array())
    if fo.size() < 20:
      return None
    fo_e = fo.quasi_normalize_structure_factors()
    fc_e = fc.quasi_normalize_structure_factors()
    n_weak = max(5, int(weak_fraction*fo_e.size()))
    order = fo_e.sort_permutation(by_value="data")  # ascending: weakest first
    weak = order[:n_weak]
    e_c = fc_e.data().select(weak)
    return flex.mean(flex.pow2(e_c))
  except Exception:
    return None


def inversion_agreement(f_calc_in_p1):
  """ The symmetry agreement factor for an inversion centre. Small => centric.

  Centrosymmetry is the one property the rest of the evidence here cannot
  settle. The interatomic difference-vector set is centrosymmetric by
  construction, whether or not the structure is -- that is just the statement
  that a Patterson function always has an inversion centre -- so no
  map-folding argument can decide it. <|E^2-1|> can, from intensity
  statistics, but only about 92% of the time. This is a second and independent
  test of the same question, taken from the phases instead, so the two fail in
  different circumstances and their agreement carries information.

  **The operator cannot simply be applied to the data.** For a data set without
  anomalous signal, F(-h) = conj(F(h)) holds identically by Friedel's law, so
  testing the bare inversion is degenerate -- in cctbx it produces an empty
  common set, because the mapped indices leave the stored hemisphere. What
  distinguishes a centrosymmetric structure is that an inversion centre exists
  *at some position*: the phases can be made real by a suitable origin shift.
  Finding that shift is a translation search, which is exactly what the
  symmetry search already performs for every candidate operator. So the value
  is read from its pool rather than recomputed.

  Inversion is always among the candidates the search tests --
  `possible_point_group_generators` calls `expand_inv` on the lattice group and
  sorts so that it comes first -- so this only reads a number the search has
  already computed and costs nothing beyond the search itself.

  Returns None if it cannot be obtained.
  """
  from cctbx import symmetry_search

  try:
    search = symmetry_search.structure_factor_symmetry(f_calc_in_p1)
  except Exception:
    return None
  inversion = (-1, 0, 0, 0, -1, 0, 0, 0, -1)
  for candidate in getattr(search, "symmetry_pool", None) or []:
    # `possible_symmetry` exposes `.r` (a rot_mx) and `.symmetry_agreement`.
    # Not `.rt`/`.phi_sym` -- those names look plausible and silently yield
    # nothing, which is how this returned None on every structure until the
    # class was actually read.
    try:
      if tuple(candidate.r.num()) == inversion:
        return candidate.symmetry_agreement
    except Exception:
      continue
  return None


def candidate_groups(unit_cell, laue_group_info=None, f_calc_in_p1=None,
                     phi_sym_acceptance_cutoff=0.50):
  """ Space groups worth considering, before any evidence is applied.

  With `laue_group_info` -- which a user normally has, because R_int over
  unmerged symmetry equivalents settles the Laue class before solving and does
  it reliably -- the candidates are every group in that Laue class whose metric
  constraints the cell satisfies.

  Without it, the acceptance-cutoff ladder over `f_calc_in_p1` is used instead.
  That is nearly free: `structure_factor_symmetry` scores every candidate
  operator once and the cutoff only decides which scores count as accepted, so
  a nested family of groups falls out of a single scoring pass, already ordered
  by confidence.
  """
  from cctbx import sgtbx, symmetry_search

  if laue_group_info is not None:
    return _settings_compatible_with(
      unit_cell, laue_number=laue_group_info.type().number())

  if f_calc_in_p1 is None:
    raise ValueError("need either laue_group_info or f_calc_in_p1")

  # Strictest cutoff first, so the list arrives in confidence order. The
  # cctbx default of 0.25 is far too strict for a charge-flipping map -- it
  # recovered the right group for only 18.5% of a 1986-entry sample against
  # 68.9% at 0.50 -- because real operators on a solution map land in the
  # "unsure" band and are discarded.
  numbers = []
  for cutoff in (0.35, phi_sym_acceptance_cutoff, 0.70):
    try:
      r = symmetry_search.structure_factor_symmetry(
        f_calc_in_p1, phi_sym_acceptance_cutoff=cutoff)
    except Exception:
      continue
    sgi = getattr(r, "space_group_info", None)
    if sgi is not None and sgi.type().number() not in numbers:
      numbers.append(sgi.type().number())
  if not numbers:
    return []
  # The search returns its answer in whatever basis and origin it settled on --
  # `P121/c1(a,b-1/4,c+1/4)` is a normal result. Handing that straight to
  # `merge_equivalents` gives nonsense, so only the group *identity* is taken
  # from the search, and the conventional settings compatible with the cell are
  # enumerated from it. The absence test then chooses among them.
  return _settings_compatible_with(unit_cell, numbers=numbers)


def suggest(f_obs, f_calc_in_p1, laue_group_info=None, n_suggestions=3,
            max_absence_ratio=DEFAULT_MAX_ABSENCE_RATIO):
  """ A ranked shortlist of space groups for a P1 solution.

  Returns a group_args with `suggestions` -- a list of group_args, best first,
  each carrying `space_group_info`, `absence_ratio`, `n_predicted_absent`,
  `centric_agrees` and `reason` -- plus the `e_sq_minus_one` statistic and the
  `centric` call derived from it.

  Ranking, in order of how much each is trusted:

  1. **Absences refute.** Any group whose predicted absences are not weak
     its predicted-absent reflections observed strong is dropped outright.
     This is the only hard filter, and the strongest one available: given a
     Laue class the candidate list runs to 19-21 groups and as many as 59, and
     nothing else narrows it comparably.
  2. **Centrosymmetry agreement**, from <|E^2-1|>. Right 92% of the time
     blind, and the only signal here that decides it at all.
  3. **Frequency**, as the final tie-break -- see FREQUENCY_ORDER.

  Correlation peak height and R1 are deliberately absent: both are measured to
  prefer the lowest-symmetry candidate whatever the truth.
  """
  f_obs_p1 = f_obs.expand_to_p1() if f_obs.space_group().order_z() > 1 \
      else f_obs

  stat = e_squared_minus_one(f_obs)
  # Two independent tests of centrosymmetry, because it is the one question
  # nothing else here can settle and a single 92%-accurate classifier is not
  # enough. <|E^2-1|> works from intensity statistics; the agreement factor
  # from the P1 phases. They fail differently, so agreement is worth something
  # and disagreement is worth flagging.
  alpha_0 = inversion_agreement(f_calc_in_p1)
  centric_by_stat = None if stat is None else bool(stat > CENTRIC_THRESHOLD)
  centric_by_alpha = None if alpha_0 is None \
      else bool(alpha_0 < ALPHA_0_CENTRIC_THRESHOLD)
  # **<|E^2-1|> decides; alpha_0 only advises.** Preferring alpha_0 was tried
  # and reverted: on the first structures examined it called two genuinely
  # centrosymmetric ones acentric and, because it was in charge, pushed P2(1)/c
  # off the top of the shortlist for a P2(1)/c structure -- a regression
  # against the statistic alone, which was right there.
  #
  # The threshold is why. ALPHA_0_CENTRIC_THRESHOLD comes from work on phases
  # that have been through dual-space refinement, and charge-flipping phases
  # are noisier, so the same agreement factor sits systematically higher: the
  # two centric structures scored 0.382 and 0.532 against an acentric 0.663.
  # The *ordering* is right and the cut is in the wrong place -- exactly what
  # was found for phi_sym_acceptance_cutoff, where the inherited default of
  # 0.25 recovered 18.5% of space groups and 0.50 recovered 68.9%.
  #
  # So alpha_0 stays supplementary until it is calibrated on this kind of map
  # the same way, against known centricity over a few hundred structures.
  # Until then its disagreement is worth showing to a user and not worth
  # acting on.
  # **alpha_0 decides inside a narrow band around the cut, and nowhere else.**
  #
  # Measured over 2,818 structures with known centricity:
  #
  #   <|E^2-1|> alone            0.894   missed 0.057   invented 0.049
  #   alpha_0 alone              0.794   missed 0.169   invented 0.037
  #   alpha_0 within +-0.02      0.909   missed 0.052   invented 0.039
  #   agree, else assume centric 0.910   missed 0.020   invented 0.071
  #
  # The last two are one structure apart on accuracy and are not equivalent:
  # **inventing a centre destroys a structure, missing one only leaves it in
  # too low a symmetry, which is recoverable.** The band rule makes 0.039
  # against 0.071 -- roughly ninety fewer ruined structures per 2,818 -- so it
  # wins on the error that matters even though it loses on the average.
  # Reading the accuracy column alone would have picked the other one.
  #
  # Every *linear* combination was tried and all twelve scored 0.79-0.82,
  # below the statistic alone. alpha_0 is not a correction to <|E^2-1|>; it is
  # a second opinion that is only worth having where the first is undecided.
  centric = centric_by_stat if centric_by_stat is not None else centric_by_alpha
  if (ALPHA_0_DECIDES_WITHIN
      and centric_by_stat is not None and centric_by_alpha is not None
      and stat is not None
      and abs(stat - CENTRIC_THRESHOLD) <= ALPHA_0_DECIDES_WITHIN):
    centric = centric_by_alpha

  candidates = candidate_groups(
    f_obs.unit_cell(), laue_group_info=laue_group_info,
    f_calc_in_p1=f_calc_in_p1)

  # Built once for the coverage test; None if no sphere could be made, in which
  # case coverage simply never applies and the intensity test stands alone.
  reference = completeness_reference(f_obs_p1)

  # The absences every candidate predicts, which therefore separate none of
  # them. Computed once: it is one boolean pass per candidate over the observed
  # list, against ninety candidates each already doing the same work.
  shared = shared_absences(f_obs_p1, candidates)

  # Decided once per file, not per candidate: see `_discriminating_ranking`.
  discriminating = _discriminating_ranking(f_obs)

  # Phases, shared across candidates: see `point_group_agreement`. Needs Friedel
  # mates, because the factor maps indices under the rotation and half of them
  # leave the stored hemisphere otherwise -- which silently yields nothing
  # rather than an error.
  phase_cache = {}
  f_calc_for_phases = None
  # **Computed whenever asked for, refuted only when the filter is on.** The
  # feature emitter needs this metric on every row so that combinations can be
  # fitted offline, and that must not require re-enabling a refutation which
  # measured -0.62 on COD. Two switches, because recording evidence and acting
  # on it are different decisions.
  if (_point_group_filter() or _point_group_compute()) and f_calc_in_p1 is not None:
    try:
      f_calc_for_phases = f_calc_in_p1.generate_bijvoet_mates()
    except Exception:
      f_calc_for_phases = None

  scored, refuted = [], []
  for sgi in candidates:
    group = sgi.group()
    ratio, n_absent = absence_ratio(f_obs_p1, group)
    own_ratio, n_own = discriminating_absence_ratio(f_obs_p1, group, shared)
    phi_point_group = point_group_agreement(f_calc_for_phases, group,
                                            cache=phase_cache)
    number = sgi.type().number()
    coverage, n_coverage = ((None, 0) if reference is None
                            else absence_coverage(reference, group))
    entry = group_args(
      space_group_info=sgi,
      absence_ratio=ratio,
      n_predicted_absent=n_absent,
      # The same statistic over this group's own absences only -- what it
      # claims that its rivals do not. See `discriminating_absence_ratio`.
      own_absence_ratio=own_ratio,
      n_own_absent=n_own,
      # Phase evidence about the rotations, independent of the absences.
      phi_point_group=phi_point_group,
      own_absence_bucket=(round(own_ratio, _absence_bucket_decimals())
                          if n_own >= MIN_ABSENCES_TO_JUDGE else None),
      coverage_margin=coverage,
      n_coverage_absent=n_coverage,
      # The same quantity on both sides: the fraction of the reflection list
      # this group declares systematically absent. Comparable across the two
      # tests, where the raw counts are not.
      evidence_fraction=(
        n_absent/max(1, f_obs_p1.size()) if n_absent >= MIN_ABSENCES_TO_JUDGE
        else (n_coverage/max(1, reference[0].size())
              if reference is not None and n_coverage else 0.0)),
      # **Which test applies is decided per candidate, not per data set.**
      #
      # The tempting design is a global switch on whether the file looks merged.
      # Per candidate is both simpler and more correct, and the reason is that
      # the two tests fail in complementary places on the *same* file:
      #
      #   On merged data the true group's absences have been deleted, so it has
      #   nothing present to measure and falls to coverage, which finds them
      #   convincingly missing. A wrong group predicting a superset still has
      #   some of its predictions present and strong, so the intensity test
      #   applies to it and refutes it.
      #
      #   On unmerged data the true group's absences were measured and came
      #   back at noise, so the intensity test applies and clears it -- and it
      #   discriminates rather than tying, which coverage cannot do.
      #
      # So each candidate is judged by whichever test has evidence about it,
      # preferring intensity: measured weakness is a stronger statement than
      # inferred absence. Measured on the group's own data the intensity test
      # is judgeable for 0.667 of structures against 0.006 on COD, and never
      # once refuted the true group.
      judged_by=("intensity" if n_absent >= MIN_ABSENCES_TO_JUDGE
                 else "coverage" if (coverage is not None
                                     and n_coverage >= MIN_ABSENCES_TO_JUDGE)
                 else "nothing"),
      centric_agrees=(None if centric is None
                      else bool(group.is_centric()) == centric),
      frequency_rank=(FREQUENCY_ORDER.index(number)
                      if number in FREQUENCY_ORDER else len(FREQUENCY_ORDER)),
      reason=None)
    entry.reason = None
    scored.append(entry)

  # **Refute relative to the best candidate, not against an absolute cut.**
  #
  # Real data always shows some intensity where a space group says there should
  # be none: multiple scattering (Renninger) feeds the strongest absences,
  # neighbouring reflections bleed into them, and detector artefacts do the
  # rest. On a real P2(1)/n structure all three settings of group 14 exceeded
  # an absolute 5% cut and every candidate was "refuted" -- the filter
  # contributed nothing and the fallback carried the whole shortlist.
  #
  # That baseline affects every candidate about equally, so the informative
  # quantity is the *excess* over the cleanest candidate. In the same run the
  # relative ordering was already right: P2(1)/n had the lowest violation
  # fraction and came first. Comparing candidates against each other is also
  # self-calibrating, which an absolute threshold on real data can never be.
  survivors = []
  for entry in scored:
    # Only groups predicting enough absences can be judged on them. A group
    # predicting none cannot violate any, and must not therefore be treated as
    # the cleanest candidate -- that is the whole failure this replaced. On a
    # real P2(1)2(1)2(1) structure, P222 and Pmmm predict zero absences and
    # scored a perfect 0.000, while the true group predicted 48 and had 10
    # observed, so every screw-containing candidate was refuted against groups
    # that had simply said nothing.
    # **Phases refute point groups; absences refute glides and screws.** This
    # is checked before the absence branches because it applies to candidates
    # the absence test cannot judge at all -- Pmmm and P222 predict nothing, so
    # they reach `judged_by == "nothing"` and survive everything above, which is
    # precisely how a P2(1)2(1)2(1) structure came back as Pmmm. The phases
    # have an opinion about those candidates even when the absences do not.
    #
    # **Gated on the switch, not on the value being available.** Separating
    # "compute this metric" from "act on it" was the whole point of the two
    # switches, and this test asked only whether the number existed. So turning
    # on computation -- which the feature emitter does, because the metric
    # belongs in the table -- silently turned the refutation back on, and the
    # refutation is the one that took COD sg_top1 from 0.9314 to 0.3145. Three
    # of fifteen structures in a probe run had the truth refuted here with no
    # absence evidence at all before this was caught.
    if (_point_group_filter()
        and entry.phi_point_group is not None
        and entry.phi_point_group > MAX_POINT_GROUP_DISAGREEMENT):
      entry.reason = (
        "the solution's phases do not obey its rotations (phi_sym %.2f, "
        "refuted above %.2f)"
        % (entry.phi_point_group, MAX_POINT_GROUP_DISAGREEMENT))
      refuted.append(entry)
      continue
    if entry.judged_by == "nothing":
      survivors.append(entry)
      continue
    if entry.judged_by == "intensity":
      if entry.absence_ratio > max_absence_ratio:
        entry.reason = (
          "its %d predicted absences carry %.0f%% of the mean intensity, "
          "so they are not absent"
          % (entry.n_predicted_absent, 100*entry.absence_ratio))
        refuted.append(entry)
      else:
        survivors.append(entry)
      continue
    # Coverage. Refutation here is deliberately weak: the statement "this
    # group's predicted absences are present in the file" is only damning when
    # the file is one that would have deleted them, and we cannot be certain it
    # is. So a candidate is dropped only when its predicted class is barely
    # more missing than the data is incomplete generally -- which is the case
    # where the group explains nothing at all.
    if entry.coverage_margin is not None \
       and entry.coverage_margin < MIN_COVERAGE_MARGIN:
      entry.reason = (
        "only %.0f%% of its %d predicted absences are missing, no more than "
        "the data is incomplete generally"
        % (100*entry.coverage_margin, entry.n_coverage_absent))
      # **Inferred evidence demotes; it does not kill.**
      #
      # The comment above says this test is uncertain -- "absent from the file"
      # only means something if the file is one that deletes absences, and that
      # is inferred rather than known -- and then removes the candidate anyway.
      # Removal is final: the fallback that restores a refuted list fires only
      # when *every* candidate is refuted, so a truth refuted here while one
      # wrong candidate survives is gone, and `truth_in_list` caps every
      # ordering built on top.
      #
      # Measured on a sample of the feature table, the truth was being refuted
      # with an absence ratio of **0.000** -- perfectly weak absences, so the
      # intensity test had no complaint and this test did it. A candidate whose
      # own measured evidence is spotless should not be discarded on an
      # inference about the file.
      #
      # So it is demoted to the back instead: the ordering benefit is kept
      # entirely, because survivors still rank above it, and the truth can be
      # mis-ranked but never lost.
      if _coverage_refutes():
        refuted.append(entry)
      else:
        entry.coverage_demoted = True
        survivors.append(entry)
    else:
      survivors.append(entry)
  scored = survivors

  # Everything refuted means the filter is wrong for this data -- twinning,
  # an incommensurate component, a bad cell -- so fall back to the unfiltered
  # list rather than returning nothing. Silently returning an empty list would
  # look like "no symmetry" when it means "this test does not apply".
  # A guard, not an expected path: the cleanest candidate has zero excess by
  # construction and so always survives the relative test above. This only
  # fires if something upstream produced no usable candidate at all, and
  # returning a shortlist is still better than returning nothing.
  if not scored and refuted:
    for entry in refuted:
      entry.reason = "no candidate is clean; showing the least bad"
    scored = refuted

  # When the two centrosymmetry tests disagree, stop using centricity to rank
  # at all and let the candidates compete on absences and frequency alone.
  #
  # Measured over 345 structures: the tests agree on 80.6% of them and are
  # 96.4% right there, but on the 19.4% where they disagree the better of them
  # is right 56.7% of the time -- a coin flip. Ranking on a coin flip pushes
  # the right answer off a three-entry shortlist for no reason, whereas leaving
  # centricity out lets both a centric and an acentric candidate appear, which
  # is exactly what a shortlist is for.
  # **Absences that are genuinely absent are positive evidence, not merely the
  # absence of a complaint.** P2(1)2(1)2(1) predicts 48 reflections missing and
  # 38 of them are indeed weak; P222 predicts none and explains nothing, yet
  # fits the data equally well. A crystallographer reads the absence table and
  # picks the group that accounts for the missing reflections, and that is what
  # `n_explained` encodes.
  #
  # This ranks above the frequency prior on purpose, and it is the one place
  # where measurements from the COD corpus do not transfer: COD stores merged
  # data with absences already removed, so absence evidence was nearly useless
  # there and frequency won by default. On real data the absences are present
  # and carry far more information than a population prior.
  for entry in scored:
    # A group only gets credit for absences it actually explains: many
    # predicted absences that really are weak -- or, on merged data, many
    # predicted absences that really are gone. A group predicting none, or one
    # whose predictions are neither weak nor missing, explains nothing.
    entry.has_absence_evidence = (
      (entry.judged_by == "intensity"
       and entry.absence_ratio <= max_absence_ratio)
      or (entry.judged_by == "coverage"
          and entry.coverage_margin is not None
          and entry.coverage_margin >= MIN_COVERAGE_MARGIN))
    # Rounded, because ratios this close are not distinguishable: the true
    # P2(1)/n of a real sample scored 0.0114 against 0.0113 for P2/n, and
    # deciding between them on the fourth decimal would be deciding on noise.
    # Within a bucket the frequency prior breaks the tie, which is what it is
    # good at.
    #
    # Coverage-judged candidates get a bucket of 0.0 -- the best possible --
    # because a margin near 1 means the class really is gone. They are not
    # ranked *ahead* of intensity-judged ones: measured weakness and inferred
    # absence are different strengths of claim, and the coverage side ties
    # heavily anyway (a mean of 8.4 candidates share the top margin), so the
    # separation it cannot provide is left to centricity and frequency below.
    if not entry.has_absence_evidence:
      entry.absence_bucket = 0.0
    elif entry.judged_by == "intensity":
      # **One decimal, not two.** Two decimals separated 0.004 from 0.006 and
      # that is not a distinction the data supports: real absences always carry
      # some intensity from multiple scattering and glide leakage, and how much
      # depends on the reflection class rather than on whether the group is
      # right.
      #
      # The cost of the finer bucket was measured. C2/c predicts strictly more
      # absences than C2/m -- the c-glide class on top of the C-centring -- so
      # it explains everything C2/m explains and more, and should win. It lost
      # three times on real data because its glide class leaked to 0.01 while
      # C2/m's pure centring class sat at 0.00, and the bucket is consulted
      # before the count. Glide-versus-mirror was five of the ten failures.
      #
      # A coarser bucket still separates what it was introduced to separate: a
      # centred group at 0.22 lands in 0.2 against a true group at 0.011 in
      # 0.0, which was the case that motivated bucketing at all.
      entry.absence_bucket = round(entry.absence_ratio,
                                   _absence_bucket_decimals())
    else:
      entry.absence_bucket = 0.0

  disputed =(centric_by_stat is not None and centric_by_alpha is not None
              and centric_by_stat != centric_by_alpha)
  # Order: candidates with real absence evidence first, then by *how
  # convincingly* absent rather than by how many absences were predicted. A
  # centred group predicts half the data absent and, on one real sample, slid
  # under the refutation threshold at a ratio of 0.22 -- not absent at all --
  # yet its raw count of 1931 buried the true group's 78 at a ratio of 0.011.
  # Counting rewards the wrong thing; the ratio is the evidence.
  def order(e):
    # Candidates the coverage test would once have removed are demoted to
    # the back rather than dropped, so the ordering benefit is unchanged
    # and the truth can be mis-ranked but never lost.
    demoted = 1 if getattr(e, "coverage_demoted", False) else 0
    # Evidence before prior, and *how convincingly* absent before *how many*.
    #
    # The bucket separates candidates whose absences are genuinely weak from
    # those that merely scraped under the threshold -- a centred group with
    # 1931 absences at 0.22 loses to the true group's 78 at 0.011 because they
    # land in different buckets.
    #
    # Within a bucket the count decides, and it must come before frequency: on
    # one real sample Pna2(1) explained 626 absences at 0.0075 while
    # P2(1)2(1)2(1) explained 64 at 0.0095 -- better on both counts -- yet the
    # frequency prior preferred P2(1)2(1)2(1) because it is the commoner group.
    # A prior should not overrule ten times the evidence.
    # The count must come from whichever test judged the candidate. A
    # coverage-judged group has `n_predicted_absent` of zero by construction --
    # that is *why* it fell to coverage -- so reading that field for it would
    # sort every merged-data candidate to the back on the one key meant to
    # reward explanatory power.
    # **A fraction, not a count.** The two tests count against different
    # denominators -- intensity over the *observed list*, coverage over the
    # whole *complete sphere*, which is tens of thousands -- so comparing raw
    # counts let a coverage-judged candidate win every tie-break on units
    # alone. On real data that cost the true C2/c to a C2/m judged by coverage
    # with zero absences present. The fraction of the list a group declares
    # absent is dimensionless and means the same thing on both sides.
    #
    # **Ranking intensity above coverage was tried here and refuted.** It is
    # the intuitive fix -- measured weakness is a stronger claim than inferred
    # absence -- and on 395 paired COD entries it gained 0 and lost 7
    # (ok1 -0.018). The reason is structural: on merged data the *true* group
    # is the one whose absences were deleted, so it is exactly the candidate
    # that falls to coverage, and a blanket demotion targets it. The unit bug
    # and the precedence were two separate claims and only the first survived.
    #
    # **The discriminating set decides first.** Both of the arguments above are
    # about how to compare two candidates' *whole* absence sets, and that is
    # the wrong comparison: candidates from one Laue class share their lattice
    # centring and differ only in screws and glides, so the whole-set ratio is
    # dominated by evidence that applies equally to both. On `DB_0148_1`, with
    # deposited phases, C2/c's c-glide class was diluted about eighty to one by
    # centring absences it shares with C2/m, and the truth ranked sixth.
    #
    # Ordering on `own_absence_bucket` first asks the question a
    # crystallographer asks -- is *this group's own* claimed class weak? -- and
    # a group whose predictions are entirely shared has no such class and so
    # makes no claim. That is the principle `absence_ratio` already states,
    # applied against the rivals rather than against the file, and it does not
    # demote coverage-judged candidates as a class, which is what the refuted
    # experiment above did wrong.
    #
    # `None` sorts last: no discriminating evidence is not a good score.
    #
    # **And coverage outranks the frequency prior.** On `DB_0148_1` the true
    # C2/c has 29,348 of its predicted-absent reflections missing from the file
    # -- a coverage margin of 0.4818 against a baseline -- while P2(1)/c, which
    # predicts almost nothing extra missing, sits at 0.0056. Nearly a hundred
    # times the evidence, and it decided nothing: the two tied on every key
    # that was consulted and P2(1)/c won on `frequency_rank` alone, because it
    # is the commoner group. That is the failure this file already names
    # elsewhere -- "A prior should not overrule ten times the evidence" --
    # arriving through a different door.
    #
    # Bucketed to two decimals so that noise-level differences in the margin do
    # not order candidates; 0.48 against 0.01 is not a close call, and two
    # candidates genuinely tied on coverage should still fall through to the
    # prior.
    if discriminating:
      margin = (e.coverage_margin
                if (e.coverage_margin is not None
                    and e.n_coverage_absent >= MIN_ABSENCES_TO_JUDGE) else 0.0)
      return (demoted,
              0 if e.own_absence_bucket is not None else 1,
              e.own_absence_bucket if e.own_absence_bucket is not None else 0.0,
              0 if e.has_absence_evidence else 1,
              e.absence_bucket,
              -round(max(0.0, margin), 2),
              -e.evidence_fraction,
              e.frequency_rank)
    return (demoted,
            0 if e.has_absence_evidence else 1,
            e.absence_bucket,
            -e.evidence_fraction,
            e.frequency_rank)

  if disputed:
    scored.sort(key=order)
  else:
    scored.sort(key=lambda e: ((0 if e.centric_agrees is not False else 1),)
                + order(e))

  if PHI_RERANK and len(scored) > 1:
    _phi = [getattr(e, "phi_point_group", None) for e in scored]
    # Every candidate needs the number or the contest is between different
    # measurements. A partial column would let a candidate win on having been
    # skipped, which is how absence turns into agreement.
    if all(p is not None for p in _phi):
      _by_phi = sorted(range(len(scored)), key=lambda i: -_phi[i])
      _phi_rank = [0]*len(scored)
      for _pos, _i in enumerate(_by_phi):
        _phi_rank[_i] = _pos
      _sorted_phi = sorted(_phi, reverse=True)
      _w = min((_sorted_phi[0] - _sorted_phi[1])/PHI_MARGIN_SCALE, 4.0)
      _fused = sorted(range(len(scored)),
                      key=lambda i: (i + _w*_phi_rank[i], i))
      scored = [scored[i] for i in _fused]

  # **A run-off among the leaders, decided on the reflections that separate
  # them.** Everything above ranks candidates against the file; this ranks them
  # against each other, which is the question when two groups differ by a single
  # glide. Confined to the leading few because it is O(k^2) absence tests and
  # because a candidate the file-level evidence has already put tenth is not in
  # contention. Stable: `sorted` keeps the existing order among equal scores, so
  # a contest that says nothing changes nothing.
  if discriminating and len(scored) > 1:
    leaders = scored[:PAIRWISE_RUNOFF_SIZE]
    wins = copeland_scores(f_obs_p1, leaders, max_ratio=max_absence_ratio)
    order_of = dict((id(e), i) for i, e in enumerate(leaders))
    leaders = sorted(leaders,
                     key=lambda e: (-wins[order_of[id(e)]], order_of[id(e)]))
    scored = leaders + scored[PAIRWISE_RUNOFF_SIZE:]

  # **Alpha fills the shortlist, it does not lead it.**
  #
  # Origin-searched symmetry agreement is a good shortlist builder and a bad
  # top-1 ranker, and that is true on both corpora rather than being a quirk of
  # one. Ranking by it alone scores top1 0.368 on COD against the module's
  # 0.895, while its top3 reaches 0.924 on measured data against the module's
  # 0.746. Rank fusion was tried across the whole sweep and **no weighting
  # improves one corpus without regressing the other**.
  #
  # So the module's leading two are left exactly where the evidence put them --
  # which makes a top-1 regression impossible by construction -- and alpha
  # decides only the remaining shortlist places, the ones the frequency prior
  # would otherwise fill. Measured:
  #
  #                     COD top3          measured top3
  #     module alone    0.9726            0.7458
  #     alpha fills     0.9775  (+0.005)  0.8729  (+0.127)
  #
  # and it held on three random held-out halves of both corpora, while keeping
  # only *one* leader regressed COD every time. Nothing is deleted: candidates
  # alpha dislikes move down the list, never off it.
  if _alpha_shortlist() and f_calc_in_p1 is not None and len(scored) > ALPHA_KEEP_LEADERS:
    try:
      from smtbx.ab_initio import symmetry_agreement
      rho = symmetry_agreement._map_array(f_calc_in_p1)
      window = scored[ALPHA_KEEP_LEADERS:ALPHA_KEEP_LEADERS + ALPHA_WINDOW]
      for e in window:
        try:
          e.alpha = symmetry_agreement.agreement(
            None, e.space_group_info.group(), rho=rho)
        except Exception:
          e.alpha = None
      # Stable, and a candidate with no alpha keeps its existing place rather
      # than being sorted to the front by a missing measurement.
      window = sorted(window,
                      key=lambda e: -(e.alpha if e.alpha is not None else -1.0))
      scored = (scored[:ALPHA_KEEP_LEADERS] + window
                + scored[ALPHA_KEEP_LEADERS + ALPHA_WINDOW:])
    except Exception:
      pass

  for entry in scored:
    if entry.reason is None:
      bits = []
      # Say which test spoke. A user comparing two suggestions needs to know
      # whether "the absences check out" means they were measured and are weak
      # or merely that they are not in the file.
      if entry.judged_by == "intensity" and entry.n_predicted_absent:
        bits.append("%d predicted absences, at %.0f%% of the mean intensity"
                    % (entry.n_predicted_absent, 100*entry.absence_ratio))
      elif entry.judged_by == "coverage" and entry.coverage_margin is not None:
        bits.append("%d predicted absences, %.0f%% of them missing from the "
                    "data (merged file, so they cannot be measured)"
                    % (entry.n_coverage_absent, 100*entry.coverage_margin))
      if entry.centric_agrees:
        bits.append("centrosymmetry agrees with <|E^2-1|>")
      entry.reason = "; ".join(bits) or "no evidence against it"

  return group_args(
    suggestions=scored[:n_suggestions],
    all_candidates=scored,
    refuted=refuted,
    e_sq_minus_one=stat,
    alpha_0=alpha_0,
    centric=centric,
    centric_by_stat=centric_by_stat,
    centric_by_alpha=centric_by_alpha,
    # True when the two independent centrosymmetry tests disagree. Worth
    # surfacing rather than hiding: it is precisely the situation where the
    # P1/P-1 choice is genuinely uncertain and the user should look at both.
    centric_disputed=(centric_by_stat is not None
                      and centric_by_alpha is not None
                      and centric_by_stat != centric_by_alpha))


def show(result, out=None):
  """ The shortlist, as a user would read it. """
  import sys
  if out is None:
    out = sys.stdout
  bits = []
  if result.e_sq_minus_one is not None:
    bits.append("<|E^2-1|> = %.3f" % result.e_sq_minus_one)
  if result.alpha_0 is not None:
    bits.append("alpha_0 = %.3f" % result.alpha_0)
  if bits:
    print("%s: %s" % (", ".join(bits),
                      "centrosymmetric" if result.centric
                      else "non-centrosymmetric"), file=out)
  if getattr(result, "centric_disputed", False):
    print("  (the two tests disagree -- treat the P1/P-1 choice as open)",
          file=out)
  for i, s in enumerate(result.suggestions):
    print("%d. %-12s %s" % (i + 1, str(s.space_group_info), s.reason),
          file=out)
  if not result.suggestions:
    print("No space group could be suggested.", file=out)
