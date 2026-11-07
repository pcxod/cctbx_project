""" Space-group suggestions from a P1 charge-flipping solution.

Produces a short ranked list of candidate space groups rather than one answer,
since a wrong assumed group is the expensive failure of solving in P1.

Fit-based figures of merit cannot rank space groups: the solution is a P1
solution, so imposing symmetry can only degrade the fit, and correlation peak
height and R1 both prefer the lowest symmetry. They are not used for ranking.

Each evidence source is used only for what it can decide:

  systematic absences  screws and glides, and a hard filter: a group
                       predicting absent reflections that are observed
                       strong is refuted
  <|E^2-1|> statistics centrosymmetry, which map folding cannot decide
  cell metric          which groups are geometrically possible
  frequency prior      tie-break
"""
from __future__ import absolute_import, division, print_function

from libtbx import group_args

# Space-group frequency in the COD, most common first, by International Tables
# number; groups never seen close the list in number order. Every group is
# listed, since number order puts the rare pure-rotation groups first (P622
# above P-62m). Only a tie-break: a prior this concentrated names P2(1)/c for
# structures that are not, and the unusual ones are where a user needs help.
FREQUENCY_ORDER = (14, 2, 15, 19, 4, 61, 33, 9, 62, 29, 60, 5, 13, 148, 11,
                   1, 12, 7, 18, 56, 43, 88, 92, 36, 96, 167, 20, 82, 64,
                   146, 41, 52, 63, 57, 161, 86, 155, 114, 31, 70, 76, 170,
                   78, 194, 176, 225, 198, 152, 169, 45, 166, 58, 145, 154,
                   144, 142, 53, 147, 173, 205, 85, 122, 8, 72, 73, 54, 110,
                   55, 186, 87, 165, 163, 121, 179, 220, 131, 74, 59, 98,
                   160, 136, 159, 71, 191, 65, 178, 227, 139, 79, 34, 113,
                   68, 26, 221, 192, 190, 140, 97, 81, 106, 150, 80, 216,
                   143, 129, 217, 130, 10, 32, 46, 164, 69, 118, 3, 218, 23,
                   128, 141, 182, 196, 212, 193, 120, 200, 37, 126, 66, 189,
                   40, 197, 39, 229, 206, 77, 21, 104, 51, 213, 204, 223,
                   135, 38, 158, 30, 42, 117, 6, 171, 137, 157, 123, 17, 100,
                   47, 175, 132, 174, 219, 95, 185, 91, 138, 222, 94, 90, 28,
                   162, 202, 228, 103, 187, 180, 50, 215, 24, 27, 102, 119,
                   109, 124, 134, 116, 108, 211, 181, 224, 127, 177, 151,
                   207, 172, 75, 230, 226, 195, 48, 22, 67, 84, 44, 168, 210,
                   107, 89, 203, 153, 133, 83, 209, 125, 16, 25, 35, 49, 93,
                   99, 101, 105, 111, 112, 115, 149, 156, 183, 184, 188, 199,
                   201, 208, 214)

# <|E^2-1|> is 0.968 centric and 0.736 acentric; the midpoint
CENTRIC_THRESHOLD = 0.858

# Symmetry agreement factor for an inversion centre below which the structure
# counts as centrosymmetric (the factor is ~1 for random phases). Not a
# calibrated classifier: alpha_0 feeds `disputed`, which suppresses centricity
# as a ranking key, and a better-calibrated cut (0.60) admitted centricity
# ahead of the absence evidence and ranked worse end to end.
ALPHA_0_CENTRIC_THRESHOLD = 0.30

# How close <|E^2-1|> must be to its own cut before alpha_0 may overrule it;
# 0 = off. To be judged on the end-to-end ranking, not centricity accuracy.
ALPHA_0_DECIDES_WITHIN = 0.0

# Fewest predicted absences before the absence test has an opinion: a 2(1)
# screw predicts only a few dozen axial reflections, P222 none.
MIN_ABSENCES_TO_JUDGE = 10

# Below that minimum a class can still refute: axial reflections stronger than
# half the mean are not absent, else frequency alone ranks P2(1) over P2.
STRONG_ABSENCE_MIN = 3
STRONG_ABSENCE_RATIO = 0.5
# merged-file mirror: a short axial class missing to the last reflection is a
# screw axis, not a small sample
COVERAGE_SURE_MARGIN = 0.9

# Where <|E^2-1|> is a coin flip; inside it the shortlist keeps one slot for
# the other centricity.
CENTRIC_AMBIGUOUS = (0.78, 0.92)

# Below this coverage margin a group's predicted absences are no more missing
# than the file is incomplete anyway. Low, because absent-from-file is only
# evidence if the file deletes absences, which is inferred.
MIN_COVERAGE_MARGIN = 0.05
# A deleted class outranks a measured one only when it is much larger and the
# measured class is genuinely weak.
COVERAGE_DOMINANCE_FACTOR = 6
COVERAGE_DOMINANCE_MAX_RATIO = 0.15

# Decimals the absence ratio is rounded to before candidates are compared on
# it, before the absence count. One decimal would let C2/c (glide leaking to
# 0.01) beat C2/m but ranks worse overall. Env override for A/B only.
ABSENCE_BUCKET_DECIMALS = 2


def _absence_bucket_decimals():
  import os

  try:
    return int(os.environ.get("SMTBX_ABSENCE_BUCKET_DECIMALS",
                              ABSENCE_BUCKET_DECIMALS))
  except ValueError:
    return ABSENCE_BUCKET_DECIMALS

# Refute a group when its predicted absences carry more than this fraction of
# the mean intensity of everything else; true absent classes sit near 0.05-0.1,
# absent-but-predicted ones near 0.9.
DEFAULT_MAX_ABSENCE_RATIO = 0.25

# Leading candidates entering the pairwise run-off, which costs O(k^2) absence
# tests.
PAIRWISE_RUNOFF_SIZE = 12


def e_squared_minus_one(f_obs):
  """ <|E^2-1|>, normalised in P1 and rescaled to <E^2> = 1, so the statistic
  does not depend on the group it is used to choose. """
  from cctbx import crystal, sgtbx
  from cctbx.array_family import flex

  try:
    p1 = f_obs.customized_copy(
      crystal_symmetry=crystal.symmetry(
        unit_cell=f_obs.unit_cell(),
        space_group_info=sgtbx.space_group_info("P 1")))
    # negative amplitudes trip an assert in the normalisation
    data = p1.data().deep_copy()
    data.set_selected(data < 0, 0.0)
    p1 = p1.customized_copy(data=data)
    p1.setup_binner_counting_sorted(reflections_per_bin=200)
    try:
      norm = p1.amplitude_quasi_normalisations()
    except AssertionError:
      # the interpolation between bin means goes non-positive on small data
      # sets; the plain bin mean never fails
      eps = p1.epsilons().data().as_double()
      n = flex.double(p1.size(), 0.0)
      for i in p1.binner().range_used():
        bsel = p1.binner().selection(i)
        if bsel.count(True):
          n.set_selected(bsel, flex.mean(
            flex.pow2(p1.data().select(bsel)) / eps.select(bsel)))
      norm = p1.customized_copy(data=flex.sqrt(n))
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

  Near zero the class is absent; near one the group is refuted.

  Relative strength, not significance: on good data multiple scattering and
  glide leakage put much of a truly absent class above I/sigma = 3, while the
  class is still weak compared to the rest. Intensities, because squaring
  separates the populations further.

  Only refutation is claimed. A merged file has already lost the true group's
  absences, so the ratio is over reflections present, and a group predicting
  none of them gets no credit.
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


PSEUDO_CENTRING_SHARE = 0.10

def pseudo_centring(f_obs_p1, symbol):
  """ Share of the reflections a lattice centring forbids that are in the file
  at I/sigma > 3, or 0 when there are under MIN_ABSENCES_TO_JUDGE of them or
  no sigmas. Significance, unlike `absence_ratio`, because pseudo-centring
  leaves the class weak on average but widely observed. """
  from cctbx import sgtbx
  from cctbx.array_family import flex
  if symbol == "P" or f_obs_p1.sigmas() is None:
    return 0.0
  try:
    absent = flex.bool(sgtbx.space_group("%s 1" % symbol).is_sys_absent(
      f_obs_p1.indices()))
  except Exception:
    return 0.0
  if absent.count(True) < MIN_ABSENCES_TO_JUDGE:
    return 0.0
  d, s = f_obs_p1.data().select(absent), f_obs_p1.sigmas().select(absent)
  # I/sigma(I) = F/(2 sigma(F))
  lim = 3.0 if f_obs_p1.is_xray_intensity_array() else 6.0
  return (d > lim*s).count(True)/absent.count(True)


# Observations per unique reflection above which a file is taken to have
# measured its absences rather than merged them away.
UNMERGED_REDUNDANCY = 1.5


def _discriminating_ranking(f_obs=None):
  """ Whether the discriminating-absence rank applies to *this file*.

  On merged data the true group's absences were deleted, so its
  discriminating set is empty and ranking on what a group uniquely claims
  demotes the right answer. On unmerged data the true group's absences were
  measured at noise level, so it is the one making a claim. Hence decided per
  file from its redundancy.

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
    # cannot tell: keep the merged-data behaviour
    return False


# Worst phase disagreement a candidate's rotations may show before it is
# refuted; cctbx's `structure_factor_symmetry` accepts an operator at 0.25 and
# rejects at 0.5.
MAX_POINT_GROUP_DISAGREEMENT = 0.5

# The same statistic as a weight instead of a veto (the veto refutes the true
# group too often, hence off). The weight is the candidate's phi margin over
# the runner-up scaled by PHI_MARGIN_SCALE, so a phi that cannot separate the
# leaders contributes nothing and a sharp one can outvote the absence order.
# Applied before the discriminating run-off, which keeps the last word among
# the leaders.
import os
PHI_RERANK = os.environ.get("SG_PHI_RERANK", "0") != "0"
PHI_MARGIN_SCALE = float(os.environ.get("SG_PHI_MARGIN_SCALE", "0.16"))


# leading candidates alpha may not reorder
ALPHA_KEEP_LEADERS = 2

# Candidates below the leaders alpha is computed for; each costs an FFT pair
# per symmetry operator, and the ranking is flat in this window.
ALPHA_WINDOW = 6


def _alpha_shortlist():
  """ Whether alpha orders the shortlist below the module's leaders.

  Off by default; `SMTBX_ALPHA_SHORTLIST=1` enables it.
  """
  import os
  return os.environ.get("SMTBX_ALPHA_SHORTLIST", "0") not in ("", "0", "false",
                                                              "False")


def _coverage_refutes():
  """ Whether the coverage test may remove a candidate outright.

  Off by default: it demotes instead. `SMTBX_COVERAGE_REFUTES=1` restores
  refutation.
  """
  import os

  return os.environ.get("SMTBX_COVERAGE_REFUTES", "0") == "1"


def _point_group_compute():
  """ Compute the phase agreement without letting it refute anything. """
  import os

  return os.environ.get("SMTBX_POINT_GROUP_COMPUTE", "0") == "1"


def _point_group_filter():
  """ Whether the phase-based point-group refutation is on. Off by default.

  On merged data it refutes the true group in most structures, for reasons
  not yet understood. `SMTBX_POINT_GROUP_PHASES=1` re-enables it.
  """
  import os

  return os.environ.get("SMTBX_POINT_GROUP_PHASES", "0") == "1"


def point_group_agreement(f_calc_in_p1, space_group, cache=None):
  """ Worst phase disagreement over a candidate's distinct rotations.

  Whether the phases of the P1 solution satisfy the candidate's rotations
  over every reflection (SHELXT's alpha, Sheldrick 2015).

  Rotations only: a c-glide and a mirror share a rotation and score alike, so
  glide versus mirror is left to the absence test.

  `cache` maps a rotation to its factor across candidates; the candidates of
  a Laue class share few distinct rotations.

  None when it cannot be computed, treated as no evidence.
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

  Pairwise, because two candidates of one Laue class differ by a specific set
  of reflections (an intersection over all candidates is empty once P1 is in
  the list):

      A_only = absent(A) and not absent(B)
      B_only = absent(B) and not absent(A)

  A strong claim refutes the group making it. When only one side claims
  anything (glide versus mirror), that claim decides.

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
  MIN_ABSENCES_TO_JUDGE = 5   # a disputed class is small: 0k0 has nine reflections
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
  # a claim that is present and strong refutes the group making it
  if wa is not None and wa > max_ratio:
    return -1
  if wb is not None and wb > max_ratio:
    return +1
  # only one side claims anything: prefer the group that explains more
  if wa is not None and wb is None:
    return +1
  if wb is not None and wa is None:
    return -1
  return 0


def copeland_scores(f_obs_p1, entries, max_ratio=None):
  """ How many pairwise contests each candidate wins. Order by this.

  `pairwise_absence_winner` need not be transitive, and a non-transitive
  comparator makes `sort` depend on input order; a win count (Copeland
  score) is well defined either way.
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

  Candidates of one Laue class have nested absence sets: C2/c predicts the
  C-centring class and the c-glide class, C2/m only the centring. Scoring over
  the whole set buries the few deciding reflections under the thousands both
  share.

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

  Restricted to the reflections this group declares absent and its rivals do
  not. A group whose predictions are entirely shared gets `(0.0, 0)`: no
  claim.
  """
  from cctbx.array_family import flex

  indices = f_obs_p1.indices()
  absent = flex.bool(space_group.is_sys_absent(indices))
  own = absent & ~shared
  n_own = own.count(True)
  if n_own == 0:
    return 0.0, 0

  intensities = flex.pow2(f_obs_p1.data())
  # against everything this group does not call absent; the shared absences in
  # the denominator would make every ratio look small
  rest = intensities.select(~absent)
  if rest.size() == 0:
    return 0.0, n_own
  mean_rest = flex.mean(rest)
  if mean_rest <= 0:
    return 0.0, n_own
  return flex.mean(intensities.select(own))/mean_rest, n_own


def completeness_reference(f_obs_p1):
  """ The reflection list this cell *could* have, and which of it is present.

  Built once and reused for every candidate.

  Returns (complete indices, present flags, baseline missing fraction), or None
  if no usable sphere can be built.
  """
  import os

  from cctbx import miller
  from cctbx.array_family import flex

  # A/B switch: None disables the coverage test, candidates fall back to the
  # intensity test
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

  The test for merged data, where deposition deleted the absent reflections
  and `absence_ratio` has nothing to work with. Measured as the excess over
  the file's general incompleteness, since a file missing much of everything
  makes any group look good on raw coverage.

  A filter, not a discriminator: many groups tie at the same margin, but it
  almost never eliminates the truth, so it is applied first.
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

  Settings, not the 230 numbers: P2(1)/n and P2(1)/c are both number 14 but
  differ in which reflections are absent, and P2(1)/n is common. The absence
  test chooses between settings.
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
  smallest observed normalised structure factors, including systematically
  absent ones.

  CC and R1 over the strong data prefer the lowest symmetry, since a P1
  solution fits P1 data by construction. A correct phase set must also
  predict the near-extinct reflections near-extinct, which a wrong group has
  no reason to. With the absences included this is a graded absence test.

  None when it cannot be computed, never a neutral value.
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

  A phase-based test of centrosymmetry, independent of <|E^2-1|>; the
  Patterson is centrosymmetric regardless, so map folding cannot decide it.

  The bare inversion is degenerate by Friedel's law (in cctbx an empty common
  set); what matters is an inversion centre at some origin, a translation
  search that `structure_factor_symmetry` already runs. Inversion is always in
  its pool (`expand_inv` on the lattice group), so the value is read from
  there.

  Returns None if it cannot be obtained.
  """
  from cctbx import symmetry_search

  try:
    search = symmetry_search.structure_factor_symmetry(f_calc_in_p1)
  except Exception:
    return None
  inversion = (-1, 0, 0, 0, -1, 0, 0, 0, -1)
  for candidate in getattr(search, "symmetry_pool", None) or []:
    # `possible_symmetry` exposes `.r` and `.symmetry_agreement`, not
    # `.rt`/`.phi_sym`; the wrong names fail silently here
    try:
      if tuple(candidate.r.num()) == inversion:
        return candidate.symmetry_agreement
    except Exception:
      continue
  return None


def candidate_groups(unit_cell, laue_group_info=None, f_calc_in_p1=None,
                     phi_sym_acceptance_cutoff=0.50):
  """ Space groups worth considering, before any evidence is applied.

  With `laue_group_info` (normally known from R_int): every group of that
  Laue class whose metric the cell satisfies. Without it: the acceptance
  cutoff ladder over `f_calc_in_p1`, a nested family of groups from one
  `structure_factor_symmetry` scoring pass.
  """
  from cctbx import sgtbx, symmetry_search

  if laue_group_info is not None:
    return _settings_compatible_with(
      unit_cell, laue_number=laue_group_info.type().number())

  if f_calc_in_p1 is None:
    raise ValueError("need either laue_group_info or f_calc_in_p1")

  # Strictest cutoff first, so the list arrives in confidence order. The
  # cctbx default of 0.25 is too strict for a charge-flipping map: real
  # operators land in the "unsure" band and are discarded.
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
  # The search answers in whatever basis and origin it settled on (e.g.
  # P121/c1(a,b-1/4,c+1/4)), which merge_equivalents cannot use, so only the
  # group identity is taken and the conventional settings compatible with the
  # cell are enumerated; the absence test chooses among them.
  return _settings_compatible_with(unit_cell, numbers=numbers)


def suggest(f_obs, f_calc_in_p1, laue_group_info=None, n_suggestions=3,
            max_absence_ratio=DEFAULT_MAX_ABSENCE_RATIO, ensure_centric=False):
  """ A ranked shortlist of space groups for a P1 solution.

  The shortlist holds one setting per group: a second setting of the same
  group with the same absences (Cc and Cn in a C-centred cell) only wastes a
  slot, and the absence test has already ordered them. With `ensure_centric`,
  a shortlist without a centrosymmetric group gets the best-ranked one in its
  last slot -- for heavy-atom structures, whose <|E^2-1|> reads acentric.

  Returns a group_args with `suggestions` -- a list of group_args, best first,
  each carrying `space_group_info`, `absence_ratio`, `n_predicted_absent`,
  `centric_agrees` and `reason` -- plus the `e_sq_minus_one` statistic and the
  `centric` call derived from it.

  Ranking, most trusted first: absences (a group whose predicted-absent
  reflections are observed strong is refuted; the only test that narrows
  the candidate list substantially), centrosymmetry agreement from
  <|E^2-1|>, then frequency (FREQUENCY_ORDER) as tie-break. Correlation peak
  height and R1 are not used: both prefer the lowest-symmetry candidate
  whatever the truth.
  """
  f_obs_p1 = f_obs.expand_to_p1() if f_obs.space_group().order_z() > 1 \
      else f_obs

  stat = e_squared_minus_one(f_obs)
  # Two independent centrosymmetry tests that fail differently: <|E^2-1|> from
  # intensity statistics, the inversion agreement alpha_0 from the P1 phases.
  alpha_0 = inversion_agreement(f_calc_in_p1)
  centric_by_stat = None if stat is None else bool(stat > CENTRIC_THRESHOLD)
  centric_by_alpha = None if alpha_0 is None \
      else bool(alpha_0 < ALPHA_0_CENTRIC_THRESHOLD)
  # <|E^2-1|> decides; alpha_0 decides only within ALPHA_0_DECIDES_WITHIN of
  # CENTRIC_THRESHOLD, where the statistic is undecided. On charge-flipping
  # phases alpha_0 sits higher than ALPHA_0_CENTRIC_THRESHOLD assumes, so alone
  # (or in any linear combination) it is worse than the statistic. "Assume
  # centric when they disagree" invents more centres, and an invented centre
  # ruins a structure while a missed one is recoverable.
  centric = centric_by_stat if centric_by_stat is not None else centric_by_alpha
  if (ALPHA_0_DECIDES_WITHIN
      and centric_by_stat is not None and centric_by_alpha is not None
      and stat is not None
      and abs(stat - CENTRIC_THRESHOLD) <= ALPHA_0_DECIDES_WITHIN):
    centric = centric_by_alpha

  candidates = candidate_groups(
    f_obs.unit_cell(), laue_group_info=laue_group_info,
    f_calc_in_p1=f_calc_in_p1)

  # None if no sphere could be made; coverage then never applies.
  reference = completeness_reference(f_obs_p1)

  # The absences every candidate predicts, which separate none of them.
  shared = shared_absences(f_obs_p1, candidates)

  # Decided once per file, not per candidate: see `_discriminating_ranking`.
  discriminating = _discriminating_ranking(f_obs)

  # Phases shared across candidates (see `point_group_agreement`). Needs
  # Friedel mates: rotated indices otherwise leave the stored hemisphere and
  # silently yield nothing. Computing the metric and refuting on it are
  # separate switches, so the metric can be recorded without the filter.
  phase_cache = {}
  f_calc_for_phases = None
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
    # The excess missing count of its predicted class (margin * n) exceeds the
    # count it can measure: the group is credited and bucketed like a
    # coverage-judged one (evidence_fraction below, absence_bucket further down).
    coverage_dominant = (coverage is not None
                         and coverage*n_coverage > COVERAGE_DOMINANCE_FACTOR*n_absent
                         and ratio < COVERAGE_DOMINANCE_MAX_RATIO)
    entry = group_args(
      space_group_info=sgi,
      absence_ratio=ratio,
      n_predicted_absent=n_absent,
      # over the absences only this group predicts
      own_absence_ratio=own_ratio,
      n_own_absent=n_own,
      # Phase evidence about the rotations, independent of the absences.
      phi_point_group=phi_point_group,
      own_absence_bucket=(round(own_ratio, _absence_bucket_decimals())
                          if n_own >= MIN_ABSENCES_TO_JUDGE else None),
      coverage_margin=coverage,
      n_coverage_absent=n_coverage,
      # Fraction of the reflection list this group declares absent, comparable
      # across the two tests where raw counts are not. Measured weakness and
      # inferred absence concern different reflection classes, so both add up
      # (otherwise C2/c, judged on its measured c-glide class, loses to C2/m
      # credited with the deleted centring class it explains equally). The
      # coverage side counts only when the group is not judged by intensity or
      # is coverage_dominant: on fully measured data every margin sits at the
      # incompleteness noise and would credit whichever group scrapes over.
      evidence_fraction=(
        (n_absent/max(1, f_obs_p1.size())
         if n_absent >= MIN_ABSENCES_TO_JUDGE else 0.0)
        + (n_coverage/max(1, reference[0].size())
           if (reference is not None and n_coverage
               and (n_absent < MIN_ABSENCES_TO_JUDGE or coverage_dominant))
           else 0.0)),
      coverage_dominant=coverage_dominant,
      # Per candidate, not per data set: on merged data the true group's
      # absences are deleted and only coverage can judge it, while a wrong
      # group predicting a superset still has strong reflections present for
      # the intensity test. Intensity is preferred where it applies: measured
      # weakness is stronger evidence than inferred absence.
      judged_by=("intensity" if n_absent >= MIN_ABSENCES_TO_JUDGE
                 else "coverage" if (coverage is not None
                                     and (n_coverage >= MIN_ABSENCES_TO_JUDGE
                                          or (n_coverage >= STRONG_ABSENCE_MIN
                                              and coverage >= COVERAGE_SURE_MARGIN)))
                 else "nothing"),
      centric_agrees=(None if centric is None
                      else bool(group.is_centric()) == centric),
      frequency_rank=(FREQUENCY_ORDER.index(number)
                      if number in FREQUENCY_ORDER else len(FREQUENCY_ORDER)),
      reason=None)
    entry.reason = None
    scored.append(entry)

  # Real data always shows some intensity at predicted absences (multiple
  # scattering, overlap with neighbours, detector artefacts), hence the
  # generous cut. A group predicting no absences cannot violate any and is
  # never judged on them.
  survivors = []
  for entry in scored:
    # Phases refute point groups, absences refute glides and screws. Checked
    # first because it also reaches candidates the absence test cannot judge
    # (Pmmm, P222). Gated on the filter switch, not on the value existing:
    # computing the metric must not turn the refutation on.
    if (_point_group_filter()
        and entry.phi_point_group is not None
        and entry.phi_point_group > MAX_POINT_GROUP_DISAGREEMENT):
      entry.reason = (
        "the solution's phases do not obey its rotations (phi_sym %.2f, "
        "refuted above %.2f)"
        % (entry.phi_point_group, MAX_POINT_GROUP_DISAGREEMENT))
      refuted.append(entry)
      continue
    if (entry.n_predicted_absent >= STRONG_ABSENCE_MIN
        and entry.absence_ratio > STRONG_ABSENCE_RATIO):
      entry.reason = (
        "its %d predicted absences are stronger than the mean intensity "
        "(%.1fx), so they are not absent"
        % (entry.n_predicted_absent, entry.absence_ratio))
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
    # Coverage. Deliberately weak: "predicted absences are present" only
    # damns a group if the file is one that deletes absences, which is
    # inferred. Only a class barely more missing than the data is incomplete
    # generally counts against the group.
    if entry.coverage_margin is not None \
       and entry.coverage_margin < MIN_COVERAGE_MARGIN:
      entry.reason = (
        "only %.0f%% of its %d predicted absences are missing, no more than "
        "the data is incomplete generally"
        % (100*entry.coverage_margin, entry.n_coverage_absent))
      # Inferred evidence demotes, it does not remove: removal is final (the
      # fallback only fires when every candidate is refuted), and the truth
      # can fail here with perfectly weak measured absences. Demoted entries
      # rank behind the other survivors.
      if _coverage_refutes():
        refuted.append(entry)
      else:
        entry.coverage_demoted = True
        survivors.append(entry)
    else:
      survivors.append(entry)
  scored = survivors

  # Everything refuted means the filter does not apply to this data (twinning,
  # an incommensurate component, a bad cell): fall back to the unfiltered list
  # rather than an empty one that would read as "no symmetry".
  if not scored and refuted:
    for entry in refuted:
      entry.reason = "no candidate is clean; showing the least bad"
    scored = refuted

  # Absences that really are absent are positive evidence, ranked above the
  # frequency prior: P2(1)2(1)2(1) explains its weak screw absences, P222
  # predicts none and explains nothing, yet fits equally well otherwise.
  for entry in scored:
    # Credit only for absences explained: predicted ones that are weak or, on
    # merged data, missing.
    entry.has_absence_evidence = (
      (entry.judged_by == "intensity"
       and entry.absence_ratio <= max_absence_ratio)
      or (entry.judged_by == "coverage"
          and entry.coverage_margin is not None
          and entry.coverage_margin >= MIN_COVERAGE_MARGIN))
    # Rounded: ratios this close are noise, and within a bucket the later keys
    # decide. Coverage-judged candidates get the best bucket, 0.0 (a margin
    # near 1 means the class really is gone), but are not ranked ahead of
    # intensity-judged ones.
    if not entry.has_absence_evidence:
      entry.absence_bucket = 0.0
    elif entry.judged_by == "intensity":
      # Coarse (see ABSENCE_BUCKET_DECIMALS): real absences carry leakage that
      # depends on the reflection class, so a finer bucket lets C2/m's clean
      # centring class beat C2/c's leaky glide class although C2/c explains
      # more. It still separates a centred group at 0.22 from a true one at
      # 0.01. A coverage_dominant group is bucketed like its coverage-judged
      # rivals so the tie falls through to evidence_fraction.
      entry.absence_bucket = (0.0 if entry.coverage_dominant
                              else round(entry.absence_ratio,
                                         _absence_bucket_decimals()))
    else:
      entry.absence_bucket = 0.0

  disputed =(centric_by_stat is not None and centric_by_alpha is not None
              and centric_by_stat != centric_by_alpha)
  # Evidence before prior; how convincingly absent (bucket) before how much
  # (evidence_fraction), so a centred group whose many absences scraped under
  # the cut at 0.22 does not bury a true group with few at 0.01.
  def order(e):
    demoted = 1 if getattr(e, "coverage_demoted", False) else 0
    # evidence_fraction is a fraction, not a count: the intensity test counts
    # over the observed list, coverage over the complete sphere, and a
    # coverage-judged group has n_predicted_absent == 0 by construction.
    # Intensity is not ranked above coverage as a class: on merged data the
    # true group is the one whose absences were deleted.
    #
    # With `discriminating`, the group's own absence class (what it claims
    # that its rivals do not) decides first: groups of one Laue class share
    # their centring, so the whole-set ratio is dominated by shared evidence.
    # None sorts last. The coverage margin (two decimals, so noise does not
    # order) comes before the frequency prior: a prior must not overrule a far
    # stronger measurement.
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

  # The centricity key is dropped only when the two tests disagree and the
  # statistic sits inside CENTRIC_AMBIGUOUS; outside that band the statistic
  # is right far more often than alpha_0 and keeps its say.
  if disputed and stat is not None and (
      CENTRIC_AMBIGUOUS[0] <= stat <= CENTRIC_AMBIGUOUS[1]):
    scored.sort(key=order)
  else:
    scored.sort(key=lambda e: ((0 if e.centric_agrees is not False else 1),)
                + order(e))

  if PHI_RERANK and len(scored) > 1:
    _phi = [getattr(e, "phi_point_group", None) for e in scored]
    # All or nothing: a partial column would let a candidate win on having
    # been skipped.
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

  # Run-off among the leaders on the reflections that separate them (two
  # groups differing by one glide). Limited to the leading few: O(k^2)
  # absence tests. Stable sort, so a silent contest changes nothing. Also run
  # on measured data with leaky absences, but not when a leader is judged by
  # coverage with almost nothing measured: it cannot win a pairwise test.
  # The contest covers the first PAIRWISE_RUNOFF_SIZE absence patterns, not
  # entries: a centric group and its acentric twin (Pbcm/Pbc2(1)) share a
  # pattern and would otherwise fill the slots.
  sig = lambda e: (e.n_predicted_absent, e.n_coverage_absent,
                   round(e.absence_ratio, 9))
  sigs, first = [], {}
  for e in scored:
    if sig(e) not in first:
      first[sig(e)] = e
      if len(sigs) < PAIRWISE_RUNOFF_SIZE:
        sigs.append(sig(e))
  top = scored[:PAIRWISE_RUNOFF_SIZE]
  top += [first[g] for g in sigs if not any(first[g] is e for e in top)]
  measured = (any(e.judged_by == "intensity" for e in top)
              and not any(e.judged_by == "coverage" and e.n_predicted_absent < 5 for e in top))
  if (discriminating or measured) and len(scored) > 1:
    reps = [first[g] for g in sigs]
    wins = dict(zip(sigs, copeland_scores(f_obs_p1, reps, max_ratio=max_absence_ratio)))
    order_of = dict((id(e), i) for i, e in enumerate(top))
    leaders = sorted(top, key=lambda e: (-wins[sig(e)], order_of[id(e)]))
    scored = leaders + [e for e in scored if id(e) not in order_of]

  # Alpha fills the shortlist, it does not lead it: origin-searched symmetry
  # agreement builds good shortlists but ranks top-1 badly, and no rank-fusion
  # weighting helped. The first ALPHA_KEEP_LEADERS stay where the evidence put
  # them; alpha orders the remaining places the frequency prior would fill.
  # Candidates move down, never off.
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
      # stable; a missing alpha sorts behind every measured one
      window = sorted(window,
                      key=lambda e: -(e.alpha if e.alpha is not None else -1.0))
      scored = (scored[:ALPHA_KEEP_LEADERS] + window
                + scored[ALPHA_KEEP_LEADERS + ALPHA_WINDOW:])
    except Exception:
      pass

  for entry in scored:
    if entry.reason is None:
      bits = []
      # say which test spoke: measured weak, or merely not in the file
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

  short, seen = [], set()
  for e in scored:
    n = e.space_group_info.type().number()
    if n not in seen:
      seen.add(n)
      short.append(e)
  head = [e.space_group_info.group().is_centric() for e in short[:n_suggestions]]
  # Extra entries appended after the head; the ranking itself is untouched.
  # Each head entry's partner of the other centricity with the same absences
  # (Pna2(1) beside Pnma) is appended: the absences cannot tell them apart and
  # an extra suggestion costs one trial. A head of one centricity rests on
  # <|E^2-1|> alone, so the best group of the other centricity goes first.
  centric_of = lambda e: e.space_group_info.group().is_centric()
  same = lambda a, b: (a.n_predicted_absent == b.n_predicted_absent
                       and a.n_coverage_absent == b.n_coverage_absent
                       and abs(a.absence_ratio - b.absence_ratio) < 1e-9)
  rest = short[n_suggestions:]
  extra = []
  for h in short[:n_suggestions]:
    twin = [e for e in rest if centric_of(e) != centric_of(h) and same(e, h)]
    if twin and twin[0] not in extra:
      extra.append(twin[0])
  if head and len(set(head)) == 1:
    other = [e for e in rest if centric_of(e) != head[0]]
    if other and other[0] not in extra:
      extra.insert(0, other[0])
  # Pseudo-centring: a head of centred groups whose forbidden class is present
  # and significant points to a heavy-atom substructure, not centring (truly
  # centred files almost never carry that class), so the best primitive group
  # goes first.
  lattice = lambda e: e.space_group_info.group().conventional_centring_type_symbol()
  if (short[:n_suggestions] and "P" not in [lattice(e) for e in short[:n_suggestions]]
      and pseudo_centring(f_obs_p1, lattice(short[0])) > PSEUDO_CENTRING_SHARE):
    prim = [e for e in rest if lattice(e) == "P"]
    if prim and prim[0] not in extra:
      extra.insert(0, prim[0])
  short = short[:n_suggestions] + extra
  n_suggestions += len(extra)
  return group_args(
    suggestions=short[:n_suggestions],
    all_candidates=scored,
    refuted=refuted,
    e_sq_minus_one=stat,
    alpha_0=alpha_0,
    centric=centric,
    centric_by_stat=centric_by_stat,
    centric_by_alpha=centric_by_alpha,
    # the two centrosymmetry tests disagree: the P1/P-1 choice is open
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
