""" One call behind the button: solve, then suggest space groups.

    result = solve.structure(f_obs)
    solve.show(result)

`f_obs` is the measured data in whatever symmetry the user currently has set.
The space group it carries is **not** trusted for the first pass: solving in P1
finds more structures than an assumed group does, is much faster, and beats a
plausible-but-wrong assumption by 13 percentage points, so this path does not
commit to a group before it has evidence for one.

But P1 is only where the symmetry is *found*. Once a group is chosen the
structure is solved **again with that symmetry enforced**, because charge
flipping in the space group averages symmetry-equivalent density every cycle
and a P1 solution never gets that benefit. Measured over 299 structures never
used for tuning: solving in-group is right 84.9% of the time against 47.5% for
solving in P1 and placing the result. Skipping the second pass was the largest
single defect this pipeline had.

What the caller gets back is a solution plus a short ranked list of candidate
space groups -- not a single answer -- because the space group is where this
pipeline actually loses. Measured on a uniform Crystallography Open Database
sample: solutions are correct 83.8% of the time and the ranking has only ~3
points of headroom left, while the space group is right 61-69% of the time from
a single answer, and 149 structures solve correctly and then lose their
symmetry for every 16 that do the reverse.
"""
from __future__ import absolute_import, division, print_function

from libtbx import group_args
from smtbx.ab_initio import charge_flipping, multi_trial, space_group_suggest


def structure(f_obs,
              n_trials=8,
              laue_group_info=None,
              n_suggestions=3,
              resolve_in_group=True,
              max_seconds=None,
              loop=None,
              callback=None,
              verbose=False,
              out=None):
  """ Solve in P1 and suggest space groups. Returns a group_args with

      f_calc_in_p1     the P1 solution, unsymmetrised
      suggestions      ranked candidate space groups, best first
      f_calc           the solution placed in the best candidate, or None
      space_group_info the best candidate, or None
      cc_peak_height   correlation peak height of the P1 solution
      solving          the full multi_trial result, for a log
      seconds

  `n_trials` defaults to 8. Sixteen scores 1.4 points better and takes 1.84x as
  long (about 10 s against 5 s on a typical small molecule); thirty-two adds a
  further 0.3 points, so the curve is flat past sixteen and there is nothing to
  gain by raising this further. Eight is the interactive default; offer sixteen
  as a "work harder" option rather than a different algorithm.

  `laue_group_info`, when known, is worth supplying. R_int over unmerged
  symmetry equivalents settles the Laue class before solving and does it far
  more reliably than anything recoverable from a solution map. Without it the
  candidate list is built from the solution instead, which is measurably worse.

  Normalisation is quasi-E and there is deliberately **no formula argument**.
  True-E normalisation with a perfect composition is worth about 1 point, but
  under every realistic corruption of a user-supplied formula -- wrong Z, rough
  counts, a missing element, or the common "one atom of each" -- it falls to or
  below plain quasi-E. Asking the user for a formula cannot pay for itself.
  """
  import sys
  import time

  if out is None:
    out = sys.stdout
  t0 = time.time()

  # Charge flipping expands to P1 internally regardless; doing it here makes
  # the choice explicit and keeps the assumed group out of the merging step,
  # where a wrong one would silently average unrelated reflections together.
  # Arrays read from a file carry an observation type; ones built in code often
  # do not, and `symmetry_search.misfit` asserts on it deep inside the
  # symmetrisation. The assertion surfaces as a bare AssertionError from every
  # trial, with nothing to say what was wrong. Label it here instead.
  if not f_obs.is_xray_amplitude_array():
    if f_obs.is_xray_intensity_array():
      f_obs = f_obs.as_amplitude_array()
    else:
      f_obs = f_obs.set_observation_type_xray_amplitude()

  f_obs_p1 = f_obs.expand_to_p1() if f_obs.space_group().order_z() > 1 \
      else f_obs

  solving = multi_trial.solve(
    f_obs_p1,
    n_trials=n_trials,
    normalisations_for=charge_flipping.amplitude_quasi_normalisations,
    max_seconds=max_seconds,
    loop=loop,
    callback=callback,
    verbose=verbose,
    out=out)

  if solving.f_calc_in_p1 is None:
    return group_args(
      f_calc_in_p1=None, suggestions=[], f_calc=None, space_group_info=None,
      cc_peak_height=None, solving=solving, suggestion_result=None,
      seconds=time.time() - t0)

  suggestion = space_group_suggest.suggest(
    f_obs, solving.f_calc_in_p1,
    laue_group_info=laue_group_info, n_suggestions=n_suggestions)

  f_calc, best_sgi = None, None
  if suggestion.suggestions:
    # **The composite choice, when enabled.** Instead of taking the module's
    # first suggestion on absence evidence alone, solve in each of the top
    # three and let a refined free-R1 vote alongside the ranking. Measured on
    # the shortlist: the module picks the truth 0.765 of the time, refined R1
    # alone 0.58, and the two fused 0.809.
    #
    # Off by default and behind an environment switch so the A/B is one code
    # version under two environments, and so Olex2 gets the cheap path unless
    # a user asks for the thorough one -- it costs three solves and three
    # refinements instead of one solve.
    best_sgi, f_calc = None, None
    if _composite_enabled() and resolve_in_group:
      try:
        from smtbx.ab_initio import composite

        n_heavy = _expected_atom_count(f_obs)
        ordered = composite.choose_space_group(
          f_obs, suggestion.suggestions, solving.f_calc_in_p1, n_heavy,
          out=out)
        if ordered and ordered[0].get("placed") is not None:
          best_sgi = ordered[0]["space_group_info"]
          f_calc = ordered[0]["placed"]
          # Report the reordered shortlist, so a user picking the second
          # suggestion in the GUI gets the composite's second and not the
          # ranking's. Reordered by symbol rather than by position: the two
          # lists hold different objects and zipping them merely relabels.
          position = dict((str(e["space_group_info"]), i)
                          for i, e in enumerate(ordered))
          suggestion.suggestions = sorted(
            suggestion.suggestions,
            key=lambda s: position.get(str(s.space_group_info), len(position)))
      except Exception:
        best_sgi, f_calc = None, None

    if best_sgi is None:
      best_sgi = suggestion.suggestions[0].space_group_info
      f_calc = solve_in(f_obs, best_sgi, solving.f_calc_in_p1,
                        n_trials=n_trials, out=out) if resolve_in_group \
          else place_in(f_obs, solving.f_calc_in_p1, best_sgi)

  return group_args(
    f_calc_in_p1=solving.f_calc_in_p1,
    suggestions=suggestion.suggestions,
    suggestion_result=suggestion,
    f_calc=f_calc,
    space_group_info=best_sgi,
    cc_peak_height=solving.cc_peak_height,
    solving=solving,
    seconds=time.time() - t0)


def _composite_enabled():
  """ Whether `structure` uses the composite choice among shortlisted groups.

  Off by default. `SMTBX_COMPOSITE=1` enables it. Olex2 reads this module
  directly, so the switch is the supported way to turn the thorough path on for
  a single run without shipping a second version of the file -- the same
  discipline as `SMTBX_ALPHA_SHORTLIST` and `SMTBX_COVERAGE_REFUTES`.
  """
  import os
  return os.environ.get("SMTBX_COMPOSITE", "0") not in ("", "0", "false",
                                                        "False")


def _expected_atom_count(f_obs):
  """ Roughly how many non-hydrogen atoms the asymmetric unit holds.

  From the cell volume and the space group order, at the usual 18.6 A^3 per
  non-hydrogen atom. The composite needs a peak budget and has no composition
  to work from -- this is the same estimate `bench_pipeline` falls back on when
  a corpus does not supply one.
  """
  try:
    unit_cell = f_obs.unit_cell()
    order = max(1, len(f_obs.space_group()))
    return max(4, int(unit_cell.volume()/18.6/order))
  except Exception:
    return 20


def solve_in(f_obs, space_group_info, f_calc_in_p1=None, n_trials=8,
             out=None):
  """ Solve again with the chosen symmetry enforced. Falls back to placing.

  **This step is worth 37 percentage points and its absence was the single
  largest defect in this pipeline.** Measured over 299 structures never used
  for tuning, scored identically: solving with the space group enforced got the
  structure right 84.9% of the time, while solving in P1 and merely placing the
  result in the same group managed 47.5%.

  The reason is that charge flipping in the space group averages
  symmetry-equivalent density on every cycle, which improves the
  signal-to-noise of the map by something like the order of the group. Placing
  a finished P1 solution applies the symmetry once, at the end, and recovers
  none of that -- the map it symmetrises was built without ever using the
  constraint.

  So P1 is the right place to *find* the symmetry and the wrong place to leave
  the answer. An earlier three-structure check suggested the two were
  equivalent, which is why this shipped the wrong way round; three structures
  cannot see a 37-point effect reliably.
  """
  import sys
  from smtbx.ab_initio import charge_flipping, multi_trial

  if out is None:
    out = sys.stdout
  try:
    f_obs_g = f_obs.customized_copy(
      space_group_info=space_group_info).merge_equivalents().array()
    if f_obs_g.size() < 20:
      raise ValueError("too few reflections after merging")
    result = multi_trial.solve(
      f_obs_g, n_trials=n_trials,
      normalisations_for=charge_flipping.amplitude_quasi_normalisations,
      out=out)
    if result.f_calc is not None:
      return result.f_calc
  except Exception as e:
    print("Could not re-solve in %s (%s); placing the P1 solution instead."
          % (space_group_info, e), file=out)
  # Falling back rather than returning nothing: a placed solution is worse but
  # is still a structure, and the user can see it.
  if f_calc_in_p1 is None:
    return None
  return place_in(f_obs, f_calc_in_p1, space_group_info)


def place_in(f_obs, f_calc_in_p1, space_group_info):
  """ The P1 solution expressed in a chosen space group, or None.

  Separate from `structure` on purpose: a user picking the second or third
  suggestion needs exactly this and nothing else re-run. Placing a solution is
  a translation search, not a re-solve, so switching suggestions is cheap and
  should feel instant in a GUI.
  """
  f_obs_g = f_obs.customized_copy(
    space_group_info=space_group_info).merge_equivalents().array()
  solutions = list(charge_flipping.f_calc_symmetrisations(
    f_obs_g, f_calc_in_p1, min_cc_peak_height=0.0))
  if not solutions:
    return None
  return solutions[0][0]


def show(result, out=None):
  """ What the button should print. """
  import sys
  if out is None:
    out = sys.stdout
  if result.f_calc_in_p1 is None:
    print("No solution found in %.1f s." % result.seconds, file=out)
    return
  print("Solved in %.1f s (correlation %.3f)."
        % (result.seconds, result.cc_peak_height or float("nan")), file=out)
  if result.suggestion_result is not None:
    space_group_suggest.show(result.suggestion_result, out=out)
