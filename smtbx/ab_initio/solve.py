""" One call behind the button: solve, then suggest space groups.

    result = solve.structure(f_obs)
    solve.show(result)

The space group `f_obs` carries is not trusted for the first pass: solving in P1
finds more structures than a plausible but wrong assumed group. Once a group is
chosen the structure is solved again with that symmetry enforced (see
`solve_in`). The caller gets a ranked shortlist of space groups rather than one
answer, since the space group is the least reliable part of the result.
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

  `n_trials`: 8 is the interactive default; 16 is the "work harder" option,
  and more gains nothing.

  `laue_group_info`, from R_int over unmerged equivalents, settles the Laue
  class more reliably than the solution map, from which the candidates are
  built otherwise.

  Normalisation is quasi-E, with deliberately no formula argument: true-E with
  a realistically wrong user formula does no better than quasi-E.
  """
  import sys
  import time

  if out is None:
    out = sys.stdout
  t0 = time.time()

  # arrays built in code often lack an observation type, on which the
  # symmetrisation asserts deep inside every trial
  if not f_obs.is_xray_amplitude_array():
    if f_obs.is_xray_intensity_array():
      f_obs = f_obs.as_amplitude_array()
    else:
      f_obs = f_obs.set_observation_type_xray_amplitude()

  # expanded here so a wrong assumed group never averages unrelated
  # reflections in the merging step
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
    # the composite choice: solve in each shortlisted group and let a refined
    # free-R1 vote alongside the ranking; off by default (SMTBX_COMPOSITE),
    # it costs three solves and refinements instead of one solve
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
          # the shortlist reordered to match, by symbol: the two lists hold
          # different objects
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

  Off by default; `SMTBX_COMPOSITE=1` enables it. """
  import os
  return os.environ.get("SMTBX_COMPOSITE", "0") not in ("", "0", "false",
                                                        "False")


def _expected_atom_count(f_obs):
  """ Roughly how many non-hydrogen atoms the asymmetric unit holds.

  At 18.6 A^3 per non-hydrogen atom: the composite's peak budget, with no
  composition to work from.
  """
  try:
    unit_cell = f_obs.unit_cell()
    order = max(1, len(f_obs.space_group()))
    return max(4, int(unit_cell.volume()/18.6/order))
  except Exception:
    return 20


def solve_in(f_obs, space_group_info, f_calc_in_p1=None, n_trials=4,
             out=None, n_threads=1):
  """ Solve again with the chosen symmetry enforced. Falls back to placing.

  Charge flipping in the space group averages symmetry-equivalent density on
  every cycle, raising the map's signal-to-noise by about the group order;
  placing a finished P1 solution applies the symmetry only once, at the end,
  and solves far fewer structures correctly.
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
      out=out, n_threads=n_threads)
    if result.f_calc is not None:
      return result.f_calc
  except Exception as e:
    print("Could not re-solve in %s (%s); placing the P1 solution instead."
          % (space_group_info, e), file=out)
  # a placed solution is worse but still a structure
  if f_calc_in_p1 is None:
    return None
  return place_in(f_obs, f_calc_in_p1, space_group_info)


def place_in(f_obs, f_calc_in_p1, space_group_info):
  """ The P1 solution expressed in a chosen space group, or None.

  A translation search, not a re-solve, so switching to another suggestion in
  a GUI is cheap.
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
