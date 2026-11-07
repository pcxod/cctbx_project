""" Dual-space building blocks: atomicity in real space, amplitudes in
reciprocal space, and a figure of merit that can compare across symmetries.

As in SHELXD: peaks become atoms (real space), the observed amplitudes take
the model's phases (reciprocal space), and round again.

The figure of merit is computed on held-out reflections. A lower-symmetry
candidate has more independent atoms and always fits the whole data better,
so a whole-data fit compares model complexity, not correctness; held-out
reflections cannot be fitted by extra freedom (R_free applied to space-group
choice). The recycling keeps the best cycle by that score, not the last: the
working residual keeps falling while the model fits noise.
"""
from __future__ import absolute_import, division, print_function

DEFAULT_FREE_FRACTION = 0.10
DEFAULT_CYCLES = 8
# peaks per expected atom: a missing atom leaves a hole the next map must
# re-find, a few spurious ones are diluted; kept low because each peak's
# scattering enters every later cycle
PEAK_FACTOR = 1.1


def split_free(f_obs, fraction=DEFAULT_FREE_FRACTION, seed=0):
  """ (working, free) -- the free set is never shown to the recycling.

  Deterministic for a given array and seed, so candidate space groups are
  judged on the same held-out reflections. No free set below 200 reflections.
  """
  from cctbx.array_family import flex

  n = f_obs.size()
  if n < 200:
    return f_obs, None
  # seeded before drawing: `flex.random_bool` uses the global generator
  flex.set_random_seed(seed)
  flags = flex.random_bool(n, fraction)
  return f_obs.select(~flags), f_obs.select(flags)


def match_arrays(a, b):
  """ (a, b) restricted to the reflections they share. None if too few.

  Not `common_sets`: even with `assert_is_similar_symmetry=False` it asserts
  on other properties such as the anomalous flag, which differs between an
  array re-merged in the candidate's symmetry and the `solve_in` phases.
  `miller.match_indices` pairs indices and asserts nothing.
  """
  from cctbx import miller

  try:
    pairs = miller.match_indices(a.indices(), b.indices()).pairs()
    if pairs.size() < 50:
      return None
    return (a.select(pairs.column(0)), b.select(pairs.column(1)))
  except Exception:
    return None


def peaks_to_model(f_map_coefficients, n_atoms, crystal_symmetry=None,
                   min_distance=1.0, u_iso=0.05):
  """ Real-space step: the map's strongest peaks become carbon atoms.

  All carbon: a guessed element inside the recycling would put a wrong
  scattering factor into every later map.
  """
  from cctbx import maptbx, xray

  fft_map = f_map_coefficients.fft_map(
    symmetry_flags=maptbx.use_space_group_symmetry)
  fft_map.apply_volume_scaling()
  search = fft_map.peak_search(
    parameters=maptbx.peak_search_parameters(
      min_distance_sym_equiv=min_distance,
      max_clusters=max(2, int(n_atoms))),
    verify_symmetry=False).all()
  sites = search.sites()
  symmetry = crystal_symmetry or f_map_coefficients.crystal_symmetry()
  model = xray.structure(symmetry.special_position_settings())
  for i, site in enumerate(sites):
    model.add_scatterer(xray.scatterer(label="C%d" % i, site=site, u=u_iso,
                                       scattering_type="C"))
  return model


def structure_factors(f_set, model, algorithm=None):
  """ F_calc from a model, by FFT where it pays and direct where it does not.

  Direct summation is O(reflections x atoms) and wins on small models; FFT
  takes over at 40 atoms. Falls back to direct if the FFT raises, as some
  small or oddly shaped cells give no usable grid.
  """
  if algorithm is None:
    algorithm = "fft" if model.scatterers().size() >= 40 else "direct"
  try:
    return f_set.structure_factors_from_scatterers(
      xray_structure=model, algorithm=algorithm).f_calc()
  except Exception:
    return f_set.structure_factors_from_scatterers(
      xray_structure=model, algorithm="direct").f_calc()


def model_to_phases(f_obs_working, model):
  """ Reciprocal-space step: observed amplitudes carrying the model's phases.

  Only the phases come from the model; the amplitudes are the measurement.
  """
  f_calc = structure_factors(f_obs_working, model)
  return f_obs_working.phase_transfer(f_calc), f_calc


def _normalised(amplitudes):
  """ E-values: amplitudes divided by their mean in resolution shells.

  A correlation on raw |F| mostly measures the low-resolution envelope that
  every candidate fits equally; normalising per shell removes it, as for the
  SHELXD correlation coefficient. None if normalisation fails.
  """
  try:
    work = amplitudes.customized_copy(
      data=amplitudes.data()).set_observation_type_xray_amplitude()
    work.setup_binner(auto_binning=True)
    return work.quasi_normalize_structure_factors()
  except Exception:
    return None


STRONG_E_FRACTION = 0.35


def strongest_e(amplitudes, fraction=STRONG_E_FRACTION):
  """ The subset with the largest E. Direct methods work on strong data.

  Weak reflections carry little phase information and agree near zero on
  both sides for every candidate alike, which dilutes a correlation. Returns
  the input unchanged if normalisation fails.
  """
  e = _normalised(amplitudes)
  if e is None or e.size() < 40:
    return amplitudes
  try:
    from cctbx.array_family import flex

    data = flex.abs(e.data())
    order = flex.sort_permutation(data, reverse=True)
    keep = max(40, int(round(e.size()*fraction)))
    selection = flex.bool(e.size(), False)
    selection.set_selected(order[:keep], True)
    return e.select(selection)
  except Exception:
    return amplitudes


def free_correlation(f_free, model, on_e=True, strong=True):
  """ Agreement on reflections the recycling never saw. Comparable across
  space groups, which whole-data agreement is not.

  `on_e=False` correlates |F| instead of E; `strong=False` scores all
  held-out reflections rather than the largest E only.
  """
  from cctbx.array_family import flex

  if f_free is None or f_free.size() < 20:
    return None
  try:
    obs = strongest_e(f_free) if (on_e and strong) else f_free
    calc = structure_factors(obs, model).as_amplitude_array()
    matched = match_arrays(obs, calc)
    if matched is None:
      return None
    obs, calc = matched
    if on_e:
      e_obs, e_calc = _normalised(obs), _normalised(calc)
      if e_calc is not None:
        calc = e_calc
      # without `strongest_e` the observed side is still |F|, not E
      if not strong and e_obs is not None:
        obs = e_obs
    # normalising can drop reflections outside a binner, so match again
    matched = match_arrays(obs, calc)
    if matched is None:
      return None
    obs, calc = matched
    return flex.linear_correlation(
      flex.abs(obs.data()), flex.abs(calc.data())).coefficient()
  except Exception:
    return None


def refine_model_smtbx(f_working, model, cycles=8):
  """ Refine with the smtbx normal-equations engine that Olex2 uses.

  SHELX weighting, a refined scale and origin-fixing restraints, without
  which a polar space group drifts along its free origin. None on failure,
  so "could not run" differs from "changed nothing".
  """
  try:
    from scitbx.lstbx import normal_eqns_solving
    from smtbx.refinement import least_squares
    from smtbx.refinement import constraints
    from smtbx.refinement.restraints import origin_fixing_restraints
    import smtbx.utils

    refined = model.deep_copy_scatterers()
    refined.scatterers().flags_set_grads(state=False)
    selection = refined.all_selection().iselection()
    refined.scatterers().flags_set_grad_site(iselection=selection)
    refined.scatterers().flags_set_grad_u_iso(iselection=selection)

    # the engine works on intensities
    f_sq = f_working.f_as_f_sq()
    observations = f_sq.as_xray_observations()

    connectivity = smtbx.utils.connectivity_table(refined)
    reparametrisation = constraints.reparametrisation(
      structure=refined, constraints=[], connectivity_table=connectivity)
    ls = least_squares.crystallographic_ls(
      observations, reparametrisation,
      weighting_scheme=least_squares.mainstream_shelx_weighting(),
      origin_fixing_restraints_type=(
        origin_fixing_restraints.atomic_number_weighting))
    normal_eqns_solving.levenberg_marquardt_iterations(
      ls, n_max_iterations=cycles, gradient_threshold=1e-7,
      step_threshold=1e-7)
    return refined
  except Exception:
    return None


def refine_model(f_working, model, cycles=12):
  """ LBFGS least-squares refinement of sites and U_iso against the working set.

  Discriminates where a fixed-model score does not: in the wrong space group
  the symmetry holds atoms where the density does not support them, so the
  fit cannot improve. Returns the input unchanged if refinement raises.
  """
  from cctbx import xray

  try:
    refined = model.deep_copy_scatterers()
    refined.scatterers().flags_set_grads(state=False)
    refined.scatterers().flags_set_grad_site(
      iselection=refined.all_selection().iselection())
    refined.scatterers().flags_set_grad_u_iso(
      iselection=refined.all_selection().iselection())
    # the iteration count goes through `lbfgs_termination_params`: a
    # `max_iterations` argument raises TypeError, which the except below
    # would hide as "converged unchanged"
    import scitbx.lbfgs

    xray.minimization.lbfgs(
      target_functor=xray.target_functors.least_squares_residual(f_working),
      xray_structure=refined,
      lbfgs_termination_params=scitbx.lbfgs.termination_parameters(
        max_iterations=cycles))
    return refined
  except Exception:
    return model


def free_r1(f_free, model):
  """ R1 on held-out reflections, with the scale refined on those same data.

  Sensitive to the size of the disagreement, not only its pattern. Lower is
  better.
  """
  from cctbx.array_family import flex

  if f_free is None or f_free.size() < 20:
    return None
  try:
    calc = structure_factors(f_free, model).as_amplitude_array()
    matched = match_arrays(f_free, calc)
    if matched is None:
      return None
    obs, cal = matched
    fo = flex.abs(obs.data())
    fc = flex.abs(cal.data())
    denominator = flex.sum(fc*fc)
    if denominator <= 0:
      return None
    scale = flex.sum(fo*fc)/denominator
    total = flex.sum(fo)
    if total <= 0:
      return None
    return flex.sum(flex.abs(fo - scale*fc))/total
  except Exception:
    return None


def omit_trial(working, free, model, fraction=0.3, trials=3, seed=0, strong=True):
  """ SHELXD's peak-list optimisation: drop atoms at random, keep it if better.

  Without it, recycling reinforces a wrong atom through its own scattering.
  An omitted set that was wrong raises the held-out score and the smaller
  model is kept. Returns (model, score), the input when no omission beats it.
  """
  import random

  from cctbx import xray

  best_model = model
  best_score = free_correlation(free, model, strong=strong)
  if best_score is None:
    return model, None
  scatterers = list(model.scatterers())
  n = len(scatterers)
  if n < 4:
    return model, best_score
  rng = random.Random(seed)
  keep_n = max(2, int(round(n*(1.0 - fraction))))
  for _ in range(max(1, trials)):
    keep = rng.sample(range(n), keep_n)
    trial = xray.structure(model.crystal_symmetry().special_position_settings())
    for i in sorted(keep):
      trial.add_scatterer(scatterers[i])
    score = free_correlation(free, trial, strong=strong)
    if score is not None and score > best_score:
      best_model, best_score = trial, score
  return best_model, best_score


def recycle(f_obs, start_phases, n_atoms, cycles=DEFAULT_CYCLES,
            free_fraction=DEFAULT_FREE_FRACTION, seed=0, omit=True,
            omit_fraction=0.3, omit_trials=3, strong=True):
  """ Iterate the two steps. Returns the best cycle by held-out agreement.

  `start_phases` is any complex array to take initial phases from.

  Returns a group_args with `model`, `f_map` (amplitudes with the model's
  phases), `cc_free`, `cycle`, and `history`, or None if nothing worked.
  """
  from libtbx import group_args

  # `phase_transfer` needs identical reflection sets, and the re-merged
  # observations never match the phase source's
  work_all, free = split_free(f_obs, fraction=free_fraction, seed=seed)
  matched = match_arrays(work_all, start_phases)
  if matched is None:
    return None
  working, matched_phases = matched
  try:
    current = working.phase_transfer(matched_phases)
  except Exception:
    return None

  best = None
  first = None
  history = []
  for cycle in range(max(1, cycles)):
    try:
      model = peaks_to_model(current, int(PEAK_FACTOR*n_atoms),
                             crystal_symmetry=f_obs.crystal_symmetry())
      if model.scatterers().size() < 2:
        break
      cc = None
      if omit:
        model, cc = omit_trial(working, free, model, fraction=omit_fraction,
                               trials=omit_trials, seed=seed + cycle,
                               strong=strong)
      current, f_calc = model_to_phases(working, model)
      if cc is None:
        cc = free_correlation(free, model, strong=strong)
      history.append(cc)
      if first is None:
        first = group_args(model=model, f_map=current, cc_free=-1.0,
                           cycle=cycle, history=None)
      # held-out agreement alone decides: the working residual falls anyway
      if cc is not None and (best is None or cc > best.cc_free):
        best = group_args(model=model, f_map=current, cc_free=cc, cycle=cycle,
                          history=None)
    except Exception:
      break

  # with no score at all, the first cycle with cc_free = -1 rather than None
  if best is None:
    best = first
  if best is not None:
    best.history = history
  return best


def improve(f_obs, f_calc_start, n_atoms, cycles=DEFAULT_CYCLES, seed=0):
  """ One call: `recycle` from a starting solution. """
  result = recycle(f_obs, f_calc_start, n_atoms, cycles=cycles, seed=seed)
  if result is None:
    return None
  return result
