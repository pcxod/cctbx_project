""" Dual-space building blocks: atomicity in real space, amplitudes in
reciprocal space, and a figure of merit that can compare across symmetries.

Charge flipping is a real-space modification plus an FFT, and nothing in it ever
says "this density should look like atoms". SHELXT and SHELXD do say that, and
they say it every cycle: pick peaks, make them atoms, compute structure factors
from that model, keep the observed amplitudes but take the model's phases, and
go round again. Atomicity is the strongest prior available for a small-molecule
structure and we have not been using it.

    real space         peaks -> carbon atoms          (atomicity)
    reciprocal space   |F_obs| with model phases      (the measurement)

**The figure of merit is computed on held-out reflections.** This is the part
that matters most and the reason the last attempt at choosing between candidate
space groups failed outright. Correlation of |F_obs| with |F_calc| over all data
scored 0.117 as a chooser -- reliably *anti*-correlated -- because a
lower-symmetry candidate has more independent atoms and therefore always fits
the data better. Degrees of freedom differ between candidates, so any
whole-data fit is a comparison of model complexity rather than of correctness.

Reflections held out of the recycling cannot be fitted by extra freedom, so the
bias cancels by construction. It is the same argument as R_free in refinement,
applied to space-group choice.

**Every block is separately callable and separately measurable.** `peaks_to_model`,
`model_to_phases`, `recycle` and `free_correlation` are the pieces; `improve`
composes them. Nothing here decides anything on its own -- combinations get
measured by `bench_dual.py` on both corpora, in the cross-validated way, before
any of it becomes the shipping path.

**Keeps the best cycle, not the last.** A recycling loop eventually starts
fitting noise, and the free-reflection score is what notices; the internal
residual will happily keep falling while the model gets worse.
"""
from __future__ import absolute_import, division, print_function

DEFAULT_FREE_FRACTION = 0.10
DEFAULT_CYCLES = 8
# Peaks are picked at slightly more than the expected atom count: a model one
# atom short leaves a hole that the next difference map has to re-find, while a
# few spurious atoms are diluted by the ones that are right. 1.1 rather than the
# 1.3 the peak-budget work settled on, because here the peaks become *atoms*
# whose scattering enters every later cycle, rather than a list handed to a user.
PEAK_FACTOR = 1.1


def split_free(f_obs, fraction=DEFAULT_FREE_FRACTION, seed=0):
  """ (working, free) -- the free set is never shown to the recycling.

  Deterministic for a given array and seed, so a re-run compares like with
  like and two candidate space groups are judged on the same held-out
  reflections rather than on two different random draws.
  """
  from cctbx.array_family import flex

  n = f_obs.size()
  if n < 200:
    # Too few reflections for a meaningful held-out set. Returning the whole
    # array as both is wrong in a way that would be invisible, so refuse.
    return f_obs, None
  # Seed *before* drawing. `flex.random_bool` uses the global generator, so
  # seeding afterwards would make this deterministic-looking and actually
  # arbitrary -- and two candidate space groups would then be judged on two
  # different held-out sets, which is the one thing this function exists to
  # prevent.
  flex.set_random_seed(seed)
  flags = flex.random_bool(n, fraction)
  return f_obs.select(~flags), f_obs.select(flags)


def match_arrays(a, b):
  """ (a, b) restricted to the reflections they share. None if too few.

  **`common_sets` is not usable here.** Even with
  `assert_is_similar_symmetry=False` it asserts on other properties -- the
  anomalous flag among them -- and raised a bare `AssertionError` for every
  candidate of five structures in thirteen, which is how half the ablation came
  back with no held-out score at all. `miller.match_indices` does the one thing
  actually wanted, pairs up Miller indices, and asserts nothing about where they
  came from.

  The observed array and the phase source genuinely differ in provenance here:
  one is re-merged in the candidate's symmetry, the other is whatever
  `solve_in` produced. Insisting they agree about anomalous scattering before
  intersecting them is a check with no bearing on the operation.
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

  All carbon deliberately. Element identity is a separate measured step and
  guessing it here would put its errors inside the recycling, where a wrong
  scattering factor corrupts every later map.
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

  Direct summation is O(reflections x atoms) and this loop calls it twice per
  cycle, eight cycles per candidate, three candidates per structure -- it is the
  whole runtime. The FFT route is asymptotically better and takes over well
  before the sizes here; below a few hundred atoms the direct sum wins on
  constant factors, so the choice is made on the model rather than fixed.

  Falls back to direct if the FFT path raises: some small or oddly-shaped cells
  do not give a usable grid, and a slow answer beats no answer.
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

  `phase_transfer` rather than using F_calc directly -- the amplitudes are the
  measurement and must survive the cycle; only the phases come from the model.
  """
  f_calc = structure_factors(f_obs_working, model)
  return f_obs_working.phase_transfer(f_calc), f_calc


def _normalised(amplitudes):
  """ E-values: amplitudes divided by their mean in resolution shells.

  **On E, not on F.** Raw |F| is dominated by a handful of strong low-angle
  reflections that every candidate fits about equally, so a correlation on F
  mostly measures the shared low-resolution envelope and barely sees the
  differences that distinguish one space group -- or one cycle -- from another.
  Normalising per resolution shell removes that envelope and is what the
  correlation coefficient in SHELXD is computed on.

  Measured consequence of getting this wrong: the F-based version failed to
  protect the model inside the recycling loop, which then degraded every band of
  starting quality (0.960 -> 0.944 on the good ones), and scored 0.41 as a
  space-group chooser against the module's 0.68.
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

  SHELXT and its ancestors phase on a few hundred to a few thousand of the
  largest normalised amplitudes rather than on everything. Weak reflections
  carry little phase information but plenty of measurement error, and including
  them dilutes a correlation with terms that are near zero on both sides and
  agree for that reason alone -- which flatters every candidate equally and is
  exactly the kind of shared baseline that stops a figure of merit from
  discriminating.

  Returns the input unchanged if normalisation fails, so this can never make a
  caller worse off than not having tried.
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

  `on_e=False` recovers the F-based version and `strong=False` scores on all
  held-out reflections rather than the largest E only. Both are switchable
  because adding the two blocks together *narrowed* the separation on the first
  structure tested -- truth against best rival went from 0.47 to 0.13 -- and
  with two changes at once there is no way to tell which did it. Ablation, not
  argument.
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
      # When the observed side was not already normalised by `strongest_e`,
      # normalise it here -- correlating E_calc against raw |F_obs| compares
      # two different quantities and was silently doing so whenever
      # `strong=False`.
      if not strong and e_obs is not None:
        obs = e_obs
    # Already index-matched above; normalising can only drop reflections that
    # fall outside a binner, so re-match rather than assume the two sides still
    # line up -- and do it with `match_arrays`, not `common_sets`, for the same
    # reason as everywhere else here.
    matched = match_arrays(obs, calc)
    if matched is None:
      return None
    obs, calc = matched
    return flex.linear_correlation(
      flex.abs(obs.data()), flex.abs(calc.data())).coefficient()
  except Exception:
    return None


def refine_model_smtbx(f_working, model, cycles=8):
  """ Refine with the **smtbx normal-equations engine** -- what Olex2 uses.

  The first version of this used `cctbx.xray.minimization.lbfgs`, which is a
  gradient minimiser against a least-squares residual on F. It is cctbx, but it
  is not the refinement engine: no weighting scheme, no scale factor as a
  refined parameter, no origin fixing, and LBFGS convergence rather than normal
  equations. Twelve LBFGS iterations is a long way from a converged refinement,
  and the refined-R1 chooser was being judged on that.

  This is the engine we ship in Olex2, with SHELX weighting and origin-fixing
  restraints -- which matter here specifically, because a polar space group has
  a free origin and an unrestrained refinement will drift along it rather than
  improve the fit.

  Returns None on failure so the caller can distinguish "this engine could not
  run" from "it ran and changed nothing" -- the distinction the LBFGS version
  lost by returning its input.
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

    # The engine works on intensities; ours are amplitudes.
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
  """ Least-squares refinement of a peak model against the working set.

  **The one thing six refuted discriminators all had in common was not doing
  this.** Whole-data correlation, held-out correlation on F, on E, on the
  strongest E, peak-height contrast and a peak-built model's correlation were
  each measured and each failed; every one of them scored a *fixed* model --
  peaks placed where the map put them, all carbon, U pinned at 0.05. SHELXT
  reports a refined R1 per candidate, and refinement is what turns "these peaks
  roughly fit" into "these peaks are a structure".

  The argument for why it should discriminate where a correlation does not: a
  model in the wrong space group is not merely a worse fit, it is a fit that
  *cannot improve*, because the symmetry constrains atoms into positions the
  density does not support. The right group's model has somewhere to go. That
  difference shows up in refinement and is invisible to a single-shot
  correlation.

  Returns the refined structure, or the input if refinement raises -- a failed
  refinement must not silently look like a converged one, so the caller is told
  by getting its own model back unchanged.
  """
  from cctbx import xray

  try:
    refined = model.deep_copy_scatterers()
    refined.scatterers().flags_set_grads(state=False)
    refined.scatterers().flags_set_grad_site(
      iselection=refined.all_selection().iselection())
    refined.scatterers().flags_set_grad_u_iso(
      iselection=refined.all_selection().iselection())
    # Iteration count goes through `lbfgs_termination_params`; there is no
    # `max_iterations` argument, and passing one raised `TypeError` on every
    # candidate -- which the `except` below turned into "refinement ran and
    # changed nothing", indistinguishable from convergence. `r1_ref` came back
    # exactly equal to `r1_raw` on every row, which is the tell.
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

  R1 rather than a correlation because it is the number a crystallographer
  reads and the number SHELXT prints, and because it is sensitive to the size
  of the disagreement rather than only to its pattern. Lower is better, so
  callers comparing candidates want the minimum.
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

  **This is the block whose absence made plain recycling harmful.** A loop that
  only re-picks peaks from its own map feeds its own mistakes back every cycle:
  a wrongly placed atom contributes scattering, which reinforces the density
  that put it there, and the model has no mechanism to let go of it. Measured,
  that loop degraded every band of starting quality -- 0.960 -> 0.944 on good
  solutions.

  Randomly omitting a fraction of the atoms and recomputing gives it one. If
  the omitted set was wrong, the figure of merit improves without them and the
  smaller model is kept; if it was right, the score falls and nothing changes.
  It is the same trick as an omit map, applied to the whole peak list, and it is
  what makes SHELXD's dual-space recursion converge rather than ossify.

  Returns (model, score) -- the input unchanged when no omission beats it, so
  the caller can apply it unconditionally.
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

  `start_phases` is any complex array to take initial phases from -- the
  charge-flipping solution, a Patterson-seeded one, or the output of an earlier
  call to this function, which is what makes the blocks composable.

  Returns a group_args with `model`, `f_map` (amplitudes with the model's
  phases), `cc_free`, `cycle`, and `history`, or None if nothing worked.
  """
  from libtbx import group_args

  # **Match the index sets first, always.** `phase_transfer` requires the two
  # arrays to carry the same reflections, and they never do here: the observed
  # array has been re-merged in the candidate's symmetry while the phases come
  # from `solve_in`'s own set. It therefore raised `cctbx Internal Error` on
  # *every* candidate, and the intersection was only ever reached as an
  # exception fallback -- which failed often enough to leave half the
  # candidates with no score at all, so the chooser was being judged on the
  # minority of cases where a fallback happened to work.
  #
  # Intersecting up front is what the operation actually needs, and it also
  # lets a genuinely empty overlap be diagnosed instead of being caught by a
  # bare except.
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
      # **Held-out agreement decides, and only it.** The working-set residual
      # falls monotonically whether or not the model is getting better, which
      # is exactly how a recycling loop talks itself into fitting noise.
      if cc is not None and (best is None or cc > best.cc_free):
        best = group_args(model=model, f_map=current, cc_free=cc, cycle=cycle,
                          history=None)
    except Exception:
      break

  # **Return something the caller can measure even when the score is
  # unavailable.** Returning None whenever `free_correlation` never produced a
  # value made 53 of 155 candidates report `cycle = -1` *and* `cc_free = -1`,
  # and the chooser comparison then took `max` over a field that was -1 for a
  # third of the rows -- picking arbitrarily and reading as a weak signal
  # rather than as a broken measurement. A result with an honest "no score" is
  # diagnosable; a None is not.
  if best is None:
    best = first
  if best is not None:
    best.history = history
  return best


def improve(f_obs, f_calc_start, n_atoms, cycles=DEFAULT_CYCLES, seed=0):
  """ One call: recycle from a starting solution, return the improved map.

  Falls back to the input when nothing improves on it, so a caller can apply
  this unconditionally without having to decide whether it helped.
  """
  result = recycle(f_obs, f_calc_start, n_atoms, cycles=cycles, seed=seed)
  if result is None:
    return None
  return result
