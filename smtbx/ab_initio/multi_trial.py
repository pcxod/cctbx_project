""" Charge flipping run several times from different random starts, ranked.

`charge_flipping.solving_iterator` already restarts itself when an attempt
fails to reach a phase transition or a sharp correlation map -- by default up
to 5x5 times -- but every restart is thrown away, and only
`f_calc_solutions[0]` of the one attempt that finally worked is ever used.
Which seed a run happens to start from therefore decides the outcome, and
nothing compares the outcomes.

This module does the same total amount of work differently: each attempt gets
its own seed, is allowed exactly one try, and is *kept*. What was an invisible
retry becomes a scored candidate, and the best of them is returned.

Measured over 377 Crystallography Open Database entries with real measured
data, scoring peak positions against the deposited model with
`cctbx.euclidean_model_matching`, the solution rate roughly doubles.

**Ranking is just the correlation peak height.** That is worth stating
plainly, because it is not what one would guess and not what this started as.
Chemical plausibility of the peak list -- the fraction of peaks with a
neighbour at a bonding distance -- looks like the better signal on a single
structure, and is actively harmful at corpus scale: it correlates -0.08 with
correctness among candidates worth ranking, against +0.38 for the correlation
peak height, and choosing by it costs about half the solutions. A ranker is
easy to make worse while making it look more principled, so anything added
here should be measured with `notes/bench_cf.py` first.

References: the algorithm itself is in `charge_flipping.py`; see the papers
cited there.
"""
from __future__ import absolute_import, division, print_function

import math

from libtbx import group_args
from smtbx.ab_initio import charge_flipping


def phases_from_seed(seed):
  """ A reproducible random starting phase set for one trial.

  Deliberately a private generator rather than `flex.set_random_seed`, which
  seeds two process-global generators and so would reach into whatever else the
  process is doing -- and, in a GUI, make the result depend on what the user
  did beforehand. Seeds are what makes a trial differ from its neighbours, so
  they have to be under this module's control.
  """
  import scitbx_array_family_flex_ext as flex_ext
  generator = flex_ext.mersenne_twister(seed=seed)
  return lambda f_obs: generator.random_double(size=f_obs.size(),
                                               factor=2*math.pi)


class trial_result(group_args):
  pass


def solve(f_obs,
          n_trials=8,
          # 0.20 measured over a uniform sample of the Crystallography Open
          # Database: success is a clean interior maximum there (0.812, against
          # 0.802 at 0.25 and 0.788 at 0.30), and switching the treatment off
          # entirely costs 8.9 points -- so the Oszlanyi-Suto weak-reflection
          # step earns its place and only its tuning was ever in question.
          # An earlier 0.3 here was fitted on an unrepresentative corpus and
          # was 2.4 points worse than this, which is also the value Olex2 has
          # always used.
          weak_reflection_fraction=0.2,
          normalisations_for=charge_flipping.amplitude_quasi_normalisations,
          max_solving_iterations=500,
          yield_solving_interval=60,
          good_enough_cc_peak_height=0.99,
          max_seconds=None,
          first_seed=1,
          initial_phases_list=None,
          keep_solutions=False,
          loop=None,
          callback=None,
          verbose=False,
          out=None):
  """ n_trials seeded charge-flipping runs; the best solution found.

  Returns a group_args with `f_calc`, `shift` and `cc_peak_height` of the
  winner -- the same triple `solving_iterator.f_calc_solutions` holds, so a
  caller that used to take `f_calc_solutions[0]` can use this unchanged -- plus
  `trials`, a record of every run, and `n_trials_run`, which is less than
  `n_trials` when the search stopped early. `f_calc` is None if nothing solved.

  loop, when given, replaces `charge_flipping.loop` for driving one trial. Olex2
  passes its own so that the progress plot and the stop button keep working;
  it must accept (solving_iterator, verbose=, out=) and, if it returns a false
  value, is taken to mean the user asked to stop.

  callback(i_trial, n_trials, result) after each trial, for progress reporting.
  Returning False from it stops the search, so the same hook serves as a second
  route for cancellation.
  """
  import sys
  import time

  if out is None:
    out = sys.stdout
  if loop is None:
    loop = charge_flipping.loop

  best = None
  best_trial = None
  trials = []
  t_start = time.time()
  stopped_because = None

  # Non-random starting points, if any were supplied, are used first and the
  # remaining trials fall back to random seeds. Ordering them first matters:
  # the search stops early once a trial is good enough, so a better starting
  # point only pays if it is tried before the random ones.
  supplied = list(initial_phases_list or [])

  # Supplied starts are **added to** the random ones, not substituted for them.
  # Substituting was measured and cost 6 of 288 control structures that had
  # solved before: with 8 trials and 4 supplied starts only 4 random trials
  # remained, and four of the six losses still found *a* solution (cc 0.95-0.98)
  # -- they lost the particular random trial that had happened to work, not the
  # ability to solve. Adding keeps every random chance and layers the better
  # starting points on top.
  #
  # The extra trials are cheaper than they look: the search stops as soon as a
  # trial is good enough, and supplied starts go first precisely so that a good
  # one ends the run early.
  for i_trial in range(n_trials + len(supplied)):
    seed = first_seed + max(0, i_trial - len(supplied))
    t0 = time.time()

    flipping = charge_flipping.weak_reflection_improved_iterator(
      delta=None, weak_reflection_fraction=weak_reflection_fraction)
    solving = charge_flipping.solving_iterator(
      flipping, f_obs,
      normalisations_for=normalisations_for,
      initial_phases_for=(supplied[i_trial] if i_trial < len(supplied)
                          else phases_from_seed(seed)),
      yield_solving_interval=yield_solving_interval,
      max_solving_iterations=max_solving_iterations,
      # One try per trial. The retries the solving iterator would do on its own
      # are exactly what this loop is taking over, and leaving them switched on
      # would mean each trial silently doing up to 25 runs and reporting one.
      max_attempts_to_get_phase_transition=1,
      max_attempts_to_get_sharp_correlation_map=1)

    user_stopped = False
    error = None
    try:
      if loop(solving, verbose=verbose, out=out) is False:
        user_stopped = True
    except Exception as e:
      # One trial failing is not the run failing: seven others may yet solve
      # the structure, and a solution path that dies on the first awkward map
      # is worse than one that reports what went wrong at the end.
      error = "%s: %s" % (type(e).__name__, str(e)[:200])

    # Every solution the trial found, not just the first. The list is ordered
    # by correlation peak height, which is also the ranking key, so the head of
    # each list is what competes -- but the whole list is recorded, because
    # `charge_flipping` only appends solutions above `min_cc_peak_height` and
    # a trial that found several says something about its quality.
    solutions = list(getattr(solving, "f_calc_solutions", None) or [])
    # The unsymmetrised P1 structure factors this trial ended on. Every
    # solution in the list above has had the assumed space group imposed on it
    # by `f_calc_symmetrisations`, so this is the only form that still carries
    # what the data alone said -- which is what a symmetry search has to be
    # given if its answer is to mean anything. `clean_up` deletes the state
    # machine's generators but leaves the flipping iterator, so reading it here
    # is safe.
    f_calc_in_p1 = getattr(solving.flipping_iterator, "f_calc", None)
    result = trial_result(
      seed=seed,
      n_solutions=len(solutions),
      had_phase_transition=bool(getattr(solving, "had_phase_transition",
                                        False)),
      cc_peak_height=(solutions[0][2] if solutions else None),
      seconds=time.time() - t0,
      error=error,
      # Off by default: a caller only wants the winner, and holding every
      # trial's f_calc keeps n_trials miller arrays alive for no purpose. The
      # benchmark turns it on, so that it can ask what the best *available*
      # answer was and measure how much the ranking gave away.
      solutions=(solutions if keep_solutions else None),
      f_calc_in_p1=f_calc_in_p1)
    trials.append(result)

    for solution in solutions:
      if best is None or solution[2] > best[2]:
        best = solution
        best_trial = result

    if callback is not None and callback(i_trial, n_trials, result) is False:
      stopped_because = "cancelled"
    if user_stopped:
      stopped_because = "cancelled"
    # A trial this good is right about 95 times in 100 (measured), so spending
    # the remaining trials to confirm it is not worth the wait. Below that the
    # correlation peak height barely separates right from wrong at all, which
    # is why there is no lower bar: it would stop early on the wrong answer.
    if (best is not None and good_enough_cc_peak_height is not None
        and best[2] >= good_enough_cc_peak_height):
      stopped_because = "good_enough"
    if (max_seconds is not None and time.time() - t_start >= max_seconds
        and stopped_because is None):
      # A bound for the interactive case, where a structure that will not solve
      # must not hold the GUI for minutes. Deliberately checked after a whole
      # trial: interrupting one leaves nothing usable behind.
      stopped_because = "out_of_time"
    if stopped_because is not None:
      break

  return group_args(
    f_calc=(best[0] if best else None),
    shift=(best[1] if best else None),
    cc_peak_height=(best[2] if best else None),
    # The winning trial's P1 structure factors, for a symmetry search that must
    # not be handed a map the assumed space group has already been imposed on.
    f_calc_in_p1=(best_trial.f_calc_in_p1 if best_trial else None),
    best_seed=(best_trial.seed if best_trial else None),
    trials=trials,
    n_trials_run=len(trials),
    stopped_because=stopped_because,
    seconds=time.time() - t_start)


def show(result, out=None):
  """ One line per trial and a summary, for the log a user actually reads. """
  import sys
  if out is None:
    out = sys.stdout
  for t in result.trials:
    if t.error is not None:
      print("  trial seed=%i failed: %s" % (t.seed, t.error), file=out)
    elif t.n_solutions:
      print("  trial seed=%i: %i solution(s), best correlation %.3f (%.1fs)"
            % (t.seed, t.n_solutions, t.cc_peak_height, t.seconds), file=out)
    else:
      print("  trial seed=%i: no solution (%.1fs)" % (t.seed, t.seconds),
            file=out)
  if result.f_calc is None:
    print("No solution from %i trial(s) in %.1fs"
          % (result.n_trials_run, result.seconds), file=out)
  else:
    print("Best of %i trial(s): correlation %.3f in %.1fs%s"
          % (result.n_trials_run, result.cc_peak_height, result.seconds,
             "" if result.stopped_because is None
             else " (stopped: %s)" % result.stopped_because), file=out)
