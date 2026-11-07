""" Charge flipping run several times from different random starts, ranked.

`charge_flipping.solving_iterator` restarts itself when an attempt fails, but
throws every restart away and uses only the attempt that finally worked. Here
each attempt gets its own seed, exactly one try, and is kept as a scored
candidate; the best is returned.

Candidates are ranked by R1; the correlation peak height decides only whether
a trial counts as a solution and when the run may stop early. Chemical
plausibility of the peak list (the fraction of peaks with a bonded neighbour)
is not used: across many structures it does not correlate with correctness.
"""
from __future__ import absolute_import, division, print_function

import math
import os

from six.moves import cStringIO as StringIO

from libtbx import group_args
from smtbx.ab_initio import charge_flipping


def phases_from_seed(seed):
  """ A reproducible random starting phase set for one trial.

  A private generator rather than `flex.set_random_seed`, which seeds two
  process-global generators and would make the result depend on whatever else
  the process (a GUI) did beforehand.
  """
  import scitbx_array_family_flex_ext as flex_ext
  generator = flex_ext.mersenne_twister(seed=seed)
  return lambda f_obs: generator.random_double(size=f_obs.size(),
                                               factor=2*math.pi)


class trial_result(group_args):
  pass


class _abortable(object):
  """ `solving` whose iteration ends early once `abort()` is true """
  def __init__(self, solving, abort):
    self._solving, self._abort = solving, abort

  def __getattr__(self, name):
    return getattr(self._solving, name)

  def __iter__(self):
    for flipping in self._solving:
      if self._abort():
        return
      yield flipping


def solve(f_obs,
          n_trials=8,
          # the measured optimum, and the value Olex2 has always used
          weak_reflection_fraction=0.2,
          normalisations_for=charge_flipping.amplitude_quasi_normalisations,
          max_solving_iterations=500,
          # Tried in turn, each a set of overrides for this call, when no trial
          # reached a phase transition and time remains: the weak-reflection
          # step and the E normalisation each lose a few structures the other
          # keeps, and a structure whose trials transition one time in ten
          # wants more of them, longer; every round starts from fresh seeds.
          fallbacks=(# first: a map that flipped but whose symmetrisation peak
                     # stays just under 0.9 needs only a lower bar; the R
                     # ranking and the caller judge the model
                     dict(min_cc_peak_height=0.75),
                     dict(weak_reflection_fraction=0.3),
                     dict(normalisations_for=None),
                     dict(d_min=0.9, max_solving_iterations=2000),
                     dict(n_trials=16, max_solving_iterations=5000)),
          yield_solving_interval=60,
          good_enough_cc_peak_height=0.99,
          # a solution with a worse R than this may be a symmetric wrong one,
          # so it does not stop the trials early: the R ranking needs them all
          good_r1=0.3,
          # a best solution worse than this is as often a symmetric wrong one
          # as a poorly converged right one, so it earns the fallback rounds
          # too when a time budget bounds them; the lower-R result of the
          # rounds is returned, never nothing
          retry_r1=0.3,
          max_seconds=None,
          first_seed=1,
          initial_phases_list=None,
          keep_solutions=False,
          loop=None,
          callback=None,
          # trials on threads (the cctbx 3D FFTs release the GIL); 1 is the
          # serial path, None reads SMTBX_SOLVE_THREADS
          n_threads=None,
          # Polled by the main thread while trials run on threads; True
          # cuts the running trials short and abandons the rest.
          stop=None,
          verbose=False,
          out=None,
          # data cut to this resolution before solving: a fallback for a
          # structure whose full data set will not flip
          d_min=None,
          # the solving iterator's threshold for a symmetrisation peak to be
          # a solution; None keeps its own (0.9)
          min_cc_peak_height=None,
          # how far a fallback round's seeds move on from this round's; None
          # moves them past the trials just run. parallel_trials passes its
          # total trial count, so no worker reuses another worker's seeds
          fallback_seed_stride=None,
          # the first call's settings, so every fallback round overrides those
          # and not the previous round's
          _base=None):
  """ n_trials seeded charge-flipping runs; the best solution found.

  Returns a group_args with `f_calc`, `shift`, `cc_peak_height` and `r1` of the
  winner (the tuple `solving_iterator.f_calc_solutions` holds), `trials`, a
  record of every run, and `n_trials_run`. `f_calc` is None if nothing solved.

  loop, when given, replaces `charge_flipping.loop` for one trial (Olex2 passes
  its own for the progress plot and stop button); it takes (solving_iterator,
  verbose=, out=), and returning False means the user asked to stop.

  callback(i_trial, n_trials, result) runs after each trial; returning False
  stops the search.
  """
  import sys
  import time

  if out is None:
    out = sys.stdout
  if loop is None:
    loop = charge_flipping.loop
  if d_min is not None and d_min > f_obs.d_min():
    f_obs = f_obs.resolution_filter(d_min=d_min)

  if n_threads is None:
    try:
      n_threads = int(os.environ.get('SMTBX_SOLVE_THREADS', '1'))
    except ValueError:
      n_threads = 1
  best = None
  best_trial = None
  trials = []
  t_start = time.time()
  stopped_because = None

  # supplied starts are added to the random ones, not substituted, and go
  # first so that a good one ends the run early
  supplied = list(initial_phases_list or [])

  # one trial; shared by the serial and the threaded path so they cannot drift
  def _compute_trial(i_trial, trial_out, abort=None):
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
      # one try per trial: this loop takes over the iterator's own retries
      max_attempts_to_get_phase_transition=1,
      max_attempts_to_get_sharp_correlation_map=1,
      **({} if min_cc_peak_height is None
         else dict(min_cc_peak_height=min_cc_peak_height)))

    user_stopped = False
    error = None
    try:
      target = solving if abort is None else _abortable(solving, abort)
      if loop(target, verbose=verbose, out=trial_out) is False:
        user_stopped = True
    except Exception as e:
      # one trial failing is not the run failing; reported at the end
      error = "%s: %s" % (type(e).__name__, str(e)[:200])

    # every solution the trial found, not just the first
    solutions = list(getattr(solving, "f_calc_solutions", None) or [])
    # the unsymmetrised P1 structure factors, the only form without the
    # assumed space group imposed, which a symmetry search needs; `clean_up`
    # leaves the flipping iterator, so reading it here is safe
    f_calc_in_p1 = getattr(solving.flipping_iterator, "f_calc", None)
    result = trial_result(
      seed=seed,
      n_solutions=len(solutions),
      had_phase_transition=bool(getattr(solving, "had_phase_transition",
                                        False)),
      cc_peak_height=(solutions[0][2] if solutions else None),
      r1=(solutions[0][3] if solutions else None),
      seconds=time.time() - t0,
      error=error,
      # off by default, to not keep n_trials miller arrays alive; a benchmark
      # turns it on to judge the ranking
      solutions=(solutions if keep_solutions else None),
      f_calc_in_p1=f_calc_in_p1)
    return result, solutions, user_stopped

  # Charge flipping is almost all cctbx FFT, which releases the GIL, so trials
  # run in parallel on threads. The answer must equal the serial one: trials
  # are computed concurrently, then the bookkeeping below walks them in seed
  # order and stops exactly where serial would; trials past that point are
  # discarded.
  #
  # `max_seconds`, `stop` and a good-enough trial raise `halt` = (reason,
  # index): trials not yet started stay None, running ones past the index are
  # cut short, and the replay stops at the first None with that reason.
  # Trials are claimed in seed order, so everything before a None is complete.
  precomputed = None
  n_all = n_trials + len(supplied)
  halt = [None, n_all]
  if n_threads > 1 and n_all > 1:
    import threading
    slots = [None]*n_all
    bufs = [StringIO() for _ in range(n_all)]
    lock = threading.Lock()
    nxt = [0]

    def _worker():
      while True:
        with lock:
          i = nxt[0]
          if i >= n_all or halt[0] is not None:
            return
          nxt[0] = i + 1
        try:
          slots[i] = _compute_trial(i, bufs[i], abort=lambda: i > halt[1])
        except Exception as e:
          # one thread dying must not take the run down
          slots[i] = (trial_result(
            seed=first_seed + max(0, i - len(supplied)), n_solutions=0,
            had_phase_transition=False, cc_peak_height=None, r1=None, seconds=0.0,
            error='%s: %s' % (type(e).__name__, str(e)[:200]),
            solutions=None, f_calc_in_p1=None), [], False)
        cc, r1 = slots[i][0].cc_peak_height, slots[i][0].r1
        if slots[i][2]:
          halt[:] = "cancelled", -1
        elif (good_enough_cc_peak_height is not None and cc is not None
              and cc >= good_enough_cc_peak_height and r1 <= good_r1
              and i < halt[1]):
          halt[:] = "good_enough", i

    workers = [threading.Thread(target=_worker)
               for _ in range(min(n_threads, n_all))]
    for w in workers:
      w.start()
    while any(w.is_alive() for w in workers):
      for w in workers:
        w.join(0.1)
      if halt[0] is None:
        if stop is not None and stop():
          halt[:] = "cancelled", -1
        elif (max_seconds is not None
              and time.time() - t_start >= max_seconds):
          halt[0] = "out_of_time"
    # per-trial buffers replayed in order, so thread output never interleaves
    for b in bufs:
      out.write(b.getvalue())
    precomputed = slots

  for i_trial in range(n_trials + len(supplied)):
    if precomputed is not None:
      if precomputed[i_trial] is None:
        stopped_because = halt[0]
        break
      result, solutions, user_stopped = precomputed[i_trial]
    else:
      result, solutions, user_stopped = _compute_trial(i_trial, out)
    trials.append(result)

    for solution in solutions:
      # R separates right from wrong where the correlation peak does not
      if best is None or solution[3] < best[3]:
        best = solution
        best_trial = result

    if callback is not None and callback(i_trial, n_trials, result) is False:
      stopped_because = "cancelled"
    if user_stopped:
      stopped_because = "cancelled"
    # a trial this good is almost always right; below it the peak height
    # barely separates right from wrong, so there is no lower bar
    if (best is not None and good_enough_cc_peak_height is not None
        and best[2] >= good_enough_cc_peak_height and best[3] <= good_r1):
      stopped_because = "good_enough"
    if (max_seconds is not None and precomputed is None
        and time.time() - t_start >= max_seconds and stopped_because is None):
      # checked after a whole trial: interrupting one leaves nothing usable
      stopped_because = "out_of_time"
    if stopped_because is not None:
      break
  if precomputed is not None and stopped_because is None:
    stopped_because = halt[0]

  elapsed = time.time() - t_start
  result = group_args(
    f_calc=(best[0] if best else None),
    shift=(best[1] if best else None),
    cc_peak_height=(best[2] if best else None),
    r1=(best[3] if best else None),
    # unsymmetrised, for the symmetry search
    f_calc_in_p1=(best_trial.f_calc_in_p1 if best_trial else None),
    best_seed=(best_trial.seed if best_trial else None),
    trials=trials,
    n_trials_run=len(trials),
    stopped_because=stopped_because,
    seconds=time.time() - t_start)
  if (fallbacks and stopped_because is None
      and (best is None or (max_seconds is not None and best[3] > retry_r1))
      and (max_seconds is None or elapsed < 0.5*max_seconds)):
    if _base is None:
      _base = dict(n_trials=n_trials,
                   weak_reflection_fraction=weak_reflection_fraction,
                   normalisations_for=normalisations_for,
                   max_solving_iterations=max_solving_iterations, d_min=d_min,
                   min_cc_peak_height=min_cc_peak_height)
    kw = dict(_base,
              fallbacks=fallbacks[1:], _base=_base,
              yield_solving_interval=yield_solving_interval,
              good_enough_cc_peak_height=good_enough_cc_peak_height,
              good_r1=good_r1, retry_r1=retry_r1,
              max_seconds=(None if max_seconds is None
                           else max_seconds - elapsed),
              first_seed=first_seed + (len(trials) if fallback_seed_stride
                                       is None else fallback_seed_stride),
              fallback_seed_stride=fallback_seed_stride,
              initial_phases_list=None,
              keep_solutions=keep_solutions, loop=loop, callback=callback,
              n_threads=n_threads, stop=stop, verbose=verbose, out=out)
    kw.update(fallbacks[0])
    print("%s in %d trials; retrying with %s" % (
      "No solution" if best is None else "Best R %.3f" % best[3], len(trials),
          ", ".join("%s=%s" % (k, getattr(v, '__name__', v))
                    for k, v in fallbacks[0].items())), file=out)
    r = solve(f_obs, **kw)
    all_trials = trials + r.trials
    if best is not None and (r.f_calc is None or r.r1 > best[3]):
      r = result
    r.trials = all_trials
    r.n_trials_run = len(r.trials)
    r.seconds = time.time() - t_start
    return r
  return result


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
    print("Best of %i trial(s): correlation %.3f R %.3f in %.1fs%s"
          % (result.n_trials_run, result.cc_peak_height, result.r1,
             result.seconds,
             "" if result.stopped_because is None
             else " (stopped: %s)" % result.stopped_because), file=out)
