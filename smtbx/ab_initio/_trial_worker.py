""" One worker process for `parallel_trials`. Not meant to be run by hand.

    <python> _trial_worker.py <job.pickle> <out.pickle> <first_seed> <n_trials>

The parent hands over its own `sys.path`, rather than the worker working out
where cctbx lives. Inside Olex2 the interpreter is embedded and libtbx's
environment is already configured; a worker that tried to rediscover it goes
looking for paths like `C:\\Python38\\Library\\share\\cctbx` that do not exist.
Whatever the parent is successfully importing from is by definition right.
"""
from __future__ import absolute_import, division, print_function

import sys


def main(argv):
  job_path, out_path, first_seed, n_trials = argv[0], argv[1], \
      int(argv[2]), int(argv[3])

  # Bootstrap the path before importing anything from cctbx.
  import pickle
  with open(job_path, "rb") as f:
    job = pickle.load(f)
  for entry in reversed(job["sys_path"]):
    if entry not in sys.path:
      sys.path.insert(0, entry)

  from smtbx.ab_initio import charge_flipping, multi_trial

  result = multi_trial.solve(
    job["f_obs"],
    n_trials=n_trials,
    first_seed=first_seed,
    weak_reflection_fraction=job["weak_reflection_fraction"],
    normalisations_for=charge_flipping.amplitude_quasi_normalisations,
    max_solving_iterations=job["max_solving_iterations"],
    # Each worker holds only its own best, so the early exit still applies
    # within a worker but cannot stop the others. That is deliberate: a worker
    # that finds a good solution should stop, and the rest are already running.
    good_enough_cc_peak_height=job["good_enough_cc_peak_height"],
    out=open(out_path + ".log", "w"))

  # Only what the parent needs. Whole trial records would drag every trial's
  # miller arrays through the pickle for no purpose.
  payload = dict(
    cc_peak_height=result.cc_peak_height,
    f_calc=result.f_calc,
    shift=result.shift,
    f_calc_in_p1=result.f_calc_in_p1,
    best_seed=result.best_seed,
    n_trials_run=result.n_trials_run)
  with open(out_path, "wb") as f:
    pickle.dump(payload, f, protocol=2)


if __name__ == "__main__":
  main(sys.argv[1:])
