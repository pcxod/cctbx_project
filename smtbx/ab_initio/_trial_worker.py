""" One worker process for `parallel_trials`. Not meant to be run by hand.

    <python> _trial_worker.py <job.pickle> <out.pickle> <first_seed> <n_trials>

The parent hands over its own `sys.path`: inside Olex2 the interpreter is
embedded, and a worker rediscovering the libtbx environment looks for paths
that do not exist.
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
    # the early exit stops this worker only
    good_enough_cc_peak_height=job["good_enough_cc_peak_height"],
    # fallback rounds step by the total, not by n_trials, which would land on
    # the next worker's seeds; this gives the seeds the serial run would use
    fallback_seed_stride=job.get("n_trials_total"),
    out=open(out_path + ".log", "w"))

  # only what the parent needs, not every trial's miller arrays
  payload = dict(
    cc_peak_height=result.cc_peak_height,
    # multi_trial ranks by R1, and so must the parent across workers
    r1=result.r1,
    f_calc=result.f_calc,
    shift=result.shift,
    f_calc_in_p1=result.f_calc_in_p1,
    best_seed=result.best_seed,
    n_trials_run=result.n_trials_run)
  with open(out_path, "wb") as f:
    pickle.dump(payload, f, protocol=2)


if __name__ == "__main__":
  main(sys.argv[1:])
