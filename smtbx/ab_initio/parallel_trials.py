""" Charge-flipping trials spread over several processes.

For an interpreter that cannot use multi_trial.solve(n_threads=...).
`multiprocessing` cannot be used inside Olex2: on Windows its spawn method
re-executes `sys.executable`, which is `olex2.exe`, and the `pyl.exe` launcher
Olex2 ships does not accept `-c`. So each worker is a `pyl.exe` running
`_trial_worker.py` over a contiguous block of seeds (process start-up costs
more than a trial), the parent passes its own `sys.path`, and picks the best
result. Falls back to running in-process on any failure.
"""
from __future__ import absolute_import, division, print_function

import os

from libtbx import group_args


def default_python_executable():
  """ `pyl.exe` next to the running Olex2, or None.

  Deliberately not `sys.executable`: inside Olex2 that is `olex2.exe`, and
  launching it would start a second copy of the program.
  """
  import sys

  candidates = []
  exe_dir = os.path.dirname(os.path.abspath(sys.executable or ""))
  if exe_dir:
    candidates.append(os.path.join(exe_dir, "pyl.exe"))
  for var in ("OLEX2_DIR", "OLEX2_DATADIR"):
    root = os.environ.get(var)
    if root:
      candidates.append(os.path.join(root, "pyl.exe"))
  for path in candidates:
    if os.path.isfile(path):
      return path
  return None


def solve(f_obs,
          n_trials=8,
          n_workers=None,
          weak_reflection_fraction=0.2,
          max_solving_iterations=500,
          good_enough_cc_peak_height=0.99,
          python_executable=None,
          timeout_seconds=600,
          out=None):
  """ Run `n_trials` charge-flipping trials across processes; best result.

  Returns the fields of `multi_trial.solve` a caller uses -- `f_calc`, `shift`,
  `cc_peak_height`, `r1`, `f_calc_in_p1`, `best_seed` -- plus `n_workers_used`
  and `fell_back`.
  """
  import pickle
  import shutil
  import subprocess
  import sys
  import tempfile
  import time

  if out is None:
    out = sys.stdout
  t0 = time.time()

  if python_executable is None:
    python_executable = default_python_executable()
  if n_workers is None:
    # leave a core for the GUI
    try:
      n_workers = max(1, (os.cpu_count() or 2) - 1)
    except Exception:
      n_workers = 2
  n_workers = max(1, min(n_workers, n_trials))

  if python_executable is None or n_workers == 1:
    return _in_process(f_obs, n_trials, weak_reflection_fraction,
                       max_solving_iterations, good_enough_cc_peak_height,
                       out, reason=("no pyl.exe found" if not python_executable
                                    else "single worker"))

  work_dir = tempfile.mkdtemp(prefix="cf_trials_")
  try:
    job_path = os.path.join(work_dir, "job.pickle")
    with open(job_path, "wb") as f:
      pickle.dump(dict(
        f_obs=f_obs,
        sys_path=list(sys.path),
        weak_reflection_fraction=weak_reflection_fraction,
        max_solving_iterations=max_solving_iterations,
        good_enough_cc_peak_height=good_enough_cc_peak_height,
        n_trials_total=n_trials), f, protocol=2)

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_trial_worker.py")
    procs, outputs = [], []
    seed = 1
    for i in range(n_workers):
      # the remainder spread over the first workers
      count = n_trials//n_workers + (1 if i < n_trials % n_workers else 0)
      if count == 0:
        continue
      out_path = os.path.join(work_dir, "out_%i.pickle" % i)
      cmd = [python_executable, worker, job_path, out_path,
             str(seed), str(count)]
      procs.append(subprocess.Popen(cmd, cwd=work_dir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT))
      outputs.append(out_path)
      seed += count

    deadline = time.time() + timeout_seconds
    for p in procs:
      remaining = max(1, int(deadline - time.time()))
      try:
        p.communicate(timeout=remaining)
      except Exception:
        p.kill()

    best = None
    for path in outputs:
      if not os.path.exists(path):
        continue
      try:
        with open(path, "rb") as f:
          payload = pickle.load(f)
      except Exception:
        continue
      if payload.get("cc_peak_height") is None:
        continue
      # lowest R1, as multi_trial.solve ranks (the peak height does not tell a
      # symmetric wrong solution from the right one); the peak height breaks
      # ties, a worker without an R1 ranks last
      def key(p):
        r1 = p.get("r1")
        return (r1 is None, r1 if r1 is not None else 0,
                -p["cc_peak_height"])
      if best is None or key(payload) < key(best):
        best = payload

    if best is None:
      # every worker failed or found nothing: a worker failure is not
      # "no solution", so redo the work in process
      return _in_process(f_obs, n_trials, weak_reflection_fraction,
                         max_solving_iterations, good_enough_cc_peak_height,
                         out, reason="all workers returned nothing")

    print("Solved across %i processes in %.1f s"
          % (len(procs), time.time() - t0), file=out)
    return group_args(
      f_calc=best.get("f_calc"),
      shift=best.get("shift"),
      cc_peak_height=best.get("cc_peak_height"),
      r1=best.get("r1"),
      f_calc_in_p1=best.get("f_calc_in_p1"),
      best_seed=best.get("best_seed"),
      n_workers_used=len(procs),
      fell_back=False,
      seconds=time.time() - t0)
  except Exception as e:
    return _in_process(f_obs, n_trials, weak_reflection_fraction,
                       max_solving_iterations, good_enough_cc_peak_height,
                       out, reason="%s: %s" % (type(e).__name__, e))
  finally:
    shutil.rmtree(work_dir, ignore_errors=True)


def _in_process(f_obs, n_trials, weak_reflection_fraction,
                max_solving_iterations, good_enough_cc_peak_height, out,
                reason):
  import time
  from smtbx.ab_initio import charge_flipping, multi_trial

  print("Running trials in this process (%s)." % reason, file=out)
  t0 = time.time()
  result = multi_trial.solve(
    f_obs, n_trials=n_trials,
    weak_reflection_fraction=weak_reflection_fraction,
    normalisations_for=charge_flipping.amplitude_quasi_normalisations,
    max_solving_iterations=max_solving_iterations,
    good_enough_cc_peak_height=good_enough_cc_peak_height,
    out=out)
  return group_args(
    f_calc=result.f_calc, shift=result.shift,
    cc_peak_height=result.cc_peak_height, r1=result.r1,
    f_calc_in_p1=result.f_calc_in_p1, best_seed=result.best_seed,
    n_workers_used=1, fell_back=True, seconds=time.time() - t0)
