""" alpha and beta (sigma_A) for maximum-likelihood refinement

The Lunin-Skovoroda estimator is the C++ of mmtbx/max_lik, built into
mmtbx_max_lik_ext by smtbx/max_lik/SConscript without the mmtbx module being
configured. The binning, smoothing and interpolation around it, which mmtbx
keeps in max_lik/maxlik.py, are carried here: the bundle copies the Python
sources of configured modules only, so an import of mmtbx.max_lik fails in
every deployed smtbx even though the extension ships. Returns alpha and beta
per reflection, ordered as the observations are.

The extension is imported lazily, so smtbx imports without it and a caller
requesting maximum likelihood where it is missing receives a single
diagnostic rather than an ImportError at module load.

References:
 - Lunin & Skovoroda (1995) Acta Cryst. A51, 880-887.
 - Afonine, Lunin & Urzhumtsev (2003) J. Appl. Cryst. 36, 158-159.
 - Read (1986) Acta Cryst. A42, 140-149.
"""
from __future__ import absolute_import, division, print_function

from cctbx.array_family import flex


class missing_estimator(RuntimeError):
  pass


def _ext():
  import boost_adaptbx.boost.python as bp
  ext = bp.import_ext("mmtbx_max_lik_ext", optional=True)
  if ext is None:
    raise missing_estimator(
      "maximum-likelihood refinement needs the alpha/beta estimator "
      "mmtbx_max_lik_ext, which this cctbx was built without (see "
      "smtbx/max_lik/SConscript). Least-squares refinement is unaffected.")
  return ext


def _smooth(x):
  """ maxlik.alpha_beta_est_manager.smooth, kept bit for bit: a running mean
  of three that reads the already-smoothed left neighbour, then every value
  below 0.01 replaced by its left neighbour - the last bin's for the first,
  through Python's x[-1]. """
  if len(x) > 1:
    x1, x2 = x[0], x[1]
    for i in range(1, len(x)-1):
      x3 = x[i+1]
      x[i] = (x1+x2+x3)/3.0
      x1, x2 = x2, x3
    for i in range(len(x)):
      if x[i] < 0.01:
        x[i] = x[i-1]
  return x


def _alpha_beta_est(f_obs, f_calc, flags, free_reflections_per_bin,
                    interpolation, epsilons):
  """ maxlik.alpha_beta_est_manager: alpha and beta per resolution zone from
  the test set (the work set if there is none), then per reflection. """
  n_free = flags.count(True)
  if n_free == 0:
    flags = ~flags
  elif free_reflections_per_bin > n_free:
    free_reflections_per_bin = n_free
  fo_test = f_obs.select(flags)
  fc_test = f_calc.select(flags)
  eps_test = epsilons.select(flags)
  fo_test.setup_binner_counting_sorted(
    reflections_per_bin=free_reflections_per_bin)
  fo_sets, fm_sets, index_sets, eps_sets = [], [], [], []
  for i_bin in fo_test.binner().range_used():
    sel = fo_test.binner().selection(i_bin)
    if sel.count(True) > 0:
      fo_sets.append(fo_test.select(sel).data())
      fm_sets.append(fc_test.select(sel).data())
      index_sets.append(fo_test.select(sel).indices())
      eps_sets.append(eps_test.select(sel))
  est = _ext().alpha_beta_est(fo_test=fo_sets, fm_test=fm_sets,
    indices=index_sets, epsilons=eps_sets, space_group=fo_test.space_group())
  alpha_zones, beta_zones = est.alpha(), est.beta()
  f_obs.setup_binner(n_bins=len(alpha_zones))
  binner = f_obs.binner()
  if interpolation:
    return (binner.interpolate(flex.double(_smooth(alpha_zones)), 0),
            binner.interpolate(flex.double(_smooth(beta_zones)), 0))
  alpha = flex.double(f_obs.size())
  beta = flex.double(f_obs.size())
  for i_bin, a, b in zip(binner.range_used(), alpha_zones, beta_zones):
    sel = binner.selection(i_bin)
    alpha.set_selected(sel, a)
    beta.set_selected(sel, b)
  return alpha, beta


def alpha_beta(f_obs, f_calc, r_free_flags,
               free_reflections_per_bin=140,
               interpolation=True,
               add_sigma_squared_to_beta=False):
  """ Per-reflection alpha and beta, as two flex.double aligned with f_obs.

  `f_obs` and `f_calc` are miller arrays over the same indices; `f_calc` may be
  complex and is reduced to amplitudes here. `r_free_flags` is a flex.bool with
  True for the test set.

  `add_sigma_squared_to_beta` folds the experimental variance into beta, the
  convention Refmac uses. Without it the maximum-likelihood weight
  2*alpha^2/(epsilon*beta) depends only on the resolution shell, so every
  reflection in a shell is weighted alike and the measured sigmas play no part
  at all. Leave it off for the intensity target, which models the experimental
  error explicitly by convolution and would otherwise count it twice.
  """
  _ext()
  assert f_obs.indices().size() == f_calc.indices().size()
  assert r_free_flags.size() == f_obs.indices().size()
  assert f_calc.indices().all_eq(f_obs.indices())
  a, b = _alpha_beta_est(
    f_obs=f_obs,
    f_calc=abs(f_calc),
    flags=r_free_flags,
    free_reflections_per_bin=free_reflections_per_bin,
    interpolation=interpolation,
    epsilons=f_obs.epsilons().data().as_double())
  if add_sigma_squared_to_beta and f_obs.sigmas() is not None:
    b += f_obs.sigmas()*f_obs.sigmas()
  return a, b


def centric_flags_and_epsilons(f_obs):
  """ The other two per-reflection quantities the targets need.

  Both come from the symmetry rather than from the data, and both are asked for
  by every likelihood target, so they are gathered in one place instead of at
  each call site.
  """
  return (f_obs.centric_flags().data(),
          f_obs.epsilons().data().as_double())


def deterministic_free_flags(f_obs, fraction=0.1):
  """ A test set derived from the data, reproducible without being stored

  The set is a pure function of the Miller indices, the unit cell and the space
  group, so the same data always yields the same free reflections and no flag
  column need be written. It follows that different reflections give a
  different set, as a test set carried over onto other data would not be one.

  The global flex random generator is not used: seeding it would affect every
  other consumer of randomness in the process, and the current seed cannot be
  read back to restore it. Hashing each index requires no state.

  f_obs is a unique set in the asymmetric unit, so symmetry-related reflections
  cannot be split between the work and test sets.
  """
  assert 0 < fraction < 1
  uc = f_obs.unit_cell().parameters()
  # a seed from the crystal rather than from the clock
  seed = hash((f_obs.space_group().type().number(),
               tuple(round(p, 4) for p in uc),
               f_obs.indices().size())) & 0x7fffffff
  flags = flex.bool(f_obs.indices().size(), False)
  for i, h in enumerate(f_obs.indices()):
    # a cheap integer mix; the constants are the usual odd multipliers
    x = (h[0]*73856093) ^ (h[1]*19349663) ^ (h[2]*83492791) ^ seed
    x &= 0xffffffff
    x = (x ^ (x >> 16))*0x45d9f3b & 0xffffffff
    x = (x ^ (x >> 16))*0x45d9f3b & 0xffffffff
    x = x ^ (x >> 16)
    flags[i] = ((x % 1000000)/1000000.0) < fraction
  return flags


def is_available():
  """ Whether alpha/beta can be estimated at all in this installation. """
  try:
    _ext()
  except missing_estimator:
    return False
  return True
