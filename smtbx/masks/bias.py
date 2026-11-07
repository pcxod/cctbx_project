"""Model bias correction for the solvent map, with nothing outside cctbx.

The void's difference map has phases from a model missing the void's content,
which can drive the void integral negative and discard it. sigma_A weighting
(Read, Acta Cryst. A42 (1986) 140) gives m*Fo - D*Fc in place of Fo - Fc.

smtbx.refinement.sigma_a needs mmtbx's compiled alpha_beta_est, which Olex2
does not ship; nothing here imports mmtbx. alpha and beta come from shell
moments, not a likelihood fit.
"""
from __future__ import absolute_import, division, print_function

from cctbx.array_family import flex
import scitbx.math


def n_bins_for(n_reflections, per_bin=500, lo=5, hi=25):
  return max(lo, min(hi, int(n_reflections//per_bin) or lo))


def alpha_beta(f_obs, f_calc, n_bins=None):
  """Per-reflection alpha and beta, by moments within resolution shells.

  In a shell <Fo^2> = alpha^2 <Fc^2> + beta, alpha the least-squares scale of
  |Fc| onto Fo. beta is a variance, floored above zero so a perfectly
  explained shell does not divide by zero downstream. Estimated on the data
  being corrected (no free set), so it under-corrects, the safe direction.
  """
  assert f_obs.indices().all_eq(f_calc.indices())
  if n_bins is None:
    n_bins = n_bins_for(f_obs.size())
  fo = f_obs.data()
  fc = flex.abs(f_calc.data())
  alpha = flex.double(fo.size(), 1.)
  beta = flex.double(fo.size(), 0.)
  work = f_obs.deep_copy()
  work.setup_binner(n_bins=n_bins)
  for i_bin in work.binner().range_used():
    sel = work.binner().selection(i_bin)
    if sel.count(True) < 3:
      continue
    o = fo.select(sel)
    c = fc.select(sel)
    cc = flex.sum(c*c)
    a = flex.sum(o*c)/cc if cc > 0 else 1.
    b = flex.mean(o*o) - a*a*flex.mean(c*c)
    floor = 1e-6*max(flex.mean(o*o), 1e-10)
    alpha.set_selected(sel, max(a, 1e-6))
    beta.set_selected(sel, max(b, floor))
  return alpha, beta


def figure_of_merit(f_obs, f_calc, alpha, beta):
  """m, the expected cosine of the phase error, per reflection.

  I1/I0 for acentric, tanh(x/2) for centric reflections, as in mlf.h.
  """
  fo = f_obs.data()
  fc = flex.abs(f_calc.data())
  eps = f_obs.epsilons().data().as_double()
  centric = f_obs.centric_flags().data()
  x = 2*alpha*fo*fc/(eps*beta)
  m = scitbx.math.bessel_i1_over_i0(x)
  if centric.count(True):
    half = x.select(centric)/2
    m.set_selected(centric, flex.tanh(half))
  return m


def difference_coefficients(f_obs, f_calc, n_bins=None):
  """m*Fo - D*Fc, as a complex miller array.

  Drop-in for f_obs.f_obs_minus_f_calc(1/scale, f_calc). f_obs may be real
  amplitudes or the complex array phase_transfer returns; when complex, its
  modulus is Fo and its (combined) phases are kept instead of Fc's.
  """
  phase_source = f_calc
  if isinstance(f_obs.data(), flex.complex_double):
    phase_source = f_obs
    f_obs = f_obs.customized_copy(
      data=flex.abs(f_obs.data()), sigmas=None).set_observation_type(None)
  alpha, beta = alpha_beta(f_obs, f_calc, n_bins=n_bins)
  m = figure_of_merit(f_obs, f_calc, alpha, beta)
  amplitude = m*f_obs.data() - alpha*flex.abs(f_calc.data())
  fc = phase_source.data()
  mod = flex.abs(fc)
  # unit vector along Fc, and zero where Fc has no phase to give
  out = flex.complex_double(fc.size(), 0)
  sel = (mod > 1e-30).iselection()
  # flex has no complex/double division, so scale by the reciprocal built as
  # a complex array
  scale = flex.complex_double(amplitude.select(sel)/mod.select(sel),
                              flex.double(sel.size(), 0))
  out.set_selected(sel, fc.select(sel)*scale)
  return f_calc.customized_copy(data=out)
