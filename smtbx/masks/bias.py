"""Model bias correction for the solvent map, with nothing outside cctbx.

The difference map that fills the void is made with phases from a model that
is missing the very atoms the void contains. The bias that produces is not
small: on the synthetic cases it drives the integrated density in the void
negative, so the whole void is discarded and the solvent is lost entirely.

The standard cure is to weight the coefficients by how much the model can be
trusted at each resolution - sigma_A, from Read, Acta Cryst. A42 (1986) 140 -
giving m*Fo - D*Fc in place of Fo - Fc.

smtbx.refinement.sigma_a does this already but reaches into mmtbx's compiled
alpha_beta_est, which Olex2 does not ship. Everything needed is in cctbx: the
binner, centric flags and epsilons off the miller array, and i1/i0 from
scitbx.math. Nothing here imports mmtbx.

alpha and beta come from the moments of each shell rather than from a
likelihood maximisation, which is the cheap estimator and is what a difference
map needs; the free-set caveat below is the price.
"""
from __future__ import absolute_import, division, print_function

import math

from cctbx.array_family import flex
import scitbx.math


def n_bins_for(n_reflections, per_bin=500, lo=5, hi=25):
  return max(lo, min(hi, int(n_reflections//per_bin) or lo))


def alpha_beta(f_obs, f_calc, n_bins=None):
  """Per-reflection alpha and beta, by moments within resolution shells.

  In a shell <Fo^2> = alpha^2 <Fc^2> + beta, and alpha is the scale that best
  maps |Fc| onto Fo, so both follow from sums over the shell. beta is floored
  at a small positive number: it is a variance, and a shell where the model
  explains everything would otherwise divide by zero downstream.

  **Estimated on the data being corrected**, there being no free set here. That
  biases alpha up and beta down - the model looks better than it is - so this
  under-corrects rather than over-corrects, which is the safe direction for a
  map whose job is to reveal something the model does not have.
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

  Rice for the acentric reflections and the hyperbolic tangent for the
  centric ones, which is the same distinction mlf.h makes.
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

  Drop-in for f_obs.f_obs_minus_f_calc(1/scale, f_calc): same set, same
  phases, coefficients that know how far the model can be trusted.

  f_obs may be real amplitudes or the complex array phase_transfer returns -
  the mask calls it both ways, once before its loop and once inside. When it
  is complex those are the combined phases and they are the ones to keep, so
  the modulus becomes Fo and the phase is taken from it rather than from Fc.
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


def solvent_electrons_from_f_mask(f_mask, n_shells=6, d_min_fit=None):
  """Electrons in the solvent region, by extrapolating f_mask to (000).

  The electron count of the solvent is its forward scattering, f_mask(000),
  and that reflection is never measured. Integrating the map instead makes the
  answer hostage to the map's mean level, which is exactly the quantity the
  missing F(000) leaves undetermined - on a void filling 41% of the cell the
  pedestal outweighs the solvent twentyfold.

  A solvent region is a compact, smooth, positive blob, so |f_mask| falls off
  smoothly from its value at the origin. Fitting ln|f_mask| against
  d*^2 over the lowest shells and extrapolating to d*^2 = 0 estimates that
  value directly, using the reflections where the solvent actually scatters
  and nothing else.

  Returns None when there are too few low-angle reflections to fit, rather
  than a number nobody should trust.
  """
  fm = f_mask
  if d_min_fit is not None:
    fm = fm.resolution_filter(d_min=d_min_fit)
  if fm.size() < 3*n_shells:
    return None
  work = fm.customized_copy(data=flex.abs(fm.data()), sigmas=None)
  work.setup_binner(n_bins=max(n_shells, 3))
  xs, ys = flex.double(), flex.double()
  for i_bin in list(work.binner().range_used())[:n_shells]:
    sel = work.binner().selection(i_bin)
    if sel.count(True) < 3:
      continue
    mean = flex.mean(work.data().select(sel))
    if mean <= 0:
      continue
    d_star_sq = flex.mean(work.d_star_sq().data().select(sel))
    xs.append(d_star_sq)
    ys.append(math.log(mean))
  if xs.size() < 2:
    return None
  # straight line through ln|f_mask| against d*^2; the intercept is ln f(000)
  n = xs.size()
  sx, sy = flex.sum(xs), flex.sum(ys)
  sxx, sxy = flex.sum(xs*xs), flex.sum(xs*ys)
  denom = n*sxx - sx*sx
  if abs(denom) < 1e-30:
    return None
  slope = (n*sxy - sx*sy)/denom
  intercept = (sy - slope*sx)/n
  return math.exp(intercept)
