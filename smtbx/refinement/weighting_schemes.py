""" Weighting schemes.

They all feature:

  o __str__: a string representation of the weighting scheme in a format
             that is appropriate for the CIF item _refine_ls_weighting_details;

  o type: the kind of weighting scheme,
          that is appropriate for the CIF item _refine_ls_weighting_scheme
"""


from __future__ import absolute_import, division, print_function
import boost_adaptbx.boost.python as bp
from six.moves import range
ext = bp.import_ext("smtbx_refinement_least_squares_ext")


from cctbx.array_family import flex

import math
import numpy as np
import scipy.optimize as optimize


@bp.inject_into(ext.mainstream_shelx_weighting)
class _():

  def __str__(self):
    if round(self.a, 4) in (0.1, 0.2):
      a = "%.1f" %self.a
    else:
      a = "%.4f" %self.a
    if round(self.b, 4) == 0: b_part=""
    else: b_part = "+%.4fP" %self.b
    return (r"w=1/[\s^2^(Fo^2^)+(%sP)^2^%s]"
            " where P=(Fo^2^+2Fc^2^)/3" %(a, b_part))

  def type(self):
    return "calc"

  def optimise_parameters(self, fo_sq, fc_sq,
                          scale_factor, n_independent_params):
    """ Find optimal values of a and b that give a flat analysis of the variance
        when binned by Fc/max(Fc), and a goodness of fit close to 1.

        This is done in a grid search fashion similar to Shelxl.

        self is not modified in place; instead a new instance of the weighting
        scheme is returned.

        It is intended that f_calc should already contain the contribution from
        f_mask (if a solvent mask is used).
    """
    assert fc_sq.is_xray_intensity_array()
    weighting = ext.mainstream_shelx_weighting(a=self.a, b=self.b)

    fo_sq = fo_sq.deep_copy()
    fo_sq.data().set_selected(fo_sq.data() < 0, 0)

    fo2 = fo_sq.data().deep_copy()
    fo2 /= scale_factor
    sigmas = fo_sq.sigmas() / scale_factor
    sigmas_sq = flex.pow2(sigmas)
    fc2 = fc_sq.data()

    # determine starting values for a and b, formulae taken from shelxl code
    p = (fo2 + 2. * fc2)/3.
    p_sq = flex.pow2(p)
    x = flex.sum((flex.pow2(fo2-fc2)-sigmas) * (p_sq/sigmas_sq))
    y = flex.sum( flex.pow2(p_sq)/sigmas_sq)
    z = flex.sum(p)
    start_a = math.sqrt(max(0.0001, 0.64 * x / max(1e-8, y)))
    start_b = 0.5 * z * start_a**2 / fo_sq.size()
    a_step = 0.2 * start_a
    b_step = 0.4 * start_b

    # sort data and setup binning by fc/fc_max
    permutation = flex.sort_permutation(fc_sq.data()/flex.max(fc_sq.data()))
    k = scale_factor
    fo = fo_sq.data().select(permutation).as_numpy_array()
    s = fo_sq.sigmas().select(permutation).as_numpy_array()
    fc = fc_sq.data().select(permutation).as_numpy_array()
    n_bins = 10
    bin_limits = [0] + [int(math.ceil((i+1) * fo.size/n_bins)) for i in range(n_bins)]
    bin_count = [bin_limits[i+1] - bin_limits[i] for i in range(n_bins)]
    # per bin what does not depend on a and b: P, sigma^2 and (fo - k fc)^2
    bins = []
    for i in range(n_bins):
      lo, hi = bin_limits[i], bin_limits[i+1]
      pb = (np.maximum(fo[lo:hi], 0.) + 2 * k * fc[lo:hi]) / 3.
      bins.append((pb, s[lo:hi] * s[lo:hi], np.square(fo[lo:hi] - k * fc[lo:hi])))

    n = fo.size//(fo.size-n_independent_params)

    # search on a 9x9 grid to determine best values of a and b, each bin as one
    # numpy block with the operations of weighting_schemes.h compute() in the
    # same order and a sequential sum (cumsum, as flex.sum, not numpy's
    # pairwise sum), so the same a and b bit for bit as one C++ call per point
    while (a_step > 1e-4 and b_step > 5e-3):
      start_a = max(start_a, 4*a_step) - 4*a_step
      start_b = max(start_b, 4*b_step) - 4*b_step
      av, bv, a, b = [], [], start_a, start_b
      for j in range(9):
        a += a_step
        b += b_step
        av.append(a)
        bv.append(b)
      av = np.array(av)[:, None, None]
      bkv = np.array([bb * k for bb in bv])[None, :, None]
      binned_chi_sq = []
      for i_bin, (pb, ssq, d2) in enumerate(bins):
        if bin_count[i_bin] == 0:
          binned_chi_sq.append(np.zeros((9, 9)))
          continue
        # a bin at a time, 256 reflections at a time: a whole large bin's
        # 9 x 9 x n blocks fall out of cache, and at ~21k reflections that
        # made this slower than the per-point C++ calls it replaced. Each
        # chunk's first term takes the running sum, so the summation stays
        # sequential and the result is the same bit for bit.
        acc = None
        for lo in range(0, pb.size, 256):
          c_pb = pb[lo:lo+256]
          w = 1. / ((ssq[lo:lo+256] + np.square(av * c_pb)) + bkv * c_pb)
          wd2 = w * d2[lo:lo+256]
          if acc is not None:
            wd2[:, :, 0] += acc
          acc = np.cumsum(wd2, axis=2)[:, :, -1]
        binned_chi_sq.append(acc.T)
      min_variance = 9e9
      j_min, k_min = (0, 0)
      for j in range(9):
        for kk in range(9):
          variance = 0
          for i_bin in range(n_bins):
            if bin_count[i_bin] == 0: continue
            goof = math.sqrt(binned_chi_sq[i_bin][j,kk]*n/bin_count[i_bin])
            variance += (goof-1)**2
          min_variance = min(variance, min_variance)
          if variance == min_variance:
            j_min = j
            k_min = kk
      start_a += k_min*a_step
      start_b += j_min*b_step
      if k_min == 8:
        a_step *= 2
        continue
      elif k_min != 0:
        a_step /= 4
      if j_min == 8:
        b_step *= 2
        continue
      elif j_min != 0:
        b_step /=4
      if start_a <= 1e-4: a_step /= 4
      if start_b <= 1e-3: b_step /= 4
    if start_a > 0.2:
      start_a = 0.2
      start_b = 0
    weighting.a = start_a
    weighting.b = start_b
    return weighting

@bp.inject_into(ext.new_shelx_weighting)
class _():

  def __str__(self):
    if round(self.a, 4) in (0.1, 0.2):
      a = "%.1f" %self.a
    else:
      a = "%.4f" %self.a
    if round(self.b, 4) == 0: b_part=""
    else: b_part = "+%.4fP" %self.b
    return (r"w=1/[\s^2^(Fo^2^)+(%sP)^2^%s]"
            " where P=(Fo^2^+2Fc^2^)/3" %(a, b_part))

  def type(self):
    return "calc"

  def optimise_parameters(self, fo_sq, fc_sq,
                              scale_factor, n_independent_params):
    """ Find optimal values of a and b that give a flat analysis of the variance
        when binned by Fc/max(Fc), and a goodness of fit close to 1.

        This is done in a bin search using scipy minimize as proposed by Martin Lutz.

        self is not modified in place; instead a new instance of the weighting
        scheme is returned.

        It is intended that f_calc should already contain the contribution from
        f_mask (if a solvent mask is used).
    """
    assert fc_sq.is_xray_intensity_array()
    weighting = ext.mainstream_shelx_weighting(a=self.a, b=self.b)

    def new_shelx_weighting_compute_chi_sq(fo_sq, fc_sq, a,b):
      weighting.a = a
      weighting.b = b
      weights = weighting(
        fo_sq.data(), fo_sq.sigmas(), fc_sq.data(), fo_sq.indices(), scale_factor)
      return (flex.sum(
        weights * flex.pow2(fo_sq.data() - scale_factor * fc_sq.data())))

    fo_sq = fo_sq.deep_copy()
    fo_sq.data().set_selected(fo_sq.data() < 0, 0)

    fo2 = fo_sq.data().deep_copy()
    fo2 /= scale_factor
    sigmas = fo_sq.sigmas() / scale_factor
    sigmas_sq = flex.pow2(sigmas)
    fc2 = fc_sq.data()

    # determine starting values for a and b, formulae taken from shelxl code
    p = (fo2 + 2. * fc2)/3.
    p_sq = flex.pow2(p)
    x = flex.sum((flex.pow2(fo2-fc2)-sigmas) * (p_sq/sigmas_sq))
    y = flex.sum( flex.pow2(p_sq)/sigmas_sq)
    z = flex.sum(p)
    start_a = math.sqrt(max(0.0001, 0.64 * x / max(1e-8, y)))
    start_b = 0.5 * z * start_a**2 / fo_sq.size()

    # sort data and setup binning by fc/fc_max
    fc_sq_over_fc_sq_max = fc_sq.data()/flex.max(fc_sq.data())
    permutation = flex.sort_permutation(fc_sq_over_fc_sq_max)
    fc_sq_over_fc_sq_max = fc_sq.customized_copy(
      data=fc_sq_over_fc_sq_max).select(permutation)
    fc_sq = fc_sq.select(permutation)
    fo_sq = fo_sq.select(permutation)
    n_bins = 10
    bin_limits = flex.size_t(1, 0)
    bin_count = flex.size_t()
    for i in range(n_bins):
      bin_limits.append(int(math.ceil((i+1) * fc_sq.size()/n_bins)))
      bin_count.append(bin_limits[i+1] - bin_limits[i])

    n = fo_sq.size()//(fo_sq.size()-n_independent_params)

    def calcres(w):
      averages = [flex.double() for i in range(n_bins)]
      for i_bin in range(n_bins):
        sel = flex.size_t_range(bin_limits[i_bin], bin_limits[i_bin+1])
        fc2 = fc_sq.select(sel)
        fo2 = fo_sq.select(sel)
        binned_chi_sq = new_shelx_weighting_compute_chi_sq(fo2, fc2, w[0], w[1])
        averages[i_bin] = math.sqrt(binned_chi_sq*n/bin_count[i_bin])
      residual = np.sum((np.array(averages)-1.0)**2)
      return residual

    # Use scipy minimize to find optimal a and b
    result = optimize.minimize(calcres, [start_a, start_b], method='SLSQP', bounds=((0,None),(0,None)))
    if result.success:
      weighting.a = result.x[0]
      weighting.b = result.x[1]
    # THIS IS A TEST TO SEE IF DIFFERENTIAL EVOLUTION WORKS BETTER
    #a_min = 0.0, a_max = 1.0
    #b_min = 0.0, b_max = 100.0
    #de_result = optimize.differential_evolution(
    #  calcres,
    #  bounds=[(a_min, a_max), (b_min, b_max)],
    #  maxiter=1000, popsize=15, tol=1e-6, polish=False
    #)
    #if de_result.success:
    #  # local refinement from best DE solution
    #  local = optimize.minimize(calcres, de_result.x, method='Nelder-Mead',
    #                            options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':2000})
    #  best = local.x if local.success else de_result.x
    #  weighting.a, weighting.b = best[0], best[1]
    else:
      #If it fails using the new routine fallback to Shelxl style grid search
      print("Warning: optimisation of weighting parameters failed, falling back to old method.")
      weighting = self.optimise_parameters_old(fo_sq, fc_sq,
                                               scale_factor, n_independent_params)
    return weighting

  def optimise_parameters_old(self, fo_sq, fc_sq,
                          scale_factor, n_independent_params):
    """ Find optimal values of a and b that give a flat analysis of the variance
        when binned by Fc/max(Fc), and a goodness of fit close to 1.

        This is done in a grid search fashion similar to Shelxl.

        self is not modified in place; instead a new instance of the weighting
        scheme is returned.

        It is intended that f_calc should already contain the contribution from
        f_mask (if a solvent mask is used).
    """
    assert fc_sq.is_xray_intensity_array()
    weighting = ext.new_shelx_weighting(a=self.a, b=self.b)

    def compute_chi_sq(fo_sq, fc_sq, a,b):
      weighting.a = a
      weighting.b = b
      weights = weighting(
        fo_sq.data(), fo_sq.sigmas(), fc_sq.data(), fo_sq.indices(), scale_factor)
      return (flex.sum(
        weights * flex.pow2(fo_sq.data() - scale_factor * fc_sq.data())))

    fo_sq = fo_sq.deep_copy()
    fo_sq.data().set_selected(fo_sq.data() < 0, 0)

    fo2 = fo_sq.data().deep_copy()
    fo2 /= scale_factor
    sigmas = fo_sq.sigmas() / scale_factor
    sigmas_sq = flex.pow2(sigmas)
    fc2 = fc_sq.data()

    # determine starting values for a and b, formulae taken from shelxl code
    p = (fo2 + 2. * fc2)/3.
    p_sq = flex.pow2(p)
    x = flex.sum((flex.pow2(fo2-fc2)-sigmas) * (p_sq/sigmas_sq))
    y = flex.sum( flex.pow2(p_sq)/sigmas_sq)
    z = flex.sum(p)
    start_a = math.sqrt(max(0.0001, 0.64 * x / max(1e-8, y)))
    start_b = 0.5 * z * start_a**2 / fo_sq.size()
    a_step = 0.2 * start_a
    b_step = 0.4 * start_b

    # sort data and setup binning by fc/fc_max
    fc_sq_over_fc_sq_max = fc_sq.data()/flex.max(fc_sq.data())
    permutation = flex.sort_permutation(fc_sq_over_fc_sq_max)
    fc_sq_over_fc_sq_max = fc_sq.customized_copy(
      data=fc_sq_over_fc_sq_max).select(permutation)
    fc_sq = fc_sq.select(permutation)
    fo_sq = fo_sq.select(permutation)
    n_bins = 10
    bin_limits = flex.size_t(1, 0)
    bin_count = flex.size_t()
    for i in range(n_bins):
      bin_limits.append(int(math.ceil((i+1) * fc_sq.size()/n_bins)))
      bin_count.append(bin_limits[i+1] - bin_limits[i])

    n = fo_sq.size()//(fo_sq.size()-n_independent_params)

    # search on a 9x9 grid to determine best values of a and b
    gridding = flex.grid(9,9)
    while (a_step > 1e-4 and b_step > 5e-3):
      tmp = flex.double(gridding, 0)
      binned_chi_sq = [tmp.deep_copy() for i in range(n_bins)]
      start_a = max(start_a, 4*a_step) - 4*a_step
      start_b = max(start_b, 4*b_step) - 4*b_step
      for i_bin in range(n_bins):
        sel = flex.size_t_range(bin_limits[i_bin], bin_limits[i_bin+1])
        fc2 = fc_sq.select(sel)
        fo2 = fo_sq.select(sel)
        b = start_b
        for j in range(9):
          a = start_a
          b += b_step
          for k in range(9):
            a += a_step
            binned_chi_sq[i_bin][j,k] += compute_chi_sq(fo2, fc2, a, b)
      min_variance = 9e9
      j_min, k_min = (0, 0)
      for j in range(9):
        for k in range(9):
          variance = 0
          for i_bin in range(n_bins):
            if bin_count[i_bin] == 0: continue
            goof = math.sqrt(binned_chi_sq[i_bin][j,k]*n/bin_count[i_bin])
            variance += (goof-1)**2
          min_variance = min(variance, min_variance)
          if variance == min_variance:
            j_min = j
            k_min = k
      start_a += k_min*a_step
      start_b += j_min*b_step
      if k_min == 8:
        a_step *= 2
        continue
      elif k_min != 0:
        a_step /= 4
      if j_min == 8:
        b_step *= 2
        continue
      elif j_min != 0:
        b_step /=4
      if start_a <= 1e-4: a_step /= 4
      if start_b <= 1e-3: b_step /= 4
    if start_a > 0.2:
      start_a = 0.2
      start_b = 0
    weighting.a = start_a
    weighting.b = start_b
    return weighting

@bp.inject_into(ext.stl_weighting)
class _():

  def __str__(self):
    return "w=[sin(theta)/lambda]^%f" %self.a

  def type(self):
    return "sin(theta)/lambda"

  def optimise_parameters(self, fo_sq, fc_sq,
                          scale_factor, n_independent_params):
    uc = fo_sq.crystal_symmetry().unit_cell()
    weighting = ext.stl_weighting(uc, a=self.a)
    def stl_weighting_compute_chi_sq(fo_sq, fc_sq, a):
      weighting.a = a
      weights = weighting(
        fo_sq.data(), fo_sq.sigmas(), fc_sq.data(), fo_sq.indices(), scale_factor)
      return (flex.sum(
        weights * flex.pow2(fo_sq.data() - scale_factor * fc_sq.data())))

    fo_sq = fo_sq.deep_copy()
    fo_sq.data().set_selected(fo_sq.data() < 0, 0)

    fo2 = fo_sq.data().deep_copy()
    fo2 /= scale_factor

    # sort data and setup binning by fc/fc_max
    fc_sq_over_fc_sq_max = fc_sq.data()/flex.max(fc_sq.data())
    permutation = flex.sort_permutation(fc_sq_over_fc_sq_max)
    fc_sq_over_fc_sq_max = fc_sq.customized_copy(
      data=fc_sq_over_fc_sq_max).select(permutation)
    fc_sq = fc_sq.select(permutation)
    fo_sq = fo_sq.select(permutation)
    n_bins = 10
    bin_limits = flex.size_t(1, 0)
    bin_count = flex.size_t()
    for i in range(n_bins):
      bin_limits.append(int(math.ceil((i+1) * fc_sq.size()/n_bins)))
      bin_count.append(bin_limits[i+1] - bin_limits[i])

    n = fo_sq.size()//(fo_sq.size()-n_independent_params)

    def calcres(w):
      averages = [flex.double() for i in range(n_bins)]
      for i_bin in range(n_bins):
        sel = flex.size_t_range(bin_limits[i_bin], bin_limits[i_bin+1])
        fc2 = fc_sq.select(sel)
        fo2 = fo_sq.select(sel)
        binned_chi_sq = stl_weighting_compute_chi_sq(fo2, fc2, w[0])
        averages[i_bin] = math.sqrt(binned_chi_sq*n/bin_count[i_bin])
      #goof = flex.mean(averages)
      residual = np.sum((np.array(averages)-1.0)**2)
      return residual
    result = optimize.minimize(calcres, [self.a], method='Nelder-Mead', bounds=((0.00001,None),))
    if result.success:
      weighting.a = result.x[0]
    else:
      print("Warning: optimisation of STL weighting parameter failed.\nNot updating the value!")
      weighting.a = self.a
    return weighting

@bp.inject_into(ext.unit_weighting)
class _():

  def __str__(self):
    return "w=1"

  def type(self):
    return "unit"

  def optimise_parameters(self, fo_sq, fc_sq,
                          scale_factor, n_independent_params):
    # no parameters to optimise!
    return self

@bp.inject_into(ext.sigma_weighting)
class _():

  def __str__(self): return "w=1/sigma^2"

  def type(self): return "sigma"

  def optimise_parameters(self, fo_sq, fc_sq,
                          scale_factor, n_independent_params):
    # no parameters to optimise!
    return self
