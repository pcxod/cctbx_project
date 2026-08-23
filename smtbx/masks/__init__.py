from __future__ import absolute_import, division, print_function

import sys

import cctbx.masks
from cctbx import maptbx, miller, sgtbx, xray
from cctbx.array_family import flex
from scitbx.math import approx_equal_relatively
from libtbx.utils import xfrange
from six.moves import range


class solvent_accessible_volume(object):
  def __init__(self, xray_structure,
               solvent_radius,
               shrink_truncation_radius,
               ignore_hydrogen_atoms=False,
               crystal_gridding=None,
               grid_step=None,
               d_min=None,
               resolution_factor=1/4,
               atom_radii_table=None,
               use_space_group_symmetry=False):
    self.xray_structure = xray_structure
    if crystal_gridding is None:
      self.crystal_gridding = maptbx.crystal_gridding(
        unit_cell=xray_structure.unit_cell(),
        space_group_info=xray_structure.space_group_info(),
        step=grid_step,
        d_min=d_min,
        resolution_factor=resolution_factor,
        symmetry_flags=sgtbx.search_symmetry_flags(
          use_space_group_symmetry=use_space_group_symmetry))
    else:
      self.crystal_gridding = crystal_gridding
    if use_space_group_symmetry:
      atom_radii = cctbx.masks.vdw_radii(
        xray_structure, table=atom_radii_table).atom_radii
      asu_mappings = xray_structure.asu_mappings(
        buffer_thickness=flex.max(atom_radii)+solvent_radius)
      scatterers_asu_plus_buffer = flex.xray_scatterer()
      frac = xray_structure.unit_cell().fractionalize
      for sc, mappings in zip(
        xray_structure.scatterers(), asu_mappings.mappings()):
        for mapping in mappings:
          scatterers_asu_plus_buffer.append(
            sc.customized_copy(site=frac(mapping.mapped_site())))
      xs = xray.structure(crystal_symmetry=xray_structure,
                          scatterers=scatterers_asu_plus_buffer)
    else:
      xs = xray_structure.expand_to_p1()
    self.vdw_radii = cctbx.masks.vdw_radii(xs, table=atom_radii_table)
    self.mask = cctbx.masks.around_atoms(
      unit_cell=xs.unit_cell(),
      space_group_order_z=xs.space_group().order_z(),
      sites_frac=xs.sites_frac(),
      atom_radii=self.vdw_radii.atom_radii,
      gridding_n_real=self.crystal_gridding.n_real(),
      solvent_radius=solvent_radius,
      shrink_truncation_radius=shrink_truncation_radius)
    if use_space_group_symmetry:
      tags = self.crystal_gridding.tags()
      tags.tags().apply_symmetry_to_mask(self.mask.data)
    self.flood_fill = cctbx.masks.flood_fill(
      self.mask.data, xray_structure.unit_cell())
    self.exclude_void_flags = [False] * self.flood_fill.n_voids()
    self.solvent_accessible_volume = self.n_solvent_grid_points() \
        / self.mask.data.size() * xray_structure.unit_cell().volume()

  def n_voids(self):
    return self.flood_fill.n_voids()

  def n_solvent_grid_points(self):
    return sum([self.mask.data.count(i+2) for i in range(self.n_voids())
                if not self.exclude_void_flags[i]])

  def show_summary(self, log=None):
    if log is None: log = sys.stdout
    print("solvent_radius: %.2f" %(self.mask.solvent_radius), file=log)
    print("shrink_truncation_radius: %.2f" %(
      self.mask.shrink_truncation_radius), file=log)
    print("van der Waals radii:", file=log)
    self.vdw_radii.show(log=log)
    print(file=log)
    print("Total solvent accessible volume / cell = %.1f Ang^3 [%.1f%%]" %(
      self.solvent_accessible_volume,
      100 * self.solvent_accessible_volume /
      self.xray_structure.unit_cell().volume()), file=log)
    n_voids = self.n_voids()
    print(file=log)
    self.flood_fill.show_summary(log=log)




class mask(object):
  def __init__(self, xray_structure, observations, use_set_completion=False):
    self.xray_structure = xray_structure
    self.fo2 = observations.as_intensity_array().average_bijvoet_mates()
    self.use_set_completion = use_set_completion
    if use_set_completion:
      self.complete_set = self.fo2.complete_set()
    else:
      self.complete_set = None
    self.mask = None
    self.solvent_radius = None
    # width of the boundary smearing, as a multiple of the solvent radius.
    # Zero reproduces the hard-edged mask exactly. See solvent_weight_map.
    self.boundary_smearing = 0
    # Fold the outward half of the taper away, so that it runs from zero at
    # the boundary inwards. A symmetric kernel already puts the half weight
    # point on the boundary, and folding there keeps the mask off the atoms
    # the boundary was drawn around. It costs about a fifth of the region
    # volume and the electrons in it, and it cannot be exact where the
    # boundary is concave, so it measured worse than the default on two
    # proteins once solvent_d_min was set. Off unless that trade is
    # wanted.
    self.boundary_smearing_inward = False
    # Hold the density inside the void at or above zero. The difference map is
    # made with phases from a model that is missing the very atoms the void
    # contains, and that bias routinely drives the integral negative; the code
    # below then throws the whole void away. Electrons are not negative, so
    # this is a statement about the map and not about the structure.
    self.void_positivity = False
    # Floor for the positivity constraint, in sigma of the difference map.
    # Clamping at zero rectifies noise: over the empty part of the void the
    # map is zero mean, and max(rho, 0) turns every negative fluctuation
    # positive, adding about V*sigma/sqrt(2 pi) of electrons that are not
    # there. Subtracting a floor of order the noise first removes that
    # pedestal and keeps the density that rises above it, which is the same
    # device as the threshold in charge flipping.
    self.void_positivity_threshold = 0.
    # Resolution limit, in Angstrom, of the data used to find the solvent.
    # Disordered solvent scatters only at low angle, so the high angle
    # reflections carry no information about it - what they do carry is the
    # model's own error, and cutting a region out of a map built from them
    # launders that error into f_mask. This is not damping f_mask: the
    # measured solvent density is all at low angle and stays, and f_mask is
    # still evaluated for every reflection. None uses all the data, which is
    # what the method did before.
    self.solvent_d_min = None
    # Weight the difference coefficients by how far the model can be trusted,
    # m*Fo - D*Fc instead of Fo - Fc. The bias this corrects is one reason the
    # void integrates negative; see smtbx.masks.bias, which needs no mmtbx.
    # Never yet judged on a refinement, only on the electron counts it gives,
    # which this module's history says is not enough to conclude from.
    self.bias_correction = False
    # Give back the solvent that partially occupied atoms wrongly exclude.
    # around_atoms knows sites and radii and no occupancies, so a quarter
    # occupancy side chain reaching into what would be solvent excludes bulk
    # solvent from all of that volume rather than a quarter of it. See
    # occupancy_weight_map. On by default: judged on held-out reflections over
    # five structures, best where the correction is large (R_free 0.1922 to
    # 0.1855) and never worse. A structure with no partial atoms takes the
    # early return below and is untouched.
    self.occupancy_weighting = True
    # How far below one an occupancy has to be before the atom counts as
    # partial. Refined free variables land a little off exact values, and a
    # structure where every atom is at 0.999 should behave as though it were
    # fully occupied.
    self.occupancy_full_tolerance = 0.01
    self._f_mask = None
    self._masked_diff_map = None
    self.f_000 = None
    self.f_000_s = None
    self.f_000_cell = None
    self._electron_counts_per_void = None

  def compute(self,
              solvent_radius,
              shrink_truncation_radius,
              ignore_hydrogen_atoms=False,
              crystal_gridding=None,
              grid_step=None,
              resolution_factor=1/4,
              atom_radii_table=None,
              use_space_group_symmetry=False):
    if grid_step is not None: d_min = None
    else: d_min = self.fo2.d_min()
    result = solvent_accessible_volume(
      self.xray_structure,
      solvent_radius,
      shrink_truncation_radius,
      ignore_hydrogen_atoms=ignore_hydrogen_atoms,
      crystal_gridding=crystal_gridding,
      grid_step=grid_step,
      d_min=d_min,
      resolution_factor=resolution_factor,
      atom_radii_table=atom_radii_table,
      use_space_group_symmetry=use_space_group_symmetry)
    self.crystal_gridding = result.crystal_gridding
    self.vdw_radii = result.vdw_radii
    self.solvent_radius = solvent_radius
    self.mask = result.mask
    self.flood_fill = result.flood_fill
    self.exclude_void_flags = [False] * self.flood_fill.n_voids()
    self.solvent_accessible_volume = self.n_solvent_grid_points() \
        / self.mask.data.size() * self.xray_structure.unit_cell().volume()

  def solvent_weight_map(self):
    """ Where the solvent region is, as a weight in [0, 1] rather than a step.

    The region is cut out of the difference map with a hard edge, and a step
    that sharp has a transform which does not decay.

    On its own that turns out to matter little. On a high-resolution
    protein |f_mask| holds about 30% of |Fc| in every shell out to 0.54 A, which no disordered
    solvent can do, and it is tempting to read that as the edge - but smearing
    the edge does not remove it. At a width of 1.0 the electron count moves by
    23% and the shell profile by 0.3%. Most of that content comes from the
    difference map itself, which carries the model's own error at atomic
    resolution straight through the region cut; solvent_d_min is what removes
    it, and this only becomes worth setting once that is done.

    What must not be done is to damp f_mask itself. This is not a protein bulk
    solvent model: the density inside the region comes from the observed
    difference map, so a Gaussian on f_mask would attenuate measured solvent
    density, and hardest exactly where the data are weakest. Only the edge is
    touched here - the weight is 1 throughout the interior, so the observed
    density there passes through untouched, and tapers to 0 across the
    boundary.

    The edge is not knowable more sharply than the probe that traced it, which
    is what sets the width: boundary_smearing * solvent_radius.

    maptbx.smooth_map builds a full box of structure factors from the map, so
    the work goes as the grid and not as the boundary being smoothed. Measured
    on a protein at a 0.12 A step, 8.3M points, that is 1.4 s for the weight and
    1.3x the mask stage overall - not free, not a problem.
    """
    # built by assignment rather than as_double() of a comparison, which
    # would drop the 3D accessor the map needs
    hard = self.mask.data.as_double()
    hard.set_selected(hard > 0, 1.)
    # computed once and added on whichever path returns. Putting it only after
    # the smearing branch left it unreachable at the default of no smearing,
    # and the whole correction silently did nothing - four structures came back
    # bit-identical with it on and off, which is what exposed it.
    correction = (self.occupancy_weight_map() if self.occupancy_weighting
                  else None)
    if not self.boundary_smearing:
      if correction is not None:
        hard = hard + correction
        hard.set_selected(hard > 1., 1.)
      return hard
    if not self.solvent_radius:
      raise RuntimeError(
        "boundary smearing needs the solvent radius, which compute() stores; "
        "call compute() before structure_factors()")
    w = maptbx.smooth_map(
      map=hard,
      crystal_symmetry=self.xray_structure.crystal_symmetry(),
      rad_smooth=self.boundary_smearing*self.solvent_radius,
      method="exp")
    if self.boundary_smearing_inward:
      # a smoothed step is 1/2 on the surface it came from, so 2w - 1 clipped
      # to [0, 1] is the same taper with its outer half folded away: zero on
      # the boundary, one where the kernel no longer reaches out of the
      # region. Costs region volume, which is the price of not reaching into
      # the model's density.
      w = w*2. - 1.
      w.set_selected(w < 0, 0.)
      w.set_selected(w > 1., 1.)
    if correction is not None:
      w = w + correction
      w.set_selected(w > 1., 1.)
    return w

  def occupancy_weight_map(self):
    """ Solvent the region cut wrongly threw away, because atoms are partial.

    cctbx::masks::around_atoms takes sites and radii and **no occupancies**, so
    a side chain at quarter occupancy excludes bulk solvent from all of the
    volume it touches instead of a quarter of it. Three quarters of that volume
    still holds solvent and currently holds none. Afonine et al. (2024) name
    this as one of the three failings of the flat model, and note that the
    error does not stay in the solvent: it surfaces as residual density over
    each conformer, or gets absorbed into wrongly refined occupancies and ADPs.

    The correction is a weight rather than a second region. For a grid point
    covered only by a disorder group of occupancy q, the solvent weight should
    be 1 - q, not 0. Groups are taken as the distinct occupancy values present,
    which for a SHELX model is what PART and its free variable produce anyway,
    and the weight multiplies over groups so that two overlapping conformers
    leave (1-q1)(1-q2).

    Returns the amount to be **added** to the hard region, so it is zero
    wherever a full-occupancy atom sits and wherever the region already counts
    as solvent. Nothing here can make the weight exceed one; the caller clamps.
    """
    xs = self.xray_structure
    occupancies = flex.double([sc.occupancy for sc in xs.scatterers()])
    partial = occupancies < 1 - self.occupancy_full_tolerance
    if partial.count(True) == 0:
      # flex reshape works in place and answers None, so it is never the last
      # thing in an expression
      zero = flex.double(self.mask.data.size(), 0)
      zero.reshape(self.mask.data.accessor())
      return zero
    # the region as it stands: 1 where solvent is already counted
    hard = self.mask.data.as_double()
    hard.set_selected(hard > 0, 1.)
    # distinct occupancies, rounded so that a free variable refined to
    # 0.2499998 and 0.25 are one group rather than two
    values = sorted(set(round(q, 3) for q in occupancies.select(partial)))

    def outside(selection):
      """ 1 where no atom of this selection reaches, 0 where one does.

      Expanded to P1 and given its own radii, exactly as the main mask is
      built, so that the two regions are drawn on the same footing and can be
      combined pointwise.
      """
      group = xs.select(selection).expand_to_p1()
      radii = cctbx.masks.vdw_radii(group).atom_radii
      r = cctbx.masks.around_atoms(
        unit_cell=group.unit_cell(),
        space_group_order_z=group.space_group().order_z(),
        sites_frac=group.sites_frac(),
        atom_radii=radii,
        gridding_n_real=self.crystal_gridding.n_real(),
        solvent_radius=self.solvent_radius,
        shrink_truncation_radius=self.mask.shrink_truncation_radius
        ).data.as_double()
      r.set_selected(r > 0, 1.)
      return r

    # A full-occupancy atom leaves no solvent behind whatever else overlaps it,
    # so the correction has to be switched off wherever one reaches. Without
    # this term every point of the model region untouched by a partial group -
    # which is most of the protein - would come back as full solvent.
    if partial.count(False) == 0:
      free_of_full = flex.double(self.mask.data.size(), 1)
      free_of_full.reshape(self.mask.data.accessor())
    else:
      free_of_full = outside(~partial)

    # Occupancies **add**, they do not multiply. Two alternate conformers of
    # one side chain at 0.6 and 0.4 are mutually exclusive: their shared volume
    # is occupied all of the time and must get no solvent back. A product of
    # (1-q) would hand back 0.4*0.6 = 0.24 of it, and on a structure with half
    # its atoms in alternates that error compounds - measured at 387% of the
    # region on one, which is what exposed it. Summing the occupancies and
    # clamping gives 1 - (0.6+0.4) = 0 there, and still gives 1-q where a lone
    # partial group sits, which is the case this exists for.
    covered = flex.double(self.mask.data.size(), 0)
    covered.reshape(self.mask.data.accessor())
    for q in values:
      if q <= 0:
        continue
      inside = 1. - outside(
        flex.bool([round(o, 3) == q for o in occupancies]))
      covered = covered + q*inside
    covered.set_selected(covered > 1., 1.)
    absent = 1. - covered
    # zero inside full atoms, zero where the region already counts as solvent,
    # and 1-q in the volume a group of occupancy q took away on its own
    return free_of_full*absent*(1. - hard)

  def _level_and_clamp(self, m, weight, sigma):
    """Put the F(000) level back into the region, and solve for it.

    F(000) is not measured, so the difference map has zero mean over the cell.
    A solvent lump of Q electrons therefore forces -Q/V everywhere, the region
    integrates to Q(1 - <w>) rather than Q, with <w> the mean of the region
    weight over the cell, and dividing that factor out is van der Sluis and
    Spek's electron count - which is what this returns when nothing is
    clamped.

    Holding the density non-negative does not remove that balance, it only
    makes the equation implicit, because the level decides which points are
    clamped and the clamped points change the level:

        Q = integral over the cell of max(w*(rho + Q/V), 0)

    The right hand side grows with Q at a rate below one, so there is a
    single fixed point. Clamping first and dividing by (1 - <w>) afterwards,
    as the two used to be written, corrects the same pedestal twice and
    overcounts by that factor.

    Solved by Newton rather than by repeated substitution: the rate is the
    weight fraction that is not clamped, which approaches one when the region
    is most of the cell, and substitution then needs thousands of passes. The
    equation is piecewise linear and that rate is its slope, so Newton has it
    in a handful.
    """
    dv = self.fft_scale
    v_cell = self.xray_structure.unit_cell().volume()
    flat = m.as_1d()
    w_flat = weight.as_1d()
    mean_w = flex.sum(w_flat)/flat.size()
    integral = flex.sum(flat)*dv
    q = integral/(1 - mean_w) if mean_w < 1 else integral
    cut = 0.
    if self.void_positivity:
      # charge flipping's device: subtracted before the clamp, so that noise
      # is rectified into electrons as little as possible
      cut = self.void_positivity_threshold*sigma
      for _ in range(50):
        trial = flat + w_flat*(q/v_cell - cut)
        active = trial > 0
        trial.set_selected(~active, 0)
        residual = flex.sum(trial)*dv - q
        if abs(residual) <= 1e-9*max(1., abs(q)): break
        slope = flex.sum(w_flat.select(active))*dv/v_cell
        if slope >= 1 - 1e-9:
          # the unclamped region is the whole cell, so the level cancels out
          # of its own equation and F(000) is simply not determined by it
          break
        q += residual/(1 - slope)
    m = m + weight*(q/v_cell - cut)
    if self.void_positivity:
      m.set_selected(m < 0, 0)
    # read the count back off the map that was built, so that what is reported
    # and what f_mask is computed from can never be two different things
    return m, flex.sum(m.as_1d())*dv

  def _difference_coefficients(self, f_obs, f_calc):
    """Fo/k - Fc, or its sigma_A weighted form when asked for."""
    if not self.bias_correction:
      return f_obs.f_obs_minus_f_calc(1/self.scale_factor, f_calc)
    from smtbx.masks import bias
    scaled = f_obs.customized_copy(data=f_obs.data()/self.scale_factor)
    return bias.difference_coefficients(scaled, f_calc)

  def structure_factors(self, max_cycles=10):
    """P. van der Sluis and A. L. Spek, Acta Cryst. (1990). A46, 194-201."""
    assert self.mask is not None
    if self.n_voids() == 0: return
    if self.use_set_completion:
      f_calc_set = self.complete_set
    else:
      f_calc_set = self.fo2.set()
    self.f_calc = f_calc_set.structure_factors_from_scatterers(
      self.xray_structure, algorithm="direct").f_calc()
    f_obs = self.f_obs()
    self.scale_factor = flex.sum(f_obs.data())/flex.sum(
      flex.abs(self.f_calc.data()))
    f_obs_minus_f_calc = self._difference_coefficients(f_obs, self.f_calc)
    self.fft_scale = self.xray_structure.unit_cell().volume()\
        / self.crystal_gridding.n_grid_points()
    epsilon_for_min_residual = 2
    # once: the mask does not change through the iteration, and smoothing it
    # costs a transform pair
    solvent_weight = self.solvent_weight_map()
    for i in range(max_cycles):
      coefficients = f_obs_minus_f_calc
      if self.solvent_d_min is not None:
        coefficients = coefficients.resolution_filter(d_min=self.solvent_d_min)
      self.diff_map = miller.fft_map(self.crystal_gridding, coefficients)
      self.diff_map.apply_volume_scaling()
      stats = self.diff_map.statistics()
      # multiplied by the weight rather than cut by a selection, so that with
      # boundary_smearing set the edge tapers instead of stepping. At the
      # default of zero the weight is exactly the old 0/1 indicator and this
      # reproduces set_selected(mask == 0, 0) value for value.
      masked_diff_map = self.diff_map.real_map_unpadded()*solvent_weight
      self.f_000 = flex.sum(masked_diff_map) * self.fft_scale
      masked_diff_map, f_000_s = self._level_and_clamp(
        masked_diff_map, solvent_weight, stats.sigma())
      for j in range(self.n_voids()):
        # a void whose own electron count is negative is not solvent, so it
        # goes. The count is read off the levelled map: before the level is
        # restored every void still carries its share of the missing F(000),
        # and the test would then discard a void merely for holding less than
        # that share rather than for holding nothing.
        selection = self.mask.data == j+2
        if self.exclude_void_flags[j]:
          masked_diff_map.set_selected(selection, 0)
          continue
        diff_map_ = masked_diff_map.deep_copy().set_selected(~selection, 0)
        electrons = flex.sum(diff_map_) * self.fft_scale
        if electrons < 0:
          masked_diff_map.set_selected(selection, 0)
          f_000_s -= electrons
          self.exclude_void_flags[j] = True
      previous_f_000_s = self.f_000_s
      self.f_000_s = f_000_s
      self._masked_diff_map = masked_diff_map
      if 0:
        from crys3d import wx_map_viewer
        wx_map_viewer.display(
          title="masked diff_map",
          raw_map=masked_diff_map.as_double(),
          unit_cell=f_obs.unit_cell())
      self._f_mask = f_obs.structure_factors_from_map(map=masked_diff_map)
      self._f_mask *= self.fft_scale
      scales = []
      residuals = []
      min_residual = 1000
      for epsilon in xfrange(epsilon_for_min_residual, 0.9, -0.2):
        f_model_ = self.f_model(epsilon=epsilon)
        scale = flex.sum(f_obs.data())/flex.sum(flex.abs(f_model_.data()))
        residual = flex.sum(flex.abs(
          1/scale * flex.abs(f_obs.data())- flex.abs(f_model_.data()))) \
                 / flex.sum(1/scale * flex.abs(f_obs.data()))
        scales.append(scale)
        residuals.append(residual)
        min_residual = min(min_residual, residual)
        if min_residual == residual:
          scale_for_min_residual = scale
          epsilon_for_min_residual = epsilon
      self.scale_factor = scale_for_min_residual
      if (previous_f_000_s is not None and
          approx_equal_relatively(previous_f_000_s, f_000_s, 0.0001)):
        break # we have reached convergence
      f_model = self.f_model(epsilon=epsilon_for_min_residual)
      f_obs = self.f_obs()
      f_obs_minus_f_calc = self._difference_coefficients(
        f_obs.phase_transfer(f_model), self.f_calc)
    return self._f_mask

  def f_obs(self):
    fo2 = self.fo2.as_intensity_array()
    f_obs = fo2.as_amplitude_array()
    if self.use_set_completion:
      if self._f_mask is not None:
        f_model = self.f_model()
      else:
        f_model = self.f_calc
      data_substitute = flex.abs(f_model.data())
      scale_factor = flex.sum(f_obs.data())/flex.sum(
        f_model.common_set(f_obs).as_amplitude_array().data())
      f_obs = f_obs.matching_set(
        other=self.complete_set,
        data_substitute=scale_factor*flex.abs(f_model.data()),
        sigmas_substitute=0)
    return f_obs

  def f_mask(self):
    return self._f_mask

  def f_model(self, f_calc=None, epsilon=None):
    if self._f_mask is None: return None
    f_mask = self.f_mask()
    if f_calc is None:
      f_calc = self.f_calc
    if epsilon is None:
      data = f_calc.data() + f_mask.data()
    else:
      data = f_calc.data() + epsilon * f_mask.data()
    return miller.array(miller_set=f_calc, data=data)

  def modified_intensities(self):
    """Intensities with the solvent contribution removed."""
    if self._f_mask is None: return None
    f_mask = self.f_mask().common_set(self.fo2)
    f_model = self.f_model().common_set(self.fo2)
    return modified_intensities(
      self.fo2, f_model, f_mask)

  def n_voids(self):
    return self.flood_fill.n_voids()

  def n_solvent_grid_points(self):
    return sum([self.mask.data.count(i+2) for i in range(self.n_voids())
                if not self.exclude_void_flags[i]])

  def electron_counts_per_void(self):
    if self._electron_counts_per_void is not None:
      return self._electron_counts_per_void
    self._electron_counts_per_void = []
    # the map structure_factors actually built, not a third reconstruction of
    # it: this used to cut the region out with a hard edge and no level
    # whatever the other two copies were set to
    masked_diff_map = self._masked_diff_map
    if masked_diff_map is None: return self._electron_counts_per_void
    for i in range(self.n_voids()):
      if self.exclude_void_flags[i]:
        electrons = 0
      else:
        diff_map = masked_diff_map.deep_copy().set_selected(
          self.mask.data != i+2, 0)
        electrons = flex.sum(diff_map) * self.fft_scale
      self._electron_counts_per_void.append(electrons)
    # a smeared boundary puts density outside the flood filled void that the
    # labels above cannot attribute, so the parts are made to sum to the whole
    total = flex.sum(masked_diff_map) * self.fft_scale
    parts = sum(self._electron_counts_per_void)
    if parts > 0 and abs(total - parts) > 1e-6*abs(total):
      self._electron_counts_per_void = [
        e*total/parts for e in self._electron_counts_per_void]
    return self._electron_counts_per_void

  def show_summary(self, log=None):
    if log is None: log = sys.stdout
    print("use_set_completion: %s" %self.use_set_completion, file=log)
    print("solvent_radius: %.2f" %(self.mask.solvent_radius), file=log)
    print("shrink_truncation_radius: %.2f" %(
      self.mask.shrink_truncation_radius), file=log)
    print("van der Waals radii:", file=log)
    self.vdw_radii.show(log=log)
    print(file=log)
    print("Total solvent accessible volume / cell = %.1f Ang^3 [%.1f%%]" %(
      self.solvent_accessible_volume,
      100 * self.solvent_accessible_volume /
      self.xray_structure.unit_cell().volume()), file=log)
    n_voids = self.n_voids()
    if n_voids > 0:
      print("Total electron count / cell = %.1f" %(self.f_000_s), file=log)
    print(file=log)
    self.flood_fill.show_summary(log=log)
    if n_voids == 0: return
    print(file=log)
    print("Void  Vol/Ang^3  #Electrons", file=log)
    grid_points_per_void = self.flood_fill.grid_points_per_void()
    com = self.flood_fill.centres_of_mass_frac()
    electron_counts = self.electron_counts_per_void()
    for i in range(self.n_voids()):
      void_vol = (
        self.xray_structure.unit_cell().volume() * grid_points_per_void[i]) \
               / self.crystal_gridding.n_grid_points()
      formatted_site = ["%6.3f" % x for x in com[i]]
      print("%4i" %(i+1), end=' ', file=log)
      print("%10.1f     " %void_vol, end=' ', file=log)
      print("%7.1f" %electron_counts[i], file=log)

  def as_cif_block(self):
    from iotbx import cif
    cif_block = cif.model.block()
    mask_loop = cif.model.loop(header=(
      "_smtbx_masks_void_nr",
      "_smtbx_masks_void_average_x",
      "_smtbx_masks_void_average_y",
      "_smtbx_masks_void_average_z",
      "_smtbx_masks_void_volume",
      "_smtbx_masks_void_count_electrons",
      "_smtbx_masks_void_content",
    ))
    n_voids = self.n_voids()
    if n_voids == 0: return cif_block
    grid_points_per_void = self.flood_fill.grid_points_per_void()
    com = self.flood_fill.centres_of_mass_frac()
    electron_counts = self.electron_counts_per_void()
    for i in range(self.n_voids()):
      void_vol = (
        self.xray_structure.unit_cell().volume() * grid_points_per_void[i]) \
               / self.crystal_gridding.n_grid_points()
      xyz = list(com[i])
      for j in range(3):
        if round(xyz[j],6) == 0: xyz[j] = 0
      site_fmt = "%.3f"
      mask_loop.add_row(
        [i+1, site_fmt % xyz[0], site_fmt % xyz[1],
         site_fmt % xyz[2], "%.1f" % void_vol,
         "%.1f" %electron_counts[i], '?'])
    cif_block.add_loop(mask_loop)
    cif_block['_smtbx_masks_special_details'] = '?'
    return cif_block


def modified_intensities(observations, f_model, f_mask,scale_factor=None):
  """Subtracts the solvent contribution from the observed structure
  factors to obtain modified structure factors, suitable for refinement
  with other refinement programs such as ShelXL"""
  f_obs = observations.as_amplitude_array()
  if f_obs.sigmas() is not None:
    weights = weights=1/flex.pow2(f_obs.sigmas())
  else:
    weights = None
  if scale_factor == None:
    scale_factor = f_obs.scale_factor(f_model, weights=weights)
  f_obs = f_obs.phase_transfer(phase_source=f_model)
  modified_f_obs = miller.array(
    miller_set=f_obs,
    data=(f_obs.data() - f_mask.data()*scale_factor))
  if observations.is_xray_intensity_array():
    # it is better to use the original sigmas for intensity if possible
    return modified_f_obs.as_intensity_array().customized_copy(
      sigmas=observations.sigmas())
  else:
    return modified_f_obs.customized_copy(
      sigmas=f_obs.sigmas()).as_intensity_array()
