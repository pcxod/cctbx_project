from __future__ import absolute_import, division, print_function

import sys

import cctbx.masks
from cctbx import maptbx, miller, sgtbx, xray
from cctbx.array_family import flex
from scitbx.math import approx_equal_relatively
from libtbx.utils import xfrange
from six.moves import range


def hydrogens_at_common_length(xray_structure, x_h_length, max_bond=1.3):
  """ A copy with every H moved along its bond to x_h_length from its parent.

  For drawing a solvent region only: the region is filled from the difference
  map, so where a model puts its H decides its volume, and an aspherical model
  (longer X-H) would otherwise get less region than IAM.

  x_h_length is one length, or a dict by parent element symbol (any case).
  The parent is the nearest non-H within max_bond under any symmetry
  operation, of the H's PART or PART 0 (an H in PART 0 takes any). An H with
  no parent, or whose parent element is not in the dict, stays put; their
  labels are returned.
  """
  xs = xray_structure.deep_copy_scatterers()
  scatterers = xs.scatterers()
  hd = xs.hd_selection()
  unit_cell = xs.unit_cell()
  sites_frac = xs.sites_frac()
  pair_sym_table = xs.pair_asu_table(distance_cutoff=max_bond)\
    .extract_pair_sym_table(skip_j_seq_less_than_i_seq=False,
                            all_interactions_from_inside_asu=True)
  if isinstance(x_h_length, dict):
    by_element = dict((e.capitalize(), l) for e, l in x_h_length.items())
  else:
    by_element = None
  without_parent = []
  for i_seq in hd.iselection():
    part = scatterers[i_seq].get_part()
    best = None
    for j_seq, sym_ops in pair_sym_table[i_seq].items():
      if hd[j_seq]: continue
      part_j = scatterers[j_seq].get_part()
      if part != 0 and part_j not in (0, part): continue
      for rt_mx in sym_ops:
        site = rt_mx * sites_frac[j_seq]
        d = unit_cell.distance(sites_frac[i_seq], site)
        if best is None or d < best[0]:
          best = (d, site, j_seq)
    length = x_h_length
    if best is not None and by_element is not None:
      length = by_element.get(
        (scatterers[best[2]].element_symbol() or '').capitalize())
    if best is None or best[0] == 0 or length is None:
      without_parent.append(scatterers[i_seq].label)
      continue
    d, parent, _ = best
    xh = unit_cell.orthogonalize(sites_frac[i_seq])
    xp = unit_cell.orthogonalize(parent)
    scatterers[i_seq].site = unit_cell.fractionalize(
      [p + length*(h - p)/d for h, p in zip(xh, xp)])
  return xs, without_parent


def _void_iselections(mask_data, n_voids):
  """ The grid points of each void (label j+2), one selection per void.

  One pass over the grid, not one per void (a protein region can have
  thousands). Indices ascend within a void, as iselection() gives them, so
  sums over a selection are bit-identical.
  """
  import numpy as np
  labels = mask_data.as_numpy_array().ravel()
  points = np.flatnonzero(labels >= 2)
  voids = labels[points] - 2
  points = points[np.argsort(voids, kind='stable')]
  ends = np.cumsum(np.bincount(voids, minlength=n_voids))
  return [flex.size_t(s.astype(np.uint64))
          for s in np.split(points, ends[:-1])][:n_voids]


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
               use_space_group_symmetry=False,
               x_h_length=None):
    self.xray_structure = xray_structure
    # what the region is drawn around; H at x_h_length when that is given
    self.region_structure = xray_structure
    self.hydrogens_without_parent = []
    if x_h_length is not None and not ignore_hydrogen_atoms:
      self.region_structure, self.hydrogens_without_parent = \
        hydrogens_at_common_length(xray_structure, x_h_length)
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
    # ignore_hydrogen_atoms is now honoured; off by default, as leaving H out
    # (as mmtbx does) measured worse on held-out reflections
    # old: drew around xray_structure itself, H always in:
    # region = xray_structure
    region = self.region_structure
    if ignore_hydrogen_atoms:
      region = xray_structure.select(~xray_structure.hd_selection())
    if use_space_group_symmetry:
      atom_radii = cctbx.masks.vdw_radii(
        region, table=atom_radii_table).atom_radii
      asu_mappings = region.asu_mappings(
        buffer_thickness=flex.max(atom_radii)+solvent_radius)
      scatterers_asu_plus_buffer = flex.xray_scatterer()
      frac = region.unit_cell().fractionalize
      for sc, mappings in zip(
        region.scatterers(), asu_mappings.mappings()):
        for mapping in mappings:
          scatterers_asu_plus_buffer.append(
            sc.customized_copy(site=frac(mapping.mapped_site())))
      xs = xray.structure(crystal_symmetry=region,
                          scatterers=scatterers_asu_plus_buffer)
    else:
      xs = region.expand_to_p1()
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
    self._void_isel = None
    self.solvent_accessible_volume = self.n_solvent_grid_points() \
        / self.mask.data.size() * xray_structure.unit_cell().volume()

  def n_voids(self):
    return self.flood_fill.n_voids()

  def void_iselection(self, i):
    """ The grid points of void i, as indices into the mask's grid.

    Built once on first use (the mask is fixed), so summing a void costs a
    pass over the void, not over the grid.
    """
    if self._void_isel is None:
      self._void_isel = _void_iselections(self.mask.data, self.n_voids())
    return self._void_isel[i]

  def n_solvent_grid_points(self):
    # old: a pass over the grid per void; the flood fill has counted them
    # return sum([self.mask.data.count(i+2) for i in range(self.n_voids())
    #             if not self.exclude_void_flags[i]])
    per_void = self.flood_fill.grid_points_per_void()
    return sum([per_void[i] for i in range(self.n_voids())
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
    # boundary smearing width in units of solvent_radius; 0 is the hard-edged
    # mask exactly. See solvent_weight_map
    self.boundary_smearing = 0
    # fold the outer half of the taper away so it runs from 0 at the boundary
    # inwards: keeps the mask off the atoms, costs region volume, and is not
    # exact where the boundary is concave
    self.boundary_smearing_inward = False
    # hold the void density >= 0: phases from a model missing the void's
    # content drive the integral negative, and a negative void is dropped
    self.void_positivity = False
    # floor for void_positivity in sigma of the difference map: clamping at 0
    # rectifies zero-mean noise into ~V*sigma/sqrt(2 pi) of false electrons,
    # subtracting a noise-sized floor first removes that (as in charge flipping)
    self.void_positivity_threshold = 0.
    # d_min (A) of the coefficients the solvent map is built from. Disordered
    # solvent scatters only at low angle; the high-angle terms carry only model
    # error into f_mask. f_mask itself is still evaluated for every reflection.
    # None uses all data
    self.solvent_d_min = None
    # m*Fo - D*Fc instead of Fo - Fc (smtbx.masks.bias), against the model
    # bias that drives voids negative
    self.bias_correction = False
    # give back the solvent partial atoms wrongly exclude: around_atoms has no
    # occupancies, so a quarter-occupied side chain excludes all of its volume.
    # See occupancy_weight_map; no effect without partial atoms
    self.occupancy_weighting = True
    # an atom counts as partial below 1 - this, so refined occupancies of
    # 0.999 behave as full
    self.occupancy_full_tolerance = 0.01
    # Fc for the difference map: "direct" or "fft". FFT is much cheaper for
    # large structures and accurate enough for a map that is only cut into
    # regions, but it changes the numbers, hence not the default
    self.f_calc_algorithm = "direct"
    # disorder group in occupancy_weight_map: 'value' groups atoms by
    # occupancy, 'part' by (PART, occupancy). With 'value' two 0.5/0.5
    # alternates are one group, so their overlap gets 0.5 back instead of 0
    self.occupancy_groups = 'value'
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
              use_space_group_symmetry=False,
              x_h_length=None):
    """x_h_length: draw the region with every H at this distance from its
    parent (see hydrogens_at_common_length), so that IAM and aspherical models
    get the same region. None uses the model's own H positions.
    """
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
      use_space_group_symmetry=use_space_group_symmetry,
      x_h_length=x_h_length)
    self.crystal_gridding = result.crystal_gridding
    self.vdw_radii = result.vdw_radii
    self.solvent_radius = solvent_radius
    # kept for occupancy_weight_map, which leaves out the same atoms and
    # draws around the same sites
    self.ignore_hydrogen_atoms = ignore_hydrogen_atoms
    self.region_structure = result.region_structure
    self.hydrogens_without_parent = result.hydrogens_without_parent
    self.mask = result.mask
    self.flood_fill = result.flood_fill
    self.exclude_void_flags = [False] * self.flood_fill.n_voids()
    self._void_isel = None
    self.solvent_accessible_volume = self.n_solvent_grid_points() \
        / self.mask.data.size() * self.xray_structure.unit_cell().volume()

  def solvent_weight_map(self):
    """ Where the solvent region is, as a weight in [0, 1] rather than a step.

    A hard edge has a transform that does not decay. Only the edge is
    tapered, over boundary_smearing * solvent_radius (the edge is not known
    more sharply than the probe that traced it); the interior keeps weight 1.
    f_mask itself must never be damped: the region holds observed density,
    and damping would attenuate measured solvent where the data are weakest.
    Most high-angle content of f_mask is model error from the difference map,
    which solvent_d_min removes, not this.
    """
    # built by assignment rather than as_double() of a comparison, which
    # would drop the 3D accessor the map needs
    hard = self.mask.data.as_double()
    hard.set_selected(hard > 0, 1.)
    # added on both return paths; the unsmeared one must not skip it
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
      # a smoothed step is 1/2 on its surface, so 2w - 1 clipped to [0, 1] is
      # the same taper with its outer half folded away
      w = w*2. - 1.
      w.set_selected(w < 0, 0.)
      w.set_selected(w > 1., 1.)
    if correction is not None:
      w = w + correction
      w.set_selected(w > 1., 1.)
    return w

  def occupancy_weight_map(self):
    """ Solvent the region cut wrongly threw away, because atoms are partial.

    around_atoms has no occupancies, so a quarter-occupied side chain excludes
    solvent from all of its volume instead of a quarter (Afonine et al. 2024).
    A grid point covered only by groups of total occupancy q gets solvent
    weight 1 - q. Groups are the distinct occupancies, or (PART, occupancy)
    with occupancy_groups 'part'.

    Returns the amount to be added to the hard region: zero wherever a full
    atom reaches and wherever the region already counts as solvent. The
    caller clamps the sum at one.
    """
    # the sites the main region was drawn around, H moved if it was asked for;
    # same scatterers in the same order as xray_structure
    xs = getattr(self, 'region_structure', self.xray_structure)
    occupancies = flex.double([sc.occupancy for sc in xs.scatterers()])
    # atoms the main region was not drawn around take no part here either
    if getattr(self, 'ignore_hydrogen_atoms', False):
      considered = ~xs.hd_selection()
    else:
      considered = flex.bool(xs.scatterers().size(), True)
    partial = (occupancies < 1 - self.occupancy_full_tolerance) & considered
    if partial.count(True) == 0:
      # flex reshape works in place and answers None, so it is never the last
      # thing in an expression
      zero = flex.double(self.mask.data.size(), 0)
      zero.reshape(self.mask.data.accessor())
      return zero
    import numpy as np
    # the region as it stands: >0 where solvent is already counted
    labels = self.mask.data.as_numpy_array().ravel()
    # one key per atom, occupancy rounded so refined 0.2499998 and 0.25 are
    # one group
    by_part = getattr(self, 'occupancy_groups', 'value') == 'part'
    keys = [(sc.get_part() if by_part else 0, round(sc.occupancy, 3))
            for sc in xs.scatterers()]
    groups = sorted(set(k for k, p in zip(keys, partial) if p))

    def outside(selection):
      """ Non-zero where no atom of this selection reaches, 0 where one does.

      Drawn on the main mask's grid, expanded to P1, so the two combine
      pointwise. Raw grid, not clamped to 0/1: callers only test for zero.
      """
      group = xs.select(selection).expand_to_p1()
      # cctbx default radii even when the main region used a caller's table:
      # the caller's (smaller) radii measured worse on held-out reflections
      radii = cctbx.masks.vdw_radii(group).atom_radii
      return cctbx.masks.around_atoms(
        unit_cell=group.unit_cell(),
        space_group_order_z=group.space_group().order_z(),
        sites_frac=group.sites_frac(),
        atom_radii=radii,
        gridding_n_real=self.crystal_gridding.n_real(),
        solvent_radius=self.solvent_radius,
        shrink_truncation_radius=self.mask.shrink_truncation_radius
        ).data.as_numpy_array().ravel()

    # a full atom leaves no solvent whatever overlaps it, else all of the model
    # region untouched by a partial group would come back as solvent. Only
    # points outside both the region and every full atom can carry a weight,
    # so the groups are evaluated there only
    inside_region = labels <= 0
    full = ~partial & considered
    if full.count(True) == 0:
      support = np.flatnonzero(inside_region)
    else:
      support = np.flatnonzero(inside_region & (outside(full) != 0))

    # occupancies add, they do not multiply: alternates at 0.6 and 0.4 are
    # mutually exclusive, so their shared volume is always occupied and gets
    # 1 - (0.6+0.4) = 0 back, where a product of (1-q) would give 0.24
    covered = np.zeros(support.size)
    for group in groups:
      q = group[1]
      if q <= 0:
        continue
      inside = outside(flex.bool([k == group for k in keys])
                       & considered)[support] == 0
      covered += q*inside
    np.minimum(covered, 1., out=covered)
    # zero inside full atoms, zero where the region already counts as solvent,
    # and 1-q in the volume a group of occupancy q took away on its own
    weight = np.zeros(labels.size)
    weight[support] = (1. - covered)*(1. - labels[support])
    result = flex.double(weight)
    result.reshape(self.mask.data.accessor())
    return result

  def _level_and_clamp(self, m, weight, sigma, mean_w=None):
    """Put the F(000) level back into the region, and solve for it.

    The difference map has zero mean over the cell, so a solvent lump of Q
    electrons integrates to Q(1 - <w>) over the region (<w> the cell mean of
    the weight); dividing that out is van der Sluis and Spek's count, and is
    what this returns when nothing is clamped. With the clamp the level
    decides what is clamped, so Q solves

        Q = integral over the cell of max(w*(rho + Q/V), 0)

    whose right side grows with Q at a rate below one: a single fixed point.
    Clamping first and then dividing by (1 - <w>) would correct the pedestal
    twice. Newton, not substitution: the rate (the unclamped weight fraction)
    nears one for a large region, and the equation is piecewise linear.
    """
    dv = self.fft_scale
    v_cell = self.xray_structure.unit_cell().volume()
    flat = m.as_1d()
    w_flat = weight.as_1d()
    # optional: Olex2's own copy of structure_factors calls with three arguments
    if mean_w is None:
      mean_w = flex.sum(w_flat)/flat.size()
    integral = flex.sum(flat)*dv
    q = integral/(1 - mean_w) if mean_w < 1 else integral
    cut = 0.
    if self.void_positivity:
      # subtracted before the clamp, so noise is not rectified into electrons
      cut = self.void_positivity_threshold*sigma
      for _ in range(50):
        trial = flat + w_flat*(q/v_cell - cut)
        active = trial > 0
        trial.set_selected(~active, 0)
        residual = flex.sum(trial)*dv - q
        if abs(residual) <= 1e-9*max(1., abs(q)): break
        slope = flex.sum(w_flat.select(active))*dv/v_cell
        if slope >= 1 - 1e-9:
          # unclamped region is the whole cell: Q drops out, undetermined
          break
        q += residual/(1 - slope)
    m = m + weight*(q/v_cell - cut)
    if self.void_positivity:
      m.set_selected(m < 0, 0)
    # count read off the built map, so it is what f_mask is made of
    return m, flex.sum(m.as_1d())*dv

  def _difference_coefficients(self, f_obs, f_calc):
    """Fo/k - Fc, or its sigma_A weighted form when asked for."""
    if not self.bias_correction:
      return f_obs.f_obs_minus_f_calc(1/self.scale_factor, f_calc)
    from smtbx.masks import bias
    scaled = f_obs.customized_copy(data=f_obs.data()/self.scale_factor)
    return bias.difference_coefficients(scaled, f_calc)

  def structure_factors(self, max_cycles=10, f_calc=None, scale_factor=None,
                        convergence=0.0001):
    """P. van der Sluis and A. L. Spek, Acta Cryst. (1990). A46, 194-201.

    f_calc: the model's structure factors on the mask's set (complete_set
    with use_set_completion, fo2 otherwise). Left out, they are computed from
    spherical atoms, which is wrong for an aspherical model: the void then
    soaks up what IAM misses.
    scale_factor: a starting scale; by default sum(Fo)/sum(|Fc|).
    """
    assert self.mask is not None
    if self.n_voids() == 0: return
    if self.use_set_completion:
      f_calc_set = self.complete_set
    else:
      f_calc_set = self.fo2.set()
    if f_calc is None:
      f_calc = f_calc_set.structure_factors_from_scatterers(
        self.xray_structure, algorithm=self.f_calc_algorithm).f_calc()
    elif not f_calc.indices().all_eq(f_calc_set.indices()):
      # put into the mask's order; asserts if f_calc misses any reflection
      permutation = miller.match_indices(
        f_calc_set.indices(), f_calc.indices()).permutation()
      f_calc = f_calc_set.array(data=f_calc.data().select(permutation))
    self.f_calc = f_calc
    f_obs = self.f_obs()
    if scale_factor is None:
      scale_factor = flex.sum(f_obs.data())/flex.sum(
        flex.abs(self.f_calc.data()))
    self.scale_factor = scale_factor
    f_obs_minus_f_calc = self._difference_coefficients(f_obs, self.f_calc)
    self.fft_scale = self.xray_structure.unit_cell().volume()\
        / self.crystal_gridding.n_grid_points()
    epsilon_for_min_residual = 2
    # once: the mask is fixed through the iteration
    solvent_weight = self.solvent_weight_map()
    mean_w = flex.sum(solvent_weight.as_1d())/solvent_weight.size()
    for i in range(max_cycles):
      # old: tested each void's sign before the F(000) level was restored, so
      # a void was dropped for holding less than its share of the missing
      # F(000); replaced by the weighted, levelled map of _level_and_clamp
      # self.diff_map = miller.fft_map(self.crystal_gridding, f_obs_minus_f_calc)
      # self.diff_map.apply_volume_scaling()
      # stats = self.diff_map.statistics()
      # masked_diff_map = self.diff_map.real_map_unpadded().set_selected(
      #   self.mask.data.as_double() == 0, 0)
      # n_solvent_grid_points = self.n_solvent_grid_points()
      # for j in range(self.n_voids()):
      #   # exclude voids with negative electron counts from the masked map
      #   # set the electron density in those areas to be zero
      #   selection = self.mask.data == j+2
      #   if self.exclude_void_flags[j]:
      #     masked_diff_map.set_selected(selection, 0)
      #     continue
      #   diff_map_ = masked_diff_map.deep_copy().set_selected(~selection, 0)
      #   f_000 = flex.sum(diff_map_) * self.fft_scale
      #   f_000_s = f_000 * (
      #     self.crystal_gridding.n_grid_points() /
      #     (self.crystal_gridding.n_grid_points() - n_solvent_grid_points))
      #   if f_000_s < 0:
      #     masked_diff_map.set_selected(selection, 0)
      #     f_000_s = 0
      #     self.exclude_void_flags[j] = True
      # self.f_000 = flex.sum(masked_diff_map) * self.fft_scale
      # f_000_s = self.f_000 * (masked_diff_map.size() /
      #   (masked_diff_map.size() - self.n_solvent_grid_points()))
      # if (self.f_000_s is not None and
      #     approx_equal_relatively(self.f_000_s, f_000_s, 0.0001)):
      #   break # we have reached convergence
      # else:
      #   self.f_000_s = f_000_s
      # masked_diff_map.add_selected(
      #   self.mask.data.as_double() > 0,
      #   self.f_000_s/self.xray_structure.unit_cell().volume())
      coefficients = f_obs_minus_f_calc
      if self.solvent_d_min is not None:
        coefficients = coefficients.resolution_filter(d_min=self.solvent_d_min)
      self.diff_map = miller.fft_map(self.crystal_gridding, coefficients)
      self.diff_map.apply_volume_scaling()
      # a full-grid pass needed only by the void_positivity cut
      sigma = self.diff_map.statistics().sigma() if self.void_positivity else 0.
      # multiplied, not cut, so a smeared edge tapers; at boundary_smearing 0
      # the weight is the 0/1 indicator and this equals set_selected(mask == 0, 0)
      masked_diff_map = self.diff_map.real_map_unpadded()*solvent_weight
      self.f_000 = flex.sum(masked_diff_map) * self.fft_scale
      masked_diff_map, f_000_s = self._level_and_clamp(
        masked_diff_map, solvent_weight, sigma, mean_w)
      for j in range(self.n_voids()):
        # a void with a negative count on the levelled map is not solvent
        selection = self.void_iselection(j)
        electrons = flex.sum(masked_diff_map.select(selection)) * self.fft_scale
        # index selections need a 1-d array; as_1d() is a view, so zeroing
        # through it zeroes the map. A dropped void takes its electrons out of
        # f_000_s, which the level above counted
        if self.exclude_void_flags[j]:
          masked_diff_map.as_1d().set_selected(selection, 0)
          f_000_s -= electrons
          continue
        if electrons < 0:
          masked_diff_map.as_1d().set_selected(selection, 0)
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
          approx_equal_relatively(previous_f_000_s, f_000_s, convergence)):
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

  def void_iselection(self, i):
    """ The grid points of void i, as indices into the mask's grid.

    Built once on first use (the mask is fixed), so summing a void costs a
    pass over the void, not over the grid.
    """
    if self._void_isel is None:
      self._void_isel = _void_iselections(self.mask.data, self.n_voids())
    return self._void_isel[i]

  def n_solvent_grid_points(self):
    # old: a pass over the grid per void; the flood fill has counted them
    # return sum([self.mask.data.count(i+2) for i in range(self.n_voids())
    #             if not self.exclude_void_flags[i]])
    per_void = self.flood_fill.grid_points_per_void()
    return sum([per_void[i] for i in range(self.n_voids())
                if not self.exclude_void_flags[i]])

  def electron_counts_per_void(self):
    if self._electron_counts_per_void is not None:
      return self._electron_counts_per_void
    self._electron_counts_per_void = []
    # old: rebuilt the map with a hard edge and no level, whatever
    # structure_factors had used; now counts on the map it built
    # masked_diff_map = self.diff_map.real_map_unpadded().set_selected(
    #   self.mask.data.as_double() == 0, 0)
    # for i in range(self.n_voids()):
    #   if self.exclude_void_flags[i]:
    #     f_000_s = 0
    #   else:
    #     diff_map = masked_diff_map.deep_copy().set_selected(
    #       self.mask.data != i+2, 0)
    #     f_000 = flex.sum(diff_map) * self.fft_scale
    #     f_000_s = f_000 * (
    #       self.crystal_gridding.n_grid_points() /
    #       (self.crystal_gridding.n_grid_points() - self.n_solvent_grid_points()))
    #   self._electron_counts_per_void.append(f_000_s)
    masked_diff_map = self._masked_diff_map
    if masked_diff_map is None: return self._electron_counts_per_void
    for i in range(self.n_voids()):
      if self.exclude_void_flags[i]:
        electrons = 0
      else:
        electrons = flex.sum(
          masked_diff_map.select(self.void_iselection(i))) * self.fft_scale
      self._electron_counts_per_void.append(electrons)
    # a smeared edge puts density outside the labelled voids; rescale so the
    # parts sum to the whole
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
