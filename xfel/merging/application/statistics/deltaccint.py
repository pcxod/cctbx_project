from __future__ import absolute_import, division, print_function
from six.moves import range
from dxtbx import flumpy
from dials.array_family import flex
import itertools
import numpy as np
from scitbx.math import five_number_summary
from xfel.merging.application.reflection_table_utils import reflection_table_utils
from xfel.merging.application.worker import worker


class deltaccint(worker):
  '''Calculates ΔCC½ using the σ-τ method from Assmann 2016'''

  def __repr__(self):
    return 'ΔCC½ statistics (σ-τ method)'

  def run(self, experiments, reflections):
    self.logger.log_step_time("STATISTICS_DELTA_CCINT")

    comm = self.mpi_helper.comm
    MPI = self.mpi_helper.MPI

    assert experiments == None, "Must be run after the group worker"
    # need at least 3 reflections to keep multiplicty > 2 when removing an image
    filtered = flex.reflection_table()
    min_mult = max(3, self.params.merging.minimum_multiplicity)
    for refls in reflection_table_utils.get_next_hkl_reflection_table(reflections=reflections):
      if len(set(refls['id'])) >= min_mult:
        filtered.extend(refls)

    expt_map = filtered.experiment_identifiers()

    all_expt_ids = sorted(set(itertools.chain.from_iterable(comm.allgather(expt_map.values()))))
    all_expts_map = {v: k for k, v in enumerate(all_expt_ids)}

    if self.mpi_helper.rank == 0:
      self.logger.main_log("Beginning ΔCC½ analysis (σ-τ method from Assmann 2016)")
      self.logger.main_log("Removing reflections with less than %d measurements"%min_mult)
      self.logger.main_log("N experiments after filtering: %d"%len(all_expt_ids))
      self.logger.main_log("")

    # We need to compute
    # 1) variance of the average intensities -> compute averages of all intensities and then compute variance per bin
    # 2) average variance of the intensities -> compute variances of all intensities and then compute average per bin
    # In reality for 2), we compute the average of the standard error of the mean squared instead of the variance (semsq)

    hkl_resolution_bins = self.params.statistics.hkl_resolution_bins
    hkl_set = [hkl for hkl in set(filtered['miller_index_asymmetric']) if hkl in hkl_resolution_bins]
    n_hkl = len(hkl_set)
    hkl_map = {v: k for k, v in enumerate(hkl_set)}

    # N expts (all ranks) x N hkl (this rank)
    sums   = np.zeros((len(all_expt_ids), n_hkl), float)
    n      = np.zeros((len(all_expt_ids), n_hkl), int)
    merged = np.zeros((len(all_expt_ids), n_hkl), float)
    semsq  = np.zeros((len(all_expt_ids), n_hkl), float)

    for refls in reflection_table_utils.get_next_hkl_reflection_table(reflections=filtered):
      hkl = refls['miller_index_asymmetric'][0]
      if hkl not in hkl_map: continue
      hkl_idx = hkl_map[hkl]
      intensity = flumpy.to_numpy(refls['intensity.sum.value'])

      # set the sum and n for all expts for this hkl
      sums[:,hkl_idx] = np.sum(intensity)
      n   [:,hkl_idx] = len(refls)

      # For each reflection, subtract it off the experiment it came from, leaving it for the rest of the experiments
      for i in range(len(refls)):
        expt_idx = all_expts_map[expt_map[refls['id'][i]]]
        sums[expt_idx,hkl_idx] -= intensity[i]
        n   [expt_idx,hkl_idx] -= 1
      merged[:,hkl_idx] = sums[:,hkl_idx]/n[:,hkl_idx]

      # compute semsq for each hkl (less the intensity from each experiment)
      diff_sq_ = np.full(len(all_expt_ids), -1, dtype=float)

      for i in range(len(refls)):
        expt_idx = all_expts_map[expt_map[refls['id'][i]]]
        mean_intensity_modified = merged[expt_idx,hkl_idx]
        # Handle the case where a single image contains same hkl twice.
        if diff_sq_[expt_idx] <0:
          diff_sq_modified = np.sum((intensity-mean_intensity_modified)**2)
          diff_sq_[expt_idx] = diff_sq_modified - (intensity[i] - mean_intensity_modified)**2
        else:
          diff_sq_[expt_idx] -= (intensity[i] - mean_intensity_modified)**2

      mean_intensity_unmodified = np.mean(intensity)
      diff_sq_unmodified = np.sum((intensity-mean_intensity_unmodified)**2)
      diff_sq_[diff_sq_<0] = diff_sq_unmodified
      semsq[:,hkl_idx] = diff_sq_ / (n[:,hkl_idx]-1) / n[:,hkl_idx]

    # N expts (all ranks)
    all_i_sums     = np.zeros(len(all_expt_ids), float) # sum of the averaged intensities
    all_i_n        = np.zeros(len(all_expt_ids), int)   # count of the averaged intensities
    all_semsq_sums = np.zeros(len(all_expt_ids), float) # sums of the semsq of the intensities

    # Sum up and reduce the bins
    for hkl in hkl_set:
      all_i_sums     += merged[:,hkl_map[hkl]]
      all_i_n        += 1
      all_semsq_sums += semsq[:,hkl_map[hkl]]

    # Broadcast the semsq and average intensities
    total_i_sums     = comm.allreduce(all_i_sums, op=MPI.SUM)
    total_i_n        = comm.allreduce(all_i_n,    op=MPI.SUM)
    total_semsq_sums = comm.reduce(all_semsq_sums,  op=MPI.SUM)
    total_i_average  = total_i_sums / total_i_n

    # Compute the variance of the average intensities
    # First, the numerator, the difference between each hkl and the average for that hkl's bin (ommiting each experiment once)
    diff_sq = np.zeros(merged.shape, float)
    for hkl in hkl_set:
      diff_sq[:,hkl_map[hkl]] = (merged[:,hkl_map[hkl]] - total_i_average) ** 2

    # N expts (all ranks) x N bins
    diff_sq_sum = np.zeros(all_i_sums.shape, float)

    # Complete the numerator for this rank
    for hkl in hkl_set:
      diff_sq_sum += diff_sq[:,hkl_map[hkl]]

    total_diff_sq_sum = comm.reduce(diff_sq_sum, MPI.SUM, 0)

    # Report
    if self.mpi_helper.rank == 0:
      sigma_sq_y = total_diff_sq_sum / (total_i_n-1) # variance of the average intensities
      sigma_sq_e = 2 * total_semsq_sums / total_i_n    # average semsq of the intensities
      deltaccint_st = (sigma_sq_y - (0.5 * sigma_sq_e)) / (sigma_sq_y + (0.5 * sigma_sq_e))

      data = flex.double(deltaccint_st) * 100
      sorted_data = data.select(flex.sort_permutation(data))

      mini, q1, med, q3, maxi = five_number_summary(data)
      self.logger.main_log("Five number summary")
      self.logger.main_log("% 8.4f%% min"%mini)
      self.logger.main_log("% 8.4f%% q1"%q1)
      self.logger.main_log("% 8.4f%% median"%med)
      self.logger.main_log("% 8.4f%% q3"%q3)
      self.logger.main_log("% 8.4f%% max"%maxi)
      self.logger.main_log("")

      if self.params.statistics.deltaccint.verbose:
        self.logger.main_log("Showing ΔCC½ for all lattices")
        for e, identifier in enumerate(all_expt_ids):
          self.logger.main_log("%s %f"%(identifier, data[e]))

      n_worst = min(len(data), 30)
      worst = sorted_data[-n_worst:]
      iqr = q3-q1

      self.logger.main_log("Showing ΔCC½ of the worst %d lattices and IQR ratio needed to remove them"%n_worst)
      self.logger.main_log(" ΔCC½ (%) IQR ratio Lattices removed")
      for i in range(len(worst)):
        self.logger.main_log("% 8.4f % 10.1f %d"%(worst[i], (worst[i]-med)/iqr, n_worst-i))
      self.logger.main_log("")

      if self.params.statistics.deltaccint.iqr_ratio:
        cut = q3 + iqr * self.params.statistics.deltaccint.iqr_ratio
        sel = data > cut
        worst_expts_ = flex.std_string(all_expt_ids).select(sel)

        self.logger.main_log("Removing %d experiments out of %d using IQR ratio %.1f"%(len(worst_expts_), len(all_expt_ids), self.params.statistics.deltaccint.iqr_ratio))

    # Broadcast the worst experiments to cut
    else:
      worst_expts_ = None

    if self.params.statistics.deltaccint.iqr_ratio:
      worst_expts = comm.bcast(worst_expts_, 0)
      self.logger.log("Starting number of reflections: %d"%len(reflections))
      self.logger.log("Reflections after filtering by minimum multiplicity of %d: %d"%(min_mult, len(filtered)))
      reflections.remove_on_experiment_identifiers(worst_expts)
      reflections.reset_ids()
      self.logger.log("Reflections after filtering by ΔCC½: %d"%len(reflections))

    self.logger.log_step_time("STATISTICS_DELTA_CCINT", True)

    return experiments, reflections

if __name__ == '__main__':
  from xfel.merging.application.worker import exercise_worker
  exercise_worker(deltaccint)
