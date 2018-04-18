from __future__ import division

from dials.array_family import flex
from matplotlib import pyplot as plt

def get_spotfinder_stats(timestamps, n_strong, n_min,
  tuple_of_timestamp_boundaries, lengths, run_numbers):
  # n_strong is a flex.double and n_min is an integer
  iterator = xrange(len(timestamps))
  half_window = min(50, max(int(len(timestamps)//20), 1))
  # spotfinding more than n_min spots rate in a sliding window
  enough_spots = n_strong >= n_min
  enough_rate = flex.double()
  for i in iterator:
    window_min = max(0, i - half_window)
    window_max = min(i + half_window, len(timestamps))
    enough_local = enough_spots[window_min:window_max]
    enough_rate_local = enough_local.count(True)/len(enough_local)
    enough_rate.append(enough_rate_local)
  return (timestamps,
          n_strong,
          enough_rate,
          n_min,
          half_window*2,
          lengths,
          tuple_of_timestamp_boundaries,
          run_numbers)

def plot_spotfinder_stats(stats,
                          run_tags=[],
                          run_statuses=[],
                          minimalist=False,
                          interactive=True,
                          xsize=30,
                          ysize=10,
                          high_vis=False,
                          title=None,
                          ext='cbf',
                          ):
  plot_ratio = max(min(xsize, ysize)/2.5, 3)
  if high_vis:
    spot_ratio = plot_ratio*4
    text_ratio = plot_ratio*4.5
  else:
    spot_ratio = plot_ratio*2
    text_ratio = plot_ratio*3
  t, n_strong, enough_rate, n_min, window, lengths, boundaries, run_numbers = stats
  if len(t) == 0:
    return None
  n_runs = len(boundaries)//2
  if len(run_tags) != n_runs:
    run_tags = [[] for i in xrange(n_runs)]
  if len(run_statuses) != n_runs:
    run_statuses = [None for i in xrange(n_runs)]
  if minimalist:
    print "Minimalist mode activated."
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    axset = (ax1, ax2)
  else:
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
    axset = (ax1, ax2, ax3)
  for a in axset:
    a.tick_params(axis='x', which='both', bottom='off', top='off')
  ax1.scatter(t, n_strong, edgecolors="none", color ='magenta', s=spot_ratio)
  ax1.set_ylim(ymin=0)
  ax1.axis('tight')
  ax1.set_ylabel("# strong spots", fontsize=text_ratio)
  ax2.plot(t, enough_rate*100, color='orange')
  #test = n_strong > 1000
  #ax2.plot(t.select(test), n_strong.select(test), color='magenta')
  ax2.set_ylim(ymin=0)
  ax2.axis('tight')
  ax2.set_ylabel("%% images with\n>=%d spots" % n_min, fontsize=text_ratio)
  f.subplots_adjust(hspace=0)
  # add lines and text summaries at the timestamp boundaries
  if not minimalist:
    for boundary in boundaries:
      if boundary is not None:
        for a in (ax1, ax2):
          a.axvline(x=boundary, ymin=0, ymax=2, linewidth=1, color='k')
  run_starts = boundaries[0::2]
  run_ends = boundaries[1::2]
  start = 0
  end = -1
  for idx in xrange(len(run_numbers)):
    start_t = run_starts[idx]
    end_t = run_ends[idx]
    if start_t is None or end_t is None: continue
    end += lengths[idx]
    slice_t = t[start:end+1]
    slice_enough = n_strong[start:end+1] > n_min
    tags = run_tags[idx]
    status = run_statuses[idx]
    if status == "DONE":
      status_color = 'blue'
    elif status in ["RUN", "PEND", "SUBMITTED"]:
      status_color = 'green'
    elif status is None:
      status_color = 'black'
    else:
      status_color = 'red'
    if minimalist:
      ax2.set_xlabel("timestamp (s)\n# images shown as all (%3.1f Angstroms)" % d_min, fontsize=text_ratio)
      ax2.set_yticks([])
    else:
      ax3.text(start_t, 2.85, " " + ", ".join(tags) + " [%s]" % status, fontsize=text_ratio, color=status_color)
      ax3.text(start_t, .85, "run %d" % run_numbers[idx], fontsize=text_ratio)
      ax3.text(start_t, .65, "%d images" % lengths[idx], fontsize=text_ratio)
      ax3.text(start_t, .45, "%d n>=%d" % (slice_enough.count(True), n_min), fontsize=text_ratio)
      ax3.set_xlabel("timestamp (s)", fontsize=text_ratio)
      ax3.set_yticks([])
    for item in axset:
      item.tick_params(labelsize=text_ratio)
    start += lengths[idx]
  if title is not None:
    plt.title(title)
  f.set_size_inches(xsize, ysize)
  f.savefig("spotfinder_tmp.png", bbox_inches='tight', dpi=100)
  plt.close(f)
  return "spotfinder_tmp.png"

def plot_multirun_spotfinder_stats(runs,
                                   run_numbers,
                                   run_tags=[],
                                   run_statuses=[],
                                   n_min=4,
                                   minimalist=False,
                                   interactive=False,
                                   spotfinder_only=False,
                                   easy_run=False,
                                   compress_runs=True,
                                   xsize=30,
                                   ysize=10,
                                   high_vis=False,
                                   title=None):
  tset = flex.double()
  nset = flex.int()
  boundaries = []
  lengths = []
  runs_with_data = []
  offset = 0
  for idx in xrange(len(runs)):
    r = runs[idx]
    if len(r[0]) > 0:
      if compress_runs:
        tslice = r[0] - r[0][0] + offset
        offset += (r[0][-1] - r[0][0] + 1/120.)
      else:
        tslice = r[0]
      last_end = r[0][-1]
      tset.extend(tslice)
      nset.extend(r[1])
      boundaries.append(tslice[0])
      boundaries.append(tslice[-1])
      lengths.append(len(tslice))
      runs_with_data.append(run_numbers[idx])
    else:
      boundaries.extend([None]*2)
  stats_tuple = get_spotfinder_stats(tset,
                                     nset,
                                     n_min,
                                     tuple(boundaries),
                                     tuple(lengths),
                                     runs_with_data)
  if easy_run:
    from libtbx import easy_run, easy_pickle
    easy_pickle.dump("plot_spotfinder_stats_tmp.pickle", (stats_tuple, run_tags, run_statuses, minimalist, interactive, xsize, ysize, high_vis, title))
    result = easy_run.fully_buffered(command="cctbx.xfel.plot_spotfinder_stats_from_stats_pickle plot_spotfinder_stats_tmp.pickle")
    try:
      png = result.stdout_lines[-1]
      if png == "None":
        return None
    except Exception:
      return None
  else:
    png = plot_spotfinder_stats(stats_tuple, run_tags=run_tags, run_statuses=run_statuses, minimalist=minimalist,
      interactive=interactive, xsize=xsize, ysize=ysize, high_vis=high_vis, title=title)
  return png