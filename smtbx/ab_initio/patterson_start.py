""" Starting phases from a Patterson superposition, instead of at random.

Random starts saturate: more of them do not solve more structures, so only a
better starting point helps. Shifting the Patterson by a genuine interatomic
vector u and taking

    M(x) = min( P(x), P(x + u) )

suppresses everything that is not an atom, leaving ideally a double image (the
structure and its inverse through u, 2N peaks instead of N^2) for two FFTs.
`solving_iterator.initial_phases_for` takes a callable of f_obs, so each strong
Patterson vector gives a different drop-in start. A vector that is not
interatomic gives no better than noise, and equal-atom structures have no
strongly preferred vectors, so this can only help where there is scattering
contrast.
"""
from __future__ import absolute_import, division, print_function


def sharpened_patterson(f_obs, resolution_factor=1/3.):
  """ The Patterson map, sharpened, on a grid suited to peak searching.

  Sharpening (E-like coefficients) narrows the peaks so single vectors can be
  picked out of a crowded map.
  """
  from cctbx import maptbx

  try:
    normalisations = f_obs.amplitude_quasi_normalisations()
    e = f_obs.customized_copy(data=f_obs.data()/normalisations.data())
  except Exception:
    e = f_obs
  patterson = e.patterson_map(resolution_factor=resolution_factor,
                              symmetry_flags=maptbx.use_space_group_symmetry)
  patterson.apply_sigma_scaling()
  return patterson


def superposition_vectors(f_obs, max_vectors=4, min_distance=1.0):
  """ The strongest non-origin Patterson peaks, as candidate atom vectors.

  Returns fractional coordinates, strongest first, without the origin peak;
  `min_distance` keeps a peak from coming back twice through symmetry.
  """
  from cctbx import maptbx

  patterson = sharpened_patterson(f_obs)
  peaks = patterson.peak_search(
    parameters=maptbx.peak_search_parameters(
      min_distance_sym_equiv=min_distance,
      max_clusters=max_vectors + 4),
    verify_symmetry=False).all()

  out = []
  for site, height in zip(peaks.sites(), peaks.heights()):
    if f_obs.unit_cell().length(site) < min_distance:
      continue
    out.append(tuple(site))
    if len(out) >= max_vectors:
      break
  return out


def superposition_map(f_obs, vector):
  """ min(P(x), P(x+u)) as a real map, or None if it cannot be built. """
  from cctbx.array_family import flex

  try:
    patterson = sharpened_patterson(f_obs)
    data = patterson.real_map_unpadded()
    n = data.accessor().all()

    # a whole-grid-step shift: exact, where interpolation would blur the peaks
    shift = [int(round(vector[i]*n[i])) % n[i] for i in range(3)]
    rolled = flex.double(flex.grid(n), 0.0)
    for i in range(n[0]):
      si = (i + shift[0]) % n[0]
      for j in range(n[1]):
        sj = (j + shift[1]) % n[1]
        for k in range(n[2]):
          rolled[i, j, k] = data[si, sj, (k + shift[2]) % n[2]]
    combined = flex.double(flex.grid(n), 0.0)
    for i in range(data.size()):
      a, b = data[i], rolled[i]
      combined[i] = a if a < b else b
    return combined
  except Exception:
    return None


def initial_phases_for(f_obs, max_vectors=4):
  """ A list of callables, one per superposition vector, for multi_trial.

  Each is a drop-in replacement for `phases_from_seed`. Returns an empty list
  if no usable Patterson vector was found, so a caller can fall back to random
  starts rather than fail.

  The phases are computed against the array the solver hands over, which
  `charge_flipping` has normalised and filtered and so is shorter than `f_obs`;
  only that final transform is deferred, the superposition maps are built here.
  """
  maps = []
  for vector in superposition_vectors(f_obs, max_vectors=max_vectors):
    combined = superposition_map(f_obs, vector)
    if combined is None:
      continue
    # negative density only adds noise to the transform
    combined.set_selected(combined < 0, 0.0)
    maps.append(combined)

  def phases_from(map_):
    def f(array):
      try:
        f_calc = array.structure_factors_from_map(map_, use_scale=True,
                                                  anomalous_flag=False)
        return f_calc.phases().data()
      except Exception:
        # random phases rather than a dead trial
        import math
        import scitbx_array_family_flex_ext as flex_ext
        gen = flex_ext.mersenne_twister(seed=0)
        return gen.random_double(size=array.size(), factor=2*math.pi)
    return f

  return [phases_from(m) for m in maps]
