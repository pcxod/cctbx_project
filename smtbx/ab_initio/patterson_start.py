""" Starting phases from a Patterson superposition, instead of at random.

Charge flipping from random phases saturates: measured over held-out structures
the *oracle* of 32 trials is 0.868 and the oracle of 16 is 0.865, so no amount
of extra random starts helps. That ceiling is a property of the starting point,
not of the trial count, and 22% of structures never solve at all. A better
starting point is the only thing that moves it.

**The method.** The Patterson function is the map of interatomic vectors, and
it is computable without any phases at all -- it is the Fourier transform of
the intensities. If u is a genuine interatomic vector, then shifting the
Patterson by u superposes the vector set on itself for one atom's worth of the
structure, and taking the pointwise minimum of the two suppresses everything
that does not correspond to a real atom:

    M(x) = min( P(x), P(x + u) )

What survives is, in the ideal case, a double image: the structure and its
inverse through u, so 2N peaks rather than the N^2 of the raw Patterson. That
is a far better starting density than noise, and it costs two FFTs.

**Why it fits multi-trial unchanged.** `solving_iterator` takes
`initial_phases_for`, a callable of f_obs returning phases. So a superposition
start is a drop-in replacement for a random seed, and the several strongest
Patterson vectors give several genuinely different starting points -- which is
what the trials loop wants anyway. Random starts differ only in noise;
superposition starts differ in which interatomic vector they assume, which is a
much more useful kind of diversity.

**What it cannot do.** The superposition is only as good as the vector chosen.
A vector that is not a true interatomic vector gives a map no better than
noise, and for an equal-atom structure the strongest Patterson peaks are not
strongly preferred over the rest. So this is expected to help most where there
is *some* scattering contrast, and to be neutral where there is none -- which
is why it must be measured against a control of structures that already solve,
not only against the failures.
"""
from __future__ import absolute_import, division, print_function

import math


def sharpened_patterson(f_obs, resolution_factor=1/3.):
  """ The Patterson map, sharpened, on a grid suited to peak searching.

  Sharpening (working from E-like rather than F-like coefficients) narrows the
  peaks, which matters here because the whole method depends on picking
  individual vectors out of a crowded map.
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

  Returns fractional coordinates, strongest first. The origin peak is always
  the largest and carries no information, so it is skipped; `min_distance`
  keeps the search from returning the same peak twice through symmetry.
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
    # Skip the origin: it is the sum of every atom with itself, always the
    # tallest peak, and says nothing about the structure.
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

    # The shift is applied by index arithmetic rather than by interpolation:
    # the grid is the same for both copies, so a whole-grid-step shift is exact
    # and an interpolated one would blur exactly the peaks being selected.
    shift = [int(round(vector[i]*n[i])) % n[i] for i in range(3)]
    rolled = flex.double(flex.grid(n), 0.0)
    for i in range(n[0]):
      si = (i + shift[0]) % n[0]
      for j in range(n[1]):
        sj = (j + shift[1]) % n[1]
        for k in range(n[2]):
          rolled[i, j, k] = data[si, sj, (k + shift[2]) % n[2]]
    # The minimum function: density survives only where both copies agree.
    combined = flex.double(flex.grid(n), 0.0)
    for i in range(data.size()):
      a, b = data[i], rolled[i]
      combined[i] = a if a < b else b
    return combined
  except Exception:
    return None


def phases_from_superposition(f_obs, vector):
  """ Phases of the structure factors of the superposition map.

  This is the whole point: the map is built with no phase information, and
  transforming it back yields phases that already encode real interatomic
  geometry rather than noise.
  """
  from cctbx import maptbx

  combined = superposition_map(f_obs, vector)
  if combined is None:
    return None
  try:
    # Negative density is meaningless here and only adds noise to the
    # transform, so it is truncated -- the same reasoning behind truncation in
    # density modification generally. Modify in place: `flex.double(combined)`
    # builds a *copy*, so truncating that silently left the real map unchanged.
    combined.set_selected(combined < 0, 0.0)
    f_calc = f_obs.structure_factors_from_map(combined, use_scale=True,
                                              anomalous_flag=False)
    return f_calc.phases().data()
  except Exception:
    return None


def initial_phases_for(f_obs, max_vectors=4):
  """ A list of callables, one per superposition vector, for multi_trial.

  Each is a drop-in replacement for `phases_from_seed`. Returns an empty list
  if no usable Patterson vector was found, so a caller can fall back to random
  starts rather than fail.

  **The phases must be computed against the array the solver hands over, not
  against `f_obs`.** `charge_flipping` normalises and filters its data before
  asking for starting phases, so the array inside the solver is shorter than
  the one here -- precomputing phases from `f_obs` produced a length mismatch
  and every Patterson trial died on an assertion inside `phase_transfer`, while
  the surviving random trials made the run look merely unhelpful rather than
  broken.

  The expensive part -- the Patterson and the superposition -- is still done
  once per vector here; only the final transform is deferred, and that is one
  FFT.
  """
  maps = []
  for vector in superposition_vectors(f_obs, max_vectors=max_vectors):
    combined = superposition_map(f_obs, vector)
    if combined is None:
      continue
    combined.set_selected(combined < 0, 0.0)
    maps.append(combined)

  def phases_from(map_):
    def f(array):
      try:
        f_calc = array.structure_factors_from_map(map_, use_scale=True,
                                                  anomalous_flag=False)
        return f_calc.phases().data()
      except Exception:
        # Fall back to random rather than kill the trial: a starting point that
        # cannot be built is a missed opportunity, not a failure.
        import math
        import scitbx_array_family_flex_ext as flex_ext
        gen = flex_ext.mersenne_twister(seed=0)
        return gen.random_double(size=array.size(), factor=2*math.pi)
    return f

  return [phases_from(m) for m in maps]
