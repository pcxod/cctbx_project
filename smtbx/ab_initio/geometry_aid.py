""" Element proposals from local geometry, using a trained SOAP model.

The classical assignment in `element_assignment.py` reads integrated density,
which is a direct measure of how many electrons sit at a peak and knows nothing
about chemistry. It is therefore good at "heavy or light" and poor exactly where
chemistry is informative: C, N and O differ by one electron and overlap badly at
ordinary resolution.

This is the complementary signal. It describes each atom by the *arrangement*
of its neighbours -- a SOAP power spectrum within a cutoff -- and asks a trained
classifier which element has that environment. A carbonyl oxygen and a ring
carbon have nearly the same density in a 0.7 A sphere and completely different
surroundings.

**The two must be measured separately and then together.** Neither is the
answer on its own: geometry cannot tell bromine from iodine, density cannot
tell carbon from nitrogen. The point of running the classical assignment first
and this second is that the comparison is then a real one -- what does geometry
add to what density already knew -- rather than a claim that one method wins.

**No scikit-learn at run time.** The model is a mean vector, a PCA projection
and four dense layers; `export_geometry_aid.py` writes them to an .npz and this
evaluates them with numpy. Unpickling a scikit-learn model in Olex2 would tie
the plugin to one library version and would execute whatever the file contains.

**Preconditions, both of which have bitten.** The descriptor is meaningless
unless the coordinates are assembled into connected molecules first, because an
atom whose bonded neighbours live in another symmetry image has a nearly empty
environment -- see `assemble.py`. And the descriptor must be computed with the
same SOAP hyperparameters the model was trained on; a vector of the wrong length
is caught here, but one of the right length computed with a different cutoff
would not be, and would silently produce confident nonsense.
"""
from __future__ import absolute_import, division, print_function

import json
import os

from libtbx import group_args

# The hyperparameters the shipped models were trained with, from the greedy
# search in geometry-aid (`multi_layer_classifier/c_only_training.py`). They
# are recorded here because the feature count alone does not pin them down and
# a mismatch is undetectable downstream.
SOAP_HYPERPARAMETERS = {
  "cutoff": {"radius": 3.5,
             "smoothing": {"type": "ShiftedCosine", "width": 0.7}},
  "density": {"type": "Gaussian", "width": 0.2, "center_atom_weight": 1.0},
  "basis": {"type": "TensorProduct", "max_angular": 12,
            "radial": {"type": "Gto", "max_radial": 6},
            "spline_accuracy": 1e-6},
}

# 11 element types -> 66 unique pairs, 7 radial functions, 13 angular momenta.
EXPECTED_N_FEATURES = 66*7*7*13


class Model(object):
  """ The exported PCA + MLP, evaluated with numpy. """

  def __init__(self, path):
    import numpy as np

    data = np.load(path, allow_pickle=False)
    self.mean = data["pca_mean"]
    self.components = data["pca_components"]
    self.whiten_scale = None
    if "pca_explained_variance" in data:
      self.whiten_scale = np.sqrt(data["pca_explained_variance"])
    meta = json.loads(str(data["meta"]))
    self.classes = list(meta["classes"])
    self.n_features = int(meta["n_features_in"])
    self.layers = [(data["w%d" % i], data["b%d" % i])
                   for i in range(int(meta["n_layers"]))]
    if meta.get("activation") != "relu":
      raise ValueError("only relu hidden layers are supported, got %s"
                       % meta.get("activation"))
    if meta.get("out_activation") != "softmax":
      raise ValueError("only a softmax output is supported, got %s"
                       % meta.get("out_activation"))

  def probabilities(self, descriptors):
    """ (n_atoms, n_classes) class probabilities for a descriptor block. """
    import numpy as np

    x = np.asarray(descriptors, dtype=np.float64)
    if x.ndim == 1:
      x = x[None, :]
    if x.shape[1] != self.n_features:
      raise ValueError(
        "descriptor has %d features, the model expects %d -- these come from "
        "different SOAP hyperparameters, and the result would be meaningless "
        "rather than merely worse" % (x.shape[1], self.n_features))
    x = (x - self.mean).dot(self.components.T)
    if self.whiten_scale is not None:
      x = x/self.whiten_scale
    for w, b in self.layers[:-1]:
      x = np.maximum(x.dot(w) + b, 0.0)
    w, b = self.layers[-1]
    x = x.dot(w) + b
    # Softmax, shifted by the row maximum so a large logit cannot overflow.
    x = np.exp(x - x.max(axis=1, keepdims=True))
    return x/x.sum(axis=1, keepdims=True)

  def top_k(self, descriptors, k=3):
    """ [(element, probability)] per atom, best first. """
    import numpy as np

    probs = self.probabilities(descriptors)
    out = []
    for row in probs:
      order = np.argsort(row)[::-1][:k]
      out.append([(self.classes[j], float(row[j])) for j in order])
    return out


def descriptor_via_nospherA2(xyz_path, exe, work_dir=None):
  """ SOAP power spectrum from NoSpherA2, which Olex2 already ships.

  This is the dependency-free route: featomic is linked into the executable,
  so nothing needs installing next to Olex2.

  **Currently unusable with the shipped models**, and deliberately not hidden.
  The build of 30 Jul 2026 hard-codes the older hyperparameters from
  `external_script.py` (cutoff 2.5, density width 0.15, max_angular 9,
  max_radial 4) and produces 16,500 features, while every trained model in
  geometry-aid was fitted on the greedy-search values above, which give 42,042.
  Passing numbers after the flag does not change it. The fix belongs in
  NoSpherA2 -- either update the constants or expose them -- and until then the
  length check in `Model.probabilities` is what stops a silent mismatch.
  """
  import subprocess

  import numpy as np

  work_dir = work_dir or os.path.dirname(os.path.abspath(xyz_path))
  subprocess.check_call(
    [exe, "-wfn", os.path.abspath(xyz_path), "-calc_featomic_descriptor"],
    cwd=work_dir)
  out = os.path.join(work_dir, "descriptor.npy")
  if not os.path.exists(out):
    raise RuntimeError("NoSpherA2 produced no descriptor.npy")
  return np.load(out)


Z_OF = dict(B=5, C=6, N=7, O=8, F=9, Si=14, P=15, S=16, Cl=17, Br=35, I=53)


def density_likelihood(observed, elements, sigma):
  """ p(observed density | element) over `elements`, normalised.

  The density estimate is a scalar on the carbon-calibrated scale, and
  `element_assignment.expected_density` says where each element should sit on
  it. A Gaussian around that expectation turns "sulfur reads 22.2" into a
  distribution rather than a single winner, which is what makes it combinable
  with the classifier's output.

  `sigma` scales with the expectation. Absolute width would be wrong: a whole
  electron of error means something very different at carbon than at bromine,
  and a fixed sigma either makes C/N/O a coin toss or makes every heavy call
  certain.
  """
  import math

  from smtbx.ab_initio import element_assignment

  weights = []
  for symbol in elements:
    expected = element_assignment.expected_density(Z_OF[symbol])
    width = max(1e-6, sigma*expected)
    weights.append(math.exp(-0.5*((observed - expected)/width)**2))
  total = sum(weights)
  if total <= 0:
    return [1.0/len(elements)]*len(elements)
  return [w/total for w in weights]


def combine(classical, proposals, geometry_weight=0.5, sigma=0.15,
            elements=None):
  """ One element per atom, from the density call and the geometry call.

  The two are combined as **distributions, not as a preference order**. An
  earlier version routed light elements to geometry and heavy ones to density,
  which scored *between* the two methods rather than above either: whichever
  source it ignored for an atom, it threw that atom's other opinion away.

  Here density becomes a likelihood over the candidate elements and the
  classifier supplies a posterior; the product is renormalised and the
  argmax taken. The two disagree in different places -- density separates
  Br from I and cannot see C from N, geometry the reverse -- so multiplying is
  the operation that lets each break the other's ties.

  `geometry_weight` raises the classifier term to a power: 0 recovers density
  alone, a large value recovers geometry alone, and 1 weights them equally.
  One knob spanning both baselines makes the comparison a scan rather than
  three unrelated numbers.

  **Default 0.5, from measurement.** It was 1.0 because that is the neutral
  value, not because anything had been measured -- the scan had never been run.
  Over 5,473 COD structures and 131,345 scorable atoms, 0.5 is the best value
  on every metric at once:

      weight   atom    exact   <=1 wrong   <=2 wrong
      0.00    0.6941   0.0682    0.1325      0.2143     density alone
      0.25    0.7592   0.0753    0.1588      0.2803
      0.50    0.7706   0.0780    0.1597      0.2794     <- here
      1.00    0.7624   0.0703    0.1513      0.2595     <- was here
      2.00    0.7522   0.0621    0.1370      0.2439
      4.00    0.7445   0.0546    0.1268      0.2289

  Equally weighted was over-trusting the classifier: geometry is worth +0.077
  per atom over density alone at 0.5, and 1.0 gave back a fifth of that. The
  gain from the change itself is +0.008 per atom and +0.008 on the fraction of
  structures within one misassignment.

  Atoms whose density call is outside the classifier's element set keep the
  density answer -- the classifier cannot express it, so there is nothing to
  combine.
  """
  candidates = list(elements or sorted(Z_OF, key=lambda s: Z_OF[s]))
  out = []
  for i, call in enumerate(classical):
    density_element = call.element
    top = proposals[i] if i < len(proposals) else []
    observed = getattr(call, "z_estimate", None)
    geometry_element = top[0][0] if top else None

    if not top or observed is None or density_element not in Z_OF:
      out.append(group_args(
        element=density_element, from_density=density_element,
        from_geometry=geometry_element, probability=0.0, used="density",
        marginal=bool(getattr(call, "marginal", False)), top_k=top))
      continue

    posterior = dict(top)
    prior = density_likelihood(observed, candidates, sigma)
    scored = []
    for symbol, p_density in zip(candidates, prior):
      # Absent from the top-k means small, not impossible; a floor keeps a
      # confident density call from being vetoed by a truncated shortlist.
      p_geometry = max(posterior.get(symbol, 0.0), 1e-4)
      scored.append((p_density*(p_geometry**geometry_weight), symbol))
    scored.sort(reverse=True)
    best_score, best = scored[0]
    total = sum(s for s, _ in scored) or 1.0
    out.append(group_args(
      element=best, from_density=density_element,
      from_geometry=geometry_element, probability=best_score/total,
      used=("density" if best == density_element else
            "geometry" if best == geometry_element else "combined"),
      marginal=bool(getattr(call, "marginal", False)), top_k=top))
  return out
