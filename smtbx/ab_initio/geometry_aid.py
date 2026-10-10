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

  # below the lightest candidate every element is far away, and the relative
  # width would hand a noise peak to the heaviest one
  observed = max(observed, min(element_assignment.expected_density(Z_OF[s])
                               for s in elements))
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


# Chemistry priors measured on the CSD (971796 ordered 3D entries, 40.4 M heavy
# atoms, CSD bond assignment, 19 Sep 2026). P(n heavy neighbours | element),
# index 6 holds six and more.
NEIGHBOUR_COUNT = {
  'C': (0.000, 0.128, 0.525, 0.312, 0.034, 0.001, 0.001),
  'N': (0.001, 0.111, 0.332, 0.520, 0.035, 0.000, 0.000),
  'O': (0.044, 0.518, 0.373, 0.058, 0.006, 0.001, 0.000),
  'F': (0.001, 0.989, 0.009, 0.001, 0.000, 0.000, 0.000),
  'Cl': (0.046, 0.829, 0.066, 0.007, 0.052, 0.000, 0.000),
  'Br': (0.058, 0.810, 0.112, 0.017, 0.002, 0.000, 0.000),
  'I': (0.065, 0.616, 0.239, 0.070, 0.009, 0.001, 0.001),
  'S': (0.000, 0.108, 0.532, 0.146, 0.211, 0.002, 0.001),
  'P': (0.000, 0.004, 0.015, 0.087, 0.820, 0.006, 0.068),
  'B': (0.009, 0.022, 0.021, 0.152, 0.253, 0.489, 0.054),
  'Si': (0.000, 0.007, 0.007, 0.028, 0.936, 0.013, 0.009),
  'Se': (0.000, 0.147, 0.538, 0.249, 0.044, 0.009, 0.011),
}
# P(element | one heavy neighbour of type X)
TERMINAL_ON = {
  'C': {'C': 0.504, 'O': 0.269, 'F': 0.110, 'N': 0.046, 'Cl': 0.044, 'Br': 0.014, 'S': 0.007},
  'N': {'C': 0.480, 'O': 0.396, 'N': 0.033, 'Cu': 0.017, 'Zn': 0.014, 'Cd': 0.013, 'Co': 0.010, 'Ag': 0.009},
  'O': {'C': 0.698, 'Zn': 0.044, 'Cu': 0.030, 'Co': 0.024, 'Cd': 0.020, 'Mn': 0.018, 'Na': 0.011, 'Ag': 0.008},
  'S': {'O': 0.807, 'C': 0.158, 'F': 0.012, 'N': 0.006},
  'P': {'F': 0.510, 'C': 0.237, 'O': 0.193, 'S': 0.022, 'Cl': 0.021, 'B': 0.007, 'Se': 0.006},
  'Cl': {'O': 0.971, 'Cu': 0.008, 'Pb': 0.005},
  'B': {'F': 0.790, 'O': 0.072, 'Cl': 0.054, 'C': 0.035, 'Br': 0.031, 'I': 0.013},
  'Si': {'C': 0.953, 'F': 0.017, 'Cl': 0.016, 'O': 0.010},
}
# P(element | bonded to this metal)
METAL_DONOR = {
  'Cu': {'N': 0.378, 'O': 0.358, 'Cl': 0.062, 'I': 0.053, 'S': 0.052, 'Br': 0.030, 'P': 0.028, 'C': 0.023},
  'Fe': {'C': 0.628, 'N': 0.163, 'O': 0.100, 'S': 0.042, 'P': 0.026, 'Cl': 0.020},
  'Co': {'O': 0.364, 'N': 0.318, 'C': 0.207, 'Cl': 0.029, 'P': 0.028, 'S': 0.024, 'B': 0.011, 'Br': 0.006},
  'Mo': {'O': 0.642, 'C': 0.188, 'S': 0.051, 'N': 0.050, 'P': 0.023, 'Cl': 0.021, 'Br': 0.008, 'I': 0.006},
  'Zn': {'O': 0.517, 'N': 0.362, 'Cl': 0.048, 'S': 0.027, 'C': 0.018, 'Br': 0.013, 'I': 0.008},
  'Ru': {'C': 0.576, 'N': 0.148, 'P': 0.081, 'O': 0.070, 'Cl': 0.064, 'S': 0.035, 'B': 0.010},
  'Ni': {'N': 0.378, 'O': 0.332, 'C': 0.104, 'S': 0.086, 'P': 0.047, 'Cl': 0.024, 'Br': 0.015},
  'W': {'O': 0.675, 'C': 0.203, 'N': 0.032, 'S': 0.030, 'P': 0.023, 'Cl': 0.018, 'I': 0.005},
  'Mn': {'O': 0.476, 'N': 0.266, 'C': 0.166, 'Cl': 0.033, 'Br': 0.018, 'S': 0.016, 'P': 0.013},
  'Cd': {'O': 0.538, 'N': 0.300, 'Cl': 0.069, 'S': 0.043, 'Br': 0.020, 'I': 0.016, 'Se': 0.007},
  'Ag': {'N': 0.316, 'O': 0.260, 'S': 0.127, 'C': 0.107, 'I': 0.054, 'P': 0.052, 'Br': 0.029, 'Cl': 0.025},
  'Pd': {'N': 0.269, 'C': 0.200, 'Cl': 0.163, 'P': 0.136, 'O': 0.092, 'S': 0.072, 'Br': 0.027, 'I': 0.019},
  'Ti': {'C': 0.427, 'O': 0.343, 'N': 0.115, 'Cl': 0.070, 'F': 0.017, 'S': 0.013, 'P': 0.008},
  'Sn': {'C': 0.354, 'O': 0.287, 'N': 0.105, 'Cl': 0.091, 'S': 0.057, 'Se': 0.030, 'I': 0.026, 'Br': 0.020},
  'Pt': {'C': 0.258, 'N': 0.245, 'P': 0.164, 'Cl': 0.118, 'S': 0.084, 'O': 0.066, 'I': 0.021, 'Br': 0.016},
  'V': {'O': 0.766, 'N': 0.101, 'C': 0.075, 'S': 0.018, 'Cl': 0.018, 'F': 0.013, 'P': 0.006},
  'Rh': {'C': 0.530, 'N': 0.119, 'P': 0.103, 'O': 0.102, 'Cl': 0.074, 'S': 0.033, 'B': 0.018, 'I': 0.008},
  'K': {'O': 0.686, 'N': 0.125, 'C': 0.116, 'F': 0.022, 'S': 0.017, 'Cl': 0.014},
  'Re': {'C': 0.403, 'N': 0.158, 'O': 0.133, 'Cl': 0.077, 'S': 0.073, 'P': 0.067, 'Se': 0.041, 'Br': 0.026},
  'Pb': {'O': 0.332, 'I': 0.261, 'Br': 0.192, 'N': 0.099, 'Cl': 0.067, 'S': 0.023, 'C': 0.020},
  'Cr': {'C': 0.550, 'O': 0.192, 'N': 0.145, 'Cl': 0.033, 'P': 0.030, 'S': 0.024, 'F': 0.008, 'Se': 0.006},
  'Ir': {'C': 0.573, 'N': 0.169, 'P': 0.088, 'Cl': 0.063, 'O': 0.046, 'S': 0.029, 'B': 0.012, 'I': 0.008},
  'U': {'O': 0.640, 'C': 0.168, 'N': 0.120, 'Cl': 0.026, 'F': 0.016, 'S': 0.009, 'I': 0.008},
  'Na': {'O': 0.812, 'N': 0.102, 'C': 0.045, 'S': 0.012, 'F': 0.011, 'Cl': 0.007},
  'Zr': {'C': 0.556, 'O': 0.230, 'N': 0.095, 'Cl': 0.075, 'F': 0.015, 'P': 0.012, 'S': 0.005},
  'Li': {'O': 0.517, 'N': 0.256, 'C': 0.138, 'Cl': 0.028, 'P': 0.015, 'S': 0.012, 'F': 0.008, 'Br': 0.007},
  'Al': {'O': 0.361, 'C': 0.288, 'N': 0.198, 'Cl': 0.076, 'F': 0.020, 'Br': 0.018, 'P': 0.011, 'S': 0.009},
  'Dy': {'O': 0.774, 'N': 0.140, 'C': 0.063, 'Cl': 0.011},
  'Sb': {'F': 0.272, 'C': 0.195, 'Cl': 0.164, 'O': 0.149, 'Br': 0.060, 'S': 0.060, 'I': 0.043, 'N': 0.039},
  'Eu': {'O': 0.802, 'N': 0.142, 'C': 0.023, 'Cl': 0.013, 'S': 0.005},
  'Os': {'C': 0.713, 'N': 0.078, 'P': 0.068, 'O': 0.043, 'Cl': 0.037, 'S': 0.035, 'Br': 0.006, 'B': 0.006},
  'Au': {'C': 0.290, 'P': 0.219, 'S': 0.167, 'Cl': 0.137, 'N': 0.100, 'Br': 0.029, 'I': 0.017, 'O': 0.014},
  'Gd': {'O': 0.829, 'N': 0.105, 'C': 0.042, 'Cl': 0.012},
  'Tb': {'O': 0.846, 'N': 0.116, 'C': 0.024, 'Cl': 0.010},
  'Sm': {'O': 0.592, 'C': 0.221, 'N': 0.138, 'Cl': 0.018, 'I': 0.008, 'S': 0.007},
  'Nd': {'O': 0.747, 'N': 0.116, 'C': 0.095, 'Cl': 0.019, 'S': 0.007, 'I': 0.005},
  'La': {'O': 0.705, 'C': 0.124, 'N': 0.114, 'I': 0.018, 'Cl': 0.017, 'S': 0.006, 'Br': 0.006},
  'Bi': {'O': 0.284, 'I': 0.203, 'Cl': 0.146, 'Br': 0.129, 'C': 0.097, 'N': 0.069, 'S': 0.059},
  'Hg': {'Cl': 0.201, 'N': 0.193, 'S': 0.140, 'O': 0.114, 'C': 0.103, 'I': 0.092, 'Br': 0.082, 'Se': 0.034},
  'Y': {'O': 0.481, 'C': 0.296, 'N': 0.172, 'Cl': 0.023, 'I': 0.008, 'B': 0.005, 'S': 0.005},
  'Mg': {'O': 0.609, 'N': 0.231, 'C': 0.090, 'Cl': 0.027, 'Br': 0.017, 'I': 0.006, 'P': 0.006},
  'Yb': {'O': 0.507, 'C': 0.260, 'N': 0.183, 'Cl': 0.021, 'S': 0.009, 'I': 0.006},
  'Ce': {'O': 0.679, 'N': 0.137, 'C': 0.129, 'Cl': 0.019, 'I': 0.014, 'Br': 0.009, 'S': 0.006},
  'Er': {'O': 0.748, 'N': 0.118, 'C': 0.099, 'Cl': 0.013, 'B': 0.007},
  'Ga': {'O': 0.274, 'C': 0.230, 'N': 0.196, 'Cl': 0.142, 'S': 0.042, 'P': 0.026, 'I': 0.023, 'Br': 0.021},
  'Ge': {'C': 0.331, 'O': 0.244, 'N': 0.140, 'Cl': 0.064, 'S': 0.061, 'Si': 0.036, 'I': 0.032, 'Se': 0.024},
  'Ca': {'O': 0.793, 'N': 0.119, 'C': 0.060, 'Cl': 0.007, 'I': 0.006},
  'Pr': {'O': 0.810, 'N': 0.094, 'C': 0.054, 'Cl': 0.024, 'I': 0.007},
  'Ba': {'O': 0.802, 'N': 0.110, 'C': 0.049, 'Cl': 0.009, 'F': 0.008, 'S': 0.006, 'I': 0.006},
  'In': {'O': 0.349, 'N': 0.158, 'Cl': 0.126, 'C': 0.120, 'S': 0.083, 'Br': 0.052, 'Se': 0.031, 'Te': 0.029},
  'Nb': {'C': 0.323, 'O': 0.299, 'Cl': 0.168, 'N': 0.088, 'S': 0.043, 'F': 0.023, 'P': 0.021, 'Se': 0.010},
  'Ta': {'C': 0.375, 'O': 0.170, 'N': 0.148, 'Cl': 0.144, 'S': 0.045, 'P': 0.038, 'F': 0.036, 'B': 0.017},
  'Sr': {'O': 0.800, 'N': 0.125, 'C': 0.035, 'I': 0.010, 'P': 0.008, 'Cl': 0.007},
  'Ho': {'O': 0.801, 'N': 0.117, 'C': 0.057, 'Cl': 0.010},
  'Cs': {'O': 0.660, 'N': 0.092, 'C': 0.090, 'Cl': 0.031, 'I': 0.031, 'Br': 0.028, 'F': 0.028, 'S': 0.014},
  'Sc': {'C': 0.500, 'O': 0.311, 'N': 0.127, 'Cl': 0.024, 'P': 0.011, 'I': 0.009, 'F': 0.009},
  'Th': {'C': 0.479, 'O': 0.306, 'N': 0.137, 'Cl': 0.028, 'S': 0.016, 'P': 0.009, 'Se': 0.008, 'F': 0.006},
  'Hf': {'C': 0.481, 'O': 0.230, 'N': 0.148, 'Cl': 0.072, 'F': 0.032, 'P': 0.013, 'S': 0.006, 'B': 0.005},
  'Rb': {'O': 0.703, 'N': 0.117, 'C': 0.078, 'F': 0.021, 'Br': 0.020, 'S': 0.015, 'I': 0.015, 'Cl': 0.014},
  'Lu': {'O': 0.435, 'C': 0.336, 'N': 0.183, 'Cl': 0.026, 'P': 0.008, 'B': 0.005},
  'Tl': {'O': 0.353, 'N': 0.205, 'C': 0.184, 'S': 0.109, 'Cl': 0.057, 'Br': 0.033, 'P': 0.019, 'Se': 0.013},
  'Tm': {'O': 0.672, 'C': 0.167, 'N': 0.124, 'Cl': 0.009, 'S': 0.007, 'I': 0.007, 'P': 0.006},
  'Tc': {'O': 0.232, 'N': 0.203, 'C': 0.193, 'Cl': 0.123, 'S': 0.110, 'P': 0.091, 'Br': 0.032, 'Se': 0.008},
  'Np': {'O': 0.730, 'N': 0.108, 'C': 0.069, 'Cl': 0.065, 'S': 0.010, 'F': 0.009},
  'Be': {'O': 0.399, 'N': 0.207, 'C': 0.179, 'Cl': 0.097, 'F': 0.033, 'Br': 0.032, 'I': 0.022, 'P': 0.017},
  'Pu': {'O': 0.708, 'Cl': 0.110, 'N': 0.078, 'C': 0.071, 'Se': 0.011, 'I': 0.010, 'S': 0.007},
  'Am': {'O': 0.713, 'N': 0.092, 'C': 0.077, 'S': 0.058, 'Se': 0.027, 'Cl': 0.019, 'Br': 0.010},
  'Cm': {'O': 0.608, 'N': 0.304, 'S': 0.082, 'Cl': 0.006},
  'Bk': {'C': 0.474, 'O': 0.437, 'N': 0.089},
}
# metal -> donor element -> (median bond length, robust sd, count); bonded = covalent
# radii sum + 0.45 A, 12 of 74 CSD export parts (bond_windows.py), pairs under 50 dropped
BOND_WINDOW = {
  'Ag': {'N': (2.243, 0.141, 16780), 'As': (2.67, 0.101, 207), 'O': (2.388, 0.131, 10713), 'P': (2.431, 0.061, 2915), 'C': (2.269, 0.245, 8036), 'Cl': (2.64, 0.145, 824), 'Br': (2.735, 0.11, 1501), 'S': (2.536, 0.091, 8386), 'Se': (2.628, 0.05, 978), 'I': (2.864, 0.08, 3378)},
  'Al': {'N': (1.926, 0.088, 5009), 'C': (1.994, 0.053, 6728), 'O': (1.827, 0.118, 12682), 'I': (2.549, 0.05, 100), 'S': (2.245, 0.093, 166), 'Cl': (2.139, 0.05, 1221), 'P': (2.45, 0.075, 382), 'Br': (2.312, 0.052, 209), 'F': (1.772, 0.061, 528), 'Si': (2.488, 0.05, 164)},
  'Am': {'S': (2.895, 0.05, 64), 'O': (2.478, 0.086, 67)},
  'Au': {'I': (2.575, 0.05, 366), 'P': (2.28, 0.05, 4213), 'Br': (2.414, 0.05, 475), 'N': (2.03, 0.05, 2085), 'Cl': (2.283, 0.05, 2413), 'C': (2.02, 0.05, 5569), 'S': (2.309, 0.05, 2586), 'O': (2.04, 0.05, 198), 'Si': (2.405, 0.05, 61), 'As': (2.407, 0.05, 116), 'Se': (2.43, 0.05, 90)},
  'Ba': {'O': (2.808, 0.097, 8257), 'C': (3.127, 0.167, 1493), 'P': (3.426, 0.156, 85), 'N': (2.895, 0.116, 1197), 'Si': (3.632, 0.064, 73), 'Cl': (3.227, 0.167, 131), 'F': (2.957, 0.141, 130), 'S': (3.232, 0.064, 321)},
  'Be': {'O': (1.624, 0.05, 763), 'Cl': (2.035, 0.08, 262), 'N': (1.723, 0.05, 277), 'C': (1.887, 0.072, 142)},
  'Bi': {'O': (2.345, 0.176, 4010), 'S': (2.769, 0.144, 475), 'N': (2.468, 0.138, 1005), 'C': (2.258, 0.05, 1169), 'Cl': (2.683, 0.162, 1541), 'Br': (2.841, 0.153, 1522), 'I': (3.064, 0.185, 1656), 'Si': (2.605, 0.05, 91), 'P': (2.65, 0.05, 58)},
  'Bk': {'C': (2.637, 0.05, 60)},
  'Ca': {'N': (2.427, 0.119, 1982), 'O': (2.404, 0.095, 7613), 'Si': (3.203, 0.111, 75), 'C': (2.778, 0.129, 1843), 'P': (3.075, 0.085, 179), 'I': (3.118, 0.065, 115), 'Cl': (2.747, 0.05, 103), 'S': (2.913, 0.05, 53)},
  'Cd': {'O': (2.319, 0.083, 28528), 'N': (2.332, 0.053, 18863), 'S': (2.59, 0.097, 2845), 'I': (2.786, 0.085, 763), 'Cl': (2.62, 0.087, 3423), 'Br': (2.707, 0.159, 777), 'Se': (2.633, 0.05, 424), 'P': (2.571, 0.05, 51), 'C': (2.203, 0.05, 411), 'As': (2.644, 0.124, 58)},
  'Ce': {'N': (2.665, 0.14, 2409), 'O': (2.493, 0.112, 11656), 'C': (2.821, 0.103, 2909), 'P': (3.459, 0.144, 110), 'S': (2.946, 0.05, 258), 'Cl': (2.786, 0.155, 135), 'Si': (3.449, 0.117, 110), 'I': (3.204, 0.091, 132), 'As': (3.303, 0.254, 64)},
  'Cm': {'S': (2.876, 0.05, 72)},
  'Co': {'O': (2.072, 0.077, 52338), 'N': (2.039, 0.135, 41043), 'S': (2.246, 0.05, 2931), 'Cl': (2.287, 0.075, 3078), 'C': (2.029, 0.097, 26633), 'Br': (2.403, 0.05, 384), 'I': (2.585, 0.05, 256), 'F': (2.036, 0.05, 302), 'P': (2.209, 0.069, 4897), 'As': (2.374, 0.088, 728), 'Se': (2.352, 0.05, 222), 'Si': (2.231, 0.071, 118), 'Te': (2.526, 0.05, 124)},
  'Cr': {'C': (2.156, 0.149, 11015), 'Cl': (2.33, 0.063, 848), 'N': (2.057, 0.051, 4514), 'O': (1.96, 0.05, 10169), 'F': (1.916, 0.05, 1657), 'As': (2.468, 0.091, 93), 'P': (2.373, 0.092, 769), 'Si': (2.79, 0.09, 53), 'S': (2.352, 0.078, 427), 'Se': (2.54, 0.05, 67)},
  'Cs': {'Br': (3.652, 0.14, 177), 'C': (3.53, 0.114, 1695), 'O': (3.186, 0.161, 6011), 'F': (3.232, 0.178, 344), 'S': (3.656, 0.119, 126), 'N': (3.27, 0.126, 693), 'P': (3.81, 0.126, 88), 'Cl': (3.511, 0.108, 322), 'As': (3.844, 0.141, 63), 'I': (3.939, 0.12, 78)},
  'Cu': {'C': (2.004, 0.133, 6567), 'N': (2.006, 0.052, 72617), 'O': (1.965, 0.05, 72173), 'P': (2.254, 0.05, 6026), 'S': (2.288, 0.064, 10137), 'I': (2.664, 0.062, 12057), 'Cl': (2.299, 0.099, 8634), 'Br': (2.451, 0.1, 4618), 'Se': (2.438, 0.084, 1879), 'Te': (2.651, 0.087, 659), 'F': (2.172, 0.228, 162), 'Si': (2.355, 0.136, 95), 'As': (2.388, 0.051, 512)},
  'Dy': {'O': (2.364, 0.079, 24466), 'C': (2.713, 0.135, 3678), 'N': (2.542, 0.094, 6138), 'Cl': (2.699, 0.077, 427), 'S': (2.881, 0.116, 223), 'F': (2.209, 0.155, 136), 'Si': (3.324, 0.123, 50), 'I': (3.058, 0.05, 87), 'Br': (2.859, 0.05, 84)},
  'Er': {'N': (2.491, 0.107, 2337), 'O': (2.341, 0.078, 10752), 'C': (2.638, 0.151, 2480), 'P': (3.335, 0.063, 97), 'Cl': (2.627, 0.077, 277), 'S': (2.885, 0.051, 130), 'Br': (2.861, 0.05, 152)},
  'Eu': {'N': (2.598, 0.089, 4201), 'O': (2.417, 0.081, 22205), 'C': (2.875, 0.085, 2247), 'I': (3.253, 0.052, 79), 'S': (2.954, 0.1, 306), 'P': (3.471, 0.05, 67), 'Cl': (2.761, 0.108, 143)},
  'Fe': {'N': (2.046, 0.138, 37694), 'C': (2.041, 0.05, 93173), 'O': (2.014, 0.074, 27091), 'Cl': (2.266, 0.086, 3770), 'P': (2.251, 0.064, 5998), 'Br': (2.424, 0.068, 568), 'S': (2.272, 0.05, 8664), 'As': (2.437, 0.081, 591), 'F': (1.956, 0.078, 336), 'Se': (2.374, 0.05, 562), 'I': (2.628, 0.05, 206), 'Te': (2.57, 0.05, 234), 'Si': (2.303, 0.057, 215)},
  'Ga': {'Cl': (2.176, 0.05, 1609), 'N': (1.976, 0.063, 2858), 'Se': (2.409, 0.05, 1065), 'O': (1.942, 0.064, 4135), 'P': (2.378, 0.063, 317), 'C': (2.003, 0.066, 2343), 'Si': (2.427, 0.05, 83), 'Br': (2.35, 0.08, 320), 'I': (2.533, 0.05, 252), 'S': (2.303, 0.05, 705), 'F': (1.901, 0.095, 127), 'As': (2.444, 0.05, 166)},
  'Gd': {'N': (2.558, 0.095, 3439), 'O': (2.404, 0.078, 21864), 'C': (2.749, 0.14, 4006), 'P': (3.14, 0.104, 131), 'S': (2.89, 0.21, 65), 'Cl': (2.735, 0.078, 405), 'Br': (2.911, 0.078, 104), 'I': (3.169, 0.075, 203), 'Si': (3.425, 0.081, 140)},
  'Hf': {'Cl': (2.422, 0.05, 167), 'P': (2.806, 0.104, 161), 'N': (2.132, 0.155, 841), 'O': (2.167, 0.076, 2291), 'C': (2.51, 0.058, 1760)},
  'Hg': {'Cl': (2.439, 0.151, 1640), 'C': (2.07, 0.05, 1531), 'N': (2.375, 0.091, 1340), 'O': (2.192, 0.162, 290), 'I': (2.722, 0.111, 1078), 'Br': (2.551, 0.108, 700), 'S': (2.479, 0.101, 1308), 'P': (2.485, 0.07, 267), 'Te': (2.78, 0.05, 461), 'Se': (2.637, 0.128, 353)},
  'Ho': {'N': (2.522, 0.097, 1480), 'O': (2.346, 0.072, 7224), 'F': (2.306, 0.05, 106), 'C': (2.743, 0.181, 891), 'Cl': (2.653, 0.068, 80), 'S': (2.893, 0.05, 105), 'P': (3.402, 0.05, 78), 'Si': (3.352, 0.158, 62)},
  'In': {'O': (2.193, 0.084, 5925), 'C': (2.194, 0.079, 1160), 'Cl': (2.492, 0.05, 1806), 'Br': (2.572, 0.091, 179), 'N': (2.272, 0.094, 1345), 'S': (2.489, 0.089, 1083), 'Se': (2.538, 0.05, 136), 'Te': (2.774, 0.05, 76), 'I': (2.75, 0.091, 74), 'F': (2.13, 0.058, 85)},
  'Ir': {'N': (2.083, 0.063, 6254), 'C': (2.144, 0.112, 18578), 'O': (2.119, 0.055, 1821), 'I': (2.7, 0.05, 172), 'P': (2.314, 0.05, 3283), 'S': (2.35, 0.053, 867), 'Cl': (2.406, 0.05, 1680), 'Si': (2.367, 0.055, 151), 'Br': (2.529, 0.076, 107), 'Se': (2.51, 0.063, 64)},
  'K': {'N': (2.931, 0.12, 5429), 'O': (2.81, 0.091, 32761), 'Si': (3.47, 0.05, 449), 'C': (3.139, 0.094, 5675), 'Cl': (3.162, 0.142, 477), 'S': (3.4, 0.062, 1312), 'I': (3.609, 0.196, 156), 'F': (2.801, 0.146, 1037), 'Br': (3.451, 0.202, 83), 'P': (3.409, 0.126, 259), 'Se': (3.389, 0.062, 102), 'Te': (3.617, 0.083, 64)},
  'La': {'N': (2.727, 0.139, 2449), 'O': (2.55, 0.091, 12812), 'C': (2.863, 0.145, 5598), 'S': (3.008, 0.056, 194), 'Si': (3.448, 0.125, 143), 'P': (3.242, 0.211, 128), 'Cl': (2.883, 0.078, 216), 'I': (3.257, 0.087, 147), 'As': (3.342, 0.222, 66)},
  'Li': {'N': (2.07, 0.076, 6244), 'Br': (2.558, 0.068, 259), 'O': (1.951, 0.065, 14217), 'C': (2.283, 0.124, 4348), 'P': (2.591, 0.078, 480), 'Cl': (2.369, 0.063, 927), 'S': (2.451, 0.05, 540), 'Si': (2.684, 0.098, 247), 'I': (2.802, 0.056, 82), 'F': (1.948, 0.139, 162), 'As': (2.709, 0.185, 65), 'Se': (2.548, 0.05, 148)},
  'Lu': {'Si': (3.298, 0.102, 53), 'O': (2.294, 0.086, 3826), 'N': (2.445, 0.143, 1886), 'C': (2.612, 0.1, 1830), 'Cl': (2.582, 0.054, 71), 'P': (3.367, 0.05, 112)},
  'Mg': {'N': (2.095, 0.074, 3311), 'C': (2.364, 0.15, 1742), 'O': (2.06, 0.05, 7275), 'Br': (2.592, 0.071, 138), 'Cl': (2.443, 0.073, 329), 'I': (2.791, 0.094, 73), 'P': (2.651, 0.061, 248), 'S': (2.569, 0.05, 64)},
  'Mn': {'Se': (2.499, 0.071, 371), 'P': (2.296, 0.068, 948), 'C': (1.925, 0.225, 9995), 'O': (2.132, 0.162, 52127), 'N': (2.218, 0.123, 19007), 'Cl': (2.505, 0.123, 2167), 'S': (2.409, 0.115, 1101), 'Br': (2.532, 0.076, 1071), 'I': (2.709, 0.05, 74), 'Te': (2.662, 0.076, 85), 'As': (2.334, 0.137, 51)},
  'Mo': {'O': (1.94, 0.258, 102577), 'C': (2.292, 0.115, 21968), 'Cl': (2.45, 0.05, 2700), 'N': (2.18, 0.127, 7533), 'S': (2.381, 0.1, 6872), 'Br': (2.575, 0.05, 533), 'Te': (3.266, 0.052, 229), 'P': (2.505, 0.068, 3356), 'Se': (2.53, 0.081, 328), 'I': (2.79, 0.05, 1176), 'F': (2.129, 0.16, 215), 'As': (2.595, 0.057, 581)},
  'Na': {'O': (2.407, 0.105, 30938), 'N': (2.48, 0.091, 4031), 'C': (2.756, 0.1, 1916), 'F': (2.38, 0.192, 356), 'P': (3.048, 0.05, 160), 'S': (2.948, 0.108, 1092), 'Si': (3.167, 0.05, 74), 'Br': (3.005, 0.101, 96), 'Cl': (2.892, 0.101, 150)},
  'Nb': {'O': (1.984, 0.134, 6088), 'F': (1.93, 0.057, 233), 'P': (2.633, 0.096, 195), 'C': (2.359, 0.135, 2440), 'N': (2.106, 0.21, 815), 'Cl': (2.449, 0.05, 1700), 'S': (2.525, 0.157, 378), 'As': (2.636, 0.087, 62), 'I': (2.933, 0.05, 58)},
  'Nd': {'O': (2.477, 0.085, 13525), 'C': (2.815, 0.121, 2639), 'N': (2.648, 0.136, 3119), 'I': (3.156, 0.062, 98), 'S': (2.932, 0.051, 340), 'Si': (3.485, 0.064, 95), 'Cl': (2.811, 0.082, 323), 'P': (3.296, 0.284, 134), 'Se': (3.08, 0.079, 133)},
  'Ni': {'N': (2.066, 0.078, 35608), 'O': (2.058, 0.052, 36293), 'S': (2.182, 0.05, 6529), 'P': (2.191, 0.05, 3994), 'Si': (2.216, 0.057, 131), 'I': (2.599, 0.109, 186), 'C': (1.994, 0.175, 9459), 'Cl': (2.353, 0.125, 2330), 'Br': (2.387, 0.09, 1028), 'F': (1.944, 0.054, 360), 'Se': (2.332, 0.127, 90), 'As': (2.337, 0.05, 367)},
  'Np': {'N': (2.936, 0.134, 183), 'O': (2.345, 0.14, 1658), 'Cl': (2.805, 0.091, 231), 'C': (2.857, 0.05, 102)},
  'Os': {'P': (2.367, 0.055, 1437), 'N': (2.086, 0.073, 1707), 'Cl': (2.401, 0.052, 669), 'O': (2.071, 0.091, 796), 'C': (1.921, 0.074, 10858), 'Br': (2.527, 0.069, 189), 'S': (2.418, 0.05, 593), 'Se': (2.509, 0.05, 121)},
  'Pa': {'O': (2.359, 0.05, 57)},
  'Pb': {'N': (2.513, 0.097, 1068), 'O': (2.446, 0.097, 2879), 'S': (2.735, 0.15, 311), 'Cl': (2.851, 0.06, 731), 'Br': (3.002, 0.056, 4082), 'C': (2.214, 0.068, 441), 'I': (3.198, 0.05, 6039), 'P': (2.715, 0.05, 101)},
  'Pd': {'N': (2.038, 0.05, 10742), 'O': (2.032, 0.056, 6447), 'Cl': (2.324, 0.05, 5935), 'C': (2.045, 0.097, 11286), 'I': (2.62, 0.05, 665), 'P': (2.292, 0.056, 5781), 'Br': (2.457, 0.05, 821), 'S': (2.317, 0.05, 2562), 'Se': (2.425, 0.05, 208), 'As': (2.404, 0.05, 121), 'F': (2.011, 0.05, 55), 'Si': (2.381, 0.1, 197), 'Te': (2.582, 0.05, 195)},
  'Pr': {'N': (2.67, 0.098, 1556), 'O': (2.499, 0.089, 9037), 'S': (2.955, 0.05, 158), 'C': (2.835, 0.115, 1348), 'Cl': (2.863, 0.091, 150)},
  'Pt': {'P': (2.295, 0.05, 6602), 'C': (2.03, 0.065, 8844), 'N': (2.041, 0.05, 7636), 'I': (2.643, 0.067, 756), 'Cl': (2.317, 0.05, 3167), 'O': (2.024, 0.05, 2167), 'S': (2.317, 0.053, 2772), 'Se': (2.451, 0.05, 190), 'Br': (2.473, 0.072, 662), 'Si': (2.316, 0.052, 167), 'As': (2.366, 0.05, 116)},
  'Pu': {'O': (2.412, 0.12, 313), 'Cl': (2.6, 0.05, 81), 'N': (2.882, 0.08, 98), 'C': (2.702, 0.089, 141)},
  'Rb': {'O': (2.944, 0.117, 2512), 'N': (3.041, 0.106, 533), 'C': (3.349, 0.067, 381), 'Br': (3.432, 0.06, 224), 'F': (2.946, 0.087, 74), 'Cl': (3.376, 0.085, 50), 'As': (3.674, 0.122, 68)},
  'Re': {'Br': (2.549, 0.076, 568), 'N': (2.17, 0.051, 3941), 'C': (1.957, 0.09, 11655), 'P': (2.438, 0.05, 2258), 'S': (2.407, 0.07, 2312), 'O': (1.921, 0.305, 3188), 'Se': (2.518, 0.05, 2028), 'Cl': (2.385, 0.061, 2012), 'Te': (2.639, 0.05, 367), 'I': (2.736, 0.057, 73)},
  'Rh': {'P': (2.294, 0.065, 4805), 'Cl': (2.4, 0.05, 2647), 'C': (2.141, 0.123, 18688), 'N': (2.091, 0.065, 5163), 'I': (2.684, 0.05, 330), 'Br': (2.576, 0.05, 212), 'O': (2.045, 0.05, 3900), 'S': (2.336, 0.05, 1441), 'Si': (2.367, 0.05, 99), 'Se': (2.464, 0.052, 56), 'As': (2.433, 0.055, 65)},
  'Ru': {'O': (2.065, 0.064, 5923), 'Cl': (2.409, 0.05, 5590), 'C': (2.178, 0.099, 41589), 'N': (2.077, 0.05, 15158), 'P': (2.336, 0.058, 7371), 'I': (2.727, 0.05, 171), 'Si': (2.381, 0.087, 105), 'As': (2.455, 0.05, 179), 'S': (2.367, 0.068, 2678), 'Se': (2.49, 0.05, 179), 'F': (2.167, 0.099, 59), 'Br': (2.543, 0.058, 282)},
  'Sc': {'Cl': (2.475, 0.095, 239), 'O': (2.09, 0.064, 1811), 'C': (2.377, 0.05, 7555), 'N': (2.167, 0.096, 1065), 'P': (2.737, 0.062, 183), 'Si': (3.027, 0.293, 89)},
  'Sm': {'N': (2.6, 0.134, 2762), 'O': (2.443, 0.091, 12691), 'C': (2.77, 0.123, 5403), 'Cl': (2.76, 0.096, 302), 'S': (2.887, 0.05, 292), 'Si': (3.438, 0.092, 264), 'P': (3.103, 0.21, 64), 'I': (3.14, 0.106, 102), 'Se': (2.942, 0.05, 286)},
  'Sn': {'N': (2.208, 0.167, 3345), 'O': (2.134, 0.101, 12310), 'C': (2.136, 0.05, 11158), 'S': (2.437, 0.057, 2928), 'Cl': (2.425, 0.06, 2826), 'I': (3.125, 0.056, 876), 'Br': (2.597, 0.055, 482), 'Si': (2.618, 0.056, 241), 'Se': (2.552, 0.062, 1447), 'F': (2.003, 0.053, 171), 'P': (2.624, 0.069, 381), 'As': (2.677, 0.05, 165)},
  'Sr': {'O': (2.603, 0.103, 5452), 'N': (2.613, 0.154, 950), 'C': (2.942, 0.128, 1000), 'P': (3.167, 0.133, 73), 'I': (3.26, 0.05, 167), 'Si': (3.396, 0.105, 87), 'Cl': (2.9, 0.111, 56), 'Br': (3.05, 0.103, 95), 'F': (2.916, 0.05, 143)},
  'Ta': {'O': (1.973, 0.098, 1759), 'F': (1.893, 0.05, 139), 'C': (2.433, 0.069, 2091), 'Cl': (2.413, 0.074, 989), 'P': (2.583, 0.128, 412), 'N': (2.075, 0.176, 549), 'Br': (2.603, 0.05, 139), 'S': (2.408, 0.152, 192)},
  'Tb': {'O': (2.384, 0.077, 23030), 'C': (2.748, 0.13, 3630), 'S': (2.946, 0.05, 257), 'N': (2.565, 0.092, 3041), 'Cl': (2.704, 0.067, 460)},
  'Tc': {'O': (1.726, 0.101, 374), 'S': (2.353, 0.113, 113), 'N': (2.083, 0.122, 349), 'Cl': (2.365, 0.051, 136), 'P': (2.431, 0.05, 99), 'C': (2.086, 0.207, 275)},
  'Th': {'Cl': (2.759, 0.052, 190), 'O': (2.469, 0.086, 5097), 'N': (2.507, 0.232, 995), 'C': (2.82, 0.101, 1981), 'Si': (3.438, 0.215, 93), 'S': (2.876, 0.05, 63)},
  'Ti': {'Cl': (2.332, 0.081, 1766), 'N': (2.088, 0.201, 4107), 'C': (2.381, 0.05, 15968), 'O': (1.964, 0.135, 28127), 'P': (2.709, 0.254, 700), 'S': (2.443, 0.077, 367), 'F': (1.973, 0.144, 410), 'Si': (3.074, 0.114, 80), 'Te': (2.63, 0.106, 71)},
  'Tl': {'O': (2.467, 0.096, 104), 'N': (2.494, 0.124, 118), 'Cl': (2.472, 0.103, 54), 'C': (2.17, 0.085, 79), 'Br': (2.552, 0.05, 95)},
  'Tm': {'N': (2.5, 0.069, 576), 'O': (2.322, 0.087, 2450), 'C': (2.802, 0.114, 333)},
  'U': {'O': (2.354, 0.163, 20882), 'P': (3.117, 0.162, 475), 'C': (2.781, 0.098, 7918), 'N': (2.52, 0.223, 5138), 'S': (2.86, 0.078, 292), 'Si': (3.422, 0.109, 760), 'I': (3.109, 0.086, 179), 'Cl': (2.671, 0.073, 805), 'F': (2.317, 0.059, 238), 'Br': (2.794, 0.05, 122), 'Se': (2.936, 0.138, 76)},
  'V': {'O': (1.93, 0.139, 34145), 'P': (2.499, 0.05, 193), 'N': (2.122, 0.072, 2890), 'S': (2.382, 0.095, 425), 'Cl': (2.345, 0.088, 551), 'C': (2.272, 0.05, 3101), 'F': (2.07, 0.197, 256), 'Te': (3.246, 0.084, 55)},
  'W': {'P': (2.497, 0.055, 1869), 'C': (2.146, 0.218, 13162), 'Se': (2.459, 0.082, 105), 'Te': (2.829, 0.208, 96), 'N': (2.198, 0.108, 2862), 'O': (1.913, 0.096, 122994), 'Cl': (2.43, 0.099, 1584), 'S': (2.372, 0.117, 3082), 'I': (2.812, 0.05, 172), 'F': (1.88, 0.05, 71), 'Br': (2.61, 0.05, 209)},
  'Y': {'N': (2.479, 0.161, 2993), 'O': (2.339, 0.071, 8827), 'C': (2.666, 0.082, 4885), 'Cl': (2.689, 0.102, 433), 'P': (3.397, 0.05, 143), 'Si': (3.357, 0.11, 229), 'S': (2.801, 0.128, 69), 'I': (3.097, 0.061, 160), 'Br': (2.884, 0.07, 208), 'F': (2.27, 0.089, 151)},
  'Yb': {'O': (2.304, 0.094, 6659), 'C': (2.658, 0.116, 3659), 'N': (2.427, 0.123, 2957), 'Cl': (2.632, 0.092, 185), 'S': (2.776, 0.116, 94), 'Si': (3.284, 0.15, 136), 'P': (3.034, 0.138, 51)},
  'Zn': {'N': (2.07, 0.076, 35733), 'O': (2.023, 0.09, 48635), 'S': (2.341, 0.05, 2399), 'Cl': (2.247, 0.05, 3565), 'P': (2.402, 0.084, 126), 'C': (2.042, 0.118, 2513), 'Br': (2.363, 0.05, 1013), 'I': (2.542, 0.05, 2035), 'Si': (2.408, 0.064, 50), 'Se': (2.441, 0.05, 294), 'F': (2.023, 0.077, 255), 'As': (2.486, 0.092, 52)},
  'Zr': {'Cl': (2.449, 0.05, 1168), 'O': (2.192, 0.082, 16295), 'N': (2.176, 0.174, 2590), 'C': (2.525, 0.05, 12425), 'F': (2.037, 0.101, 130), 'Si': (3.163, 0.163, 170), 'P': (2.859, 0.163, 499), 'S': (2.601, 0.096, 329)},
}


def full_label(element):
  """ What the label-aware descriptor sees: the element itself for the eleven
  species, Zn for any other. """
  return element if element in Z_OF else "Zn"


def specialist_label(element):
  """ What the row-2 specialist descriptor sees at an atom typed `element`:
  C for row 2, the element itself for Si P S Cl Br I, Zn for any other. """
  from cctbx.eltbx import tiny_pse
  if element in Z_OF:
    return "C" if Z_OF[element] <= 10 else element
  try:
    return "C" if tiny_pse.table(element).atomic_number() <= 10 else "Zn"
  except Exception:
    return "C"
BASE_RATE = {'C': 0.699, 'N': 0.077, 'O': 0.121, 'F': 0.020, 'Cl': 0.014, 'Br': 0.004, 'I': 0.003, 'S': 0.012, 'P': 0.008, 'B': 0.005, 'Si': 0.003}


def heavy_neighbours(xs, slack=0.45):
  """ label -> (distance, element) of the bonded non-H neighbours, nearest
  first, a bond being closer than the covalent radii sum plus `slack`. """
  from cctbx.eltbx import covalent_radii

  def radius(e):
    try:
      return covalent_radii.table(e).radius()
    except Exception:
      return 1.5

  sc = list(xs.scatterers())
  el = [s.scattering_type.strip().capitalize() for s in sc]
  pat = xs.pair_asu_table(distance_cutoff=3.2)
  table, maps = pat.table(), pat.asu_mappings().mappings()
  out = {}
  for i, s in enumerate(sc):
    if el[i] in ("H", "D", "Q"):
      continue
    ci, found = maps[i][0].mapped_site(), []
    for j, groups in table[i].items():
      if el[j] in ("H", "D", "Q"):
        continue
      for group in groups:
        for j_sym in group:
          cj = maps[j][j_sym].mapped_site()
          d = sum((a - b)**2 for a, b in zip(ci, cj))**0.5
          if 0.5 < d < radius(el[i]) + radius(el[j]) + slack:
            found.append((d, el[j]))
    out[str(s.label)] = sorted(found)
  return out


def neighbour_prior(element, neighbours, floor=0.02):
  """ p(these heavy neighbours | element), the chemistry the density read
  cannot see: a lone atom on Cl is an O, an atom with no neighbour is not a
  C, a donor on a hard metal is O before S. Posterior tables are turned into
  likelihoods by the COD base rate; every table is floored so no one of them
  can veto the density. """
  import math
  neighbours = [x if isinstance(x, tuple) else (None, x) for x in neighbours]
  n, syms = len(neighbours), [e for d, e in neighbours]
  base = BASE_RATE.get(element, 0.01)
  # every table enters as its square root: the full ratios won 22 mistypes
  # on 100 hard cases and lost 10 on a 1000-id holdout
  p = max(NEIGHBOUR_COUNT.get(element, (0.05,)*7)[min(n, 6)], floor)**0.5
  if n == 1 and syms[0] in TERMINAL_ON:
    p *= (max(TERMINAL_ON[syms[0]].get(element, 0.0), floor)/base)**0.5
  for e in set(syms):
    if e in METAL_DONOR:
      p *= (max(METAL_DONOR[e].get(element, 0.0), floor)/base)**0.5
  # ponytail: a metal-donor bond has a length per element pair (Cu-O 1.98,
  # Cu-Cl 2.30, Cu-I 2.66 on the CSD); a Gaussian on the CSD median, its
  # spread widened by the 0.08 A a raw solution is off, floored and damped
  # like the tables; a pair the CSD has under 50 of carries no opinion
  for d, e in neighbours:
    w = BOND_WINDOW.get(e, {}).get(element) if d is not None else None
    if w:
      p *= max(math.exp(-0.5*((d - w[0])/math.hypot(w[1], 0.08))**2), floor)**0.5
  return p
