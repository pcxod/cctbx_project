from __future__ import absolute_import, division, print_function

from cctbx import crystal, xray
from cctbx.array_family import flex
from libtbx.test_utils import Exception_expected
from smtbx import utils
from smtbx.refinement import constraints

def structure():
  xs = xray.structure(
    crystal_symmetry=crystal.symmetry((10, 10, 10, 90, 90, 90), "P1"),
    scatterers=flex.xray_scatterer([
      xray.scatterer("C1", (0.1, 0.1, 0.1), u=0.02),
      xray.scatterer("H1", (0.2, 0.1, 0.1), u=0.03)]))
  return xs, utils.connectivity_table(xs)

class broken(object):
  pivot = 0
  constrained_site_indices = (1,)
  constrained_parameters = ()

  def add_to(self, reparametrisation):
    raise RuntimeError("boom")

def exercise_failing_constraint_is_named():
  """ the exception of a failing add_to names the constraint, once """
  xs, ct = structure()
  try:
    constraints.reparametrisation(xs, [broken()], ct)
    raise Exception_expected
  except RuntimeError as e:
    assert str(e).startswith("boom"), str(e)
    # in str(e) itself, which is what Olex2 prints
    assert "while adding broken pivot=C1(#0)" in str(e), str(e)
    # what a traceback shows: the message, then the notes (Python >= 3.11)
    shown = "\n".join([str(e)] + list(getattr(e, '__notes__', ())))
    assert shown.count(
      "while adding broken pivot=C1(#0) constrained=H1(#1)") == 1, shown

def exercise_undefined_direction():
  """ raised a str before: a TypeError in Python 3 that lost the message """
  xs, ct = structure()
  r = constraints.reparametrisation(xs, [], ct)
  try:
    r.find_direction("nope")
    raise Exception_expected
  except constraints.InvalidConstraint as e:
    assert "nope" in str(e), str(e)

def run():
  exercise_failing_constraint_is_named()
  exercise_undefined_direction()
  print('OK')

if __name__ == '__main__':
  run()
