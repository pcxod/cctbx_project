from __future__ import absolute_import, division, print_function

import contextlib, io

from iotbx import shelx
from iotbx.shelx.tst_lex_parse_build import template_ins
import iotbx.builders
from libtbx.test_utils import Exception_expected

def parse(text):
  """ (builder, what the restraint parser printed) """
  builder = iotbx.builders.restrained_crystal_structure_builder()
  stream = shelx.command_stream(file=io.StringIO(text))
  l_cs = shelx.crystal_symmetry_parser(stream, builder)
  l_afix = shelx.afix_parser(l_cs.filtered_commands(), builder)
  l_xs = shelx.atom_parser(l_afix.filtered_commands(), builder)
  l_r = shelx.restraint_parser(l_xs.filtered_commands(), builder)
  out = io.StringIO()
  with contextlib.redirect_stdout(out):
    l_r.parse()
  return builder, out.getvalue()

def exercise_unreadable_atom_reference():
  """ the lexer read 'C2345' as C234: an error now, not another atom """
  try:
    parse(template_ins("DFIX 1.5 C1 C2345"))
    raise Exception_expected
  except RuntimeError as e:
    assert "C2345" in str(e), str(e)
  parse(template_ins("DFIX 1.5 C1 C2"))

def exercise_residue_references_not_bound_to_another_atom():
  """ 'C2_A:26' was read as C2 of the current residue, 'N_-3' as N of the
  previous one, DANG_A:45 failed later with a bare KeyError. Each is either
  refused at its line or, read as a residue this model does not have, reported
  and dropped - never a restraint on a different atom. """
  for ins, needle in (("DFIX 1.5 C1 C2_A:26", "C2_A:26"),
                      ("DFIX 1.5 C1 C2_-3", "C2_-3"),
                      ("DANG_A:45 2.5 C1 C3", "A:45")):
    try:
      b, out = parse(template_ins(ins))
    except RuntimeError as e:
      assert needle in str(e), (ins, str(e))
    else:
      assert "not in the model" in out and not b.proxies().get("bond"), (ins, out)

def exercise_unresolved_restraint_reported():
  """ an explicit restraint naming a missing atom used to vanish silently """
  b, out = parse(template_ins("DFIX 1.5 C1 C9\nDFIX 1.5 C1 C2"))
  assert "1 restraint(s) name atoms that are not in the model" in out, out
  assert "C9" in out, out
  assert len(b.proxies()["bond"]) == 1

def exercise_part_on_scatterer():
  """ PART reaches the scatterer, and a restraint on a name two PARTs share is
  not silently bound to the last-read atom: reported, or applied per PART """
  b, out = parse("""\
CELL 0 5 10 15 90 90 90
SFAC C
DFIX 1.5 C1 C2
C1    1     0.28967  0.31613 -0.02773  11.00000  0.05
PART 1
C2    1     0.30768  0.41649  0.03308  10.50000  0.05
PART 2
C2    1     0.31768  0.42649  0.04308  10.50000  0.05
PART 0
""")
  parts = [sc.get_part() for sc in b.structure.scatterers()]
  assert parts == [0, 1, 2], parts
  pairs = sorted(tuple(p.i_seqs) for p in b.proxies()["bond"])
  assert pairs == [(0, 1), (0, 2)] or "more than one PART" in out, (pairs, out)

def run():
  import libtbx.load_env
  if (not libtbx.env.has_module(name="smtbx")):
    print("Skipping: smtbx module is not available.")
    return
  exercise_unreadable_atom_reference()
  exercise_residue_references_not_bound_to_another_atom()
  exercise_unresolved_restraint_reported()
  exercise_part_on_scatterer()
  print('OK')

if __name__ == '__main__':
  run()
