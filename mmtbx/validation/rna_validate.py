
# TODO reduce to one outlier per residue

from __future__ import absolute_import, division, print_function
from mmtbx.monomer_library import rna_sugar_pucker_analysis
from mmtbx.monomer_library import pdb_interpretation
from mmtbx.validation import utils
from mmtbx import monomer_library
from mmtbx.validation import rna_geometry
from mmtbx.validation import residue
from mmtbx.validation import atoms
from mmtbx.validation import get_atoms_info
from iotbx.pdb import common_residue_names_get_class as get_res_class
from cctbx import geometry_restraints
from scitbx.array_family import flex
from libtbx.str_utils import make_sub_header, format_value
from libtbx import slots_getstate_setstate
from math import sqrt
from mmtbx.suitename import suitealyze
import sys
import json

rna_backbone_atoms = set([
  "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C1'",
  "C3'", "O3'", "C2'", "O2'", "N1", "N9" ]) #version 3.x naming

# individual validation results
class rna_bond(atoms):
  __slots__ = atoms.__slots__ + ["sigma", "delta"]

  @staticmethod
  def header():
    return "%-20s  %6s  %6s  %6s" % ("residue", "atom 1", "atom 2", "sigmas")

  def id_str(self, spacer=" "):
    return "%s%s%s" % (self.atoms_info[0].id_str(), spacer,
      self.atoms_info[1].id_str())

  def format_values(self):
    return "%-20s  %6s  %6s  %6.2f" % (self.id_str(), self.atoms_info[0].name,
      self.atoms_info[1].name, self.score)

  def as_string(self, prefix=""):
    return prefix + self.format_values()

  def as_JSON(self):
    atoms_dict = {}
    for s in self.atoms_info[0].__slots__:
      atoms_dict["atoms_"+s] = [getattr(self.atoms_info[0], s), getattr(self.atoms_info[1], s)]
    serializable_slots = [s for s in self.__slots__ if s != 'atoms_info' and hasattr(self, s)]
    slots_as_dict = ({s: getattr(self, s) for s in serializable_slots})
    return json.dumps(self.merge_two_dicts(slots_as_dict, atoms_dict), indent=2)

  def as_hierarchical_JSON(self):
    hierarchical_dict = {}
    hierarchy_nest_list = ['model_id', 'chain_id', 'resid', 'altloc']
    return json.dumps(self.nest_dict(hierarchy_nest_list, hierarchical_dict), indent=2)

  def as_table_row_phenix(self):
    return [ self.id_str(), self.atoms_info[0].name, self.atoms_info[1].name,
             self.score ]

class rna_angle(atoms):
  __slots__ = atoms.__slots__ + ["sigma", "delta"]

  @staticmethod
  def header():
    return "%-20s  %6s  %6s  %6s  %6s" % ("residue", "atom 1", "atom 2",
      "atom 3", "sigmas")

  def id_str(self, spacer=" "):
    return "%s%s%s%s%s" % (self.atoms_info[0].id_str(), spacer,
      self.atoms_info[1].id_str(), spacer,
      self.atoms_info[2].id_str())

  def format_values(self):
    return "%-20s  %6s  %6s  %6s  %6.2f" % (self.id_str(),
      self.atoms_info[0].name, self.atoms_info[1].name,
      self.atoms_info[2].name, self.score)

  def as_string(self, prefix=""):
    return prefix + self.format_values()

  def as_JSON(self):
    atoms_dict = {}
    for s in self.atoms_info[0].__slots__:
      atoms_dict["atoms_"+s] = [getattr(self.atoms_info[0], s), getattr(self.atoms_info[1], s), getattr(self.atoms_info[2], s)]
    serializable_slots = [s for s in self.__slots__ if s != 'atoms_info' and hasattr(self, s)]
    slots_as_dict = ({s: getattr(self, s) for s in serializable_slots})
    return json.dumps(self.merge_two_dicts(slots_as_dict, atoms_dict), indent=2)

  def as_hierarchical_JSON(self):
    hierarchical_dict = {}
    hierarchy_nest_list = ['model_id', 'chain_id', 'resid', 'altloc']
    return json.dumps(self.nest_dict(hierarchy_nest_list, hierarchical_dict), indent=2)

  def as_table_row_phenix(self):
    return [ self.id_str(), self.atoms_info[0].name, self.atoms_info[1].name,
             self.atoms_info[2].name, self.score ]

class rna_pucker(residue):
  """
  Validation using pucker-specific restraints library.
  """
  __slots__ = residue.__slots__ + [
    "delta_angle",
    "is_delta_outlier",
    "epsilon_angle",
    "is_epsilon_outlier",
    "Pperp_distance",
    "probable_pucker",
    "model_id",
  ]

  @staticmethod
  def header():
    return "%-20s  %8s  %8s  %8s  %8s" % ("residue", "delta", "outlier",
      "epsilon", "outlier")

  def format_values(self):
    def format_outlier_flag(flag):
      if (flag) : return "yes"
      else : return "no"
    def format_angle(val):
      return format_value("%8.1f", val, replace_none_with="---")
    return "%-20s  %8s  %8s  %8s  %8s" % (self.id_str(),
      format_angle(self.delta_angle),
      format_outlier_flag(self.is_delta_outlier),
      format_angle(self.epsilon_angle),
      format_outlier_flag(self.is_epsilon_outlier))

  def as_string(self, prefix=""):
    return prefix + self.format_values()

  def as_JSON(self):
    serializable_slots = [s for s in self.__slots__ if hasattr(self, s)]
    slots_as_dict = ({s: getattr(self, s) for s in serializable_slots})
    return json.dumps(slots_as_dict, indent=2)

  def as_hierarchical_JSON(self):
    hierarchical_dict = {}
    hierarchy_nest_list = ['model_id', 'chain_id', 'resid', 'altloc']
    return json.dumps(self.nest_dict(hierarchy_nest_list, hierarchical_dict), indent=2)

  def as_table_row_phenix(self):
    return [ self.id_str(), self.delta_angle, self.epsilon_angle ]

# analysis objects
class rna_bonds(rna_geometry):
  output_header = "#residue:atom_1:atom_2:num_sigmas"
  label = "Backbone bond lenths"
  gui_list_headers = ["Residue", "Atom 1", "Atom 2", "Sigmas"]
  gui_formats = ["%s", "%s", "%s", "%.2f"]
  wx_column_widths = [160] * 4
  def __init__(self, pdb_hierarchy, pdb_atoms, geometry_restraints_manager,
                outliers_only=True):
    self.n_outliers_large_by_model = {}
    self.n_outliers_small_by_model = {}
    rna_geometry.__init__(self)
    cutoff = 4
    sites_cart = pdb_atoms.extract_xyz()
    flags = geometry_restraints.flags.flags(default=True)
    pair_proxies = geometry_restraints_manager.pair_proxies(
      flags=flags,
      sites_cart=sites_cart)
    bond_proxies = pair_proxies.bond_proxies
    for proxy in bond_proxies.simple:
      restraint = geometry_restraints.bond(
        sites_cart=sites_cart,
        proxy=proxy)
      atom1 = pdb_atoms[proxy.i_seqs[0]].name
      atom2 = pdb_atoms[proxy.i_seqs[1]].name
      labels = pdb_atoms[proxy.i_seqs[0]].fetch_labels()
      model_id = labels.model_id
      if model_id not in self.n_total_by_model.keys():
        self.n_total_by_model[model_id] = 0
        self.n_outliers_by_model[model_id] = 0
        self.n_outliers_small_by_model[model_id] = 0
        self.n_outliers_large_by_model[model_id] = 0
      if (atom1.strip() not in rna_backbone_atoms or
          atom2.strip() not in rna_backbone_atoms):
        continue
      self.n_total += 1
      self.n_total_by_model[model_id] += 1
      sigma = sqrt(1 / restraint.weight)
      num_sigmas = - restraint.delta / sigma
      is_outlier = (abs(num_sigmas) >= cutoff)
      if is_outlier:
        self.n_outliers += 1
        self.n_outliers_by_model[model_id] += 1
        if num_sigmas < 0:
          self.n_outliers_small_by_model[model_id] += 1
        else:
          self.n_outliers_large_by_model[model_id] += 1
      if (is_outlier or not outliers_only):
        self.results.append(rna_bond(
          atoms_info=get_atoms_info(pdb_atoms, proxy.i_seqs),
          sigma=sigma,
          score=num_sigmas,
          delta=restraint.delta,
          xyz=flex.vec3_double([pdb_atoms[proxy.i_seqs[0]].xyz, pdb_atoms[proxy.i_seqs[1]].xyz]).mean(),
          outlier=is_outlier))

  def get_result_class(self) : return rna_bond

  def as_JSON(self, addon_json={}):
    if not addon_json:
      addon_json = {}
    addon_json["validation_type"] = "rna_bonds"
    data = addon_json
    flat_results = []
    hierarchical_results = {}
    summary_results = {}
    for result in self.results:
      flat_results.append(json.loads(result.as_JSON()))
      hier_result = json.loads(result.as_hierarchical_JSON())
      hierarchical_results = self.merge_dict(hierarchical_results, hier_result)

    data['flat_results'] = flat_results
    data['hierarchical_results'] = hierarchical_results
    for mod_id in self.n_total_by_model.keys():
      summary_results[mod_id] = {"num_outliers": self.n_outliers_by_model[mod_id],
                                 "num_total": self.n_total_by_model[mod_id],
                                 "num_outliers_too_small": self.n_outliers_small_by_model[mod_id],
                                 "num_outliers_too_large": self.n_outliers_large_by_model[mod_id]}
    data['summary_results'] = summary_results
    return json.dumps(data, indent=2)

  def show_summary(self, out=sys.stdout, prefix=""):
    if (self.n_total == 0):
      print(prefix + "No RNA backbone atoms found.", file=out)
    elif (self.n_outliers == 0):
      print(prefix + "All bonds within 4.0 sigma of ideal values.", file=out)
    else :
      print(prefix + "%d/%d bond outliers present" % (self.n_outliers,
        self.n_total), file=out)

class rna_angles(rna_geometry):
  output_header = "#residue:atom_1:atom_2:atom_3:num_sigmas"
  label = "Backbone bond angles"
  gui_list_headers = ["Residue", "Atom 1", "Atom 2", "Atom 3", "Sigmas"]
  gui_formats = ["%s", "%s", "%s", "%s", "%.2f"]
  wx_column_widths = [160] * 5
  def __init__(self, pdb_hierarchy, pdb_atoms, geometry_restraints_manager,
                outliers_only=True):
    self.n_outliers_large_by_model = {}
    self.n_outliers_small_by_model = {}
    rna_geometry.__init__(self)
    cutoff = 4
    sites_cart = pdb_atoms.extract_xyz()
    flags = geometry_restraints.flags.flags(default=True)
    i_seq_name_hash = utils.build_name_hash(pdb_hierarchy=pdb_hierarchy)
    for proxy in geometry_restraints_manager.angle_proxies:
      restraint = geometry_restraints.angle(
        sites_cart=sites_cart,
        proxy=proxy)
      atom1 = pdb_atoms[proxy.i_seqs[0]].name
      atom2 = pdb_atoms[proxy.i_seqs[1]].name
      atom3 = pdb_atoms[proxy.i_seqs[2]].name
      labels = pdb_atoms[proxy.i_seqs[0]].fetch_labels()
      model_id = labels.model_id
      if model_id not in self.n_total_by_model.keys():
        self.n_total_by_model[model_id] = 0
        self.n_outliers_by_model[model_id] = 0
        self.n_outliers_small_by_model[model_id] = 0
        self.n_outliers_large_by_model[model_id] = 0
      if (atom1.strip() not in rna_backbone_atoms or
          atom2.strip() not in rna_backbone_atoms or
          atom3.strip() not in rna_backbone_atoms):
        continue
      self.n_total += 1
      self.n_total_by_model[model_id] += 1
      sigma = sqrt(1 / restraint.weight)
      num_sigmas = - restraint.delta / sigma
      is_outlier = (abs(num_sigmas) >= cutoff)
      if is_outlier:
        self.n_outliers += 1
        self.n_outliers_by_model[model_id] += 1
        if num_sigmas < 0:
          self.n_outliers_small_by_model[model_id] += 1
        else:
          self.n_outliers_large_by_model[model_id] += 1
      if (is_outlier or not outliers_only):
        self.results.append(rna_angle(
          atoms_info=get_atoms_info(pdb_atoms, proxy.i_seqs),
          sigma=sigma,
          score=num_sigmas,
          delta=restraint.delta,
          xyz=pdb_atoms[proxy.i_seqs[1]].xyz,
          outlier=is_outlier))

  def get_result_class(self) : return rna_angle

  def as_JSON(self, addon_json={}):
    if not addon_json:
      addon_json = {}
    addon_json["validation_type"] = "rna_angles"
    data = addon_json
    flat_results = []
    hierarchical_results = {}
    summary_results = {}
    for result in self.results:
      flat_results.append(json.loads(result.as_JSON()))
      hier_result = json.loads(result.as_hierarchical_JSON())
      hierarchical_results = self.merge_dict(hierarchical_results, hier_result)

    data['flat_results'] = flat_results
    data['hierarchical_results'] = hierarchical_results
    for mod_id in self.n_total_by_model.keys():
      summary_results[mod_id] = {"num_outliers": self.n_outliers_by_model[mod_id],
                                 "num_total": self.n_total_by_model[mod_id],
                                 "num_outliers_too_small": self.n_outliers_small_by_model[mod_id],
                                 "num_outliers_too_large": self.n_outliers_large_by_model[mod_id]}
    data['summary_results'] = summary_results
    return json.dumps(data, indent=2)

  def show_summary(self, out=sys.stdout, prefix=""):
    if (self.n_total == 0):
      print(prefix + "No RNA backbone atoms found.", file=out)
    elif (self.n_outliers == 0):
      print(prefix + "All angles within 4.0 sigma of ideal values.", file=out)
    else :
      print(prefix + "%d/%d angle outliers present" % (self.n_outliers,
        self.n_total), file=out)

class rna_puckers(rna_geometry):
  __slots__ = rna_geometry.__slots__ + [
    "pucker_states",
    "pucker_perp_xyz",
    "pucker_dist",
  ]
  output_header = "#residue:delta_angle:is_delta_outlier:epsilon_angle:is_epsilon_outler"
  label = "Sugar pucker"
  gui_list_headers = ["Residue", "Delta", "Epsilon"]
  gui_formats = ["%s", "%.2f", "%.2f"]
  wx_column_widths = [200]*3
  def __init__(self, pdb_hierarchy, params=None, outliers_only=True):
    if (params is None):
      params = rna_sugar_pucker_analysis.master_phil.extract()
    self.pucker_states = []
    self.pucker_perp_xyz = {}
    self.pucker_dist = {}
    rna_geometry.__init__(self)
    from iotbx.pdb.rna_dna_detection import residue_analysis
    for model in pdb_hierarchy.models():
      self.n_outliers_by_model[model.id] = 0
      self.n_total_by_model[model.id] = 0
      for chain in model.chains():
        first_altloc = [conformer.altloc for conformer in chain.conformers()][0]
        #can skip some calculations on later altlocs
        for conformer in chain.conformers():
          residues = conformer.residues()
          for i_residue,residue in enumerate(residues):
            def _get_next_residue():
              j = i_residue + 1
              if (j == len(residues)): return None
              return residues[j]
            ra1 = residue_analysis(
              residue_atoms=residue.atoms(),
              distance_tolerance=params.bond_detection_distance_tolerance)
            if (ra1.problems is not None): continue
            if (not ra1.is_rna): continue
            residue_2_p_atom = None
            next_pdb_residue = _get_next_residue()
            if (next_pdb_residue is not None):
              residue_2_p_atom = next_pdb_residue.find_atom_by(name=" P  ")
            if (get_res_class(residue.resname) != "common_rna_dna"):
              continue
            if conformer.altloc.strip() == '':
              local_altloc = ''
            else:
              local_altloc = self.local_altloc_from_atoms(ra1.deoxy_ribo_atom_dict, ra1.c1p_outbound_atom, residue_2_p_atom)
            if local_altloc == '' and local_altloc != conformer.altloc and conformer.altloc != first_altloc:
              #if the pucker atoms contain no alternates, then the calculation can be skipped if this isn't the first
              #  conformer to be encountered
              continue
            ana = rna_sugar_pucker_analysis.evaluate(
              params=params,
              residue_1_deoxy_ribo_atom_dict=ra1.deoxy_ribo_atom_dict,
              residue_1_c1p_outbound_atom=ra1.c1p_outbound_atom,
              residue_2_p_atom=residue_2_p_atom)
            self.pucker_states.append(ana)
            self.n_total += 1
            self.n_total_by_model[model.id] += 1
            is_outlier = ana.is_delta_outlier or ana.is_epsilon_outlier
            if (is_outlier):
              self.n_outliers += 1
              self.n_outliers_by_model[model.id] += 1
            if (is_outlier or not outliers_only):
              pucker = rna_pucker(
                model_id=model.id,
                chain_id=chain.id,
                resseq=residue.resseq,
                icode=residue.icode,
                altloc=local_altloc, #'' if none
                resname=residue.resname,
                delta_angle=ana.delta,
                is_delta_outlier=ana.is_delta_outlier,
                epsilon_angle=ana.epsilon,
                is_epsilon_outlier=ana.is_epsilon_outlier,
                xyz=self.get_sugar_xyz_mean(residue),
                outlier=is_outlier)
              self.results.append(pucker)
              key = pucker.id_str() #[8:-1]
              self.pucker_perp_xyz[key] = [ana.p_perp_xyz, ana.o3p_perp_xyz]
              self.pucker_dist[key] = [ana.p_distance_c1p_outbound_line,
                                       ana.o3p_distance_c1p_outbound_line]
              pucker.Pperp_distance=ana.p_distance_c1p_outbound_line
              if not pucker.Pperp_distance:
                pucker.probable_pucker=None
              elif pucker.Pperp_distance and pucker.Pperp_distance < 2.9:
                pucker.probable_pucker="C2'-endo"
              else:
                pucker.probable_pucker="C3'-endo"

  def get_sugar_xyz_mean(self, residue):
    sugar_atoms = [" C1'", " C2'", " C3'", " C4'", " O4'"]
    atom_xyzs = []
    sums = [0]*3
    for at_name in sugar_atoms:
      sug_atom = residue.find_atom_by(name = at_name)
      if sug_atom is not None:
        atom_xyzs.append(sug_atom.xyz)
    assert len(atom_xyzs) > 0, "RNA sugar pucker validation found zero sugar atoms, probably due to non-standard atom names"
    mean = flex.vec3_double(atom_xyzs).mean()
    return mean

  def get_result_class(self) : return rna_pucker

  def as_JSON(self, addon_json={}):
    if not addon_json:
      addon_json = {}
    addon_json["validation_type"] = "rna_puckers"
    data = addon_json
    flat_results = []
    hierarchical_results = {}
    summary_results = {}
    for result in self.results:
      flat_results.append(json.loads(result.as_JSON()))
      hier_result = json.loads(result.as_hierarchical_JSON())
      hierarchical_results = self.merge_dict(hierarchical_results, hier_result)

    data['flat_results'] = flat_results
    data['hierarchical_results'] = hierarchical_results
    for mod_id in self.n_total_by_model.keys():
      summary_results[mod_id] = {"num_outliers": self.n_outliers_by_model[mod_id],
                                 "num_residues": self.n_total_by_model[mod_id]}
    data['summary_results'] = summary_results
    return json.dumps(data, indent=2)

  def show_summary(self, out=sys.stdout, prefix=""):
    if (self.n_total == 0):
      print(prefix + "No RNA sugar groups found.", file=out)
    elif (self.n_outliers == 0):
      print(prefix + "All puckers have reasonable geometry.", file=out)
    else :
      print(prefix + "%d/%d pucker outliers present" % (self.n_outliers,
        self.n_total), file=out)

  def local_altloc_from_atoms(self, residue_1_deoxy_ribo_atom_dict, residue_1_c1p_outbound_atom, residue_2_p_atom):
    #conformer.altloc masks whether a residue has true alternate conformations
    #check atom.parent().altloc for whether any atoms have alt positions
    #only run this if conformer.altloc != ''
    for atom in [residue_1_c1p_outbound_atom, residue_2_p_atom]:
      if atom is None: continue
      altloc = atom.parent().altloc
      if altloc != "":
        return altloc
    for atom in residue_1_deoxy_ribo_atom_dict.values():
      if atom is None: continue
      altloc = atom.parent().altloc
      if altloc != "":
        return altloc
    return ""

class rna_validation(slots_getstate_setstate):
  __slots__ = ["bonds", "angles", "puckers", "suites"]

  def __init__(self,
      pdb_hierarchy,
      geometry_restraints_manager=None,
      params=None,
      outliers_only=True):
    if (geometry_restraints_manager is None):
      mon_lib_srv = monomer_library.server.server()
      ener_lib = monomer_library.server.ener_lib()
      processed_pdb_file = pdb_interpretation.process(
        mon_lib_srv=mon_lib_srv,
        ener_lib=ener_lib,
        pdb_hierarchy=pdb_hierarchy,
        substitute_non_crystallographic_unit_cell_if_necessary=True)
      geometry_restraints_manager = \
        processed_pdb_file.geometry_restraints_manager()
      pdb_hierarchy = \
        processed_pdb_file.all_chain_proxies.pdb_hierarchy
    pdb_atoms = pdb_hierarchy.atoms()
    self.bonds = rna_bonds(
      pdb_hierarchy=pdb_hierarchy,
      pdb_atoms=pdb_atoms,
      geometry_restraints_manager=geometry_restraints_manager,
      outliers_only=outliers_only)
    self.angles = rna_angles(
      pdb_hierarchy=pdb_hierarchy,
      pdb_atoms=pdb_atoms,
      geometry_restraints_manager=geometry_restraints_manager,
      outliers_only=outliers_only)
    self.puckers = rna_puckers(
      pdb_hierarchy=pdb_hierarchy,
      params=getattr(params, "rna_sugar_pucker_analysis", None),
      outliers_only=outliers_only)
    self.suites = suitealyze.suitealyze(
      pdb_hierarchy=pdb_hierarchy,
      outliers_only=outliers_only)
    #self.suites = rna_suites(
    #  pdb_hierarchy=pdb_hierarchy,
    #  geometry_restraints_manager=geometry_restraints_manager,
    #  outliers_only=outliers_only)

  def show_summary(self, out=sys.stdout, prefix=""):
    pass

  def show(self, out=sys.stdout, prefix="", outliers_only=None,
      verbose=True):
    for geo_type in self.__slots__ :
      rv = getattr(self, geo_type)
      if (rv.n_outliers > 0) or (not outliers_only):
        make_sub_header(rv.label, out=out)
        rv.show(out=out)

  def as_JSON(self, addon_json={}):
    if not addon_json:
      addon_json = {}
    rna_json = addon_json
    for slot in self.__slots__:
      slot_json = json.loads(getattr(self, slot).as_JSON())
      rna_json["rna_"+slot] = slot_json
    return json.dumps(rna_json, indent=2)

class rna_pucker_ref(object):
  def __init__(self):
    #residue 21 is 3', residue 22 is 2', residue 3 included for P only
    self.sample_bases = """
CRYST1  132.298   35.250   42.225  90.00  90.95  90.00 C 1 2 1       4
ATOM    132  P     A A  21       8.804   4.707  -0.950  1.00 21.07           P
ATOM    133  OP1   A A  21       8.958   5.041  -2.388  1.00 19.56           O
ATOM    134  OP2   A A  21       9.726   3.742  -0.339  1.00 17.87           O
ATOM    135  O5'   A A  21       8.914   6.057  -0.104  1.00 18.44           O
ATOM    136  C5'   A A  21       8.311   7.240  -0.587  1.00 16.49           C
ATOM    137  C4'   A A  21       8.458   8.398   0.384  1.00 15.52           C
ATOM    138  O4'   A A  21       7.781   8.097   1.635  1.00 16.13           O
ATOM    139  C3'   A A  21       9.885   8.660   0.849  1.00 16.12           C
ATOM    140  O3'   A A  21      10.578   9.415  -0.119  1.00 15.84           O
ATOM    141  C2'   A A  21       9.670   9.462   2.123  1.00 14.81           C
ATOM    142  O2'   A A  21       9.390  10.824   1.853  1.00 16.92           O
ATOM    143  C1'   A A  21       8.420   8.794   2.696  1.00 16.03           C
ATOM    144  N9    A A  21       8.745   7.898   3.805  1.00 14.49           N
ATOM    145  C8    A A  21       9.200   6.611   3.807  1.00 14.35           C
ATOM    146  N7    A A  21       9.500   6.161   5.002  1.00 14.14           N
ATOM    147  C5    A A  21       9.192   7.227   5.846  1.00 13.55           C
ATOM    148  C6    A A  21       9.288   7.400   7.255  1.00 13.46           C
ATOM    149  N6    A A  21       9.760   6.467   8.083  1.00 13.17           N
ATOM    150  N1    A A  21       8.881   8.588   7.778  1.00 14.41           N
ATOM    151  C2    A A  21       8.419   9.532   6.935  1.00 14.69           C
ATOM    152  N3    A A  21       8.294   9.483   5.600  1.00 16.23           N
ATOM    153  C4    A A  21       8.702   8.288   5.122  1.00 13.95           C
ATOM    154  P     U A  22      11.833   8.778  -0.860  1.00 17.19           P
ATOM    155  OP1   U A  22      12.143   9.766  -1.930  1.00 14.62           O
ATOM    156  OP2   U A  22      11.572   7.360  -1.223  1.00 16.11           O
ATOM    157  O5'   U A  22      12.955   8.795   0.274  1.00 17.34           O
ATOM    158  C5'   U A  22      14.086   7.879   0.260  1.00 15.80           C
ATOM    159  C4'   U A  22      14.668   7.761   1.654  1.00 13.76           C
ATOM    160  O4'   U A  22      14.980   9.104   2.130  1.00 13.53           O
ATOM    161  C3'   U A  22      13.575   7.229   2.579  1.00 14.34           C
ATOM    162  O3'   U A  22      14.069   6.400   3.635  1.00 16.58           O
ATOM    163  C2'   U A  22      12.950   8.479   3.204  1.00 14.31           C
ATOM    164  O2'   U A  22      12.419   8.301   4.510  1.00 12.30           O
ATOM    165  C1'   U A  22      14.140   9.444   3.207  1.00 13.55           C
ATOM    166  N1    U A  22      13.832  10.876   3.147  1.00 14.29           N
ATOM    167  C2    U A  22      14.055  11.619   4.286  1.00 13.16           C
ATOM    168  O2    U A  22      14.534  11.153   5.327  1.00 15.12           O
ATOM    169  N3    U A  22      13.706  12.925   4.186  1.00 13.49           N
ATOM    170  C4    U A  22      13.172  13.569   3.099  1.00 13.63           C
ATOM    171  O4    U A  22      12.789  14.728   3.228  1.00 12.03           O
ATOM    172  C5    U A  22      12.990  12.747   1.948  1.00 12.63           C
ATOM    173  C6    U A  22      13.320  11.453   2.009  1.00 13.42           C
ATOM    174  P     A A  23      14.502   4.869   3.397  1.00 16.90           P
ATOM    175  OP1   A A  23      13.773   4.291   2.214  1.00 17.04           O
ATOM    176  OP2   A A  23      14.351   4.224   4.726  1.00 16.03           O
ATOM    177  O5'   A A  23      16.057   4.949   3.092  1.00 16.99           O
ATOM    178  C5'   A A  23      16.937   5.684   3.940  1.00 16.50           C
ATOM    179  C4'   A A  23      18.220   5.970   3.200  1.00 16.03           C
ATOM    180  O4'   A A  23      17.896   6.559   1.914  1.00 15.15           O
ATOM    181  C3'   A A  23      19.123   7.000   3.865  1.00 15.88           C
ATOM    182  O3'   A A  23      20.007   6.350   4.779  1.00 16.04           O
ATOM    183  C2'   A A  23      19.932   7.535   2.689  1.00 14.37           C
ATOM    184  O2'   A A  23      21.038   6.689   2.437  1.00 14.07           O
ATOM    185  C1'   A A  23      18.924   7.447   1.539  1.00 15.13           C
ATOM    186  N9    A A  23      18.313   8.717   1.157  1.00 14.71           N
ATOM    187  C8    A A  23      18.027   9.134  -0.113  1.00 14.98           C
ATOM    188  N7    A A  23      17.513  10.334  -0.171  1.00 15.52           N
ATOM    189  C5    A A  23      17.440  10.727   1.159  1.00 14.12           C
ATOM    190  C6    A A  23      16.984  11.892   1.762  1.00 13.27           C
ATOM    191  N6    A A  23      16.517  12.944   1.074  1.00 13.02           N
ATOM    192  N1    A A  23      17.029  11.962   3.109  1.00 13.36           N
ATOM    193  C2    A A  23      17.522  10.921   3.791  1.00 12.32           C
ATOM    194  N3    A A  23      17.996   9.768   3.328  1.00 12.80           N
ATOM    195  C4    A A  23      17.924   9.737   1.988  1.00 13.37           C
"""
    self.rna_backbone_atoms = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
      "C1'", "C3'", "O3'", "C2'", "O2'"] #version 3.x naming
    self.get_rna_pucker_ref()

  def get_rna_pucker_ref(self, test_pucker=False):
    flags = geometry_restraints.flags.flags(default=True)
    mon_lib_srv = monomer_library.server.server()
    ener_lib = monomer_library.server.ener_lib()
    self.bond_dict = {}
    self.angle_dict = {}
    self.processed_pdb_file = pdb_interpretation.process(
      mon_lib_srv=mon_lib_srv,
      ener_lib=ener_lib,
      file_name=None,
      raw_records=self.sample_bases)
    self.geometry = self.processed_pdb_file.geometry_restraints_manager()
    #confirm puckers are correct
    if test_pucker:
      r = rna_validate()
      r.pucker_evaluate(
        hierarchy=self.processed_pdb_file.all_chain_proxies.pdb_hierarchy)
      assert not r.pucker_states[0].is_2p
      assert r.pucker_states[1].is_2p
    i_seq_name_hash = utils.build_name_hash(
      pdb_hierarchy=self.processed_pdb_file.all_chain_proxies.pdb_hierarchy)
    pair_proxies = self.geometry.pair_proxies(
      flags=flags,
      sites_cart=self.processed_pdb_file.all_chain_proxies.sites_cart)
    bond_proxies = pair_proxies.bond_proxies
    for bond in bond_proxies.simple:
      atom1 = i_seq_name_hash[bond.i_seqs[0]][0:4]
      atom2 = i_seq_name_hash[bond.i_seqs[1]][0:4]
      if (atom1.strip() not in self.rna_backbone_atoms and
          atom2.strip() not in self.rna_backbone_atoms):
        continue
      key = atom1+atom2
      sigma = (1/bond.weight)**(.5)
      try:
        self.bond_dict[key].append((bond.distance_ideal, sigma))
      except Exception:
        self.bond_dict[key] = []
        self.bond_dict[key].append((bond.distance_ideal, sigma))
    for angle in self.geometry.angle_proxies:
      atom1 = i_seq_name_hash[angle.i_seqs[0]][0:4]
      atom2 = i_seq_name_hash[angle.i_seqs[1]][0:4]
      atom3 = i_seq_name_hash[angle.i_seqs[2]][0:4]
      if (atom1.strip() not in self.rna_backbone_atoms and
          atom2.strip() not in self.rna_backbone_atoms and
          atom3.strip() not in self.rna_backbone_atoms):
        continue
      key = atom1+atom2+atom3
      sigma = (1/angle.weight)**(.5)
      try:
        self.angle_dict[key].append((angle.angle_ideal, sigma))
      except Exception:
        self.angle_dict[key] = []
        self.angle_dict[key].append((angle.angle_ideal, sigma))
