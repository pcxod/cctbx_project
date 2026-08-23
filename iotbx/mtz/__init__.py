"""Routines to read and write mtz formatted files
"""
from __future__ import absolute_import, division, print_function
import cctbx.array_family.flex

import boost_adaptbx.boost.python as bp
from six.moves import range
from six.moves import zip

from iotbx.mtz import extract_from_symmetry_lib
from cctbx import xray
import cctbx.xray.observation_types
from cctbx import miller
import cctbx.crystal
from cctbx import sgtbx
from cctbx import uctbx
from cctbx.array_family import flex
from libtbx.str_utils import show_string, overwrite_at, contains_one_of
from libtbx.utils import Sorry
from libtbx import adopt_init_args, slots_getstate_setstate
import warnings
import string
import re
import sys, os

expected_cmtz_struct_sizes = (
  (56, 84, 176, 500, 12336, 12488), # Linux 32-bit
  (64, 88, 184, 504, 12336, 12528), # Tru64
  #
  # 2010/04/19, addition of spg_confidence to struct SYMGRP in mtzdata.h
  (56, 84, 176, 500, 12340, 12492), # 32-bit
  (64, 88, 184, 504, 12340, 12536), # 64-bit
  #
  # 2010/06/30, addition of colsource...grpposn to struct MTZCOL
  # and xml...n_unknown_headers to struct MTZ in mtzdata.h
  (136, 84, 176, 500, 12340, 12504), # 32-bit
  (144, 88, 184, 504, 12340, 12560), # 64-bit
)

column_type_legend_source = "ccp4/doc/mtzformat.doc"
column_type_legend = {
  "H": "index h,k,l",
  "J": "intensity",
  "F": "amplitude",
  "D": "anomalous difference",
  "Q": "standard deviation",
  "G": "F(+) or F(-)",
  "L": "standard deviation",
  "K": "I(+) or I(-)",
  "M": "standard deviation",
  "E": "normalized amplitude",
  "P": "phase angle in degrees",
  "W": "weight (of some sort)",
  "A": "phase probability coefficients (Hendrickson/Lattman)",
  "B": "BATCH number",
  "Y": "M/ISYM, packed partial/reject flag and symmetry number",
  "I": "integer",
  "R": "real",
}

column_type_as_miller_array_type_hints = {
  "H": "miller_index",
  "J": "intensity",
  "F": "amplitude",
  "D": "anomalous_difference",
  "Q": "standard_deviation",
  "G": "amplitude",
  "L": "standard_deviation",
  "K": "intensity",
  "M": "standard_deviation",
  "E": "normalized_amplitude",
  "P": "phase_degrees",
  "W": "weight",
  "A": "hendrickson_lattman",
  "B": "batch_number",
  "Y": "m_isym",
  "I": "integer",
  "R": "real",
}

assert sorted(column_type_as_miller_array_type_hints.keys()) \
    == sorted(column_type_legend.keys())

def default_column_types(miller_array):
  result = None
  if (miller_array.is_complex_array() and miller_array.sigmas() is None):
    assert not miller_array.is_xray_intensity_array()
    if (miller_array.anomalous_flag()):
      result = "GP"
    else:
      result = "FP"
  elif (    miller_array.is_xray_reconstructed_amplitude_array()
        and miller_array.anomalous_flag()
        and miller_array.sigmas() is not None):
    result = "FQDQY"
  elif ((   miller_array.is_bool_array()
         or miller_array.is_integer_array())
        and miller_array.sigmas() is None):
    result = "I"
  elif (miller_array.is_xray_intensity_array()):
    if (miller_array.anomalous_flag()):
      result = "K"
      if (miller_array.sigmas() is not None):
        result += "M"
    else:
      result = "J"
      if (miller_array.sigmas() is not None):
        result += "Q"
  elif (miller_array.is_xray_amplitude_array()
        or (miller_array.is_real_array()
            and miller_array.sigmas() is not None)):
    if (miller_array.anomalous_flag()):
      result = "G"
      if (miller_array.sigmas() is not None):
        result += "L"
    else:
      result = "F"
      if (miller_array.sigmas() is not None):
        result += "Q"
  elif (miller_array.is_real_array()):
    if (miller_array.anomalous_flag()):
      result = "G"
    else:
      result = "F"
  elif (miller_array.is_hendrickson_lattman_array()):
    result = "AAAA"
  return result

anomalous_label_patterns = {
  "+": ("+", "PLUS"),
  "-": ("-", "MINU")
}

def is_anomalous_label(sign, label):
  return contains_one_of(label.upper(), anomalous_label_patterns[sign])

def are_anomalous_labels(sign, labels):
  for label in labels:
    if (not is_anomalous_label(sign, label)): return False
  return True

class label_decorator(__builtins__["object"]):

  def __init__(self,
        anomalous_plus_suffix="(+)",
        anomalous_minus_suffix="(-)",
        sigmas_prefix="SIG",
        sigmas_suffix="",
        delta_anomalous_prefix="DANO",
        delta_anomalous_suffix="",
        delta_anomalous_isym_prefix="ISYM",
        delta_anomalous_isym_suffix="",
        phases_prefix="PHI",
        phases_suffix="",
        hendrickson_lattman_suffix_list=["A","B","C","D"]):
    assert len(hendrickson_lattman_suffix_list) == 4
    adopt_init_args(self, locals())
    self.maximum_characters=30

  def check_character_length(self,label):
    if len(label)>self.maximum_characters:
      raise Sorry(
        "The label %s would be more than the maximum of %d characters" %(
         label,self.maximum_characters))

  def anomalous(self, root_label, sign=None):
    assert sign in (None, "+", "-")
    if (sign is None):
      label = root_label
    elif (sign == "+"):
      label = root_label + self.anomalous_plus_suffix
    else:
      label = root_label + self.anomalous_minus_suffix
    self.check_character_length(label)
    return label

  def sigmas(self, root_label, anomalous_sign=None):
    return self.anomalous(
      self.sigmas_prefix + root_label + self.sigmas_suffix,
      anomalous_sign)

  def delta_anomalous(self, root_label):
    label = self.delta_anomalous_prefix + root_label \
         + self.delta_anomalous_suffix
    self.check_character_length(label)
    return label

  def delta_anomalous_sigmas(self, root_label):
    return self.sigmas(self.delta_anomalous(root_label))

  def delta_anomalous_isym(self, root_label):
    label = self.delta_anomalous_isym_prefix + root_label \
         + self.delta_anomalous_isym_suffix
    self.check_character_length(label)
    return label

  def phases(self, root_label, anomalous_sign=None):
    return self.anomalous(
      self.phases_prefix + root_label + self.phases_suffix,
      anomalous_sign)

  def hendrickson_lattman(self, root_label, i_coeff, anomalous_sign=None):
    assert 0 <= i_coeff < 4
    return self.anomalous(
      root_label + self.hendrickson_lattman_suffix_list[i_coeff],
      anomalous_sign)

# XXX this will generate output labels recognizable by the Coot auto-open
# command, but only for specific root label types.
class ccp4_label_decorator(label_decorator):
  def phases(self, root_label, anomalous_sign=None):
    assert (root_label in ["FWT", "DELFWT"])
    root_label = re.sub("F", "", root_label)
    return self.anomalous(
      "PH" + root_label + self.phases_suffix,
      anomalous_sign)

def format_min_max(func, values):
  if (len(values) == 0): return "None"
  value = func(values)
  result = "%.2f" % value
  if (len(result) > 12): result = "%12.4E" % value
  return result

show_column_data_format_keywords = [
  "human_readable",
  "machine_readable",
  "spreadsheet"]

def tidy_show_column_data_format_keyword(input):
  if (input is None): return show_column_data_format_keywords[0]
  input = input.lower()
  for k in show_column_data_format_keywords:
    if (input.startswith(k[0])): return k
  raise Sorry(
      "Column data format keyword not recognized: %s\n" % show_string(input)
    + "  Valid keywords are: %s" % ", ".join(show_column_data_format_keywords))

def column_group(
      crystal_symmetry_from_file,
      crystal_symmetry,
      base_array_info,
      primary_column_type,
      labels,
      group,
      observation_type,
      anomalous=None):
  assert group.data.size() == group.indices.size()
  sigmas = getattr(group, "sigmas", None)
  if (sigmas is not None): assert sigmas.size() == group.indices.size()
  if anomalous is not None:
    miller_set = miller.set(
      crystal_symmetry=crystal_symmetry,
      indices=group.indices,
      anomalous_flag=anomalous)
  else:
    if (group.anomalous_flag):
      miller_set = miller.set(
        crystal_symmetry=crystal_symmetry,
        indices=group.indices,
        anomalous_flag=True)
      if (miller_set.n_bijvoet_pairs() == 0):
        # account for non-sensical files generated via
        # ccp4i "import merged data" tab with default parameters
        miller_set = miller.set(
          crystal_symmetry=crystal_symmetry,
          indices=group.indices,
          anomalous_flag=False)
    else:
      miller_set = miller.set(
        crystal_symmetry=crystal_symmetry,
        indices=group.indices).auto_anomalous(min_fraction_bijvoet_pairs=2/3.)
  result = miller_set.array(
    data=group.data,
    sigmas=sigmas).set_info(
      base_array_info.customized_copy(
        labels=labels,
        crystal_symmetry_from_file=crystal_symmetry_from_file,
        type_hints_from_file=column_type_as_miller_array_type_hints.get(
          primary_column_type)))
  if (observation_type is not None):
    result.set_observation_type(observation_type)
  elif (not result.is_complex_array()):
    if   (primary_column_type in "FG"):
      result.set_observation_type_xray_amplitude()
    elif (primary_column_type in "JK"):
      result.set_observation_type_xray_intensity()
  return result

def mend_non_conforming_anomalous_column_types(all_types, all_labels):

  replacements = {
    "JQJQ": "KMKM",
    "FQFQ": "GLGL",
    "JJQQ": "KKMM",
    "FFQQ": "GGLL",
    "FPFP": "GPGP",
    "FFPP": "GGPP",
  }

  def are_anomalous_labels():
    if (group_types[1] == "Q"):
      permutation = ((0,1),(2,3))
    else:
      permutation = ((0,2),(1,3))
    for offsets,sign in zip(permutation, ("+", "-")):
      for offs in offsets:
        if (not is_anomalous_label(sign, all_labels[i_group_start + offs])):
          return False
    return True

  def find_group():
    for group_types in replacements.keys():
      i_group_start = all_types_x.find(group_types)
      if (i_group_start >= 0):
        return group_types, i_group_start
    return None, None

  all_types = "".join(all_types)
  if (0): # for debugging only
    all_types = all_types.replace("GLGL", "FQFQ")
    all_types = all_types.replace("KMKM", "JQJQ")
  all_types_x = all_types
  while 1:
    group_types, i_group_start = find_group()
    if (group_types is None):
      break
    if (are_anomalous_labels()):
      replacement = replacements[group_types]
      all_types = overwrite_at(all_types, i_group_start, replacement)
    else:
      replacement = "X"
    all_types_x = overwrite_at(all_types_x, i_group_start, replacement)
  return all_types

class column_values_and_selection_valid(slots_getstate_setstate):

  __slots__ = ["values", "selection_valid"]

  def __init__(O, values, selection_valid):
    O.values = values
    O.selection_valid = selection_valid

  def as_tuple(O):
    return (O.values, O.selection_valid)

def miller_array_as_mtz_dataset(self,
      column_root_label,
      column_types=None,
      label_decorator=None,
      title=None,
      crystal_name="crystal",
      project_name="project",
      dataset_name="dataset",
      wavelength=0.0):
  if (title is None):
    title = str(self.info())
  if (title is None):
    title = "cctbx.miller.array"
  unit_cell = self.unit_cell()
  if (unit_cell is None):
    unit_cell = uctbx.unit_cell((1,1,1,90,90,90))
  space_group_info = self.space_group_info()
  if (space_group_info is None):
    space_group_info = sgtbx.space_group_info(symbol="P 1")
  mtz_object = object() \
    .set_title(title=title) \
    .set_space_group_info(space_group_info=space_group_info)
  mtz_object.set_hkl_base(unit_cell=unit_cell)
  return mtz_object.add_crystal(
    name=crystal_name,
    project_name=project_name,
    unit_cell=unit_cell).add_dataset(
      name=dataset_name,
      wavelength=wavelength).add_miller_array(
        miller_array=self,
        column_root_label=column_root_label,
        column_types=column_types,
        label_decorator=label_decorator)

def cutoff_data(file_name, d_min_cut):
  """
  Utility function for applying a global resolution cutoff to an MTZ file
  without reference to the contents.  This is only used internally in cases
  where the input MTZ has already been processed by CCTBX.
  """
  mtz_in = object(file_name=file_name)
  cut_arrays = []
  labels = ["H","K","L"]
  uppercase = ['']
  for miller_array in mtz_in.as_miller_arrays():
    cut_arrays.append(miller_array.resolution_filter(d_min=d_min_cut))
    labels.extend(miller_array.info().labels)
  mtz_dataset = cut_arrays[0].as_mtz_dataset(column_root_label="A")
  for k, other_array in enumerate(cut_arrays[1:], start=1):
    mtz_dataset.add_miller_array(other_array,
      column_root_label=string.ascii_uppercase[k])
  mtz_obj = mtz_dataset.mtz_object()
  assert (len(list(mtz_obj.columns())) == len(labels))
  for k, column in enumerate(mtz_obj.columns()):
    column.set_label(labels[k])
  mtz_obj.write(file_name)
