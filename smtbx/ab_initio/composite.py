""" The composite solver: every validated block, composed, each one switchable.

Florian, 8 August 2026: *"we are building a composite solver, not a single
method, but the combined power of all building blocks"*.

That is what the measurements have been saying all along. Not one idea tried
this week beat the pipeline on its own -- alpha scored 0.48 against the module's
0.64, refined free-R1 scores 0.58 against 0.76, held-out correlation 0.51 -- and
two of them are worth real points *in combination*. A solver built as one method
has one failure mode; a solver built as weighted blocks degrades gracefully when
any single block is uninformative.

## The blocks, and what each is measured to be worth

    P1 multi-trial            the starting point                  baseline
    candidate generation      truth present 0.9995 of the time
    absence evidence          the module's ranking     COD top1 0.8718
    alpha shortlist           orders below the top two  measured +0.107 top3
    solve_in                  re-solve in the chosen group        +37 points
    refined free-R1           fused with rank, chooses among      +0.04 top1
                              the shortlist
    peak typing               geometry + density                  0.7692
    cleanup and retype        after compaq                        +0.046

Refuted and therefore absent: dual-space recycling, peak-list omission, height
and connectivity prunes, whole-data fit as a chooser. They are listed in
`Story-And-Statistics-8-Aug` with the measurements, so they do not get rebuilt.

## Two rules the composition obeys

**Nothing deletes.** Every block contributes to an ordering; none removes a
candidate. A shortlist is a hard test and was measured as one -- alpha's top
three gained 0.18 on measured data and lost 0.20 on COD in exactly that form.

**Nothing ships unmeasured on both corpora.** Each block is a switch here so
that an A/B is one code version under two environments, and so a block that
turns out to invert between merged and unmerged data can be turned off without
unpicking the pipeline.
"""
from __future__ import absolute_import, division, print_function

import os

# Weight on the R1 ranking against the module's own, in the final choice among
# shortlisted groups. 0.4 was selected on training halves in four random splits
# and reproduced on every held-out half (+0.037 mean, never negative), so it is
# a fitted number with a held-out justification rather than a chosen one.
R1_FUSION_WEIGHT = 0.4

# How many shortlisted groups are solved and scored. Three is what the
# suggester reports to a user; the cost is linear in this.
N_SHORTLIST = 3


def _enabled(name, default=True):
  value = os.environ.get(name)
  if value is None:
    return default
  return value not in ("", "0", "false", "False")


def settings():
  """ Which blocks are active. Read once per call so an A/B is an environment.

  `alpha` is the shortlist ordering, read by `space_group_suggest` itself;
  it is reported here so a run records what it was doing.
  """
  return dict(
    alpha=_enabled("SMTBX_ALPHA_SHORTLIST", False),
    solve_each=_enabled("COMPOSITE_SOLVE_EACH", True),
    r1_fusion=_enabled("COMPOSITE_R1_FUSION", True),
    retype=_enabled("COMPOSITE_RETYPE", True))


def fusion_weight(f_obs, base=R1_FUSION_WEIGHT):
  """ How loudly refined R1 votes, given how much the ranking already knows.

  A single global weight was measured on both corpora and inverts between them:
  at 0.4 it gains +0.044 on measured unmerged data (0.7647 -> 0.8088) and loses
  0.0035 on COD (0.9194 -> 0.9159).

  That is not noise, it has a mechanism. COD is merged with its absences
  deleted, so the coverage test is highly informative and the ranking already
  reaches 0.9194 against an oracle of 0.9835 -- six points of room. Unmerged
  data is judged by the intensity test instead, the ranking reaches 0.765
  against an oracle of 0.92, and there are fifteen points of room. R1 should
  speak in proportion to how little the ranking has to say.

  Redundancy is the observable that separates the two regimes, and this module
  already conditions absence precedence on it at 1.5. **Graded, not gated**:
  the weight ramps between merged and clearly-unmerged rather than switching,
  so a file at redundancy 1.4 is not treated as categorically different from
  one at 1.6.
  """
  try:
    indices = f_obs.indices()
    n_unique = len(set(indices))
    redundancy = indices.size()/max(1, n_unique)
  except Exception:
    return 0.0
  # 1.0 -> 0 (merged: the ranking is already near its ceiling)
  # 2.0 -> base (unmerged: the ranking has room and R1 is complementary)
  ramp = (redundancy - 1.0)/1.0
  return base*max(0.0, min(1.0, ramp))


def rank_fusion(entries, weight=R1_FUSION_WEIGHT):
  """ Combine the module's order with an R1 order. Lower is better in both.

  Rank fusion rather than a score threshold, deliberately. The threshold form
  ("override the leader when R1 beats it by more than 0.20") scored higher on
  the corpus it was chosen on and **regressed on one of four held-out splits**,
  because it is a hard test: a candidate 0.19 better is ignored and one 0.21
  better takes over. The fused form never regressed on any split.

  `entries` is a list of dicts carrying `rank` (the module's position) and
  `r1` (refined free-R1, or None). Candidates with no R1 keep their module
  position, which is the right default: a missing measurement is not evidence.
  """
  scored = [e for e in entries if e.get("r1") is not None]
  if len(scored) < 2 or weight <= 0:
    return sorted(entries, key=lambda e: e["rank"])

  by_rank = dict((id(e), i) for i, e in
                 enumerate(sorted(entries, key=lambda e: e["rank"])))
  by_r1 = dict((id(e), i) for i, e in
               enumerate(sorted(scored, key=lambda e: e["r1"])))
  n = float(len(entries))
  unscored_position = len(scored)

  def key(e):
    r1_position = by_r1.get(id(e), unscored_position)
    return (weight*r1_position + (1.0 - weight)*by_rank[id(e)])/n

  return sorted(entries, key=key)


def choose_space_group(f_obs, suggestions, f_calc_in_p1, n_heavy, out=None):
  """ Solve in each shortlisted group, score it, and order them.

  Returns a list of dicts, best first, each with `space_group_info`, `placed`,
  `rank`, `r1`. The caller takes the first and can report the rest.

  **The expensive block, and the one that earns its cost.** Every ranking metric
  before this reads the P1 solution, which is identical for all candidates, so
  none of them can separate the shortlist -- the truth sits in it 0.76 of the
  time and the module puts it first in 0.76 of those. The three *solutions* are
  the only evidence that differs.
  """
  from six.moves import cStringIO as StringIO

  from smtbx.ab_initio import solve as ab_initio_solve
  from smtbx.ab_initio import dual_space

  out = out or StringIO()
  weight = fusion_weight(f_obs)
  entries = []
  for rank, s in enumerate(suggestions[:N_SHORTLIST]):
    info = s.space_group_info
    entry = dict(space_group_info=info, rank=rank, placed=None, r1=None)
    try:
      entry["placed"] = ab_initio_solve.solve_in(
        f_obs, info, f_calc_in_p1=f_calc_in_p1, out=out)
    except Exception:
      entry["placed"] = None
    if entry["placed"] is not None:
      entry["r1"] = refined_r1(f_obs, info, entry["placed"], n_heavy)
    entries.append(entry)
  return rank_fusion(entries, weight=weight)


def refined_r1(f_obs, space_group_info, placed, n_heavy):
  """ R1 on held-out reflections after refining a peak model. None on failure.

  Refinement is what distinguishes this from the six discriminators that were
  measured and refuted, all of which scored a *fixed* peak model. A model in the
  wrong group is not merely a worse fit but one that cannot improve, because the
  symmetry holds atoms where the density does not support them; that shows up
  only when the model is allowed to move.

  It is still not sufficient on its own -- 0.58 against the module's 0.76 -- so
  it enters through `rank_fusion` and never decides alone.
  """
  from smtbx.ab_initio import dual_space

  try:
    f_here = f_obs.customized_copy(
      space_group_info=space_group_info).merge_equivalents().array()
    work_all, free = dual_space.split_free(f_here)
    if free is None:
      return None
    matched = dual_space.match_arrays(work_all, placed)
    if matched is None:
      return None
    working, phases = matched
    current = working.phase_transfer(phases)
    model = dual_space.peaks_to_model(
      current, int(1.1*max(1, n_heavy)),
      crystal_symmetry=f_here.crystal_symmetry())
    if model.scatterers().size() < 2:
      return None
    refined = dual_space.refine_model(working, model)
    return dual_space.free_r1(free, refined)
  except Exception:
    return None
