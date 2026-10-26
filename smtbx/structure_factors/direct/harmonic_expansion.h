#ifndef SMTBX_STRUCTURE_FACTORS_DIRECT_HARMONIC_EXPANSION_H
#define SMTBX_STRUCTURE_FACTORS_DIRECT_HARMONIC_EXPANSION_H

/// Form factors from atomic densities expanded in Gaussians and real
/// spherical harmonics: the .tshb file.
/** A .tsc/.tscb table stores the form factor of every atom at every
    reflection and every symmetry equivalent of it. A .tshb stores instead the
    density each form factor is the transform of, in the form

    \f[ \rho_j(\mathbf r) = \sum_{\mathrm{shells}\ s} r^{l_s}
          e^{-\alpha_s r^2} \sum_{m=-l_s}^{l_s} c_{s,m}\, Y_{l_s m}(\hat{\mathbf r}) \f]

    about the atom's position, in the Cartesian frame of the unit cell (cctbx's
    orthogonalisation: a along x, b in the xy plane), in Angstrom and
    electrons per cubic Angstrom. \f$ Y_{lm} \f$ are the orthonormal real
    spherical harmonics (see real_spherical_harmonics). Its transform is
    analytic,

    \f[ f_j(\mathbf s) = \sum_s i^{l_s} \left(\frac{\pi}{\alpha_s}\right)^{3/2}
          \left(\frac{\pi s}{\alpha_s}\right)^{l_s} e^{-\pi^2 s^2/\alpha_s}
          \sum_m c_{s,m}\, Y_{l_s m}(\hat{\mathbf s}) \f]

    with \f$ f(\mathbf s) = \int \rho(\mathbf r) e^{2\pi i \mathbf s\cdot\mathbf r}
    d^3r \f$, the convention of cctbx's structure factors, and
    \f$ \mathbf s \f$ the Cartesian reciprocal vector of h. So the file holds
    a few numbers per atom, which do not depend on the reflections, and any
    reflection can be evaluated - including ones a table would not have held.
    Like the tables, the values exclude f' and f'', which are added
    downstream, and the atomic displacement, which is applied downstream too.

    The symmetry equivalent of an atom under a rotation R has form factor
    \f$ f(\mathbf h R) \f$, which is what get_full returns per operator, so an
    improper rotation is handled by the parity of the harmonics without
    special cases.

    File layout (little endian):

      char[4]  "TSHB"
      uint32   version (1 to 4; the later ones add the fields below)
      uint32   header length in bytes, then the header: lines "Key: value",
               as in a .tsc header. "AD: false" is honoured as it is there;
               "FRAME", if present, must be "cartesian"; the rest is
               informational.
      uint32   number of atoms
      per atom:
        uint32 label length, then the label (matched to the structure's
               labels, case-insensitively, like a .tsc SCATTERERS line)
        uint32 number of shells
        per shell:
          int32   l
          float64 alpha (1/A^2)
          float64 c[2l+1], for m = -l .. l

      version 2 adds, per atom after its shells, three spherical record lists
      for what a Gaussian expansion does not hold (NoSpherA2 writes ECP cores
      into them):
        uint32 n, then n x { float64 c, float64 a }    l = 0 Gaussians
                                                        c exp(-a r^2)
        uint32 n, then n x { float64 c, float64 n, float64 zeta }
                                                        Slater terms
                                                        c r^n exp(-zeta r),
                                                        n real, -2 < n <= 60
        uint32 n, then n x { float64 a, float64 b }    form factor terms
                                                        a exp(-b s), s in 1/A

      version 3 adds, per atom after those lists,
        int32  Z, the nuclear charge
      and the header key "ED: true" makes the form factors electron ones, by
      the Mott-Bethe formula NoSpherA2's convert_to_ED applies to its tables:
      \f$ 0.023934\,(Z - f_x)/(\sin\theta/\lambda)^2 \f$, with
      \f$ \sin\theta/\lambda = s/2 \f$, per symmetry equivalent.

      version 4 keys each record the way a .tscb keys its columns, so that
      tables and densities merge and match alike. Directly after the label:
        uint8  key tag: 0 = the label is the key, 1 = a scatterer id follows
        (tag 1) 16 bytes, the scatterer id as in SCATTERER_IDS
               (cctbx::xray::scatterer_id_big: element, PART, coordinate)
        int32  PART
      An id is matched through scatterer_ID_lookup, by identity and, after the
      structure has moved, by position, as for a .tscb; a label by name, as in
      versions 1-3. All records of a file are keyed the same way, label or
      id. A key may occur once in a file, and two records may not
      resolve to the same atom: either is an error naming them. Records that
      match no atom are kept in unmatched for the caller to report.

      An atom that move, compaq, sgen -m or inv has put on a symmetry image
      (R, t) of the site its id records is found there too - over the space
      group's operations and, in a group without a centre, each composed with
      the inversion, as inv -f does. As Olex2 matches a table's columns, an
      atom within 0.1 A of the site is in place, and otherwise the nearest of
      the site and its images wins. The shells are rotated with the atom:
      the density becomes rho(Q^-1 r) with Q = O R O^-1, so that its form
      factor at h is the record's at h R, which is what a table recomputed
      for the moved model would hold. See rotate_shells; such records are
      listed in moved. (The table reader has no such fallback; for a table
      Olex2 rewrites the moved columns before refining,
      tsc_scatterer_resync.reorient_moved_columns.)

    Reading a damaged file is an error, never a crash: every count is checked
    against the bytes left before anything is allocated for it, and every
    number must be finite (alpha, zeta > 0; b >= 0; -2 < n <= 60).

    The Slater terms transform analytically too, for any real n > -2 (Gamma
    replacing the factorial when n is not an integer):
    \f$ 4\pi c\,\Gamma(n+2)\,\sin(m\theta)/(k\rho^m) \f$ with m = n+2,
    \f$ k = 2\pi s \f$, \f$ \rho = \sqrt{\zeta^2 + k^2} \f$ and
    \f$ \theta = \arctan(k/\zeta) \f$, which is
    \f$ 4\pi c\,(n+2)!/\zeta^{n+3} \f$ at s = 0.

    A contracted function is written as one shell per primitive with the
    contraction coefficients folded into c.
 */

#include <smtbx/structure_factors/direct/standard_xray.h>
#include <smtbx/error.h>
#include <cctbx/uctbx.h>
#include <cctbx/sgtbx/space_group.h>
#include <cctbx/xray/scatterer_lookup.h>
#include <cctbx/sgtbx/rt_mx.h>
#include <cctbx/sgtbx/site_symmetry.h>
#include <scitbx/constants.h>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace smtbx { namespace structure_factors { namespace table_based {

  /// All orthonormal real spherical harmonics up to lmax at a unit vector.
  /** Written to y[l*l + l + m], m = -l..l. The convention is the one obtained
      from the complex harmonics with the Condon-Shortley phase:
      \f$ Y_{l0} \f$; \f$ \sqrt2 (-1)^m \mathrm{Re} Y_l^m \f$ for m > 0; and
      \f$ \sqrt2 (-1)^m \mathrm{Im} Y_l^{|m|} \f$ for m < 0. That makes
      \f$ Y_{11} \propto x \f$, \f$ Y_{1,-1} \propto y \f$ and
      \f$ Y_{10} \propto z \f$ with positive constants.
   */
  template <typename FloatType>
  void real_spherical_harmonics(int lmax, FloatType x, FloatType y,
                                FloatType z, FloatType *out)
  {
    typedef FloatType f_t;
    f_t const pi = scitbx::constants::pi;
    f_t const ct = z;                        // cos theta
    f_t const rho = std::sqrt(x*x + y*y);    // sin theta
    // cos(m phi), sin(m phi) by recurrence from cos phi, sin phi
    f_t cp = 1, sp = 0;
    if (rho > 0) { cp = x/rho; sp = y/rho; }
    // on the stack: called per reflection and symmetry operator
    f_t cm[21], sm[21], p[21*21];
    SMTBX_ASSERT(lmax <= 20);
    cm[0] = 1; sm[0] = 0;
    for (int m = 1; m <= lmax; m++) {
      cm[m] = cm[m-1]*cp - sm[m-1]*sp;
      sm[m] = sm[m-1]*cp + cm[m-1]*sp;
    }
    // P_l^m(cos theta) with the Condon-Shortley phase, l >= m
    #define SMTBX_P(l, m) p[(l)*(lmax + 1) + (m)]
    SMTBX_P(0, 0) = 1;
    for (int m = 1; m <= lmax; m++) {
      SMTBX_P(m, m) = -(2*m - 1)*rho*SMTBX_P(m - 1, m - 1);
    }
    for (int m = 0; m < lmax; m++) {
      SMTBX_P(m + 1, m) = (2*m + 1)*ct*SMTBX_P(m, m);
    }
    for (int m = 0; m <= lmax; m++) {
      for (int l = m + 2; l <= lmax; l++) {
        SMTBX_P(l, m) = ((2*l - 1)*ct*SMTBX_P(l - 1, m)
                         - (l + m - 1)*SMTBX_P(l - 2, m))/(l - m);
      }
    }
    for (int l = 0; l <= lmax; l++) {
      // N_l^m = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!), built up over m
      f_t n = std::sqrt((2*l + 1)/(4*pi));
      out[l*l + l] = n*SMTBX_P(l, 0);
      for (int m = 1; m <= l; m++) {
        n /= std::sqrt(f_t((l + m)*(l - m + 1)));
        f_t const sign = (m % 2) ? -1 : 1;
        f_t const a = std::sqrt(f_t(2))*sign*n*SMTBX_P(l, m);
        out[l*l + l + m] = a*cm[m];
        out[l*l + l - m] = a*sm[m];
      }
    }
    #undef SMTBX_P
  }

  /// The densities of a .tshb: per atom, its shells.
  template <typename FloatType>
  struct harmonic_expansion_data {
    struct shell {
      int l;
      FloatType alpha;
      std::vector<FloatType> c;   // 2l+1, m = -l..l
    };
    struct slater { FloatType c; FloatType n; FloatType zeta; };
    struct exponential { FloatType a, b; };
    /* The Slater terms of one exponent, as the coefficient of each power:
       sum over terms of 4 pi c (n+1)! Im (zeta - i k)^-(n+2) / k is
       (4 pi/k) sum_m w_m Im (zeta - i k)^-m, so one complex power series per
       distinct exponent serves every term that has it. A Thakkar core is a
       few hundred terms over a few dozen exponents.
     */
    struct slater_group { FloatType zeta; std::vector<FloatType> w; };
    std::vector<std::vector<shell> > atoms;   // indexed like the structure
    std::vector<std::vector<slater> > slaters;
    // the terms of non-integer n, evaluated one by one
    std::vector<std::vector<slater> > fractional_slaters;
    std::vector<std::vector<slater_group> > slater_groups;
    std::vector<std::vector<exponential> > exponentials;
    af::shared<bool> covered;
    // the structure's sites, indexed like it: for the stabilisers of special
    // positions (see harmonic_expansion's dedup)
    std::vector<cctbx::fractional<FloatType> > sites;
    std::vector<int> z;      // nuclear charges, version 3
    // records that matched no atom of the structure, by label and, for an id
    // key, the id in hex
    std::vector<std::string> unmatched;
    // records found on a symmetry image of their site and rotated there
    std::vector<std::string> moved;
    std::string coverage;    // the header's COVERAGE value, if any
    bool use_ad, electron;
    int lmax;

    harmonic_expansion_data() : use_ad(true), electron(false), lmax(0) {}

    harmonic_expansion_data(
      uctbx::unit_cell const &unit_cell,
      af::shared<xray::scatterer<FloatType> > const &scatterers,
      std::string const &file_name,
      sgtbx::space_group const &space_group = sgtbx::space_group())
      : atoms(scatterers.size()),
        slaters(scatterers.size()),
        fractional_slaters(scatterers.size()),
        slater_groups(scatterers.size()),
        exponentials(scatterers.size()),
        covered(scatterers.size(), false),
        z(scatterers.size(), 0),
        use_ad(true),
        electron(false),
        lmax(0)
    {
      for (std::size_t i = 0; i < scatterers.size(); i++) {
        sites.push_back(scatterers[i].site);
      }
      std::ifstream in(file_name.c_str(), std::ios::in | std::ios::binary);
      SMTBX_ASSERT(in.good())(file_name);
      /* Every count read from the file is checked against what is left of
         it before anything is allocated for it: a corrupt or truncated file
         is an error, never a huge allocation or a read past the end.
       */
      in.seekg(0, std::ios::end);
      file_size = static_cast<std::uint64_t>(in.tellg());
      in.seekg(0, std::ios::beg);
      char magic[4];
      read(in, magic, 4);
      SMTBX_ASSERT(std::string(magic, 4) == "TSHB")(file_name);
      std::uint32_t version = read_u32(in);
      SMTBX_ASSERT(version >= 1 && version <= 4)(version);
      std::string header = read_string(in);
      std::vector<std::string> lines;
      boost::split(lines, header, boost::is_any_of("\n"));
      for (std::size_t i = 0; i < lines.size(); i++) {
        std::size_t ci = lines[i].find(':');
        if (ci == std::string::npos) continue;
        std::string key = boost::trim_copy(lines[i].substr(0, ci));
        std::string value = boost::trim_copy(lines[i].substr(ci + 1));
        if (boost::iequals(key, "AD")) {
          use_ad = !boost::iequals(value, "false");
        }
        else if (boost::iequals(key, "FRAME")) {
          SMTBX_ASSERT(boost::iequals(value, "cartesian"))(value);
        }
        else if (boost::iequals(key, "ED")) {
          electron = boost::iequals(value, "true");
        }
        else if (boost::iequals(key, "COVERAGE")) {
          coverage = value;
        }
      }
      SMTBX_ASSERT(!electron || version >= 3)
        ("ED needs the nuclear charges of a version 3 .tshb");
      std::map<std::string, std::size_t> by_label;
      for (std::size_t i = 0; i < scatterers.size(); i++) {
        by_label[boost::to_upper_copy(scatterers[i].label)] = i;
      }
      typedef cctbx::xray::scatterer_id_big<FloatType, cctbx::fractional<FloatType> >
        id_t;
      // built on the first id, as for a .tscb: the parts are the scatterers'
      boost::shared_ptr<cctbx::xray::scatterer_ID_lookup<FloatType> > lookup;
      std::map<std::string, std::string> keys;         // key -> its record
      std::vector<std::string> matched_by(scatterers.size());
      unsigned char first_tag = 0;
      std::uint32_t n_atoms = read_u32(in);
      for (std::uint32_t a = 0; a < n_atoms; a++) {
        std::string const record_label = read_string(in);
        std::string label = boost::to_upper_copy(record_label);
        std::string key = "label " + label, record = record_label;
        boost::shared_ptr<id_t> id;
        if (version >= 4) {
          unsigned char tag;
          read(in, reinterpret_cast<char *>(&tag), 1);
          SMTBX_ASSERT(tag <= 1)(record_label)(int(tag));
          // all records keyed one way, as NoSpherA2's -merge refuses a
          // table mixing labels and ids
          if (a == 0) first_tag = tag;
          else if (tag != first_tag) {
            throw SMTBX_ERROR("the .tshb mixes label and scatterer id keys: "
              "record " + record_label + " is keyed by "
              + (tag ? "id" : "label") + ", the first by "
              + (first_tag ? "id" : "label"));
          }
          if (tag == 1) {
            id.reset(new id_t(in));
            key = "id " + id->to_hex_string();
            record = record_label + " (id " + id->to_hex_string() + ")";
          }
          std::int32_t part;
          read(in, reinterpret_cast<char *>(&part), 4);
          if (part != 0) {
            record += " PART " + std::to_string(part);
          }
        }
        std::map<std::string, std::string>::const_iterator dup = keys.find(key);
        if (dup != keys.end()) {
          throw SMTBX_ERROR("the .tshb holds the " + key + " twice: records "
            + dup->second + " and " + record);
        }
        keys[key] = record;
        std::uint32_t n_shells = read_u32(in);
        fits(in, std::uint64_t(n_shells)*12, record_label);
        std::vector<shell> shells(n_shells);
        for (std::uint32_t s = 0; s < n_shells; s++) {
          std::int32_t l;
          read(in, reinterpret_cast<char *>(&l), 4);
          SMTBX_ASSERT(l >= 0 && l <= 20)(label)(l);
          shells[s].l = l;
          double alpha;
          read(in, reinterpret_cast<char *>(&alpha), 8);
          SMTBX_ASSERT(alpha > 0 && finite(alpha))(label)(alpha);
          shells[s].alpha = alpha;
          std::vector<double> c(2*l + 1);
          read(in, reinterpret_cast<char *>(&c[0]), 8*c.size());
          for (std::size_t m = 0; m < c.size(); m++) {
            SMTBX_ASSERT(finite(c[m]))(label)(l);
          }
          shells[s].c.assign(c.begin(), c.end());
          if (l > lmax) lmax = l;
        }
        std::vector<slater> sl;
        std::vector<exponential> ex;
        if (version >= 2) {
          FloatType const y00 = 1/std::sqrt(4*scitbx::constants::pi);
          std::uint32_t n = read_u32(in);
          fits(in, std::uint64_t(n)*16, record_label);
          for (std::uint32_t k = 0; k < n; k++) {
            double ca[2];
            read(in, reinterpret_cast<char *>(ca), 16);
            SMTBX_ASSERT(finite(ca[0]) && ca[1] > 0 && finite(ca[1]))(label);
            // an l = 0 shell: c exp(-a r^2) = (c/Y00) Y00 exp(-a r^2)
            shell g;
            g.l = 0; g.alpha = ca[1]; g.c.assign(1, ca[0]/y00);
            shells.push_back(g);
          }
          n = read_u32(in);
          fits(in, std::uint64_t(n)*24, record_label);
          for (std::uint32_t k = 0; k < n; k++) {
            double t[3];
            read(in, reinterpret_cast<char *>(t), 24);
            slater x = { t[0], t[1], t[2] };
            // r^n e^(-zeta r) r^2 must be integrable at 0, and the power
            // series of an integer n sized by it
            SMTBX_ASSERT(finite(x.c) && x.n > -2 && x.n <= 60 && x.zeta > 0
              && finite(x.zeta))(label)(x.n)(x.zeta);
            sl.push_back(x);
          }
          n = read_u32(in);
          fits(in, std::uint64_t(n)*16, record_label);
          for (std::uint32_t k = 0; k < n; k++) {
            double t[2];
            read(in, reinterpret_cast<char *>(t), 16);
            exponential x = { t[0], t[1] };
            SMTBX_ASSERT(finite(x.a) && x.b >= 0 && finite(x.b))(label);
            ex.push_back(x);
          }
        }
        std::int32_t atom_z = 0;
        if (version >= 3) read(in, reinterpret_cast<char *>(&atom_z), 4);
        std::size_t j = ~std::size_t(0);
        if (id) {
          if (!lookup) {
            af::shared<int> parts(scatterers.size());
            for (std::size_t q = 0; q < scatterers.size(); q++) {
              parts[q] = scatterers[q].get_part();
            }
            lookup.reset(new cctbx::xray::scatterer_ID_lookup<FloatType>(
              unit_cell, scatterers, parts));
          }
          j = lookup->index_from_id(*id);
          if (j == ~std::size_t(0)) {
            j = locate(*lookup, *id, unit_cell, space_group, scatterers,
                       shells, record);
          }
        }
        else {
          std::map<std::string, std::size_t>::const_iterator i =
            by_label.find(label);
          if (i != by_label.end()) j = i->second;
        }
        if (j == ~std::size_t(0)) {         // an atom no longer in the model
          unmatched.push_back(record);
          continue;
        }
        if (!matched_by[j].empty()) {
          throw SMTBX_ERROR("two .tshb records resolve to the atom "
            + scatterers[j].label + ": " + matched_by[j] + " and " + record);
        }
        matched_by[j] = record;
        atoms[j].swap(shells);
        z[j] = atom_z;
        std::map<FloatType, std::size_t> by_zeta;
        std::vector<slater_group> &groups = slater_groups[j];
        for (std::size_t k = 0; k < sl.size(); k++) {
          if (sl[k].n != std::floor(sl[k].n) || sl[k].n < 0) {
            fractional_slaters[j].push_back(sl[k]);
            continue;
          }
          typename std::map<FloatType, std::size_t>::iterator g =
            by_zeta.find(sl[k].zeta);
          if (g == by_zeta.end()) {
            g = by_zeta.insert(std::make_pair(sl[k].zeta, groups.size())).first;
            slater_group ng;
            ng.zeta = sl[k].zeta;
            groups.push_back(ng);
          }
          slater_group &grp = groups[g->second];
          int const m = static_cast<int>(sl[k].n) + 2;
          if (grp.w.size() <= static_cast<std::size_t>(m)) grp.w.resize(m + 1, 0);
          FloatType fact = 1;                  // (n+1)! = (m-1)!
          for (int f = 2; f < m; f++) fact *= f;
          grp.w[m] += sl[k].c*fact;
        }
        slaters[j].swap(sl);
        exponentials[j].swap(ex);
        covered[j] = true;
      }
    }

  private:
    std::uint64_t file_size;

    static bool finite(double x) { return std::isfinite(x); }

    // that n more bytes are there to be read
    void fits(std::istream &in, std::uint64_t n, std::string const &label) const {
      std::uint64_t const at = static_cast<std::uint64_t>(in.tellg());
      SMTBX_ASSERT(in.good() && at <= file_size && n <= file_size - at)
        ("truncated or corrupt .tshb")(label);
    }

    /* The atom an id names when it is not the exact id of one: the atom
       shifted as by a refinement, or on a symmetry image of the site, its
       shells then rotated onto it; ~0 if none. As Olex2 matches the columns
       of a table (tsc_scatterer_resync._symmetry_matches): within 0.1 A of
       the site an atom is in place; otherwise the nearest of the site and
       its images within the lookup's tolerance, each match unambiguous, wins.
       The operations are the space group's and, without a centre, each
       composed with the inversion, as inv -f does.
     */
    std::size_t locate(
      cctbx::xray::scatterer_ID_lookup<FloatType> const &lookup,
      cctbx::xray::scatterer_id_big<FloatType, cctbx::fractional<FloatType> > const &id,
      uctbx::unit_cell const &unit_cell,
      sgtbx::space_group const &space_group,
      af::shared<xray::scatterer<FloatType> > const &scatterers,
      std::vector<shell> &shells,
      std::string const &record)
    {
      typedef cctbx::xray::scatterer_id_big<FloatType, cctbx::fractional<FloatType> > id_t;
      cctbx::fractional<FloatType> const site = id.get_crd();
      std::size_t best = lookup.index_cartesian(id);
      double best_d = best == ~std::size_t(0) ? 1e300
        : unit_cell.mod_short_distance(scatterers[best].site, site);
      if (best_d <= 0.1) return best;
      std::vector<sgtbx::rt_mx> ops;
      for (std::size_t i = 0; i < space_group.order_z(); i++) {
        ops.push_back(space_group(i));
      }
      if (!space_group.is_centric()) {
        // -x,-y,-z after each operation
        for (std::size_t i = 0; i < space_group.order_z(); i++) {
          ops.push_back(-space_group(i));
        }
      }
      std::size_t best_op = ops.size();
      for (std::size_t o = 0; o < ops.size(); o++) {
        if (ops[o].is_unit_mx()) continue;
        cctbx::fractional<FloatType> const image = ops[o]*site;
        std::size_t const j = lookup.index_cartesian(
          id_t(image, id.get_data(), id.get_z()));
        if (j == ~std::size_t(0)) continue;
        double const d = unit_cell.mod_short_distance(scatterers[j].site, image);
        if (d < best_d) {
          best = j; best_d = d; best_op = o;
        }
      }
      if (best_op == ops.size()) return best;
      sgtbx::rot_mx const r = ops[best_op].r();
      if (!r.is_unit_mx()) {
        scitbx::mat3<double> rf;
        for (int k = 0; k < 9; k++) rf[k] = double(r.num()[k])/r.den();
        scitbx::mat3<double> const q = unit_cell.orthogonalization_matrix()
          *rf*unit_cell.fractionalization_matrix();
        rotate_shells(shells, q);
      }
      moved.push_back(record + " by " + ops[best_op].as_xyz());
      return best;
    }

  public:
    /// shells of a density rho(r) rewritten for rho(Q^-1 r), Q orthogonal
    /** Per l the coefficients of u -> sum_n c_n Y_n(Q^T u) on the Y_m:
        c'_m = int Y_m(u) sum_n c_n Y_n(Q^T u) du, by a product quadrature,
        Gauss-Legendre in cos theta (l+1 points) and 2l+1 equal steps in phi,
        exact for the degree 2l of the integrand. An improper Q needs no
        special case: its parity comes out of the harmonics.
     */
    static void rotate_shells(std::vector<shell> &shells,
                              scitbx::mat3<double> const &q)
    {
      double const pi = scitbx::constants::pi;
      for (std::size_t s = 0; s < shells.size(); s++) {
        int const l = shells[s].l;
        if (l == 0) continue;
        std::vector<double> x, w;
        gauss_legendre(l + 1, x, w);
        int const n_phi = 2*l + 1;
        std::vector<FloatType> y((l + 1)*(l + 1)), yq((l + 1)*(l + 1));
        std::vector<FloatType> c(2*l + 1, 0);
        for (std::size_t t = 0; t < x.size(); t++) {
          double const ct = x[t], st = std::sqrt(std::max(0., 1 - ct*ct));
          for (int p = 0; p < n_phi; p++) {
            double const phi = 2*pi*p/n_phi;
            double const u[3] = { st*std::cos(phi), st*std::sin(phi), ct };
            double v[3];      // Q^T u
            for (int i = 0; i < 3; i++) {
              v[i] = q(0, i)*u[0] + q(1, i)*u[1] + q(2, i)*u[2];
            }
            double const nv = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            real_spherical_harmonics<FloatType>(l, u[0], u[1], u[2], &y[0]);
            real_spherical_harmonics<FloatType>(l, v[0]/nv, v[1]/nv, v[2]/nv, &yq[0]);
            double g = 0;
            for (int m = 0; m < 2*l + 1; m++) g += shells[s].c[m]*yq[l*l + m];
            double const weight = w[t]*2*pi/n_phi;
            for (int m = 0; m < 2*l + 1; m++) c[m] += weight*y[l*l + m]*g;
          }
        }
        shells[s].c = c;
      }
    }

  private:
    static void gauss_legendre(int n, std::vector<double> &x,
                               std::vector<double> &w)
    {
      double const pi = scitbx::constants::pi;
      x.resize(n); w.resize(n);
      for (int i = 0; i < n; i++) {
        double z = std::cos(pi*(i + 0.75)/(n + 0.5)), dp = 0;
        for (int it = 0; it < 100; it++) {
          double p0 = 1, p1 = z;
          for (int k = 2; k <= n; k++) {
            double const p2 = ((2*k - 1)*z*p1 - (k - 1)*p0)/k;
            p0 = p1; p1 = p2;
          }
          if (n == 1) { p1 = z; p0 = 1; }
          dp = n*(z*p1 - p0)/(z*z - 1);
          double const dz = p1/dp;
          z -= dz;
          if (std::abs(dz) < 1e-15) break;
        }
        x[i] = z;
        w[i] = 2/((1 - z*z)*dp*dp);
      }
    }

    static void read(std::istream &in, char *buffer, std::size_t n) {
      in.read(buffer, n);
      SMTBX_ASSERT(in.gcount() == static_cast<std::streamsize>(n))
        ("truncated .tshb");
    }
    static std::uint32_t read_u32(std::istream &in) {
      std::uint32_t v;
      read(in, reinterpret_cast<char *>(&v), 4);
      return v;
    }
    std::string read_string(std::istream &in) const {
      std::uint32_t n = read_u32(in);
      fits(in, n, "");
      std::string s(n, '\0');
      if (n) read(in, &s[0], n);
      return s;
    }
  };

  /// The form factors of a .tshb, for the structure-factor code.
  template <typename FloatType>
  class harmonic_expansion
    : public direct::one_scatterer_one_h::scatterer_contribution<FloatType>
  {
    typedef direct::one_scatterer_one_h::scatterer_contribution<FloatType>
      base_type;
  public:
    typedef FloatType float_type;
    typedef std::complex<float_type> complex_type;
    typedef harmonic_expansion_data<FloatType> data_type;
  private:
    /* The radial factor of a primitive depends on its exponent, its l and |s|
       alone, and the atoms of one element share their basis, so the distinct
       (alpha, l) pairs of the whole structure - a hundred or so - are
       evaluated once per reflection instead of once per primitive per atom.
     */
    // the primitives of one l of one atom: their (alpha, l) pairs and their
    // coefficients, row by row, contiguous
    struct l_block {
      int l;
      int pos;                      // of m = -l in the atom's parity order
      std::vector<int> u;
      std::vector<float_type> c;    // u.size() x (2l+1)
    };
    /* Atoms whose primitives are the same (alpha, l) pairs in the same order
       - the atoms of one element, from one auxiliary basis - form a group,
       and the sums over primitives are done for all of a group's atoms at
       once: row p of a group block holds c_p,lm of every member, so that the
       sum is a few long, contiguous multiply-adds instead of a short loop
       over 2l+1 values per atom.
     */
    struct group_block {
      int l, pos;                   // pos as in l_block, the same for all
      std::vector<int> u;
      std::vector<float_type> c;    // u.size() x n_members x (2l+1)
    };
    struct basis_group {
      std::vector<std::size_t> members;
      std::vector<group_block> blocks;
    };
    struct shared_tables {
      std::vector<float_type> alpha, pi_over_alpha_32;
      std::vector<int> l;
      std::vector<std::vector<l_block> > atoms;
      std::vector<int> atom_lmax;
      std::vector<int> n_even;      // per atom: its coefficients of even l
      std::vector<basis_group> groups;
      std::vector<std::size_t> coef_offset;   // per atom, into coef
      std::size_t n_coef_total, max_row;
    };
    /* An atom's coefficients a_lm are kept in parity order: those of even l
       first, then those of odd l, each in increasing l. i^l makes the even l
       contribute to the real part of f alone and the odd l to the imaginary
       part alone, so in this order each part is one contiguous dot product,
       with no multiplications by the zeros the other parity would leave.
     */
    static int n_even(int lmax) {
      int n = 0;
      for (int l = 0; l <= lmax; l += 2) n += 2*l + 1;
      return n;
    }
    static int parity_offset(int l, int lmax) {
      int pos = (l & 1) ? n_even(lmax) : 0;
      for (int k = l & 1; k < l; k += 2) pos += 2*k + 1;
      return pos;
    }
    /* Every reflection is worked in the frame of its orbit's canonical member
       c, the largest of +-h R_i: with c = sigma h R_m and R_m R_k = tau R_i,
       f(h R_i) = f(c R_k), conjugated if sigma tau = -1 (a real density has
       f(-s) = f(s)*; Mott-Bethe keeps that). So the n_smx values f(c R_k) of
       an atom serve every equivalent and every Friedel mate of h.

       The densities do not change during a refinement and the contribution
       outlives the least-squares cycles (Olex2 keeps it), so those values are
       kept per orbit, shared by all forks, up to a memory budget
       (SMTBX_TSHB_CACHE_MB, default 1024, 0 for none; past it, computed every
       time). Per atom whichever is smaller is stored: the n_smx complex
       values, or the (lmax+1)^2 coefficients sum_p g_p(s) c_p,lm - which
       depend on |s| alone - with the spherical records and Mott-Bethe folded
       into them, from which the values are regenerated by n_smx dot products
       with Y_lm(c R_k). Low symmetry stores the values (4 reals per atom in
       P2_1/c), high symmetry the coefficients (25 for lmax 4, against 48
       values in m-3m).
     */
    struct cache_entry {
      std::atomic<bool> ready;   // set once v is filled, by its one writer
      std::vector<float_type> v;
      cache_entry() : ready(false) {}
    };
    struct shared_cache {
      // fixed at construction
      std::vector<int> product_k, product_tau;  // [m n_smx + i]: R_m R_k = tau R_i
      std::vector<int> n_coef;                  // (atom lmax + 1)^2
      std::vector<char> folded;                 // stores the coefficients
      std::vector<std::size_t> offset;          // into cache_entry::v
      std::size_t entry_size, max_entries;
      // the cache proper
      std::mutex mutex;
      std::unordered_map<std::uint64_t, cache_entry *> index;
      std::deque<cache_entry> entries;          // never moves an element
    };
    boost::shared_ptr<data_type const> data;   // shared between forks
    boost::shared_ptr<shared_tables const> tables;
    boost::shared_ptr<shared_cache> cache;
    scitbx::mat3<double> frac;                 // x_frac = frac x_cart
    std::vector<scitbx::mat3<int> > rotations;
    int n_lm, n_lm_even;                       // all, and those of even l
    /* Per reflection: the harmonics of every c R_k, |s| and the radial
       factors, where each h R_i comes from, and every atom's coefficients.
       Every atom of one reflection is asked for before the next reflection
       (see base_fc::compute_), so this is set up once per reflection and not
       once per atom; each thread works through its own fork.
     */
    // i^l Y_lm(c R_k): even l, real, n_smx x n_lm_even; odd l, imaginary
    mutable std::vector<float_type> ylm_even, ylm_odd, ylm;
    mutable std::vector<float_type> radial;    // per (alpha, l) pair
    mutable float_type s;
    mutable miller::index<> cached_h, canonical;
    mutable bool cache_valid, ylm_valid;
    mutable cache_entry const *hit;            // 0: computed, not looked up
    mutable std::vector<int> source_k;         // h R_i from c R_k ...
    mutable std::vector<char> source_conj;     // ... conjugated or not
    mutable std::vector<complex_type> f_c, tmp;  // f(c R_k), f(h R_i)
    // every atom's coefficients for the reflection, and a group's sums
    mutable std::vector<float_type> coef, row;
    /* Per atom on a special position whose density its stabiliser leaves
       unchanged: for each operator k the first operator with the same value,
       k' with R_k = R_k' S for an S of the stabiliser, so that f(h R_k) =
       f(h R_k' S) = f(h R_k'); empty for every other atom. Shared by forks.
     */
    boost::shared_ptr<std::vector<std::vector<int> > const> same_as;

    static boost::shared_ptr<shared_tables const> make_tables(data_type const &d) {
      boost::shared_ptr<shared_tables> t(new shared_tables);
      std::map<std::pair<float_type, int>, int> index;
      float_type const pi = scitbx::constants::pi;
      t->atoms.resize(d.atoms.size());
      t->atom_lmax.assign(d.atoms.size(), 0);
      for (std::size_t a = 0; a < d.atoms.size(); a++) {
        for (std::size_t k = 0; k < d.atoms[a].size(); k++) {
          typename data_type::shell const &sh = d.atoms[a][k];
          std::pair<float_type, int> key(sh.alpha, sh.l);
          typename std::map<std::pair<float_type, int>, int>::iterator i =
            index.find(key);
          int u;
          if (i == index.end()) {
            u = static_cast<int>(t->alpha.size());
            index[key] = u;
            t->alpha.push_back(sh.alpha);
            t->l.push_back(sh.l);
            t->pi_over_alpha_32.push_back(std::pow(pi/sh.alpha, float_type(1.5)));
          }
          else u = i->second;
          std::vector<l_block> &blocks = t->atoms[a];
          if (blocks.size() <= static_cast<std::size_t>(sh.l)) {
            blocks.resize(sh.l + 1);
            for (int l = 0; l <= sh.l; l++) blocks[l].l = l;
          }
          blocks[sh.l].u.push_back(u);
          blocks[sh.l].c.insert(blocks[sh.l].c.end(), sh.c.begin(), sh.c.end());
          if (sh.l > t->atom_lmax[a]) t->atom_lmax[a] = sh.l;
        }
        for (std::size_t b = 0; b < t->atoms[a].size(); b++) {
          t->atoms[a][b].pos = parity_offset(t->atoms[a][b].l, t->atom_lmax[a]);
        }
        t->n_even.push_back(n_even(t->atom_lmax[a]));
      }
      // the groups: atoms with the same primitives, in the same order
      std::map<std::vector<int>, std::size_t> by_basis;
      t->n_coef_total = 0;
      for (std::size_t a = 0; a < t->atoms.size(); a++) {
        std::vector<int> key(1, t->atom_lmax[a]);
        std::vector<l_block> const &blocks = t->atoms[a];
        for (std::size_t b = 0; b < blocks.size(); b++) {
          if (blocks[b].u.empty()) continue;
          key.push_back(blocks[b].l);
          key.push_back(static_cast<int>(blocks[b].u.size()));
          key.insert(key.end(), blocks[b].u.begin(), blocks[b].u.end());
        }
        typename std::map<std::vector<int>, std::size_t>::iterator g =
          by_basis.find(key);
        if (g == by_basis.end()) {
          g = by_basis.insert(std::make_pair(key, t->groups.size())).first;
          t->groups.push_back(basis_group());
        }
        t->groups[g->second].members.push_back(a);
        t->coef_offset.push_back(t->n_coef_total);
        t->n_coef_total += (t->atom_lmax[a] + 1)*(t->atom_lmax[a] + 1);
      }
      t->max_row = 0;
      for (std::size_t g = 0; g < t->groups.size(); g++) {
        basis_group &grp = t->groups[g];
        std::size_t const n_mem = grp.members.size();
        std::vector<l_block> const &first = t->atoms[grp.members[0]];
        for (std::size_t b = 0; b < first.size(); b++) {
          if (first[b].u.empty()) continue;
          group_block gb;
          gb.l = first[b].l;
          gb.pos = first[b].pos;
          gb.u = first[b].u;
          std::size_t const w = 2*gb.l + 1, n_p = gb.u.size();
          gb.c.resize(n_p*n_mem*w);
          for (std::size_t k = 0; k < n_mem; k++) {
            std::vector<float_type> const &c = t->atoms[grp.members[k]][b].c;
            for (std::size_t q = 0; q < n_p; q++) {
              std::copy(&c[q*w], &c[q*w] + w, &gb.c[(q*n_mem + k)*w]);
            }
          }
          t->max_row = std::max(t->max_row, n_mem*w);
          grp.blocks.push_back(gb);
        }
      }
      return t;
    }

    static boost::shared_ptr<shared_cache> make_cache(shared_tables const &t,
      std::vector<scitbx::mat3<int> > const &r)
    {
      boost::shared_ptr<shared_cache> c(new shared_cache);
      std::size_t const n = r.size();
      c->product_k.resize(n*n);
      c->product_tau.resize(n*n);
      for (std::size_t m = 0; m < n; m++) {
        for (std::size_t k = 0; k < n; k++) {
          scitbx::mat3<int> const p = r[m]*r[k];
          bool found = false;
          for (std::size_t i = 0; i < n && !found; i++) {
            for (int tau = 1; tau >= -1 && !found; tau -= 2) {
              if (p == r[i]*tau) {
                c->product_k[m*n + i] = static_cast<int>(k);
                c->product_tau[m*n + i] = tau;
                found = true;
              }
            }
          }
          SMTBX_ASSERT(found)("rotation parts not closed");
        }
      }
      std::size_t offset = 0;
      for (std::size_t j = 0; j < t.atoms.size(); j++) {
        int const n_j = (t.atom_lmax[j] + 1)*(t.atom_lmax[j] + 1);
        bool const fold = static_cast<std::size_t>(n_j) < 2*n;
        c->n_coef.push_back(n_j);
        c->folded.push_back(fold);
        c->offset.push_back(offset);
        offset += fold ? n_j : 2*n;
      }
      c->entry_size = offset;
      double mb = 1024;
      if (char const *e = std::getenv("SMTBX_TSHB_CACHE_MB")) mb = std::atof(e);
      // the values, the entry and its hash node
      double const per_entry = offset*sizeof(float_type) + sizeof(cache_entry) + 64;
      c->max_entries = mb > 0 ? static_cast<std::size_t>(mb*1048576/per_entry) : 0;
      return c;
    }

  public:
    harmonic_expansion(boost::shared_ptr<data_type const> const &data_,
                       uctbx::unit_cell const &unit_cell,
                       sgtbx::space_group const &space_group)
      : data(data_),
        tables(make_tables(*data_)),
        frac(unit_cell.fractionalization_matrix()),
        n_lm((data_->lmax + 1)*(data_->lmax + 1)),
        n_lm_even(n_even(data_->lmax)),
        ylm_even(space_group.n_smx()*n_lm_even),
        ylm_odd(space_group.n_smx()*(n_lm - n_lm_even)),
        ylm(n_lm),
        radial(tables->alpha.size()),
        s(0),
        cache_valid(false),
        ylm_valid(false),
        hit(0),
        source_k(space_group.n_smx()),
        source_conj(space_group.n_smx()),
        f_c(space_group.n_smx()),
        tmp(space_group.n_smx()),
        coef(tables->n_coef_total),
        row(tables->max_row)
    {
      for (std::size_t i = 0; i < space_group.n_smx(); i++) {
        SMTBX_ASSERT(space_group.smx(i).r().den() == 1);
        rotations.push_back(space_group.smx(i).r().num());
      }
      /* Nothing here depends on where the identity is - the canonical member
         of an orbit is the largest of +-h R_i over all i, and f(h R_i) is
         found through R_m R_k = tau R_i for any order - but the structure
         factor code pairs get_full's values with smx(i) one for one, and
         cctbx keeps the identity first (space_group::reset). Said here, so
         that a group built otherwise fails loudly rather than quietly.
       */
      SMTBX_ASSERT(rotations.size() && rotations[0] == scitbx::mat3<int>(1, 0, 0, 0, 1, 0, 0, 0, 1))
        ("the first operator of the space group must be the identity");
      cache = make_cache(*tables, rotations);
      same_as = make_same_as(*data_, unit_cell, space_group, rotations);
    }

  private:
    static boost::shared_ptr<std::vector<std::vector<int> > const> make_same_as(
      data_type const &d, uctbx::unit_cell const &unit_cell,
      sgtbx::space_group const &space_group,
      std::vector<scitbx::mat3<int> > const &rotations)
    {
      boost::shared_ptr<std::vector<std::vector<int> > > r(
        new std::vector<std::vector<int> >(d.atoms.size()));
      std::size_t const n = rotations.size();
      if (n == 1) return r;
      for (std::size_t j = 0; j < d.atoms.size() && j < d.sites.size(); j++) {
        // only a site that is special exactly: one merely near a special
        // position is a general position here
        sgtbx::site_symmetry ss(unit_cell, space_group, d.sites[j], 0.01, false);
        if (ss.is_point_group_1() || ss.distance_moved() > 1e-6) continue;
        std::vector<scitbx::mat3<int> > stab;
        for (std::size_t k = 0; k < ss.matrices().size(); k++) {
          sgtbx::rot_mx const rm = ss.matrices()[k].r();
          if (rm.is_unit_mx() || rm.den() != 1) continue;
          stab.push_back(rm.num());
        }
        if (stab.empty()) continue;
        // and only a density that is in fact invariant: a numerical one need
        // not be exactly, and then every value is computed as before
        bool invariant = true;
        for (std::size_t k = 0; k < stab.size() && invariant; k++) {
          scitbx::mat3<double> sf;
          for (int q = 0; q < 9; q++) sf[q] = stab[k][q];
          scitbx::mat3<double> const qm = unit_cell.orthogonalization_matrix()
            *sf*unit_cell.fractionalization_matrix();
          std::vector<typename data_type::shell> rotated = d.atoms[j];
          data_type::rotate_shells(rotated, qm);
          for (std::size_t sh = 0; sh < rotated.size() && invariant; sh++) {
            for (std::size_t m = 0; m < rotated[sh].c.size(); m++) {
              FloatType const a = d.atoms[j][sh].c[m];
              if (std::abs(rotated[sh].c[m] - a) > 1e-12*std::max(FloatType(1), std::abs(a))) {
                invariant = false; break;
              }
            }
          }
        }
        if (!invariant) continue;
        std::vector<int> map(n);
        bool any = false;
        for (std::size_t k = 0; k < n; k++) {
          map[k] = static_cast<int>(k);
          for (std::size_t kk = 0; kk < k && map[k] == int(k); kk++) {
            if (map[kk] != int(kk)) continue;
            for (std::size_t q = 0; q < stab.size(); q++) {
              if (rotations[kk]*stab[q] == rotations[k]) {
                map[k] = static_cast<int>(kk); any = true; break;
              }
            }
          }
        }
        if (any) (*r)[j].swap(map);
      }
      return r;
    }

  public:

    harmonic_expansion(harmonic_expansion const &other)
      : data(other.data),
        tables(other.tables),
        cache(other.cache),
        frac(other.frac),
        rotations(other.rotations),
        n_lm(other.n_lm),
        n_lm_even(other.n_lm_even),
        ylm_even(other.ylm_even.size()),
        ylm_odd(other.ylm_odd.size()),
        ylm(other.ylm.size()),
        radial(other.radial.size()),
        s(0),
        cache_valid(false),
        ylm_valid(false),
        hit(0),
        source_k(other.source_k.size()),
        source_conj(other.source_conj.size()),
        f_c(other.f_c.size()),
        tmp(other.tmp.size()),
        coef(other.coef.size()),
        row(other.row.size()),
        same_as(other.same_as)
    {}

  private:
    /* The canonical member of h's orbit, where each h R_i comes from, and
       the cached entry: looked up, filled here if new and within budget, or
       none - another thread still filling it, or the budget spent - and then
       computed per atom.
     */
    void set_reflection(miller::index<> const &h) const {
      std::size_t const n = rotations.size();
      int sigma = 1;
      std::size_t m = 0;
      for (std::size_t i = 0; i < n; i++) {
        scitbx::mat3<int> const &r = rotations[i];
        int v[3];
        for (int c = 0; c < 3; c++) {
          v[c] = h[0]*r(0, c) + h[1]*r(1, c) + h[2]*r(2, c);
        }
        for (int sg = 1; sg >= -1; sg -= 2) {
          miller::index<> const w(sg*v[0], sg*v[1], sg*v[2]);
          if ((i == 0 && sg == 1) || w[0] > canonical[0] ||
              (w[0] == canonical[0] && (w[1] > canonical[1] ||
                (w[1] == canonical[1] && w[2] > canonical[2]))))
          {
            canonical = w; sigma = sg; m = i;
          }
        }
      }
      shared_cache &c = *cache;
      for (std::size_t i = 0; i < n; i++) {
        source_k[i] = c.product_k[m*n + i];
        source_conj[i] = sigma*c.product_tau[m*n + i] < 0;
      }
      cached_h = h;
      cache_valid = true;
      ylm_valid = false;
      hit = 0;
      if (c.max_entries == 0) {
        prepare(canonical);
        all_coefficients();
        return;
      }
      for (int k = 0; k < 3; k++) {
        SMTBX_ASSERT(canonical[k] > -(1 << 20) && canonical[k] < (1 << 20));
      }
      std::uint64_t const key =
        (std::uint64_t(canonical[0] + (1 << 20)) << 42) |
        (std::uint64_t(canonical[1] + (1 << 20)) << 21) |
        std::uint64_t(canonical[2] + (1 << 20));
      cache_entry *e = 0;
      bool fill = false;
      {
        std::lock_guard<std::mutex> lock(c.mutex);
        typename std::unordered_map<std::uint64_t, cache_entry *>::iterator
          i = c.index.find(key);
        if (i != c.index.end()) e = i->second;
        else if (c.entries.size() < c.max_entries) {
          c.entries.emplace_back();
          e = &c.entries.back();
          c.index[key] = e;
          fill = true;
        }
      }
      if (fill) {
        prepare(canonical);
        all_coefficients();
        e->v.resize(c.entry_size);
        for (std::size_t j = 0; j < c.offset.size(); j++) {
          float_type *out = &e->v[c.offset[j]];
          float_type const *a = &coef[tables->coef_offset[j]];
          if (c.folded[j]) {
            std::copy(a, a + c.n_coef[j], out);
          }
          else {
            values(a, j);
            for (std::size_t k = 0; k < n; k++) {
              out[2*k] = f_c[k].real();
              out[2*k + 1] = f_c[k].imag();
            }
          }
        }
        e->ready.store(true, std::memory_order_release);
        hit = e;
      }
      else if (e != 0 && e->ready.load(std::memory_order_acquire)) hit = e;
      else {
        prepare(canonical);
        all_coefficients();
      }
    }

    /* sum over primitives of g(s) c_lm, per l, for every atom, to coef in
       parity order, group by group (see basis_group); then per atom the
       spherical records and Mott-Bethe folded in through Y00, so that the
       form factors are exactly the dot products of values()
     */
    void all_coefficients() const {
      shared_tables const &t = *tables;
      // the l an atom has no primitive of
      std::fill(coef.begin(), coef.end(), float_type(0));
      for (std::size_t g = 0; g < t.groups.size(); g++) {
        basis_group const &grp = t.groups[g];
        std::size_t const n_mem = grp.members.size();
        for (std::size_t b = 0; b < grp.blocks.size(); b++) {
          group_block const &bl = grp.blocks[b];
          std::size_t const w = 2*bl.l + 1, len = n_mem*w;
          std::size_t const n_p = bl.u.size();
          float_type *out = &row[0];
          float_type const *c = &bl.c[0];
          // the first primitive assigns, the rest add two at a time: two
          // independent chains of multiply-adds instead of one
          float_type const g0 = radial[bl.u[0]];
          for (std::size_t i = 0; i < len; i++) out[i] = g0*c[i];
          c += len;
          std::size_t p = 1;
          for (; p + 1 < n_p; p += 2, c += 2*len) {
            float_type const g1 = radial[bl.u[p]], g2 = radial[bl.u[p + 1]];
            for (std::size_t i = 0; i < len; i++) out[i] += g1*c[i] + g2*c[len + i];
          }
          if (p < n_p) {
            float_type const g1 = radial[bl.u[p]];
            for (std::size_t i = 0; i < len; i++) out[i] += g1*c[i];
          }
          for (std::size_t k = 0; k < n_mem; k++) {
            std::copy(out + k*w, out + (k + 1)*w,
                      &coef[t.coef_offset[grp.members[k]] + bl.pos]);
          }
        }
      }
      for (std::size_t j = 0; j < t.atoms.size(); j++) {
        finish(j, &coef[t.coef_offset[j]]);
      }
    }

    void finish(std::size_t j, float_type *a) const {
      float_type const y00 = 1/std::sqrt(4*float_type(scitbx::constants::pi));
      // the spherical records, the same for every symmetry equivalent
      float_type const extra = spherical_records(j);
      if (data->electron) {
        // Mott-Bethe, as NoSpherA2's convert_to_ED, k (Z - f): |hR| = |h|, so
        // one factor for every equivalent. s = 0 is no reflection and divides
        // by zero there, as in NoSpherA2.
        float_type const k = float_type(0.023934)*4/(s*s);
        int const n = cache->n_coef[j];
        for (int q = 0; q < n; q++) a[q] *= -k;
        a[0] += k*(data->z[j] - extra)/y00;
      }
      else a[0] += extra/y00;
    }

    /* f_c[k] = f(c R_k) from the coefficients of atom j, in parity order.
       Each sum in two halves: a chain of dependent additions, not the
       multiplications, is what bounds a sum this short.
     */
    void values(float_type const *a, std::size_t j) const {
      int const n_e = tables->n_even[j], n_o = cache->n_coef[j] - n_e;
      int const n_lm_odd = n_lm - n_lm_even;
      float_type const *a_o = a + n_e;
      std::vector<int> const &same = (*same_as)[j];
      for (std::size_t k = 0; k < f_c.size(); k++) {
        if (!same.empty() && same[k] != int(k)) {
          f_c[k] = f_c[same[k]];
          continue;
        }
        float_type const *ye = &ylm_even[k*n_lm_even];
        float_type const *yo = &ylm_odd[k*n_lm_odd];
        float_type re0 = 0, re1 = 0, im0 = 0, im1 = 0;
        int q = 0;
        for (; q + 1 < n_e; q += 2) { re0 += a[q]*ye[q]; re1 += a[q + 1]*ye[q + 1]; }
        if (q < n_e) re0 += a[q]*ye[q];
        q = 0;
        for (; q + 1 < n_o; q += 2) { im0 += a_o[q]*yo[q]; im1 += a_o[q + 1]*yo[q + 1]; }
        if (q < n_o) im0 += a_o[q]*yo[q];
        f_c[k] = complex_type(re0 + re1, im0 + im1);
      }
    }

    void prepare(miller::index<> const &h) const {
      prepare_harmonics(h);
      // (pi/alpha)^(3/2) (pi s/alpha)^l exp(-pi^2 s^2/alpha), i^l applied later
      float_type const pi = scitbx::constants::pi;
      shared_tables const &t = *tables;
      for (std::size_t u = 0; u < radial.size(); u++) {
        float_type const a = t.alpha[u];
        float_type g0 = t.pi_over_alpha_32[u]*std::exp(-pi*pi*s*s/a);
        float_type const x = pi*s/a;
        for (int k = 0; k < t.l[u]; k++) g0 *= x;
        radial[u] = g0;
      }
    }

    void prepare_harmonics(miller::index<> const &h) const {
      for (std::size_t i = 0; i < rotations.size(); i++) {
        scitbx::mat3<int> const &r = rotations[i];
        // h R as a row vector, then its Cartesian reciprocal vector frac^T k
        double k[3];
        for (int c = 0; c < 3; c++) {
          k[c] = h[0]*r(0, c) + h[1]*r(1, c) + h[2]*r(2, c);
        }
        double v[3];
        for (int c = 0; c < 3; c++) {
          v[c] = k[0]*frac(0, c) + k[1]*frac(1, c) + k[2]*frac(2, c);
        }
        double norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (i == 0) s = norm;
        float_type *y = &ylm[0];
        if (norm > 0) {
          real_spherical_harmonics<float_type>(data->lmax,
            v[0]/norm, v[1]/norm, v[2]/norm, y);
        }
        else {
          // only l = 0 survives at s = 0, whatever direction is taken
          real_spherical_harmonics<float_type>(data->lmax, 0, 0, 1, y);
        }
        /* i^l folded in, by parity: the even l go to the real part with
           signs +, - for l = 0, 2 mod 4, the odd l to the imaginary part with
           +, - for l = 1, 3 mod 4. Two plain dot products per operator then,
           and no products with zeros.
         */
        float_type *ye = &ylm_even[i*n_lm_even];
        float_type *yo = &ylm_odd[i*(n_lm - n_lm_even)];
        for (int l = 0; l <= data->lmax; l++) {
          float_type const sign = (l & 2) ? -1 : 1;
          float_type *out = (l & 1) ? yo + parity_offset(l, data->lmax) - n_lm_even
                                    : ye + parity_offset(l, data->lmax);
          for (int m = 0; m < 2*l + 1; m++) out[m] = sign*y[l*l + m];
        }
      }
      ylm_valid = true;
    }

  public:
    virtual complex_type get(std::size_t scatterer_idx,
      miller::index<> const &h) const
    {
      SMTBX_NOT_IMPLEMENTED();
      throw 1;
    }

    virtual std::vector<complex_type> const &get_full(
      std::size_t scatterer_idx, miller::index<> const &h) const
    {
      if (!cache_valid || cached_h != h) set_reflection(h);
      shared_cache const &c = *cache;
      if (hit == 0) {
        values(&coef[tables->coef_offset[scatterer_idx]], scatterer_idx);
      }
      else {
        float_type const *v = &hit->v[c.offset[scatterer_idx]];
        if (c.folded[scatterer_idx]) {
          if (!ylm_valid) prepare_harmonics(canonical);
          values(v, scatterer_idx);
        }
        else {
          for (std::size_t k = 0; k < f_c.size(); k++) {
            f_c[k] = complex_type(v[2*k], v[2*k + 1]);
          }
        }
      }
      for (std::size_t i = 0; i < tmp.size(); i++) {
        complex_type const &f = f_c[source_k[i]];
        tmp[i] = source_conj[i] ? std::conj(f) : f;
      }
      return tmp;
    }

    float_type spherical_records(std::size_t j) const {
      float_type const pi = scitbx::constants::pi;
      float_type f = 0;
      std::vector<typename data_type::exponential> const &ex =
        data->exponentials[j];
      for (std::size_t k = 0; k < ex.size(); k++) {
        f += ex[k].a*std::exp(-ex[k].b*s);
      }
      float_type const k = 2*pi*s;
      // non-integer n: 4 pi c Gamma(n+2) sin(m theta)/(k rho^m), m = n+2
      std::vector<typename data_type::slater> const &fs =
        data->fractional_slaters[j];
      for (std::size_t t = 0; t < fs.size(); t++) {
        float_type const m = fs[t].n + 2, z = fs[t].zeta;
        if (k == 0) {
          f += 4*pi*fs[t].c*std::tgamma(m + 1)/std::pow(z, m + 1);
        }
        else {
          float_type const rho = std::sqrt(z*z + k*k);
          f += 4*pi*fs[t].c*std::tgamma(m)*std::sin(m*std::atan2(k, z))
               /(k*std::pow(rho, m));
        }
      }
      std::vector<typename data_type::slater_group> const &groups =
        data->slater_groups[j];
      if (groups.empty()) return f;
      float_type sum = 0;
      for (std::size_t g = 0; g < groups.size(); g++) {
        typename data_type::slater_group const &grp = groups[g];
        std::size_t const n_m = grp.w.size();
        if (k == 0) {
          // Im (zeta - ik)^-m / k -> m / zeta^(m+1)
          float_type p = 1/grp.zeta;
          for (std::size_t m = 1; m < n_m; m++) {
            p /= grp.zeta;
            sum += grp.w[m]*m*p;
          }
        }
        else {
          // 1/(zeta - ik) = (zeta + ik)/(zeta^2 + k^2), its powers by hand:
          // std::complex guards against infinities and NaN in every
          // multiplication, which cost a third of this
          float_type const d = 1/(grp.zeta*grp.zeta + k*k);
          float_type const wr = grp.zeta*d, wi = k*d;
          float_type pr = wr, pi_ = wi, acc = 0;
          for (std::size_t m = 2; m < n_m; m++) {
            float_type const t = pr*wr - pi_*wi;
            pi_ = pr*wi + pi_*wr;
            pr = t;
            acc += grp.w[m]*pi_;
          }
          sum += acc/k;
        }
      }
      return f + 4*pi*sum;
    }

    virtual base_type &at_d_star_sq(float_type d_star_sq) {
      return *this;
    }

    virtual bool is_spherical() const {
      return false;
    }

    virtual af::shared<std::string> records_not_in_structure() const {
      af::shared<std::string> r;
      for (std::size_t i = 0; i < data->unmatched.size(); i++) {
        r.push_back(data->unmatched[i]);
      }
      return r;
    }

    virtual af::shared<std::string> records_moved_by_symmetry() const {
      af::shared<std::string> r;
      for (std::size_t i = 0; i < data->moved.size(); i++) {
        r.push_back(data->moved[i]);
      }
      return r;
    }

    virtual std::size_t n_scatterers() const { return data->atoms.size(); }

    virtual base_type *raw_fork() const {
      return new harmonic_expansion(*this);
    }
  };

}}} // smtbx::structure_factors::table_based

#endif
