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
      uint32   version (1)
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
                                                        c r^n exp(-zeta r)
        uint32 n, then n x { float64 a, float64 b }    form factor terms
                                                        a exp(-b s), s in 1/A

      version 3 adds, per atom after those lists,
        int32  Z, the nuclear charge
      and the header key "ED: true" makes the form factors electron ones, by
      the Mott-Bethe formula NoSpherA2's convert_to_ED applies to its tables:
      \f$ 0.023934\,(Z - f_x)/(\sin\theta/\lambda)^2 \f$, with
      \f$ \sin\theta/\lambda = s/2 \f$, per symmetry equivalent.

    The Slater terms transform analytically too:
    \f$ 4\pi c\,(n+1)!\,\sin(m\theta)/(k\rho^m) \f$ with m = n+2,
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
#include <scitbx/constants.h>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>

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
    struct slater { FloatType c; int n; FloatType zeta; };
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
    std::vector<std::vector<slater_group> > slater_groups;
    std::vector<std::vector<exponential> > exponentials;
    af::shared<bool> covered;
    std::vector<int> z;      // nuclear charges, version 3
    bool use_ad, electron;
    int lmax;

    harmonic_expansion_data() : use_ad(true), electron(false), lmax(0) {}

    harmonic_expansion_data(
      af::shared<xray::scatterer<FloatType> > const &scatterers,
      std::string const &file_name)
      : atoms(scatterers.size()),
        slaters(scatterers.size()),
        slater_groups(scatterers.size()),
        exponentials(scatterers.size()),
        covered(scatterers.size(), false),
        z(scatterers.size(), 0),
        use_ad(true),
        electron(false),
        lmax(0)
    {
      std::ifstream in(file_name.c_str(), std::ios::in | std::ios::binary);
      SMTBX_ASSERT(in.good())(file_name);
      char magic[4];
      read(in, magic, 4);
      SMTBX_ASSERT(std::string(magic, 4) == "TSHB")(file_name);
      std::uint32_t version = read_u32(in);
      SMTBX_ASSERT(version >= 1 && version <= 3)(version);
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
      }
      SMTBX_ASSERT(!electron || version >= 3)
        ("ED needs the nuclear charges of a version 3 .tshb");
      std::map<std::string, std::size_t> by_label;
      for (std::size_t i = 0; i < scatterers.size(); i++) {
        by_label[boost::to_upper_copy(scatterers[i].label)] = i;
      }
      std::uint32_t n_atoms = read_u32(in);
      for (std::uint32_t a = 0; a < n_atoms; a++) {
        std::string label = boost::to_upper_copy(read_string(in));
        std::uint32_t n_shells = read_u32(in);
        std::vector<shell> shells(n_shells);
        for (std::uint32_t s = 0; s < n_shells; s++) {
          std::int32_t l;
          read(in, reinterpret_cast<char *>(&l), 4);
          SMTBX_ASSERT(l >= 0 && l <= 20)(label)(l);
          shells[s].l = l;
          double alpha;
          read(in, reinterpret_cast<char *>(&alpha), 8);
          SMTBX_ASSERT(alpha > 0)(label)(alpha);
          shells[s].alpha = alpha;
          std::vector<double> c(2*l + 1);
          read(in, reinterpret_cast<char *>(&c[0]), 8*c.size());
          shells[s].c.assign(c.begin(), c.end());
          if (l > lmax) lmax = l;
        }
        std::vector<slater> sl;
        std::vector<exponential> ex;
        if (version >= 2) {
          FloatType const y00 = 1/std::sqrt(4*scitbx::constants::pi);
          std::uint32_t n = read_u32(in);
          for (std::uint32_t k = 0; k < n; k++) {
            double ca[2];
            read(in, reinterpret_cast<char *>(ca), 16);
            SMTBX_ASSERT(ca[1] > 0)(label);
            // an l = 0 shell: c exp(-a r^2) = (c/Y00) Y00 exp(-a r^2)
            shell g;
            g.l = 0; g.alpha = ca[1]; g.c.assign(1, ca[0]/y00);
            shells.push_back(g);
          }
          n = read_u32(in);
          for (std::uint32_t k = 0; k < n; k++) {
            double t[3];
            read(in, reinterpret_cast<char *>(t), 24);
            slater x = { t[0], static_cast<int>(t[1]), t[2] };
            SMTBX_ASSERT(x.n >= 0 && x.n == t[1] && x.zeta > 0)(label);
            sl.push_back(x);
          }
          n = read_u32(in);
          for (std::uint32_t k = 0; k < n; k++) {
            double t[2];
            read(in, reinterpret_cast<char *>(t), 16);
            exponential x = { t[0], t[1] };
            ex.push_back(x);
          }
        }
        std::int32_t atom_z = 0;
        if (version >= 3) read(in, reinterpret_cast<char *>(&atom_z), 4);
        std::map<std::string, std::size_t>::const_iterator i =
          by_label.find(label);
        if (i == by_label.end()) continue;   // an atom no longer in the model
        atoms[i->second].swap(shells);
        z[i->second] = atom_z;
        std::map<FloatType, std::size_t> by_zeta;
        std::vector<slater_group> &groups = slater_groups[i->second];
        for (std::size_t k = 0; k < sl.size(); k++) {
          typename std::map<FloatType, std::size_t>::iterator g =
            by_zeta.find(sl[k].zeta);
          if (g == by_zeta.end()) {
            g = by_zeta.insert(std::make_pair(sl[k].zeta, groups.size())).first;
            slater_group ng;
            ng.zeta = sl[k].zeta;
            groups.push_back(ng);
          }
          slater_group &grp = groups[g->second];
          int const m = sl[k].n + 2;
          if (grp.w.size() <= static_cast<std::size_t>(m)) grp.w.resize(m + 1, 0);
          FloatType fact = 1;                  // (n+1)! = (m-1)!
          for (int f = 2; f < m; f++) fact *= f;
          grp.w[m] += sl[k].c*fact;
        }
        slaters[i->second].swap(sl);
        exponentials[i->second].swap(ex);
        covered[i->second] = true;
      }
    }

  private:
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
    static std::string read_string(std::istream &in) {
      std::uint32_t n = read_u32(in);
      SMTBX_ASSERT(n < (1u << 26))(n);
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
      std::vector<int> u;
      std::vector<float_type> c;    // u.size() x (2l+1)
    };
    struct shared_tables {
      std::vector<float_type> alpha, pi_over_alpha_32;
      std::vector<int> l;
      std::vector<std::vector<l_block> > atoms;
      std::vector<int> atom_lmax;
    };
    boost::shared_ptr<data_type const> data;   // shared between forks
    boost::shared_ptr<shared_tables const> tables;
    scitbx::mat3<double> frac;                 // x_frac = frac x_cart
    std::vector<scitbx::mat3<int> > rotations;
    int n_lm;
    std::vector<int> l_mod_4;                  // per (l, m) index
    /* Per reflection: the harmonics of every symmetry equivalent, |s| and the
       radial factors. Every atom of one reflection is asked for before the
       next reflection (see base_fc::compute_), so this is computed once per
       reflection and not once per atom; each thread works through its own
       fork.
     */
    mutable std::vector<float_type> ylm;       // n_smx x n_lm
    mutable std::vector<float_type> radial;    // per (alpha, l) pair
    mutable float_type s;
    mutable miller::index<> cached_h;
    mutable bool cache_valid;
    mutable std::vector<complex_type> tmp;

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
      }
      return t;
    }

  public:
    harmonic_expansion(boost::shared_ptr<data_type const> const &data_,
                       uctbx::unit_cell const &unit_cell,
                       sgtbx::space_group const &space_group)
      : data(data_),
        tables(make_tables(*data_)),
        frac(unit_cell.fractionalization_matrix()),
        n_lm((data_->lmax + 1)*(data_->lmax + 1)),
        ylm(space_group.n_smx()*n_lm),
        radial(tables->alpha.size()),
        s(0),
        cache_valid(false),
        tmp(space_group.n_smx())
    {
      for (int l = 0; l <= data_->lmax; l++)
        for (int m = 0; m < 2*l + 1; m++) l_mod_4.push_back(l & 3);
      for (std::size_t i = 0; i < space_group.n_smx(); i++) {
        SMTBX_ASSERT(space_group.smx(i).r().den() == 1);
        rotations.push_back(space_group.smx(i).r().num());
      }
    }

    harmonic_expansion(harmonic_expansion const &other)
      : data(other.data),
        tables(other.tables),
        frac(other.frac),
        rotations(other.rotations),
        n_lm(other.n_lm),
        l_mod_4(other.l_mod_4),
        ylm(other.ylm.size()),
        radial(other.radial.size()),
        s(0),
        cache_valid(false),
        tmp(other.tmp.size())
    {}

    void prepare(miller::index<> const &h) const {
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
        float_type *y = &ylm[i*n_lm];
        if (norm > 0) {
          real_spherical_harmonics<float_type>(data->lmax,
            v[0]/norm, v[1]/norm, v[2]/norm, y);
        }
        else {
          // only l = 0 survives at s = 0, whatever direction is taken
          real_spherical_harmonics<float_type>(data->lmax, 0, 0, 1, y);
        }
      }
      // (pi/alpha)^(3/2) (pi s/alpha)^l exp(-pi^2 s^2/alpha), i^l applied later
      float_type const pi = scitbx::constants::pi;
      shared_tables const &t = *tables;
      for (std::size_t u = 0; u < radial.size(); u++) {
        float_type const a = t.alpha[u];
        float_type g = t.pi_over_alpha_32[u]*std::exp(-pi*pi*s*s/a);
        float_type const x = pi*s/a;
        for (int k = 0; k < t.l[u]; k++) g *= x;
        radial[u] = g;
      }
      cached_h = h;
      cache_valid = true;
    }

    virtual complex_type get(std::size_t scatterer_idx,
      miller::index<> const &h) const
    {
      SMTBX_NOT_IMPLEMENTED();
      throw 1;
    }

    virtual std::vector<complex_type> const &get_full(
      std::size_t scatterer_idx, miller::index<> const &h) const
    {
      if (!cache_valid || cached_h != h) prepare(h);
      shared_tables const &t = *tables;
      std::vector<l_block> const &blocks = t.atoms[scatterer_idx];
      // the radial-weighted coefficients of this atom, independent of the
      // symmetry operator: sum over primitives of g(s) c_lm, per l
      float_type a[441];
      for (std::size_t b = 0; b < blocks.size(); b++) {
        l_block const &bl = blocks[b];
        int const w = 2*bl.l + 1;
        float_type *out = a + bl.l*bl.l;
        for (int m = 0; m < w; m++) out[m] = 0;
        float_type const *c = bl.c.empty() ? 0 : &bl.c[0];
        int const n_p = static_cast<int>(bl.u.size());
        // two primitives at a time: two independent chains of multiply-adds
        // instead of one, which is what bounds a short sum like this one
        int p = 0;
        for (; p + 1 < n_p; p += 2, c += 2*w) {
          float_type const g0 = radial[bl.u[p]], g1 = radial[bl.u[p + 1]];
          for (int m = 0; m < w; m++) out[m] += g0*c[m] + g1*c[w + m];
        }
        if (p < n_p) {
          float_type const g = radial[bl.u[p]];
          for (int m = 0; m < w; m++) out[m] += g*c[m];
        }
      }
      int const n = static_cast<int>(blocks.size()*blocks.size());
      std::size_t const n_smx = rotations.size();
      for (std::size_t i = 0; i < n_smx; i++) {
        float_type const *y = &ylm[i*n_lm];
        // by l mod 4, for i^l; four independent sums
        float_type part[4] = {0, 0, 0, 0};
        for (int k = 0; k < n; k++) part[l_mod_4[k]] += a[k]*y[k];
        tmp[i] = complex_type(part[0] - part[2], part[1] - part[3]);
      }
      // the spherical records, the same for every symmetry equivalent
      float_type const extra = spherical_records(scatterer_idx);
      if (extra != 0) {
        for (std::size_t i = 0; i < n_smx; i++) tmp[i] += extra;
      }
      if (data->electron) {
        // Mott-Bethe, as NoSpherA2's convert_to_ED: |hR| = |h|, so one factor
        // for every equivalent. s = 0 is no reflection and divides by zero
        // there, as in NoSpherA2.
        float_type const k = float_type(0.023934)*4/(s*s);
        float_type const z = data->z[scatterer_idx];
        for (std::size_t i = 0; i < n_smx; i++) {
          tmp[i] = complex_type(k*(z - tmp[i].real()), -k*tmp[i].imag());
        }
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
      std::vector<typename data_type::slater_group> const &groups =
        data->slater_groups[j];
      if (groups.empty()) return f;
      float_type const k = 2*pi*s;
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
          complex_type const w = float_type(1)/complex_type(grp.zeta, -k);
          complex_type p = w;
          for (std::size_t m = 2; m < n_m; m++) {
            p *= w;
            sum += grp.w[m]*p.imag()/k;
          }
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

    virtual base_type *raw_fork() const {
      return new harmonic_expansion(*this);
    }
  };

}}} // smtbx::structure_factors::table_based

#endif
