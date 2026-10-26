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
    int n_lm;
    std::vector<int> l_mod_4;                  // per (l, m) index
    /* Per reflection: the harmonics of every c R_k, |s| and the radial
       factors, and where each h R_i comes from. Every atom of one reflection
       is asked for before the next reflection (see base_fc::compute_), so
       this is set up once per reflection and not once per atom; each thread
       works through its own fork.
     */
    mutable std::vector<float_type> ylm_re, ylm_im;   // n_smx x n_lm, i^l in
    mutable std::vector<float_type> radial;    // per (alpha, l) pair
    mutable float_type s;
    mutable miller::index<> cached_h, canonical;
    mutable bool cache_valid, ylm_valid;
    mutable cache_entry const *hit;            // 0: computed, not looked up
    mutable std::vector<int> source_k;         // h R_i from c R_k ...
    mutable std::vector<char> source_conj;     // ... conjugated or not
    mutable std::vector<complex_type> f_c, tmp;  // f(c R_k), f(h R_i)

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
        ylm_re(space_group.n_smx()*n_lm),
        ylm_im(space_group.n_smx()*n_lm),
        radial(tables->alpha.size()),
        s(0),
        cache_valid(false),
        ylm_valid(false),
        hit(0),
        source_k(space_group.n_smx()),
        source_conj(space_group.n_smx()),
        f_c(space_group.n_smx()),
        tmp(space_group.n_smx())
    {
      for (int l = 0; l <= data_->lmax; l++)
        for (int m = 0; m < 2*l + 1; m++) l_mod_4.push_back(l & 3);
      for (std::size_t i = 0; i < space_group.n_smx(); i++) {
        SMTBX_ASSERT(space_group.smx(i).r().den() == 1);
        rotations.push_back(space_group.smx(i).r().num());
      }
      cache = make_cache(*tables, rotations);
    }

    harmonic_expansion(harmonic_expansion const &other)
      : data(other.data),
        tables(other.tables),
        cache(other.cache),
        frac(other.frac),
        rotations(other.rotations),
        n_lm(other.n_lm),
        l_mod_4(other.l_mod_4),
        ylm_re(other.ylm_re.size()),
        ylm_im(other.ylm_im.size()),
        radial(other.radial.size()),
        s(0),
        cache_valid(false),
        ylm_valid(false),
        hit(0),
        source_k(other.source_k.size()),
        source_conj(other.source_conj.size()),
        f_c(other.f_c.size()),
        tmp(other.tmp.size())
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
        e->v.resize(c.entry_size);
        float_type a[441];
        for (std::size_t j = 0; j < c.offset.size(); j++) {
          float_type *out = &e->v[c.offset[j]];
          coefficients(j, a);
          if (c.folded[j]) {
            std::copy(a, a + c.n_coef[j], out);
          }
          else {
            values(a, c.n_coef[j]);
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
      else prepare(canonical);
    }

    /* sum over primitives of g(s) c_lm, per l, to a[0 .. n_coef), with the
       spherical records and Mott-Bethe folded in through Y00, so that the
       form factors are exactly the dot products of values()
     */
    void coefficients(std::size_t j, float_type *a) const {
      shared_tables const &t = *tables;
      std::vector<l_block> const &blocks = t.atoms[j];
      a[0] = 0;   // an atom of spherical records alone has no blocks
      for (std::size_t b = 0; b < blocks.size(); b++) {
        l_block const &bl = blocks[b];
        int const w = 2*bl.l + 1;
        float_type *out = a + bl.l*bl.l;
        int const n_p = static_cast<int>(bl.u.size());
        if (n_p == 0) {
          for (int m = 0; m < w; m++) out[m] = 0;
          continue;
        }
        float_type const *c = &bl.c[0];
        // the first primitive assigns, the rest add two at a time: two
        // independent chains of multiply-adds instead of one, which is what
        // bounds a short sum like this one
        float_type const g0 = radial[bl.u[0]];
        for (int m = 0; m < w; m++) out[m] = g0*c[m];
        c += w;
        int p = 1;
        for (; p + 1 < n_p; p += 2, c += 2*w) {
          float_type const g1 = radial[bl.u[p]], g2 = radial[bl.u[p + 1]];
          for (int m = 0; m < w; m++) out[m] += g1*c[m] + g2*c[w + m];
        }
        if (p < n_p) {
          float_type const g1 = radial[bl.u[p]];
          for (int m = 0; m < w; m++) out[m] += g1*c[m];
        }
      }
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

    // f_c[k] = f(c R_k) from the coefficients
    void values(float_type const *a, int n) const {
      for (std::size_t k = 0; k < f_c.size(); k++) {
        float_type const *yr = &ylm_re[k*n_lm], *yi = &ylm_im[k*n_lm];
        float_type re = 0, im = 0;
        for (int q = 0; q < n; q++) {
          re += a[q]*yr[q];
          im += a[q]*yi[q];
        }
        f_c[k] = complex_type(re, im);
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
        float_type *y = &ylm_re[i*n_lm], *yi = &ylm_im[i*n_lm];
        if (norm > 0) {
          real_spherical_harmonics<float_type>(data->lmax,
            v[0]/norm, v[1]/norm, v[2]/norm, y);
        }
        else {
          // only l = 0 survives at s = 0, whatever direction is taken
          real_spherical_harmonics<float_type>(data->lmax, 0, 0, 1, y);
        }
        /* i^l folded in, in place: the real part keeps l = 0, 2 mod 4 with
           signs +, -, the imaginary part takes l = 1, 3 with +, -. Two plain
           dot products per operator then, which vectorise, instead of a sum
           indexed by l mod 4.
         */
        for (int k = 0; k < n_lm; k++) {
          int const q = l_mod_4[k];
          float_type const v = y[k];
          y[k] = q == 0 ? v : (q == 2 ? -v : 0);
          yi[k] = q == 1 ? v : (q == 3 ? -v : 0);
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
      int const n = c.n_coef[scatterer_idx];
      if (hit == 0) {
        float_type a[441];
        coefficients(scatterer_idx, a);
        values(a, n);
      }
      else {
        float_type const *v = &hit->v[c.offset[scatterer_idx]];
        if (c.folded[scatterer_idx]) {
          if (!ylm_valid) prepare_harmonics(canonical);
          values(v, n);
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

    virtual base_type *raw_fork() const {
      return new harmonic_expansion(*this);
    }
  };

}}} // smtbx::structure_factors::table_based

#endif
