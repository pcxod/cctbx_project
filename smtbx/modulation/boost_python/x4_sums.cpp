// The per-class x4 sums of smtbx.modulation.calculator for one atom: F and the complex gradient columns, the
// numpy path of calculator._compute_rows (atom_sum, x4_sums) fused into one loop over reflections, operation
// classes and x4 nodes. The arrays come in through the buffer protocol (C-contiguous numpy arrays, checked on the
// Python side) and the GIL is released for the whole loop, so the calculator's atom pool runs atoms in parallel.
// The extension itself does not thread: one level of parallelism, the pool's.

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/extract.hpp>
#include <Python.h>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace smtbx { namespace modulation { namespace {

  namespace bp = boost::python;
  typedef std::complex<double> cd;

  // a C-contiguous buffer of a numpy array, released with the object
  struct buffer {
    Py_buffer view;
    bool ok;
    buffer(bp::object const &o, bool writable, std::size_t itemsize) : ok(false) {
      int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0);
      if (PyObject_GetBuffer(o.ptr(), &view, flags) != 0) bp::throw_error_already_set();
      ok = true;
      if ((std::size_t)view.itemsize != itemsize) throw std::invalid_argument("x4_sums: wrong element size");
    }
    ~buffer() { if (ok) PyBuffer_Release(&view); }
    std::size_t size() const { return view.len / view.itemsize; }
    template <typename T> T *ptr() const { return static_cast<T *>(view.buf); }
  };

  // vectorisable sin/cos and exp on arrays: Cody-Waite reduction and the fdlibm kernels (|error| < 1e-15 for the
  // arguments here), so that the compiler can use SIMD, which scalar libm calls prevent
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__INTEL_COMPILER)
#define X4_CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define X4_CLONES
#endif

  X4_CLONES
  void sincos_array(const double *x, double *s, double *c, std::size_t n)
  {
    const double two_over_pi = 0.63661977236758134308, p1 = 1.57079632673412561417e+00,
                 p2 = 6.07710050650619224932e-11, p3 = 2.02226624879595063154e-21, magic = 6755399441055744.0;
#pragma omp simd
    for (std::size_t i = 0; i < n; i++) {
      double q = (x[i] * two_over_pi + magic) - magic;     // round to nearest, no libm call (|x| < 2^50)
      double r = ((x[i] - q * p1) - q * p2) - q * p3;
      double z = r * r;
      double sr = r + r * z * (-1.66666666666666324348e-01 + z * (8.33333333332248946124e-03
                  + z * (-1.98412698298579493134e-04 + z * (2.75573137070700676789e-06
                  + z * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10)))));
      double cr = 1.0 - 0.5 * z + z * z * (4.16666666666666019037e-02 + z * (-1.38888888888741095749e-03
                  + z * (2.48015872894767294178e-05 + z * (-2.75573143513906633035e-07
                  + z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11)))));
      double qm = q - 4.0 * (((q - 1.5) * 0.25 + magic) - magic);     // the quadrant, 0..3
      bool odd = qm == 1.0 || qm == 3.0;
      double sv = odd ? cr : sr, cv = odd ? sr : cr;
      s[i] = qm >= 2.0 ? -sv : sv;
      c[i] = (qm == 1.0 || qm == 2.0) ? -cv : cv;
    }
  }

  X4_CLONES
  void exp_array(double *x, std::size_t n)
  {
    const double log2e = 1.44269504088896338700, ln2hi = 6.93147180369123816490e-01,
                 ln2lo = 1.90821492927058770002e-10, magic = 6755399441055744.0;
#pragma omp simd
    for (std::size_t i = 0; i < n; i++) {
      double v = std::min(std::max(x[i], -708.0), 709.0);
      double k = (v * log2e + magic) - magic;
      double r = (v - k * ln2hi) - k * ln2lo;
      double p = 1.0 + r * (1.0 + r * (1.0 / 2 + r * (1.0 / 6 + r * (1.0 / 24 + r * (1.0 / 120 + r * (1.0 / 720
                 + r * (1.0 / 5040 + r * (1.0 / 40320 + r * (1.0 / 362880 + r * (1.0 / 3628800
                 + r * (1.0 / 39916800 + r * (1.0 / 479001600 + r / 6227020800.0))))))))))));
      // 2^k from the bits of k + 1023 placed in the exponent field: k + 1023 + 2^52 holds it in its low bits
      double t = k + (1023.0 + 4503599627370496.0);
      std::uint64_t b;
      std::memcpy(&b, &t, 8);
      b <<= 52;
      double scale;
      std::memcpy(&scale, &b, 8);
      x[i] = p * scale;
    }
  }

  // sum_k e_k w_k over the nodes for every column, e = c + i s, w (M, K) contiguous along k
  X4_CLONES
  void node_sums(const double *c, const double *s, const double *w, std::size_t K, std::size_t M, cd *r)
  {
    for (std::size_t j = 0; j < M; j++) {
      const double *wj = w + j * K;
      double re = 0, im = 0;
#pragma omp simd reduction(+:re,im)
      for (std::size_t k = 0; k < K; k++) { re += c[k] * wj[k]; im += s[k] * wj[k]; }
      r[j] += cd(re, im);
    }
  }

  const double two_pi = 6.283185307179586476925286766559;
  const double two_pi_sq = 19.739208802178717237668981999752;

  /* ops: (hp (O,N,3), ph0 (O,N), dw6 (O,N,6), sp (O,N,3), mp (O,N), s_sq (N), cls (O) int64, conj (O) int64,
           f (O or 1, N) complex)
     atom: (site (3), u_iso, u_star (6 or 0), weight, x4 (K), disp (3,K), uw (6,K or 0), w (M,K), du (3,K or 0),
            duw (6,K or 0), w2 (M2,K or 0), specs (S,4) int64 [source 0 w / 1 w2, column, kind, first output
            column], spec_c (S) complex)
     out: (f_out (N) complex, g (N,G) complex or None); both accumulated into.
     kinds: 0 c E, 1 c O hp, 2 c O sp, 3 c E dw6, 4 c E s_sq with E = B+ r + B- r*, O = B+ r - B- r* the even and
     odd combinations of the class sums r over the class's operations and their conjugates (B the base factor). */
  void atom_sums(bp::tuple ops, bp::tuple atom, bp::tuple out)
  {
    buffer hp(ops[0], false, 8), ph0(ops[1], false, 8), dw6(ops[2], false, 8), sp(ops[3], false, 8),
           mp(ops[4], false, 8), s_sq(ops[5], false, 8), cls(ops[6], false, 8), cnj(ops[7], false, 8),
           ff(ops[8], false, 16);
    buffer site(atom[0], false, 8), u_star(atom[2], false, 8), x4(atom[4], false, 8), disp(atom[5], false, 8),
           uw(atom[6], false, 8), w(atom[7], false, 8), du(atom[8], false, 8), duw(atom[9], false, 8),
           w2(atom[10], false, 8), specs(atom[11], false, 8), spec_c(atom[12], false, 16);
    double u_iso = bp::extract<double>(atom[1]), weight = bp::extract<double>(atom[3]);
    buffer f_out(out[0], true, 16);
    bool grad = bp::object(out[1]).ptr() != Py_None;
    std::size_t n_ref = s_sq.size(), n_op = cls.size(), K = x4.size();
    std::size_t M = K ? w.size() / K : 0, M2 = K ? w2.size() / K : 0, S = spec_c.size();
    std::size_t n_f = ff.size() / (n_ref ? n_ref : 1);
    std::size_t G = 0;
    cd *g = 0;
    buffer *gb = 0;
    if (grad) {
      gb = new buffer(out[1], true, 16);
      G = gb->size() / (n_ref ? n_ref : 1);
      g = gb->ptr<cd>();
    }
    if (hp.size() != 3 * n_op * n_ref || mp.size() != n_op * n_ref || specs.size() != 4 * S
        || disp.size() != 3 * K || (uw.size() && uw.size() != 6 * K) || (M2 && du.size() != 3 * K)
        || (n_f != 1 && n_f != n_op)) {
      delete gb;
      throw std::invalid_argument("x4_sums: inconsistent array sizes");
    }
    {
      const long long *sc = specs.ptr<long long>();
      const long long width[5] = {1, 3, 3, 6, 1};
      bool bad = M == 0 || dw6.size() != 6 * n_op * n_ref || sp.size() != 3 * n_op * n_ref
                 || ph0.size() != n_op * n_ref || cnj.size() != n_op;
      for (std::size_t o = 0; o < n_op; o++) {
        long long k = cls.ptr<long long>()[o];
        if (k < 0 || (std::size_t)k >= n_op) bad = true;
      }
      for (std::size_t s = 0; s < S && grad; s++) {
        const long long *q = sc + 4 * s;
        if (q[2] < 0 || q[2] > 4 || q[1] < 0 || (std::size_t)q[1] >= (q[0] ? M2 : M) || q[3] < 0
            || (std::size_t)(q[3] + width[q[2]]) > G) bad = true;
      }
      if (bad) {
        delete gb;
        throw std::invalid_argument("x4_sums: inconsistent array sizes or column specifications");
      }
    }
    const double *Hp = hp.ptr<double>(), *Ph0 = ph0.ptr<double>(), *Dw = dw6.ptr<double>(),
                 *Sp = sp.ptr<double>(), *Mp = mp.ptr<double>(), *Ssq = s_sq.ptr<double>(),
                 *Site = site.ptr<double>(), *Ustar = u_star.size() ? u_star.ptr<double>() : 0,
                 *X4 = x4.ptr<double>(), *Disp = disp.ptr<double>(), *Uw = uw.size() ? uw.ptr<double>() : 0,
                 *W = w.ptr<double>(), *Du = du.size() ? du.ptr<double>() : 0,
                 *Duw = duw.size() ? duw.ptr<double>() : 0, *W2 = M2 ? w2.ptr<double>() : 0;
    const long long *Cls = cls.ptr<long long>(), *Cnj = cnj.ptr<long long>(), *Spec = specs.ptr<long long>();
    const cd *F = ff.ptr<cd>(), *Sc = spec_c.ptr<cd>();
    cd *Fo = f_out.ptr<cd>();

    // class leaders: the operations that some operation maps to
    std::vector<int> slot(n_op, -1);
    std::vector<std::size_t> leaders;
    for (std::size_t o = 0; o < n_op; o++) {
      std::size_t k = (std::size_t)Cls[o];
      if (slot[k] < 0) { slot[k] = (int)leaders.size(); leaders.push_back(k); }
    }
    std::size_t L = leaders.size();

    Py_BEGIN_ALLOW_THREADS
    std::vector<cd> bplus(L), bminus(L), r(M), r2(M2);
    std::vector<double> ph(K), es(K), ec(K), dwf(K), dc(K), ds(K);
    for (std::size_t n = 0; n < n_ref; n++) {
      for (std::size_t c = 0; c < L; c++) bplus[c] = bminus[c] = 0.0;
      for (std::size_t o = 0; o < n_op; o++) {
        std::size_t on = o * n_ref + n;
        double t0;
        if (Ustar) {
          double a = 0;
          for (int i = 0; i < 6; i++) a += Dw[6 * on + i] * Ustar[i];
          t0 = std::exp(a);
        }
        else t0 = std::exp(-two_pi_sq * u_iso * Ssq[n]);
        const double *h = Hp + 3 * on;
        double arg = two_pi * (Ph0[on] + (h[0] * Site[0] + h[1] * Site[1] + h[2] * Site[2]));
        cd base = weight * F[(n_f == 1 ? 0 : o) * n_ref + n] * t0 * cd(std::cos(arg), std::sin(arg));
        int s = slot[Cls[o]];
        if (Cnj[o]) bminus[s] += base; else bplus[s] += base;
      }
      for (std::size_t c = 0; c < L; c++) {
        std::size_t kn = leaders[c] * n_ref + n;
        const double *s3 = Sp + 3 * kn, *d6 = Dw + 6 * kn;
        double m = Mp[kn];
        for (std::size_t j = 0; j < M; j++) r[j] = 0.0;
        for (std::size_t j = 0; j < M2; j++) r2[j] = 0.0;
        {
          const double a0 = two_pi * s3[0], a1 = two_pi * s3[1], a2 = two_pi * s3[2], am = two_pi * m;
          const double *u0 = Disp, *u1 = Disp + K, *u2 = Disp + 2 * K;
#pragma omp simd
          for (std::size_t k = 0; k < K; k++) ph[k] = ((a0 * u0[k] + a1 * u1[k]) + a2 * u2[k]) + am * X4[k];
        }
        sincos_array(&ph[0], &es[0], &ec[0], K);
        if (Uw) {
#pragma omp simd
          for (std::size_t k = 0; k < K; k++)
            dwf[k] = d6[0] * Uw[k] + d6[1] * Uw[K + k] + d6[2] * Uw[2 * K + k] + d6[3] * Uw[3 * K + k]
                     + d6[4] * Uw[4 * K + k] + d6[5] * Uw[5 * K + k];
          exp_array(&dwf[0], K);
#pragma omp simd
          for (std::size_t k = 0; k < K; k++) { ec[k] *= dwf[k]; es[k] *= dwf[k]; }
        }
        node_sums(&ec[0], &es[0], W, K, M, &r[0]);
        if (M2) {
          // e times the x4 derivative of the phase (imaginary) and of the Debye-Waller exponent (real)
          for (std::size_t k = 0; k < K; k++) {
            double rate_im = two_pi * ((s3[0] * Du[k] + s3[1] * Du[K + k] + s3[2] * Du[2 * K + k]) + m), rate_re = 0;
            if (Duw)
              for (int i = 0; i < 6; i++) rate_re += d6[i] * Duw[i * K + k];
            dc[k] = ec[k] * rate_re - es[k] * rate_im;
            ds[k] = ec[k] * rate_im + es[k] * rate_re;
          }
          node_sums(&dc[0], &ds[0], W2, K, M2, &r2[0]);
        }
        cd bp_ = bplus[c], bm_ = bminus[c];
        Fo[n] += bp_ * r[0] + bm_ * std::conj(r[0]);
        if (!grad) continue;
        cd *gn = g + n * G;
        const double *h = Hp + 3 * kn;
        for (std::size_t s = 0; s < S; s++) {
          const long long *sp_ = Spec + 4 * s;
          cd x = sp_[0] ? r2[sp_[1]] : r[sp_[1]];
          long long kind = sp_[2], d = sp_[3];
          cd ev = bp_ * x + bm_ * std::conj(x), od = bp_ * x - bm_ * std::conj(x);
          cd cc = Sc[s];
          switch (kind) {
            case 0: gn[d] += cc * ev; break;
            case 1: for (int i = 0; i < 3; i++) gn[d + i] += cc * od * h[i]; break;
            case 2: for (int i = 0; i < 3; i++) gn[d + i] += cc * od * s3[i]; break;
            case 3: for (int i = 0; i < 6; i++) gn[d + i] += cc * ev * d6[i]; break;
            case 4: gn[d] += cc * ev * Ssq[n]; break;
          }
        }
      }
    }
    Py_END_ALLOW_THREADS
    delete gb;
  }

}}} // smtbx::modulation::<anonymous>

BOOST_PYTHON_MODULE(smtbx_modulation_ext)
{
  boost::python::def("atom_sums", smtbx::modulation::atom_sums);
}
