#include "kernel.hpp"
#include "mwu_main.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

#include <iostream>
#include <iterator>
#include <iomanip>

#ifndef VERBOSE
#define VERBOSE(n) if(verbose >= n) std::cout
#endif

#ifndef VERBOSE_ITER
#define VERBOSE_ITER(n,b,e,t,s) if(verbose >= n)                    \
    std::copy((b), (e), std::ostream_iterator<t>(std::cout, (s)))
#endif

using namespace mwumkl::kernel;

namespace {

  struct primal_var 
  {
    primal_var(int m_, int n_) 
      : ea(0.0), m(m_), n(n_), l11_(m_,0.0), l12_(m_,0.0)
    {}

    void normalize(int verbose = 0)
    {
      // trace is sum(l11) + sum(l22) + m(n-1)e^a
      double trace = std::accumulate(l11_.begin(), l11_.end(), 0.0, std::plus<double>());
      trace *= 2; // l22 is the same as l11
      trace += m*(n-1)*ea;
      for( unsigned int i = 0; i < l11_.size(); ++i ) {
        l11_[i] /= trace;
        l12_[i] /= trace;
      }
      ea /= trace;
    }

    const std::vector<double> & l11() const { return l11_; }
    const std::vector<double> & l12() const { return l12_; }
    const std::vector<double> & l22() const { return l11_; }

    void update(int i, double l11, double l12, double) {
      l11_[i] = l11;
      l12_[i] = l12;
    }

    double ea;
    const int m, n;
  private:
    std::vector<double> l11_, l12_;
  };

  bool oracle(std::pair<int, int> & alhat_idx, // OUTPUT
              const std::vector<double> & g,   // INPUT
              const int * const y,             // INPUT
              const int verbose)               // INPUT
  {
    double pmax = -std::numeric_limits<double>::infinity();
    double nmax = -std::numeric_limits<double>::infinity();

    int n = g.size();

    int pidx = -1, nidx = -1;

    for( int j = 0; j < n; ++j ) {
      if (y[j] > 0) {
        if (g[j] > pmax) {
          pidx = j;
          pmax = g[j];
        }
      }
      else {
        if (g[j] > nmax) {
          nidx = j;
          nmax = g[j];
        }
      }
    }

    alhat_idx.first = pidx;
    alhat_idx.second = nidx;

    // pmax + nmax = 2*g'alpha, so this equiv. to g'alpha >= -1
    if (pmax + nmax < -2) {
      VERBOSE(0) << "Oracle failed: g'alpha = " << (pmax + nmax)/2 << "\n" 
                 << std::flush;
      return false;
    }

    return true;
  }

  void exponentiateM(primal_var & L,                     // OUTPUT
                     const std::vector<double> & alGal,  // INPUT
                     const std::vector<double> & Galpha, // INPUT
                     const int n,                        // INPUT
                     const int m,                        // INPUT
                     const double rho,                   // INPUT
                     const double epsp,                  // INPUT
                     const int t,                        // INPUT
                     const double cutoff,                // INPUT
                     const int verbose = 0)              // INPUT
  {
    std::vector<double> normu(m);
    std::transform(alGal.begin(), alGal.end(), normu.begin(), 
                   (double (*)(double)) &std::sqrt);

    // const double coeff = -epsp/(2*rho); // always negative
    // const double acoeff = std::abs(coeff);
    const double acoeff = epsp/(2*rho);

    std::vector<double> ps(m,0.0);
    std::transform(normu.begin(), normu.end(), ps.begin(), 
        [&](double nmu){ return acoeff * nmu; });

    // For large x, sinh(x) and cosh(x) are essentially exp(x) 
    // -- this will also overflow for large x, so quash it
    double quash = *std::max_element(ps.begin(), ps.end());
    if (quash > cutoff) {
      for( int i = 0; i < m; ++i ) {
        double epsquash = std::exp(ps[i] - quash)/2;
        L.update(i, epsquash, -epsquash, epsquash);
      }
      L.ea = std::exp(-quash); // probably insignificant
    }
    else {
      // Factoring out exp(ph), as it gets factored out by the normalize anyway
      for( int i = 0; i < m; ++i ) {
        double expp = exp(ps[i])/2;
        double expn = exp(-ps[i])/2;
        // cosh, -sinh, cosh
        L.update(i, expp + expn, expn - expp, expp + expn);
      }
      L.ea = 1;
    }

    L.normalize(verbose);
  }

  bool try_solve(double * alpha,  // OUTPUT: Support vector
                 primal_var & L,  // OUTPUT: primal variable
                 std::vector<double> & alGal,
                 std::vector<double> & Galpha,
                 std::vector<double> & g,
                 //                  OUTPUT: auxiliary variables
                 const std::vector<Kernel *> & K,
                 //                  INPUT:  Kernels as kernel objects
                 const std::vector<double> & r,
                 //                  INPUT:  Kernel traces
                 int * y,         // INPUT:  Labels, +/-1
                 double c,        // INPUT:  Desired output trace
                 double eps,      // INPUT:  Epsilon parameter
                 double ratio,    // INPUT:  Iteration multiplier
                 double cutoff,   // INPUT:  Exponentiation cutoff
                 double C,        // INPUT:  Soft margin parameter
                 double norm1or2, // INPUT:  Is the soft margin 1-norm (1) or 2-norm(2) 
                 //                          or is it a hard margin (0)
                 int verbose = 0  // INPUT:  Be noisy or not (boolean)
                 )
  {
    int m = K.size();
    int n = K[0]->ntr;

    double rho = sqrt(c)/2;
    double eps0 = eps/(2*rho); // eps/sqrt(c);
    double eps0sq = eps*eps/c;
    double epsp = -log1p(-eps0);

    const int tau = std::ceil(2*ratio*std::log(n)/eps0sq);

    std::vector<double> Kij1(n, 0.0);
    std::vector<double> Kij2(n, 0.0);

    for( int t = 0; t < tau; ++t) {
      std::pair<int, int> alhat_idx(-1,-1);
      if (!oracle(alhat_idx, g, y, verbose)) {
        alpha[0] = t/tau;
        alpha[1] = t;
        VERBOSE(0) << "t/tau: " << t << "/" << tau << "\n" << std::flush;
        return false;
      }

      int j1 = alhat_idx.first;
      int j2 = alhat_idx.second;
      alpha[j1] += 0.5; 
      alpha[j2] += 0.5;

      for( int i = 0; i < m; ++i ) {
        K[i]->gram_column(Kij1.begin(), j1);
        K[i]->gram_column(Kij2.begin(), j2);

        alGal[i] += Galpha[i*n + j1] * 0.5;
        alGal[i] += Galpha[i*n + j2] * 0.5;

        for( int k = 0; k < n; ++k ) {
          double ijk = 0.5 * (Kij1[k] * y[j1] + Kij2[k] * y[j2]);
          Galpha[i*n + k] +=  ijk * (c / r[i]) * y[k];
        }
        if (norm1or2 == 2) {
          Galpha[i*n + j1] += 0.5 / C;
          Galpha[i*n + j2] += 0.5 / C;
        }

        alGal[i] += Galpha[i*n + j1] * 0.5;
        alGal[i] += Galpha[i*n + j2] * 0.5;
      }

      exponentiateM(L, alGal, Galpha, n, m, rho, epsp, t, cutoff, verbose);

      g.assign(n, 0.0);
      for( int i = 0; i < m; ++i ) {
        for( int j = 0; j < n; ++j ) {
          g[j] += Galpha[i*n + j] * 2*L.l12()[i]/sqrt(alGal[i]);
        }
      }
    }
    
    auto divide_tau = [&](double x){ return x/tau; };
    // std::binder2nd< std::divides<double> > 
    //   divide_tau( std::divides<double>(), tau );
    
    std::transform(alpha, alpha+n, alpha, divide_tau);

    std::transform(Galpha.begin(), Galpha.end(), Galpha.begin(),
                   divide_tau);

    std::transform(alGal.begin(), alGal.end(), alGal.begin(),
                   divide_tau);
    std::transform(alGal.begin(), alGal.end(), alGal.begin(),
                   divide_tau);

    VERBOSE(2) << "alGal[i]:";
    VERBOSE_ITER(2, alGal.begin(), alGal.end(), double, " ");
    VERBOSE(2) << "\n" << std::flush;

    return true;
  }
}

/*
  void run_mwu_cpp_dynamic(...);

  |-------------------+----------+---------------------------------------|
  | OUTPUT Parameters |          |                                       |
  |                   |          |                                       |
  | Sigma             | double * | The kernel weights                    |
  |                   |          |                                       |
  | alpha             | double * | Support vector                        |
  |                   |          |                                       |
  | bsvm              | double * | Bias                                  |
  |                   |          |                                       |
  | posw              | int *    | Support indicators                    |
  |-------------------+----------+---------------------------------------|
  | INPUT Parameters  |          |                                       |
  |                   |          |                                       |
  | kerns             | int *    | Kernel types, as IDs                  |
  |                   |          |                                       |
  | kern_params       | double * | Parameters for kernels (one each)     |
  |                   |          |                                       |
  | feature_sel       | int *    | Features that kernel will use         |
  |                   |          | 1-d array, -1 means use all features  |
  |                   |          |                                       |
  | Xdata             | double * | Data matrix as 1-d array, by columns  |
  |                   |          | (columns are features)                |
  |                   |          |                                       |
  | y                 | int *    | Labels, +/-1                          |
  |                   |          |                                       |
  | d                 | int      | Number of total features              |
  |                   |          |                                       |
  | n                 | int      | Number of data points                 |
  |                   |          |                                       |
  | m                 | int      | Number of kernels                     |
  |                   |          |                                       |
  | eps               | double   | Epsilon parameter                     |
  |                   |          |                                       |
  | ratio             | double   | Iteration multiplier                  |
  |                   |          |                                       |
  | cutoff            | double   | Exponentiation cutoff                 |
  |                   |          |                                       |
  | C                 | double   | Margin parameter                      |
  |                   |          |                                       |
  | norm1or2          | int      | Is the soft margin 1-norm (1) or      |
  |                   |          | 2-norm (2) or is it a hard margin (0) |
  |                   |          |                                       |
  | verbose           | int      | Be noisy or not (boolean)             |
  |-------------------+----------+---------------------------------------|
*/

void run_mwu_cpp_dynamic(// OUTPUT
                         int * success,
                         double * Sigma,  
                         double * alpha,  
                         double * bsvm,   
                         int * posw,

                         // INPUT
                         int * kerns,
                         double * kern_params,
                         int * feature_sel,
                         double * Xdata,
                         int * y,   
                         int d,
                         int n,           
                         int m,           
                         double eps,      
                         double ratio,    
                         double cutoff,   
                         double C,        
                         int norm1or2,    
                         int verbose      
                         )
{
  std::vector<Kernel *> K(m, (Kernel *)NULL);
  Kernel::MakeKernels(K.begin(), 
                      kerns, kern_params, feature_sel, 
                      Xdata, (double *) NULL, m, n, 0, d);

  // trace of each kernel
  std::vector<double> r(m,0);

  for( int i = 0; i < m; ++i ) {
    r[i] = K[i]->gram_trace();
  }
  VERBOSE(2) << "Traces:";
  VERBOSE_ITER(2, r.begin(), r.end(), double, " ");
  VERBOSE(2) << "\n" << std::flush;

  primal_var L(m, n);
  double c = 1;

  std::vector<double> alGal(m,0.0);
  std::vector<double> Galpha(n*m,0.0);
  std::vector<double> g(n,0.0);
  
  bool success_b = try_solve(alpha, L, alGal, Galpha, g,
                             K, r, y, c, 
                             eps, ratio, cutoff, 
                             C, norm1or2, verbose);
  *success = success_b ? 1 : 0;

  // Clean up
  for( int i = 0; i < m; ++i ) {
    delete K[i];
    K[i] = NULL;
  }

  if (!success_b) {
    return;
  }

  // compute posw
  int nsupp = std::count_if(alpha, alpha+n, 
    [](double a){return a!=0.0;});
  // int psupp = 0, nsupp = 0;
  // for( int j = 0; j < n; ++j ) {
  //   if (alpha[j] != 0.0) {
  //     posw[j] = 1;
  //     if (y[j] > 0) { ++psupp; }
  //     if (y[j] < 0) { ++nsupp; }
  //   }
  // }

  std::vector<double> mu(m,0.0);
  double mu_sum = 0.0;
  // TODO Need to look at these to find constraints that are tight, and only keep those
  // Compute mu
  VERBOSE(1) << "Kernels (phase 1):";
  for (int i = 0; i < m; ++i) {
    VERBOSE(1) << "+";
    mu[i] = -2*L.l12()[i]/std::sqrt(alGal[i]); // the l12 are negative
    mu_sum += mu[i];
  }
  VERBOSE(1) << "\n" << std::flush;
  // mu = mu/mu_sum
  std::transform(mu.begin(), mu.end(), mu.begin(), 
                 [&](double x){ return x/mu_sum; });
  //                std::bind2nd(std::divides<double>(), mu_sum));

  // Recompute g
  g.assign(n,0.0);
  for( int i = 0; i < m; ++i ) {
    if (mu[i] == 0.0) continue;
    for( int j = 0; j < n; ++j ) {
      g[j] += mu[i]*Galpha[i*n + j];
    }
  }

  double pavg = 0.0, navg = 0.0;
  for (int j = 0; j < n; ++j) {
    if (alpha[j] == 0.0) continue;
    if (y[j] > 0) pavg += alpha[j]*g[j];
    if (y[j] < 0) navg += alpha[j]*g[j];
  }

  // compute final alpha and bsvm
  double scale = pavg + navg; // 1/|omega|
  *bsvm = (navg - pavg)/(pavg + navg);
  for (int j = 0; j < n; ++j) {
    alpha[j] /= scale;
  }

  // Give an idea of how mu looks
  double mu_ent = 0.0;
  int mu_cnt = 0;
  for (int i = 0; i < m; ++i) {
    if (mu[i] == 0.0) continue;
    ++mu_cnt;
    mu_ent -= mu[i] * std::log(mu[i]);
  }

  // compute Sigma
  std::transform(mu.begin(),mu.end(),r.begin(),Sigma, 
    [c](double mu, double r){ return c*mu/r; });
  // for (int i = 0; i < m; ++i) {
  //   Sigma[i] = c*mu[i]/r[i];
  // }

  VERBOSE(2) << "mu:";
  VERBOSE_ITER(2, mu.begin(), mu.end(), double, " ");
  VERBOSE(2) << "\n" << std::flush;

  VERBOSE(2) << "Sigma:";
  VERBOSE_ITER(2, Sigma, Sigma+m, double, " ");
  VERBOSE(2) << "\n" << std::flush;

  VERBOSE(1) << std::setw(10) << "supp" << " | "
             << std::setw(10) << "bsvm" << " | "
             << std::setw(10) << "|omega|" << " | "
             << std::setw(10) << "H(mu)" << " | "
             << std::setw(10) << "H(mu_sel)" << "\n";

  VERBOSE(1) << std::setw(8) << 100*nsupp/double(n) << " % | "
             << std::setw(10) << *bsvm << " | "
             << std::setw(10) << 1/scale << " | "
             << std::setw(10) << mu_ent/std::log(m) << " | "
             << std::setw(10) << mu_ent/std::log(mu_cnt) << "\n" 
             << std::flush;

  return;
}

/*
  void test_mwu_cpp_dynamic(...);

  |-------------------+----------+------------------------------------------|
  | OUTPUT Parameters |          |                                          |
  |                   |          |                                          |
  | results           | int *    | Predicted labels, +/-1                   |
  |-------------------+----------+------------------------------------------|
  | INPUT Parameters  |          |                                          |
  |                   |          |                                          |
  | Sigma             | double * | The kernel weights                       |
  |                   |          |                                          |
  | alpha             | double * | Support vector                           |
  |                   |          | NB: Non-support coordinates should have  |
  |                   |          | been removed                             |
  |                   |          |                                          |
  | kerns             | int *    | Kernel types, as IDs                     |
  |                   |          |                                          |
  | kern_params       | double * | Parameters for kernels (one each)        |
  |                   |          |                                          |
  | feature_sel       | int *    | Features that kernel will use            |
  |                   |          | 1-d array, -1 means use all features     |
  |                   |          |                                          |
  | Xtr               | double * | Training matrix as 1-d array, by columns |
  |                   |          | (columns are features)                   |
  |                   |          | NB: Non-support points should have been  |
  |                   |          | removed                                  |
  |                   |          |                                          |
  | Xte               | double * | Test matrix as 1-d array, by columns     |
  |                   |          | (columns are features)                   |
  |                   |          |                                          |
  | ytr               | int *    | Training labels, +/-1                    |
  |                   |          | NB: Non-support points should have been  |
  |                   |          | removed                                  |
  |                   |          |                                          |
  | yte               | int *    | Test labels, +/-1                        |
  |                   |          |                                          |
  | d                 | int      | Number of total features                 |
  |                   |          |                                          |
  | ntr               | int      | Number of training points                |
  |                   |          | NB: This should only count support       |
  |                   |          | points                                   |
  |                   |          |                                          |
  | nte               | int      | Number of test points                    |
  |                   |          |                                          |
  | m                 | int      | Number of kernels                        |
  |                   |          |                                          |
  | verbose           | int      | Be noisy or not (boolean)                |
  |-------------------+----------+------------------------------------------|
*/

void test_mwu_cpp_dynamic(// OUTPUT
                          double * results,

                          // INPUT
                          double * Sigma,  
                          double * alpha,  
                          int * kerns,
                          double * kern_params,
                          int * feature_sel,
                          double * Xtr,
                          double * Xte,
                          int * ytr,
                          int d,
                          int ntr,           
                          int nte,           
                          int m,           
                          int verbose
                          )
{
  std::vector<Kernel *> K(m, (Kernel *)NULL);
  Kernel::MakeKernels(K.begin(), 
                      kerns, kern_params, feature_sel, 
                      Xtr, Xte, 
                      m, ntr, nte, d);

  std::vector<double> results_int(nte, 0.0);
  std::fill(results, results+nte, 0.0);

  for (int i = 0; i < m; ++i) {
    if (Sigma[i] == 0.0) continue;
    K[i]->predict(results_int.begin(), alpha, ytr);
    for (int t = 0; t < nte; ++t) {
      results[t] += Sigma[i] * results_int[t];
    }
    // Clean up
    delete K[i];
  }
}

