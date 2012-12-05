#include "mwu_main.h"

#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>
#include <limits>
#include <typeinfo>

#include <iostream>

namespace mwumkl { namespace kernel {

    struct dot_elem {
      inline double operator()(double r, double c) {
        return r*c;
      }
    };

    struct diff_sq_elem {
      inline double operator()(double r, double c) {
        double out = r - c;
        return out*out;
      }
    };
    
    struct outer_fun {
      typedef double first_argument_type;
      typedef double second_argument_type;
      typedef double result_type;
    };

    struct identity_outer_fun : outer_fun {
      double operator()(double in, double) const {
        return in;
      }
    };

    struct poly_outer_fun : outer_fun  {
      double operator()(double in, double param) const {
        return std::pow(in+1, param);
      }
    };

    struct tanh_outer_fun : outer_fun  {
      double operator()(double in, double param) const {
        return std::tanh(param*in);
      }
    };

    struct logistic_outer_fun : outer_fun  {
      double operator()(double in, double param) const {
        return 1/(std::exp(-param*in) + 1);
      }
    };

    struct gauss_outer_fun : outer_fun  {
      double operator()(double in, double param) const {
        return std::exp(-in/param);
      }
    };

    template < typename AggElem, typename OuterFun = identity_outer_fun >
    struct gram {
      // double entry(int r, int c, const std::vector<double * > & features) const {
      //   double out = 0.0;
      //   int d = features.size();
      //   AggElem agg;
      //   for (int i = 0; i < d; ++i) {
      //     out += agg(features[i][r], features[i][c]);
      //   }
      //   return OuterFun()(out, param);
      // }

      template < typename RandIt >
      void column(RandIt out, int n, int c, const std::vector<const double * > & features) const {
        int d = features.size();
        AggElem e;

        std::fill(out, out+n, 0.0);

        for (int k = 0; k < d; ++k) {
          const double * feature = features[k];
          for (int j = 0; j < n; ++j) {
            out[j] += e(feature[j], feature[c]);
          }
        }
        std::transform(out, out+n, out, std::bind2nd(OuterFun(), param));
      }

      template < typename RandIt >
      void predict(RandIt out,
                   double * alpha, int * ytr,
                   int ntr, int nte, 
                   const std::vector<const double *> & features_tr, 
                   const std::vector<const double *> & features_te) const
      {
        int d = features_te.size();        
        AggElem e;
        OuterFun o;

        std::fill(out, out+nte, 0.0);
        
        std::vector<double> kern(ntr, 0.0);
        for (int t = 0; t < nte; ++t) {
          kern.assign(ntr, 0.0);
          for (int k = 0; k < d; ++k) {
            const double * feat_tr = features_tr[k];
            const double * feat_te = features_te[k];
            for (int j = 0; j < ntr; ++j) {
              kern[j] += e(feat_tr[j], feat_te[t]);
            }
          }
          for (int j = 0; j < ntr; ++j) {
            out[t] += o(kern[j], param)*alpha[j]*ytr[j];
          }
        }
      }

      double trace(int n, const std::vector<const double *> & features) const {
        double trace = 0.0;
        int d = features.size();
        AggElem e;
        OuterFun o;
        for (int j = 0; j < n; ++j) {
          double elem = 0.0;
          for (int i = 0; i < d; ++i) {
            double el = features[i][j];
            elem += e(el, el);
          }
          trace += o(elem, param);
        }
        return trace;
      }

      explicit gram(double param_) : param(param_) {}

      double param;
    };

    class Kernel {

      std::vector<const double *> features_tr;
      std::vector<const double *> features_te;

      virtual void column_(std::vector<double>::iterator out, int c) const = 0;
      virtual double trace_() const = 0;
      virtual void predict_(std::vector<double>::iterator out,
                            double * alpha, int * ytr) const = 0;

      void build_features(std::vector<const double *> & features, int n, const double * X) const {
        if (features.empty()) {
          if (feature < 0) {
            features.assign(-feature, (const double *)NULL);
            for (int i = 0; i < -feature; ++i) {
              features[i] = X + i*n;
            }
          }
          else {
            features.assign(1, X + n*feature);
          }
        }        
      }
      
    protected:
      const std::vector<const double *> & features_tr_() const { return features_tr; }
      const std::vector<const double *> & features_te_() const { return features_te; }

    public:
      Kernel(int ntr_, int nte_, double * Xtr_, double * Xte_,
             int feature_) 
        : ntr(ntr_), nte(nte_), Xtr(Xtr_), Xte(Xte_), feature(feature_)
      {}

      const int ntr, nte;
      const double * const Xtr;
      const double * const Xte;
      const int feature;

      void gram_column(std::vector<double>::iterator out, int c) {
        build_features(features_tr, ntr, Xtr);
        column_(out, c);
      }

      double gram_trace() {
        build_features(features_tr, ntr, Xtr);
        return trace_();
      }

      template < typename RandIt >
      void predict(RandIt out, double * alpha, int * ytr) {
        build_features(features_tr, ntr, Xtr);
        build_features(features_te, nte, Xte);
        predict_(out, alpha, ytr);
      }

      template < typename OutIt > 
      static void MakeKernels(OutIt oi, 
                              int * kerns,
                              double * kern_params,
                              int * feature_sel,
                              double * Xtr,
                              double * Xte,
                              int m,
                              int ntr,
                              int nte,
                              int d);
    };

    template < typename AggFun, typename OuterFun >
    class KernelImp : public Kernel, private gram< AggFun, OuterFun >
    {
      typedef gram< AggFun, OuterFun > gram_base;

      virtual void column_(std::vector<double>::iterator out, int c) const {
        gram_base::column(out, ntr, c, features_tr_());
      }

      virtual double trace_() const {
        return gram_base::trace(ntr, features_tr_());
      }

      virtual void predict_(std::vector<double>::iterator out, 
                            double * alpha, int * ytr) const {
        gram_base::predict(out, alpha, ytr, ntr, nte, 
                           features_tr_(), features_te_());
      }

    public:
      KernelImp(double param_, 
                int ntr_, int nte_, double * Xtr_, double * Xte_, 
                int feature_)
        : Kernel(ntr_, nte_, Xtr_, Xte_, feature_), 
          gram_base(param_)
      {}
    };

    typedef KernelImp< dot_elem, identity_outer_fun > LinearKern;   // LINEAR
    typedef KernelImp< dot_elem, poly_outer_fun > PolyKern;         // POLY
    typedef KernelImp< diff_sq_elem, gauss_outer_fun > RbfKern;     // RBF, GAUSSIAN
    typedef KernelImp< dot_elem, logistic_outer_fun > LogisticKern; // LOGISTIC, SIGMOID
    typedef KernelImp< dot_elem, tanh_outer_fun > TanhKern;         // TANH

    template < typename RandIt >
    void Kernel::MakeKernels(RandIt out, 
                             int * kerns,
                             double * kern_params,
                             int * feature_sel,
                             double * Xtr,
                             double * Xte,
                             int m,
                             int ntr, 
                             int nte, 
                             int d)
    {
      for (int i = 0; i < m; ++i) {
        int feature = feature_sel[i] < 0 ? -d : feature_sel[i];
        switch (kerns[i]) {
        case LINEAR:
          out[i] = new LinearKern(kern_params[i], ntr, nte, Xtr, Xte, feature);
          break;
        case POLY:
          out[i] = new PolyKern(kern_params[i], ntr, nte, Xtr, Xte, feature);
          break;
        case RBF: // case GAUSSIAN:
          out[i] = new RbfKern(kern_params[i], ntr, nte, Xtr, Xte, feature);
          break;
        case LOGISTIC: // case SIGMOID:
          out[i] = new LogisticKern(kern_params[i], ntr, nte, Xtr, Xte, feature);
          break;
        case TANH:
          out[i] = new TanhKern(kern_params[i], ntr, nte, Xtr, Xte, feature);
          break;
        default:
          out[i] = NULL;
        };
      }
    }

  } // namespace kernel
} // namespace mwumkl
