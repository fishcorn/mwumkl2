%module mwumkl

%{
#define SWIG_FILE_WITH_INIT
#include "../mwu_main.h"
%}

%include "typemaps.i"
%include "numpy.i"

%init %{
import_array();
%}

%apply double * OUTPUT { double *_bsvm_out };

%apply int * OUTPUT { int *_success_out };

%apply (double * ARGOUT_ARRAY1, int DIM1) { 
  (double * _Sigma_out, int dim_Sigma_out),
    (double * _alpha_out, int dim_alpha_out), 
    (double * _results_out, int dim_results_out) }

%apply (int * ARGOUT_ARRAY1, int DIM1) {
  (int * _posw_out, int dim_posw_out) }

%apply (double * INPLACE_ARRAY1, int DIM1) {
  (double * _Sigma, int dim_Sigma),
    (double * _alpha, int dim_alpha),
    (double * _kern_params, int dim_kern_params) }

%apply (double * INPLACE_ARRAY2, int DIM1, int DIM2) {
  (double * _Xtr, int dim_Xtr_1, int dim_Xtr_2), 
    (double * _Xte, int dim_Xte_1, int dim_Xte_2) }


%apply (int * INPLACE_ARRAY1, int DIM1) {
    (int * _kerns, int dim_kerns),
    (int * _feature_sel, int dim_feature_sel),
    (int * _ytr, int dim_ytr) }

%rename (_train_mwu_mkl) run_wrapper;
%rename (_test_mkl) test_wrapper;

%inline%{
  void run_wrapper(int * _success_out,
                   double * _Sigma_out, int dim_Sigma_out,
                   double * _alpha_out, int dim_alpha_out,
                   double * _bsvm_out,
                   int * _posw_out, int dim_posw_out,
                   int * _kerns, int dim_kerns,
                   double * _kern_params, int dim_kern_params,
                   int * _feature_sel, int dim_feature_sel,
                   double * _Xtr, int dim_Xtr_1, int dim_Xtr_2,
                   int * _ytr, int dim_ytr,
                   int d,
                   int ntr,
                   int m,
                   double eps,      
                   double ratio,    
                   double cutoff,   
                   double C,        
                   int norm1or2,    
                   int verbose
                   ) 
  {
    if (m != dim_Sigma_out || 
        m != dim_kerns || 
        m != dim_kern_params || 
        m != dim_feature_sel) {
      PyErr_Format(PyExc_ValueError, 
                   "Mismatch with # of kernels: "
                   "m=%d, |Sigma|=%d, |kerns|=%d, "
                   "|params|=%d, |features|=%d", 
                   m, dim_Sigma_out, dim_kerns, 
                   dim_kern_params, dim_feature_sel);
      return;
    }

    if (d != dim_Xtr_1) {
      PyErr_Format(PyExc_ValueError, 
                   "Mismatch with # of dimensions: "
                   "d=%d, |Xtr_1|=%d", 
                   d, dim_Xtr_1);
      return;
    }
    
    if (ntr != dim_alpha_out || 
        ntr != dim_posw_out || 
        ntr != dim_Xtr_2 || 
        ntr != dim_ytr) {
      PyErr_Format(PyExc_ValueError, 
                   "Mismatch with # of training points: "
                   "ntr=%d, |alpha|=%d, |posw|=%d, "
                   "|Xtr_2|=%d, |ytr|=%d", 
                   ntr, dim_alpha_out, dim_posw_out, 
                   dim_Xtr_2, dim_ytr);
      return;
    }
    
    run_mwu_cpp_dynamic(_success_out, _Sigma_out, _alpha_out, 
                        _bsvm_out, _posw_out,
                        _kerns, _kern_params, _feature_sel, 
                        _Xtr, _ytr, 
                        d, ntr, m, 
                        eps, ratio, cutoff, C, norm1or2,
                        verbose);
  }

  void test_wrapper(double * _results_out, int dim_results_out,
                    double * _Sigma, int dim_Sigma,
                    double * _alpha, int dim_alpha,
                    int * _kerns, int dim_kerns,
                    double * _kern_params, int dim_kern_params,
                    int * _feature_sel, int dim_feature_sel,
                    double * _Xtr, int dim_Xtr_1, int dim_Xtr_2,
                    double * _Xte, int dim_Xte_1, int dim_Xte_2,
                    int * _ytr, int dim_ytr,
                    int d,
                    int ntr,           
                    int nte,           
                    int m,           
                    int verbose
                    ) 
  {
    if (m != dim_Sigma || 
        m != dim_kerns || 
        m != dim_kern_params || 
        m != dim_feature_sel) {
      PyErr_Format(PyExc_ValueError, 
                   "Mismatch with # of kernels: "
                   "m=%d, |Sigma|=%d, |kerns|=%d, "
                   "|params|=%d, |features|=%d", 
                   m, dim_Sigma, dim_kerns, 
                   dim_kern_params, dim_feature_sel);
      return;
    }

    if (d != dim_Xtr_1 || d != dim_Xte_1) {
      PyErr_Format(PyExc_ValueError, 
                   "Mismatch with # of dimensions: "
                   "d=%d, |Xtr_1|=%d, |Xte_1|=%d", 
                   d, dim_Xtr_1, dim_Xte_1);
      return;
    }
    
    if (ntr != dim_alpha || 
        ntr != dim_Xtr_2 || 
        ntr != dim_ytr) {
      PyErr_Format(PyExc_ValueError, 
                   "Mismatch with # of support points: "
                   "ntr=%d, |alpha|=%d, |Xtr_2|=%d, |ytr|=%d", 
                   ntr, dim_alpha, dim_Xtr_2, dim_ytr);
      return;
    }
    
    if (nte != dim_Xte_2 || nte != dim_results_out) {
      PyErr_Format(PyExc_ValueError, 
                   "Mismatch with # of test points: "
                   "ntr=%d, |Xte_2|=%d, |argout|=%d", 
                   ntr, dim_Xtr_2, dim_results_out);
      return;
    }

    test_mwu_cpp_dynamic(_results_out, _Sigma, _alpha, 
                         _kerns, _kern_params, _feature_sel, 
                         _Xtr, _Xte, _ytr, 
                         d, ntr, nte, m, 
                         verbose);
  }
%}
 
%pythoncode %{
    def train_mwu_mkl(kerns, kern_params, features, 
                      Xtr, ytr,
                      eps = 0.2, C = 1000.0, norm1or2 = 2, 
                      verbose = 0):
        """
        (success, Sigma, alpha, bsvm, posw) = 
        train_mwu_mkl(
          kerns, kern_params, features, 
          Xtr, ytr, 
          eps = 0.2, C = 1000.0, verbose = 0
        )
        """
        (m,) = kerns.shape
        (d,ntr) = Xtr.shape
        return _train_mwu_mkl(m, ntr, ntr, 
                              kerns, kern_params, features, 
                              Xtr, ytr, d, ntr, m, 
                              eps, 1.0, 20.0, C, norm1or2, 
                              verbose)

    def test_mkl(Sigma, alpha, 
                 kerns, kern_params, feature_sel, 
                 Xtr, Xte, ytr, 
                 verbose = 0):
        """
        (results) = test_mkl(
                      Sigma, alpha, 
                      kerns, kern_params, feature_sel, 
                      Xtr, Xte, ytr, 
                      verbose = 0)
        """
        (m,) = kerns.shape
        (d, ntr) = Xtr.shape
        (d, nte) = Xte.shape
        return _test_mkl(nte, 
                         Sigma, alpha, 
                         kerns, kern_params, feature_sel, 
                         Xtr, Xte, ytr, 
                         d, ntr, nte, m, 
                         verbose)
%}
