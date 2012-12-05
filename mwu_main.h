#ifndef _mwu_main_h_
#define _mwu_main_h_

#ifdef __cplusplus
extern "C" {
#endif

enum KernelType {
  LINEAR,
  POLY,
  RBF,
  GAUSSIAN = RBF,
  LOGISTIC,
  SIGMOID = LOGISTIC,
  TANH,
  MAX_KERNEL_TYPE_1, /* New kernel types should go after this  */
  /* FANCY_KERNEL = 100, For example */
};

void run_mwu_cpp_dynamic(int * success,
                         double * Sigma,  
                         double * alpha,  
                         double * bsvm,   
                         int * posw,      
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
                         );

/*
  void run_mwu_cpp_dynamic(...);

  |-------------------+--------------+---------------------------------------|
  | OUTPUT Parameters |              |                                       |
  |                   |              |                                       |
  | success           | int *        | 1 for success, 0 for failure          |
  |                   |              |                                       |
  | Sigma             | double *     | The kernel weights                    |
  |                   |              |                                       |
  | alpha             | double *     | Support vector                        |
  |                   |              |                                       |
  | bsvm              | double *     | Bias                                  |
  |                   |              |                                       |
  | posw              | int *        | Support indicators                    |
  |-------------------+--------------+---------------------------------------|
  | INPUT Parameters  |              |                                       |
  |                   |              |                                       |
  | kerns             | KernelType * | Kernel types, as IDs                  |
  |                   |              |                                       |
  | kern_params       | double *     | Parameters for kernels (one each)     |
  |                   |              |                                       |
  | feature_sel       | int *        | Features that kernel will use         |
  |                   |              | 1-d array, -1 means use all features  |
  |                   |              |                                       |
  | Xdata             | double *     | Data matrix as 1-d array, by columns  |
  |                   |              | (columns are features)                |
  |                   |              |                                       |
  | y                 | int *        | Labels, +/-1                          |
  |                   |              |                                       |
  | d                 | int          | Number of total features              |
  |                   |              |                                       |
  | n                 | int          | Number of data points                 |
  |                   |              |                                       |
  | m                 | int          | Number of kernels                     |
  |                   |              |                                       |
  | eps               | double       | Epsilon parameter                     |
  |                   |              |                                       |
  | ratio             | double       | Iteration multiplier                  |
  |                   |              |                                       |
  | cutoff            | double       | Exponentiation cutoff                 |
  |                   |              |                                       |
  | C                 | double       | Margin parameter                      |
  |                   |              |                                       |
  | norm1or2          | int          | Is the soft margin 1-norm (1) or      |
  |                   |              | 2-norm (2) or is it a hard margin (0) |
  |                   |              |                                       |
  | verbose           | int          | Noisiness: 0 = not noisy              |
  |                   |              |            nonzero = noisy level      |
  |-------------------+--------------+---------------------------------------|
*/


void test_mwu_cpp_dynamic(double * results,
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
                          );

/*
  void test_mwu_cpp_dynamic(...);

  |--------------------+--------------+------------------------------------------|
  | OUTPUT Parameters  |              |                                          |
  |                    |              |                                          |
  | results            | double *     | Predicted outputs                        |
  |--------------------+--------------+------------------------------------------|
  | INPUT Parameters   |              |                                          |
  |                    |              |                                          |
  | Sigma              | double *     | The kernel weights                       |
  |                    |              |                                          |
  | alpha              | double *     | Support vector                           |
  |                    |              | NB: Non-support coordinates should have  |
  |                    |              | been removed                             |
  |                    |              |                                          |
  | kerns              | KernelType * | Kernel types, as IDs                     |
  |                    |              |                                          |
  | kern_params        | double *     | Parameters for kernels (one each)        |
  |                    |              |                                          |
  | feature_sel        | int *        | Features that kernel will use            |
  |                    |              | 1-d array, -1 means use all features     |
  |                    |              |                                          |
  | Xtr                | double *     | Training matrix as 1-d array, by columns |
  |                    |              | (columns are features)                   |
  |                    |              | NB: Non-support points should have been  |
  |                    |              | removed                                  |
  |                    |              |                                          |
  | Xte                | double *     | Test matrix as 1-d array, by columns     |
  |                    |              | (columns are features)                   |
  |                    |              |                                          |
  | ytr                | int *        | Training labels, +/-1                    |
  |                    |              | NB: Non-support points should have been  |
  |                    |              | removed                                  |
  |                    |              |                                          |
  | d                  | int          | Number of total features                 |
  |                    |              |                                          |
  | ntr                | int          | Number of training points                |
  |                    |              | NB: This should only count support       |
  |                    |              | points                                   |
  |                    |              |                                          |
  | nte                | int          | Number of test points                    |
  |                    |              |                                          |
  | m                  | int          | Number of kernels                        |
  |                    |              |                                          |
  | verbose            | int          | Be noisy or not (boolean)                |
  |--------------------+--------------+------------------------------------------|
*/


#ifdef __cplusplus
}
#endif

#endif 
