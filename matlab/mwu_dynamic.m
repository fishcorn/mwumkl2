function [ model ] = mwu_dynamic( X, y, options )
%MWU_DYNAMIC Run MWU-MKL algorithm with "dynamic" (i.e. lazy) kernels
%   X:          Data, with features as columns, points as rows n x d
%   y:          Labels, +/- 1, n x 1
%   options:    Algorithm options

    model = {};
    libname = 'libmwu';
    if ~libisloaded(libname)
        libfile = [libname '.so'];
        hdrfile = 'mwu_main.h';  
        
        loadlibrary(libfile, hdrfile);
        assert(libisloaded(libname));
    end
    
    [ n, d ] = size(X);

    options.kernel = kern_parse(options.kernel, d);
    
    if options.loss.C > 1e6
      options.loss.normtype = 0;
    end
    
    % void run_mwu_cpp_dynamic(...
    %     int * success, double * Sigma, double * alpha, double * bsvm, int * posw,...
    %     int * kerns, double * kern_params, int * feature_sel,...
    %     double * Xdata, int * y, int d, int n, int m,...
    %     double eps, double ratio, double cutoff,...
    %     double C, int norm1or2, int verbose)

    % [int32Ptr, doublePtr, doublePtr, doublePtr, int32Ptr] run_mwu_cpp_dynamic(
    %     int32Ptr, doublePtr, doublePtr, doublePtr, int32Ptr, ...
    %     int32Ptr, doublePtr, int32Ptr, ...
    %     doublePtr, int32Ptr, int32, int32, int32, ...
    %     double, double, double, ...
    %     double, int32, int32)

    % The indices have to be changed to 0-based
    features_zero = [options.kernel.cache(:).feature] - 1;
    m = length(options.kernel.cache);
    param = [options.kernel.cache(:).param1];
    ids = [options.kernel.cache(:).id];
    
    tic
    [ success Sigma alpha bsvm posw ] ...
        = calllib(libname, 'run_mwu_cpp_dynamic', ...
        0, zeros(m,1), zeros(n,1), 0, zeros(n,1), ...
        ids(:), param(:), features_zero(:), ...
        X(:), y, d, n, m, ...
        options.mwu.epsilon, options.mwu.iter_mult, options.mwu.cutoff, ...
        options.loss.C, options.loss.normtype, options.verbose);
    time = toc;

    if ~success
        fprintf('MWU-MKL Algorithm failed.\n');
        return;
    end
    
    model.method = 'mwu-mkl';
    model.Sigma = Sigma;
    model.nKern = nnz(Sigma);
    model.bias = bsvm;
    model.supp = nnz(posw)/n;
    posw = (posw == 1);
    model.alpha = alpha(posw);
    model.posw = find(posw);
    model.time = time;
    model.options = options;
end
