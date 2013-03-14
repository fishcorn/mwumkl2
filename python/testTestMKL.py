#! /usr/bin/env python

# System imports
from   distutils.util import get_platform
import os
import sys
import unittest

# Import NumPy
import numpy as np
major, minor = [ int(d) for d in np.__version__.split(".")[:2] ]
if major == 0:
    BadListError = TypeError
else:
    BadListError = ValueError

from mwumkl import test_mkl, train_mwu_mkl

######################################################################

class TestMKL1TestCase(unittest.TestCase):

    def setUp(self):
        return

    def testSimpleMargin1(self):
        "Test against two 1-dimensional support points (SIGN), 1 kernel (linear)"
        Sigma = np.double([1.]); # 1 kernel, weight 1.0
        alpha = np.double([1.,1.])/2; # 2 support points of equal weight
        kerns = np.int32([0]); # 1 linear kernel
        params = np.double([0.]); # params not important
        features = np.int32([-1]); # use all features, but there's only 1
        Xtr = np.double([[-2.,2.]]); # support points
        Xte = Xtr/2; # test points
        ytr = np.int32([-1,1]); # labels
        results = test_mkl(Sigma, alpha, kerns, params, features, Xtr, Xte, ytr)
        self.assertTrue((np.sign(results) == ytr).all(), 
                        msg='results={0}, ytr={1}'.format(results, ytr))

    def testSimpleMargin2(self):
        "Test against four 2-dimensional support points (XOR), 1 kernel (quad)"
        Sigma = np.double([1.]); # 1 kernel, weight 1.0
        alpha = np.double([1.,1.,1.,1.])/4; # 4 support points of equal weight
        kerns = np.int32([1]); # 1 polynomial kernel
        params = np.double([2.]); # quadratic
        features = np.int32([-1]); # use all features
        Xtr = np.double([[-2.,2.,-2.,2.],
                         [-2.,-2.,2.,2.]]); # support points
        Xte = Xtr/2; # test points
        ytr = np.int32([-1,1,1,-1]); # labels
        results = test_mkl(Sigma, alpha, kerns, params, features, Xtr, Xte, ytr)
        self.assertTrue((np.sign(results) == ytr).all(), 
                        msg='results={0}, ytr={1}'.format(results, ytr))

######################################################################

class TrainMKL1TestCase(unittest.TestCase):

    def setUp(self):
        return

    def testSimpleTrain1(self):
        "Two 1-dimensional input points (SIGN), 1 kernel (linear)"
        kerns = np.int32([0]); # 1 linear kernel
        params = np.double([0.]); # params not important
        features = np.int32([-1]); # use all features, but there's only 1
        Xtr = np.double([[-2.,2.]]); # support points
        ytr = np.int32([-1,1]); # labels
        (success, Sigma, alpha, bsvm, posw) = train_mwu_mkl(kerns, params, features, Xtr, ytr)
        self.assertTrue(success)
        self.assertTrue((posw == 1).all(), msg='posw={0}'.format(posw))
        self.assertTrue((Sigma > 0).all(), msg='Sigma={0}'.format(Sigma))
        self.assertTrue((alpha >= 0).all(), msg='alpha={0}'.format(alpha))
        self.assertAlmostEqual(bsvm, 0.0, msg='bsvm={0}'.format(bsvm))

    def testSimpleTrain2(self):
        "Four 2-dimensional input points (XOR), 1 kernel (quad)"
        kerns = np.int32([1]); # 1 polynomial kernel
        params = np.double([2.]); # quadratic
        features = np.int32([-1]); # use all features, but there's only 1
        Xtr = np.double([[-2.,2.,-2.,2.],
                         [-2.,-2.,2.,2.]]); # support points
        ytr = np.int32([-1,1,1,-1]); # labels
        (success, Sigma, alpha, bsvm, posw) = train_mwu_mkl(kerns, params, features, Xtr, ytr)
        self.assertTrue(success)
        self.assertTrue((posw == 1).all(), msg='posw={0}'.format(posw))
        self.assertTrue((Sigma > 0).all(), msg='Sigma={0}'.format(Sigma))
        self.assertTrue((alpha >= 0).all(), msg='alpha={0}'.format(alpha))
        self.assertAlmostEqual(bsvm, 0.0, msg='bsvm={0}'.format(bsvm))

######################################################################

if __name__ == "__main__":

    # Build the test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMKL1TestCase))
    suite.addTest(unittest.makeSuite(TrainMKL1TestCase))

    # Execute the test suite
    print "Testing Classes of Module mwumkl"
    print "NumPy version", np.__version__
    print
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(len(result.errors) + len(result.failures))
