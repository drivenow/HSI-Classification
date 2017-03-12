import inspect
from numpy import ones, zeros, where, argmin, unique, array
from numpy import logical_and, logical_or, arange, sqrt
from numpy import maximum, pi, log
from numpy.random import choice
from numpy.linalg import norm, det
from scipy.stats import binom, uniform
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean
from scipy.sparse.linalg import LinearOperator, cg
from random import randint
from numpy.random import randint as nrandint
from multiprocessing import Pool, Array
import ctypes
from numpy.ctypeslib import as_array
from sklearn.neighbors import KNeighborsClassifier as KClass
import pickle
import sys


_sharedX = None
_sharedX2 = None


def para_func(arg):
    num, shape, metric, cnum = arg
    X = _sharedX
    centers = choice(X.shape[0], cnum, False)
    mod = KClass(1, metric=metric)
    mod.fit(X[centers, :], range(centers.size))
    dist, m = mod.kneighbors(X, return_distance=True)
    return m


def para_func2(arg):
    num, shape, shape2, metric, cnum = arg
    X = _sharedX
    X2 = _sharedX2
    centers = choice(X.shape[0], cnum, False)
    mod = KClass(1, metric=metric)
    mod.fit(X[centers, :], range(centers.size))
    dista1, ma1 = mod.kneighbors(X, return_distance=True)
    distb1, mb1 = mod.kneighbors(X2, return_distance=True)

    mall = ma1
    mall2 = mb1

    return mall2, mall


def initShared(X):
    global _sharedX
    _sharedX = X


def initShared2(X, X2):
    global _sharedX
    global _sharedX2
    _sharedX = X
    _sharedX2 = X2


def load_model(model_folder):
    model = FastKernel(None, None, None)
    with open(model_folder + "/model.cfg", 'r') as f:
        model.X = load()
    return model

class FastKernel:
    def __init__(self, X, y, m=200, h=8, distance='euclidean', sigma=0.01, eps=0.05, num_proc=8):
        self.cnum = 3*X.shape[0]//4
        self.d = distance
        self.X = X
        self.y = y
        self.num_proc = num_proc
        self.v = None
        self.m = m
        self.h = h
        self.sigma = sigma
        self.eps = eps
        self.cs = None
        self.selected = False
        # the number of centers for each m
        if len(X.shape) == 1:
            yt = 1
        else:
            x, yt = X.shape
        if yt is None:
            yt = 1

    def _select_centers(self, X):
        if self.selected:
            return
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        self.selected = True

    def K(self, X):
        # the cluster class assigned to each example use
        self._select_centers(X)
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        c = zeros((X.shape[0], self.m))
        share_base = Array(ctypes.c_double, X.shape[0]*X.shape[1], lock=False)
        share = as_array(share_base)
        share = share.reshape(X.shape)
        share[:, :] = X

        if self.cs is None:
            pool = Pool(self.num_proc, maxtasksperchild=50, initializer=initShared, initargs=[share])
            cs = pool.imap(para_func, ((i, X.shape, self.d, self.cnum) for i in xrange(self.m)), 10)
            self.cs = list(cs)
            pool.close()
            pool.join()
        for i, cv in enumerate(self.cs):
            c[:, i] = cv.flatten()
        return c

    def K2y(self, X, X2, y):
        res = zeros(X.shape[0])
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        if len(X2.shape) == 1:
            X2 = X2.reshape((X2.shape[0], 1))
        share_base = Array(ctypes.c_double, X.shape[0]*X.shape[1], lock=False)
        share = as_array(share_base)
        share = share.reshape(X.shape)
        share[:, :] = X

        share2_base = Array(ctypes.c_double, X2.shape[0]*X2.shape[1], lock=False)
        share2 = as_array(share2_base)
        share2 = share2.reshape(X2.shape)
        share2[:, :] = X2

        pool = Pool(self.num_proc, maxtasksperchild=50, initializer=initShared2, initargs=[share2, share])
        cs = pool.imap(para_func2, ((i, X2.shape, X.shape, self.d, self.cnum) for i in xrange(self.m)), 10)
        for c, c2 in cs:
            for cls in unique(c):
                if cls > -1:
                    res[c.flatten() == cls] += y[c2.flatten() == cls].sum()
        res /= self.m
        pool.close()
        pool.join()
        return res

    def K2(self, X, X2):
        #if X.ndim == 0:
        #    X = X.reshape((1, 1))
        #if X2.ndim == 0:
        #    X2 = X2.reshape((1, 1))
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
        if X2.ndim == 1:
            X2 = X2.reshape((X2.shape[0], 1))
        if X.ndim == 0:
            Xsh = 1
            Xsh2 = 1
        else:
            Xsh = X.shape[0]
            Xsh2 = X.shape[1]
        if X2.ndim == 0:
            X2sh = 1
            X2sh2 = 1
        else:
            X2sh = X2.shape[0]
            X2sh2 = X2.shape[1]
        res = zeros((Xsh, X2sh))
        share_base = Array(ctypes.c_double, Xsh*Xsh2, lock=False)
        share = as_array(share_base)
        share = share.reshape((Xsh, Xsh2))
        share[:, :] = X

        share2_base = Array(ctypes.c_double, X2sh*X2sh2, lock=False)
        share2 = as_array(share2_base)
        share2 = share2.reshape(X2.shape)
        share2[:, :] = X2
        pool = Pool(self.num_proc, maxtasksperchild=50, initializer=initShared2, initargs=[share2, share])
        cs = pool.imap(para_func2, ((i, X2.shape, X.shape, self.d, self.cnum) for i in xrange(self.m)), 10)
        for c, c2 in cs:
            for i, c_v in enumerate(c):
                for j, c_v2 in enumerate(c2):
                    if c_v == c_v2 and c_v != -1:
                        res[i, j] += 1.
        res /= self.m
        pool.close()
        pool.join()
        if X.ndim == 0:
            res = res.flatten()
        return res

    def Ky(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        res = zeros(X.shape[0])
        c = self.K(X)
        a = 1.0
        #a = 0.95
        for i in range(self.m):
            for j in unique(c[:, i]):
                if j < 0:
                    continue
                ind = where(c[:, i] == j)[0]
                for k in ind:
                    res[k] += (1.-a)*y[k] + a*y[ind].sum()
            if (c[:, i] == -1).any():
                res[c[:, i] == -1] += y[c[:, i] == -1] # JOE remove if not doing semi
        res /= float(self.m)
        return res

    def B(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        res = zeros(X.shape[0])
        c = self.K(X)
        for i in range(self.m):
            for j in unique(c[:, i]):
                ind = c[:, i] == j
                if j < 0:
                    res[ind] += (1./(1. + self.sigma))*y[ind]
                    continue
                res[ind] += (1./(float(where(ind)[0].size) + self.sigma))*y[ind].sum()
        res /= self.m
        res = (1./self.sigma)*y - res
        return res

    def train(self, X, y):
        if self.v is None:
            A = LinearOperator((X.shape[0], X.shape[0]), lambda x: self.Ky(X, x) + self.sigma*x)
            M = LinearOperator((X.shape[0], X.shape[0]), lambda x: self.B(X, x))
            self.v, info = cg(A, y, M=M, maxiter=40, tol=self.eps, callback=resid_callback)

    def predict_mean(self, X2, X, y):
        self.train(X, y)
        self.cs = None
        res = self.K2y(X2, X, self.v)
        return res

    def predict_var(self, X2, X, y):
        vs = zeros(X2.shape[0])
        for i in range(X2.shape[0]):
            self.cs = None
            # v = self.K2(X2[i, :], X2[i, :])
            v = 1. # by definition of partition kernel K(x, x) = 1
            A = LinearOperator((X.shape[0], X.shape[0]), lambda x: self.Ky(X, x) + self.sigma*x)
            M = LinearOperator((X.shape[0], X.shape[0]), lambda x: self.B(X, x))
            self.cs = None
            if X2.ndim == 1:
                k_star = self.K2(X2[i], X)
            else:
                k_star = self.K2(X2[i, :], X)
            tmp, info = cg(A, k_star.T, M=M, maxiter=40, tol=self.eps)
            vs[i] = v - k_star.dot(tmp)
        return vs

    def likelihood(self, X, y):
        self.train(X, y)
        A = self.K2(X, X)
        res = -.5*y.dot(self.v)-y.shape[0]*log(2.*pi)-.5*log(det(A))
        return res


def resid_callback(xk):
    res = inspect.currentframe().f_back.f_locals['resid']
    with open('residuals.dat', 'a') as f:
        f.write('%s\n' % res)
