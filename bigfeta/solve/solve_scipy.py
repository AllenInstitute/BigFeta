"""solves using the python scientific stack"""
import time

import numpy as np
from scipy.sparse.linalg import factorized


def solve(A, weights, reg, x0, rhs):
    """regularized, weighted linear least squares solve

    Parameters
    ----------
    A : :class:`scipy.sparse.csr_matrix`
        the matrix, N (equations) x M (degrees of freedom)
    weights : :class:`scipy.sparse.csr_matrix`
        N x N diagonal matrix containing weights
    reg : :class:`scipy.sparse.csr_matrix`
        M x M diagonal matrix containing regularizations
    x0 : :class:`numpy.ndarray`
        M x nsolve float constraint values for the DOFs
    rhs : :class:`numpy.ndarray`
        rhs vector(s)
        N x nsolve float right-hand-side(s)

    Returns
    -------
    results : dict
       includes solution "x" and summary metrics
    """
    time0 = time.time()
    # regularized least squares
    # ensure symmetry of K
    weights.data = np.sqrt(weights.data)
    atwt = A.transpose().dot(weights.transpose())
    wa = weights.dot(A)
    K = atwt.dot(wa) + reg

    # save this for calculating error
    w = weights.diagonal() != 0

    del wa

    # factorize, then solve, efficient for large affine
    x = np.zeros_like(x0)
    err = np.zeros((A.shape[0], x.shape[1]))
    precision = [0] * x.shape[1]

    solve = factorized(K)
    for i in range(x0.shape[1]):
        # can solve for same A, but multiple x0's
        Lm = reg.dot(x0[:, i]) + atwt.dot(weights.dot(rhs[:, i]))
        x[:, i] = solve(Lm)
        err[:, i] = A.dot(x[:, i]) - rhs[:, i]
        precision[i] = \
            np.linalg.norm(K.dot(x[:, i]) - Lm) / np.linalg.norm(Lm)
        del Lm
    del K, A, atwt

    # only report errors where weights != 0
    err = err[w, :]

    results = {}
    results['precision'] = precision
    results['error'] = np.linalg.norm(err, axis=0).tolist()
    mag = np.linalg.norm(err, axis=1)
    results['mag'] = [mag.mean(), mag.std()]
    results['err'] = [
            [m, e] for m, e in
            zip(err.mean(axis=0), err.std(axis=0))]
    results['x'] = x
    results['time'] = time.time() - time0
    return results


__all__ = ["solve"]
