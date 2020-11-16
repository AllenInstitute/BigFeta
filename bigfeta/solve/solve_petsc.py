"""solves using petsc4py"""
import time

import numpy as np

try:
    import petsc4py.PETSc
    HAS_PETSC = True
except ImportError:
    HAS_PETSC = False


def petsc_mtx_from_scipycsr(m):
    """construct a petsc matrix object from a scipy CSR sparse martrix

    Parameters
    ----------
    m : :class:`scipy.sparse.csr_matrix`
        a matrix to represent in petsc

    Returns
    -------
    M : :class:`petsc4py.PETSc.Mat`
      petsc4py representation of matrix m
    """
    return petsc4py.PETSc.Mat().createAIJ(
        size=m.shape, csr=(
            m.indptr, m.indices, m.data))


def petsc_factorize(A):
    """return petsc ksp for solving sparse linear system,
    with A pre-factorized

    Parameters
    ----------
    A : :class:`petsc4py.PETSc.Mat`
        input matrix

    Returns
    -------
    ksp : :class:`petsc.PETSc.KSP`
        KSP for solving
    """
    pc = petsc4py.PETSc.PC().create()
    pc.setType("lu")
    # pc.setFactorSolverType("pastix")

    ksp = petsc4py.PETSc.KSP().create()
    ksp.setPC(pc)
    ksp.setType("preonly")

    ksp.setFromOptions()
    ksp.setOperators(A)

    return ksp


def petsc_factorize_scipycsr(A):
    """get petsc4py ksp from a scipy sparse CSR

    Parameters
    ----------
    A : :class:`scipy.sparse.csr_matrix`
        input matrix to factorize

    Returns
    -------
    ksp : :class:`petsc.PETSc.KSP`
        KSP for solving
    """
    A_petsc = petsc_mtx_from_scipycsr(A)
    return petsc_factorize(A_petsc)


def petsc_solve_to_numpy(s, b, x):
    """solve system with numpy array b and x using petsc4py ksp

    Parameters
    ----------
    s : :class:`petsc.PETSc.KSP`
        KSP for solving
    b : :class:`numpy.ndarray`
        b for Ax=b solve
    x : :class:`numpy.ndarray`
        x for Ax=b solve

    Returns
    -------
    x : :class:`numpy.ndarray`
        numpy representation of solution

    """
    b_petsc = petsc4py.PETSc.Vec().createWithArray(b)
    x_petsc = petsc4py.PETSc.Vec().createWithArray(x)
    s.solve(b_petsc, x_petsc)
    return x_petsc.getArray()


def solve_petsc_factorization(A, weights, reg, x0, rhs):
    """regularized, weighted linear least squares solve
    using petsc factorization

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
    ksp = petsc_factorize_scipycsr(K)

    for i in range(x0.shape[1]):
        # can solve for same A, but multiple x0's
        Lm = reg.dot(x0[:, i]) + atwt.dot(weights.dot(rhs[:, i]))
        x[:, i] = petsc_solve_to_numpy(ksp, Lm, x[:, i])
        err[:, i] = A.dot(x[:, i]) - rhs[:, i]
        precision[i] = \
            np.linalg.norm(K.dot(x[:, i]) - Lm) / np.linalg.norm(Lm)
        del Lm
    del K, A, atwt, ksp

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


solve = solve_petsc_factorization

__all__ = ["solve", "HAS_PETSC"]
