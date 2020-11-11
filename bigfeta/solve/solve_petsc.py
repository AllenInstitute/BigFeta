"""solves using petsc4py"""
import time

import numpy as np

try:
    import petsc4py.PETSc
    HAS_PETSC = True
except ImportError:
    HAS_PETSC = False


def petsc_mtx_from_scipycsr(m):
    return petsc4py.PETSc.Mat().createAIJ(
        size=m.shape, csr=(
            m.indptr, m.indices, m.data))


def petsc_factorize(m):
    pc = petsc4py.PETSc.PC().create()
    pc.setType("lu")
    # pc.setFactorSolverType("pastix")

    ksp = petsc4py.PETSc.KSP().create()
    ksp.setPC(pc)
    ksp.setType("preonly")

    ksp.setFromOptions()
    ksp.setOperators(m)

    return ksp


def petsc_factorize_scipycsr(m):
    m_petsc = petsc_mtx_from_scipycsr(m)
    return petsc_factorize(m_petsc)


def petsc_solve_to_numpy(s, b, x):
    b_petsc = petsc4py.PETSc.Vec().createWithArray(b)
    x_petsc = petsc4py.PETSc.Vec().createWithArray(x)
    s.solve(b_petsc, x_petsc)
    return x_petsc.getArray()


def solve_petsc_factorization(A, weights, reg, x0, rhs):
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
