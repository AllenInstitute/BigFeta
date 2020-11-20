import numpy as np


class AlignerTransformException(Exception):
    """Exception class for AlignerTransforms"""
    pass


def aff_matrix(theta, offs=None):
    """affine matrix or augmented affine matrix
    given a rotation angle.

    Parameters
    ----------
    theta : float
        rotation angle in radians
    offs : :class:`numpy.ndarray`
        the translations to include

    Returns
    -------
    M : :class:`numpy.ndarray`
        2 x 2 (for offs=None) affine matrix
        or 3 x 3 augmented matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    if offs is None:
        return R
    M = np.eye(3)
    M[0:2, 0:2] = R
    M[0, 2] = offs[0]
    M[1, 2] = offs[1]
    return M


def aff_matrices(thetas, offs=None):
    """affine matrices from thetas
    """
    c, s = np.cos(thetas), np.sin(thetas)
    matrices = np.zeros((len(thetas), 3, 3))

    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    if offs is None:
        return matrices[:, :-1, :-1]
    matrices[:, :, -1] = np.insert(offs, offs.shape[1], 1, 1)
    return matrices
