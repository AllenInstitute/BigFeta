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
