import numpy as np
import renderapi
from scipy.sparse import csr_matrix

from .utils import AlignerTransformException

__all__ = ['AlignerSimilarityModel']


class AlignerSimilarityModel(renderapi.transform.AffineModel):
    """
    Object for implementing similarity transform.
    """

    def __init__(self, transform=None):
        """
        Parameters
        ----------

        transform : :class:`renderapi.transform.Transform`
            The new AlignerTransform will
            inherit from this transform, if possible.
        """

        if transform is not None:
            if isinstance(transform, renderapi.transform.AffineModel):
                super(AlignerSimilarityModel, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            super(AlignerSimilarityModel, self).__init__()

        self.DOF_per_tile = 4
        self.rows_per_ptmatch = 4

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : :class:`numpy.ndarray`
            N x 1 transform parameters in solve form
        """
        vec = np.array([
            self.M[0, 0],
            self.M[0, 1],
            self.M[0, 2],
            self.M[1, 2]])
        vec = vec.reshape((vec.size, 1))
        return vec

    def from_solve_vec(self, vec):
        """reads values from solution and sets transform parameters

        Parameters
        ----------
        vec : :class:`numpy.ndarray`
            input to this function is sliced so that vec[0] is the
            first harvested value for this transform

        Returns
        -------
        n : int
            number of rows read from vec. Used to increment vec slice
            for next transform
        """
        vec = vec.reshape(vec.shape[0])
        self.M[0, 0] = vec[0]
        self.M[0, 1] = vec[1]
        self.M[0, 2] = vec[2]
        self.M[1, 0] = -vec[1]
        self.M[1, 1] = vec[0]
        self.M[1, 2] = vec[3]
        n = 4
        return n

    def regularization(self, regdict):
        """regularization vector

        Parameters
        ----------
        regdict : dict
           bigfeta.schemas.regularization. controls
           regularization values

        Return
        ------
        reg : :class:`numpy.ndarray`
            array of regularization values of length DOF_per_tile
        """
        reg = np.ones(self.DOF_per_tile).astype('float64') * \
            regdict['default_lambda']
        reg[2::4] *= regdict['translation_factor']
        reg[3::4] *= regdict['translation_factor']
        return reg

    def block_from_pts(self, pts, w, col_ind, col_max):
        """partial sparse block for a transform/match.
           similarity constrains the center-of-mass coordinates
           to transform according to the same affine transform
           as the coordinates, save translation.

        Parameters
        ----------
        pts :  :class:`numpy.ndarray`
            N x 2, the x, y values of the match (either p or q)
        w : :class:`numpy.ndarray`
            the weights associated with the pts
        col_ind : int
            the starting column index for this tile
        col_max : int
            number of columns in the matrix

        Returns
        -------
        block : :class:`scipy.sparse.csr_matrix`
            the partial block for this transform
        w : :class:`numpy.ndarray`
            the weights associated with the rows of this block
        rhs : :class:`numpy.ndarray`
            N x 1 (fullsize)
            right hand side for this transform.
            generally all zeros. could implement fixed tiles in
            rhs later.
        """
        px = pts[:, 0]
        py = pts[:, 1]
        npts = px.size
        pxm = px - px.mean()
        pym = py - py.mean()
        ones = np.ones_like(px)

        data = np.concatenate((
            np.vstack((px, py, ones)).transpose().flatten(),
            np.vstack((-px, py, ones)).transpose().flatten(),
            np.vstack((pxm, pym)).transpose().flatten(),
            np.vstack((-pxm, pym)).transpose().flatten()))

        indices = np.concatenate((
            np.tile([0, 1, 2], npts),
            np.tile([1, 0, 3], npts),
            np.tile([0, 1], npts),
            np.tile([1, 0], npts))) + col_ind

        i = np.concatenate(([3] * npts, [3] * npts, [2] * npts, [2] * npts))
        indptr = np.concatenate(([0], np.cumsum(i)))

        block = csr_matrix((data, indices, indptr), shape=(npts * 4, col_max))
        rhs = np.zeros((npts * 4, 1))
        return block, np.hstack((w, w, w, w)), rhs
