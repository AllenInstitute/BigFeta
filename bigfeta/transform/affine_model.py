import numpy as np
import renderapi
from scipy.sparse import csr_matrix

from .utils import AlignerTransformException

__all__ = ['AlignerAffineModel']


class AlignerAffineModel(renderapi.transform.AffineModel):
    """
    Object for implementing full or half-size affine transforms
    """

    def __init__(self, transform=None, fullsize=False):
        """
        Parameters
        ----------

        transform : :class:`renderapi.transform.Transform`
            The new AlignerTransform will
            inherit from this transform, if possible.
        fullsize : bool
            only applies to affine transform. Remains for legacy reason as an
            explicit demonstration of the equivalence of fullsize and halfsize
            transforms.
        """
        self.fullsize = fullsize

        if transform is not None:
            if isinstance(transform, renderapi.transform.AffineModel):
                super(AlignerAffineModel, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            super(AlignerAffineModel, self).__init__()

        self.DOF_per_tile = 3
        self.rows_per_ptmatch = 1
        if self.fullsize:
            self.DOF_per_tile = 6
            self.rows_per_ptmatch = 2

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : :class:`numpy.ndarray`
            N/2 x 2 for halfsize, N x 2 for fullsize
        """

        vec = np.array([
            self.M[0, 0],
            self.M[0, 1],
            self.M[0, 2],
            self.M[1, 0],
            self.M[1, 1],
            self.M[1, 2]])
        if not self.fullsize:
            # split in half for half-size solve
            # transpose into Nx2
            vec = np.transpose(vec.reshape((2, int(vec.size/2))))
        else:
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
        vsh = vec.shape
        if vsh[1] == 1:
            # fullsize
            vec = vec.reshape(vec.shape[0])
            self.M[0, 0] = vec[0]
            self.M[0, 1] = vec[1]
            self.M[0, 2] = vec[2]
            self.M[1, 0] = vec[3]
            self.M[1, 1] = vec[4]
            self.M[1, 2] = vec[5]
            n = 6
        else:
            # halfsize
            self.M[0, 0] = vec[0, 0]
            self.M[0, 1] = vec[1, 0]
            self.M[0, 2] = vec[2, 0]
            self.M[1, 0] = vec[0, 1]
            self.M[1, 1] = vec[1, 1]
            self.M[1, 2] = vec[2, 1]
            n = 3
        return n

    def regularization(self, regdict):
        """regularization vector from this transform

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
        reg[2::3] *= regdict['translation_factor']

        return reg

    def block_from_pts(self, pts, w, col_ind, col_max):
        """partial sparse block for a transform/match

        Parameters
        ----------
        pts :  :class:`numpy.ndarray`
            N x 2, the x, y values of the match (either p or q)
        w : :class:`numpy.ndarray`
            size N, the weights associated with the pts
        col_ind : int
            the starting column index for this tile
        col_max : int
            total number of columns in the matrix

        Returns
        -------
        block : :class:`scipy.sparse.csr_matrix`
            the partial block for this transform
        w : :class:`numpy.ndarray`
            the weights associated with the rows of this block
        rhs : :class:`numpy.ndarray`
            N/2 x 2 (halfsize) or N x 1 (fullsize)
            right hand side for this transform.
            generally all zeros. could implement fixed tiles in
            rhs later.
        """
        data = np.hstack((pts, np.ones((pts.shape[0], 1)))).flatten()
        i0 = col_ind + np.arange(3)
        nrow = pts.shape[0]
        indices = np.tile(i0, nrow)
        rhs = np.zeros((nrow, 2))
        if self.fullsize:
            nrow *= 2
            data = np.concatenate((data, data))
            indices = np.hstack((indices, indices + 3))
            rhs = np.zeros((nrow, 1))
            w = np.hstack((w, w))

        indptr = np.arange(0, nrow + 1) * 3
        block = csr_matrix((data, indices, indptr), shape=(nrow, col_max))
        return block, w, rhs
