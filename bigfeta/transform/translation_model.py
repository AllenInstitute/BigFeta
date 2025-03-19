import renderapi
import numpy as np
from scipy.sparse import csr_matrix

from .utils import AlignerTransformException

__all__ = ['AlignerTranslationModel']


class AlignerTranslationModel(renderapi.transform.AffineModel):
    """
    Object for implementing translation transform
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
                super(AlignerTranslationModel, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            super(AlignerTranslationModel, self).__init__()

        self.DOF_per_tile = 1
        self.rows_per_ptmatch = 1

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : :class:`numpy.ndarray`
            1 x 2 transform parameters in solve form
        """
        return np.array([[0.0, 0.0]])

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
        self.M[0, 2] += vec[0, 0]
        self.M[1, 2] += vec[0, 1]
        n = 1
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
            regdict['default_lambda'] * regdict['translation_factor']
        return reg

    def block_from_pts(self, pts, w, col_ind, col_max):
        """partial sparse block for a tilepair/match

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
            N/2 x 2 
            right hand side for this transform.
        """
        data = np.ones(pts.shape[0])
        indices = data * col_ind
        indptr = np.arange(0, pts.shape[0] + 1)

        block = csr_matrix(
                (data, indices, indptr),
                shape=(pts.shape[0], col_max))
        rhs = pts + np.array([self.B0, self.B1])
        return block, w, rhs
