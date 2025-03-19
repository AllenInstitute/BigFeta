import numpy as np
import renderapi
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from .utils import AlignerTransformException

__all__ = ['AlignerThinPlateSplineTransform']


class AlignerThinPlateSplineTransform(renderapi.transform.ThinPlateSplineTransform):
    """
    Object for implementing thin plate spline transform
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
            if isinstance(transform, renderapi.transform.ThinPlateSplineTransform):
                super(AlignerThinPlateSplineTransform, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            raise AlignerTransformException(
                    "not sure how to iniztialize thin plate spline")

        self.DOF_per_tile = self.nLm
        if self.aMtx is not None:
            self.DOF_per_tile += 3
        self.rows_per_ptmatch = 1

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : :class:`numpy.ndarray`
            N x 2 transform parameters in solve form
        """
        vec = self.dMtxDat.transpose()
        if self.aMtx is not None:
            vec = np.vstack((
                self.bVec,
                self.aMtx.transpose(),
                vec))
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
        n0 = 0
        n1 = self.nLm
        if self.aMtx is not None:
            self.bVec = vec[0, :]
            self.aMtx = vec[1:3, :].transpose()
            n0 += 3
            n1 += 3
        self.dMtxDat = vec[n0:n1, :].transpose()
        return n1

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
        n0 = 0
        if self.aMtx is not None:
            reg[0] *= regdict['translation_factor']
            reg[1:3] *= 1.0
            n0 += 3
        reg[n0:] *= regdict['thinplate_factor']
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
        data = cdist(
                pts,
                self.srcPts.transpose(),
                metric='sqeuclidean')
        data *= np.ma.log(np.sqrt(data)).filled(0.0)
        ncol = data.shape[1]
        if self.aMtx is not None:
            ones = np.ones(pts.shape[0]).reshape(-1, 1)
            data = np.hstack((ones, pts, data))
            ncol += 3
        data = data.flatten()
        indices = np.tile(np.arange(ncol) + col_ind, pts.shape[0])
        indptr = np.arange(0, pts.shape[0] + 1) * ncol

        block = csr_matrix(
                (data, indices, indptr),
                shape=(pts.shape[0], col_max))
        rhs = pts
        return block, w, rhs

    @property
    def scale(self):
        """tuple of scale for x, y.
           For setting regularization, it is useful to watch
           scale (logged output for the solver) to look for 
           unwanted distortions and shrinking. Other transforms
           have scale implemented inside of renderapi.
        """
        src = self.srcPts.transpose()
        dst = self.tform(src)
        a = renderapi.transform.AffineModel()
        a.estimate(src, dst)
        return a.scale
