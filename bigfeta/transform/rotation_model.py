import numpy as np
import renderapi
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix

from .utils import aff_matrix, aff_matrices, AlignerTransformException

__all__ = ['AlignerRotationModel']


class AlignerRotationModel(renderapi.transform.AffineModel):
    """
    Object for implementing rotation transform
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
            if isinstance(
                    transform, renderapi.transform.AffineModel):
                super(AlignerRotationModel, self).__init__(
                        json=transform.to_dict())
            else:
                raise AlignerTransformException(
                        "can't initialize %s with %s" % (
                            self.__class__, transform.__class__))
        else:
            super(AlignerRotationModel, self).__init__()

        self.DOF_per_tile = 1
        self.rows_per_ptmatch = 1

    def to_solve_vec(self):
        """sets solve vector values from transform parameters

        Returns
        -------
        vec : :class:`numpy.ndarray`
            N x 1 transform parameters in solve form
        """

        return np.array([self.rotation]).reshape(-1, 1)

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
        newr = aff_matrix(vec[0][0])
        self.M[0:2, 0:2] = newr.dot(self.M[0:2, 0:2])
        return 1

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
        return reg

    def block_from_pts(self, pts, w, col_ind, col_max):
        """partial sparse block for a transform/match.
           Note: for rotation, a pre-processing step is
           called at the tilepair level.

        Parameters
        ----------
        pts :  :class:`numpy.ndarray`
            N x 1, preprocessed from preprocess()
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
            N x 1
            right hand side for this transform.
        """

        data = np.ones(pts.size)
        indices = np.ones(pts.size) * col_ind
        indptr = np.arange(pts.size + 1)
        rhs = pts.reshape(-1, 1) + self.rotation

        block = csr_matrix((data, indices, indptr), shape=(pts.size, col_max))
        return block, w, rhs

    @staticmethod
    def preprocess(ppts, qpts, w, npts_max=None, choose_random=True,
                   cutoff_distance=15):
        """tilepair-level preprocessing step for rotation transform.
           derives the relative center-of-mass angles between all
           p's and q's to avoid angular discontinuity. Will filter
           out points very close to center-of-mass.
           Tilepairs with relative rotations near 180deg will not avoid
           the discontinuity.

        Parameters
        ----------
        ppts : :class:`numpy.ndarray`
            N x 2. The p tile correspondence coordinates
        qpts : :class:`numpy.ndarray`
            N x 2. The q tile correspondence coordinates
        w : :class:`numpy.ndarray`
            size N. The weights.
        npts_max : int
            maximum points to include after processing
        choose_random : bool
            whether to randomly reduce the input points to npts_max
        cutoff_distance : float
            distance from center of mass of each point cloud in which to
            exclude points

        Returns
        -------
        pa : :class:`numpy.ndarray`
            M x 1 preprocessed angular distances. -0.5 x delta angle
            M <= N depending on filter
        qa : :class:`numpy.ndarray`
            M x 1 preprocessed angular distances. 0.5 x delta angle
            M <= N depending on filter
        w : :class:`numpy.ndarray`
            size M. filtered weights.
        """
        # center of mass
        pcm = ppts - ppts.mean(axis=0)
        qcm = qpts - qpts.mean(axis=0)

        # points very close to center of mass are noisy
        rfilter = np.argwhere(
            (np.linalg.norm(pcm, axis=1) > cutoff_distance) &
            (np.linalg.norm(qcm, axis=1) > cutoff_distance)).flatten()
        if npts_max is not None:
            if choose_random:
                rfilter = np.random.choice(
                    rfilter, min(npts_max, rfilter.size), replace=False)
            else:
                rfilter = rfilter[:npts_max]

        pcm = pcm[rfilter]
        qcm = qcm[rfilter]
        w = w[rfilter]

        pangs = np.arctan2(pcm[:, 1], pcm[:, 0])

        # rotate all the q values relative to p
        mtxs = aff_matrices(-1. * pangs)
        qrot = np.einsum("ijk,ik->ij", mtxs, qcm)

        delta_angs = np.arctan2(qrot[:, 1], qrot[:, 0])

        pa = (-0.5 * delta_angs).reshape(-1, 1)
        qa = (0.5 * delta_angs).reshape(-1, 1)
        return pa, qa, w
