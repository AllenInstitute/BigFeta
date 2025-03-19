from .affine_model import AlignerAffineModel
from .polynomial_model import AlignerPolynomial2DTransform
from .rotation_model import AlignerRotationModel
from .similarity_model import AlignerSimilarityModel
from .thinplatespline_model import AlignerThinPlateSplineTransform
from .translation_model import AlignerTranslationModel
from .utils import AlignerTransformException

__all__ = ['AlignerTransform']


class AlignerTransform(object):
    """general transform object that the solver expects
    """

    def __init__(self, name=None, transform=None, fullsize=False, order=2):
        """
        Parameters
        ----------

        name : str
            specifies the intended transform for the type of solve
        transform : :class:`renderapi.transform.Transform`
            The new AlignerTransform will
            inherit from this transform, if possible.
        fullsize : bool
            only applies to affine transform. Remains for legacy reason as an
            explicit demonstration of the equivalence of fullsize and halfsize
            transforms.
        order : int
            used in Polynomial2DTransform

        """
        if (name is None):
            raise AlignerTransformException(
                   'must specify transform name')

        # backwards compatibility
        if name == 'affine':
            name = 'AffineModel'
        if name == 'affine_fullsize':
            name = 'AffineModel'
            fullsize = True
        if name == 'rigid':
            name = 'SimilarityModel'

        # renderapi-consistent names
        if (name == 'AffineModel'):
            self.__class__ = AlignerAffineModel
            AlignerAffineModel.__init__(
                    self, transform=transform, fullsize=fullsize)
        elif (name == 'SimilarityModel'):
            self.__class__ = AlignerSimilarityModel
            AlignerSimilarityModel.__init__(self, transform=transform)
        elif (name == 'RotationModel'):
            self.__class__ = AlignerRotationModel
            AlignerRotationModel.__init__(self, transform=transform)
        elif (name == 'TranslationModel'):
            self.__class__ = AlignerTranslationModel
            AlignerTranslationModel.__init__(self, transform=transform)
        elif (name == 'Polynomial2DTransform'):
            self.__class__ = AlignerPolynomial2DTransform
            AlignerPolynomial2DTransform.__init__(
                    self, transform=transform,
                    order=order)
        elif (name == 'ThinPlateSplineTransform'):
            self.__class__ = AlignerThinPlateSplineTransform
            AlignerThinPlateSplineTransform.__init__(
                    self, transform=transform)
        else:
            raise AlignerTransformException(
                    'transform %s not in possible choices:' % name)
