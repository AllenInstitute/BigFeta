transforms
==========

The EMaligner solver uses AlignerTransform objects to construct linear least squares elements from tilespecs. The final transform in each tilespec transform list is converted into an AlignerTransform object, which has a renderapi.transform.transform object as its base class. The AlignerTransform object has a few other methods to enable constructing the matrix, setting regularizations, extracting the starting transform values and setting the solved transform values.

.. autoclass:: EMaligner.transform.transform.AlignerTransform
    :members: __init__
    :undoc-members:
    :show-inheritance:

.. autoclass:: EMaligner.transform.affine_model.AlignerAffineModel
    :members: __init__, to_solve_vec, from_solve_vec, regularization,
      block_from_pts
    :undoc-members:
    :show-inheritance:

.. autoclass:: EMaligner.transform.polynomial_model.AlignerPolynomial2DTransform
    :members: __init__, to_solve_vec, from_solve_vec, regularization,
      block_from_pts
    :undoc-members:
    :show-inheritance:

.. autoclass:: EMaligner.transform.rotation_model.AlignerRotationModel
    :members: __init__, to_solve_vec, from_solve_vec, regularization,
      block_from_pts, preprocess
    :undoc-members:
    :show-inheritance:

.. autoclass:: EMaligner.transform.similarity_model.AlignerSimilarityModel
    :members: __init__, to_solve_vec, from_solve_vec, regularization,
      block_from_pts
    :undoc-members:
    :show-inheritance:

.. autoclass:: EMaligner.transform.thinplatespline_model.AlignerThinPlateSplineTransform
    :members: __init__, to_solve_vec, from_solve_vec, regularization, scale,
      block_from_pts
    :undoc-members:
    :show-inheritance:

.. autoclass:: EMaligner.transform.translation_model.AlignerTranslationModel
    :members: __init__, to_solve_vec, from_solve_vec, regularization,
      block_from_pts
    :undoc-members:
    :show-inheritance:

.. automodule:: EMaligner.transform.utils
    :members:
    :undoc-members:
    :show-inheritance:
