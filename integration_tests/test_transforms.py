import pytest
import renderapi
from EMaligner.transform.transform import AlignerTransform
from EMaligner.transform.affine_model import AlignerAffineModel
from EMaligner.transform.similarity_model import AlignerSimilarityModel
from EMaligner.transform.rotation_model import AlignerRotationModel
from EMaligner.transform.polynomial_model import AlignerPolynomial2DTransform
from EMaligner.transform.utils import AlignerTransformException
import numpy as np


def test_aliases():
    t = AlignerTransform(name='affine')
    assert(t.__class__ == AlignerAffineModel)
    assert(not t.fullsize)

    t = AlignerTransform(name='affine_fullsize')
    assert(t.__class__ == AlignerAffineModel)
    assert(t.fullsize)

    t = AlignerTransform(name='rigid')
    assert(t.__class__ == AlignerSimilarityModel)


def test_transform():
    # must specify something
    with pytest.raises(AlignerTransformException):
        t = AlignerTransform()

    # two ways to load affine
    t = AlignerTransform(name='AffineModel')
    assert(t.__class__ == AlignerAffineModel)
    del t
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(name='AffineModel', transform=rt)
    assert(t.__class__ == AlignerAffineModel)

    # two ways to load similarity
    t = AlignerTransform(name='SimilarityModel')
    assert(t.__class__ == AlignerSimilarityModel)
    del t
    rt = renderapi.transform.SimilarityModel()
    t = AlignerTransform(name='SimilarityModel', transform=rt)
    assert(t.__class__ == AlignerSimilarityModel)

    # two ways to load rotation
    t = AlignerTransform(name='RotationModel')
    assert(t.__class__ == AlignerRotationModel)
    del t
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(name='RotationModel', transform=rt)
    assert(t.__class__ == AlignerRotationModel)

    # two ways to load polynomial
    t = AlignerTransform(name='Polynomial2DTransform')
    assert(t.__class__ == AlignerPolynomial2DTransform)
    del t
    rt = renderapi.transform.Polynomial2DTransform(identity=True)
    t = AlignerTransform(name='Polynomial2DTransform', transform=rt)
    assert(t.__class__ == AlignerPolynomial2DTransform)

    # specifying something not real
    with pytest.raises(AlignerTransformException):
        t = AlignerTransform(name='LudicrousModel')


def example_match(npts):
    match = {}
    match['matches'] = {
            "w": list(np.ones(npts)),
            "p": [list(np.random.randn(npts)), list(np.random.randn(npts))],
            "q": [list(np.random.randn(npts)), list(np.random.randn(npts))]
            }
    return match


def test_affine_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        t = AlignerAffineModel(transform=rt)

    # check args
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(name='AffineModel', transform=rt)
    assert(t.__class__ == AlignerAffineModel)
    assert(not t.fullsize)
    t = AlignerTransform(name='AffineModel', transform=rt, fullsize=True)
    assert(t.__class__ == AlignerAffineModel)
    assert(t.fullsize)

    # make block (fullsize)
    t = AlignerTransform(name='AffineModel', transform=rt, fullsize=True)
    nmatch = 100
    match = example_match(nmatch)
    ncol = 1000
    icol = 73
    block, weights, rhs = t.block_from_pts(
            np.array(match['matches']['p']).transpose(),
            np.array(match['matches']['w']),
            icol,
            ncol)

    assert np.all(np.isclose(rhs, 0.0))
    assert block.check_format() is None
    assert weights.size == nmatch * t.rows_per_ptmatch
    assert block.shape == (nmatch * t.rows_per_ptmatch, ncol)
    assert block.nnz == 2 * nmatch * 3

    # make CSR (halfsize)
    t = AlignerTransform(name='AffineModel', transform=rt, fullsize=False)
    nmatch = 100
    match = example_match(nmatch)
    ncol = 1000
    icol = 73
    block, weights, rhs = t.block_from_pts(
            np.array(match['matches']['p']).transpose(),
            np.array(match['matches']['w']),
            icol,
            ncol)
    assert np.all(np.isclose(rhs, 0.0))
    assert block.check_format() is None
    assert weights.size == nmatch * t.rows_per_ptmatch
    assert block.shape == (nmatch * t.rows_per_ptmatch, ncol)
    assert block.nnz == nmatch * 3

    # to vec
    t = AlignerTransform(name='AffineModel', transform=rt, fullsize=True)
    v = t.to_solve_vec()
    assert np.all(v == np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(6, 1))
    t.fullsize = False
    v = t.to_solve_vec()
    assert np.all(v == np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

    # from vec
    t.fullsize = True
    ntiles = 6
    vi = [1.0, 0.2, 0.0, -0.1, 1.0, 0.0]
    vec = np.tile(vi, ntiles)
    vec = vec.reshape(-1, 1)
    index = 0
    for i in range(ntiles):
        index += t.from_solve_vec(vec[index:, :])
        assert np.all(np.isclose(t.M[0:2, :].flatten(), vi))

    t.fullsize = False
    vi = np.array([[1.0, 0.2, 0.0], [-0.1, 1.0, 0.0]]).transpose()
    vec = np.tile(vi, reps=[ntiles, 1])
    index = 0
    for i in range(ntiles):
        index += t.from_solve_vec(vec[index:, :])
        assert np.all(np.isclose(t.M[0:2, :], vi.transpose()))

    # reg
    rdict = {
            "default_lambda": 1.0,
            "translation_factor": 0.1}
    t = AlignerTransform(name='AffineModel', transform=rt, fullsize=True)
    r = t.regularization(rdict)
    assert np.all(r[[0, 1, 3, 4]] == 1.0)
    assert np.all(r[[2, 5]] == 0.1)
    t = AlignerTransform(name='AffineModel', transform=rt, fullsize=False)
    r = t.regularization(rdict)
    assert np.all(r[[0, 1]] == 1.0)
    assert np.all(r[[2]] == 0.1)


def test_similarity_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        t = AlignerSimilarityModel(transform=rt)

    # check args
    rt = renderapi.transform.SimilarityModel()
    t = AlignerTransform(name='SimilarityModel', transform=rt)
    assert(t.__class__ == AlignerSimilarityModel)

    # make block
    t = AlignerTransform(name='SimilarityModel', transform=rt, fullsize=True)
    nmatch = 100
    match = example_match(nmatch)
    ncol = 1000
    icol = 73
    block, weights, rhs = t.block_from_pts(
            np.array(match['matches']['p']).transpose(),
            np.array(match['matches']['w']),
            icol,
            ncol)

    assert np.all(np.isclose(rhs, 0.0))
    assert block.check_format() is None
    assert weights.size == nmatch * t.rows_per_ptmatch
    assert block.shape == (nmatch * t.rows_per_ptmatch, ncol)
    assert block.nnz == 10 * nmatch

    # to vec
    t = AlignerTransform(name='SimilarityModel')
    v = t.to_solve_vec()
    assert np.all(v == np.array([1.0, 0.0, 0.0, 0.0]).reshape(-1, 1))

    # from vec
    ntiles = 6
    vi = [1.0, 0.02, -10.0, 12.1]
    vec = np.tile(vi, ntiles)
    vec = vec.reshape(-1, 1)
    index = 0
    for i in range(ntiles):
        index += t.from_solve_vec(vec[index:, :])
        msub = t.M.flatten()[[0, 1, 2, 5]]
        assert np.all(np.isclose(msub, vi))

    # reg
    rdict = {
            "default_lambda": 1.0,
            "translation_factor": 0.1}
    t = AlignerTransform(name='SimilarityModel')
    r = t.regularization(rdict)
    assert np.all(r[[0, 1]] == 1.0)
    assert np.all(r[[2, 3]] == 0.1)


def test_polynomial_model():
    # check args
    for o in range(4):
        t = AlignerTransform(name="Polynomial2DTransform", order=o)
        assert(t.__class__ == AlignerPolynomial2DTransform)
        assert(t.order == o)

    rt = renderapi.transform.AffineModel()
    for o in range(4):
        t = AlignerTransform(
                name="Polynomial2DTransform", order=o, transform=rt)
        assert(t.__class__ == AlignerPolynomial2DTransform)
        assert(t.order == o)

    # make block
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        params = np.zeros((2, n))
        rt = renderapi.transform.Polynomial2DTransform(params=params)
        t = AlignerTransform(name='Polynomial2DTransform', transform=rt)

        nmatch = 100
        match = example_match(nmatch)
        ncol = 1000
        icol = 73
        block, weights, rhs = t.block_from_pts(
                np.array(match['matches']['p']).transpose(),
                np.array(match['matches']['w']),
                icol,
                ncol)

        assert np.all(np.isclose(rhs, 0.0))
        assert block.check_format() is None
        assert weights.size == nmatch
        assert block.shape == (nmatch, ncol)
        assert block.nnz == n * nmatch

    # to vec
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        params = np.random.randn(2, n)
        rt = renderapi.transform.Polynomial2DTransform(params=params)
        t = AlignerTransform(name='Polynomial2DTransform', transform=rt)
        v = t.to_solve_vec()
        assert np.all(np.isclose(v, np.transpose(params)))

    # from vec
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        v0 = np.random.randn(n, 2)
        rt0 = renderapi.transform.Polynomial2DTransform(
                params=np.zeros((2, n)))
        t = AlignerTransform(name='Polynomial2DTransform', transform=rt0)
        assert t.order == order
        vec = np.concatenate((v0, v0, v0, v0))
        index = 0
        for i in range(4):
            index += t.from_solve_vec(vec[index:, :])
            assert np.all(np.isclose(t.params.transpose(), v0))

    # reg
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        vec = np.zeros((n, 2))
        rt0 = renderapi.transform.Polynomial2DTransform(params=vec)
        t = AlignerTransform(name='Polynomial2DTransform', transform=rt0)

        rdict = {
                "default_lambda": 1.0,
                "translation_factor": 0.1,
                "poly_factors": None}
        r = t.regularization(rdict)
        assert np.isclose(r[0], 0.1)
        assert np.all(np.isclose(r[1:], 1.0))

    # reg
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        vec = np.zeros((n, 2))
        rt0 = renderapi.transform.Polynomial2DTransform(params=vec)
        t = AlignerTransform(name='Polynomial2DTransform', transform=rt0)

        pf = np.random.randn(order + 1)
        rdict = {
                "default_lambda": 1.0,
                "translation_factor": 0.1,
                "poly_factors": pf.tolist()}
        r = t.regularization(rdict)
        ni = 0
        for i in range(order + 1):
            for j in range(i + 1):
                assert np.all(r[ni::n] == pf[i])
                ni += 1


def test_rotation_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        t = AlignerRotationModel(transform=rt)

    # check args
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(name='RotationModel', transform=rt)
    assert(t.__class__ == AlignerRotationModel)

    # make block
    t = AlignerTransform(name='RotationModel', transform=rt, fullsize=True)
    nmatch = 100
    match = example_match(nmatch)
    ncol = 1000
    icol = 73

    # scale up because rotation filters out things near center-of-mass
    ppts, qpts, w = AlignerRotationModel.preprocess(
            np.array(match['matches']['p']).transpose() * 1000,
            np.array(match['matches']['q']).transpose() * 1000,
            np.array(match['matches']['w']))

    assert ppts.shape == qpts.shape == (nmatch, 1)

    block, weights, rhs = t.block_from_pts(
            ppts,
            w,
            icol,
            ncol)

    assert rhs.shape == (nmatch, 1)
    assert block.check_format() is None
    assert weights.size == nmatch * t.rows_per_ptmatch
    assert block.shape == (nmatch * t.rows_per_ptmatch, ncol)
    assert block.nnz == 1 * nmatch

    # to vec
    t = AlignerTransform(name='RotationModel')
    v = t.to_solve_vec()
    assert np.all(v == np.array([0.0]).reshape(-1, 1))

    # from vec
    ntiles = 6
    vec = np.random.randn(ntiles)
    vec = vec.reshape(-1, 1)
    index = 0
    for i in range(ntiles):
        t = AlignerTransform(name='RotationModel')
        index += t.from_solve_vec(vec[index:, :])
        msub = t.rotation
        assert np.isclose(msub, vec[i][0])

    # reg
    rdict = {
            "default_lambda": 1.2345,
            "translation_factor": 0.1}
    t = AlignerTransform(name='RotationModel')
    r = t.regularization(rdict)
    assert np.all(np.isclose(r, 1.2345))
