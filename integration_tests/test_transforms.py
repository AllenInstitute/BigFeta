import pytest

import numpy as np
import renderapi

from bigfeta.transform.transform import AlignerTransform
from bigfeta.transform.affine_model import AlignerAffineModel
from bigfeta.transform.similarity_model import AlignerSimilarityModel
from bigfeta.transform.rotation_model import AlignerRotationModel
from bigfeta.transform.translation_model import AlignerTranslationModel
from bigfeta.transform.polynomial_model import AlignerPolynomial2DTransform
from bigfeta.transform.thinplatespline_model import \
        AlignerThinPlateSplineTransform
from bigfeta.transform.utils import AlignerTransformException, aff_matrix

random_seed = None
rng = np.random.default_rng(random_seed)

def test_aff_matrix():
    a = aff_matrix(0.0)
    assert np.all(np.isclose(a, np.eye(2)))
    a = aff_matrix(0.0, offs=[0.0, 0.0])
    assert np.all(np.isclose(a, np.eye(3)))
    a = aff_matrix(0.0, offs=[1.0, 2.0])
    assert np.isclose(a[0, 2], 1.0)
    assert np.isclose(a[1, 2], 2.0)


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
        _ = AlignerTransform()

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

    # two ways to load translation
    t = AlignerTransform(name='TranslationModel')
    assert(t.__class__ == AlignerTranslationModel)
    del t
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(name='TranslationModel', transform=rt)
    assert(t.__class__ == AlignerTranslationModel)

    # two ways to load polynomial
    t = AlignerTransform(name='Polynomial2DTransform')
    assert(t.__class__ == AlignerPolynomial2DTransform)
    del t
    rt = renderapi.transform.Polynomial2DTransform(identity=True)
    t = AlignerTransform(name='Polynomial2DTransform', transform=rt)
    assert(t.__class__ == AlignerPolynomial2DTransform)

    # one way to load thinplatespline
    x = y = np.linspace(0, 2000, 4)
    xt, yt = np.meshgrid(x, y)
    src = np.vstack((xt.flatten(), yt.flatten())).transpose()
    dst = rt.tform(src)
    tps = renderapi.transform.ThinPlateSplineTransform()
    tps.estimate(src, dst)
    t = AlignerTransform(name='ThinPlateSplineTransform', transform=tps)
    assert(t.__class__ == AlignerThinPlateSplineTransform)
    del t

    # specifying something not real
    with pytest.raises(AlignerTransformException):
        AlignerTransform(name='LudicrousModel')


def example_match(npts, fac=1):
    match = {}
    match['matches'] = {
            "w": list(np.ones(npts)),
            "p": [
                list(rng.random(npts) * fac),
                list(rng.random(npts) * fac)],
            "q": [
                list(rng.random(npts) * fac),
                list(rng.random(npts) * fac)]
            }
    return match


def test_affine_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        _ = AlignerAffineModel(transform=rt)

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
    for _ in range(ntiles):
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
    assert np.allclose(r[[0, 1, 3, 4]], 1.0)
    assert np.allclose(r[[2, 5]], 0.1)
    t = AlignerTransform(name='AffineModel', transform=rt, fullsize=False)
    r = t.regularization(rdict)
    assert np.allclose(r[[0, 1]], 1.0)
    assert np.allclose(r[[2]], 0.1)


def test_similarity_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        _ = AlignerSimilarityModel(transform=rt)

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
    for _ in range(ntiles):
        index += t.from_solve_vec(vec[index:, :])
        msub = t.M.flatten()[[0, 1, 2, 5]]
        assert np.all(np.isclose(msub, vi))

    # reg
    rdict = {
            "default_lambda": 1.0,
            "translation_factor": 0.1}
    t = AlignerTransform(name='SimilarityModel')
    r = t.regularization(rdict)
    assert np.allclose(r[[0, 1]], 1.0)
    assert np.allclose(r[[2, 3]], 0.1)


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
        params = rng.normal(size=(2, n))
        rt = renderapi.transform.Polynomial2DTransform(params=params)
        t = AlignerTransform(name='Polynomial2DTransform', transform=rt)
        v = t.to_solve_vec()
        assert np.all(np.isclose(v, np.transpose(params)))

    # from vec
    for order in range(4):
        n = int((order + 1) * (order + 2) / 2)
        v0 = rng.normal(size=(n, 2))
        rt0 = renderapi.transform.Polynomial2DTransform(
                params=np.zeros((2, n)))
        t = AlignerTransform(name='Polynomial2DTransform', transform=rt0)
        assert t.order == order
        vec = np.concatenate((v0, v0, v0, v0))
        index = 0
        for _ in range(4):
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

        pf = rng.normal(size=(order + 1))
        rdict = {
                "default_lambda": 1.0,
                "translation_factor": 0.1,
                "poly_factors": pf.tolist()}
        r = t.regularization(rdict)
        ni = 0
        for i in range(order + 1):
            for _ in range(i + 1):
                assert np.all(r[ni::n] == pf[i])
                ni += 1


def test_rotation_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        _ = AlignerRotationModel(transform=rt)

    # check args
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(name='RotationModel', transform=rt)
    assert(t.__class__ == AlignerRotationModel)

    # make block
    t = AlignerTransform(name='RotationModel', transform=rt, fullsize=True)
    nmatch = 100
    # scale up because rotation filters out things near center-of-mass
    match = example_match(nmatch, fac=1000)
    ncol = 1000
    icol = 73

    ppts, qpts, w = AlignerRotationModel.preprocess(
            np.array(match['matches']['p']).transpose(),
            np.array(match['matches']['q']).transpose(),
            np.array(match['matches']['w']))

    assert ppts.shape == qpts.shape
    assert ppts.shape[0] <= nmatch

    block, weights, rhs = t.block_from_pts(
            ppts,
            w,
            icol,
            ncol)

    assert rhs.shape == ppts.shape
    assert block.check_format() is None
    assert weights.size == ppts.shape[0]
    assert block.shape == (ppts.shape[0] * t.rows_per_ptmatch, ncol)
    assert block.nnz == ppts.shape[0]

    # to vec
    t = AlignerTransform(name='RotationModel')
    v = t.to_solve_vec()
    assert np.all(v == np.array([0.0]).reshape(-1, 1))

    # from vec
    ntiles = 6
    vec = rng.normal(size=ntiles)
    vec = vec.reshape(-1, 1)
    index = 0
    for i in range(ntiles):
        t = AlignerTransform(name='RotationModel')
        index += t.from_solve_vec(vec[index:, :])
        msub = t.rotation
        assert np.isclose(np.mod(np.abs(msub - vec[i][0]), 2.0 * np.pi), 0.0)

    # reg
    rdict = {
            "default_lambda": 1.2345,
            "translation_factor": 0.1}
    t = AlignerTransform(name='RotationModel')
    r = t.regularization(rdict)
    assert np.all(np.isclose(r, 1.2345))


def test_translation_model():
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        _ = AlignerTranslationModel(transform=rt)

    # check args
    rt = renderapi.transform.AffineModel()
    t = AlignerTransform(name='TranslationModel', transform=rt)
    assert(t.__class__ == AlignerTranslationModel)

    # make block
    t = AlignerTransform(name='TranslationModel', transform=rt)
    nmatch = 100
    match = example_match(nmatch)
    ncol = 1000
    icol = 73

    block, weights, rhs = t.block_from_pts(
            np.array(match['matches']['p']).transpose(),
            np.array(match['matches']['w']),
            icol,
            ncol)

    assert rhs.shape == (nmatch, 2)
    assert block.check_format() is None
    assert weights.size == nmatch * t.rows_per_ptmatch
    assert block.shape == (nmatch * t.rows_per_ptmatch, ncol)
    assert block.nnz == 1 * nmatch

    # to vec
    t = AlignerTransform(name='TranslationModel')
    v = t.to_solve_vec()
    assert np.all(v == np.array([0.0, 0.0]).reshape(-1, 2))

    # from vec
    ntiles = 6
    vec = rng.normal(size=ntiles * 2)
    vec = vec.reshape(-1, 2)
    index = 0
    for i in range(ntiles):
        t = AlignerTransform(name='TranslationModel')
        index += t.from_solve_vec(vec[index:, :])
        msub = t.translation
        assert np.all(np.isclose(msub, vec[i]))

    # reg
    rdict = {
            "default_lambda": 1.2345,
            "translation_factor": 0.1}
    t = AlignerTransform(name='TranslationModel')
    r = t.regularization(rdict)
    assert np.all(np.isclose(r, 0.12345))


@pytest.mark.parametrize('computeAffine', [True, False])
def test_thinplate_model(computeAffine):
    # can't do this
    rt = renderapi.transform.Polynomial2DTransform()
    with pytest.raises(AlignerTransformException):
        _ = AlignerThinPlateSplineTransform(transform=rt)

    # or this
    with pytest.raises(AlignerTransformException):
        _ = AlignerTransform(name='ThinPlateSplineTransform')

    hw = 5000
    x = y = np.linspace(0, hw, 4)
    xt, yt = np.meshgrid(x, y)
    src = np.vstack((xt.flatten(), yt.flatten())).transpose()
    dst = src + rng.normal(size=(src.shape[0], src.shape[1])) * 50

    # check args
    rt = renderapi.transform.ThinPlateSplineTransform()
    rt.estimate(src, dst, computeAffine=computeAffine)
    t = AlignerTransform(name='ThinPlateSplineTransform', transform=rt)
    assert(t.__class__ == AlignerThinPlateSplineTransform)

    # make block
    nmatch = 100
    match = example_match(nmatch, fac=hw)
    ncol = 1000
    icol = 73

    block, weights, rhs = t.block_from_pts(
            np.array(match['matches']['p']).transpose(),
            np.array(match['matches']['w']),
            icol,
            ncol)

    assert rhs.shape == (nmatch, 2)
    assert block.check_format() is None
    assert weights.size == nmatch * t.rows_per_ptmatch
    assert block.shape == (nmatch * t.rows_per_ptmatch, ncol)
    if computeAffine:
        assert block.nnz == nmatch * (t.srcPts.shape[1] + 3)
    else:
        assert block.nnz == nmatch * t.srcPts.shape[1]

    # to vec
    v = t.to_solve_vec()
    if computeAffine:
        assert v.shape == (t.srcPts.shape[1] + 3, 2)
    else:
        assert v.shape == (t.srcPts.shape[1], 2)

    # from vec
    ntiles = 6
    vec = np.tile(v, (ntiles, 1))
    vec = vec.reshape(-1, 2)
    index = 0
    orig = renderapi.transform.ThinPlateSplineTransform()
    orig.estimate(src, dst, computeAffine=computeAffine)
    for _ in range(ntiles):
        index += t.from_solve_vec(vec[index:, :])
        assert np.all(np.isclose(orig.dMtxDat, t.dMtxDat))
        if computeAffine:
            assert np.all(np.isclose(orig.aMtx, t.aMtx))
            assert np.all(np.isclose(orig.bVec, t.bVec))

    # reg
    rdict = {
            "default_lambda": 1.2345,
            "translation_factor": 0.1,
            "thinplate_factor": 1000}
    r = t.regularization(rdict)
    n0 = 0
    if computeAffine:
        assert np.isclose(
                r[0], rdict['default_lambda'] * rdict['translation_factor'])
        assert np.all(np.isclose(r[1:3], rdict['default_lambda']))
        n0 += 3
    assert np.all(np.isclose(
        r[n0:], rdict['default_lambda'] * rdict['thinplate_factor']))

    # scale
    assert isinstance(t.scale, tuple)
    assert len(t.scale) == 2
