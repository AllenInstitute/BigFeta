import pytest
from bigfeta.utils import (
        blocks_from_tilespec_pair, ready_transforms)
import numpy as np
import os
import json
import renderapi
import scipy.sparse
import copy

dname = os.path.dirname(os.path.abspath(__file__))
FILE_PMS = os.path.join(
        dname, 'test_files', 'montage_pointmatches.json')
FILE_TSP = os.path.join(
        dname, 'test_files', 'montage_raw_tilespecs.json')


@pytest.fixture(scope='module')
def matches():
    with open(FILE_PMS, 'r') as f:
        j = json.load(f)
    yield j


@pytest.fixture(scope='module')
def tilespecs():
    with open(FILE_TSP, 'r') as f:
        j = json.load(f)
    tspecs = [renderapi.tilespec.TileSpec(json=i) for i in j]
    ready_transforms(tspecs, 'AffineModel', False, 2)
    yield tspecs


def test_sparse_block(matches, tilespecs):
    tids = [t.layout.sectionId for t in tilespecs]

    ncol = 1000
    pcol = 123
    qcol = 789

    ma = {
            'npts_min': 5,
            'npts_max': 500,
            'choose_random': False
            }

    for match in matches:
        if not ((match['pGroupId'] in tids) & (match['qGroupId'] in tids)):
            continue
        pi = tids.index(match['pGroupId'])
        qi = tids.index(match['qGroupId'])
        pspec = tilespecs[pi]
        qspec = tilespecs[qi]

        pblock, qblock, weights, rhs = blocks_from_tilespec_pair(
                pspec,
                qspec,
                match,
                pcol,
                qcol,
                ncol,
                ma)

        assert pblock.shape == qblock.shape == \
            (len(match['matches']['w']), ncol)

        assert pblock.shape[0] == rhs.shape[0]

        pndof = pspec.tforms[-1].DOF_per_tile
        qndof = qspec.tforms[-1].DOF_per_tile

        for block, nd, nc in zip(
                [pblock, qblock], [pndof, qndof], [pcol, qcol]):
            i, j, v = scipy.sparse.find(block)
            assert np.all(np.isin(j, nc + np.arange(nd)))


@pytest.mark.parametrize('random', [True, False])
def test_sparse_npts(matches, tilespecs, random):
    fmatches = copy.deepcopy(matches)
    tids = [t.layout.sectionId for t in tilespecs]

    ncol = 1000
    pcol = 123
    qcol = 789

    ma = {
            'npts_min': 5,
            'npts_max': 100,
            'choose_random': random
            }

    nz = 1
    fmatches[nz]['matches']['w'] = [0] * len(fmatches[nz]['matches']['w'])
    nmin = 4
    fmatches[nmin]['matches']['w'] = \
        fmatches[nmin]['matches']['w'][0:ma['npts_min'] - 1]
    for pq in ['p', 'q']:
        for i in [0, 1]:
            fmatches[nmin]['matches'][pq][i] = \
                fmatches[nmin]['matches'][pq][i][0:ma['npts_min'] - 1]

    for k, match in enumerate(fmatches):
        if not ((match['pGroupId'] in tids) & (match['qGroupId'] in tids)):
            continue
        npts = len(match['matches']['w'])

        pi = tids.index(match['pGroupId'])
        qi = tids.index(match['qGroupId'])
        pspec = tilespecs[pi]
        qspec = tilespecs[qi]

        pblock, qblock, weights, rhs = blocks_from_tilespec_pair(
                pspec,
                qspec,
                match,
                pcol,
                qcol,
                ncol,
                ma)

        if np.all(np.array(match['matches']['w']) == 0):
            for x in [pblock, qblock, weights, rhs]:
                assert x is None
            continue

        if len(match['matches']['w']) < ma['npts_min']:
            for x in [pblock, qblock, weights, rhs]:
                assert x is None
            continue

        nrow = npts
        if npts > ma['npts_max']:
            nrow = ma['npts_max']

        assert pblock.shape == qblock.shape == (nrow, ncol)

        pndof = pspec.tforms[-1].DOF_per_tile
        qndof = qspec.tforms[-1].DOF_per_tile

        for block, nd, nc in zip(
                [pblock, qblock], [pndof, qndof], [pcol, qcol]):
            i, j, v = scipy.sparse.find(block)
            assert np.all(np.isin(j, nc + np.arange(nd)))
