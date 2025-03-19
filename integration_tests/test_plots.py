import pytest
import renderapi
from test_data import (render_params,
                       montage_raw_tilespecs_json,
                       montage_parameters)
from bigfeta.qctools.CheckPointMatches import CheckPointMatches
# from bigfeta.qctools.CheckResiduals import CheckResiduals  # residual has bad dep
from bigfeta.qctools.CheckTransforms import CheckTransforms, fixpi
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import copy

TEST_PLOTS = os.getenv("BIGFETA_TEST_PLOTS", False)
TEST_PLOTS = (True if (TEST_PLOTS is True or TEST_PLOTS in ("true", "1", "yes")) else False)
if not TEST_PLOTS:
    pytestmark = pytest.mark.skip("Plotting tests are disabled")

FILE_PMS = './integration_tests/test_files/montage_pointmatches.json'


@pytest.fixture(scope='module')
def render():
    render = renderapi.connect(**render_params)
    return render


@pytest.fixture(scope='module')
def raw_stack(render):
    test_raw_stack = 'input_raw_stack'
    tilespecs = [
            renderapi.tilespec.TileSpec(json=d)
            for d in montage_raw_tilespecs_json]
    renderapi.stack.create_stack(test_raw_stack, render=render)
    renderapi.client.import_tilespecs(
        test_raw_stack, tilespecs, render=render, use_rest=True)
    renderapi.stack.set_stack_state(
        test_raw_stack, 'COMPLETE', render=render)
    yield test_raw_stack
    renderapi.stack.delete_stack(test_raw_stack, render=render)


@pytest.fixture(scope='module')
def raw_stack_offset_z(render):
    dz = 10
    test_raw_stack_offset = 'input_raw_stack_offset'
    tilespecs = [
            renderapi.tilespec.TileSpec(json=d)
            for d in montage_raw_tilespecs_json]
    for i in range(len(tilespecs)):
        tilespecs[i].z += dz
        tilespecs[i].layout.sectionId = '%0.1f' % tilespecs[i].z

    renderapi.stack.create_stack(
        test_raw_stack_offset, render=render)
    renderapi.client.import_tilespecs(
        test_raw_stack_offset, tilespecs, render=render, use_rest=True)
    renderapi.stack.set_stack_state(
        test_raw_stack_offset, 'COMPLETE', render=render)
    yield test_raw_stack_offset
    renderapi.stack.delete_stack(
        test_raw_stack_offset, render=render)


@pytest.fixture(scope='module')
def raw_stack_loading(render):
    test_raw_stack = 'input_raw_stack_2'
    tilespecs = [
            renderapi.tilespec.TileSpec(json=d)
            for d in montage_raw_tilespecs_json]
    renderapi.stack.create_stack(test_raw_stack, render=render)
    renderapi.client.import_tilespecs(
            test_raw_stack, tilespecs, render=render, use_rest=True)
    yield test_raw_stack
    renderapi.stack.delete_stack(test_raw_stack, render=render)


@pytest.fixture(scope='module')
def montage_pointmatches(render):
    test_montage_collection = 'montage_collection'
    pms_from_json = json.load(open(FILE_PMS, 'r'))
    renderapi.pointmatch.import_matches(
            test_montage_collection, pms_from_json, render=render)
    yield test_montage_collection


def test_fixpi():
    arr = np.arange(0, 10.0 * np.pi, 0.1)
    narr = fixpi(arr)
    assert(narr.size == arr.size)
    assert(np.abs(narr).max() <= np.pi)


def test_pmplot(render, montage_pointmatches, raw_stack, tmpdir):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack
    p['pointmatch']['name'] = montage_pointmatches
    mod = CheckPointMatches(input_data=p, args=[])
    mod.args['z1'] = 1015
    mod.args['z2'] = 1015
    mod.args['plot_dir'] = str(tmpdir.mkdir('plotoutput'))
    mod.run()
    assert os.path.exists(mod.outputname)
    mod.args['plot'] = False
    mod.run()


def test_pmplot_loading(
        render, montage_pointmatches, raw_stack_loading, tmpdir):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack_loading
    p['pointmatch']['name'] = montage_pointmatches
    mod = CheckPointMatches(input_data=p, args=[])
    mod.args['z1'] = 1015
    mod.args['z2'] = 1015
    mod.args['plot_dir'] = str(tmpdir.mkdir('plotoutput'))
    mod.run()
    assert os.path.exists(mod.outputname)


def test_resplot(render, montage_pointmatches, raw_stack, tmpdir):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack
    p['output_stack']['name'] = raw_stack
    p['pointmatch']['name'] = montage_pointmatches
    mod = CheckResiduals(input_data=p, args=[])
    mod.args['z1'] = 1015
    mod.args['z2'] = 1015
    mod.args['plot_dir'] = str(tmpdir.mkdir('plotoutput'))
    mod.args['savefig'] = "True"
    mod.run()
    assert os.path.exists(mod.outputname)
    fig = plt.figure(12)
    mod.make_lc_plots(fig)


def test_trplot(render, montage_pointmatches, raw_stack, tmpdir):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack
    p['output_stack']['name'] = raw_stack
    p['pointmatch']['name'] = montage_pointmatches
    mod = CheckTransforms(input_data=p, args=[])
    mod.args['z1'] = 1015
    mod.args['plot_dir'] = str(tmpdir.mkdir('plotoutput'))
    mod.args['plot'] = "False"
    mod.run()
    assert os.path.exists(mod.outputname)


def test_resplot_zoff(
        render, montage_pointmatches, raw_stack_offset_z, tmpdir):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack_offset_z
    p['output_stack']['name'] = raw_stack_offset_z
    p['pointmatch']['name'] = montage_pointmatches
    mod = CheckResiduals(input_data=p, args=[])
    mod.args['z1'] = 1025
    mod.args['z2'] = 1025
    mod.args['zoff'] = -10
    mod.args['plot'] = False
    mod.args['savefig'] = "True"
    mod.run()
