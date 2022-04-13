import pytest
import renderapi
from test_data import (
        render_params,
        rough_parameters)
from bigfeta import bigfeta, utils
from marshmallow import ValidationError
import json
import os
import copy
import numpy as np


dname = os.path.dirname(os.path.abspath(__file__))
FILE_ROUGH_TILES = os.path.join(
        dname, 'test_files', 'rough_input_tiles.json')
FILE_ROUGH_PMS = os.path.join(
        dname, 'test_files', 'rough_input_matches.json')
FILE_ROUGH_PMS_S1 = os.path.join(
        dname, 'test_files', 'rough_input_matches_split1.json')
FILE_ROUGH_PMS_S2 = os.path.join(
        dname, 'test_files', 'rough_input_matches_split2.json')


@pytest.fixture(scope='module')
def render():
    render = renderapi.connect(**render_params)
    return render


@pytest.fixture(scope='function')
def output_stack_name(render):
    name = 'solver_output_stack'
    yield name
    renderapi.stack.delete_stack(name, render=render)


# raw stack tiles
@pytest.fixture(scope='module')
def rough_input_stack(render):
    test_rough_stack = 'rough_input_stack'
    with open(FILE_ROUGH_TILES, 'r') as f:
        j = json.load(f)
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in j]
    renderapi.stack.create_stack(test_rough_stack, render=render)
    renderapi.client.import_tilespecs(
            test_rough_stack, tilespecs, render=render, use_rest=True)
    renderapi.stack.set_stack_state(
            test_rough_stack, 'COMPLETE', render=render)
    yield test_rough_stack
    renderapi.stack.delete_stack(test_rough_stack, render=render)


# raw stack tiles with tileids renamed
@pytest.fixture(scope='module')
def rough_input_stack_renamed(render):
    test_rough_stack = 'rough_input_stack_renamed'
    with open(FILE_ROUGH_TILES, 'r') as f:
        j = json.load(f)
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in j]
    for i in range(len(tilespecs)):
        tilespecs[i].tileId = 'garbage_%d' % i
    # no matches will match this
    renderapi.stack.create_stack(test_rough_stack, render=render)
    renderapi.client.import_tilespecs(
            test_rough_stack, tilespecs, render=render, use_rest=True)
    renderapi.stack.set_stack_state(
            test_rough_stack, 'COMPLETE', render=render)
    yield test_rough_stack
    renderapi.stack.delete_stack(test_rough_stack, render=render)


# raw stack tiles with one z removed
@pytest.fixture(scope='module')
def rough_input_stack_2(render):
    test_rough_stack2 = 'rough_input_stack_2'
    with open(FILE_ROUGH_TILES, 'r') as f:
        j = json.load(f)
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in j]
    renderapi.stack.create_stack(
            test_rough_stack2, render=render)
    renderapi.client.import_tilespecs(
            test_rough_stack2, tilespecs, render=render, use_rest=True)
    z_values = renderapi.stack.get_z_values_for_stack(
            test_rough_stack2, render=render)
    renderapi.stack.delete_section(
            test_rough_stack2, z_values[3], render=render)
    renderapi.stack.set_stack_state(
            test_rough_stack2, 'COMPLETE', render=render)
    yield test_rough_stack2
    renderapi.stack.delete_stack(test_rough_stack2, render=render)


# raw stack with some extra transforms inserted
@pytest.fixture(scope='module')
def rough_input_stack_3(render):
    test_rough_stack3 = 'rough_input_stack_3'
    with open(FILE_ROUGH_TILES, 'r') as f:
        j = json.load(f)
    tilespecs = [renderapi.tilespec.TileSpec(json=d) for d in j]
    tfs = [
            renderapi.transform.AffineModel(M00=2.0, M11=2.0),
            renderapi.transform.AffineModel(M00=0.5, M11=0.5)]
    for i in range(len(tilespecs)):
        tilespecs[i].tforms.insert(0, tfs[1])
        tilespecs[i].tforms.insert(0, tfs[0])

    renderapi.stack.create_stack(
            test_rough_stack3, render=render)
    renderapi.client.import_tilespecs(
            test_rough_stack3, tilespecs, render=render, use_rest=True)
    renderapi.stack.set_stack_state(
            test_rough_stack3, 'COMPLETE', render=render)
    yield test_rough_stack3
    renderapi.stack.delete_stack(test_rough_stack3, render=render)


@pytest.fixture(scope='module')
def rough_pointmatches(render):
    test_rough_collection = 'rough_collection'
    with open(FILE_ROUGH_PMS, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_rough_collection, pms_from_json, render=render)
    yield test_rough_collection
    renderapi.pointmatch.delete_collection(
            test_rough_collection, render=render)


@pytest.fixture(scope='module')
def rough_pointmatches_missing(render):
    test_rough_collection = 'rough_collection_missing'
    with open(FILE_ROUGH_PMS, 'r') as f:
        pms_from_json = json.load(f)
    # pop out one match
    pms_from_json.pop(100)
    renderapi.pointmatch.import_matches(
            test_rough_collection, pms_from_json, render=render)
    yield test_rough_collection
    renderapi.pointmatch.delete_collection(
            test_rough_collection, render=render)


@pytest.fixture(scope='module')
def split_rough_pointmatches(render):
    test_rough_collection1 = 'rough_collection_split1'
    test_rough_collection2 = 'rough_collection_split2'
    with open(FILE_ROUGH_PMS_S1, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_rough_collection1, pms_from_json, render=render)
    with open(FILE_ROUGH_PMS_S2, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_rough_collection2, pms_from_json, render=render)
    yield [test_rough_collection1, test_rough_collection2]
    renderapi.pointmatch.delete_collection(
            test_rough_collection1, render=render)
    renderapi.pointmatch.delete_collection(
            test_rough_collection2, render=render)


def test_rough_similarity_explicit_depth(
        render,
        rough_pointmatches,
        rough_input_stack,
        output_stack_name):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    rough_parameters2['matrix_assembly']['depth'] = [0, 1, 2]
    rough_parameters2['matrix_assembly']['explicit_weight_by_depth'] = \
        [0, 0.5, 0.33]
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)
    del mod

    with pytest.raises(ValidationError):
        rough_parameters2['matrix_assembly']['depth'] = [0, 1]
        # not the same length as weights
        bigfeta.BigFeta(
                input_data=copy.deepcopy(rough_parameters2), args=[])


def test_multi_stack_name_exception(
        render,
        rough_pointmatches,
        rough_input_stack,
        output_stack_name):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    with pytest.raises(ValidationError):
        rough_parameters2['input_stack']['name'] = [
                rough_parameters2['input_stack']['name']] * 2
        # stacks should only have 1 name (so far)
        bigfeta.BigFeta(
                input_data=copy.deepcopy(rough_parameters2), args=[])


def test_multi_profile_exception(
        render,
        rough_pointmatches,
        rough_input_stack,
        output_stack_name):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    with pytest.raises(utils.BigFetaException):
        mod.args['profile_data_load'] = True
        mod.run()
    del mod


def test_rough_similarity_2(
        render,
        rough_pointmatches,
        rough_input_stack_2,
        output_stack_name):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_2
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)

    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)
    del mod


@pytest.mark.parametrize("choose_random", [True, False])
def test_rough_rotation(
        render,
        rough_pointmatches,
        rough_input_stack,
        output_stack_name,
        choose_random):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'RotationModel'
    rough_parameters2["matrix_assembly"]["choose_random"] = choose_random
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)

    assert np.all(np.array(mod.results['precision']) < 1e-10)
    assert len(tin) == len(tout)
    del mod


def test_apply_list(
        render,
        rough_pointmatches,
        rough_input_stack_3,
        output_stack_name):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_3
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'RotationModel'
    rough_parameters2['transform_apply'] = [0, 1]
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert np.all(np.array(mod.results['precision']) < 1e-10)
    assert len(tin) == len(tout)

    rough_parameters2['transform_apply'] = [4]
    with pytest.raises(IndexError):
        mod = bigfeta.BigFeta(
                input_data=copy.deepcopy(rough_parameters2), args=[])
        mod.run()
    del mod


@pytest.mark.parametrize("pm_db_intfc", ['render', 'mongo'])
def test_rough_similarity_split(
        render,
        split_rough_pointmatches,
        rough_input_stack_2,
        output_stack_name,
        pm_db_intfc):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_2
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = split_rough_pointmatches
    rough_parameters2['pointmatch']['db_interface'] = pm_db_intfc
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)
    del mod


@pytest.mark.parametrize("pm_db_intfc", ['render', 'mongo'])
def test_missing_section(
        render,
        rough_pointmatches,
        rough_input_stack_2,
        output_stack_name,
        pm_db_intfc):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack_2
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    rough_parameters2['input_stack']['db_interface'] = pm_db_intfc

    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)
    del mod


def test_missing_match(
        # this case was covered elsewhere, but one more test
        render,
        rough_pointmatches_missing,
        rough_input_stack,
        output_stack_name):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['pointmatch']['name'] = rough_pointmatches_missing
    rough_parameters2['transformation'] = 'SimilarityModel'

    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['input_stack']['name'], render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            rough_parameters2['output_stack']['name'], render=render)
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    assert len(tin) == len(tout)
    del mod


def test_affine_on_similarity(
        render,
        rough_pointmatches,
        rough_input_stack,
        output_stack_name):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['output_stack']['name'] = 'sim_out'
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'SimilarityModel'
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    del mod

    rough_parameters2['input_stack']['name'] = 'sim_out'
    rough_parameters2['output_stack']['name'] = output_stack_name
    rough_parameters2['transformation'] = 'AffineModel'
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()

    renderapi.stack.delete_stack(
            rough_parameters2['input_stack']['name'], render=render)
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    del mod


def test_output_mode_none(
        render,
        rough_pointmatches,
        rough_input_stack,
        output_stack_name):
    rough_parameters2 = copy.deepcopy(rough_parameters)
    rough_parameters2['input_stack']['name'] = rough_input_stack
    rough_parameters2['pointmatch']['name'] = rough_pointmatches
    rough_parameters2['transformation'] = 'AffineModel'
    rough_parameters2['output_mode'] = 'none'
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(rough_parameters2), args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    stacks = renderapi.render.get_stacks_by_owner_project(render=render)
    assert output_stack_name not in stacks
    del mod
