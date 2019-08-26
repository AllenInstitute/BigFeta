import pytest
import renderapi
from test_data import (render_params,
                       montage_raw_tilespecs_json,
                       montage_parameters)
from bigfeta import bigfeta
from bigfeta import jsongz
import json
from marshmallow.exceptions import ValidationError
import copy
import os
import numpy as np
import shutil

dname = os.path.dirname(os.path.abspath(__file__))
FILE_PMS = os.path.join(
        dname, 'test_files', 'montage_pointmatches.json')


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
    renderapi.stack.set_stack_state(test_raw_stack, 'COMPLETE', render=render)
    yield test_raw_stack
    renderapi.stack.delete_stack(test_raw_stack, render=render)


@pytest.fixture(scope='module')
def montage_pointmatches(render):
    test_montage_collection = 'montage_collection'
    pms_from_json = []
    with open(FILE_PMS, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_montage_collection, pms_from_json, render=render)
    yield test_montage_collection
    renderapi.pointmatch.delete_collection(
            test_montage_collection, render=render)


@pytest.fixture(scope='module')
def solved_montage(render, raw_stack, montage_pointmatches):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack
    p['output_stack']['name'] = 'solver_output_stack'
    p['pointmatch']['name'] = montage_pointmatches
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    yield mod
    renderapi.stack.delete_stack('output_stack_name', render=render)


def test_validation(raw_stack, montage_pointmatches):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['db_interface'] = 'file'
    p['input_stack']['input_file'] = None
    p['output_mode'] = 'none'
    p['pointmatch']['name'] = montage_pointmatches
    with pytest.raises(ValidationError):
        tmod = bigfeta.BigFeta(input_data=p, args=[])
        del tmod

    p['input_stack']['db_interface'] = 'mongo'
    p['input_stack']['name'] = raw_stack
    p['output_stack']['db_interface'] = 'file'
    p['output_stack']['output_file'] = None
    with pytest.raises(ValidationError):
        tmod = bigfeta.BigFeta(input_data=p, args=[])
        del tmod


@pytest.mark.parametrize("compress", [True, False])
def test_input_stack_file(
        render, raw_stack, montage_pointmatches,
        tmpdir, solved_montage, compress):
    p = copy.deepcopy(montage_parameters)
    resolved = renderapi.resolvedtiles.get_resolved_tiles_from_z(
            raw_stack,
            p['first_section'],
            render=render)
    tmp_file_dir = str(tmpdir.mkdir('file_test_dir'))
    input_stack_file = os.path.join(
            tmp_file_dir, "input_stack.json")
    input_stack_file = jsongz.dump(
            resolved.to_dict(),
            input_stack_file,
            compress=compress)

    p['input_stack']['db_interface'] = 'file'
    p['input_stack']['input_file'] = input_stack_file
    p['output_mode'] = 'none'
    p['pointmatch']['name'] = montage_pointmatches

    tmod = bigfeta.BigFeta(input_data=p, args=[])
    tmod.run()

    for k in ['precision', 'error', 'err']:
        assert np.all(
                np.isclose(
                    np.array(tmod.results[k]),
                    np.array(solved_montage.results[k]), atol=1e-7))

    assert np.all(
            np.isclose(
                np.linalg.norm(solved_montage.results['x'], axis=0),
                np.linalg.norm(tmod.results['x'], axis=0), atol=1e-7))

    orig_ids = np.array([
        t.tileId for t in solved_montage.resolvedtiles.tilespecs])
    for t in tmod.resolvedtiles.tilespecs:
        i = np.argwhere(orig_ids == t.tileId).flatten()[0]
        assert np.all(
                np.isclose(
                    solved_montage.resolvedtiles.tilespecs[i].tforms[-1].M,
                    t.tforms[-1].M, atol=1e-7))

    del tmod
    shutil.rmtree(tmp_file_dir)


@pytest.mark.parametrize("compress", [True, False])
def test_match_file(
        render, raw_stack, montage_pointmatches,
        tmpdir, solved_montage, compress):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack
    p['pointmatch']['name'] = montage_pointmatches

    # get the matches and write them to a file
    sectionData = renderapi.stack.get_stack_sectionData(
            p['input_stack']['name'],
            render=render)
    sections = [sd['sectionId'] for sd in sectionData]
    matches = []
    for s in sections:
        matches += renderapi.pointmatch.get_matches_with_group(
                p['pointmatch']['name'],
                s,
                render=render)
    tmp_file_dir = str(tmpdir.mkdir('file_test_dir'))
    match_file = os.path.join(
            tmp_file_dir, "matches.json")
    match_file = jsongz.dump(matches, match_file, compress=compress)

    p['pointmatch']['db_interface'] = 'file'
    p['pointmatch']['input_file'] = match_file
    p['output_mode'] = 'none'

    tmod = bigfeta.BigFeta(input_data=p, args=[])
    tmod.run()

    for k in ['precision', 'error', 'err']:
        assert np.all(
                np.isclose(
                    np.array(tmod.results[k]),
                    np.array(solved_montage.results[k]), atol=1e-7))

    assert np.all(
            np.isclose(
                np.linalg.norm(solved_montage.results['x'], axis=0),
                np.linalg.norm(tmod.results['x'], axis=0), atol=1e-7))

    orig_ids = np.array([
        t.tileId for t in solved_montage.resolvedtiles.tilespecs])
    for t in tmod.resolvedtiles.tilespecs:
        i = np.argwhere(orig_ids == t.tileId).flatten()[0]
        assert np.all(
                np.isclose(
                    solved_montage.resolvedtiles.tilespecs[i].tforms[-1].M,
                    t.tforms[-1].M, atol=1e-7))

    del tmod
    shutil.rmtree(tmp_file_dir)


@pytest.mark.parametrize("compress", [True, False])
def test_output_file(
        render, raw_stack, montage_pointmatches,
        tmpdir, solved_montage, compress):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = raw_stack
    p['pointmatch']['name'] = montage_pointmatches

    tmp_file_dir = str(tmpdir.mkdir('file_test_dir'))
    p['output_stack']['db_interface'] = 'file'
    p['output_stack']['output_file'] = os.path.join(
            tmp_file_dir,
            "resolvedtiles.json")
    p['output_stack']['compress_output'] = compress

    tmod = bigfeta.BigFeta(input_data=p, args=[])
    tmod.run()

    solved = renderapi.resolvedtiles.ResolvedTiles(
            json=jsongz.load(tmod.args['output_stack']['output_file']))

    orig_ids = np.array([
        t.tileId for t in solved_montage.resolvedtiles.tilespecs])
    for t in solved.tilespecs:
        i = np.argwhere(orig_ids == t.tileId).flatten()[0]
        assert np.all(
                np.isclose(
                    solved_montage.resolvedtiles.tilespecs[i].tforms[-1].M,
                    t.tforms[-1].M, atol=1e-7))

    del tmod
    shutil.rmtree(tmp_file_dir)
