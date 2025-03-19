import os
import pytest
import renderapi
from test_data import (
        render_params,
        rough_parameters)
from bigfeta import bigfeta
import json
import os
import copy
import numpy as np
from shutil import rmtree
import subprocess

TEST_PETSC = os.getenv("BIGFETA_TEST_PETSC", False)
TEST_PETSC = (True if (TEST_PETSC is True or TEST_PETSC in ("true", "1", "yes")) else False)

if not TEST_PETSC:
    pytestmark = pytest.mark.skip("PETSc tests are disabled")

dname = os.path.dirname(os.path.abspath(__file__))
FILE_ROUGH_TILES = os.path.join(
        dname, 'test_files', 'rough_input_tiles.json')
FILE_ROUGH_PMS = os.path.join(
        dname, 'test_files', 'rough_input_matches.json')


@pytest.fixture(scope='module')
def render():
    render = renderapi.connect(**render_params)
    return render


# raw stack tiles
@pytest.fixture(scope='module')
def rough_input_stack(render):
    test_rough_stack = 'hdf5_rough_input_stack'
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


@pytest.fixture(scope='module')
def rough_pointmatches(render):
    test_rough_collection = 'hdf5_rough_collection'
    with open(FILE_ROUGH_PMS, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_rough_collection, pms_from_json, render=render)
    yield test_rough_collection
    renderapi.pointmatch.delete_collection(
            test_rough_collection, render=render)


@pytest.fixture(scope='function')
def output_directory(tmpdir_factory):
    outdir = str(tmpdir_factory.mktemp("tmp_outputs"))
    yield outdir
    rmtree(outdir)


@pytest.mark.parametrize('transform', ['RotationModel', 'AffineModel'])
def test_petsc_solver(
        render,
        rough_input_stack,
        rough_pointmatches,
        output_directory,
        transform):

    # setup the solver to write out hdf5 files
    parameters = copy.deepcopy(rough_parameters)
    parameters['hdf5_options']['output_dir'] = output_directory
    parameters['input_stack']['name'] = rough_input_stack
    parameters['pointmatch']['name'] = rough_pointmatches
    parameters['output_mode'] = 'hdf5'
    parameters['hdf5_options']['chunks_per_file'] = 10
    parameters['transformation'] = transform
    if transform == 'AffineModel':
        parameters['regularization'] = {
                "default_lamda": 1e3,
                "translation_factor": 1.0e-10
                }

    mod = bigfeta.BigFeta(
            input_data=parameters, args=[])
    mod.run()
    indexfile = os.path.join(
            parameters['hdf5_options']['output_dir'],
            'solution_input.h5')
    assert os.path.isfile(indexfile)

    # call the petsc solver
    outfile = os.path.join(
            parameters['hdf5_options']['output_dir'],
            'solution_output.h5')
    # /tmp is automatically bound
    cmd = [
            'singularity',
            'run',
            './bigfeta/distributed/bin/petsc_solver.simf']
    cmd += ['-input', indexfile]
    cmd += ['-output', outfile]
    # this is a direct PaStiX solve
    cmd += ['-ksp_type', 'preonly', '-pc_type', 'lu']
    subprocess.call(cmd)
    assert os.path.isfile(outfile)

    # read the petsc result back into render
    parameters['ingest_from_file'] = outfile
    parameters['output_mode'] = 'stack'
    parameters['output_stack']['name'] = 'from_petsc'
    mod = bigfeta.BigFeta(
            input_data=parameters, args=[])
    mod.run()

    petsc_solved = renderapi.tilespec.get_tile_specs_from_stack(
            parameters['output_stack']['name'][0],
            render=render)

    # have scipy solve the same thing
    parameters['ingest_from_file'] = ''
    parameters['output_mode'] = 'stack'
    parameters['output_stack']['name'] = 'from_scipy'
    smod = bigfeta.BigFeta(
            input_data=parameters, args=[])
    smod.run()

    scipy_solved = renderapi.tilespec.get_tile_specs_from_stack(
            parameters['output_stack']['name'][0],
            render=render)

    assert len(petsc_solved) == len(scipy_solved)
    assert np.all(
            np.isclose(
                mod.results['precision'],
                smod.results['precision'],
                rtol=10.0,
                atol=1e-10))
    assert np.all(
            np.isclose(
                mod.results['error'],
                smod.results['error'],
                rtol=0.01,
                atol=1.0))
    assert np.all(
            np.isclose(
                mod.results['err'],
                smod.results['err'],
                rtol=0.01,
                atol=0.2))
    ptids = np.array([t.tileId for t in petsc_solved])
    stids = np.array([t.tileId for t in scipy_solved])
    in1d = np.intersect1d(ptids, stids)
    for tid in in1d:
        p = np.argwhere(ptids == tid).flatten()[0]
        s = np.argwhere(stids == tid).flatten()[0]
        ptf = petsc_solved[p].tforms[-1]
        stf = scipy_solved[s].tforms[-1]
        assert np.isclose(ptf.rotation, stf.rotation, rtol=10.0, atol=0.05)
        assert np.all(np.isclose(ptf.scale, stf.scale, rtol=0.001, atol=0.001))
        assert np.isclose(ptf.shear, stf.shear, rtol=0.1, atol=0.001)
        assert np.all(
                np.isclose(
                    ptf.translation, stf.translation, rtol=0.1, atol=0.5))


@pytest.mark.parametrize("chunks", [-1, 1, 2])
def test_hdf5_mode_similarity(
        render,
        rough_input_stack,
        rough_pointmatches,
        tmpdir,
        chunks):
    # general parameters
    parameters = copy.deepcopy(rough_parameters)
    parameters['hdf5_options']['output_dir'] = str(tmpdir.mkdir('hdf5output'))
    parameters['input_stack']['name'] = rough_input_stack
    parameters['pointmatch']['name'] = rough_pointmatches
    parameters['output_mode'] = 'hdf5'
    parameters['hdf5_options']['chunks_per_file'] = chunks

    # check output mode HDF5
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(parameters), args=[])
    mod.run()
    indexfile = os.path.join(
            parameters['hdf5_options']['output_dir'],
            'solution_input.h5')
    assert os.path.exists(indexfile)
    del mod

    # check assemble from file
    parameters['output_mode'] = 'none'
    parameters['assemble_from_file'] = indexfile
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(parameters), args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 1e6)
    del mod

    # delete the output stack, just in case
    renderapi.stack.delete_stack(
                parameters['output_stack']['name'],
                render=render)

    parameters['ingest_from_file'] = indexfile
    parameters['output_mode'] = 'stack'
    mod = bigfeta.BigFeta(
            input_data=copy.deepcopy(parameters), args=[])
    mod.run()
    tin = renderapi.tilespec.get_tile_specs_from_stack(
            parameters['input_stack']['name'],
            render=render)
    tout = renderapi.tilespec.get_tile_specs_from_stack(
            parameters['output_stack']['name'],
            render=render)
    assert len(tin) == len(tout)
    os.remove(indexfile)
    del mod

    renderapi.stack.delete_stack(
                parameters['output_stack']['name'],
                render=render)
