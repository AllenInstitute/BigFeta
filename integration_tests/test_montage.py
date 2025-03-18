import pytest
import itertools
import renderapi
from test_data import (render_params,
                       montage_raw_tilespecs_json,
                       montage_parameters)
from bigfeta import bigfeta
from bigfeta import solve
import json
from marshmallow.exceptions import ValidationError
import copy
import os
import numpy as np

from bigfeta.solve_tools import regularization_sweep

dname = os.path.dirname(os.path.abspath(__file__))
FILE_PMS = os.path.join(
        dname, 'test_files', 'montage_pointmatches.json')
FILE_PMS_S1 = os.path.join(
        dname, 'test_files', 'montage_pointmatches_split1.json')
FILE_PMS_S2 = os.path.join(
        dname, 'test_files', 'montage_pointmatches_split2.json')


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


@pytest.fixture(scope='function')
def loading_raw_stack(render):
    test_raw_stack = 'input_raw_stack_loading'
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
    pms_from_json = []
    with open(FILE_PMS, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_montage_collection, pms_from_json, render=render)
    yield test_montage_collection
    renderapi.pointmatch.delete_collection(
            test_montage_collection, render=render)


@pytest.fixture(scope='module')
def split_montage_pointmatches(render):
    test_montage_collection1 = 'montage_collection_split_1'
    test_montage_collection2 = 'montage_collection_split_2'
    pms_from_json = []
    with open(FILE_PMS_S1, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_montage_collection1, pms_from_json, render=render)
    with open(FILE_PMS_S2, 'r') as f:
        pms_from_json = json.load(f)
    renderapi.pointmatch.import_matches(
            test_montage_collection2, pms_from_json, render=render)
    yield [test_montage_collection1, test_montage_collection2]
    renderapi.pointmatch.delete_collection(
            test_montage_collection1, render=render)
    renderapi.pointmatch.delete_collection(
            test_montage_collection2, render=render)


@pytest.fixture(scope='module')
def montage_pointmatches_weighted(render):
    test_montage_collection2 = 'montage_collection2'
    pms_from_json = []
    with open(FILE_PMS, 'r') as f:
        pms_from_json = json.load(f)
    n = len(pms_from_json[0]['matches']['w'])
    pms_from_json[0]['matches']['w'] = [0.0 for i in range(n)]

    renderapi.pointmatch.import_matches(
            test_montage_collection2, pms_from_json, render=render)
    yield test_montage_collection2
    renderapi.pointmatch.delete_collection(
            test_montage_collection2, render=render)


@pytest.fixture(scope='function')
def output_stack_name(render):
    name = 'solver_output_stack'
    yield name
    renderapi.stack.delete_stack(name, render=render)


@pytest.mark.parametrize("stack_state", ["COMPLETE", "LOADING"])
def test_weighted(
        render, montage_pointmatches_weighted, loading_raw_stack,
        stack_state, tmpdir, output_stack_name):
    renderapi.stack.set_stack_state(
            loading_raw_stack, stack_state, render=render)
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['output_stack']['name'] = output_stack_name
    p['pointmatch']['name'] = montage_pointmatches_weighted
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 200)
    del mod


def test_multi_pm(
        render,
        split_montage_pointmatches,
        loading_raw_stack,
        tmpdir,
        output_stack_name):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['output_stack']['name'] = output_stack_name
    p['pointmatch']['name'] = split_montage_pointmatches
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 200)
    del mod


@pytest.mark.parametrize(
        "transform, fullsize, order",
        [("AffineModel", True, 0),
         ("AffineModel", False, 0),
         ("SimilarityModel", False, 0),
         ("Polynomial2DTransform", False, 0),
         ("Polynomial2DTransform", False, 1)])
def test_different_transforms(
        render, montage_pointmatches, loading_raw_stack,
        transform, fullsize, output_stack_name, order):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['output_stack']['name'] = output_stack_name
    p['pointmatch']['name'] = montage_pointmatches
    p['transformation'] = transform
    p['fullsize'] = fullsize
    p['poly_order'] = order
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 200)
    del mod


def test_polynomial(
        render, montage_pointmatches, loading_raw_stack,
        output_stack_name):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['output_stack']['name'] = output_stack_name
    p['pointmatch']['name'] = montage_pointmatches
    p['transformation'] = 'Polynomial2DTransform'
    p['poly_order'] = 2
    p['regularization'] = {
            'default_lambda': 1000.0,
            'translation_factor': 1e-5,
            'poly_factors': [1e-5, 1000.0, 1e6]}
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-4)
    assert np.all(np.array(mod.results['error']) < 200)
    del mod


def test_thinplate(
        render, montage_pointmatches, loading_raw_stack,
        output_stack_name):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['output_stack']['name'] = output_stack_name
    p['pointmatch']['name'] = montage_pointmatches
    p['transformation'] = 'ThinPlateSplineTransform'
    p['regularization'] = {
            'default_lambda': 1000.0,
            'translation_factor': 1e-5,
            'thinplate_factor': 1e-5}
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-4)
    assert np.all(np.array(mod.results['error']) < 200)
    del mod


def test_poly_validation(output_stack_name):
    p = copy.deepcopy(montage_parameters)
    p['regularization'] = {
            'default_lambda': 1000.0,
            'translation_factor': 1e-5,
            'poly_factors': [1e-5, 1000.0, 1e6, 1e3]}
    p['output_stack']['name'] = output_stack_name
    p['transformation'] = 'Polynomial2DTransform'
    p['poly_order'] = 2
    with pytest.raises(ValidationError):
        # because poly_factors should be length 3
        bigfeta.BigFeta(input_data=p, args=[])


@pytest.mark.parametrize("stack_state", ["COMPLETE", "LOADING"])
def test_stack_state(
        render, montage_pointmatches, output_stack_name,
        loading_raw_stack, stack_state, tmpdir):
    renderapi.stack.set_stack_state(
            loading_raw_stack, stack_state, render=render)
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['output_stack']['name'] = output_stack_name
    p['pointmatch']['name'] = montage_pointmatches
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 200)
    del mod


@pytest.mark.parametrize("db_intfc", ["render", "mongo"])
def test_basic(
        render, montage_pointmatches, output_stack_name,
        loading_raw_stack, db_intfc):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['output_stack']['name'] = output_stack_name
    p['pointmatch']['name'] = montage_pointmatches
    p['transformation'] = 'AffineModel'
    p['fullsize_transform'] = True
    p['input_stack']['db_interface'] = db_intfc
    p['output_stack']['db_interface'] = 'render'
    p['pointmatch']['db_interface'] = db_intfc
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 200)
    del mod


@pytest.mark.parametrize("render_output", ["null", "anything else"])
def test_render_output(
        render, montage_pointmatches, output_stack_name,
        loading_raw_stack, render_output, tmpdir):
    p = copy.deepcopy(montage_parameters)
    p['input_stack']['name'] = loading_raw_stack
    p['output_stack']['name'] = output_stack_name
    p['pointmatch']['name'] = montage_pointmatches
    p['render_output'] = render_output
    mod = bigfeta.BigFeta(input_data=p, args=[])
    mod.run()
    assert np.all(np.array(mod.results['precision']) < 1e-7)
    assert np.all(np.array(mod.results['error']) < 200)
    del mod


@pytest.fixture(scope="module")
def resolvedtiles_obj():
    yield renderapi.resolvedtiles.ResolvedTiles(
        tilespecs=[renderapi.tilespec.TileSpec(json=d)
                   for d in montage_raw_tilespecs_json])


@pytest.fixture(scope="module")
def matches_obj():
    with open(FILE_PMS, 'r') as f:
        matches = json.load(f)
    yield matches


def test_run_resolvedtiles(resolvedtiles_obj, matches_obj):
    p = copy.deepcopy(montage_parameters)

    rts = renderapi.resolvedtiles.ResolvedTiles(
        tilespecs=[ts for ts in resolvedtiles_obj.tilespecs
                   if p["first_section"] <= ts.z <= p["last_section"]],
        transformList=resolvedtiles_obj.transforms)

    fr, draft_rts = bigfeta.create_CSR_A_fromobjects(
        rts, matches_obj, p["transformation"],
        [],  # default empty transform_apply
        p["regularization"], p["matrix_assembly"],
        return_draft_resolvedtiles=True)

    sol = solve.solve(
        fr["A"], fr["weights"], fr["reg"], fr["x"], fr["rhs"])

    assert np.all(np.array(sol['precision']) < 1e-7)
    assert np.all(np.array(sol['error']) < 200)


@pytest.mark.parametrize(
    "transform, fullsize, order",
    [("AffineModel", True, 0),
     ("AffineModel", False, 0),
     ("SimilarityModel", False, 0),
     ("Polynomial2DTransform", False, 0),
     ("Polynomial2DTransform", False, 1)])
def test_different_solvers(resolvedtiles_obj, matches_obj,
                           transform, fullsize, order):
    solve_funcs = [v for k, v in solve.solve_funcs.items() if k != "default"]
    p = copy.deepcopy(montage_parameters)
    p['transformation'] = transform
    p['fullsize'] = fullsize
    p['poly_order'] = order
    p["regularization"]["poly_factors"] = None

    solve_results = []
    for solve_func in solve_funcs:
        copied_rts = copy.deepcopy(resolvedtiles_obj)
        copied_matches = copy.deepcopy(matches_obj)
        rts = renderapi.resolvedtiles.ResolvedTiles(
            tilespecs=[ts for ts in copied_rts.tilespecs
                       if p["first_section"] <= ts.z <= p["last_section"]],
            transformList=copied_rts.transforms)

        fr, draft_rts = bigfeta.create_CSR_A_fromobjects(
            rts, copied_matches, p["transformation"],
            [],  # default empty transform_apply
            p["regularization"], p["matrix_assembly"],
            return_draft_resolvedtiles=True)

        sol = solve_func(
            fr["A"], fr["weights"], fr["reg"], fr["x"], fr["rhs"])

        solve_results.append(sol)
    for sol1, sol2 in itertools.combinations(solve_results, 2):
        assert np.allclose(sol1["x"], sol2["x"], rtol=5e-4)


def test_affine_sweep(resolvedtiles_obj, matches_obj):
    p = copy.deepcopy(montage_parameters)

    rts = renderapi.resolvedtiles.ResolvedTiles(
        tilespecs=[ts for ts in resolvedtiles_obj.tilespecs
                   if p["first_section"] <= ts.z <= p["last_section"]],
        transformList=resolvedtiles_obj.transforms)

    solver_results = regularization_sweep.sweep_parameters(
        rts, matches_obj, p["transformation"],
        [],  # default empty transform_apply
        p["regularization"], p["matrix_assembly"],
        processes=2,
        params_iter=[(
            p["regularization"]["default_lambda"],
            p["regularization"]["translation_factor"]
        )])
    sol = solver_results[0].solution_dict

    assert np.all(np.array(sol['precision']) < 1e-7)
    assert np.all(np.array(sol['error']) < 200)


def test_affine_sweep_logarithmic(resolvedtiles_obj, matches_obj):
    p = copy.deepcopy(montage_parameters)

    rts = renderapi.resolvedtiles.ResolvedTiles(
        tilespecs=[ts for ts in resolvedtiles_obj.tilespecs
                   if p["first_section"] <= ts.z <= p["last_section"]],
        transformList=resolvedtiles_obj.transforms)

    solver_results = regularization_sweep.sweep_parameters_logarithmic(
        rts, matches_obj, p["transformation"],
        [],  # default empty transform_apply
        p["regularization"], p["matrix_assembly"],
        processes=4,
        lambda_log_range=(-1, 8),
        num_lambda=2,
        transfac_log_range=(-7, -3),
        num_transfac=2)

    assert len(solver_results) == 4


def test_filter_solvestats():
    good_solvestat = regularization_sweep.SolveStats(
        scale_mean=np.array((1, 1)),
        scale_median=np.array((1, 1)),
        scale_mins=np.array((1, 1)),
        scale_maxs=np.array((1, 1)),
        scale_stdevs=np.array((0, 0)),
        scale_mads=np.array((0, 0)),
        err_means=np.array((0, 0)),
        err_stds=np.array((0, 0)),
        error=np.array((0, 0)),
    )

    bad_solvestat_mad = regularization_sweep.SolveStats(
        scale_mean=good_solvestat.scale_mean,
        scale_median=good_solvestat.scale_median,
        scale_mins=good_solvestat.scale_mins,
        scale_maxs=good_solvestat.scale_maxs,
        scale_stdevs=good_solvestat.scale_stdevs,
        scale_mads=np.array((0.1, 0.2)),
        err_means=good_solvestat.err_means,
        err_stds=good_solvestat.err_stds,
        error=good_solvestat.error
    )

    bad_solvestat_error = regularization_sweep.SolveStats(
        scale_mean=good_solvestat.scale_mean,
        scale_median=good_solvestat.scale_median,
        scale_mins=good_solvestat.scale_mins,
        scale_maxs=good_solvestat.scale_maxs,
        scale_stdevs=good_solvestat.scale_stdevs,
        scale_mads=good_solvestat.scale_mads,
        err_means=good_solvestat.err_means,
        err_stds=good_solvestat.err_stds,
        error=np.array((1000, 1000))
    )

    filtered_stats = regularization_sweep.filter_solvestats(
        [
            good_solvestat,
            bad_solvestat_error,
            bad_solvestat_mad
        ],
        mad_target=(0.003, 0.003),
        mad_cutoff=(0.005, 0.007),
        mad_step_size=(0.001, 0.001),
        scalemin_cutoff=(0, 0),
        min_inliers=1,
        scaled_error_cutoff=4
    )

    assert len(filtered_stats) == 1

    filtered_stats = regularization_sweep.filter_solvestats(
        [
            bad_solvestat_error,
            bad_solvestat_mad
        ],
        mad_target=(0.003, 0.003),
        mad_cutoff=(0.005, 0.007),
        mad_step_size=(0.001, 0.001),
        scalemin_cutoff=(0, 0),
        min_inliers=1,
        scaled_error_cutoff=4
    )

    assert not filtered_stats
