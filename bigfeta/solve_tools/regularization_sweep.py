#!/usr/bin/env python

import concurrent.futures
import dataclasses
import itertools

import numpy
import renderapi

import bigfeta.bigfeta
import bigfeta.utils

import bigfeta.qctools.statistics


@dataclasses.dataclass
class SolveStats:
    """Statistics derived from a solve"""
    scale_mean: numpy.ndarray
    scale_median: numpy.ndarray
    scale_mins: numpy.ndarray
    scale_maxs: numpy.ndarray
    scale_stdevs: numpy.ndarray
    scale_mads: numpy.ndarray
    err_means: numpy.ndarray
    err_stds: numpy.ndarray
    error: numpy.ndarray


@dataclasses.dataclass
class ParameterResult:
    """ParameterResult from sweep"""
    reg_d: dict
    result_resolvedtiles: renderapi.resolvedtiles.ResolvedTiles
    solution_dict: dict
    solve_statistics: SolveStats


def solve_with_regd_for_parameter_result(draft_resolvedtiles, fr, reg_d):
    l_output_rts = bigfeta.utils.copy_resolvedtiles(draft_resolvedtiles)
    l_reg = bigfeta.utils.tilespecs_regularization_from_reg_d(
        l_output_rts.tilespecs, reg_d)
    l_sol = bigfeta.solve.solve(
        fr["A"], fr["weights"], l_reg, fr["x"], fr["rhs"])

    bigfeta.utils.update_tilespecs(l_output_rts, l_sol["x"])

    # TODO can get scales from l_sol and resolvedtiles r.t. copying tilespecs
    scales = numpy.array([ts.tforms[-1].scale
                          for ts in l_output_rts.tilespecs])
    mean = numpy.mean(scales, axis=0)
    median = numpy.median(scales, axis=0)
    mins = numpy.min(scales, axis=0)
    maxs = numpy.max(scales, axis=0)
    stdevs = numpy.std(scales, axis=0)
    mads = bigfeta.qctools.statistics.get_2d_filtered_mad(
        scales, axis=0)

    err_means = [e[0] for e in l_sol["err"]]
    err_stds = [e[1] for e in l_sol["err"]]
    errors = numpy.array(l_sol["error"]) / len(l_output_rts.tilespecs)

    return ParameterResult(
        reg_d, l_output_rts, l_sol, 
        SolveStats(
            mean, median, mins,
            maxs, stdevs, mads,
            err_means, err_stds, errors))


def filter_solvestats(solvestats, associated_values=None,
                      mad_target=(0.003, 0.003), mad_cutoff=(0.005, 0.007),
                      mad_step_size=(0.001, 0.001),
                      scalemin_cutoff=(0, 0),
                      min_inliers=1, max_inliers=None,
                      scaled_error_cutoff=None):
    associated_values = (
        itertools.repeat(None)
        if associated_values is None
        else associated_values)

    # list of targets sorted as the product of x and y cutoffs
    targets_to_check = sorted(itertools.product(
        numpy.arange(
            mad_target[0], mad_cutoff[0] + mad_step_size[0], mad_step_size[0]),
        numpy.arange(
            mad_target[1], mad_cutoff[1] + mad_step_size[1], mad_step_size[1])
    ), key=lambda x: x[0] * x[1])

    for tgt in targets_to_check:
        inliers = []
        for ss, ss_values in zip(solvestats, associated_values):
            if (
                    ss.scale_mads[0] <= tgt[0] and
                    ss.scale_mads[1] <= tgt[1]):
                err_param = numpy.linalg.norm(ss.error / ss.scale_mean)
                scale_param = numpy.linalg.norm(ss.scale_mean)
                if 1.4 < scale_param:  # TODO unnecessary
                    if (
                            scalemin_cutoff[0] < ss.scale_mins[0] and
                            scalemin_cutoff[1] < ss.scale_mins[1]):
                        if scaled_error_cutoff is not None:
                            if err_param > scaled_error_cutoff:
                                continue
                        inliers.append((ss, ss_values))
        if len(inliers) >= min_inliers:
            # return sorted with minimum residual first
            if max_inliers is None:
                max_inliers = len(inliers)
            return sorted(
                inliers, key=lambda x: numpy.linalg.norm(
                    x[0].error / x[0].scale_mean))[:max_inliers]


def sweep_parameters(rts, matches, transform_name="affine",
                     transform_apply=None,
                     regularization_dict=None, matrix_assembly_dict=None,
                     order=2, fullsize=False,
                     processes=1, params_iter=None):
    create_CSR_A_input = (
        rts, matches, transform_name,
        ([] if transform_apply is None else transform_apply),
        ({} if regularization_dict is None else regularization_dict),
        matrix_assembly_dict, order, fullsize)

    fr, draft_resolvedtiles = bigfeta.bigfeta.create_CSR_A_fromobjects(
        *create_CSR_A_input, return_draft_resolvedtiles=True)

    sweep_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as e:
        fut_to_regd = {}
        for lambda_val, transfac_val in params_iter:
            reg_d = dict(regularization_dict, **{
                "default_lambda": lambda_val,
                "translation_factor": transfac_val
            })
            fut_to_regd[e.submit(
                solve_with_regd_for_parameter_result, draft_resolvedtiles,
                fr, reg_d)] = reg_d
        for fut in concurrent.futures.as_completed(fut_to_regd.keys()):
            parameter_result = fut.result()
            sweep_results.append(parameter_result)

    return sweep_results


def sweep_parameters_logarithmic(
        *args, lambda_log_range=(-1, 8), num_lambda=15,
        transfac_log_range=(-7, -3), num_transfac=15, **kwargs):
    lambda_list = numpy.logspace(
        *lambda_log_range, num=num_lambda)
    transfac_list = numpy.logspace(
        *transfac_log_range, num=num_transfac)

    params_iter = itertools.product(lambda_list, transfac_list)

    return sweep_parameters(*args, params_iter=params_iter, **kwargs)
