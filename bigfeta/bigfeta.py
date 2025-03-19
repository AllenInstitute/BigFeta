import copy
import concurrent.futures
import json
import logging
import os
import sys
import time
import warnings

import argschema
import h5py
import numpy as np
import renderapi
import scipy.sparse as sparse
from scipy.sparse import csr_matrix

from .schemas import BigFetaSchema
from . import utils
from . import jsongz
from . import solve

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.resetwarnings()

logger = logging.getLogger(__name__)
default_logger = logger


def calculate_processing_chunk(fargs):
    """job to parallelize for creating a sparse matrix block
    and associated vectors from a pair of sections

    Parameters
    ----------
    fargs : List
        serialized inputs for multiprocessing job

    Returns
    -------
    chunk : dict
        keys are 'zlist', 'block', 'weights', and 'rhs'

    """
    t0 = time.time()
    # set up for calling using multiprocessing pool
    chunks = []
    dbconnection = utils.make_dbconnection(args['pointmatch'])
    for farg in fargs:
        [pair, args, tspecs, tforms, col_ind, ncol] = farg

        tile_ids = np.array([t.tileId for t in tspecs])

        sorter = np.argsort(tile_ids)


        # get point matches
        nmatches = utils.get_matches(
                pair['section1'],
                pair['section2'],
                args['pointmatch'],
                dbconnection)

        # extract IDs for fast checking
        pid_set = set(m['pId'] for m in nmatches)
        qid_set = set(m['qId'] for m in nmatches)

        tile_set = set(tile_ids)

        pid_set.intersection_update(tile_set)
        qid_set.intersection_update(tile_set)

        matches = [m for m in nmatches if m['pId']
                   in pid_set and m['qId'] in qid_set]
        del nmatches

        if len(matches) == 0:
            logger.debug(
                "no tile pairs in "
                "stack for pointmatch groupIds %s and %s" % (
                    pair['section1'], pair['section2']))
            return None

        pids = np.array([m['pId'] for m in matches])
        qids = np.array([m['qId'] for m in matches])

        logger.debug(
                "loaded %d matches, using %d, "
                "for groupIds %s and %s in %0.1f sec "
                "using interface: %s" % (
                    len(pid_set.union(qid_set)),
                    len(matches),
                    pair['section1'],
                    pair['section2'],
                    time.time() - t0,
                    args['pointmatch']['db_interface']))

        # for the given point matches, these are the indices in tile_ids
        # these determine the column locations in A for each tile pair
        # this is a fast version of np.argwhere() loop
        pinds = sorter[np.searchsorted(tile_ids, pids, sorter=sorter)]
        qinds = sorter[np.searchsorted(tile_ids, qids, sorter=sorter)]

        tilepair_weightfac = tilepair_weight(
            pair['z1'],
            pair['z2'],
            args['matrix_assembly'])

        wts = []
        pblocks = []
        qblocks = []
        rhss = []
        for k, match in enumerate(matches):

            match = utils.transform_match(
                match,
                tspecs[pinds[k]],
                tspecs[qinds[k]],
                args['transform_apply'],
                tforms)

            pblock, qblock, weights, rhs = utils.blocks_from_tilespec_pair(
                    tspecs[pinds[k]],
                    tspecs[qinds[k]],
                    match,
                    col_ind[pinds[k]],
                    col_ind[qinds[k]],
                    ncol,
                    args['matrix_assembly'])

            if pblock is None:
                continue

            pblocks.append(pblock)
            qblocks.append(qblock)
            wts.append(weights * tilepair_weightfac)
            rhss.append(rhs)

        chunk = {}
        chunk['zlist'] = np.array([pair['z1'], pair['z2']])
        chunk['block'] = sparse.vstack(pblocks) - sparse.vstack(qblocks)
        chunk['weights'] = np.concatenate(wts)
        chunk['rhs'] = np.concatenate(rhss)

        chunks.append(chunk)

    return chunks


def tilepair_weight(z1, z2, matrix_assembly):
    """get weight factor between two tilepairs

    Parameters
    ----------
    z1 : int or float
        z value for first section
    z2 : int or float
        z value for second section
    matrix_assembly : dict
        bigfeta.schemas.matrix assembly


    Returns
    -------
    tp_weight : float
        weight factor

    """
    if matrix_assembly.get('explicit_weight_by_depth') is not None:
        ind = matrix_assembly['depth'].index(int(np.abs(z1 - z2)))
        tp_weight = matrix_assembly['explicit_weight_by_depth'][ind]
    else:
        if z1 == z2:
            tp_weight = matrix_assembly['montage_pt_weight']
        else:
            tp_weight = matrix_assembly['cross_pt_weight']
            if matrix_assembly['inverse_dz']:
                tp_weight = tp_weight/(np.abs(z2 - z1) + 1)
    return tp_weight


def _filter_to_products(contains_products, iterables_it):
    """return list of values filtered to only those values which
        are in the product of a list of iterables.

    Parameters
    ----------
    contains_products : list
        list to be filtered to values containing products
    iterables_it : iterable of iterables
        iterable of iterables as in inputs to itertools.product

    Returns
    -------
    filtered_products : list
        list of values in contains_products which are products of the iterables
    """
    return [p for p in contains_products
            if all([p[i] in it for i, it in enumerate(iterables_it)])]


def create_CSR_A_fromprepared(resolvedtiles, matches, regularization_dict,
                              matrix_assembly_dict, transform_apply=[],
                              return_draft_resolvedtiles=False,
                              copy_resolvedtiles=True,
                              results_in_chunks=False):
    """Assembles results as in BigFeta.create_CSR_A from prepared resolvedtiles

    Parameters
    ----------
    resolvedtiles : renderapi.resolvedtiles.ResolvedTiles
        resolvedtiles object containing tilespecs with transforms ending in an AlignerTransform
    matches : list of dict
        pointmatches in render format
    regularization_dict : dict
        regularization parameters (see Schema)
    matrix_assembly_dict : dict
        matrix assembly parameters (see Schema)
    return_draft_resolvedtiles : boolean
        whether to return draft_resolvedtiles -- used to apply transforms
    copy_resolvedtiles : boolean
        whether to make copy of the input resolvedtiles or process in place
    results_in_chunks : boolean
        whether to return another dicitonary item "A_weights_rhs_z_chunks"
        with chunked results (for writing to hdf5)

    Returns
    -------
    func_result : dict
        dictionary with keys "x", "reg", "A", "weights", "rhs"
    draft_resolvedtiles : renderapi.resolvedtiles.ResolvedTiles
        resolvedtiles object with AlignerTransforms used to derive result
    """
    func_result = {}

    draft_resolvedtiles = (
        copy.deepcopy(resolvedtiles) if copy_resolvedtiles
        else resolvedtiles)
    # this emulates the pre_load behavior of the schema
    depth = (
        matrix_assembly_dict["depth"] if isinstance(
           matrix_assembly_dict["depth"], list)
        else list(range(matrix_assembly_dict["depth"]+1)))

    pairs = utils.determine_zvalue_pairs(
        draft_resolvedtiles, depth)

    # the column indices for each tilespec
    col_ind = np.cumsum(
        np.hstack((
            [0],
            [t.tforms[-1].DOF_per_tile
             for t in draft_resolvedtiles.tilespecs])))
    ncol = col_ind.max()

    tId_to_col_idx = {
        ts.tileId: col_ind[i]
        for i, ts in enumerate(draft_resolvedtiles.tilespecs)}
    tId_to_tspec = {
        ts.tileId: ts for ts in draft_resolvedtiles.tilespecs}

    # map like {(pGroupId, qGroupId): {(pId, qId): match}}
    group_id_tree = utils.matches_to_match_id_tree(matches)

    # map like {{z, sectionId}: {tileId: tilespec}}
    z_section_tree = utils.tilespecs_to_z_section_tree(draft_resolvedtiles.tilespecs)

    # NOTE possible that chunking is not required structurally or for performance
    chunks = []

    for pair in pairs:
        try:
            id_tree = group_id_tree[(pair["section1"], pair["section2"])]
        except KeyError:
            logger.debug(
                "no tile pairs for pointmatch groupIds: {} and {}".format(
                    pair["section1"], pair["section2"]))
            continue

        tspecs1 = z_section_tree[(pair["z1"], pair["section1"])].keys()
        tspecs2 = z_section_tree[(pair["z2"], pair["section2"])].keys()

        tilepair_weightfac = tilepair_weight(
            pair["z1"], pair["z2"], matrix_assembly_dict)

        valid_match_keys = _filter_to_products(
            id_tree.keys(), [tspecs1, tspecs2])

        wts = []
        pblocks = []
        qblocks = []
        rhss = []
        for (pId, qId), match in ((k, id_tree[k]) for k in valid_match_keys):
            tformed_match = utils.transform_match(
                match, tId_to_tspec[pId], tId_to_tspec[qId], transform_apply,
                draft_resolvedtiles.transforms)

            pblock, qblock, weights, rhs = utils.blocks_from_tilespec_pair(
                tId_to_tspec[pId],
                tId_to_tspec[qId],
                tformed_match,
                tId_to_col_idx[pId],
                tId_to_col_idx[qId],
                ncol,
                matrix_assembly_dict)
            if pblock is None:
                continue

            pblocks.append(pblock)
            qblocks.append(qblock)
            wts.append(weights * tilepair_weightfac)
            rhss.append(rhs)

        # some use cases will not poduce blocks
        if not pblocks:
            logger.debug(
                "cannot assemble matrix block "
                "for pointmatch groupIds: {} and {}".format(
                    pair["section1"], pair["section2"]
                ))
            continue

        chunk = {
            'zlist': np.array([pair['z1'], pair['z2']]),
            'block': sparse.vstack(pblocks) - sparse.vstack(qblocks),
            'weights': np.concatenate(wts),
            'rhs': np.concatenate(rhss)
            }
        chunks.append(chunk)

    func_result["x"], reg = map(list, zip(*(
        (ts.tforms[-1].to_solve_vec(),
         ts.tforms[-1].regularization(regularization_dict))
        for ts in draft_resolvedtiles.tilespecs)))

    if len(func_result['x']) == 0:
        raise utils.BigFetaException(
                "no matrix was assembled. Likely scenarios: "
                "your tilespecs and pointmatches do not key "
                " to each other in group or tileId. Or, your match "
                " collection is empty")

    func_result["x"] = np.concatenate(func_result["x"])
    func_result["reg"] = sparse.diags(
            [np.concatenate(reg)], [0], format='csr')

    func_result["A"], func_result["weights"], func_result["rhs"], _ = (
        utils.concatenate_results(np.array(chunks)))

    if results_in_chunks:
        func_result["A_weights_rhs_z_chunks"] = chunks

    return ((func_result, draft_resolvedtiles)
            if return_draft_resolvedtiles else func_result)


def create_CSR_A_fromobjects(
        resolvedtiles, matches, transform_name,
        transform_apply, regularization_dict, matrix_assembly_dict,
        order=2, fullsize=False,
        return_draft_resolvedtiles=False, copy_resolvedtiles=True,
        results_in_chunks=False):
    """Assembles results as in BigFeta.create_CSR_A from
        resolvedtiles and pointmatches

    Parameters
    ----------
    resolvedtiles : renderapi.resolvedtiles.ResolvedTiles
        resolvedtiles object containing tiles to consider during assembly
    matches : list of dict
        pointmatches in render format
    transform_name : string
        string describing model for which to solve (see Schema)
    transform_apply : list of int
        additional transforms to apply to pointmatches
    regularization_dict : dict
        regularization parameters (see Schema)
    matrix_assembly_dict : dict
        matrix assembly parameters (see Schema)
    order : int
        order for polynomial transform
    fullsize : boolean
        whether to use fullsize matrices
    return_draft_resolvedtiles : boolean
        whether to return draft_resolvedtiles -- used to apply transforms
    copy_resolvedtiles : boolean
        whether to make copy of the input resolvedtiles or process in place
    results_in_chunks : boolean
        whether to return another dicitonary item "A_weights_rhs_z_chunks"
        with chunked results (for writing to hdf5)

    Returns
    -------
    func_result : dict
        dictionary with keys "x", "reg", "A", "weights", "rhs"
    draft_resolvedtiles : renderapi.resolvedtiles.ResolvedTiles
        resolvedtiles object with AlignerTransforms used to derive result
    """
    draft_resolvedtiles = (
        copy.deepcopy(resolvedtiles) if copy_resolvedtiles else resolvedtiles)

    utils.ready_transforms(
        draft_resolvedtiles.tilespecs, transform_name,
        fullsize, order)

    return create_CSR_A_fromprepared(
        draft_resolvedtiles, matches, regularization_dict,
        matrix_assembly_dict, transform_apply=transform_apply,
        return_draft_resolvedtiles=return_draft_resolvedtiles,
        copy_resolvedtiles=False, results_in_chunks=results_in_chunks)


class BigFeta(argschema.ArgSchemaParser):
    default_schema = BigFetaSchema
    renderapi.client.WithPool = \
        renderapi.external.processpools.stdlib_pool.WithThreadPool

    def run(self):
        """main function call for BigFeta solver
        """
        logger.setLevel(self.args['log_level'])
        utils.logger.setLevel(self.args['log_level'])
        t0 = time.time()
        zvals = np.arange(
            self.args['first_section'],
            self.args['last_section'] + 1)

        # the parallel workers will need this stack ready
        if self.args['output_mode'] == 'stack':
            utils.create_or_set_loading(self.args['output_stack'])

        # montage
        if self.args['solve_type'] == 'montage':
            zvals = utils.get_z_values_for_stack(
                    self.args['input_stack'],
                    zvals)
            for z in zvals:
                self.results = self.assemble_and_solve(np.array([z]))

        # 3D
        elif self.args['solve_type'] == '3D':
            self.results = self.assemble_and_solve(zvals)

        if (self.args['output_mode'] == 'stack') & self.args['close_stack']:
            utils.set_complete(self.args['output_stack'])

        logger.info(' total time: %0.1f' % (time.time() - t0))

    def assemble_and_solve(self, zvals):
        """retrieves a ResolvedTiles object from some source
           and then assembles/solves, outputs to hdf5 and/or outputs to an
           output_stack object.

           Parameters
           ----------
           zvals : :class:`numpy.ndarray`
               int or float, z of :class:`renderapi.tilespec.TileSpec`

        """
        t0 = time.time()

        if self.args['ingest_from_file'] != '':
            assemble_result, results = self.assemble_from_hdf5(
                self.args['ingest_from_file'],
                zvals,
                read_data=False)
            results['x'] = assemble_result['x']

        else:
            if self.args['assemble_from_file'] != '':
                assemble_result, _ = self.assemble_from_hdf5(
                    self.args['assemble_from_file'],
                    zvals)
            else:
                # read in the tilespecs
                self.resolvedtiles = utils.get_resolved_tilespecs(
                    self.args['input_stack'],
                    self.args['transformation'],
                    self.args['n_parallel_jobs'],
                    zvals,
                    fullsize=self.args['fullsize_transform'],
                    order=self.args['poly_order'])
                assemble_result = self.assemble_from_db(zvals)

            logger.info(' A created in %0.1f seconds' % (time.time() - t0))

            if self.args['profile_data_load']:
                raise utils.BigFetaException(
                    "exiting after timing profile")

            # solve
            message, results = \
                self.solve_or_not(
                    assemble_result['A'],
                    assemble_result['weights'],
                    assemble_result['reg'],
                    assemble_result['x'],
                    assemble_result['rhs'])
            logger.info('\n' + message)
            del assemble_result['A']

        if results:
            utils.update_tilespecs(
                    self.resolvedtiles,
                    results['x'])
            scales = np.array(
                    [t.tforms[-1].scale
                     for t in self.resolvedtiles.tilespecs])
            smn = scales.mean(axis=0)
            ssd = scales.std(axis=0)
            logger.info("\n scales: %0.2f +/- %0.2f, %0.2f +/- %0.2f" % (
                smn[0], ssd[0], smn[1], ssd[1]))
            if self.args['output_mode'] == 'stack':
                res_for_file = {a: b for a, b in results.items() if a != 'x'}
                self.args['output_stack'] = utils.write_to_new_stack(
                        self.resolvedtiles,
                        self.args['output_stack'],
                        self.args['render_output'],
                        self.args['overwrite_zlayer'],
                        # for file output, these go too
                        self.args,
                        res_for_file)
                if self.args['render_output'] == 'stdout':
                    logger.info(message)
            del assemble_result['x']

        return results

    def assemble_from_hdf5(self, filename, zvals, read_data=True):
        """assembles and solves from an hdf5 matrix assembly
           previously created with output_mode = "hdf5".

           Parameters
           ----------
           zvals : :class:`numpy.ndarray`
               int or float, z of :class:`renderapi.tilespec.TileSpec`

        """
        assemble_result = {}

        with h5py.File(filename, 'r') as f:
            k = 0
            key = 'x'
            assemble_result[key] = []
            while True:
                name = 'x_%d' % k
                if name in f.keys():
                    assemble_result[key].append(f.get(name)[()])
                    k += 1
                else:
                    break

            if len(assemble_result[key]) == 1:
                n = assemble_result[key][0].size
                assemble_result[key] = np.array(
                    assemble_result[key]).flatten().reshape((n, 1))
            else:
                assemble_result[key] = np.transpose(
                    np.array(assemble_result[key]))

            reg = f.get('reg')[()]
            datafile_names = f.get('datafile_names')[()]
            file_args = json.loads(f.get('input_args')[()][0].decode('utf-8'))
            results = {}
            if "results" in f.keys():
                results = json.loads(f.get('results')[()][0].decode('utf-8'))

            r = f.get('resolved_tiles')[()][0]
            r = (r.decode() if not isinstance(r, str) else r)

            rname = os.path.join(
                    os.path.dirname(filename),
                    r)
            self.resolvedtiles = renderapi.resolvedtiles.ResolvedTiles(
                    json=jsongz.load(rname))

            logger.info(
                "\n loaded %d tile specs from %s" % (
                    len(self.resolvedtiles.tilespecs),
                    filename))

            utils.ready_transforms(
                    self.resolvedtiles.tilespecs,
                    file_args['transformation'],
                    file_args['fullsize_transform'],
                    file_args['poly_order'])

        assemble_result['reg'] = sparse.diags([reg], [0], format='csr')

        if read_data:
            data = np.array([]).astype('float64')
            weights = np.array([]).astype('float64')
            indices = np.array([]).astype('int64')
            indptr = np.array([]).astype('int64')
            rhs = [np.array([]), np.array([])]

            fdir = os.path.dirname(filename)
            i = 0
            for fname in datafile_names:
                with h5py.File(
                        os.path.join(
                            fdir, fname.decode('utf-8')), 'r') as f:
                    data = np.append(data, f.get('data')[()])
                    indices = np.append(indices, f.get('indices')[()])
                    if i == 0:
                        indptr = np.append(indptr, f.get('indptr')[()])
                        i += 1
                    else:
                        indptr = np.append(
                            indptr,
                            f.get('indptr')[()][1:] + indptr[-1])
                    weights = np.append(weights, f.get('weights')[()])

                    k = 0
                    while True:
                        name = 'rhs_%d' % k
                        if name in f.keys():
                            rhs[k] = np.append(rhs[k], f.get(name)[()])
                            k += 1
                        else:
                            break
                    logger.info('  %s read' % fname)

            assemble_result['A'] = csr_matrix((data, indices, indptr))
            assemble_result['rhs'] = rhs[0].reshape(-1, 1)
            if rhs[1].size > 0:
                assemble_result['rhs'] = np.hstack((
                    assemble_result['rhs'],
                    rhs[1].reshape(-1, 1)))
            assemble_result['weights'] = sparse.diags(
                    [weights], [0], format='csr')

        return assemble_result, results

    def assemble_from_db(self, zvals):
        """assembles a matrix from a pointmatch source given
           the already-retrieved ResolvedTiles object. Then solves
           or outputs to hdf5.

           Parameters
           ----------
           zvals :
               int or float, z of :class:`renderapi.tilespec.TileSpec`

        """
        # create A matrix in compressed sparse row (CSR) format
        pairs = utils.determine_zvalue_pairs(
            self.resolvedtiles,
            self.args['matrix_assembly']['depth'])

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args["n_parallel_jobs"]) as e:

            matches_fut = [
                e.submit(
                    utils.get_matches,
                    pair['section1'],
                    pair['section2'],
                    self.args['pointmatch'],
                    utils.make_dbconnection(self.args['pointmatch']))
                for pair in pairs]

            matches = []
            for f in concurrent.futures.as_completed(matches_fut):
                matches += f.result()

        CSR_A = create_CSR_A_fromobjects(
            self.resolvedtiles, matches, self.args["transformation"],
            self.args["transform_apply"], self.args["regularization"],
            self.args["matrix_assembly"], self.args["poly_order"],
            self.args["fullsize_transform"], copy_resolvedtiles=False,
            results_in_chunks=(self.args["output_mode"] == "hdf5"))

        assemble_result = {}
        assemble_result['A'] = CSR_A.pop('A')
        assemble_result['weights'] = CSR_A.pop('weights')
        assemble_result['reg'] = CSR_A.pop('reg')
        assemble_result['x'] = CSR_A.pop('x')
        assemble_result['rhs'] = CSR_A.pop('rhs')

        # output the regularization vectors to hdf5 file
        if self.args['output_mode'] == 'hdf5':
            results = np.array(CSR_A["A_weights_rhs_z_chunks"])

            if self.args['hdf5_options']['chunks_per_file'] == -1:
                proc_chunks = [np.arange(results.size)]
            else:
                proc_chunks = np.array_split(
                    np.arange(results.size),
                    np.ceil(
                        results.size /
                        self.args['hdf5_options']['chunks_per_file']))

            metadata = []
            for pchunk in proc_chunks:
                A, w, rhs, z = utils.concatenate_results(results[pchunk])
                if A is not None:
                    fname = self.args['hdf5_options']['output_dir'] + \
                        '/%d_%d.h5' % (z.min(), z.max())
                    metadata.append(
                        utils.write_chunk_to_file(fname, A, w.data, rhs))

            utils.write_reg_and_tforms(
                dict(self.args),
                self.resolvedtiles,
                metadata,
                assemble_result['x'],
                assemble_result['reg'])

        return assemble_result

    def create_CSR_A(self, resolved):  # pragma: no cover
        """distributes the work of reading pointmatches and
           assembling results

        Parameters
        ----------
        resolved : :class:`renderapi.resolvedtiles.ResolvedTiles`
            resolved tiles object from which to create A matrix

        """
        warnings.warn(
            ("create_CSR_A is deprecated.  Use create_CSR_A_fromobjects "
             "or create_CRS_A_fromprepared."),
            DeprecationWarning, stacklevel=2)

        func_result = {
            'A': None,
            'x': None,
            'reg': None,
            'weights': None,
            'rhs': None,
            'metadata': None}

        # the processing will be distributed according to these pairs
        pairs = utils.determine_zvalue_pairs(
                resolved,
                self.args['matrix_assembly']['depth'])

        # the column indices for each tilespec
        col_ind = np.cumsum(
                np.hstack((
                    [0],
                    [t.tforms[-1].DOF_per_tile for t in resolved.tilespecs])))

        fargs = [[
            pair,
            self.args,
            [resolved.tilespecs[k] for k in pair['ind']],
            resolved.transforms,
            col_ind[pair['ind']],
            col_ind.max()] for pair in pairs]

        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        with renderapi.client.WithPool(self.args['n_parallel_jobs']) as pool:
            results =  pool.map(
                        calculate_processing_chunk,
                        list(chunks(fargs, self.args['processing_chunk_size'])))
        results = np.concatenate([i for i in results if i])

        func_result['x'] = []
        reg = []
        for t in np.array(resolved.tilespecs):
            func_result['x'].append(t.tforms[-1].to_solve_vec())
            reg.append(
                    t.tforms[-1].regularization(self.args['regularization']))

        if len(func_result['x']) == 0:
            raise utils.BigFetaException(
                    "no matrix was assembled. Likely scenarios: "
                    "your tilespecs and pointmatches do not key "
                    " to each other in group or tileId. Or, your match "
                    " collection is empty")

        func_result['x'] = np.concatenate(func_result['x'])
        func_result['reg'] = sparse.diags(
                [np.concatenate(reg)], [0], format='csr')

        if self.args['output_mode'] == 'hdf5':
            results = np.array(results)

            if self.args['hdf5_options']['chunks_per_file'] == -1:
                proc_chunks = [np.arange(results.size)]
            else:
                proc_chunks = np.array_split(
                    np.arange(results.size),
                    np.ceil(
                        results.size /
                        self.args['hdf5_options']['chunks_per_file']))

            func_result['metadata'] = []
            for pchunk in proc_chunks:
                A, w, rhs, z = utils.concatenate_results(results[pchunk])
                if A is not None:
                    fname = self.args['hdf5_options']['output_dir'] + \
                        '/%d_%d.h5' % (z.min(), z.max())
                    func_result['metadata'].append(
                        utils.write_chunk_to_file(fname, A, w.data, rhs))

        else:
            func_result['A'], func_result['weights'], func_result['rhs'], _ = \
                    utils.concatenate_results(results)

        return func_result

    def solve_or_not(self, A, weights, reg, x0, rhs):
        """solves or outputs assembly to hdf5 files

        Parameters
        ----------
        A : :class:`scipy.sparse.csr`
            the matrix, N (equations) x M (degrees of freedom)
        weights : :class:`scipy.sparse.csr_matrix`
            N x N diagonal matrix containing weights
        reg : :class:`scipy.sparse.csr_matrix`
            M x M diagonal matrix containing regularizations
        x0 : :class:`numpy.ndarray`
            M x nsolve float constraint values for the DOFs
        rhs : :class:`numpy.ndarray`:
            rhs vector(s)
            N x nsolve float right-hand-side(s)

        Returns
        -------
        message : str
            solver or hdf5 output message for logging
        results : dict
            keys are "x" (the results), "precision", "error"
            "err", "mag", and "time"
        """
        # not
        if self.args['output_mode'] in ['hdf5']:
            message = '*****\nno solve for file output\n'
            message += 'solve from the files you just wrote:\n\n'
            message += 'python '
            for arg in sys.argv:
                message += arg+' '
            message = message + '--assemble_from_file ' + \
                self.args['hdf5_options']['output_dir']
            message = message + ' --output_mode none'
            message += '\n\nor, run it again to solve with no output:\n\n'
            message += 'python '
            for arg in sys.argv:
                message += arg + ' '
            message = message.replace(' hdf5 ', ' none ')
            results = None
        else:
            solve_func = solve.solve_funcs[
                self.args.get("solve_implementation", "default")]
            results = solve_func(A, weights, reg, x0, rhs)
            message = utils.message_from_solve_results(results)

        return message, results


if __name__ == '__main__':  # pragma: no cover
    mod = BigFeta(schema_type=BigFetaSchema)
    mod.run()
