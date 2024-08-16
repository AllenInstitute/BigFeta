#!/usr/bin/env python

from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        String, Int, Boolean, Nested, Float,
        List, InputFile, OutputFile)
import marshmallow as mm
import numpy as np


class db_params(DefaultSchema):
    owner = String(
        default='',
        required=False,
        description='render or mongo owner')
    project = String(
        default='',
        required=False,
        description='render or mongo project')
    name = List(
        String,
        cli_as_single_argument=True,
        required=False,
        many=True,
        description='render or mongo collection name')
    host = String(
        required=False,
        description='render host')
    port = Int(
        default=8080,
        required=False,
        description='render port')
    mongo_host = String(
        default='em-131fs',
        required=False,
        description='mongodb host')
    mongo_port = Int(
        default=27017,
        required=False,
        description='mongodb port')
    mongo_userName = String(
        default='',
        required=False,
        description='mongo user name')
    mongo_authenticationDatabase = String(
        default='',
        required=False,
        description='mongo admin db')
    mongo_password = String(
        default='',
        required=False,
        description='mongo pwd')
    db_interface = String(
        default='mongo',
        validator=mm.validate.OneOf(['render', 'mongo', 'file']),
        description=("render: read or write via render\n"
                     "mongo: read or write via pymongo\n"
                     "file: read or write to file"))
    client_scripts = String(
        default=("/allen/aibs/pipeline/image_processing/"
                 "volume_assembly/render-jars/production/scripts"),
        required=False,
        description='see renderapi.render.RenderClient')
    memGB = String(
        required=False,
        default='5G',
        description='see renderapi.render.RenderClient')
    validate_client = Boolean(
        required=False,
        default=False,
        description='see renderapi.render.RenderClient')

    @mm.pre_load
    def tolist(self, data):
        if 'name' in data:
            if not isinstance(data['name'], list):
                data['name'] = [data['name']]


class hdf5_options(DefaultSchema):
    output_dir = String(
        default="",
        description="path to directory to hold hdf5 output.")
    chunks_per_file = Int(
        default=5,
        description=("how many sections with upward-looking"
                     " cross section to write per .h5 file"))


class matrix_assembly(DefaultSchema):
    depth = List(
        Int,
        cli_as_single_argument=True,
        default=[0, 1, 2],
        required=False,
        description='depth in z for matrix assembly point matches')
    explicit_weight_by_depth = List(
        Float,
        cli_as_single_argument=True,
        default=None,
        missing=None,
        description='explicitly set solver weights by depth')

    @mm.pre_load
    def tolist(self, data):
        if not isinstance(data['depth'], list):
            data['depth'] = np.arange(0, data['depth'] + 1).tolist()

    @mm.post_load
    def check_explicit(self, data):
        if data['explicit_weight_by_depth'] is not None:
            if (
                    len(data['explicit_weight_by_depth']) !=
                    len(data['depth'])):
                raise mm.ValidationError(
                        "matrix_assembly['explicit_weight_by_depth'] "
                        "must be the same length as matrix_assembly['depth']")
    cross_pt_weight = Float(
        default=1.0,
        required=False,
        description='weight of cross section point matches')
    montage_pt_weight = Float(
        default=1.0,
        required=False,
        description='weight of montage point matches')
    npts_min = Int(
        default=5,
        missing=5,
        required=False,
        description='disregard any tile pairs with fewer points than this')
    npts_max = Int(
        default=500,
        required=False,
        description='truncate any tile pairs to this size')
    choose_random = Boolean(
        default=False,
        required=False,
        description=("choose random pts to meet npts_max"
                     " vs. just first npts_max"))
    inverse_dz = Boolean(
        default=True,
        required=False,
        description='cross section point match weighting fades with z')


class regularization(DefaultSchema):
    default_lambda = Float(
        default=0.005,
        description='common regularization value')
    translation_factor = Float(
        default=0.005,
        description=("translation regularization factor."
                     " multiplies default_lambda"))
    poly_factors = List(
        Float,
        required=False,
        default=None,
        missing=None,
        cli_as_single_argument=True,
        description=("List of regularization factors by order "
                     "(0, 1, ...,  n) will override other settings "
                     "for Polynomial2DTransform. multiplies default_lambda"))
    thinplate_factor = Float(
        required=False,
        default=1e-5,
        missing=1e-5,
        description=("regularization factor for thin plate spline "
                     "control points. multiplies default_lambda."))


class input_db(db_params):
    input_file = InputFile(
        required=False,
        missing=None,
        default=None,
        description=("json or json.gz serialization of input"))

    @mm.post_load
    def validate_file(self, data):
        if data['db_interface'] == 'file':
            if data['input_file'] is None:
                raise mm.ValidationError("with db_interface 'file', "
                                         "'input_file' must be a file")


class pointmatch(input_db):
    collection_type = String(
        default='pointmatch',
        description="'stack' or 'pointmatch'")


class input_stack(input_db):
    collection_type = String(
        default='stack',
        description="'stack' or 'pointmatch'")
    use_rest = Boolean(
        default=False,
        description="passed as arg in import_tilespecs_parallel")

    @mm.post_load
    def validate_data(self, data):
        if 'name' in data:
            if len(data['name']) != 1:
                raise mm.ValidationError("only one input or output "
                                         "stack name is allowed")


class output_stack(db_params):
    output_file = OutputFile(
        required=False,
        missing=None,
        default=None,
        description=("json or json.gz serialization of input stack"
                     "ResolvedTiles."))
    compress_output = Boolean(
        required=False,
        default=True,
        missing=True,
        description=("if writing file, compress with gzip."))
    collection_type = String(
        default='stack',
        description="'stack' or 'pointmatch'")
    use_rest = Boolean(
        default=False,
        description=("passed as kwarg to "
                     "renderapi.client.import_tilespecs_parallel"))

    @mm.post_load
    def validate_file(self, data):
        if data['db_interface'] == 'file':
            if data['output_file'] is None:
                raise mm.ValidationError("with db_interface 'file', "
                                         "'output_file' must be a file")

    @mm.post_load
    def validate_data(self, data):
        if 'name' in data:
            if len(data['name']) != 1:
                raise mm.ValidationError("only one input or output "
                                         "stack name is allowed")


class BigFetaSchema(ArgSchema):
    """The input schema used by the BigFeta solver
    """

    first_section = Int(
        required=True,
        description='first section for matrix assembly')
    last_section = Int(
        required=True,
        description='last section for matrix assembly')
    n_parallel_jobs = Int(
        default=4,
        required=False,
        description=("number of parallel jobs that will run for "
                     "retrieving tilespecs, assembly from pointmatches, "
                     "and import_tilespecs_parallel"))
    processing_chunk_size = Int(
        default=1,
        required=False,
        description=("number of pairs per multiprocessing job. can help "
                     "parallelizing pymongo calls."))
    solve_type = String(
        default='montage',
        required=False,
        description='Solve type options (montage, 3D)',
        validator=mm.validate.OneOf(['montage', '3D']))
    close_stack = Boolean(
        default=True,
        required=False,
        description='Set output stack to state COMPLETE?')
    overwrite_zlayer = Boolean(
        default=True,
        required=False,
        description='delete section before import tilespecs?')
    profile_data_load = Boolean(
        default=False,
        description="module will raise exception after timing tilespec read")
    transformation = String(
        default='AffineModel',
        validate=mm.validate.OneOf([
            'AffineModel', 'SimilarityModel', 'Polynomial2DTransform',
            'affine', 'rigid', 'affine_fullsize', 'RotationModel',
            'TranslationModel', 'ThinPlateSplineTransform']),
        description="transformation to use for the solve")
    fullsize_transform = Boolean(
        default=False,
        description='use fullsize affine transform')
    poly_order = Int(
        default=2,
        required=False,
        description='order of polynomial transform.')
    output_mode = String(
        default='none',
        validate=mm.validate.OneOf(['none', 'hdf5', 'stack']),
        description=("none: just solve and show logging output\n"
                     "hdf5: assemble to hdf5_options.output_dir\n"
                     "stack: write to output stack"))
    assemble_from_file = String(
        default='',
        description=("path to an hdf5 file for solving from hdf5 output."
                     "mainly for testing purposes. hdf5 output usually to "
                     "be solved by external solver"))
    ingest_from_file = String(
        default='',
        description='path to an hdf5 file output from the external solver.')
    render_output = String(
        default='null',
        description=("anything besides the default will "
                     "show all the render stderr/stdout"))
    input_stack = Nested(
            input_stack,
            description=("specifies the origin of the tilespecs."))
    output_stack = Nested(
            output_stack,
            description=("specifies the destination of the tilespecs."))
    pointmatch = Nested(
            pointmatch,
            description=("specifies the origin of the point correspondences"))
    hdf5_options = Nested(
            hdf5_options,
            description=("options invoked if output_mode is \"hdf5\""))
    matrix_assembly = Nested(
            matrix_assembly,
            description=("options that control which correspondences are"
                         " included in the matrix equation and their weights"))
    regularization = Nested(
            regularization,
            description=("options that contol the regularization of different"
                         " types of variables in the solve"))
    transform_apply = List(
            Int,
            default=[],
            missing=[],
            description=("tilespec.tforms[i].tform() for i in transform_apply "
                         "will be performed on the matches before matrix "
                         "assembly."),
            cli_as_single_argument=True
    )
    solve_implementation = String(
        required=False, default="default", missing="default",
        validator=mm.validate.OneOf(["petsc", "scipy", "default"]),
        description=(
            "solve type to use: petsc, scipy, or default to one of the two"))

    @mm.post_load
    def validate_data(self, data):
        if (data['regularization']['poly_factors'] is not None) & \
                (data['transformation'] == 'Polynomial2DTransform'):
            n = len(data['regularization']['poly_factors'])
            if n != data['poly_order'] + 1:
                raise mm.ValidationError(
                        "regularization.poly_factors must be a list"
                        " of length poly_order + 1")


class BigFetaPlotSchema(BigFetaSchema):
    z1 = Int(
        default=1000,
        description='first z for plot')
    z2 = Int(
        default=1000,
        description='second z for plot')
    zoff = Int(
        default=0,
        description='z offset between pointmatches and tilespecs')
    plot = Boolean(
        default=True,
        description='make a plot, otherwise, just text output')
    savefig = Boolean(
        default=False,
        description='save to a pdf')
    plot_dir = String(
        default='./')
    threshold = Float(
        default=5.0,
        description='threshold for colors in residual plot [pixels]')
    density = Boolean(
        default=True,
        description=("whether residual plot is density "
                     " (for large numbers of points) or just points"))
