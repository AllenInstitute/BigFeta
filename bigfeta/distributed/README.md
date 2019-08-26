# BigFeta distributed

For distributed solves across multiple compute nodes. This solver is built with the PETSc libraries

https://www.mcs.anl.gov/petsc/index.html

## Systems

This code has been run on NERSC's Cori and on the Allen Insitute cluster.

For Cori, Cray modules make simple work of compiling and running this code. See `makefile_cori` and `cori_example_script`.

For the Allen cluster, one way to run this is via Singularity containers. `Singularity.petsc` is a definition file for an image with compiled PETSc. This is a fairly lengthy process and should not change very often. The build of the image is manually triggered and maintained on Singularity Hub: https://singularity-hub.org/collections/2940. The solver code in this repository is then compiled in another container that builds from the PETSc image. This is `Singularity.petsc_solver`. For an example of how to run this compilation step, look in `.travis.yml` in this repository. For one example of how to use the built singularity image, see `integration_tests/test_hdf5.py`. For another example, see `allen_example_script.pbs`.

## Usage

```
em_solver_cori -input <input file> -output <output_file> <ksp options>
```
or
```
singularity run --bind <external>:<internal> em_distributed.simf -input <input_file> -output <output_file> <ksp options>
```

`ksp options` specify how PETSc should handle the system. For example whether to use a direct or iterative solver, what type of preconditioner to use, what external packages to invoke.

* direct solve with Pastix: `-ksp_type preonly -pc_type lu`

There are many PETSc options, and not all of them are necessarily installed in the Singularity image here.

## File Formats
For transferring distributed A matrices to the distributed solver.
HDFView is a convenient utility for inspecting the contents of hdf5 files. h5py and HDFView sometimes report different object types so we report both here, when necessary.

format of input\_file.h5 (output\_file.h5 should be a copy with x\_0 and x\_1 replaced by the new solution)

    dataset name: datafile_names
        type (h5py):
            object
        type (HDFView):
            String, length = variable, padding = H5T_STR_NULLTERM,
            cset = H5T_CSET_ASCII
        shape:
            (nfile, 1)
        description:
            relative paths of files that contain the distributed A matrix. 

    dataset name: datafile_maxcol
        type:
            int64
        shape:
            (nfile, 1)
        description:
            maximum column index contained in each file.

    dataset name: datafile_mincol
        type:
            int64
        shape:
            (nfile, 1)
        description:
            minimum column index contained in each file.

    dataset name: datafile_nnz
        type:
            int64
        shape:
            (nfile, 1)
        description:
            number of nonzeros in each file

    dataset name: datafile_nrows
        type: 
            int64
        shape:
            (nfile, 1)
        description:
            number of sub-matrix rows in each file

    dataset name: input_args
        type (h5py):
            object
        type (HDFView):
            String, length = variable, padding = H5T_STR_NULLTERM,
            cset = H5T_CSET_ASCII
        shape: (1,)
        description:
            copy of the input args dict used as input to the aligner
            to create this particular hdf5 file

    dataset name: reg
        type:
            float64
        shape:
            (nvar,)
        description:
            regularization vector. One entry per variable.

    dataset name: resolved_tiles
        type (h5py):
            object
        type (HDFView):
            String, length = variable, padding = H5T_STR_NULLTERM,
            cset = H5T_CSET_ASCII
        shape:
            (1,)
        description:
            relative path of resolved tiles (json or json.gz).
            this was easier than trying to embed and encoded dict
            within this file.

    dataset name: solve_list
        type:
            int32
        shape:
            (nsolve, 1)
        desctiption:
            overly explicit way to tell the C solver that there are
            1 or two solves. But, it works.

    dataset name: x_0
        type:
            float64
        shape:
            (nvar,)
        description:
            input variables for first solve,
            constraint vector for regularizations.

    dataset name: x_1
        type:
            float64
        shape:
            (nvar,)
        description:
            input variables for second solve (if present),
            constraint vector for regularizations.

format of z1_z2.h5 (one of the distributed A files. z1 and z2 are the min and max section number represented by the block, used to create unique file names)

    dataset name: data
        type:
           float64
        shape:
           (nnz,)
        description:
           the non-zero sub-matrix entries

    dataset name: indices
        type:
            int64
        shape:
            (nnz, 1)
        description:
            the globally-indexed column indices for
            the entries

    dataset name: indptr
        type:
            int64
        shape:
            (nrow + 1, 1)
        description:
            index ptr for sub-matrix rows.
            See definition of CSR matrix format.
            
    dataset name: rhs_0
        type:
            float64
        shape:
            (nrow,)
        description:
            right hand side for first solve

    dataset name: rhs_1
        type:
            float64
        shape:
            (nrow,)
        description:
            right hand side for second solve

    dataset name: rhs_list
        type:
            int32
        shape:
            (nsolve, 1)
        description:
            overly explicit callout for number of solves.

    dataset name: weights
        type:
            float64
        shape:
            (nrow,)
        description:
            weight sub-vector


## Documentation

https://bigfeta.readthedocs.io/en/latest/index.html

## Author

[Dan Kapner](https://github.com/djkapner) e-mail: danielk@alleninstitute.org
