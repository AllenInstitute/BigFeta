User Guide
==========

Use cases
---------
- **montage** (or "stitching" or "registration"): stitching of images in a single section of tissue. Even for a section comprised of thousands of images, these solves can generally be performed on a single workstation or compute node in under a minute.
- **rough alignment**: stitching of downsampled images in 3D. Typically each section is a single downsampled image. Even for thousands of downsampled sections, these solves can generally be performed on a single workstation or compute node in a few minutes. Compared to montages the computation time is increased due to the underlying database structure. Compared to fine alignment, this is just a very small 3D solve.
- **fine alignment**: these solves comprise thousands to millions of images across many z values. The solves are memory-limited and the size of the solve determines whether they can be run on a single node, need distribution across multiple nodes, and, potentially, whether one should choose a direct or iterative distributed solver.

The size of a solve is determined by the number of images, the number of point correspondences derived between the images, and, the number of degrees-of-freedom attributed to each image.

Important Dependencies
----------------------

- `render-python <https://github.com/fcollman/render-python>`_ This package provides underlying python interfaces to `render <https://github.com/saalfeldlab/render>`_
- `argschema <https://github.com/AllenInstitute/argschema>`_ This package provides the means for setting the many parameters that are inputs to this solver.
- `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ For single-node solves, this package is used for factorization and solving.
- `PETSc <https://www.mcs.anl.gov/petsc/>`_ This is a large package that supports distributed linear algebra and many different preconditioners and solvers.
- `Singularity <http://singularity.lbl.gov/archive/docs/v2-3/index.html>`_ The compilation and use of PETSc is a steep learning curve. This repo includes a PETSc build and a solver compilation in singularity containers for ease-of-use.

Detailed Argument Descriptions
------------------------------

see :ref:`bigfeta-schema-label`

Distributed Usage
-----------------
.. toctree::
   :maxdepth: 1

   ../distributed_rm


