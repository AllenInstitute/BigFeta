.. BigFeta documentation master file, created by
   sphinx-quickstart on Mon May 20 10:03:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BigFeta's documentation!
=====================================


BigFeta is a scalable linear least squares image stitching and alignment solver that can align millions of images via point correspondences. The solver runs on systems from individual workstations to distributed clusters.

This solver was developed to provide the image alignment steps for :cite:`mouse`. The starting point for this package was the MATLAB-based package described in :cite:`KDS`. This solver is described in :cite:`thiswork`.

Compared to that repository, this repository has a number of changes and additions:

.. |pastix| replace:: PaStiX
.. _pastix: https://gitlab.inria.fr/solverstack/pastix
.. |petsc| replace:: PETSc
.. _petsc: https://www.mcs.anl.gov/petsc/

+----------------+---------------------------+---------------------------------+
|                | :cite:`KDS`               | this repo                       |
+================+===========================+=================================+
| language       | MATLAB                    | Python,                         |
|                |                           | C for                           |  
|                |                           | external                        |
+----------------+---------------------------+---------------------------------+
| external solver| |pastix|_                 | multiple,                       |
|                |                           | via |petsc|_                    |
+----------------+---------------------------+---------------------------------+
| external solver| independent               | included in                     |
| installation   | of repo                   | Singularity                     |
|                |                           | container                       |
+----------------+---------------------------+---------------------------------+
| transforms     | - translation             | - translation                   |
|                | - rigid approximation     | - rigid via rotation transform  |
|                | - affine                  | - affine                        |
|                | - polynomial to 3rd degree| - polynomial to arbitrary degree|
|                |                           | - thin plate spline             |
+----------------+---------------------------+---------------------------------+
| automated tests| N/A                       | TravisCI                        |
+----------------+---------------------------+---------------------------------+


.. toctree::
   :maxdepth: 2
   :caption: Contents:

The User Guide
--------------

Use cases and detailed explanations of input parameters.

.. toctree::
   :maxdepth: 2

   user/intro


API
---

This contains the complete documentation of the api

.. toctree::
   :maxdepth: 2

   api/bigfeta/modules
   distributed/distributed


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. bibliography:: refs.bib

Acknowledgement of Government Sponsorship
=========================================

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior / Interior Business Center (DoI/IBC) contract number D16PC00004. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DoI/IBC, or the U.S. Government.

