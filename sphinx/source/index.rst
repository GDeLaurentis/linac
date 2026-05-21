Linear Algebra with CUDA
========================

``Linac`` is a Python library for GPU-accelerated linear algebra using
CUDA. Its main functionality is dense Gaussian elimination to
row-reduced echelon form, with support for real and complex
floating-point arithmetic, finite fields, and leading-digit
``p``-adic workflows.

The package was developed for applications to finite-field
reconstruction of scattering amplitude in quantum field theory, but the
core routines are general-purpose tools for dense linear algebra over
the supported arithmetic domains.

Main features
-------------

* GPU row reduction to reduced row-echelon form via
  :func:`linac.cuda_row_reduce`.
* CPU row reduction via :func:`linac.row_reduce`.
* Matrix construction for polynomial linear systems via
  :func:`linac.cuda_load_matrix` and :func:`linac.load_matrices`.
* Utilities for vector spaces over number fields and function fields.
* Support for CUDA acceleration through ``pycuda``.
* Optional CPU finite-field acceleration through ``galois`` when
  available.

Installation
------------

The latest stable release is available from PyPI:

.. code-block:: bash

   pip install linac

To enable GPU acceleration, install the CUDA extra:

.. code-block:: bash

   pip install "linac[cuda]"

For development, clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/GDeLaurentis/linac.git
   pip install -e "linac[full]"

The available extras are

.. code-block:: text

   cuda, dev, full (= cuda + dev)

GPU acceleration requires a working CUDA development environment,
including ``nvcc``, compatible NVIDIA drivers, and access to a
CUDA-enabled GPU.

Quick start
-----------

The main entry point is :func:`linac.cuda_row_reduce`, which computes a
row-reduced echelon form on the GPU.

.. code-block:: python
   :linenos:

   import numpy
   from linac import cuda_row_reduce

   p = 2**31 - 1

   A = numpy.random.randint(0, p, size=(2000, 2000), dtype="uint32")
   rref = cuda_row_reduce(A, field_characteristic=p)

Setting ``field_characteristic=0`` selects floating-point arithmetic.

.. code-block:: python
   :linenos:

   import numpy
   from linac import cuda_row_reduce

   A = numpy.random.rand(2000, 2000)
   rref = cuda_row_reduce(A, field_characteristic=0)

For CPU row reduction, use :func:`linac.row_reduce`.

.. code-block:: python
   :linenos:

   from linac import row_reduce

   rref, variable_order = row_reduce(A, prime=p)

Constructing linear systems
---------------------------

``Linac`` also provides utilities for constructing dense matrices from
polynomial ansätze. These are useful when fitting unknown coefficients
from numerical evaluations.

.. code-block:: python
   :linenos:

   from linac import load_matrices

   matrices = load_matrices(prefactors, ansatze, points, use_cuda=True)

For lower-level GPU matrix construction, see
:func:`linac.cuda_load_matrix`.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   modules

API reference
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   linac.cuda_row_reduce
   linac.cuda_load_matrix
   linac.load_matrices
   linac.row_reduce
   linac.tensor_function
   linac.ColumnVectorSpace
   linac.VectorSpaceOfFunctions

Project links
-------------

* GitHub repository: https://github.com/GDeLaurentis/linac
* PyPI package: https://pypi.org/project/linac/

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
