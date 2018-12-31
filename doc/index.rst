NumLoopy
========

Numloopy is an array library whose syntax for array computations is NumPy-like,
and generates `Loopy <https://github.com/inducer/loopy>`_ kernels. The loopy
kernels can then be translated to various `backends
<https://github.com/inducer/loopy/tree/master/loopy/target>`_:

* OpenCL
* CUDA
* ISPC
* Plain C

In order to fully understand the transformation model of NumLoopy, it is
strongly advised be familiar with loopy's semantics.

The following example is included in the :mod:`numloopy` distribution as
:download:`examples/hello-numloopy.py <../examples/hello-numloopy.py>`.

.. literalinclude:: ../examples/hello-numloopy.py


When the above script is run, the following kernel is generated:

.. literalinclude:: ../examples/hello-numloopy.cl
    :language: c


Lazy Evaluation
---------------

The variables which are of interest at the end of the computation must be
passed to  ``evaluate`` argument of the
:func:`numloopy.Stack.end_computation_stack`. All other variables are
substituted as expression into the computations. Thereby, only lazily
evaluating the specified variables.

Transformation model of NumLoopy
--------------------------------

The transformation information can be obtained by passing ``True`` for the
``transform`` argument to :func:`numloopy.Stack.end_computation_stack`. The
transformations data is returned as a mapping from the name of the variables
involved to the tuple of inames involved in their corresponding assignments.
The following example gives an example of the transformation.

.. literalinclude:: ../examples/introduce-tf_data.py

And when the above script is run, the generated code is

.. literalinclude:: ../examples/introduce-tf_data_inner.cl
    :language: c

The same code with ``UNROLLED_LOOP=2`` (*outermost loop is unrolled*) generates

.. literalinclude:: ../examples/introduce-tf_data_outer.cl
    :language: c

Places on the web related to Loopy
----------------------------------

* `Github <http://github.com/kaushikcfd/numloopy>`_ (get latest source code, file bugs)
* `Homepage <https://kaushikcfd.github.io/numloopy>`_

Table of Contents
-----------------

.. toctree::
    :maxdepth: 2

    ref_stack
    ref_array_sym

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
