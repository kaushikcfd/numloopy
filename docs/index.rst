NumLoopy
========

Numloopy is an array library whose syntax for array computations is
NumPy-like, and generates `Loopy <https://github.com/inducer/loopy>`_ kernels. The loopy kernels can then be translated to various `backends <https://github.com/inducer/loopy/tree/master/loopy/target>`:
+ OpenCL
+ CUDA
+ ISPC
+ Plain C

In order to fully take advantage of the transformations of NumLoopy, it is strongly advised to be familiar with loopy's semantics and transformation model.

loopy is a code generator for array-based code in the OpenCL/CUDA execution
model. Here's a very simple example of how to double the entries of a vector
using loopy:

.. literalinclude:: ../examples/python/hello-loopy.py
   :end-before: ENDEXAMPLE

This example is included in the :mod:`loopy` distribution as
:download:`examples/python/hello-loopy.py <../examples/python/hello-loopy.py>`.

When you run this script, the following kernel is generated, compiled, and executed:

.. literalinclude:: ../examples/python/hello-loopy.cl
    :language: c

(See the full example for how to print the generated code.)

.. _static-binary:

Want to try out loopy?
----------------------

There's no need to go through :ref:`installation` if you'd just like to get a
feel for what loopy is.  Instead, you may
`download a self-contained Linux binary <https://gitlab.tiker.net/inducer/loopy/-/jobs/66778/artifacts/browse/build-helpers/>`_.
This is purposefully built on an ancient Linux distribution, so it should work
on most versions of Linux that are currently out there.

Once you have the binary, do the following::

    chmod +x ./loopy-centos6
    ./loopy-centos6 --target=opencl hello-loopy.loopy
    ./loopy-centos6 --target=cuda hello-loopy.loopy
    ./loopy-centos6 --target=ispc hello-loopy.loopy

Grab the example here: :download:`examples/python/hello-loopy.loopy <../examples/python/hello-loopy.loopy>`.

You may also donwload the most recent version by going to the `list of builds
<https://gitlab.tiker.net/inducer/loopy/builds>`_, clicking on the newest one
of type "CentOS binary", clicking on "Browse" under "Build Artifacts", then
navigating to "build-helpers", and downloading the binary from there.

Places on the web related to Loopy
----------------------------------

* `Python package index <http://pypi.python.org/pypi/loo.py>`_ (download releases) Note the extra '.' in the PyPI identifier!

* `Github <http://github.com/kaushikcfd/numloopy>`_ (get latest source code, file bugs)
* `Homepage <https://kaushikcfd.github.io/numloopy>`_

Table of Contents
-----------------

.. toctree::
    :maxdepth: 2

    tutorial
    ref_creation
    ref_kernel
    ref_transform
    ref_other
    misc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
