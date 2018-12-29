import sys
import numpy
import pyopencl as cl
import numloopy as nplp

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()

from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests


__all__ = [
        "pytest_generate_tests",
        "cl"  # 'cl.create_some_context'
        ]


def test_reshape(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    def func_nplp():
        np = nplp.begin_computation_stack()

        a = np.arange(8)
        b = a.reshape((2, 4), order='F')

        knl = np.end_computation_stack([a, b])
        evt, (out_a, out_b) = knl(queue)

        return out_a.get(), out_b.get()

    def func_np():
        import numpy as np
        a = np.arange(8)
        b = a.reshape((2, 4), order='F')

        return a, b

    nplp_a, nplp_b = func_np()
    np_a, np_b = func_np()

    assert numpy.allclose(np_a, nplp_a)
    assert numpy.allclose(np_b, nplp_b)


def test_broadcast(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    def func_nplp():
        np = nplp.begin_computation_stack()

        A = np.arange(9).reshape((3, 3))  # noqa: N806
        x = np.arange(3)
        y = np.sum(A * x, axis=1)

        knl = np.end_computation_stack([A, x, y])
        evt, (out_A, out_x, out_y) = knl(queue)

        return out_A.get(), out_x.get(), out_y.get()

    def func_np():
        import numpy as np

        A = np.arange(9).reshape((3, 3))  # noqa: N806
        x = np.arange(3)
        y = np.sum(A * x, axis=1)

        return A, x, y

    nplp_A, nplp_x, nplp_y = func_np()
    np_A, np_x, np_y = func_np()

    assert numpy.allclose(np_A, nplp_A)
    assert numpy.allclose(np_x, nplp_x)
    assert numpy.allclose(np_y, nplp_y)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
