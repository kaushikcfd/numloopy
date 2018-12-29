import sys
import numpy
import loopy as lp
import pyopencl as cl
import numloopy as nplp


def test_axpy_like(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    def func_nplp():
        np = nplp.begin_computation_stack()

        a = np.ones(10)
        b = np.arange(10)
        c = 2*a + 3*b

        knl = np.end_computation_stack([a, c])
        evt, (out_a, out_c) = knl(queue)

        return out_a.get(), out_c.get()

    def func_np():
        import numpy as np
        a = np.ones(10)
        b = np.arange(10)
        c = 2*a + 3*b

        return a, c

    nplp_a, nplp_c = func_np()
    np_a, np_c = func_np()

    assert numpy.allclose(np_a, nplp_a)
    assert numpy.allclose(np_c, nplp_c)


def test_qudrant_splitting(ctx_factory):

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    def func_nplp(x_in):

        np = nplp.begin_computation_stack()

        x = np.argument((1000, 2))
        x_new = np.argument((1000, 2))

        midpoint = np.sum(x, axis=0)/x.shape[0]

        is_left = x[:, 0] < midpoint[0]
        is_lower = x[:, 1] < midpoint[1]

        is_ll = is_lower*is_left
        is_ul = (1-is_lower)*is_left
        is_lr = is_lower*(1-is_left)
        is_ur = (1-is_lower)*(1-is_left)

        num_ll = np.sum(is_ll)
        num_ul = np.sum(is_ul)
        num_lr = np.sum(is_lr)

        idx_ll = np.cumsum(is_ll)*(is_ll)
        idx_ul = (np.cumsum(is_ul) + num_ll)*(is_ul)
        idx_lr = (np.cumsum(is_lr) + num_ll + num_ul)*(is_lr)
        idx_ur = (np.cumsum(is_ur) + num_ll + num_ul + num_lr)*(is_ur)

        indices = idx_ll + idx_ul + idx_lr + idx_ur - 1
        x_new[indices, :] = x

        knl = np.end_computation_stack([midpoint, is_lower, is_left,
            num_ll, num_ul, num_lr, indices, x_new])
        knl = lp.set_options(knl, return_dict=True)

        knl, out_dict = knl(queue, **{x.name: x_in})

        return out_dict[x_new.name].get()

    def func_np():
        import numpy as np

        x = np.argument((1000, 2))
        x_new = np.argument((1000, 2))

        midpoint = np.sum(x, axis=0)/x.shape[0]

        is_left = x[:, 0] < midpoint[0]
        is_lower = x[:, 1] < midpoint[1]

        is_ll = is_lower*is_left
        is_ul = (1-is_lower)*is_left
        is_lr = is_lower*(1-is_left)
        is_ur = (1-is_lower)*(1-is_left)

        num_ll = np.sum(is_ll)
        num_ul = np.sum(is_ul)
        num_lr = np.sum(is_lr)

        idx_ll = np.cumsum(is_ll)*(is_ll)
        idx_ul = (np.cumsum(is_ul) + num_ll)*(is_ul)
        idx_lr = (np.cumsum(is_lr) + num_ll + num_ul)*(is_lr)
        idx_ur = (np.cumsum(is_ur) + num_ll + num_ul + num_lr)*(is_ur)

        indices = idx_ll + idx_ul + idx_lr + idx_ur - 1
        x_new[indices, :] = x

        return x_new

    nplp_xnew = func_np()
    np_xnew, np_c = func_np()

    assert numpy.allclose(np_xnew, nplp_xnew)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
