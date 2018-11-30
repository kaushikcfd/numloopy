import loopy as lp
import numpy as np
import islpy as isl
from faster_array.array import ArraySymbol
from pytools import UniqueNameGenerator, Record
from pymbolic import parse
from loopy.isl_helpers import make_slab


def fill_array(arg, value, name_generator):
    iname_names = []
    for i in range(len(arg.shape)):
        iname_names.append(name_generator(based_on='i'))

    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, iname_names)
    domain = isl.BasicSet.universe(space)

    for iname_name, axis_length in zip(iname_names, arg.shape):
        domain &= make_slab(space, iname_name, 0, axis_length)

    assignee = parse('{}[{}]'.format(arg.name,
        ', '.join(iname_names)))
    stmnt = lp.Assignment(assignee=assignee, expression=value)

    return stmnt, domain, name_generator


class Stack(Record):
    """
    Records the information about the computation stack.

    .. attribute domains::

        An instance of :class:`list` having the similar format as
        :attr:`loopy.LoopKernel.domains`.

    .. attribute data::

        An instance of :class:`list` including the data that would be involved
        in the computation stack.

    .. attribute name_generator::

        An instance of :class:`pytools.UniqueNameGenerator`.
    """
    def __init__(self, domains=[], statements=[], data=[],
            name_generator=UniqueNameGenerator()):

        self.domains = domains
        self.statements = statements
        self.data = data

        self.name_generator = name_generator

    def zeros(self, shape, dtype=np.float64):
        """
        Adds statements and domains to initialize an array to 0.

        :return: An instance of :class:`loopy.ArraySymbol`, corresponding to the
            array which and created and initialized to 0.
        """
        if isinstance(shape, int):
            shape = (shape, )
        assert isinstance(shape, tuple)

        arg = ArraySymbol(
                stack=self,
                name=self.name_generator(based_on='array'),
                shape=shape,
                dtype=dtype)
        stmnt, domain = fill_array(arg, value=0)

        self.statements.append(stmnt)
        self.domains.append(domain)
        self.data.append(arg)

        return arg

    def ones(self, shape, dtype=np.float64):
        """
        Adds statements and domains to initialize an array to 1.

        :return: An instance of :class:`loopy.ArraySymbol`, corresponding to the
            array which and created and initialized to 0.
        """
        if isinstance(shape, int):
            shape = (shape, )
        assert isinstance(shape, tuple)

        arg = ArraySymbol(
                stack=self,
                name=self.name_generator(based_on='array'),
                shape=shape,
                dtype=dtype)
        stmnt, domain, self.name_generator = fill_array(arg, value=1,
                name_generator=self.name_generator)

        self.statements.append(stmnt)
        self.domains.append(domain)
        self.data.append(arg)

        return arg

    def sum(self, arg, axis=None):
        """
        Sums all the elements of the elements according to the axis.
        """
        if isinstance(axis, int):
            axis = (axis, )

        if not axis:
            axis = tuple(range(len(arg.shape)))

        inames = [self.name_generator(based_on="i") for _ in
                arg.shape]

        space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
                inames)
        domain = isl.BasicSet.universe(space)
        for axis_len, iname in zip(arg.shape, inames):
            domain &= make_slab(space, iname, 0, axis_len)

        reduction_inames = tuple(iname for i, iname in enumerate(inames) if i in
                axis)
        left_inames = tuple(iname for i, iname in enumerate(inames) if i not in
                axis)
        if len(left_inames) == 0:
            left_inames = ['0']

        def _one_if_empty(t):
            if t:
                return t
            else:
                return (1, )

        summed_arg = ArraySymbol(
                stack=self,
                name=self.name_generator(based_on="array_sum"),
                shape=_one_if_empty(tuple(axis_len for i, axis_len in
                    enumerate(arg.shape) if i not in axis)),
                dtype=arg.dtype)

        from loopy.library.reduction import SumReductionOperation

        insn = lp.Assignment(
                assignee=parse('{}[{}]'.format(summed_arg.name,
                    ', '.join(left_inames))),
                expression=lp.Reduction(
                    SumReductionOperation(),
                    reduction_inames,
                    parse('{}[{}]'.format(arg.name,
                        ', '.join(inames)))))

        self.data.append(summed_arg)
        self.domains.append(domain)
        self.statements.append(insn)

        return axis

    def end_computation_stack(self):
        """
        Returns an instance :class:`loopy.LoopKernel` corresponding to the
        computations pushed in the computation stack.
        """

        knl = lp.make_kernel(
                self.domains,
                self.statements,
                kernel_data=self.data,
                seq_dependencies=True,
                lang_version=(2018, 2))

        return knl


def begin_computation_stack():
    """
    Must be called to initialize a copmutational stack.
    """
    return Stack()
