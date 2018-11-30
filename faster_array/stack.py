import loopy as lp
import numpy as np
import islpy as isl
from pytools import UniqueNameGenerator, Record
from pymbolic import parse
from loopy.isl_helpers import make_slab


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
        """
        if isinstance(shape, int):
            shape = (shape, )
        unique_name = self.name_generator(based_on='array')
        self.data.append(
                lp.GlobalArg(
                    name=unique_name,
                    shape=shape,
                    dtype=dtype))

        iname_names = []

        for i in range(shape):
            iname_names.append(self.name_generator(based_on='i'))

        space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, iname_names)
        domain = isl.BasicSet.universe(space)

        for iname_name, axis_length in zip(iname_names, shape):
            domain &= make_slab(space, iname_name, 0, axis_length)

        assignee = parse('{}[{}]'.format(self.data[-1].name,
            ','.join(iname_names)))
        stmnt = lp.Assignment(assignee=assignee, expression=0)

        self.statements.append(stmnt)
        self.domains.append(domain)
