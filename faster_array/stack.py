import loopy as lp
import numpy as np
import islpy as isl
from faster_array.array import ArraySymbol
from pytools import UniqueNameGenerator, Record
from pymbolic import parse
from loopy.isl_helpers import make_slab
from numbers import Number


# space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, iname_names)
# domain = isl.BasicSet.universe(space)

# for iname_name, axis_length in zip(iname_names, arg.shape):
#     domain &= make_slab(space, iname_name, 0, axis_length)

# assignee = parse('{}[{}]'.format(arg.name,
#     ', '.join(iname_names)))
# stmnt = lp.Assignment(assignee=assignee, expression=value)

def fill_array(shape, value, name_generator):
    inames = tuple(name_generator(based_on='i') for _ in shape)

    rhs = value
    subst_name = name_generator(based_on='subst')
    rule = lp.SubstitutionRule(subst_name, inames, rhs)

    return subst_name, rule, name_generator


class Stack(Record):
    """
    Records the information about the computation stack.

    .. attribute name_generator::

        An instance of :class:`pytools.UniqueNameGenerator`.

    .. attribute substitutions::

        A mapping from an instance of :class:`str`(name of the substitution) to
        an instance of :class:`SubstitutionRule`.

    .. attribute parameters::

        An instance of :class:`list` which contains the parameters
    """
    def __init__(self, substitutions={}, parameters=[],
            name_generator=UniqueNameGenerator()):

        self.substitutions = substitutions
        self.parameters = parameters

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

        for axis_length in shape:
            # Note: I have not implemented anything here, but maybe in the
            # future we would need to do something to let the stack know about
            # the vairable
            if isinstance(axis_length, Number):
                pass
            elif isinstance(axis_length, str):
                pass
            else:
                raise TypeError("Shape can only be initialized by numbers or"
                        " strings")

        subst_name, rule, name_generator = fill_array(shape, value=0,
                name_generator=self.name_generator)
        arg = ArraySymbol(
                stack=self,
                name=subst_name,
                shape=shape,
                dtype=dtype)

        self.substitutions[subst_name] = rule

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

        for axis_length in shape:
            # Note: I have not implemented anything here, but maybe in the
            # future we would need to do something to let the stack know about
            # the vairable
            if isinstance(axis_length, Number):
                pass
            elif isinstance(axis_length, str):
                pass
            else:
                raise TypeError("Shape can only be initialized by numbers or"
                        " strings")

        subst_name, rule, name_generator = fill_array(shape, value=1,
                name_generator=self.name_generator)
        arg = ArraySymbol(
                stack=self,
                name=subst_name,
                shape=shape,
                dtype=dtype)

        self.substitutions[subst_name] = rule

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

        def _one_if_empty(t):
            if t:
                return t
            else:
                return (1, )

        subst_name = self.name_generator(based_on="subst")

        summed_arg = ArraySymbol(
                stack=self,
                name=subst_name,
                shape=_one_if_empty(tuple(axis_len for i, axis_len in
                    enumerate(arg.shape) if i not in axis)),
                dtype=arg.dtype)

        from loopy.library.reduction import SumReductionOperation

        rule = lp.SubstitutionRule(
                subst_name,
                left_inames,
                lp.Reduction(
                    SumReductionOperation(),
                    reduction_inames,
                    parse('{}({})'.format(arg.name,
                        ', '.join(inames)))))
        self.substitutions[subst_name] = rule

        return summed_arg

    def end_computation_stack(self, variables_needed=()):
        """
        Returns an instance :class:`loopy.LoopKernel` corresponding to the
        computations pushed in the computation stack.

        :param variables_needed: An instance of :class:`tuple` of the variables
        that must be computed
        """
        statements = []
        domains = []
        data = []
        for arg in variables_needed:
            inames = tuple(self.name_generator(based_on='i') for _ in arg.shape)

            space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, inames)
            domain = isl.BasicSet.universe(space)

            for iname_name, axis_length in zip(inames, arg.shape):
                domain &= make_slab(space, iname_name, 0, axis_length)

            arg_name = self.name_generator(arg.name+'_arg')
            data.append(arg.copy(name=arg_name))

            assignee = parse('{}[{}]'.format(arg_name,
                ', '.join(inames)))
            stmnt = lp.Assignment(assignee=assignee,
                    expression=parse('{}({})'.format(arg.name,
                        ', '.join(inames))))

            statements.append(stmnt)
            domains.append(domain)

        knl = lp.make_kernel(
                domains=domains,
                instructions=statements,
                kernel_data=data,
                seq_dependencies=True,
                lang_version=(2018, 2))

        return knl.copy(substitutions=self.substitutions.copy())


def begin_computation_stack():
    """
    Must be called to initialize a copmutational stack.
    """
    return Stack()
