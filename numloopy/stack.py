import loopy as lp
import numpy as np
import islpy as isl
from loopy.symbolic import IdentityMapper
from numloopy.array import ArraySymbol
from pytools import UniqueNameGenerator, Record, memoize_method
from pymbolic import parse
from pymbolic.primitives import Variable, Subscript
from loopy.isl_helpers import make_slab
from numbers import Number


def fill_array(shape, value, name_generator):
    """
    Helper function to fill an array of shape ``shape`` with ``value`` .

    :arg shape: An instance of :class:`tuple` denoting the shape of the array.
    :arg value: The value with the array should be filled.
    :arg name_generator: An instance of :class:`pytools.UniqueNameGenerator`,
        for generating the name of the new array.

    :return: A tuple of the name of the subtitution, the substitution rule and
        the modified name generator.
    """
    inames = tuple(name_generator(based_on='i') for _ in shape)

    rhs = value
    subst_name = name_generator(based_on='subst')
    rule = lp.SubstitutionRule(subst_name, inames, rhs)

    return subst_name, rule, name_generator


class SubstToArrayExapander(IdentityMapper):
    """
    Mapper to change the substitution calls in :attr:`subst_to_args` to array
    instances.

    .. attribute substs_to_args::

        A mapping from substitution rules to arguments which needs to be
        represented as arrays.
    """
    def __init__(self, substs_to_args):
        self.substs_to_args = substs_to_args

    def map_call(self, expr):
        if expr.function.name in self.substs_to_args:
            def _zero_if_none(_t):
                if _t == ():
                    return (0, )
                return _t
            return Subscript(
                Variable(self.substs_to_args[expr.function.name]),
                _zero_if_none(tuple(self.rec(par) for par in expr.parameters)))

        return super(SubstToArrayExapander, self).map_call(expr)


class Stack(Record):
    """
    Records the information about the computation stack.

    .. attribute domains::

        An instance of :class:`list` representing the domains of the ``inames``
        used in :attr:`implicit_assignments`.

    .. attribute registered_substitutions::

        A mapping from an instance of :class:`str`(name of the substitution) to
        an instance of :class:`SubstitutionRule`. Similar to
        :attr:`loop.LoopKernel.substitutions`.


    .. attribute name_generator::

        An instance of :class:`pytools.UniqueNameGenerator`.

    .. attribute implicit_assignments

        A mapping from scheduled number to instance of
        :class:`loopy.Assignment` which denote the instruction to be flushed
        before at schedule number.

    .. attribute data::

        An instance of :class:`list` containing the data created due to
        :attr:`implicit_assignments`.

    .. attribute substs_to_arrays::

        A mapping from from substitution names to arrays that are equivalently
        used.

    """
    def __init__(self, domains=[], registered_substitutions=[],
            implicit_assignments={},
            data=[], substs_to_arrays={},
            name_generator=UniqueNameGenerator()):

        self.domains = domains
        self.data = data
        self.substs_to_arrays = substs_to_arrays
        self.registered_substitutions = registered_substitutions
        self.implicit_assignments = implicit_assignments

        self.name_generator = name_generator

    def register_substitution(self, rule):
        """
        Registers a substitution rule on the top of the stack.

        :arg rule: An instance of :class:`loopy.SubstitutionRule`.
        """
        assert isinstance(rule, lp.SubstitutionRule)

        self.registered_substitutions.append(rule)

    def register_implicit_assignment(self, insn):
        """
        Registers an instruction ``insn`` at the top of the stack.
        """
        # representation of implicit_assignment
        # implicit_assignment[int] = [insn1, insn2, ...]
        # all these assingments should be made before dereferencing any other
        # substitution of value int+1

        assignments_till_now = self.implicit_assignments.pop(
            len(self.registered_substitutions), [])[:]
        assignments_till_now.append(insn)
        self.implicit_assignments[len(self.registered_substitutions)] = (
                assignments_till_now)

    @memoize_method
    def get_substitution(self, name):
        """
        Returns the substiution rule corresponding to the substitution
        registered with the name ``name``.

        :arg name: An instance of :class:`str`.

        :return: An instance of :class:`loopy.SubstitutionRule` registered
            with the name ``name``.
        """
        for rule in self.registered_substitutions:
            if rule.name == name:
                return rule

        raise KeyError("Did not find the required substitution.")

    def zeros(self, shape, dtype=np.float64):
        """
        Registers a substitution rule on to the stack with an array whose values
        are filled with 0.

        :return: An instance of :class:`loopy.ArraySymbol`, corresponding to the
            substitution rule which was registered.
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

        self.register_substitution(rule)

        return arg

    def ones(self, shape, dtype=np.float64):
        """
        Registers a substitution rule on to the stack with an array whose values
        are filled with 1.

        :return: An instance of :class:`loopy.ArraySymbol`, corresponding to the
            substitution rule which was registered.
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

        self.register_substitution(rule)

        return arg

    def arange(self, stop):
        """
        Registers a substitution rule on to the stack with an array whose values
        are filled equivalent to ``numpy.arange``.

        :arg stop: An instance of :class:`int` denoting the extent of the
        array.

        :return: An instance of :class:`numloopy.ArraySymbol` of shape
        ``(stop,)``, corresponding to the substitution rule which was
        registered.
        """
        assert isinstance(stop, int)
        subst_name = self.name_generator(based_on="subst")
        arg = ArraySymbol(
                stack=self,
                name=subst_name,
                shape=(stop, ),
                dtype=np.int)
        iname = self.name_generator(based_on="i")
        rhs = Variable(iname)
        rule = lp.SubstitutionRule(subst_name, (iname, ), rhs)

        self.register_substitution(rule)

        return arg

    def sum(self, arg, axis=None):
        """
        Registers  a substitution rule in order to sum the elements of array
        ``arg`` along ``axis``.

        :return: An instance of :class:`numloopy.ArraySymbol` which is
            which is registered as the sum-substitution rule.
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

        self.domains.append(domain)

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
        self.register_substitution(rule)

        return summed_arg

    def argument(self, shape, dtype=np.float64):
        """
        Return an instance of :class:`numloopy.ArraySymbol` which the loop
        kernel expects as an input argument.
        """
        if isinstance(shape, int):
            shape = (shape, )
        assert isinstance(shape, tuple)

        inames = tuple(
               self.name_generator(based_on='i') for _ in
               shape)

        arg_name = self.name_generator(based_on='arr')

        rhs = Subscript(Variable(arg_name),
                tuple(Variable(iname) for iname in inames))
        subst_name = self.name_generator(based_on='subst')
        self.register_substitution(lp.SubstitutionRule(subst_name,
                inames, rhs))
        self.substs_to_arrays[subst_name] = arg_name

        self.data.append(lp.GlobalArg(name=arg_name, shape=shape, dtype=dtype))

        return ArraySymbol(stack=self, name=subst_name, dtype=dtype,
                shape=shape)

    def cumsum(self, arg):
        """
        Registers  a substitution rule in order to cumulatively sum the
        elements of array ``arg`` along ``axis``. Mimics :func:`numpy.cumsum`.

        :return: An instance of :class:`numloopy.ArraySymbol` which is
            which is registered as the cumulative summed-substitution rule.
        """
        # Note: this can remain as a substitution but loopy does not have
        # support for translating inames for substitutions to the kernel
        # domains
        assert len(arg.shape) == 1
        i_iname = self.name_generator(based_on="i")
        j_iname = self.name_generator(based_on="i")

        space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, [i_iname,
            j_iname])
        domain = isl.BasicSet.universe(space)
        arg_name = self.name_generator(based_on="arr")
        subst_name = self.name_generator(based_on="subst")
        domain = domain & make_slab(space, i_iname, 0, arg.shape[0])
        domain = domain.add_constraint(
                isl.Constraint.ineq_from_names(space, {j_iname: 1}))
        domain = domain.add_constraint(
                isl.Constraint.ineq_from_names(space,
                    {j_iname: -1, i_iname: 1, 1: -1}))
        cumsummed_arg = ArraySymbol(
                stack=self,
                name=arg_name,
                shape=arg.shape,
                dtype=arg.dtype)
        cumsummed_subst = ArraySymbol(
                stack=self,
                name=subst_name,
                shape=arg.shape,
                dtype=arg.dtype)
        subst_iname = self.name_generator(based_on="i")
        rule = lp.SubstitutionRule(
                subst_name, (subst_iname,), Subscript(Variable(arg_name),
                    (Variable(subst_iname), )))

        from loopy.library.reduction import SumReductionOperation

        insn = lp.Assignment(
                assignee=Subscript(Variable(arg_name), (Variable(i_iname), )),
                expression=lp.Reduction(
                    SumReductionOperation(),
                    (j_iname, ),
                    parse('{}({})'.format(arg.name,
                        j_iname))))
        self.data.append(cumsummed_arg)
        self.substs_to_arrays[subst_name] = arg_name
        self.register_implicit_assignment(insn)
        self.domains.append(domain)

        self.register_substitution(rule)
        return cumsummed_subst

    def end_computation_stack(self, evaluate=(), transform=False):
        """
        Returns an instance :class:`loopy.LoopKernel` corresponding to the
        computations pushed in the computation stack.

        :param variables_needed: An instance of :class:`tuple` of the variables
        that must be computed
        """
        statements = []
        tf_data = {}
        domains = self.domains[:]
        data = self.data[:]
        substitutions = {}
        substitutions_needed = [array_sym.name for array_sym in
                evaluate if array_sym.name not in self.substs_to_arrays]

        substs_to_arrays = self.substs_to_arrays.copy()

        for i, rule in enumerate(self.registered_substitutions):
            substs_to_arg_mapper = SubstToArrayExapander(
                    substs_to_arrays.copy())
            statements.extend([insn.with_transformed_expressions(
                substs_to_arg_mapper) for insn in
                self.implicit_assignments.pop(i, [])])
            if rule.name in substitutions_needed:
                rule = rule.copy(
                        expression=substs_to_arg_mapper(rule.expression))
                arg_name = self.name_generator(based_on="arr")
                arg = evaluate[
                        substitutions_needed.index(rule.name)]
                data.append(arg.copy(name=arg_name))
                substs_to_arrays[arg.name] = arg_name

                if arg.shape != (1, ) and arg.shape != (1):
                    inames = tuple(self.name_generator(based_on='i') for _ in
                            arg.shape)
                    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, inames)
                    domain = isl.BasicSet.universe(space)

                    for iname_name, axis_length in zip(inames, arg.shape):
                        domain &= make_slab(space, iname_name, 0, axis_length)

                    assignee = substs_to_arg_mapper(parse('{}[{}]'.format(arg_name,
                        ', '.join(inames))))
                    stmnt = lp.Assignment(assignee=assignee,
                            expression=parse('{}({})'.format(arg.name,
                                ', '.join(inames))))
                    domains.append(domain)
                    tf_data[arg.name] = inames
                else:
                    assignee = parse('{}[0]'.format(arg_name))
                    stmnt = lp.Assignment(assignee=assignee,
                            expression=parse('{}()'.format(arg.name)))
                    tf_data[arg.name] = ()
                statements.append(stmnt.with_transformed_expressions(
                    substs_to_arg_mapper))

            substitutions[rule.name] = rule.copy(
                    expression=substs_to_arg_mapper(rule.expression))

        substs_to_arg_mapper = SubstToArrayExapander(
                substs_to_arrays.copy())

        statements.extend([insn.with_transformed_expressions(
            substs_to_arg_mapper) for insn in
            self.implicit_assignments.pop(i+1, [])])

        knl = lp.make_kernel(
                domains=domains,
                instructions=statements,
                kernel_data=data,
                seq_dependencies=True,
                lang_version=(2018, 2))
        knl = knl.copy(substitutions=substitutions,
                target=lp.CTarget())
        if transform:
            return knl, tf_data
        else:
            return knl


def begin_computation_stack():
    """
    Must be called to initialize a computational stack.
    """
    return Stack()
