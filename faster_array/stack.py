import loopy as lp
import numpy as np
import islpy as isl
from loopy.symbolic import IdentityMapper
from faster_array.array import ArraySymbol
from pytools import UniqueNameGenerator, Record, memoize_method
from pymbolic import parse
from pymbolic.primitives import Variable, Subscript
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


class SubstToArrayExapander(IdentityMapper):
    def __init__(self, substs_to_args):
        self.substs_to_args = substs_to_args

    def map_call(self, expr):
        if expr.function.name in self.substs_to_args:
            return Subscript(
                Variable(self.substs_to_args[expr.function.name]),
                tuple(self.rec(par) for par in expr.parameters))

        return super(SubstToArrayExapander, self).map_call(expr)


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
    def __init__(self, domains=[], registered_substitutions=[], parameters=[],
            implicit_assignments={},
            data=[], substs_to_arrays={}, stack_compute_num=0,
            name_generator=UniqueNameGenerator()):

        self.domains = domains
        self.parameters = parameters
        self.data = data
        self.substs_to_arrays = substs_to_arrays
        self.registered_substitutions = registered_substitutions
        self.implicit_assignments = implicit_assignments

        self.name_generator = name_generator

    def register_substitution(self, rule):
        assert isinstance(rule, lp.SubstitutionRule)

        self.registered_substitutions.append(rule)

    def register_implicit_assignment(self, insn):
        # current data representation for implicit assignments
        # make note of the length of:
        # implicit_assignment[int] = [assign1, assign2, ...]
        # all these assingments should be made before dereferencing any other
        # substitution of value int+1

        assignments_till_now = self.implicit_assignments.pop(
            len(self.registered_substitutions), [])[:]
        assignments_till_now.append(insn)
        self.implicit_assignments[len(self.registered_substitutions)] = (
                assignments_till_now)

    @memoize_method
    def get_substitution(self, name):
        for rule in self.registered_substitutions:
            if rule.name == name:
                return rule

        raise KeyError("Did not find the required substitution.")

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

        self.register_substitution(rule)

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

        self.register_substitution(rule)

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

    def end_computation_stack(self, variables_needed=()):
        """
        Returns an instance :class:`loopy.LoopKernel` corresponding to the
        computations pushed in the computation stack.

        :param variables_needed: An instance of :class:`tuple` of the variables
        that must be computed
        """
        statements = []
        domains = self.domains[:]
        data = self.data[:]
        substitutions = {}
        substitutions_needed = [array_sym.name for array_sym in
                variables_needed]

        substs_to_arrays = self.substs_to_arrays.copy()

        for i, rule in enumerate(self.registered_substitutions):
            substs_to_arg_mapper = SubstToArrayExapander(
                    substs_to_arrays.copy())
            statements.extend([insn.with_transformed_expresssion(
                substs_to_arg_mapper) for insn in
                self.implicit_assignments.pop(i, [])])
            if rule.name in substitutions_needed:
                rule = rule.copy(
                        expression=substs_to_arg_mapper(rule.expression))
                arg_name = self.name_generator(based_on="arr")
                arg = variables_needed[
                        substitutions_needed.index(rule.name)]
                data.append(arg.copy(name=arg_name))
                substs_to_arrays[arg.name] = arg_name

                if arg.shape != (1, ):
                    inames = tuple(self.name_generator(based_on='i') for _ in
                            arg.shape)
                    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, inames)
                    domain = isl.BasicSet.universe(space)

                    for iname_name, axis_length in zip(inames, arg.shape):
                        domain &= make_slab(space, iname_name, 0, axis_length)

                    assignee = parse('{}[{}]'.format(arg_name,
                        ', '.join(inames)))
                    stmnt = lp.Assignment(assignee=assignee,
                            expression=parse('{}({})'.format(arg.name,
                                ', '.join(inames))))
                    domains.append(domain)
                else:
                    assignee = parse('{}[0]'.format(arg_name))
                    stmnt = lp.Assignment(assignee=assignee,
                            expression=parse('{}()'.format(arg.name)))
                statements.append(stmnt.with_transformed_expressions(
                    substs_to_arg_mapper))

            substitutions[rule.name] = rule.copy(
                    expression=substs_to_arg_mapper(rule.expression))

        knl = lp.make_kernel(
                domains=domains,
                instructions=statements,
                kernel_data=data,
                seq_dependencies=True,
                lang_version=(2018, 2))
        knl = knl.copy(substitutions=substitutions,
                target=lp.CTarget())

        return knl


def begin_computation_stack():
    """
    Must be called to initialize a copmutational stack.
    """
    return Stack()
