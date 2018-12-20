import loopy as lp
import numpy as np
from numbers import Number
import islpy as isl
from pymbolic.primitives import Variable, Call, Subscript
from pymbolic import parse
from loopy.isl_helpers import make_slab, simplify_via_aff


class ArraySymbol(lp.ArrayArg):
    __doc__ = lp.ArrayArg.__doc__ + (
            """
            :attribute stack: An instance of :class:`numloopy.Stack`
            """)
    allowed_extra_kwargs = [
            "address_space",
            "is_output_only",
            "stack"]

    def __init__(self, *args, **kwargs):
        address_space = kwargs.pop("address_space", lp.AddressSpace.GLOBAL)
        stack = kwargs.pop("stack", None)
        assert isinstance(kwargs['shape'], tuple)
        if stack is None:
            raise TypeError("must pass 'stack' to ArraySymbol")

        kwargs["address_space"] = address_space
        kwargs["stack"] = stack
        # setting the default order as C
        # note that this state is not stored, instead reflected in
        # :attr:dim_tags
        kwargs["order"] = kwargs.pop("order", 'C')
        self.order = kwargs["order"]

        super(ArraySymbol, self).__init__(*args, **kwargs)

    def _arithmetic_op(self, other, op):
        assert op in ['+', '-', '*', '/', '<', '<=', '>', '>=']

        def _apply_op(var1, var2):
            if op == '+':
                return var1+var2, self.dtype
            if op == '-':
                return var1-var2, self.dtype
            if op == '*':
                return var1*var2, self.dtype
            if op == '/':
                return var1/var2, self.dtype
            if op == '<':
                return var1.lt(var2), np.int
            if op == '<=':
                return var1.le(var2), np.int
            if op == '>':
                return var1.gt(var2), np.int
            if op == '>=':
                return var1.ge(var2), np.int
            raise RuntimeError()

        if isinstance(other, Number):
            inames = tuple(
                   self.stack.name_generator(based_on='i') for _ in
                   self.shape)
            rhs, dtype = _apply_op(Call(function=Variable(self.name),
                    parameters=tuple(Variable(iname) for iname in inames)),
                    other)
            subst_name = self.stack.name_generator(based_on='subst')
            self.stack.register_substitution(lp.SubstitutionRule(subst_name,
                    inames, rhs))
            return self.copy(name=subst_name,
                dtype=dtype)
        elif isinstance(other, ArraySymbol):
            if self.shape == other.shape:
                inames = tuple(
                       self.stack.name_generator(based_on='i') for _ in
                       self.shape)
                rhs, dtype = _apply_op(
                        Variable(self.name)(*tuple(Variable(iname) for iname in
                            inames)),
                        Variable(other.name)(*tuple(Variable(iname) for iname
                            in inames)))
                subst_name = self.stack.name_generator(based_on='subst')
                self.stack.register_substitution(lp.SubstitutionRule(subst_name,
                        inames, rhs))
                return self.copy(name=subst_name, dtype=dtype)
            else:
                left = self
                right = other
                new_left_shape = self.shape[:]
                new_right_shape = other.shape[:]
                if len(left.shape) > len(right.shape):
                    new_right_shape = (
                            (1, )*(len(left.shape)-len(right.shape))
                            + new_right_shape)
                if len(left.shape) < len(right.shape):
                    new_left_shape = (
                            (1, )*(len(right.shape)-len(left.shape))
                            + new_left_shape)

                # now both have the same shapes
                # let compute the shape of these guys
                new_shape = [1]*len(new_left_shape)

                for i, (left_len, right_len) in enumerate(zip(new_left_shape,
                        new_right_shape)):
                    if left_len == right_len:
                        new_shape[i] = left_len
                    else:
                        if left_len != 1 and right_len != 1:
                            raise ValueError("Cannot be broadcasted.")
                        else:
                            new_shape[i] = left_len*right_len

                new_shape = tuple(new_shape)

                if new_shape != left.shape:
                    subst_name = self.stack.name_generator(based_on="subst")
                    inames = tuple(self.stack.name_generator(based_on="i") for
                            _ in new_left_shape)

                    def _empty_if_zero(_idx):
                        if _idx == (0, ):
                            return ()
                        return _idx

                    indices = _empty_if_zero(
                            tuple(Variable(iname) if axis_len != 1 else 0 for
                            iname, axis_len in zip(inames[-len(left.shape):],
                                left.shape)))
                    rule = lp.SubstitutionRule(left.name,
                            inames,
                            Variable(left.name)(*indices))
                    self.stack.register_substitution(rule)
                    new_left = left.copy(stack=self.stack, name=subst_name,
                            shape=new_shape, dim_tags=None, order=left.order)
                else:
                    new_left = left

                if new_shape != right.shape:
                    subst_name = self.stack.name_generator(based_on="subst")
                    inames = tuple(self.stack.name_generator(based_on="i") for
                            _ in new_right_shape)

                    def _empty_if_zero(_idx):
                        if _idx == (0, ):
                            return ()
                        return _idx

                    indices = _empty_if_zero(
                            tuple(Variable(iname) if axis_len != 1 else 0 for
                            iname, axis_len in zip(inames[-len(right.shape):],
                                right.shape)))
                    rule = lp.SubstitutionRule(subst_name,
                            inames,
                            Variable(right.name)(*indices))
                    self.stack.register_substitution(rule)
                    new_right = right.copy(stack=self.stack, name=subst_name,
                            shape=new_shape, dim_tags=None, order=right.order)
                else:
                    new_right = right

                return new_left._arithmetic_op(new_right, op)

        else:
            raise NotImplementedError('__mul__ for', type(other))

    def __add__(self, other):
        return self._arithmetic_op(other, '+')

    def __sub__(self, other):
        return self._arithmetic_op(other, '-')

    def __rsub__(self, other):
        return -1*self._arithmetic_op(other, '-')

    def __mul__(self, other):
        return self._arithmetic_op(other, '*')

    def __truediv__(self, other):
        return self._arithmetic_op(other, '/')

    def __lt__(self, other):
        return self._arithmetic_op(other, '<')

    def __gt__(self, other):
        return self._arithmetic_op(other, '>')

    def __setitem__(self, index, value):
        if isinstance(index, (Number, slice, ArraySymbol)):
            index = (index, )
        assert isinstance(index, tuple)

        # current algorithm: assumes that the dereferenced guys are
        # always arguments and not temporary variables, maybe  we need to fix
        # this later?
        try:
            arg_name = self.stack.substs_to_arrays[self.name]
        except KeyError:
            inames = tuple(self.stack.name_generator(based_on="i") for _ in
                    self.shape)
            arg_name = self.stack.name_generator(based_on="arr")
            insn = lp.Assignment(assignee=parse('{}[{}]'.format(arg_name, ', '
                .join(inames))), expression=parse('{}({})'.format(self.name, ', '
                    .join(inames))))
            self.stack.register_implicit_assignment(insn)

            space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, inames)
            domain = isl.BasicSet.universe(space)

            for iname_name, axis_length in zip(inames, self.shape):
                domain &= make_slab(space, iname_name, 0, axis_length)

            self.stack.domains.append(domain)

        # now handling the second assignment

        try:
            inames, iname_lens = zip(
                    *tuple((self.stack.name_generator(based_on="i"), axis_len) for
                    idx, axis_len in zip(index, self.shape)
                    if isinstance(idx, slice) or isinstance(idx, ArraySymbol)))
            space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, inames)
            domain = isl.BasicSet.universe(space)

            for iname_name, axis_length in zip(inames, iname_lens):
                domain &= make_slab(space, iname_name, 0, axis_length)

            self.stack.domains.append(domain)
        except ValueError:
            inames = ()
            iname_lens = ()

        indices = []
        _k = 0
        for idx in index:
            if isinstance(idx, slice):
                indices.append(Variable(inames[_k]))
                _k += 1
            elif isinstance(idx, ArraySymbol):
                indices.append(Variable(idx.name)(Variable(inames[_k])))
                _k += 1
            else:
                indices.append(idx)
        assert _k == len(inames)
        indices = tuple(indices)

        if isinstance(value, ArraySymbol):
            insn = lp.Assignment(assignee=Subscript(Variable(arg_name), indices),
                    expression='{}({})'.format(value.name, ', '.join(str(iname)
                        for iname in inames)))
        elif isinstance(value, Number):
            insn = lp.Assignment(assignee=Subscript(Variable(arg_name), indices),
                    expression=value)
        else:
            raise TypeError("arrays can be only assigned with number or other "
                    "arrays")
        self.stack.register_implicit_assignment(insn)
        if self.name not in self.stack.substs_to_arrays:
            subst_name = self.stack.name_generator(based_on="subst")
            inames = tuple(self.stack.name_generator(based_on='i') for _ in
                    self.shape)
            rule = lp.SubstitutionRule(subst_name, inames,
                    expression=Subscript(Variable(arg_name),
                        tuple(Variable(iname) for iname in inames)))
            self.stack.register_substitution(rule)
            self.stack.data.append(self.copy(name=arg_name))

            self.stack.substs_to_arrays[subst_name] = arg_name
            self.name = subst_name

    def reshape(self, new_shape, order='C'):
        # need an error here complain if there is a shape mismatch
        # how to do this:
        # look at how loopy sets its dim tags, from shape and order.
        subst_name = self.stack.name_generator(based_on='subst')
        inames = tuple(self.stack.name_generator(based_on="i") for _ in new_shape)
        new_arg = self.copy(
                stack=self.stack,
                name=subst_name,
                shape=new_shape,
                dim_tags=None,
                order=order)

        linearized_idx = sum(Variable(iname)*dim_tag.stride for iname, dim_tag in
                zip(inames, new_arg.dim_tags))
        strides = tuple(dim_tag.stride for dim_tag in self.dim_tags)
        if self.dim_tags[0].stride == 1:
            pass
        elif self.dim_tags[-1].stride == 1:
            strides = strides[::-1]
        else:
            raise ValueError()

        indices = []
        for stride in strides[::-1]:
            current_idx = linearized_idx // stride
            indices.append(simplify_via_aff(current_idx))
            linearized_idx -= current_idx*stride

        # should we assert that the linearized index is 0?

        if self.dim_tags[-1].stride == 1:
            indices = indices[::-1]

        rule = lp.SubstitutionRule(
                subst_name,
                inames,
                expression=Variable(self.name)(*tuple(indices)))

        self.stack.register_substitution(rule)

        return ArraySymbol(stack=self.stack, name=subst_name, shape=new_shape)

    def __getitem__(self, index):
        if isinstance(index, Number):
            index = (index, )
        assert isinstance(index, tuple)

        right_inames = []
        left_inames = []
        shape = []
        for axis_len, idx in zip(self.shape, index):
            if isinstance(idx, int):
                right_inames.append(idx)
            elif isinstance(idx, slice):
                # right now only support complete slices
                # future plan is to make it diverse by adding it more support
                # its easy, anyone intereseted can do it, with a small change
                # to make_slab, or wait for kernel_callables_v3 to go through
                # into master
                assert idx.start is None
                assert idx.stop is None
                assert idx.step is None
                iname = self.stack.name_generator(based_on='i')
                right_inames.append(Variable(iname))
                left_inames.append(iname)
                shape.append(axis_len)
            else:
                raise TypeError('can be subscripted only with slices or '
                        'integers')

        rhs = Call(Variable(self.name), tuple(right_inames))
        subst_name = self.stack.name_generator(based_on='subst')
        self.stack.register_substitution(lp.SubstitutionRule(subst_name,
                    tuple(left_inames), rhs))

        def _one_if_empty(t):
            if t:
                return t
            else:
                return (1, )

        return ArraySymbol(stack=self.stack, name=subst_name, dtype=self.dtype,
                shape=_one_if_empty(tuple(shape)))

    __rmul__ = __mul__
    __radd__ = __add__
