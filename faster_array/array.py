import loopy as lp
import numpy as np
from numbers import Number
from pymbolic.primitives import Variable, Call


class ArraySymbol(lp.ArrayArg):
    __doc__ = lp.ArrayArg.__doc__ + (
            """
            :attribute stack: An instance of :class:`faster_array.Stack`
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
            self.stack.substitutions[subst_name] = lp.SubstitutionRule(subst_name,
                    inames, rhs)
            return self.copy(name=subst_name, dtype=dtype)
        elif isinstance(other, ArraySymbol):
            if other.shape == (1, ):
                # extend this to the broadcasting logic
                inames = tuple(
                       self.stack.name_generator(based_on='i') for _ in
                       self.shape)
                rhs, dtype = _apply_op(Call(function=Variable(self.name),
                        parameters=tuple(Variable(iname) for iname in inames)),
                        Variable(other.name)())
                subst_name = self.stack.name_generator(based_on='subst')
                self.stack.substitutions[subst_name] = lp.SubstitutionRule(
                        subst_name, inames, rhs)
                return self.copy(name=subst_name, dtype=dtype)

            if not self.shape == other.shape:
                raise NotImplementedError('At this moment broadcasting is not '
                        'supported. But support for it will be added soon!')
            inames = tuple(
                   self.stack.name_generator(based_on='i') for _ in
                   self.shape)
            rhs, dtype = _apply_op(Variable(self.name)(*tuple(Variable(iname) for
                iname in inames)), Variable(other.name)(*tuple(Variable(iname) for
                iname in inames)))
            subst_name = self.stack.name_generator(based_on='subst')
            self.stack.substitutions[subst_name] = lp.SubstitutionRule(subst_name,
                    inames, rhs)
            return self.copy(name=subst_name, dtype=dtype)
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
        self.stack.substitutions[subst_name] = lp.SubstitutionRule(subst_name,
                    tuple(left_inames), rhs)

        def _one_if_empty(t):
            if t:
                return t
            else:
                return (1, )

        return ArraySymbol(stack=self.stack, name=subst_name, dtype=self.dtype,
                shape=_one_if_empty(tuple(shape)))

    __rmul__ = __mul__
    __radd__ = __add__
