import loopy as lp
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

    def __add__(self, other):
        if isinstance(other, Number):
            inames = tuple(
                   self.stack.name_generator(based_on='i') for _ in
                   self.shape)
            rhs = Call(function=Variable(self.name),
                    parameters=tuple(Variable(iname) for iname in inames)) + other
            subst_name = self.stack.name_generator(based_on='subst')
            self.stack.substitutions[subst_name] = lp.SubstitutionRule(subst_name,
                    inames, rhs)
            return self.copy(name=subst_name)
        elif isinstance(other, ArraySymbol):
            if not self.shape == other.shape:
                raise NotImplementedError('At this moment broadcasting is not '
                        'supported. But support for it will be added soon!')
            inames = tuple(
                   self.stack.name_generator(based_on='i') for _ in
                   self.shape)
            rhs = Variable(self.name)(*tuple(Variable(iname) for
                iname in inames)) + Variable(other.name)(*tuple(Variable(iname) for
                iname in inames))
            subst_name = self.stack.name_generator(based_on='subst')
            self.stack.substitutions[subst_name] = lp.SubstitutionRule(subst_name,
                    inames, rhs)
            return self.copy(name=subst_name)
        else:
            raise NotImplementedError('__add__ for', type(other))

    def __mul__(self, other):
        if isinstance(other, Number):
            inames = tuple(
                   self.stack.name_generator(based_on='i') for _ in
                   self.shape)
            rhs = Call(function=Variable(self.name),
                    parameters=tuple(Variable(iname) for iname in inames)) * other
            subst_name = self.stack.name_generator(based_on='subst')
            self.stack.substitutions[subst_name] = lp.SubstitutionRule(subst_name,
                    inames, rhs)
            return self.copy(name=subst_name)
        elif isinstance(other, ArraySymbol):
            if not self.shape == other.shape:
                raise NotImplementedError('At this moment broadcasting is not '
                        'supported. But support for it will be added soon!')
            inames = tuple(
                   self.stack.name_generator(based_on='i') for _ in
                   self.shape)
            rhs = Variable(self.name)(*tuple(Variable(iname) for
                iname in inames)) * Variable(other.name)(*tuple(Variable(iname) for
                iname in inames))
            subst_name = self.stack.name_generator(based_on='subst')
            self.stack.substitutions[subst_name] = lp.SubstitutionRule(subst_name,
                    inames, rhs)
            return self.copy(name=subst_name)
        else:
            raise NotImplementedError('__mul__ for', type(other))
