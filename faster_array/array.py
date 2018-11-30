import loopy as lp


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

        if stack is None:
            raise TypeError("must pass 'stack' to ArraySymbol")

        kwargs["address_space"] = address_space
        kwargs["stack"] = stack

        super(ArraySymbol, self).__init__(*args, **kwargs)
