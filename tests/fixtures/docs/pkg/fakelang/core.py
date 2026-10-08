"""Core operations of fakelang."""


def builtin(fn):
    return fn


@builtin
def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None):
    """Returns the matrix product of two microscaled blocks.

    `lhs_scale` and `rhs_scale` hold one e8m0 exponent per 32 elements along K; the
    formats are strings: "e2m1", "e4m3", "e5m2", "bf16".
    """
    return lhs


def dot(a, b, acc=None, allow_tf32=True):
    """Returns the matrix product of two blocks; the accumulator is fp32."""
    return a


def program_id(axis):
    return axis


def long_undocumented(x):
    y = x
    y = y + 1
    y = y + 2
    y = y + 3
    y = y + 4
    y = y + 5
    y = y + 6
    y = y + 7
    y = y + 8
    y = y + 9
    y = y + 10
    y = y + 11
    y = y + 12
    y = y + 13
    y = y + 14
    y = y + 15
    y = y + 16
    y = y + 17
    y = y + 18
    y = y + 19
    y = y + 20
    y = y + 21
    y = y + 22
    y = y + 23
    y = y + 24
    return y


def _private_helper(x):
    """Not part of the API."""
    return x


class Tensor:
    """A block of values in registers or shared memory."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def reshape(self, shape):
        """Returns a tensor with the same data and a new shape; the number of elements must
        not change, and the order of the elements is kept."""
        return self

    def to(self, dtype):
        return self
