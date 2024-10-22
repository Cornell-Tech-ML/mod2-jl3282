from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given Scalar values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for logarithm."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for logarithm."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    """$f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Compute the forward pass for multiplication."""
        ctx.save_for_backward(x, y)
        c = x * y
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for multiplication."""
        x, y = ctx.saved_values
        return d_output * y, d_output * x


class Inv(ScalarFunction):
    """$f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass for inverse."""
        ctx.save_for_backward(x)
        return operators.inv(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for inverse."""
        (x,) = ctx.saved_values
        return operators.inv_back(x, d_output)


class Neg(ScalarFunction):
    """$f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass for negation."""
        return -x

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for negation."""
        return -d_output


class Sigmoid(ScalarFunction):
    """$f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass for Sigmoid."""
        result = operators.sigmoid(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for Sigmoid."""
        sigma: float = ctx.saved_values[0]
        return d_output * sigma * (1 - sigma)


class ReLU(ScalarFunction):
    r"""$f(x) = \max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass for ReLU."""
        ctx.save_for_backward(x)
        return operators.relu(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for ReLU."""
        (x,) = ctx.saved_values
        return operators.relu_back(x, d_output)


class Exp(ScalarFunction):
    """$f(x) = e^{x}$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Compute the forward pass for exponential."""
        result = operators.exp(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for exponential."""
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """$f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Compute the forward pass for less than comparison."""
        return 1.0 if x < y else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for less than comparison."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Compute the forward pass for equality comparison."""
        return 1.0 if x == y else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for equality comparison."""
        return 0.0, 0.0
