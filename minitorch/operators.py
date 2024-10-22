"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(a: float, b: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The product of a and b.

    """
    return a * b


# - id
def id(a: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        a (float): The input number.

    Returns:
    -------
        float: The same input number.

    """
    return a


# - add
def add(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The sum of a and b.

    """
    return a + b


# - neg
def neg(a: float) -> float:
    """Negates a number.

    Args:
    ----
        a (float): The input number.

    Returns:
    -------
        float: The negated value of a.

    """
    return -a


# - lt
def lt(a: float, b: float) -> float:
    """Checks if one number is less than another.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is less than b, otherwise False.

    """
    return 1.0 if a < b else 0.0


# - eq
def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a is equal to b, otherwise False.

    """
    return 1.0 if a == b else 0.0


# - max
def max(a: float, b: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The larger of a and b.

    """
    return a if a > b else b


# - is_close
def is_close(a: float, b: float) -> bool:
    """Checks if two numbers are close in value.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        bool: True if a and b are close within a tolerance, otherwise False.

    """
    return (a - b < 1e-2) and (b - a < 1e-2)


# - sigmoid
def sigmoid(a: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        a (float): The input number.

    Returns:
    -------
        float: The sigmoid of a.

    """
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1.0 + math.exp(a))


# - relu
def relu(a: float) -> float:
    """Applies the ReLU activation function.

    Args:
    ----
        a (float): The input number.

    Returns:
    -------
        float: The output of the ReLU function.

    """
    return a if a > 0 else 0.0


EPS = 1e-6


# - log
def log(a: float) -> float:
    """Calculates the natural logarithm of a number.

    Args:
    ----
        a (float): The input number (must be > 0).

    Returns:
    -------
        float: The natural logarithm of a.

    Raises:
    ------
        ValueError: If a <= 0.

    """
    return math.log(a + EPS)


# - exp
def exp(a: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        a (float): The input number.

    Returns:
    -------
        float: The exponential of a.

    """
    return math.exp(a)


# - log_back
def log_back(a: float, b: float) -> float:
    """Computes the derivatives of log times a second argument.

    Args:
    ----
        a (float): The input number.
        b (float): The second argument.

    Returns:
    -------
        float: The derivative of log(a) multiplied by b.

    """
    return b / (a + EPS)


# - inv
def inv(a: float) -> float:
    """Calculates the reciprocal of a number.

    Args:
    ----
        a (float): The input number (must be != 0).

    Returns:
    -------
        float: The reciprocal of a.

    Raises:
    ------
        ZeroDivisionError: If a == 0.

    """
    return 1.0 / a


# - inv_back
def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second argument.

    Args:
    ----
        a (float): The input number.
        b (float): The second argument.

    Returns:
    -------
        float: The derivative of the reciprocal of a multiplied by b.

    """
    return -(1.0 / a**2) * b


# - relu_back
def relu_back(a: float, b: float) -> float:
    """Computes the derivative of ReLU times a second argument.

    Args:
    ----
        a (float): The input number.
        b (float): The second argument.

    Returns:
    -------
        float: The derivative of ReLU multiplied by b.

    """
    return b if a > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        fn (Callable[[float], float]): A function that takes a float and returns a float.
        a (Iterable[float]): An iterable collection of floats.

    Returns:
    -------
        Iterable[float]: A new iterable with the results of applying fn to each element of a.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


# - zipWith
def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        fn (Callable[[float, float], float]): A function that takes two floats and returns a float.
        a (Iterable[float]): The first iterable collection of floats.
        b (Iterable[float]): The second iterable collection of floats.

    Returns:
    -------
        Iterable[float]: A new iterable with the results of applying fn to corresponding elements of a and b.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


# - reduce
def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function."""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


#
# Use these to implement
# - negList : negate a list
def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map."""
    return map(neg)(ls)


# - addLists : add two lists together
def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith."""
    return zipWith(add)(ls1, ls2)


# - sum: sum lists
def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, 0.0)(ls)


# - prod: take the product of lists
def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce."""
    return reduce(mul, 1.0)(ls)
