# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

"""
This module provides a collection of classes to represent the MySQL functions.
Each class corresponds to a distinct MySQL function (e.g., TO_DAYS, MD5, etc.),
and all of them derive from a common base class `<'SQLFunction'>`.

The function classes in this module are more like a wrapper for the function values
so the 'escape()' method in the 'sqlcycli' package can handle them correctly.

Among all the classes, the `<'RawText'>` class is a special case. It is not a MySQL
function but rather a wrapper for a string value that should not be escaped in a SQL
statement. Using this class requires users to validate the text to prevent potential
SQL injection attacks.

All other classes' values are escaped by the 'escape()' function and can be used
directly in a SQL statement.

For more information about MySQL functions, please refer to the MySQL official
[documentation](https://dev.mysql.com/doc/refman/8.4/en/built-in-function-reference.html).
"""

# Cython imports
import cython
from cython.cimports.cpython.dict import PyDict_Size as dict_len  # type: ignore
from cython.cimports.cpython.set import PySet_Contains as set_contains  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GET_SIZE as tuple_len  # type: ignore
from cython.cimports.sqlcycli.charset import _charsets, Charset  # type: ignore
from cython.cimports.sqlcycli.sqlintvl import SQLInterval, INTERVAL_UNITS  # type: ignore

# Python imports
from typing import Any
import numpy as np, pandas as pd
from sqlcycli.charset import Charset
from sqlcycli.sqlintvl import SQLInterval
from sqlcycli import errors

# Constant -----------------------------------------------------------------------------------------------------------
TRIM_MODES: set[str] = {"BOTH", "LEADING", "TRAILING"}
JSON_VALUE_RETURNING_TYPES: set[str] = {
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "SIGNED",
    "UNSIGNED",
    "DATE",
    "TIME",
    "DATETIME",
    "YEAR",
    "CHAR",
    "JSON",
}


# Base class ---------------------------------------------------------------------------------------------------------
@cython.cclass
class SQLFunction:
    """Represents the base class for MySQL function."""

    _name: str
    _arg_count: cython.int
    _args: tuple
    _kwargs: str
    _sep: str
    _hashcode: cython.Py_ssize_t

    def __init__(
        self,
        function_name: str,
        arg_count: cython.int,
        *args,
        sep: str = ",",
        **kwargs,
    ):
        """The base class for MySQL function.

        :param function_name `<'str'>`: The name of the MySQL function (e.g. 'ABS', 'COUNT', 'SUM')
        :param arg_count `<'int'>`: The number of arguments the function takes, if `-1` count the number of arguments passed.
        :param args `<'*Any'>`: The arguments of the function.
        :param sep `<'str'>`: The seperator between arguments. Defaults to `','`.
        :param kwargs `<'**Any'>`: The keyword arguments of the function.
        """
        self._name = function_name
        if arg_count == -1:
            self._arg_count = tuple_len(args)
        else:
            self._arg_count = arg_count
        self._args = args
        self._kwargs = self._validate_kwargs(kwargs)
        self._sep = sep
        self._hashcode = -1

    @property
    def name(self) -> str:
        """The name of the MySQL function."""
        return self._name

    @property
    def args(self) -> tuple[object]:
        """The arguments of the MySQL function."""
        return self._args

    @cython.cfunc
    @cython.inline(True)
    def _validate_kwargs(self, kwargs: dict) -> str:
        """(internal) Compose the keyword arguments of the function `<'str/None'>."""
        if dict_len(kwargs) == 0:
            return None

        res = []
        for k, v in kwargs.items():
            if v is IGNORED:
                continue
            if not isinstance(v, str):
                raise errors.SQLFunctionError(
                    "The value for argument '%s' must be <'str'> type, "
                    "instead got %s '%r'." % (k, type(v), v)
                )
            res.append("%s %s" % (k, v))
        return " ".join(res) if res else None

    @cython.ccall
    def syntax(self) -> str:
        """Generate the function syntax with the correct 
        placeholders for the arguments (args) `<'str'>`.
        """
        if self._arg_count == 0:
            if self._kwargs is None:
                return self._name + "()"
            return self._name + "(%s)" % self._kwargs
        elif self._arg_count == 1:
            if self._kwargs is None:
                return self._name + "(%s)"
            return self._name + "(%s %s)" % ("%s", self._kwargs)
        else:
            args: str = self._sep.join(["%s" for _ in range(self._arg_count)])
            if self._kwargs is None:
                return "%s(%s)" % (self._name, args)
            return "%s(%s %s)" % (self._name, args, self._kwargs)

    def __repr__(self) -> str:
        if self._kwargs is None:
            return "<SQLFunction: %s(%s)>" % (
                self._name,
                self._sep.join(map(repr, self._args)),
            )
        else:
            return "<SQLFunction: %s(%s %s)>" % (
                self._name,
                self._sep.join(map(repr, self._args)),
                self._kwargs,
            )

    def __str__(self) -> str:
        return self.syntax() % tuple([str(i) for i in self._args])

    def __hash__(self) -> int:
        if self._hashcode == -1:
            self._hashcode = hash(
                (
                    self.__class__.__name__,
                    self._name,
                    self._args,
                    self._kwargs,
                    self._sep,
                )
            )
        return self._hashcode


# Custom class -------------------------------------------------------------------------------------------------------
@cython.cclass
class Sentinel:
    """Represents a sentinel value for SQL functions."""

    def __repr__(self) -> str:
        return "<Sentinel>"


IGNORED: Sentinel = Sentinel()


@cython.cclass
class RawText:
    """Represents the raw text in a SQL statement.

    This class is not a MySQL function, but rather a wrapper for a `<'str'>`
    value, so the 'escape()' function will know not to escape the value but
    instead returns the raw string directly.

    This is usefull for SQL keywords such as 'FROM', 'USING', or a column
    name that should not be escaped in a SQL statement.

    ## Notice
    Since text wrapped in this class is not escaped, it is important to
    validate the text before using it in a SQL statement to prevent SQL
    injection attacks.
    """

    _value: str
    _hashcode: cython.Py_ssize_t

    def __init__(self, value: object):
        """The RawText in a SQL statement.

        This class is not a MySQL function, but rather a wrapper for a `<'str'>`
        value, so the 'escape()' function will know not to escape the value but
        instead returns the raw string directly.

        This is usefull for SQL keywords such as 'FROM', 'USING', or a column
        name that should not be escaped in a SQL statement.

        ## Notice
        Since text wrapped in this class is not escaped, it is important to
        validate the text before using it in a SQL statement to prevent SQL
        injection attacks.

        :param value `<'object'>`: The RawText in a SQL statement, or a string value that should not be escaped in a SQL statement.

        ## Example
        ```python
        from sqlcycli import sqlfunc, escape

        escape("USING utf8mb4", b"utf8")
        # Escape output: "'USING utf8mb4'"

        escape(RawText("USING utf8mb4"), b"utf8")
        # Escape output: "USING utf8mb4"
        ```
        """
        if not isinstance(value, str):
            value = str(value)
        self._value = value
        self._hashcode = -1

    @property
    def value(self) -> str:
        """The raw text string that should not be escaped in a SQL statement `<'str'>."""
        return self._value

    def __repr__(self) -> str:
        return "<RawText: %s>" % self._value

    def __str__(self) -> str:
        return self._value

    def __hash__(self) -> int:
        if self._hashcode == -1:
            self._hashcode = hash((self.__class__.__name__, self._value))
        return self._hashcode


@cython.cclass
class ObjStr:
    """For any subclass of <'ObjStr'>, the 'escape()' function will
    call its '__str__()' method and use the result as the escaped value.

    The '__str__()' method must be implemented in the subclass.
    """

    def __str__(self) -> str:
        raise NotImplementedError("<'%s'> must implement its '__str__()' method")


# Utils --------------------------------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def _validate_args_paris(paris: tuple, func_name: str, pair_name: str) -> cython.bint:
    """(internal) Validate pairs of arguments."""
    size: cython.Py_ssize_t = tuple_len(paris)
    if size == 0:
        raise errors.SQLFunctionError(
            "SQL function '%s' expects at least one pair of "
            "'%s' arguments, instead got 0." % (func_name, pair_name)
        )
    elif size % 2:
        raise errors.SQLFunctionError(
            "SQL function '%s' expects an even number of "
            "'%s' arguments, instead got %d." % (func_name, pair_name, size)
        )
    return True


@cython.cfunc
@cython.inline(True)
def _validate_interval_unit(unit: object, func_name: str) -> str:
    """(internal) Validate the interval unit."""
    if type(unit) is type and issubclass(unit, SQLInterval):
        _unit: SQLInterval = unit("")
        return _unit._name
    if isinstance(unit, SQLInterval):
        _unit: SQLInterval = unit
        return _unit._name
    if isinstance(unit, str):
        _unit_s: str = unit
        if not set_contains(INTERVAL_UNITS, _unit_s):
            _unit_s = _unit_s.upper()
            if not set_contains(INTERVAL_UNITS, _unit_s):
                raise errors.SQLFunctionError(
                    "SQL function '%s' unit argument is invalid '%s'."
                    % (func_name, unit)
                )
        return _unit_s
    raise errors.SQLFunctionError(
        "SQL function '%s' unit expects <'str/SQLInterval'> type, "
        "instead got %s %r." % (func_name, type(unit), unit)
    )


# Functions: Custom --------------------------------------------------------------------------------------------------
@cython.cclass
class RANDINT(SQLFunction):
    """Represents the custom `RANDINT(i, j)` function.

    MySQL description:
    - Return a random integer R in the range i <= R < j,
      through the following expression: FLOOR(i + RAND() * (j - i)).
    """

    def __init__(self, i: cython.longlong, j: cython.longlong):
        """The custom `RANDINT(i, j)` function.

        MySQL description:
        - Return a random integer R in the range i <= R < j,
          through the following expression: FLOOR(i + RAND() * (j - i)).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RANDINT(1, 10)
        # Escape output: "FLOOR(1 + RAND() * 9)"
        # Expect result: 4.0
        ```
        """
        if i > j:
            i, j = j, i
        super().__init__("FLOOR", 1, RawText("%d + RAND() * %d" % (i, j - i)))


# Functions: A -------------------------------------------------------------------------------------------------------
@cython.cclass
class ABS(SQLFunction):
    """Represents the `ABS(X)` function.

    MySQL description:
    - Returns the absolute value of X, or NULL if X is NULL.
    - The result type is derived from the argument type. An implication of this
      is that ABS(-9223372036854775808) produces an error because the result
      cannot be stored in a signed BIGINT value.
    """

    def __init__(self, X):
        """The `ABS(X)` function.

        MySQL description:
        - Returns the absolute value of X, or NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ABS(-1)
        # Escape output: "ABS(-1)"
        # Expect result: 1

        sqlfunc.ABS("-2")
        # Escape output: "ABS('-2')"
        # Expect result: 2
        ```
        """
        super().__init__("ABS", 1, X)


@cython.cclass
class ACOS(SQLFunction):
    """Represents the `ACOS(X)` function.

    MySQL description:
    - Returns the arc cosine of X, that is, the value whose cosine is X.
    - Returns NULL if X is not in the range -1 to 1, or if X is NULL.
    """

    def __init__(self, X):
        """The `ACOS(X)` function.

        MySQL description:
        - Returns the arc cosine of X, that is, the value whose cosine is X.
        - Returns NULL if X is not in the range -1 to 1, or if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ACOS(1)
        # Escape output: "ACOS(1)"
        # Expect result: 0

        sqlfunc.ACOS(1.0001)
        # Escape output: "ACOS(1.0001)"
        # Expect result: NULL

        sqlfunc.ACOS("0")
        # Escape output: "ACOS('0')"
        # Expect result: 1.5707963267949
        ```
        """
        super().__init__("ACOS", 1, X)


@cython.cclass
class ADDDATE(SQLFunction):
    """Represents the `ADDDATE(date, INTERVAL expr unit)` function.

    MySQL description:
    - When invoked with the INTERVAL form of the second argument,
      ADDDATE() is a synonym for DATE_ADD().
    - When invoked with the days form of the second argument, MySQL
      treats it as an integer number of days to be added to expr.
    """

    def __init__(self, date, expr):
        """The `ADDDATE(date, INTERVAL expr unit)` function.

        MySQL description:
        - When invoked with the INTERVAL form of the second argument,
          ADDDATE() is a synonym for DATE_ADD().
        - When invoked with the days form of the second argument, MySQL
          treats it as an integer number of days to be added to expr.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc, sqlintvl

        sqlfunc.ADDDATE("2008-01-02", 31)
        # Escape output: "ADDDATE('2008-01-02',31)"
        # Expect result: '2008-02-02'

        sqlfunc.ADDDATE(datetime.date(2008, 1, 2), sqlintvl.DAY(31))
        # Escape output: "ADDDATE('2008-01-02',INTERVAL 31 DAY)"
        # Expect result: '2008-02-02'

        sqlfunc.ADDDATE(datetime.datetime(2008, 1, 2), sqlintvl.DAY(31))
        # Escape output: "ADDDATE('2008-01-02 00:00:00',INTERVAL 31 DAY)"
        # Expect result: '2008-02-02 00:00:00'
        ```
        """
        super().__init__("ADDDATE", 2, date, expr)


@cython.cclass
class ADDTIME(SQLFunction):
    """Represents the `ADDTIME(expr1, expr2)` function.

    MySQL description:
    - Adds expr2 to expr1 and returns the result.
    - expr1 is a time or datetime expression, and expr2 is a time expression.
    - Returns NULL if expr1or expr2 is NULL.
    """

    def __init__(self, expr1, expr2):
        """The `ADDTIME(expr1, expr2)` function.

        MySQL description:
        - Adds expr2 to expr1 and returns the result.
        - expr1 is a time or datetime expression, and expr2 is a time expression.
        - Returns NULL if expr1 or expr2 is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.ADDTIME("2007-12-31 23:59:59.999999", "1 1:1:1.000002")
        # Escape output: "ADDTIME('2007-12-31 23:59:59.999999','1 1:1:1.000002')"
        # Expect result: '2008-01-02 01:01:01.000001'

        sqlfunc.ADDTIME(datetime.datetime(2007, 12, 31, 23,59, 59, 999999), "1 1:1:1.000002")
        # Escape output: "ADDTIME('2007-12-31 23:59:59.999999','1 1:1:1.000002')"
        # Expect result: '2008-01-02 01:01:01.000001'

        sqlfunc.ADDTIME(datetime.time(1, 0, 0, 999999), "02:00:00.999998")
        # Escape output: "ADDTIME('01:00:00.999999','02:00:00.999998')"
        # Expect result: '03:00:01.999997'
        ```
        """
        super().__init__("ADDTIME", 2, expr1, expr2)


@cython.cclass
class ASCII(SQLFunction):
    """Represents the `ASCII(string)` function.

    MySQL description:
    - Returns the numeric value of the leftmost character of the string str.
    - Returns 0 if str is the empty string. Returns NULL if str is NULL.
    - ASCII() works for 8-bit characters.
    """

    def __init__(self, string):
        """The `ASCII(string)` function.

        MySQL description:
        - Returns the numeric value of the leftmost character of the string str.
        - Returns 0 if str is the empty string. Returns NULL if str is NULL.
        - ASCII() works for 8-bit characters.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ASCII("2")
        # Escape output: "ASCII('2')"
        # Expect result: 50

        sqlfunc.ASCII(2)
        # Escape output: "ASCII(2)"
        # Expect result: 50

        sqlfunc.ASCII("dx")
        # Escape output: "ASCII('dx')"
        # Expect result: 100
        ```
        """
        super().__init__("ASCII", 1, string)


@cython.cclass
class ASIN(SQLFunction):
    """Represents the `ASIN(X)` function.

    MySQL description:
    - Returns the arc sine of X, that is, the value whose sine is X.
    - Returns NULL if X is not in the range -1 to 1, or if X is NULL.
    """

    def __init__(self, X):
        """The `ASIN(X)` function.

        MySQL description:
        - Returns the arc sine of X, that is, the value whose sine is X.
        - Returns NULL if X is not in the range -1 to 1, or if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ASIN(0.2)
        # Escape output: "ASIN(0.2)"
        # Expect result: 0.20135792079033

        sqlfunc.ASIN("0.2")
        # Escape output: "ASIN('0.2')"
        # Expect result: 0.20135792079033
        ```
        """
        super().__init__("ASIN", 1, X)


@cython.cclass
class ATAN(SQLFunction):
    """Represents the `ATAN(X[,Y])` function.

    MySQL description:
    - If both 'X' and 'Y' are invoked: Returns the arc tangent of the two
      variables X and Y. It is similar to calculating the arc tangent of
      X / Y, except that the signs of both arguments are used to determine
      the quadrant of the result. Returns NULL if X or Y is NULL.
    - If only 'X' is invoked: Returns the arc tangent of X, that is, the
      value whose tangent is X. Returns NULL if X is NULL.
    """

    def __init__(self, X, Y: Any | Sentinel = IGNORED):
        """The `ATAN(X[,Y])` function.

        MySQL description:
        - If both 'X' and 'Y' are invoked: Returns the arc tangent of the two
          variables X and Y. It is similar to calculating the arc tangent of
          X / Y, except that the signs of both arguments are used to determine
          the quadrant of the result. Returns NULL if X or Y is NULL.
        - If only 'X' is invoked: Returns the arc tangent of X, that is, the
          value whose tangent is X. Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ATAN(2)
        # Escape output: "ATAN(2)"
        # Expect result: 1.1071487177941

        sqlfunc.ATAN("-2")
        # Escape output: "ATAN('-2')"
        # Expect result: -1.1071487177941

        sqlfunc.ATAN(-2, 2)
        # Escape output: "ATAN(-2,2)"
        # Expect result: -0.78539816339745
        ```
        """
        if Y is IGNORED:
            super().__init__("ATAN", 1, X)
        else:
            super().__init__("ATAN", 2, X, Y)


# Functions: B -------------------------------------------------------------------------------------------------------
@cython.cclass
class BIN(SQLFunction):
    """Represents the `BIN(N)` function.

    MySQL description:
    - Returns a string representation of the binary value of N, where N
      is a longlong (BIGINT) number. This is equivalent to CONV(N,10,2).
    - Returns NULL if N is NULL.
    """

    def __init__(self, N):
        """The `BIN(N)` function.

        MySQL description:
        - Returns a string representation of the binary value of N, where N
          is a longlong (BIGINT) number. This is equivalent to CONV(N,10,2).
        - Returns NULL if N is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.BIN(12)
        # Escape output: "BIN(12)"
        # Expect result: '1100'

        sqlfunc.BIN("12")
        # Escape output: "BIN('12')"
        # Expect result: '1100'
        ```
        """
        super().__init__("BIN", 1, N)


@cython.cclass
class BIN_TO_UUID(SQLFunction):
    """Represents the `BIN_TO_UUID(binary_uuid[, swap_flag])` function.

    MySQL description:
    - BIN_TO_UUID() is the inverse of UUID_TO_BIN(). It converts a binary UUID
      to a string UUID and returns the result. The binary value should be a UUID
      as a VARBINARY(16) value. The return value is a string of five hexadecimal
      numbers separated by dashes.
    - If the UUID argument is NULL, the return value is NULL. If any argument is
      invalid, an error occurs.
    - For optional swap_flag argument, If swap_flag is 0, the two-argument form
      is equivalent to the one-argument form. The string result is in the same
      order as the binary argument. If swap_flag is 1, the UUID value is assumed
      to have its time-low and time-high parts swapped. These parts are swapped
      back to their original position in the result value.
    """

    def __init__(self, binary_uuid, swap_flag: int | Sentinel = IGNORED):
        """The `BIN_TO_UUID(binary_uuid[, swap_flag])` function.

        MySQL description:
        - BIN_TO_UUID() is the inverse of UUID_TO_BIN(). It converts a binary UUID
          to a string UUID and returns the result. The binary value should be a UUID
          as a VARBINARY(16) value. The return value is a string of five hexadecimal
          numbers separated by dashes.
        - If the UUID argument is NULL, the return value is NULL. If any argument is
          invalid, an error occurs.
        - For optional swap_flag argument, If swap_flag is 0, the two-argument form
          is equivalent to the one-argument form. The string result is in the same
          order as the binary argument. If swap_flag is 1, the UUID value is assumed
          to have its time-low and time-high parts swapped. These parts are swapped
          back to their original position in the result value.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        uuid = "6ccd780c-baba-1026-9564-5b8c656024db"

        sqlfunc.BIN_TO_UUID(sqlfunc.UUID_TO_BIN(uuid))
        # Escape output: "BIN_TO_UUID(UUID_TO_BIN('6ccd780c-baba-1026-9564-5b8c656024db'))"
        # Expect result: '6ccd780c-baba-1026-9564-5b8c656024db'

        sqlfunc.BIN_TO_UUID(sqlfunc.UUID_TO_BIN(uuid), 1)
        # Escape output: "BIN_TO_UUID(UUID_TO_BIN('6ccd780c-baba-1026-9564-5b8c656024db'),1)"
        # Expect result: 'baba1026-780c-6ccd-9564-5b8c656024db'
        ```
        """
        if swap_flag is IGNORED:
            super().__init__("BIN_TO_UUID", 1, binary_uuid)
        else:
            super().__init__("BIN_TO_UUID", 2, binary_uuid, swap_flag)


@cython.cclass
class BIT_COUNT(SQLFunction):
    """Represents the `BIT_COUNT(N)` function.

    MySQL description:
    - Returns the number of bits that are set in the argument N as
      an unsigned 64-bit integer, or NULL if the argument is NULL.
    """

    def __init__(self, N):
        """The `BIT_COUNT(N)` function.

        MySQL description:
        - Returns the number of bits that are set in the argument N as
          an unsigned 64-bit integer, or NULL if the argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.BIT_COUNT(64)
        # Escape output: "BIT_COUNT(64)"
        # Expect result: 1

        sqlfunc.BIT_COUNT(b"64")
        # Escape output: "BIT_COUNT(_binary'64')"
        # Expect result: 7
        ```
        """
        super().__init__("BIT_COUNT", 1, N)


@cython.cclass
class BIT_LENGTH(SQLFunction):
    """Represents the `BIT_LENGTH(string)` function.

    MySQL description:
    - Returns the length of the string str in bits.
    - Returns NULL if str is NULL.
    """

    def __init__(self, string):
        """The `BIT_LENGTH(string)` function.

        MySQL description:
        - Returns the length of the string str in bits.
        - Returns NULL if str is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.BIT_LENGTH("text")
        # Escape output: "BIT_LENGTH('text')"
        # Expect result: 32
        ```
        """
        super().__init__("BIT_LENGTH", 1, string)


# Functions: C -------------------------------------------------------------------------------------------------------
@cython.cclass
class CEIL(SQLFunction):
    """Represents the `CEIL(X)` function.

    MySQL description:
    - Returns the smallest integer value not less than X.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `CEIL(X)` function.

        MySQL description:
        - Returns the smallest integer value not less than X.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CEIL(1.23)
        # Escape output: "CEIL(1.23)"
        # Expect result: 2

        sqlfunc.CEIL("-1.23")
        # Escape output: "CEIL('-1.23')"
        # Expect result: -1
        ```
        """
        super().__init__("CEIL", 1, X)


@cython.cclass
class CHAR(SQLFunction):
    """Represents the `CHAR(N,... [USING charset_name])` function.

    MySQL description:
    - CHAR() interprets each argument N as an integer and returns a string
      consisting of the characters given by the code values of those integers.
    - NULL values are skipped.
    """

    def __init__(self, *N, using: str | Sentinel = IGNORED):
        """The `CHAR(N,... [USING charset_name])` function.

        MySQL description:
        - CHAR() interprets each argument N as an integer and returns a string
          consisting of the characters given by the code values of those integers.
        - NULL values are skipped.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CHAR(77, 121, 83, 81, "76")
        # Escape output: "CHAR(77,121,83,81,'76')"
        # Expect result: 0x4D7953514C

        sqlfunc.CHAR(77, 121, 83, 81, "76", using="utf8mb4")
        # Escape output: "CHAR(77,121,83,81,'76' USING utf8mb4)"
        # Expect result: 'MySQL'
        """
        if using is IGNORED:
            super().__init__("CHAR", -1, *N)
        else:
            try:
                _ch: Charset = _charsets.by_name(using)
            except Exception as err:
                raise errors.SQLFunctionError(
                    "SQL function CHAR argument 'using' is invalid, %s" % err
                ) from err
            super().__init__("CHAR", -1, *N, USING=_ch._name)


@cython.cclass
class CHAR_LENGTH(SQLFunction):
    """Represents the `CHAR_LENGTH(string)` function.

    MySQL description:
    - Returns the length of the string str, measured in code points.
    - A multibyte character counts as a single code point. This means that,
      for a string containing two 3-byte characters, LENGTH() returns 6,
      whereas CHAR_LENGTH() returns 2
    """

    def __init__(self, string):
        """The `CHAR_LENGTH(string)` function.

        MySQL description:
        - Returns the length of the string str, measured in code points.
        - A multibyte character counts as a single code point. This means that,
          for a string containing two 3-byte characters, LENGTH() returns 6,
          whereas CHAR_LENGTH() returns 2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CHAR_LENGTH("text")
        # Escape output: "CHAR_LENGTH('text')"
        # Expect result: 4

        sqlfunc.CHAR_LENGTH("海豚")
        # Escape output: "CHAR_LENGTH('海豚')"
        # Expect result: 2
        ```
        """
        super().__init__("CHAR_LENGTH", 1, string)


@cython.cclass
class CHARSET(SQLFunction):
    """Represents the `CHARSET(string)` function.

    MySQL description:
    - Returns the character set of the string argument.
    - Returns NULL if the argument is NULL.
    """

    def __init__(self, string):
        """The `CHARSET(string)` function.

        MySQL description:
        - Returns the character set of the string argument.
        - Returns NULL if the argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CHARSET("abs")
        # Escape output: "CHARSET('abs')"
        # Expect result: 'utf8mb3'

        sqlfunc.CHARSET(sqlfunc.CONVERT("abc", "latin1"))
        # Escape output: "CHARSET(CONVERT('abc' USING latin1))"
        # Expect result: 'latin1'
        ```
        """
        super().__init__("CHARSET", 1, string)


@cython.cclass
class COALESCE(SQLFunction):
    """Represents the `COALESCE(value, ...)` function.

    MySQL description:
    - Returns the first non-NULL value in the list, or NULL if there are no non-NULL values.
    - The return type of COALESCE() is the aggregated type of the argument types.
    """

    def __init__(self, *values):
        """The `COALESCE(X, Y, ...)` function.

        MySQL description:
        - Returns the first non-NULL value in the list, or NULL if there are no non-NULL values.
        - The return type of COALESCE() is the aggregated type of the argument types.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.COALESCE(None, 1)
        # Escape output: "COALESCE(NULL,1)"
        # Expect result: 1

        sqlfunc.COALESCE(None, None, None)
        # Escape output: "COALESCE(NULL,NULL,NULL)"
        # Expect result: NULL
        ```
        """
        super().__init__("COALESCE", -1, *values)


@cython.cclass
class COERCIBILITY(SQLFunction):
    """Represents the `COERCIBILITY(string)` function.

    MySQL description:
    - Returns the collation coercibility value of the string argument.
    """

    def __init__(self, string):
        """The `COERCIBILITY(string)` function.

        MySQL description:
        - Returns the collation coercibility value of the string argument.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.COERCIBILITY("abc")
        # Escape output: "COERCIBILITY('abc')"
        # Expect result: 4
        ```
        """
        super().__init__("COERCIBILITY", 1, string)


@cython.cclass
class COLLATION(SQLFunction):
    """Represents the `COLLATION(string)` function.

    MySQL description:
    - Returns the collation of the string argument.
    """

    def __init__(self, string):
        """The `COLLATION(string)` function.

        MySQL description:
        - Returns the collation of the string argument.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.COLLATION("abc")
        # Escape output: "COLLATION('abc')"
        # Expect result: 'utf8mb4_0900_ai_ci'
        ```
        """
        super().__init__("COLLATION", 1, string)


@cython.cclass
class COMPRESS(SQLFunction):
    """Represents the `COMPRESS(string)` function.

    MySQL description:
    - Compresses a string and returns the result as a binary string.
    - This function requires MySQL to have been compiled with a compression
      library such as zlib. Otherwise, the return value is always NULL.
      The return value is also NULL if string_to_compress is NULL.
    - The compressed string can be uncompressed with UNCOMPRESS().
    """

    def __init__(self, string):
        """The `COMPRESS(string)` function.

        MySQL description:
        - Compresses a string and returns the result as a binary string.
        - This function requires MySQL to have been compiled with a compression
          library such as zlib. Otherwise, the return value is always NULL.
          The return value is also NULL if string_to_compress is NULL.
        - The compressed string can be uncompressed with UNCOMPRESS().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LENGTH(sqlfunc.COMPRESS(""))
        # Escape output: "LENGTH(COMPRESS(''))"
        # Expect result: 0

        sqlfunc.LENGTH(sqlfunc.COMPRESS("a"))
        # Escape output: "LENGTH(COMPRESS('a'))"
        # Expect result: 13
        ```
        """
        super().__init__("COMPRESS", 1, string)


@cython.cclass
class CONCAT(SQLFunction):
    """Represents the `CONCAT(str1, str2, ...)` function.

    MySQL description:
    - Returns the string that results from concatenating the arguments.
    - May have one or more arguments. If all arguments are nonbinary strings,
      the result is a nonbinary string. If the arguments include any binary
      strings, the result is a binary string. A numeric argument is converted
      to its equivalent nonbinary string form.
    - Returns NULL if any argument is NULL.
    """

    def __init__(self, *strings):
        """The `CONCAT(str1, str2, ...)` function.

        MySQL description:
        - Returns the string that results from concatenating the arguments.
        - May have one or more arguments. If all arguments are nonbinary strings,
          the result is a nonbinary string. If the arguments include any binary
          strings, the result is a binary string. A numeric argument is converted
          to its equivalent nonbinary string form.
        - Returns NULL if any argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CONCAT("My", "S", "QL")
        # Escape output: "CONCAT('My','S','QL')"
        # Expect result: 'MySQL'

        sqlfunc.CONCAT("My", None, "QL")
        # Escape output: "CONCAT('My',NULL,'QL')"
        # Expect result: NULL

        sqlfunc.CONCAT(14.3)
        # Escape output: "CONCAT(14.3)"
        # Expect result: '14.3'
        ```
        """
        super().__init__("CONCAT", -1, *strings)


@cython.cclass
class CONCAT_WS(SQLFunction):
    """Represents the `CONCAT_WS(separator, str1, str2, ...)` function.

    MySQL description:
    - CONCAT_WS() stands for Concatenate With Separator and is a special form of CONCAT().
    - The first argument is the separator for the rest of the arguments. The separator is
      added between the strings to be concatenated. The separator can be a string, as can
      the rest of the arguments.
    - If the separator is NULL, the result is NULL
    """

    def __init__(self, sep, *strings):
        """The `CONCAT_WS(separator, str1, str2, ...)` function.

        MySQL description:
        - CONCAT_WS() stands for Concatenate With Separator and is a special form of CONCAT().
        - The first argument is the separator for the rest of the arguments. The separator is
          added between the strings to be concatenated. The separator can be a string, as can
          the rest of the arguments.
        - If the separator is NULL, the result is NULL

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CONCAT_WS(",", "First name", "Second name", "Last Name")
        # Escape output: "CONCAT_WS(',','First name','Second name','Last Name')"
        # Expect result: 'First name,Second name,Last Name'

        sqlfunc.CONCAT_WS(",", "First name", None, "Last Name")
        # Escape output: "CONCAT_WS(',','First name',NULL,'Last Name')"
        # Expect result: 'First name,Last Name'
        ```
        """
        super().__init__("CONCAT_WS", -1, sep, *strings)


@cython.cclass
class CONNECTION_ID(SQLFunction):
    """Represents the `CONNECTION_ID()` function.

    MySQL description:
    - Returns the connection ID (thread ID) for the connection. Every connection
      has an ID that is unique among the set of currently connected clients.
    - The value returned by CONNECTION_ID() is the same type of value as displayed
      in the ID column of the Information Schema PROCESSLIST table, the Id column
      of SHOW PROCESSLIST output, and the PROCESSLIST_ID column of the Performance
      Schema threads table.
    """

    def __init__(self):
        """The `CONNECTION_ID()` function.

        MySQL description:
        - Returns the connection ID (thread ID) for the connection. Every connection
          has an ID that is unique among the set of currently connected clients.
        - The value returned by CONNECTION_ID() is the same type of value as displayed
          in the ID column of the Information Schema PROCESSLIST table, the Id column
          of SHOW PROCESSLIST output, and the PROCESSLIST_ID column of the Performance
          Schema threads table.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CONNECTION_ID()
        # Escape output: "CONNECTION_ID()"
        # Expect result: 23786
        ```
        """
        super().__init__("CONNECTION_ID", 0)


@cython.cclass
class CONV(SQLFunction):
    """Represents the `CONV(N, from_base, to_base)` function.

    MySQL description:
    - Converts numbers between different number bases. Returns a string
      representation of the number N, converted from base from_base to
      base to_base. Returns NULL if any argument is NULL.
    - The argument N is interpreted as an integer, but may be specified
      as an integer or a string. The minimum base is 2 and the maximum
      base is 36. If from_base is a negative number, N is regarded as a
      signed number. Otherwise, N is treated as unsigned. CONV() works
      with 64-bit precision.
    """

    def __init__(self, N, from_base, to_base):
        """The `CONV(N, from_base, to_base)` function.

        MySQL description:
        - Converts numbers between different number bases. Returns a string
          representation of the number N, converted from base from_base to
          base to_base. Returns NULL if any argument is NULL.
        - The argument N is interpreted as an integer, but may be specified
          as an integer or a string. The minimum base is 2 and the maximum
          base is 36. If from_base is a negative number, N is regarded as a
          signed number. Otherwise, N is treated as unsigned. CONV() works
          with 64-bit precision.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CONV("a",16,2)
        # Escape output: "CONV('a',16,2)"
        # Expect result: '1010'

        sqlfunc.CONV("6E",18,8)
        # Escape output: "CONV('6E',18,8)"
        # Expect result: '172'
        ```
        """
        super().__init__("CONV", 3, N, from_base, to_base)


@cython.cclass
class CONVERT(SQLFunction):
    """Represents the `CONVERT(expr USING transcoding_name)` function.

    MySQL description:
    - Converts data between different character sets. In MySQL, transcoding
      names are the same as the corresponding character set names.
    - Returns NULL if expr is NULL.
    """

    def __init__(self, expr, using):
        """The `CONVERT(expr USING transcoding_name)` function.

        MySQL description:
        - Converts data between different character sets. In MySQL, transcoding
          names are the same as the corresponding character set names.
        - Returns NULL if expr is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CONVERT("MySQL", "utf8mb4")
        # Escape output: "CONVERT('MySQL' USING utf8mb4)"
        # Expect result: 'MySQL'

        sqlfunc.CONVERT("MySQL", "latin1")
        # Escape output: "CONVERT('MySQL' USING latin1)"
        # Expect result: 'MySQL'
        """
        try:
            _ch: Charset = _charsets.by_name(using)
        except Exception as err:
            raise errors.SQLFunctionError(
                "SQL function CONVERT argument 'using' is invalid, %s" % err
            ) from err
        super().__init__("CONVERT", 1, expr, USING=_ch._name)


@cython.cclass
class CONVERT_TZ(SQLFunction):
    """Represents the `CONVERT_TZ(dt, from_tz, to_tz)` function.

    MySQL description:
    - CONVERT_TZ() converts a datetime value dt from the time zone given by from_tz
      to the time zone given by to_tz and returns the resulting value.
    - Returns NULL if any of the arguments are invalid, or if any of them are NULL.
    """

    def __init__(self, dt, from_tz, to_tz):
        """The `CONVERT_TZ(dt, from_tz, to_tz)` function.

        MySQL description:
        - CONVERT_TZ() converts a datetime value dt from the time zone given by from_tz
          to the time zone given by to_tz and returns the resulting value.
        - Returns NULL if any of the arguments are invalid, or if any of them are NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.CONVERT_TZ("2004-01-01 12:00:00", "+00:00", "+10:00")
        # Escape output: "CONVERT_TZ('2004-01-01 12:00:00','+00:00','+10:00')"
        # Expect result: '2004-01-01 22:00:00'

        sqlfunc.CONVERT_TZ(datetime.datetime(2004, 1, 1, 12), "+00:00", "+10:00")
        # Escape output: "CONVERT_TZ('2004-01-01 12:00:00','+00:00','+10:00')"
        # Expect result: '2004-01-01 22:00:00'
        ```
        """
        super().__init__("CONVERT_TZ", 3, dt, from_tz, to_tz)


@cython.cclass
class COS(SQLFunction):
    """Represents the `COS(X)` function.

    MySQL description:
    - Returns the cosine of X, where X is given in radians.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `COS(X)` function.

        MySQL description:
        - Returns the cosine of X, where X is given in radians.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.COS(0)
        # Escape output: "COS(0)"
        # Expect result: 1

        sqlfunc.COS(sqlfunc.PI())
        # Escape output: "COS(PI())"
        # Expect result: -1
        ```
        """
        super().__init__("COS", 1, X)


@cython.cclass
class COT(SQLFunction):
    """Represents the `COT(X)` function.

    MySQL description:
    - Returns the cotangent of X.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `COT(X)` function.

        MySQL description:
        - Returns the cotangent of X.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.COT(12)
        # Escape output: "COT(12)"
        # Expect result: -1.5726734063977

        sqlfunc.COT("12")
        # Escape output: "COT('12')"
        # Expect result: -1.5726734063977
        ```
        """
        super().__init__("COT", 1, X)


@cython.cclass
class CRC32(SQLFunction):
    """Represents the `CRC32(expr)` function.

    MySQL description:
    - Computes a cyclic redundancy check value and returns a 32-bit
      unsigned value. The result is NULL if the argument is NULL.
    - The argument is expected to be a string and (if possible) is
      treated as one if it is not.
    """

    def __init__(self, expr):
        """The `CRC32(expr)` function.

        MySQL description:
        - Computes a cyclic redundancy check value and returns a 32-bit
          unsigned value. The result is NULL if the argument is NULL.
        - The argument is expected to be a string and (if possible) is
          treated as one if it is not.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CRC32("MySQL")
        # Escape output: "CRC32('MySQL')"
        # Expect result: 3259397556

        sqlfunc.CRC32("mysql")
        # Escape output: "CRC32('mysql')"
        # Expect result: 2501908538
        ```
        """
        super().__init__("CRC32", 1, expr)


@cython.cclass
class CUME_DIST(SQLFunction):
    """Represents the `CUME_DIST() OVER (window_spec)` function.

    MySQL description:
    - Returns the cumulative distribution of a value within a group of values;
      that is, the percentage of partition values less than or equal to the
      value in the current row. This represents the number of rows preceding
      or peer with the current row in the window ordering of the window
      partition divided by the total number of rows in the window partition.
      Return values range from 0 to 1.
    - This function should be used with ORDER BY to sort partition rows into
      the desired order. Without ORDER BY, all rows are peers and have value
      N/N = 1, where N is the partition size.
    """

    def __init__(self):
        """The `CUME_DIST() OVER (window_spec)` function.

        MySQL description:
        - Returns the cumulative distribution of a value within a group of values;
          that is, the percentage of partition values less than or equal to the
          value in the current row. This represents the number of rows preceding
          or peer with the current row in the window ordering of the window
          partition divided by the total number of rows in the window partition.
          Return values range from 0 to 1.
        - This function should be used with ORDER BY to sort partition rows into
          the desired order. Without ORDER BY, all rows are peers and have value
          N/N = 1, where N is the partition size.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CUME_DIST()
        # Escape output: "CUME_DIST()"
        ```
        """
        super().__init__("CUME_DIST", 0)


@cython.cclass
class CURRENT_DATE(SQLFunction):
    """Represents the `CURRENT_DATE()` function.

    MySQL description:
    - Returns the current date as a value in 'YYYY-MM-DD' or YYYYMMDD format,
      depending on whether the function is used in string or numeric context.
    """

    def __init__(self):
        """The `CURRENT_DATE()` function.

        MySQL description:
        - Returns the current date as a value in 'YYYY-MM-DD' or YYYYMMDD format,
          depending on whether the function is used in string or numeric context.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CURRENT_DATE()
        # Escape output: "CURRENT_DATE()"
        # Expect result: '2008-06-13'
        ```
        """
        super().__init__("CURRENT_DATE", 0)


@cython.cclass
class CURRENT_ROLE(SQLFunction):
    """Represents the `CURRENT_ROLE()` function.

    MySQL description:
    - Returns a utf8mb3 string containing the current active roles for the current
      session, separated by commas, or NONE if there are none.
    - The value reflects the setting of the sql_quote_show_create system variable.
    """

    def __init__(self):
        """The `CURRENT_ROLE()` function.

        MySQL description:
        - Returns a utf8mb3 string containing the current active roles for the current
          session, separated by commas, or NONE if there are none.
        - The value reflects the setting of the sql_quote_show_create system variable.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CURRENT_ROLE()
        # Escape output: "CURRENT_ROLE()"
        ```
        """
        super().__init__("CURRENT_ROLE", 0)


@cython.cclass
class CURRENT_TIME(SQLFunction):
    """Represents the `CURRENT_TIME([fps])` function.

    MySQL description:
    - Returns the current time as a value in 'hh:mm:ss' or hhmmss format,
      depending on whether the function is used in string or numeric context.
      The value is expressed in the session time zone.

    - If the fsp argument is given to specify a fractional seconds precision
      from 0 to 6, the return value includes a fractional seconds part of that
      many digits.
    """

    def __init__(self, fsp: Any | Sentinel = IGNORED):
        """The `CURRENT_TIME([fps])` function.

        MySQL description:
        - Returns the current time as a value in 'hh:mm:ss' or hhmmss format,
          depending on whether the function is used in string or numeric context.
          The value is expressed in the session time zone.

        - If the fsp argument is given to specify a fractional seconds precision
          from 0 to 6, the return value includes a fractional seconds part of that
          many digits.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CURRENT_TIME()
        # Escape output: "CURRENT_TIME()"
        # Expect result: '19:25:37'

        sqlfunc.CURRENT_TIME(3)
        # Escape output: "CURRENT_TIME(3)"
        # Expect result: '19:25:37.840'
        ```
        """
        if fsp is IGNORED:
            super().__init__("CURRENT_TIME", 0)
        else:
            super().__init__("CURRENT_TIME", 1, fsp)


@cython.cclass
class CURRENT_TIMESTAMP(SQLFunction):
    """Represents the `CURRENT_TIMESTAMP([fps])` function.

    MySQL description:
    - Returns the current date and time as a value in 'YYYY-MM-DD hh:mm:ss' or
      YYYYMMDDhhmmss format, depending on whether the function is used in string
      or numeric context. The value is expressed in the session time zone.
    - If the fsp argument is given to specify a fractional seconds precision from
      0 to 6, the return value includes a fractional seconds part of that many digits.
    """

    def __init__(self, fsp: Any | Sentinel = IGNORED):
        """The `CURRENT_TIMESTAMP([fps])` function.

        MySQL description:
        - Returns the current date and time as a value in 'YYYY-MM-DD hh:mm:ss' or
          YYYYMMDDhhmmss format, depending on whether the function is used in string
          or numeric context. The value is expressed in the session time zone.
        - If the fsp argument is given to specify a fractional seconds precision from
          0 to 6, the return value includes a fractional seconds part of that many digits.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CURRENT_TIMESTAMP()
        # Escape output: "CURRENT_TIMESTAMP()"
        # Expect result: '2021-12-25 19:25:37'

        sqlfunc.CURRENT_TIMESTAMP(3)
        # Escape output: "CURRENT_TIMESTAMP(3)"
        # Expect result: '2021-12-25 19:25:37.840'
        ```
        """
        if fsp is IGNORED:
            super().__init__("CURRENT_TIMESTAMP", 0)
        else:
            super().__init__("CURRENT_TIMESTAMP", 1, fsp)


@cython.cclass
class CURRENT_USER(SQLFunction):
    """Represents the `CURRENT_USER()` function.

    MySQL description:
    - Returns the user name and host name combination for the MySQL account that the
      server used to authenticate the current client. This account determines your
      access privileges. The return value is a string in the utf8mb3 character set.
    - The value of CURRENT_USER() can differ from the value of USER().
    """

    def __init__(self):
        """The `CURRENT_USER()` function.

        MySQL description:
        - Returns the user name and host name combination for the MySQL account that the
          server used to authenticate the current client. This account determines your
          access privileges. The return value is a string in the utf8mb3 character set.
        - The value of CURRENT_USER() can differ from the value of USER().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.CURRENT_USER()
        # Escape output: "CURRENT_USER()"
        # Expect result: 'root@localhost'
        ```
        """
        super().__init__("CURRENT_USER", 0)


# Functions: D -------------------------------------------------------------------------------------------------------
@cython.cclass
class DATABASE(SQLFunction):
    """Represents the `DATABASE()` function.

    MySQL description:
    - Returns the default (current) database name as a string in the utf8mb3 character set.
    - If there is no default database, DATABASE() returns NULL. Within a stored routine,
      the default database is the database that the routine is associated with, which is
      not necessarily the same as the database that is the default in the calling context.
    """

    def __init__(self):
        """The `DATABASE()` function.

        MySQL description:
        - Returns the default (current) database name as a string in the utf8mb3 character set.
        - If there is no default database, DATABASE() returns NULL. Within a stored routine,
          the default database is the database that the routine is associated with, which is
          not necessarily the same as the database that is the default in the calling context.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.DATABASE()
        # Escape output: "DATABASE()"
        # Expect result: 'test'
        ```
        """
        super().__init__("DATABASE", 0)


@cython.cclass
class DATE(SQLFunction):
    """Represents the `DATE(expr)` function.

    MySQL description:
    - Extracts the date part of the date or datetime expression expr.
    - Returns NULL if expr is NULL.
    """

    def __init__(self, expr):
        """The `DATE(expr)` function.

        MySQL description:
        - Extracts the date part of the date or datetime expression expr.
        - Returns NULL if expr is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.DATE("2003-12-31 01:02:03")
        # Escape output: "DATE('2003-12-31 01:02:03')"
        # Expect result: '2003-12-31'

        sqlfunc.DATE(datetime.datetime(2003, 12, 31, 1, 2, 3))
        # Escape output: "DATE('2003-12-31 01:02:03')"
        # Expect result: '2003-12-31'
        ```
        """
        super().__init__("DATE", 1, expr)


@cython.cclass
class DATE_ADD(SQLFunction):
    """Represents the `DATE_ADD(date, INTERVAL expr unit)` function.

    MySQL description:
    - These functions perform date arithmetic. The date argument specifies the starting date
      or datetime value. expr is an expression specifying the interval value to be added or
      subtracted from the starting date. expr is evaluated as a string; it may start with a
      - for negative intervals. unit is a keyword indicating the units in which the expression
      should be interpreted.
    """

    def __init__(self, date, expr):
        """The `DATE_ADD(date, INTERVAL expr unit)` function.

        MySQL description:
        - These functions perform date arithmetic. The date argument specifies the starting date
          or datetime value. expr is an expression specifying the interval value to be added or
          subtracted from the starting date. expr is evaluated as a string; it may start with a
          '-' for negative intervals. unit is a keyword indicating the units in which the expression
          should be interpreted.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc, sqlintvl

        sqlfunc.DATE_ADD("2021-01-01", sqlintvl.WEEK(1))
        # Escape output: "DATE_ADD('2021-01-01',INTERVAL 1 WEEK)"
        # Expect result: '2021-01-08'

        sqlfunc.DATE_ADD(datetime.date(2021, 1, 1), sqlintvl.DAY(1))
        # Escape output: "DATE_ADD('2021-01-01',INTERVAL 1 DAY)"
        # Expect result: '2021-01-02'
        ```
        """
        super().__init__("DATE_ADD", 2, date, expr)


@cython.cclass
class DATE_FORMAT(SQLFunction):
    """Represents the `DATE_FORMAT(date, format)` function.

    MySQL description:
    - Formats the date value according to the format string. If either
      argument is NULL, the function returns NULL.
    - The % character is required before format specifier characters.
    - For MySQL format specifiers, please refer to
      [link](https://dev.mysql.com/doc/refman/8.4/en/date-and-time-functions.html#function_date-format).
    """

    def __init__(self, date, format):
        """The `DATE_FORMAT(date, format)` function.

        MySQL description:
        - Formats the date value according to the format string. If either
          argument is NULL, the function returns NULL.
        - The % character is required before format specifier characters.
        - For MySQL format specifiers, please refer to
          [link](https://dev.mysql.com/doc/refman/8.4/en/date-and-time-functions.html#function_date-format).

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.DATE_FORMAT("2009-10-04 22:23:00", "%W %M %Y")
        # Escape output: "DATE_FORMAT('2009-10-04 22:23:00','%W %M %Y')"
        # Expect result: 'Sunday October 2009'

        sqlfunc.DATE_FORMAT(datetime.datetime(2007, 10, 4, 22, 23), "%H:%i:%s")
        # Escape output: "DATE_FORMAT('2007-10-04 22:23:00','%H:%i:%s')"
        # Expect result: '22:23:00'
        ```
        """
        super().__init__("DATE_FORMAT", 2, date, format)


@cython.cclass
class DATE_SUB(SQLFunction):
    """Represents the `DATE_SUB(date, INTERVAL expr unit)` function.

    MySQL description:
    - These functions perform date arithmetic. The date argument specifies the starting date
      or datetime value. expr is an expression specifying the interval value to be added or
      subtracted from the starting date. expr is evaluated as a string; it may start with a
      '-' or negative intervals. unit is a keyword indicating the units in which the expression
      should be interpreted.
    """

    def __init__(self, date, expr):
        """The `DATE_SUB(date, INTERVAL expr unit)` function.

        MySQL description:
        - These functions perform date arithmetic. The date argument specifies the starting date
          or datetime value. expr is an expression specifying the interval value to be added or
          subtracted from the starting date. expr is evaluated as a string; it may start with a
          '-' for negative intervals. unit is a keyword indicating the units in which the expression
          should be interpreted.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc, sqlintvl

        sqlfunc.DATE_SUB("2021-01-08", sqlintvl.WEEK(1))
        # Escape output: "DATE_SUB('2021-01-08',INTERVAL 1 WEEK)"
        # Expect result: '2021-01-01'

        sqlfunc.DATE_SUB(datetime.date(2021, 1, 2), sqlintvl.DAY(1))
        # Escape output: "DATE_SUB('2021-01-02',INTERVAL 1 DAY)"
        # Expect result: '2021-01-01'
        ```
        """
        super().__init__("DATE_SUB", 2, date, expr)


@cython.cclass
class DATEDIFF(SQLFunction):
    """Represents the `DATEDIFF(expr1, expr2)` function.

    MySQL description:
    - DATEDIFF() returns expr1 - expr2 expressed as a value in days from
      one date to the other. expr1 and expr2 are date or date-and-time
      expressions.
    - Only the date parts of the values are used in the calculation.
    """

    def __init__(self, expr1, expr2):
        """The `DATEDIFF(expr1, expr2)` function.

        MySQL description:
        - DATEDIFF() returns expr1 - expr2 expressed as a value in days from
          one date to the other. expr1 and expr2 are date or date-and-time
          expressions.
        - Only the date parts of the values are used in the calculation.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.DATEDIFF("2007-12-31 23:59:59", "2007-12-30")
        # Escape output: "DATEDIFF('2007-12-31 23:59:59','2007-12-30')"
        # Expect result: 1

        sqlfunc.DATEDIFF(datetime.datetime(2010, 11, 30, 23, 59, 59), "2010-12-31")
        # Escape output: "DATEDIFF('2010-11-30 23:59:59','2010-12-31')"
        # Expect result: -31
        ```
        """
        super().__init__("DATEDIFF", 2, expr1, expr2)


@cython.cclass
class DAYNAME(SQLFunction):
    """Represents the `DAYNAME(date)` function.

    MySQL description:
    - Returns the name of the weekday for date. The language used for the
      name is controlled by the value of the lc_time_names system variable.
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `DAYNAME(date)` function.

        MySQL description:
        - Returns the name of the weekday for date. The language used for the
          name is controlled by the value of the lc_time_names system variable.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.DAYNAME("2007-02-03")
        # Escape output: "DAYNAME('2007-02-03')"
        # Expect result: 'Saturday'

        sqlfunc.DAYNAME(datetime.date(2007, 2, 3))
        # Escape output: "DAYNAME('2007-02-03')"
        # Expect result: 'Saturday'
        ```
        """
        super().__init__("DAYNAME", 1, date)


@cython.cclass
class DAYOFMONTH(SQLFunction):
    """Represents the `DAYOFMONTH(date)` function.

    MySQL description:
    - Returns the day of the month for date, in the range 1 to 31, or 0 for
      dates such as '0000-00-00' or '2008-00-00' that have a zero day part.
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `DAYOFMONTH(date)` function.

        MySQL description:
        - Returns the day of the month for date, in the range 1 to 31, or 0 for
          dates such as '0000-00-00' or '2008-00-00' that have a zero day part.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.DAYOFMONTH("2007-02-03")
        # Escape output: "DAYOFMONTH('2007-02-03')"
        # Expect result: 3

        sqlfunc.DAYOFMONTH(datetime.date(2007, 2, 3))
        # Escape output: "DAYOFMONTH('2007-02-03')"
        # Expect result: 3
        ```
        """
        super().__init__("DAYOFMONTH", 1, date)


@cython.cclass
class DAYOFWEEK(SQLFunction):
    """Represents the `DAYOFWEEK(date)` function.

    MySQL description:
    - Returns the weekday index for date (1 = Sunday, 2 = Monday, …, 7 = Saturday).
      These index values correspond to the ODBC standard.
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `DAYOFWEEK(date)` function.

        MySQL description:
        - Returns the weekday index for date (1 = Sunday, 2 = Monday, …, 7 = Saturday).
          These index values correspond to the ODBC standard.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.DAYOFWEEK("2007-02-03")
        # Escape output: "DAYOFWEEK('2007-02-03')"
        # Expect result: 7

        sqlfunc.DAYOFWEEK(datetime.date(2007, 2, 3))
        # Escape output: "DAYOFWEEK('2007-02-03')"
        # Expect result: 7
        ```
        """
        super().__init__("DAYOFWEEK", 1, date)


@cython.cclass
class DAYOFYEAR(SQLFunction):
    """Represents the `DAYOFYEAR(date)` function.

    MySQL description:
    - Returns the day of the year for date, in the range 1 to 366.
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `DAYOFYEAR(date)` function.

        MySQL description:
        - Returns the day of the year for date, in the range 1 to 366.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.DAYOFYEAR("2007-02-03")
        # Escape output: "DAYOFYEAR('2007-02-03')"
        # Expect result: 34

        sqlfunc.DAYOFYEAR(datetime.date(2007, 2, 3))
        # Escape output: "DAYOFYEAR('2007-02-03')"
        # Expect result: 34
        ```
        """
        super().__init__("DAYOFYEAR", 1, date)


@cython.cclass
class DEGREES(SQLFunction):
    """Represents the `DEGREES(X)` function.

    MySQL description:
    - Returns the argument X, converted from radians to degrees.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `DEGREES(X)` function.

        MySQL description:
        - Returns the argument X, converted from radians to degrees.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.DEGREES(sqlfunc.PI())
        # Escape output: "DEGREES(PI())"
        # Expect result: 180
        ```
        """
        super().__init__("DEGREES", 1, X)


@cython.cclass
class DENSE_RANK(SQLFunction):
    """Represents the `DENSE_RANK() OVER (window_spec)` function.

    MySQL description:
    - Returns the rank of the current row within its partition, without gaps.
      Peers are considered ties and receive the same rank. This function assigns
      consecutive ranks to peer groups; the result is that groups of size greater
      than one do not produce noncontiguous rank numbers. For an example, see the
      RANK() function description.
    """

    def __init__(self):
        """The `DENSE_RANK() OVER (window_spec)` function.

        MySQL description:
        - Returns the rank of the current row within its partition, without gaps.
          Peers are considered ties and receive the same rank. This function assigns
          consecutive ranks to peer groups; the result is that groups of size greater
          than one do not produce noncontiguous rank numbers. For an example, see the
          RANK() function description.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.DENSE_RANK()
        # Escape output: "DENSE_RANK()"
        ```
        """
        super().__init__("DENSE_RANK", 0)


# Functions: E -------------------------------------------------------------------------------------------------------
@cython.cclass
class ELT(SQLFunction):
    """Represents the `ELT(N, str1, str2, ...)` function.

    MySQL description:
    - ELT() returns the Nth element of the list of strings: str1 if N = 1,
      str2 if N = 2, and so on.
    - Returns NULL if N is less than 1, greater than the number of arguments,
      or NULL. ELT() is the complement of FIELD().
    """

    def __init__(self, N, *strings):
        """The `ELT(N, str1, str2, ...)` function.

        MySQL description:
        - ELT() returns the Nth element of the list of strings: str1 if N = 1,
          str2 if N = 2, and so on.
        - Returns NULL if N is less than 1, greater than the number of arguments,
          or NULL. ELT() is the complement of FIELD().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ELT(1, "Aa", "Bb", "Cc", "Dd")
        # Escape output: "ELT(1,'Aa','Bb','Cc','Dd')"
        # Expect result: 'Aa'

        sqlfunc.ELT(4, "Aa", "Bb", "Cc", "Dd")
        # Escape output: "ELT(4,'Aa','Bb','Cc','Dd')"
        # Expect result: 'Dd'
        ```
        """
        super().__init__("ELT", -1, N, *strings)


@cython.cclass
class EXP(SQLFunction):
    """Represents the `EXP(X)` function.

    MySQL description:
    - Returns the value of e (the base of natural logarithms) raised to the power of X.
    - The inverse of this function is LOG() (using a single argument only) or LN().
    """

    def __init__(self, X):
        """The `EXP(X)` function.

        MySQL description:
        - Returns the value of e (the base of natural logarithms) raised to the power of X.
        - The inverse of this function is LOG() (using a single argument only) or LN().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.EXP(2)
        # Escape output: "EXP(2)"
        # Expect result: 7.3890560989307

        sqlfunc.EXP("-2")
        # Escape output: "EXP('-2')"
        # Expect result: 0.13533528323661

        sqlfunc.EXP(0)
        # Escape output: "EXP(0)"
        # Expect result: 1
        ```
        """
        super().__init__("EXP", 1, X)


@cython.cclass
class EXPORT_SET(SQLFunction):
    """Represents the `EXPORT_SET(bits, on, off, [separator[, number_of_bits]])` function.

    MySQL description:
    - Returns a string such that for every bit set in the value bits, you get an on string
      and for every bit not set in the value, you get an off string.
    - Bits in bits are examined from right to left (from low-order to high-order bits).
      Strings are added to the result from left to right, separated by the separator
      string (the default being the comma character ,).
    - The number of bits examined is given by number_of_bits, which has a default of 64
      if not specified. number_of_bits is silently clipped to 64 if larger than 64. It
      is treated as an unsigned integer, so a value of '-1' is effectively the same as 64.
    """

    def __init__(
        self,
        bits,
        on,
        off,
        separator: str | Sentinel = IGNORED,
        number_of_bits: int | Sentinel = IGNORED,
    ):
        """The `EXPORT_SET(bits, on, off, [separator[, number_of_bits]])` function.

        MySQL description:
        - Returns a string such that for every bit set in the value bits, you get an on string
          and for every bit not set in the value, you get an off string.
        - Bits in bits are examined from right to left (from low-order to high-order bits).
          Strings are added to the result from left to right, separated by the separator
          string (the default being the comma character ,).
        - The number of bits examined is given by number_of_bits, which has a default of 64
          if not specified. number_of_bits is silently clipped to 64 if larger than 64. It
          is treated as an unsigned integer, so a value of '-1' is effectively the same as 64.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.EXPORT_SET(5,"Y","N",",",4)
        # Escape output: "EXPORT_SET(5,'Y','N',',',4)"
        # Expect result: 'Y,N,Y,N'

        sqlfunc.EXPORT_SET(6,"1","0",",",10)
        # Escape output: "EXPORT_SET(6,'1','0',',',10)"
        # Expect result: '0,1,1,0,0,0,0,0,0,0'

        sqlfunc.EXPORT_SET(6,"1","0","")
        # Escape output: "EXPORT_SET(6,'1','0','')"
        # Expect result: '0110000000000000000000000000000000000000000000000000000000000000'
        ```
        """
        if separator is IGNORED:
            super().__init__("EXPORT_SET", 3, bits, on, off)
        elif number_of_bits is IGNORED:
            super().__init__("EXPORT_SET", 4, bits, on, off, separator)
        else:
            super().__init__("EXPORT_SET", 5, bits, on, off, separator, number_of_bits)


@cython.cclass
class EXTRACT(SQLFunction):
    """Represents the `EXTRACT(unit FROM date)` function.

    MySQL description:
    - The EXTRACT() function uses the same kinds of unit specifiers as DATE_ADD()
      or DATE_SUB(), but extracts parts from the date rather than performing
      date arithmetic.
    - Returns NULL if date is NULL.
    """

    def __init__(self, unit, date):
        """The `EXTRACT(unit FROM date)` function.

        MySQL description:
        - The EXTRACT() function uses the same kinds of unit specifiers as DATE_ADD()
          or DATE_SUB(), but extracts parts from the date rather than performing
          date arithmetic.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc, sqlintvl

        sqlfunc.EXTRACT("YEAR", datetime.date(2007, 12, 31))
        # Escape output: "EXTRACT(YEAR FROM '2007-12-31')"
        # Expect result: 2007

        sqlfunc.EXTRACT(sqlintvl.MONTH, "2007-12-31")
        # Escape output: "EXTRACT(MONTH FROM '2007-12-31')"
        # Expect result: 12

        sqlfunc.EXTRACT(sqlintvl.YEAR_MONTH, datetime.date(2007, 12, 31))
        # Escape output: "EXTRACT(YEAR_MONTH FROM '2007-12-31')"
        # Expect result: 200712
        ```
        """
        unit = RawText(_validate_interval_unit(unit, "EXTRACT") + " FROM")
        super().__init__("EXTRACT", 2, unit, date, sep=" ")


# Functions: F -------------------------------------------------------------------------------------------------------
@cython.cclass
class FIELD(SQLFunction):
    """Represents the `FIELD(str, str1, str2, str3,...)` function.

    MySQL description:
    - Returns the index (position) of str in the str1, str2, str3, ... list.
      Returns 0 if str is not found.
    - If all arguments to FIELD() are strings, all arguments are compared as
      strings. If all arguments are numbers, they are compared as numbers.
      Otherwise, the arguments are compared as double.
    - If str is NULL, the return value is 0 because NULL fails equality
      comparison with any value. FIELD() is the complement of ELT().
    """

    def __init__(self, *strings):
        """The `FIELD(str, str1, str2, str3,...)` function.

        MySQL description:
        - Returns the index (position) of str in the str1, str2, str3, ... list.
          Returns 0 if str is not found.
        - If all arguments to FIELD() are strings, all arguments are compared as
          strings. If all arguments are numbers, they are compared as numbers.
          Otherwise, the arguments are compared as double.
        - If str is NULL, the return value is 0 because NULL fails equality
          comparison with any value. FIELD() is the complement of ELT().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.FIELD("Bb", "Aa", "Bb", "Cc", "Dd", "Ff")
        # Escape output: "FIELD('Bb','Aa','Bb','Cc','Dd','Ff')"
        # Expect result: 2

        sqlfunc.FIELD("Gg", "Aa", "Bb", "Cc", "Dd", "Ff")
        # Escape output: "FIELD('Gg','Aa','Bb','Cc','Dd','Ff')"
        # Expect result: 0
        ```
        """
        super().__init__("FIELD", -1, *strings)


@cython.cclass
class FIND_IN_SET(SQLFunction):
    """Represents the `FIND_IN_SET(string, string_list)` function.

    MySQL description:
    - Returns a value in the range of 1 to N if the string str is in
      the string list strlist consisting of N substrings.
    - A string list is a string composed of substrings separated by
      "," characters. If the first argument is a constant string and
      the second is a column of type SET, the FIND_IN_SET() function
      is optimized to use bit arithmetic.
    - Returns 0 if str is not in strlist or if strlist is the empty
      string. Returns NULL if either argument is NULL. This function
      does not work properly if the first argument contains a comma
      (,) character.
    """

    def __init__(self, string, string_list):
        """The `FIND_IN_SET(string, string_list)` function.

        MySQL description:
        - Returns a value in the range of 1 to N if the string str is in
          the string list strlist consisting of N substrings.
        - A string list is a string composed of substrings separated by
          "," characters. If the first argument is a constant string and
          the second is a column of type SET, the FIND_IN_SET() function
          is optimized to use bit arithmetic.
        - Returns 0 if str is not in strlist or if strlist is the empty
          string. Returns NULL if either argument is NULL. This function
          does not work properly if the first argument contains a comma
          (,) character.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.FIND_IN_SET("b", "a,b,c,d")
        # Escape output: "FIND_IN_SET('b','a,b,c,d')"
        # Expect result: 2

        sqlfunc.FIND_IN_SET("b", ("a", "b", "c", "d"))
        # Escape output: "FIND_IN_SET('b','a,b,c,d')"
        # Expect result: 2
        ```
        """
        if isinstance(string_list, str):
            super().__init__("FIND_IN_SET", 2, string, string_list)
        elif isinstance(
            string_list, (set, list, tuple, frozenset, np.ndarray, pd.Series)
        ):
            string_list = ",".join(map(str, string_list))
            super().__init__("FIND_IN_SET", 2, string, string_list)
        else:
            raise errors.SQLFunctionError(
                "SQL function FIND_IN_SET argument 'string_list' is invalid %s '%r'."
                % (type(string_list), string_list)
            )


@cython.cclass
class FLOOR(SQLFunction):
    """Represents the `FLOOR(X)` function.

    MySQL description:
    - Returns the largest integer value not greater than X.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `FLOOR(X)` function.

        MySQL description:
        - Returns the largest integer value not greater than X.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.FLOOR(1.23)
        # Escape output: "FLOOR(1.23)"
        # Expect result: 1

        sqlfunc.FLOOR("-1.23")
        # Escape output: "FLOOR('-1.23')"
        # Expect result: -2
        ```
        """
        super().__init__("FLOOR", 1, X)


@cython.cclass
class FORMAT(SQLFunction):
    """Represents the `FORMAT(X, D[, locale])` function.

    MySQL description:
    - Formats the number X to a format like '#,###,###.##', rounded to D
      decimal places, and returns the result as a string. If D is 0, the
      result has no decimal point or fractional part. If X or D is NULL,
      the function returns NULL.
    - The optional third parameter enables a locale to be specified to be
      used for the result number's decimal point, thousands separator, and
      grouping between separators. Permissible locale values are the same
      as the legal values for the lc_time_names system variable
      ([link](https://dev.mysql.com/doc/refman/8.4/en/locale-support.html)).
      If the locale is NULL or not specified, the default locale is 'en_US'.
    """

    def __init__(self, X, D, locale: str | Sentinel = IGNORED):
        """The `FORMAT(X, D[, locale])` function.

        MySQL description:
        - Formats the number X to a format like '#,###,###.##', rounded to D
          decimal places, and returns the result as a string. If D is 0, the
          result has no decimal point or fractional part. If X or D is NULL,
          the function returns NULL.
        - The optional third parameter enables a locale to be specified to be
          used for the result number's decimal point, thousands separator, and
          grouping between separators. Permissible locale values are the same
          as the legal values for the lc_time_names system variable
          ([link](https://dev.mysql.com/doc/refman/8.4/en/locale-support.html)).
          If the locale is NULL or not specified, the default locale is 'en_US'.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.FORMAT(12332.123456, 4)
        # Escape output: "FORMAT(12332.123456,4)"
        # Expect result: '12,332.1235'

        sqlfunc.FORMAT("12332.2", 0)
        # Escape output: "FORMAT('12332.2',0)"
        # Expect result: '12,332'

        sqlfunc.FORMAT(12332.2, 2, "de_DE")
        # Escape output: "FORMAT(12332.2,2,'de_DE')"
        # Expect result: '12.332,20'
        ```
        """
        if locale is IGNORED:
            super().__init__("FORMAT", 2, X, D)
        else:
            super().__init__("FORMAT", 3, X, D, locale)


@cython.cclass
class FORMAT_BYTES(SQLFunction):
    """Represents the `FORMAT_BYTES(count)` function.

    MySQL description:
    - Given a numeric byte count, converts it to human-readable format and returns
      a string consisting of a value and a units indicator. The string contains the
      number of bytes rounded to 2 decimal places and a minimum of 3 significant
      digits. Numbers less than 1024 bytes are represented as whole numbers and are
      not rounded.
    - Returns NULL if count is NULL.
    """

    def __init__(self, count):
        """The `FORMAT_BYTES(count)` function.

        MySQL description:
        - Given a numeric byte count, converts it to human-readable format and returns
          a string consisting of a value and a units indicator. The string contains the
          number of bytes rounded to 2 decimal places and a minimum of 3 significant
          digits. Numbers less than 1024 bytes are represented as whole numbers and are
          not rounded.
        - Returns NULL if count is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.FORMAT_BYTES(512)
        # Escape output: "FORMAT_BYTES(512)"
        # Expect result: '512 bytes'

        sqlfunc.FORMAT_BYTES(18446644073709551615)
        # Escape output: "FORMAT_BYTES(18446644073709551615)"
        # Expect result: '16.00 EiB'
        ```
        """
        super().__init__("FORMAT_BYTES", 1, count)


@cython.cclass
class FORMAT_PICO_TIME(SQLFunction):
    """Represents the `FORMAT_PICO_TIME(time_val)` function.

    MySQL description:
    - Given a numeric Performance Schema latency or wait time in picoseconds, converts
      it to human-readable format and returns a string consisting of a value and a units
      indicator. The string contains the decimal time rounded to 2 decimal places and a
      minimum of 3 significant digits. Times under 1 nanosecond are represented as whole
      numbers and are not rounded.
    - If time_val is NULL, this function returns NULL.
    """

    def __init__(self, time_val):
        """The `FORMAT_PICO_TIME(time_val)` function.

        MySQL description:
        - Given a numeric Performance Schema latency or wait time in picoseconds, converts
          it to human-readable format and returns a string consisting of a value and a units
          indicator. The string contains the decimal time rounded to 2 decimal places and a
          minimum of 3 significant digits. Times under 1 nanosecond are represented as whole
          numbers and are not rounded.
        - If time_val is NULL, this function returns NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.FORMAT_PICO_TIME(3501)
        # Escape output: "FORMAT_PICO_TIME(3501)"
        # Expect result: '3.50 ns'

        sqlfunc.FORMAT_PICO_TIME(188732396662000)
        # Escape output: "FORMAT_PICO_TIME(188732396662000)"
        # Expect result: '3.15 minutes'
        ```
        """
        super().__init__("FORMAT_PICO_TIME", 1, time_val)


@cython.cclass
class FROM_DAYS(SQLFunction):
    """Represents the `FROM_DAYS(N)` function.

    MySQL description:
    - Given a day number N, returns a DATE value.
    - Returns NULL if N is NULL.
    """

    def __init__(self, N):
        """The `FROM_DAYS(N)` function.

        MySQL description:
        - Given a day number N, returns a DATE value.
        - Returns NULL if N is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.FROM_DAYS(730669)
        # Escape output: "FROM_DAYS(730669)"
        # Expect result: '2000-07-03'

        sqlfunc.FROM_DAYS("736695")
        # Escape output: "FROM_DAYS('736695')"
        # Expect result: '2017-01-01'
        ```
        """
        super().__init__("FROM_DAYS", 1, N)


@cython.cclass
class FROM_UNIXTIME(SQLFunction):
    """Represents the `FROM_UNIXTIME(unix_timestamp[, format])` function.

    MySQL description:
    - Returns a representation of unix_timestamp as a datetime or character string
      value. The value returned is expressed using the session time zone. Unix_timestamp
      is an internal timestamp value representing seconds since '1970-01-01 00:00:00'
      UTC, such as produced by the UNIX_TIMESTAMP() function.
    - If format is omitted, this function returns a DATETIME value.
    - If unix_timestamp or format is NULL, this function returns NULL.
    - If unix_timestamp is an integer, the fractional seconds precision of the DATETIME
      is zero. When unix_timestamp is a decimal value, the fractional seconds precision
      of the DATETIME is the same as the precision of the decimal value, up to a maximum
      of 6. When unix_timestamp is a floating point number, the fractional seconds
      precision of the datetime is 6.
    """

    def __init__(self, ts, format: str | Sentinel = IGNORED):
        """The `FROM_UNIXTIME(unix_timestamp[, format])` function.

        MySQL description:
        - Returns a representation of unix_timestamp as a datetime or character string
          value. The value returned is expressed using the session time zone. Unix_timestamp
          is an internal timestamp value representing seconds since '1970-01-01 00:00:00'
          UTC, such as produced by the UNIX_TIMESTAMP() function.
        - If format is omitted, this function returns a DATETIME value.
        - If unix_timestamp or format is NULL, this function returns NULL.
        - If unix_timestamp is an integer, the fractional seconds precision of the DATETIME
          is zero. When unix_timestamp is a decimal value, the fractional seconds precision
          of the DATETIME is the same as the precision of the decimal value, up to a maximum
          of 6. When unix_timestamp is a floating point number, the fractional seconds
          precision of the datetime is 6.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.FROM_UNIXTIME(1447430881)
        # Escape output: "FROM_UNIXTIME(1447430881)"
        # Expect result: '2015-11-13 10:08:01'

        sqlfunc.FROM_UNIXTIME(1447430881, "%Y %D %M %h:%i:%s %x")
        # Escape output: "FROM_UNIXTIME(1447430881,'%Y %D %M %h:%i:%s %x')"
        # Expect result: '2015 13th November 10:08:01 2015'
        ```
        """
        if format is IGNORED:
            super().__init__("FROM_UNIXTIME", 1, ts)
        else:
            super().__init__("FROM_UNIXTIME", 2, ts, format)


# Functions: G -------------------------------------------------------------------------------------------------------
@cython.cclass
class GeomCollection(SQLFunction):
    """Represents the `GeomCollection(g [, g] ...)` function.

    MySQL description:
    - Constructs a GeomCollection value from the geometry arguments.
    - Returns all the proper geometries contained in the arguments even
      if a nonsupported geometry is present.
    - No arguments is permitted as a way to create an empty geometry.
    """

    def __init__(self, *g):
        """The `GeomCollection(g [, g] ...)` function.

        MySQL description:
        - Constructs a GeomCollection value from the geometry arguments.
        - Returns all the proper geometries contained in the arguments even
          if a nonsupported geometry is present.
        - No arguments is permitted as a way to create an empty geometry.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.GeomCollection(sqlfunc.Point(1, 1), sqlfunc.Point(2, 2))
        # Escape output: "GeomCollection(Point(1,1),Point(2,2))"
        # Expect result: 'GEOMETRYCOLLECTION(POINT(1 1), POINT(2 2))'
        """
        super().__init__("GeomCollection", -1, *g)


@cython.cclass
class GET_FORMAT(SQLFunction):
    """Represents the `GET_FORMAT({DATE|TIME|DATETIME}, {'EUR'|'USA'|'JIS'|'ISO'|'INTERNAL'})` function.

    MySQL description:
    - Returns a format string. This function is useful in combination with
      the DATE_FORMAT() and the STR_TO_DATE() functions.
    - If format is NULL, this function returns NULL.
    - The possible values for the first and second arguments result in
      several possible format strings (for the specifiers used, see the
      [table](https://dev.mysql.com/doc/refman/8.4/en/date-and-time-functions.html#function_get-format)
      ISO format refers to ISO 9075, not ISO 8601.
    """

    def __init__(self, date: str, format: str):
        """The `GET_FORMAT({DATE|TIME|DATETIME}, {'EUR'|'USA'|'JIS'|'ISO'|'INTERNAL'})` function.

        MySQL description:
        - Returns a format string. This function is useful in combination with
          the DATE_FORMAT() and the STR_TO_DATE() functions.
        - If format is NULL, this function returns NULL.
        - The possible values for the first and second arguments result in
          several possible format strings (for the specifiers used, see the
          [table](https://dev.mysql.com/doc/refman/8.4/en/date-and-time-functions.html#function_get-format)
          ISO format refers to ISO 9075, not ISO 8601.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.GET_FORMAT("DATE", "EUR")
        # Escape output: "GET_FORMAT(DATE,'EUR')"
        # Expect result: '%d.%m.%Y'

        sqlfunc.GET_FORMAT("DATETIME", "USA")
        # Escape output: "GET_FORMAT(DATETIME,'USA')"
        # Expect result: '%Y-%m-%d %H.%i.%s'
        ```
        """
        if date not in ("DATE", "DATETIME", "TIME"):
            date = date.upper()
            if date not in ("DATE", "DATETIME", "TIME"):
                raise errors.SQLFunctionError(
                    "SQL function GET_FORMAT 'date' expects 'DATE|TIME|DATETIME', "
                    "instead got '%r'." % date
                )
        super().__init__("GET_FORMAT", 2, RawText(date), format)


@cython.cclass
class GET_LOCK(SQLFunction):
    """Represents the `GET_LOCK(string, timeout)` function.

    MySQL description:
    - Tries to obtain a lock with a name given by the string str, using a timeout
      of timeout seconds. A negative timeout value means infinite timeout. The lock
      is exclusive. While held by one session, other sessions cannot obtain a lock
      of the same name.
    - Returns 1 if the lock was obtained successfully, 0 if the attempt timed out
      (for example, because another client has previously locked the name), or NULL
      if an error occurred (such as running out of memory or the thread was killed
      with mysqladmin kill).
    - A lock obtained with GET_LOCK() is released explicitly by executing RELEASE_LOCK()
      or implicitly when your session terminates (either normally or abnormally). Locks
      obtained with GET_LOCK() are not released when transactions commit or roll back.
    """

    def __init__(self, string, timeout):
        """The `GET_LOCK(string, timeout)` function.

        MySQL description:
        - Tries to obtain a lock with a name given by the string str, using a timeout
          of timeout seconds. A negative timeout value means infinite timeout. The lock
          is exclusive. While held by one session, other sessions cannot obtain a lock
          of the same name.
        - Returns 1 if the lock was obtained successfully, 0 if the attempt timed out
          (for example, because another client has previously locked the name), or NULL
          if an error occurred (such as running out of memory or the thread was killed
          with mysqladmin kill).
        - A lock obtained with GET_LOCK() is released explicitly by executing RELEASE_LOCK()
          or implicitly when your session terminates (either normally or abnormally). Locks
          obtained with GET_LOCK() are not released when transactions commit or roll back.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.GET_LOCK("lock1", 10)
        # Escape output: "GET_LOCK('lock1',10)"
        ```
        """
        super().__init__("GET_LOCK", 2, string, timeout)


@cython.cclass
class GREATEST(SQLFunction):
    """Represents the `GREATEST(value1, value2, ...)` function.

    MySQL description:
    - With two or more arguments, returns the largest (maximum-valued) argument.
      The arguments are compared using the same rules as for LEAST().
    """

    def __init__(self, *values):
        """The `GREATEST(value1, value2, ...)` function.

        MySQL description:
        - With two or more arguments, returns the largest (maximum-valued) argument.
          The arguments are compared using the same rules as for LEAST().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.GREATEST(2, 0)
        # Escape output: "GREATEST(2,0)"
        # Expect result: 2

        sqlfunc.GREATEST("B", "A", "C")
        # Escape output: "GREATEST('B','A','C')"
        # Expect result: 'C'
        ```
        """
        super().__init__("GREATEST", -1, *values)


# Functions: H -------------------------------------------------------------------------------------------------------
@cython.cclass
class HEX(SQLFunction):
    """Represents the `HEX(X)` function.

    MySQL description:
    - For a string argument str, HEX() returns a hexadecimal string
      representation of str where each byte of each character in str
      is converted to two hexadecimal digits. (Multibyte characters
      therefore become more than two digits.) The inverse of this
      operation is performed by the UNHEX() function.
    - For a numeric argument N, HEX() returns a hexadecimal string
      representation of the value of N treated as a longlong (BIGINT)
      number. This is equivalent to CONV(N,10,16). The inverse of
      this operation is performed by CONV(HEX(N),16,10).
    """

    def __init__(self, X):
        """The `HEX(X)` function.

        MySQL description:
        - For a string argument str, HEX() returns a hexadecimal string
          representation of str where each byte of each character in str
          is converted to two hexadecimal digits. (Multibyte characters
          therefore become more than two digits.) The inverse of this
          operation is performed by the UNHEX() function.
        - For a numeric argument N, HEX() returns a hexadecimal string
          representation of the value of N treated as a longlong (BIGINT)
          number. This is equivalent to CONV(N,10,16). The inverse of
          this operation is performed by CONV(HEX(N),16,10).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.HEX("abc")
        # Escape output: "HEX('abc')"
        # Expect result: '616263'

        sqlfunc.HEX(255)
        # Escape output: "HEX(255)"
        # Expect result: 'FF'
        ```
        """
        super().__init__("HEX", 1, X)


@cython.cclass
class HOUR(SQLFunction):
    """Represents the `HOUR(time)` function.

    MySQL description:
    - Returns the hour for time. The range of the return value is 0 to 23
      for time-of-day values. However, the range of TIME values actually
      is much larger, so HOUR can return values greater than 23.
    - Returns NULL if time is NULL.
    """

    def __init__(self, time):
        """The `HOUR(time)` function.

        MySQL description:
        - Returns the hour for time. The range of the return value is 0 to 23
          for time-of-day values. However, the range of TIME values actually
          is much larger, so HOUR can return values greater than 23.
        - Returns NULL if time is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.HOUR("10:05:03")
        # Escape output: "HOUR('10:05:03')"
        # Expect result: 10

        sqlfunc.HOUR(datetime.time(11, 59, 59))
        # Escape output: "HOUR('11:59:59')"
        # Expect result: 11
        ```
        """
        super().__init__("HOUR", 1, time)


# Functions: I -------------------------------------------------------------------------------------------------------
@cython.cclass
class IFNULL(SQLFunction):
    """Represents the `IFNULL(expr1, expr2)` function.

    MySQL description:
    - If expr1 is not NULL, IFNULL() returns expr1; otherwise it returns expr2.
    """

    def __init__(self, expr1, expr2):
        """The `IFNULL(expr1, expr2)` function.

        MySQL description:
        - If expr1 is not NULL, IFNULL() returns expr1; otherwise it returns expr2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.IFNULL(1, 0)
        # Escape output: "IFNULL(1,0)"
        # Expect result: 1

        sqlfunc.IFNULL(None, "a")
        # Escape output: "IFNULL(NULL,'a')"
        # Expect result: 'a'
        ```
        """
        super().__init__("IFNULL", 2, expr1, expr2)


@cython.cclass
class IN(SQLFunction):
    """Represents the `IN (value,...)` function.

    MySQL description:
    - Returns 1 (true) if the proceeding expr is equal to any
      of the values in the IN() list, else returns 0 (false).
    """

    def __init__(self, *values):
        """The `IN (value,...)` function.

        MySQL description:
        - Returns 1 (true) if proceeding expr is equal to any of the values
        in the IN() list, else returns 0 (false).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.IN(0, 3, 5, 7)
        # Escape output: "IN(0,3,5,7)"

        sqlfunc.IN(*["a", "b", "c"])
        # Escape output: "IN('a','b','c')"
        """
        super().__init__("IN", -1, *values)


@cython.cclass
class INET_ATON(SQLFunction):
    """Represents the `INET_ATON(expr)` function.

    MySQL description:
    - Given the dotted-quad representation of an IPv4 network address as a string,
      returns an integer that represents the numeric value of the address in network
      byte order (big endian).
    - Returns NULL if it does not understand its argument, or if expr is NULL.
    """

    def __init__(self, expr):
        """The `INET_ATON(expr)` function.

        MySQL description:
        - Given the dotted-quad representation of an IPv4 network address as a string,
          returns an integer that represents the numeric value of the address in network
          byte order (big endian).
        - Returns NULL if it does not understand its argument, or if expr is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.INET_ATON("10.0.5.9")
        # Escape output: "INET_ATON('10.0.5.9')"
        # Expect result: 167773449
        ```
        """
        super().__init__("INET_ATON", 1, expr)


@cython.cclass
class INET6_ATON(SQLFunction):
    """Represents the `INET6_ATON(expr)` function.

    MySQL description:
    - Given an IPv6 or IPv4 network address as a string, returns a binary string that
      represents the numeric value of the address in network byte order (big endian).
      Because numeric-format IPv6 addresses require more bytes than the largest integer
      type, the representation returned by this function has the VARBINARY data type:
      VARBINARY(16) for IPv6 addresses and VARBINARY(4) for IPv4 addresses.
    - If the argument is not a valid address, or if it is NULL, INET6_ATON() returns NULL.
    """

    def __init__(self, expr):
        """The `INET6_ATON(expr)` function.

        MySQL description:
        - Given an IPv6 or IPv4 network address as a string, returns a binary string that
          represents the numeric value of the address in network byte order (big endian).
          Because numeric-format IPv6 addresses require more bytes than the largest integer
          type, the representation returned by this function has the VARBINARY data type:
          VARBINARY(16) for IPv6 addresses and VARBINARY(4) for IPv4 addresses.
        - If the argument is not a valid address, or if it is NULL, INET6_ATON() returns NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.HEX(sqlfunc.INET6_ATON("fdfe::5a55:caff:fefa:9089"))
        # Escape output: "HEX(INET6_ATON('fdfe::5a55:caff:fefa:9089'))"
        # Expect result: 'FDFE0000000000005A55CAFFFEFA9089'

        sqlfunc.HEX(sqlfunc.INET6_ATON("10.0.5.9"))
        # Escape output: "HEX(INET6_ATON('10.0.5.9'))"
        # Expect result: '0A000509'
        ```
        """
        super().__init__("INET6_ATON", 1, expr)


@cython.cclass
class INET_NTOA(SQLFunction):
    """Represents the `INET_NTOA(expr)` function.

    MySQL description:
    - Given a numeric IPv4 network address in network byte order, returns the dotted-quad
      string representation of the address as a string in the connection character set.
    - Returns NULL if it does not understand its argument.
    """

    def __init__(self, expr):
        """The `INET_NTOA(expr)` function.

        MySQL description:
        - Given a numeric IPv4 network address in network byte order, returns the dotted-quad
          string representation of the address as a string in the connection character set.
        - Returns NULL if it does not understand its argument.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.INET_NTOA(167773449)
        # Escape output: "INET_NTOA(167773449)"
        # Expect result: '10.0.5.9'
        ```
        """
        super().__init__("INET_NTOA", 1, expr)


@cython.cclass
class INET6_NTOA(SQLFunction):
    """Represents the `INET6_NTOA(expr)` function.

    MySQL description:
    - Given an IPv6 or IPv4 network address represented in numeric form as a binary string,
      returns the string representation of the address as a string in the connection character set.
    - If the argument is not a valid address, or if it is NULL, INET6_NTOA() returns NULL.
    """

    def __init__(self, expr):
        """The `INET6_NTOA(expr)` function.

        MySQL description:
        - Given an IPv6 or IPv4 network address represented in numeric form as a binary string,
          returns the string representation of the address as a string in the connection character set.
        - If the argument is not a valid address, or if it is NULL, INET6_NTOA() returns NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.INET6_NTOA(sqlfunc.INET6_ATON("fdfe::5a55:caff:fefa:9089"))
        # Escape output: "INET6_NTOA(INET6_ATON('fdfe::5a55:caff:fefa:9089'))"
        # Expect result: 'fdfe::5a55:caff:fefa:9089'

        sqlfunc.INET6_NTOA(sqlfunc.INET6_ATON("10.0.5.9"))
        # Escape output: "INET6_NTOA(INET6_ATON('10.0.5.9'))"
        # Expect result: '10.0.5.9'
        ```
        """
        super().__init__("INET6_NTOA", 1, expr)


@cython.cclass
class INSERT(SQLFunction):
    """Represents the `INSERT(string, pos, length, newstr)` function.

    MySQL description:
    - Returns the string str, with the substring beginning at position pos
      and len characters long replaced by the string newstr. Returns the
      original string if pos is not within the length of the string. Replaces
      the rest of the string from position pos if len is not within the length
      of the rest of the string.
    - Returns NULL if any argument is NULL.
    """

    def __init__(self, string, pos, length, newstr):
        """The `INSERT(string, pos, length, newstr)` function.

        MySQL description:
        - Returns the string str, with the substring beginning at position pos
          and len characters long replaced by the string newstr. Returns the
          original string if pos is not within the length of the string. Replaces
          the rest of the string from position pos if len is not within the length
          of the rest of the string.
        - Returns NULL if any argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.INSERT("Quadratic", 3, 4, "What")
        # Escape output: "INSERT('Quadratic',3,4,'What')"
        # Expect result: 'QuWhattic'

        sqlfunc.INSERT("Quadratic", -1, 4, "What")
        # Escape output: "INSERT('Quadratic',-1,4,'What')"
        # Expect result: 'Quadratic'

        sqlfunc.INSERT("Quadratic", 3, 100, "What")
        # Escape output: "INSERT('Quadratic',3,100,'What')"
        # Expect result: 'QuWhat'
        ```
        """
        super().__init__("INSERT", 4, string, pos, length, newstr)


@cython.cclass
class INSTR(SQLFunction):
    """Represents the `INSTR(string, substr)` function.

    MySQL description:
    - Returns the position of the first occurrence of substring substr in
      string str. This is the same as the two-argument form of LOCATE(),
      except that the order of the arguments is reversed.
    - This function is multibyte safe, and is case-sensitive only if at
      least one argument is a binary string. If either argument is NULL,
      this functions returns NULL.
    """

    def __init__(self, string, substr):
        """The `INSTR(string, substr)` function.

        MySQL description:
        - Returns the position of the first occurrence of substring substr in
          string str. This is the same as the two-argument form of LOCATE(),
          except that the order of the arguments is reversed.
        - This function is multibyte safe, and is case-sensitive only if at
          least one argument is a binary string. If either argument is NULL,
          this functions returns NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.INSTR("foobarbar", "bar")
        # Escape output: "INSTR('foobarbar','bar')"
        # Expect result: 4

        sqlfunc.INSTR("xbar", "foobar")
        # Escape output: "INSTR('xbar','foobar')"
        # Expect result: 0
        """
        super().__init__("INSTR", 2, string, substr)


@cython.cclass
class INTERVAL(SQLFunction):
    """Represents the `INTERVAL(N,N1,N2,N3,...)` function.

    MySQL description:
    - Returns 0 if N ≤ N1, 1 if N ≤ N2 and so on, or -1 if N is NULL. All arguments
      are treated as integers. It is required that N1 ≤ N2 ≤ N3 ≤ ... ≤ Nn for this
      function to work correctly. This is because a binary search is used (very fast).
    """

    def __init__(self, N, N1, *Nn):
        """The `INTERVAL(N,N1,N2,N3,...)` function.

        MySQL description:
        - Returns 0 if N ≤ N1, 1 if N ≤ N2 and so on, or -1 if N is NULL. All arguments
          are treated as integers. It is required that N1 ≤ N2 ≤ N3 ≤ ... ≤ Nn for this
          function to work correctly. This is because a binary search is used (very fast).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.INTERVAL(23, 1, 15, 17, 30, 44, 200)
        # Escape output: "INTERVAL(23,1,15,17,30,44,200)"
        # Expect result: 3

        sqlfunc.INTERVAL(22, 23, 30, 44, 200)
        # Escape output: "INTERVAL(22,23,30,44,200)"
        # Expect result: 0
        ```
        """
        super().__init__("INTERVAL", -1, N, N1, *Nn)


@cython.cclass
class IS_FREE_LOCK(SQLFunction):
    """Represents the `IS_FREE_LOCK(string)` function.

    MySQL description:
    - Checks whether the lock named str is free to use (that is, not locked).
      Returns 1 if the lock is free (no one is using the lock), 0 if the lock
      is in use, and NULL if an error occurs (such as an incorrect argument).
    - This function is unsafe for statement-based replication. A warning is
      logged if you use this function when binlog_format is set to STATEMENT.
    """

    def __init__(self, string):
        """The `IS_FREE_LOCK(string)` function.

        MySQL description:
        - Checks whether the lock named str is free to use (that is, not locked).
          Returns 1 if the lock is free (no one is using the lock), 0 if the lock
          is in use, and NULL if an error occurs (such as an incorrect argument).
        - This function is unsafe for statement-based replication. A warning is
          logged if you use this function when binlog_format is set to STATEMENT.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.IS_FREE_LOCK("lock1")
        # Escape output: "IS_FREE_LOCK('lock1')"
        ```
        """
        super().__init__("IS_FREE_LOCK", 1, string)


@cython.cclass
class IS_USER_LOCK(SQLFunction):
    """Represents the `IS_USER_LOCK(string)` function.

    MySQL description:
    - Checks whether the lock named str is in use (that is, locked). If so, it
      returns the connection identifier of the client session that holds the lock.
      Otherwise, it returns NULL.
    - This function is unsafe for statement-based replication. A warning is logged
      if you use this function when binlog_format is set to STATEMENT.
    """

    def __init__(self, string):
        """The `IS_USER_LOCK(string)` function.

        MySQL description:
        - Checks whether the lock named str is in use (that is, locked). If so, it
          returns the connection identifier of the client session that holds the lock.
          Otherwise, it returns NULL.
        - This function is unsafe for statement-based replication. A warning is logged
          if you use this function when binlog_format is set to STATEMENT.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.IS_USER_LOCK("user1")
        # Escape output: "IS_USER_LOCK('user1')"
        ```
        """
        super().__init__("IS_USER_LOCK", 1, string)


@cython.cclass
class IS_UUID(SQLFunction):
    """Represents the `IS_UUID(string_uuid)` function.

    MySQL description:
    - Returns 1 if the argument is a valid string-format UUID
    - Returns 0 if the argument is not a valid UUID, and NULL if the argument is NULL.
    """

    def __init__(self, string_uuid):
        """The `IS_UUID(string_uuid)` function.

        MySQL description:
        - Returns 1 if the argument is a valid string-format UUID
        - Returns 0 if the argument is not a valid UUID, and NULL if the argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.IS_UUID("6ccd780c-baba-1026-9564-5b8c656024db")
        # Escape output: "IS_UUID('6ccd780c-baba-1026-9564-5b8c656024db')"
        # Expect result: 1

        sqlfunc.IS_UUID("6ccd780c-baba-1026-9564-5b8c6560")
        # Escape output: "IS_UUID('6ccd780c-baba-1026-9564-5b8c6560')"
        # Expect result: 0
        ```
        """
        super().__init__("IS_UUID", 1, string_uuid)


@cython.cclass
class ISNULL(SQLFunction):
    """Represents the `ISNULL(expr)` function.

    MySQL description:
    - If expr is NULL, ISNULL() returns 1, otherwise it returns 0.
    """

    def __init__(self, expr):
        """The `ISNULL(expr)` function.

        MySQL description:
        - If expr is NULL, ISNULL() returns 1, otherwise it returns 0.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ISNULL(None)
        # Escape output: "ISNULL(NULL)"
        # Expect result: 1

        sqlfunc.ISNULL(0)
        # Escape output: "ISNULL(0)"
        # Expect result: 0
        ```
        """
        super().__init__("ISNULL", 1, expr)


# Functions: J -------------------------------------------------------------------------------------------------------
@cython.cclass
class JSON_ARRAY(SQLFunction):
    """Represents the `JSON_ARRAY([val[, val] ...])` function.

    MySQL description:
    - Evaluates a (possibly empty) list of values and returns a JSON array containing those values.
    """

    def __init__(self, *values):
        """The `JSON_ARRAY([val[, val] ...])` function.

        MySQL description:
        - Evaluates a (possibly empty) list of values and returns a JSON array containing those values.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_ARRAY(1, "abc", None, sqlfunc.CURRENT_TIME())
        # Escape output: "JSON_ARRAY(1,'abc',NULL,CURRENT_TIME())"
        # Expect result: '[1, "abc", null, "17:26:10.000000"]'
        ```
        """
        super().__init__("JSON_ARRAY", -1, *values)


@cython.cclass
class JSON_ARRAY_APPEND(SQLFunction):
    """Represents the `JSON_ARRAY_APPEND(json_doc, path, val[, path, val] ...)` function.

    MySQL description:
    - Appends values to the end of the indicated arrays within a JSON document and returns
      the result. Returns NULL if any argument is NULL. An error occurs if the json_doc
      argument is not a valid JSON document or any path argument is not a valid path
      expression or contains a * or ** wildcard.
    - The path-value pairs are evaluated left to right. The document produced by evaluating
      one pair becomes the new value against which the next pair is evaluated.
    - If a path selects a scalar or object value, that value is autowrapped within an array
      and the new value is added to that array. Pairs for which the path does not identify
      any value in the JSON document are ignored.
    """

    def __init__(self, json_doc, *path_val_pairs):
        """The `JSON_ARRAY_APPEND(json_doc, path, val[, path, val] ...)` function.

        MySQL description:
        - Appends values to the end of the indicated arrays within a JSON document and returns
          the result. Returns NULL if any argument is NULL. An error occurs if the json_doc
          argument is not a valid JSON document or any path argument is not a valid path
          expression or contains a * or ** wildcard.
        - The path-value pairs are evaluated left to right. The document produced by evaluating
          one pair becomes the new value against which the next pair is evaluated.
        - If a path selects a scalar or object value, that value is autowrapped within an array
          and the new value is added to that array. Pairs for which the path does not identify
          any value in the JSON document are ignored.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '["a", ["b", "c"], "d"]'

        sqlfunc.JSON_ARRAY_APPEND(js, "$[1]", 1)
        # Escape output: "JSON_ARRAY_APPEND('["a", ["b", "c"], "d"]','$[1]',1)"
        # Expect result: '["a", ["b", "c", 1], "d"]'

        sqlfunc.JSON_ARRAY_APPEND(js, "$[1]", 2, "$[1]", 3)
        # Escape output: "JSON_ARRAY_APPEND('["a", ["b", "c"], "d"]','$[1]',2,'$[1]',3)"
        # Expect result: '["a", ["b", "c", 2, 3], "d"]'
        ```
        """
        _validate_args_paris(path_val_pairs, "JSON_ARRAY_APPEND", "path_val_pairs")
        super().__init__("JSON_ARRAY_APPEND", -1, json_doc, *path_val_pairs)


@cython.cclass
class JSON_ARRAY_INSERT(SQLFunction):
    """Represents the `JSON_ARRAY_INSERT(json_doc, path, val[, path, val] ...)` function.

    MySQL description:
    - Updates a JSON document, inserting into an array within the document and returning
      the modified document. Returns NULL if any argument is NULL. An error occurs if the
      json_doc argument is not a valid JSON document or any path argument is not a valid
      path expression or contains a * or ** wildcard or does not end with an array element
      identifier.
    - The path-value pairs are evaluated left to right. The document produced by evaluating
      one pair becomes the new value against which the next pair is evaluated.
    - Pairs for which the path does not identify any array in the JSON document are ignored.
      If a path identifies an array element, the corresponding value is inserted at that
      element position, shifting any following values to the right. If a path identifies an
      array position past the end of an array, the value is inserted at the end of the array.
    """

    def __init__(self, json_doc, *path_val_pairs):
        """The `JSON_ARRAY_INSERT(json_doc, path, val[, path, val] ...)` function.

        MySQL description:
        - Updates a JSON document, inserting into an array within the document and returning
          the modified document. Returns NULL if any argument is NULL. An error occurs if the
          json_doc argument is not a valid JSON document or any path argument is not a valid
          path expression or contains a * or ** wildcard or does not end with an array element
          identifier.
        - The path-value pairs are evaluated left to right. The document produced by evaluating
          one pair becomes the new value against which the next pair is evaluated.
        - Pairs for which the path does not identify any array in the JSON document are ignored.
          If a path identifies an array element, the corresponding value is inserted at that
          element position, shifting any following values to the right. If a path identifies an
          array position past the end of an array, the value is inserted at the end of the array.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '["a", {"b": [1, 2]}, [3, 4]]'

        sqlfunc.JSON_ARRAY_INSERT(js, "$[1]", "x")
        # Escape output: "JSON_ARRAY_INSERT('["a", {"b": [1, 2]}, [3, 4]]','$[1]','x')"
        # Expect result: '["a", "x", {"b": [1, 2]}, [3, 4]]'

        sqlfunc.JSON_ARRAY_INSERT(js, "$[0]", "x", "$[2][1]", "y")
        # Escape output: "JSON_ARRAY_INSERT('["a", {"b": [1, 2]}, [3, 4]]','$[0]','x','$[2][1]','y')"
        # Expect result: '["x", "a", {"b": [1, 2]}, [3, 4]]'
        ```
        """
        _validate_args_paris(path_val_pairs, "JSON_ARRAY_INSERT", "path_val_pairs")
        super().__init__("JSON_ARRAY_INSERT", -1, json_doc, *path_val_pairs)


@cython.cclass
class JSON_CONTAINS(SQLFunction):
    """Represents the `JSON_CONTAINS(target, candidate[, path])` function.

    MySQL description:
    - Indicates by returning 1 or 0 whether a given candidate JSON document
      is contained within a target JSON document, or—if a path argument was
      supplied—whether the candidate is found at a specific path within the
      target.
    - Returns NULL if any argument is NULL, or if the path argument does not
      identify a section of the target document. An error occurs if target or
      candidate is not a valid JSON document, or if the path argument is not
      a valid path expression or contains a * or ** wildcard.
    """

    def __init__(self, target, candidate, path: str | Sentinel = IGNORED):
        """The `JSON_CONTAINS(target, candidate[, path])` function.

        MySQL description:
        - Indicates by returning 1 or 0 whether a given candidate JSON document
          is contained within a target JSON document, or—if a path argument was
          supplied—whether the candidate is found at a specific path within the
          target.
        - Returns NULL if any argument is NULL, or if the path argument does not
          identify a section of the target document. An error occurs if target or
          candidate is not a valid JSON document, or if the path argument is not
          a valid path expression or contains a * or ** wildcard.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js1, js2 = '{"a": 1, "b": 2, "c": {"d": 4}}', "1"

        sqlfunc.JSON_CONTAINS(js1, js2)
        # Escape output: "JSON_CONTAINS('{"a": 1, "b": 2, "c": {"d": 4}}','1')"
        # Expect result: 0

        sqlfunc.JSON_CONTAINS(js1, js2, "$.a")
        # Escape output: "JSON_CONTAINS('{"a": 1, "b": 2, "c": {"d": 4}}','1','$.a')"
        # Expect result: 1
        ```
        """
        if path is IGNORED:
            super().__init__("JSON_CONTAINS", 2, target, candidate)
        else:
            super().__init__("JSON_CONTAINS", 3, target, candidate, path)


@cython.cclass
class JSON_CONTAINS_PATH(SQLFunction):
    """Represents the `JSON_CONTAINS_PATH(json_doc, one_or_all, path[, path] ...)` function.

    MySQL description:
    - Returns 0 or 1 to indicate whether a JSON document contains data at a given path
      or paths. Returns NULL if any argument is NULL. An error occurs if the json_doc
      argument is not a valid JSON document, any path argument is not a valid path
      expression, or one_or_all is not 'one' or 'all'.
    - To check for a specific value at a path, use JSON_CONTAINS() instead.
    - The return value is 0 if no specified path exists within the document. Otherwise,
      the return value depends on the one_or_all argument: 'one' 1 if at least one path
      exists within the document, 0 otherwise; 'all' 1 if all paths exist within the
      document, 0 otherwise.
    """

    def __init__(self, json_doc, one_or_all, *paths):
        """The `JSON_CONTAINS_PATH(json_doc, one_or_all, path[, path] ...)` function.

        MySQL description:
        - Returns 0 or 1 to indicate whether a JSON document contains data at a given path
          or paths. Returns NULL if any argument is NULL. An error occurs if the json_doc
          argument is not a valid JSON document, any path argument is not a valid path
          expression, or one_or_all is not 'one' or 'all'.
        - To check for a specific value at a path, use JSON_CONTAINS() instead.
        - The return value is 0 if no specified path exists within the document. Otherwise,
          the return value depends on the one_or_all argument: 'one' 1 if at least one path
          exists within the document, 0 otherwise; 'all' 1 if all paths exist within the
          document, 0 otherwise.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '{"a": 1, "b": 2, "c": {"d": 4}}'

        sqlfunc.JSON_CONTAINS_PATH(js, "one", "$.a", "$.e")
        # Escape output: "JSON_CONTAINS_PATH('{"a": 1, "b": 2, "c": {"d": 4}}','one','$.a','$.e')"
        # Expect result: 1

        sqlfunc.JSON_CONTAINS_PATH(js, "all", "$.a", "$.e")
        # Escape output: "JSON_CONTAINS_PATH('{"a": 1, "b": 2, "c": {"d": 4}}','all','$.a','$.e')"
        # Expect result: 0
        ```
        """
        super().__init__("JSON_CONTAINS_PATH", -1, json_doc, one_or_all, *paths)


@cython.cclass
class JSON_DEPTH(SQLFunction):
    """Represents the `JSON_DEPTH(json_doc)` function.

    MySQL description:
    - Returns the maximum depth of a JSON document. Returns NULL if the argument
      is NULL. An error occurs if the argument is not a valid JSON document.
    - An empty array, empty object, or scalar value has depth 1. A nonempty array
      containing only elements of depth 1 or nonempty object containing only member
      values of depth 1 has depth 2. Otherwise, a JSON document has depth greater
      than 2.
    """

    def __init__(self, json_doc):
        """The `JSON_DEPTH(json_doc)` function.

        MySQL description:
        - Returns the maximum depth of a JSON document. Returns NULL if the argument
          is NULL. An error occurs if the argument is not a valid JSON document.
        - An empty array, empty object, or scalar value has depth 1. A nonempty array
          containing only elements of depth 1 or nonempty object containing only member
          values of depth 1 has depth 2. Otherwise, a JSON document has depth greater
          than 2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_DEPTH("{}")
        # Escape output: "JSON_DEPTH('{}')"
        # Expect result: 1

        sqlfunc.JSON_DEPTH("[10, 20]")
        # Escape output: "JSON_DEPTH('[10, 20]')"
        # Expect result: 2

        sqlfunc.JSON_DEPTH('[10, {"a": 20}]')
        # Escape output: "JSON_DEPTH('[10, {"a": 20}]')"
        # Expect result: 3
        ```
        """
        super().__init__("JSON_DEPTH", 1, json_doc)


@cython.cclass
class JSON_EXTRACT(SQLFunction):
    """Represents the `JSON_EXTRACT(json_doc, path[, path] ...)` function.

    MySQL description:
    - Returns data from a JSON document, selected from the parts of the document
      matched by the path arguments. Returns NULL if any argument is NULL or no
      paths locate a value in the document. An error occurs if the json_doc argument
      is not a valid JSON document or any path argument is not a valid path expression.
    - The return value consists of all values matched by the path arguments. If it is
      possible that those arguments could return multiple values, the matched values
      are autowrapped as an array, in the order corresponding to the paths that produced
      them. Otherwise, the return value is the single matched value.
    """

    def __init__(self, json_doc, *paths):
        """The `JSON_EXTRACT(json_doc, path[, path] ...)` function.

        MySQL description:
        - Returns data from a JSON document, selected from the parts of the document
          matched by the path arguments. Returns NULL if any argument is NULL or no
          paths locate a value in the document. An error occurs if the json_doc argument
          is not a valid JSON document or any path argument is not a valid path expression.
        - The return value consists of all values matched by the path arguments. If it is
          possible that those arguments could return multiple values, the matched values
          are autowrapped as an array, in the order corresponding to the paths that produced
          them. Otherwise, the return value is the single matched value.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_EXTRACT("[10, 20, [30, 40]]", "$[1]")
        # Escape output: "JSON_EXTRACT('[10, 20, [30, 40]]','$[1]')"
        # Expect result: 20

        sqlfunc.JSON_EXTRACT("[10, 20, [30, 40]]", "$[1]", "$[0]")
        # Escape output: "JSON_EXTRACT('[10, 20, [30, 40]]','$[1]','$[0]')"
        # Expect result: [20, 10]
        ```
        """
        super().__init__("JSON_EXTRACT", -1, json_doc, *paths)


@cython.cclass
class JSON_INSERT(SQLFunction):
    """Represents the `JSON_INSERT(json_doc, path, val[, path, val] ...)` function.

    MySQL description:
    - Inserts data into a JSON document and returns the result. Returns NULL
      if any argument is NULL. An error occurs if the json_doc argument is
      not a valid JSON document or any path argument is not a valid path
      expression or contains a * or ** wildcard.
    - The path-value pairs are evaluated left to right. The document produced
      by evaluating one pair becomes the new value against which the next pair
      is evaluated.
    - A path-value pair for an existing path in the document is ignored and does
      not overwrite the existing document value. A path-value pair for a nonexisting
      path in the document adds the value to the document if the path identifies one
      of these types of values:
    - A member not present in an existing object. The member is added to the object
      and associated with the new value.
    - A position past the end of an existing array. The array is extended with the
      new value. If the existing value is not an array, it is autowrapped as an
      array, then extended with the new value.
    - Otherwise, a path-value pair for a nonexisting path in the document is ignored
      and has no effect.
    """

    def __init__(self, json_doc, *path_val_pairs):
        """The `JSON_INSERT(json_doc, path, val[, path, val] ...)` function.

        MySQL description:
        - Inserts data into a JSON document and returns the result. Returns NULL
          if any argument is NULL. An error occurs if the json_doc argument is
          not a valid JSON document or any path argument is not a valid path
          expression or contains a * or ** wildcard.
        - The path-value pairs are evaluated left to right. The document produced
          by evaluating one pair becomes the new value against which the next pair
          is evaluated.
        - A path-value pair for an existing path in the document is ignored and does
          not overwrite the existing document value. A path-value pair for a nonexisting
          path in the document adds the value to the document if the path identifies one
          of these types of values:
        - A member not present in an existing object. The member is added to the object
          and associated with the new value.
        - A position past the end of an existing array. The array is extended with the
          new value. If the existing value is not an array, it is autowrapped as an
          array, then extended with the new value.
        - Otherwise, a path-value pair for a nonexisting path in the document is ignored
          and has no effect.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '{ "a": 1, "b": [2, 3]}'

        sqlfunc.JSON_INSERT(js, "$.a", 10, "$.c", "[true, false]")
        # Escape output: "JSON_INSERT('{ "a": 1, "b": [2, 3]}','$.a',10,'$.c','[true, false]')"
        # Expect result: '{"a": 1, "b": [2, 3], "c": "[true, false]"}'
        """
        _validate_args_paris(path_val_pairs, "JSON_INSERT", "path_val_pairs")
        super().__init__("JSON_INSERT", -1, json_doc, *path_val_pairs)


@cython.cclass
class JSON_KEYS(SQLFunction):
    """Represents the `JSON_KEYS(json_doc[, path])` function.

    MySQL description:
    - Returns the keys from the top-level value of a JSON object as a JSON array, or,
      if a path argument is given, the top-level keys from the selected path. Returns
      NULL if any argument is NULL, the json_doc argument is not an object, or path,
      if given, does not locate an object. An error occurs if the json_doc argument is
      not a valid JSON document or the path argument is not a valid path expression or
      contains a * or ** wildcard.
    - The result array is empty if the selected object is empty. If the top-level value
      has nested subobjects, the return value does not include keys from those subobjects.
    """

    def __init__(self, json_doc, path: str | Sentinel = IGNORED):
        """The `JSON_KEYS(json_doc[, path])` function.

        MySQL description:
        - Returns the keys from the top-level value of a JSON object as a JSON array, or,
          if a path argument is given, the top-level keys from the selected path. Returns
          NULL if any argument is NULL, the json_doc argument is not an object, or path,
          if given, does not locate an object. An error occurs if the json_doc argument is
          not a valid JSON document or the path argument is not a valid path expression or
          contains a * or ** wildcard.
        - The result array is empty if the selected object is empty. If the top-level value
          has nested subobjects, the return value does not include keys from those subobjects.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_KEYS('{"a": 1, "b": {"c": 30}}')
        # Escape output: "JSON_KEYS('{"a": 1, "b": {"c": 30}}')"
        # Expect result: '["a", "b"]'

        sqlfunc.JSON_KEYS('{"a": 1, "b": {"c": 30}}', '$.b')
        # Escape output: "JSON_KEYS('{"a": 1, "b": {"c": 30}}','$.b')"
        # Expect result: '["c"]'
        ```
        """
        if path is IGNORED:
            super().__init__("JSON_KEYS", 1, json_doc)
        else:
            super().__init__("JSON_KEYS", 2, json_doc, path)


@cython.cclass
class JSON_LENGTH(SQLFunction):
    """Represents the `JSON_LENGTH(json_doc[, path])` function.

    MySQL description:
    - Returns the length of a JSON document, or, if a path argument is given, the
      length of the value within the document identified by the path. Returns NULL
      if any argument is NULL or the path argument does not identify a value in the
      document. An error occurs if the json_doc argument is not a valid JSON document
      or the path argument is not a valid path expression.
    - The length of a document is determined as follows: the length of a scalar is 1;
      the length of an array is the number of array elements; the length of an object
      is the number of object members; the length does not count the length of nested
      arrays or objects.
    """

    def __init__(self, json_doc, path: str | Sentinel = IGNORED):
        """The `JSON_LENGTH(json_doc[, path])` function.

        MySQL description:
        - Returns the length of a JSON document, or, if a path argument is given, the
          length of the value within the document identified by the path. Returns NULL
          if any argument is NULL or the path argument does not identify a value in the
          document. An error occurs if the json_doc argument is not a valid JSON document
          or the path argument is not a valid path expression.
        - The length of a document is determined as follows: the length of a scalar is 1;
          the length of an array is the number of array elements; the length of an object
          is the number of object members; the length does not count the length of nested
          arrays or objects.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_LENGTH('[1, 2, {"a": 3}]')
        # Escape output: "JSON_LENGTH('[1, 2, {"a": 3}]')"
        # Expect result: 3

        sqlfunc.JSON_LENGTH('{"a": 1, "b": {"c": 30}}', '$.b')
        # Escape output: "JSON_LENGTH('{"a": 1, "b": {"c": 30}}','$.b')"
        # Expect result: 1
        ```
        """
        if path is IGNORED:
            super().__init__("JSON_LENGTH", 1, json_doc)
        else:
            super().__init__("JSON_LENGTH", 2, json_doc, path)


@cython.cclass
class JSON_MERGE_PATCH(SQLFunction):
    """Represents the `JSON_MERGE_PATCH(json_doc, json_doc[, json_doc] ...)` function.

    MySQL description:
    - Performs an RFC 7396 compliant merge of two or more JSON documents and returns
      the merged result, without preserving members having duplicate keys. Raises an
      error if at least one of the documents passed as arguments to this function is
      not valid.
    - If the first argument is not an object, the result of the merge is the same as
      if an empty object had been merged with the second argument.
    - If the second argument is not an object, the result of the merge is the second argument.
    - If both arguments are objects, the result of the merge is an object with the
      following members: all members of the first object which do not have a corresponding
      member with the same key in the second object; All members of the second object which
      do not have a corresponding key in the first object, and whose value is not the JSON
      null literal; all members with a key that exists in both the first and the second
      object, and whose value in the second object is not the JSON null literal. The values
      of these members are the results of recursively merging the value in the first object
      with the value in the second object.
    """

    def __init__(self, *json_docs):
        """The `JSON_MERGE_PATCH(json_doc, json_doc[, json_doc] ...)` function.

        MySQL description:
        - Performs an RFC 7396 compliant merge of two or more JSON documents and returns
          the merged result, without preserving members having duplicate keys. Raises an
          error if at least one of the documents passed as arguments to this function is
          not valid.
        - If the first argument is not an object, the result of the merge is the same as
          if an empty object had been merged with the second argument.
        - If the second argument is not an object, the result of the merge is the second argument.
        - If both arguments are objects, the result of the merge is an object with the
          following members: all members of the first object which do not have a corresponding
          member with the same key in the second object; All members of the second object which
          do not have a corresponding key in the first object, and whose value is not the JSON
          null literal; all members with a key that exists in both the first and the second
          object, and whose value in the second object is not the JSON null literal. The values
          of these members are the results of recursively merging the value in the first object
          with the value in the second object.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_MERGE_PATCH("[1, 2]", "[true, false]")
        # Escape output: "JSON_MERGE_PATCH('[1, 2]','[true, false]')"
        # Expect result: '[true, false]'

        sqlfunc.JSON_MERGE_PATCH('{"name": "x"}', '{"id": 47}')
        # Escape output: "JSON_MERGE_PATCH('{"name": "x"}','{"id": 47}')"
        # Expect result: '{"id": 47, "name": "x"}'

        sqlfunc.JSON_MERGE_PATCH('{"a": 1, "b":2}','{"a": 3, "c":4}','{"a": 5, "d":6}')
        # Escape output: "JSON_MERGE_PATCH('{"a": 1, "b":2}','{"a": 3, "c":4}','{"a": 5, "d":6}')"
        # Expect result: '{"a": 5, "b": 2, "c": 4, "d": 6}'
        ```
        """
        if tuple_len(json_docs) < 2:
            raise errors.SQLFunctionError(
                "SQL function 'JSON_MERGE_PATCH' requires at least two JSON documents, "
                "instead got '%s'." % tuple_len(json_docs)
            )
        super().__init__("JSON_MERGE_PATCH", -1, *json_docs)


@cython.cclass
class JSON_MERGE_PRESERVE(SQLFunction):
    """Represents the `JSON_MERGE_PRESERVE(json_doc, json_doc[, json_doc] ...)` function.

    MySQL description:
    - Merges two or more JSON documents and returns the merged result.
      Returns NULL if any argument is NULL. An error occurs if any argument
      is not a valid JSON document.
    - Merging takes place according to the following rules: adjacent arrays
      are merged to a single array; adjacent objects are merged to a single
      object; a scalar value is autowrapped as an array and merged as an array;
      an adjacent array and object are merged by autowrapping the object as an
      array and merging the two arrays.
    """

    def __init__(self, *json_docs):
        """The `JSON_MERGE_PRESERVE(json_doc, json_doc[, json_doc] ...)` function.

        MySQL description:
        - Merges two or more JSON documents and returns the merged result.
          Returns NULL if any argument is NULL. An error occurs if any argument
          is not a valid JSON document.
        - Merging takes place according to the following rules: adjacent arrays
          are merged to a single array; adjacent objects are merged to a single
          object; a scalar value is autowrapped as an array and merged as an array;
          an adjacent array and object are merged by autowrapping the object as an
          array and merging the two arrays.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_MERGE_PRESERVE("[1, 2]", "[true, false]")
        # Escape output: "JSON_MERGE_PRESERVE('[1, 2]','[true, false]')"
        # Expect result: '[1, 2, true, false]'

        sqlfunc.JSON_MERGE_PRESERVE('{"name": "x"}', '{"id": 47}')
        # Escape output: "JSON_MERGE_PRESERVE('{"name": "x"}','{"id": 47}')"
        # Expect result: '{"id": 47, "name": "x"}'

        sqlfunc.JSON_MERGE_PRESERVE('{ "a": 1, "b": 2 }','{ "a": 3, "c": 4 }','{ "a": 5, "d": 6 }')
        # Escape output: "JSON_MERGE_PRESERVE('{ "a": 1, "b": 2 }','{ "a": 3, "c": 4 }','{ "a": 5, "d": 6 }')"
        # Expect result: '{"a": [1, 3, 5], "b": 2, "c": 4, "d": 6}'
        ```
        """
        if tuple_len(json_docs) < 2:
            raise errors.SQLFunctionError(
                "SQL function 'JSON_MERGE_PRESERVE' requires at least two JSON documents, "
                "instead got '%s'." % tuple_len(json_docs)
            )
        super().__init__("JSON_MERGE_PRESERVE", -1, *json_docs)


@cython.cclass
class JSON_OBJECT(SQLFunction):
    """Represents the `JSON_OBJECT([key, val[, key, val] ...])` function.

    MySQL description:
    - Evaluates a (possibly empty) list of key-value pairs and returns a JSON object containing
      those pairs. An error occurs if any key name is NULL or the number of arguments is odd.
    """

    def __init__(self, *key_val_pairs):
        """The `JSON_OBJECT([key, val[, key, val] ...])` function.

        MySQL description:
        - Evaluates a (possibly empty) list of key-value pairs and returns a JSON object containing
          those pairs. An error occurs if any key name is NULL or the number of arguments is odd.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_OBJECT("id", 87, "name", "carrot")
        # Escape output: "JSON_OBJECT('id',87,'name','carrot')"
        # Expect result: '{"id": 87, "name": "carrot"}'
        ```
        """
        _validate_args_paris(key_val_pairs, "JSON_OBJECT", "key_val_pairs")
        super().__init__("JSON_OBJECT", -1, *key_val_pairs)


@cython.cclass
class JSON_OVERLAPS(SQLFunction):
    """Represents the `JSON_OVERLAPS(json_doc1, json_doc2)` function.

    MySQL description:
    - Compares two JSON documents. Returns true (1) if the two document have any key-value
      pairs or array elements in common. If both arguments are scalars, the function performs
      a simple equality test. If either argument is NULL, the function returns NULL.
    - This function serves as counterpart to JSON_CONTAINS(), which requires all elements of
      the array searched for to be present in the array searched in. Thus, JSON_CONTAINS()
      performs an AND operation on search keys, while JSON_OVERLAPS() performs an OR operation.
    - Queries on JSON columns of InnoDB tables using JSON_OVERLAPS() in the WHERE clause can
      be optimized using multi-valued indexes. Multi-Valued Indexes, provides detailed information
      and examples.
    """

    def __init__(self, json_doc1, json_doc2):
        """The `JSON_OVERLAPS(json_doc1, json_doc2)` function.

        MySQL description:
        - Compares two JSON documents. Returns true (1) if the two document have any key-value
          pairs or array elements in common. If both arguments are scalars, the function performs
          a simple equality test. If either argument is NULL, the function returns NULL.
        - This function serves as counterpart to JSON_CONTAINS(), which requires all elements of
          the array searched for to be present in the array searched in. Thus, JSON_CONTAINS()
          performs an AND operation on search keys, while JSON_OVERLAPS() performs an OR operation.
        - Queries on JSON columns of InnoDB tables using JSON_OVERLAPS() in the WHERE clause can
          be optimized using multi-valued indexes. Multi-Valued Indexes, provides detailed information
          and examples.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_OVERLAPS("[1,3,5,7]", "[2,5,7]")
        # Escape output: "JSON_OVERLAPS('[1,3,5,7]','[2,5,7]')"
        # Expect result: 1

        sqlfunc.JSON_OVERLAPS("[1,3,5,7]", "[2,6,8]")
        # Escape output: "JSON_OVERLAPS('[1,3,5,7]','[2,6,8]')"
        # Expect result: 0
        ```
        """
        super().__init__("JSON_OVERLAPS", 2, json_doc1, json_doc2)


@cython.cclass
class JSON_PRETTY(SQLFunction):
    """Represents the `JSON_PRETTY(json_doc)` function.

    MySQL description:
    - Provides pretty-printing of JSON values similar to that implemented in PHP
      and by other languages and database systems. The value supplied must be a
      JSON value or a valid string representation of a JSON value. Extraneous
      whitespaces and newlines present in this value have no effect on the output.
    - For a NULL value, the function returns NULL. If the value is not a JSON
      document, or if it cannot be parsed as one, the function fails with an error.
    """

    def __init__(self, json_doc):
        """The `JSON_PRETTY(json_doc)` function.

        MySQL description:
        - Provides pretty-printing of JSON values similar to that implemented in PHP
          and by other languages and database systems. The value supplied must be a
          JSON value or a valid string representation of a JSON value. Extraneous
          whitespaces and newlines present in this value have no effect on the output.
        - For a NULL value, the function returns NULL. If the value is not a JSON
          document, or if it cannot be parsed as one, the function fails with an error.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_PRETTY("[1,3,5]")
        # Escape output: "JSON_PRETTY('[1,3,5]')"
        # Expect result: '[\n  1,\n  3,\n  5\n]'

        sqlfunc.JSON_PRETTY('{"a":"10","b":"15","x":"25"}')
        # Escape output: "JSON_PRETTY('{"a":"10","b":"15","x":"25"}')"
        # Expect result: '{\n  "a": "10",\n  "b": "15",\n  "x": "25"\n}'
        ```
        """
        super().__init__("JSON_PRETTY", 1, json_doc)


@cython.cclass
class JSON_QUOTE(SQLFunction):
    """Represents the `JSON_QUOTE(string)` function.

    MySQL description:
    - Quotes a string as a JSON value by wrapping it with double quote characters
      and escaping interior quote and other characters, then returning the result
      as a utf8mb4 string. Returns NULL if the argument is NULL.
    - This function is typically used to produce a valid JSON string literal for
      inclusion within a JSON document.
    """

    def __init__(self, string):
        """The `JSON_QUOTE(string)` function.

        MySQL description:
        - Quotes a string as a JSON value by wrapping it with double quote characters
          and escaping interior quote and other characters, then returning the result
          as a utf8mb4 string. Returns NULL if the argument is NULL.
        - This function is typically used to produce a valid JSON string literal for
          inclusion within a JSON document.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_QUOTE("null")
        # Escape output: "JSON_QUOTE('null')"
        # Expect result: '"null"'

        sqlfunc.JSON_QUOTE('"null"')
        # Escape output: "JSON_QUOTE('\"null\"')"
        # Expect result: '"\\"null\\""'

        sqlfunc.JSON_QUOTE("[1, 2, 3]")
        # Escape output: "JSON_QUOTE('[1, 2, 3]')"
        # Expect result: '"[1, 2, 3]"'
        ```
        """
        super().__init__("JSON_QUOTE", 1, string)


@cython.cclass
class JSON_REMOVE(SQLFunction):
    """Represents the `JSON_REMOVE(json_doc, path[, path] ...)` function.

    MySQL description:
    - Removes data from a JSON document and returns the result. Returns NULL if
      any argument is NULL. An error occurs if the json_doc argument is not a
      valid JSON document or any path argument is not a valid path expression
      or is $ or contains a * or ** wildcard.
    - The path arguments are evaluated left to right. The document produced by
      evaluating one path becomes the new value against which the next path is
      evaluated.
    - It is not an error if the element to be removed does not exist in the
      document; in that case, the path does not affect the document.
    """

    def __init__(self, json_doc, *paths):
        """The `JSON_REMOVE(json_doc, path[, path] ...)` function.

        MySQL description:
        - Removes data from a JSON document and returns the result. Returns NULL if
          any argument is NULL. An error occurs if the json_doc argument is not a
          valid JSON document or any path argument is not a valid path expression
          or is $ or contains a * or ** wildcard.
        - The path arguments are evaluated left to right. The document produced by
          evaluating one path becomes the new value against which the next path is
          evaluated.
        - It is not an error if the element to be removed does not exist in the
          document; in that case, the path does not affect the document.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '["a", ["b", "c"], "d"]'

        sqlfunc.JSON_REMOVE(js, "$[1]")
        # Escape output: "JSON_REMOVE('["a", ["b", "c"], "d"]','$[1]')"
        # Expect result: '["a", "d"]'

        sqlfunc.JSON_REMOVE(js, "$[1]", "$[1]")
        # Escape output: "JSON_REMOVE('["a", ["b", "c"], "d"]','$[1]','$[1]')"
        # Expect result: '["a"]'
        ```
        """
        super().__init__("JSON_REMOVE", -1, json_doc, *paths)


@cython.cclass
class JSON_REPLACE(SQLFunction):
    """Represents the `JSON_REPLACE(json_doc, path, val[, path, val] ...)` function.

    MySQL description:
    - Replaces existing values in a JSON document and returns the result. Returns NULL
      if json_doc or any path argument is NULL. An error occurs if the json_doc argument
      is not a valid JSON document or any path argument is not a valid path expression
      or contains a * or ** wildcard.
    - The path-value pairs are evaluated left to right. The document produced by evaluating
      one pair becomes the new value against which the next pair is evaluated.
    - A path-value pair for an existing path in the document overwrites the existing document
      value with the new value. A path-value pair for a nonexisting path in the document is
      ignored and has no effect.
    - The optimizer can perform a partial, in-place update of a JSON column instead of removing
      the old document and writing the new document in its entirety to the column. This optimization
      can be performed for an update statement that uses the JSON_REPLACE() function and meets the
      conditions outlined in Partial Updates of JSON Values.
    """

    def __init__(self, json_doc, *path_val_pairs):
        """The `JSON_REPLACE(json_doc, path, val[, path, val] ...)` function.

        MySQL description:
        - Replaces existing values in a JSON document and returns the result. Returns NULL
          if json_doc or any path argument is NULL. An error occurs if the json_doc argument
          is not a valid JSON document or any path argument is not a valid path expression
          or contains a * or ** wildcard.
        - The path-value pairs are evaluated left to right. The document produced by evaluating
          one pair becomes the new value against which the next pair is evaluated.
        - A path-value pair for an existing path in the document overwrites the existing document
          value with the new value. A path-value pair for a nonexisting path in the document is
          ignored and has no effect.
        - The optimizer can perform a partial, in-place update of a JSON column instead of removing
          the old document and writing the new document in its entirety to the column. This optimization
          can be performed for an update statement that uses the JSON_REPLACE() function and meets the
          conditions outlined in Partial Updates of JSON Values.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '{ "a": 1, "b": [2, 3]}'

        sqlfunc.JSON_REPLACE(js, "$.a", 10, "$.c", "[true, false]")
        # Escape output: "JSON_REPLACE('{ "a": 1, "b": [2, 3]}','$.a',10,'$.c','[true, false]')"
        # Expect result: '{"a": 10, "b": [2, 3]}'

        sqlfunc.JSON_REPLACE(js, "$.a", None, "$.c", "[true, false]")
        # Escape output: "JSON_REPLACE('{ "a": 1, "b": [2, 3]}','$.a',NULL,'$.c','[true, false]')"
        # Expect result: '{"a": null, "b": [2, 3]}'

        sqlfunc.JSON_REPLACE(None, "$.a", 10, "$.c", "[true, false]")
        # Escape output: "JSON_REPLACE(NULL,'$.a',10,'$.c','[true, false]')"
        # Expect result: NULL

        sqlfunc.JSON_REPLACE(js, None, 10, "$.c", "[true, false]")
        # Escape output: "JSON_REPLACE('{ "a": 1, "b": [2, 3]}',NULL,10,'$.c','[true, false]')"
        # Expect result: NULL
        ```
        """
        _validate_args_paris(path_val_pairs, "JSON_REPLACE", "path_val_pairs")
        super().__init__("JSON_REPLACE", -1, json_doc, *path_val_pairs)


@cython.cclass
class JSON_SEARCH(SQLFunction):
    """Represents the `JSON_SEARCH(json_doc, one_or_all, search_str[, escape_char[, path] ...])` function.

    MySQL description:
    - Returns the path to the given string within a JSON document. Returns NULL if any of the json_doc,
      search_str, or path arguments are NULL; no path exists within the document; or search_str is not
      found. An error occurs if the json_doc argument is not a valid JSON document, any path argument
      is not a valid path expression, one_or_all is not 'one' or 'all', or escape_char is not a constant
      expression.
    - The one_or_all argument affects the search as follows:
    - 'one': The search terminates after the first match and returns one path string. It is undefined
      which match is considered first.
    - 'all': The search returns all matching path strings such that no duplicate paths are included.
      If there are multiple strings, they are autowrapped as an array. The order of the array elements
      is undefined.
    - Within the search_str search string argument, the % and _ characters work as for the LIKE operator:
      % matches any number of characters (including zero characters), and _ matches exactly one character.
    - To specify a literal % or _ character in the search string, precede it by the escape character.
      The default is \ if the escape_char argument is missing or NULL. Otherwise, escape_char must be
      a constant that is empty or one character.
    """

    def __init__(
        self,
        json_doc,
        one_or_all,
        search_str,
        escape_char: str | Sentinel = IGNORED,
        path: str | Sentinel = IGNORED,
    ):
        """The `JSON_SEARCH(json_doc, one_or_all, search_str[, escape_char[, path] ...])` function.

        MySQL description:
        - Returns the path to the given string within a JSON document. Returns NULL if any of the json_doc,
          search_str, or path arguments are NULL; no path exists within the document; or search_str is not
          found. An error occurs if the json_doc argument is not a valid JSON document, any path argument
          is not a valid path expression, one_or_all is not 'one' or 'all', or escape_char is not a constant
          expression.
        - The one_or_all argument affects the search as follows:
        - 'one': The search terminates after the first match and returns one path string. It is undefined
          which match is considered first.
        - 'all': The search returns all matching path strings such that no duplicate paths are included.
          If there are multiple strings, they are autowrapped as an array. The order of the array elements
          is undefined.
        - Within the search_str search string argument, the % and _ characters work as for the LIKE operator:
          % matches any number of characters (including zero characters), and _ matches exactly one character.
        - To specify a literal % or _ character in the search string, precede it by the escape character.
          The default is \ if the escape_char argument is missing or NULL. Otherwise, escape_char must be
          a constant that is empty or one character.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '["abc", [{"k": "10"}, "def"], {"x":"abc"}, {"y":"bcd"}]'

        sqlfunc.JSON_SEARCH(js, "one", "abc")
        # Escape output: "JSON_SEARCH('["abc", [{"k": "10"}, "def"], {"x":"abc"}, {"y":"bcd"}]','one','abc')"
        # Expect result: '"$[0]"'

        sqlfunc.JSON_SEARCH(js, "all", "abc")
        # Escape output: "JSON_SEARCH('["abc", [{"k": "10"}, "def"], {"x":"abc"}, {"y":"bcd"}]','all','abc')"
        # Expect result: '["$[0]", "$[2].x"]'

        sqlfunc.JSON_SEARCH(js, "all", "10", None, "$")
        # Escape output: "JSON_SEARCH('["abc", [{"k": "10"}, "def"], {"x":"abc"}, {"y":"bcd"}]','all','10',NULL,'$')"
        # Expect result: '"$[1][0].k"'

        sqlfunc.JSON_SEARCH(js, "all", "%b%", "", "$[3]")
        # Escape output: "JSON_SEARCH('["abc", [{"k": "10"}, "def"], {"x":"abc"}, {"y":"bcd"}]','all','%b%','','$[3]')"
        # Expect result: '"$[3].y"'
        ```
        """
        if escape_char is IGNORED:
            super().__init__("JSON_SEARCH", 3, json_doc, one_or_all, search_str)
        elif path is IGNORED:
            super().__init__(
                "JSON_SEARCH", 4, json_doc, one_or_all, search_str, escape_char
            )
        else:
            super().__init__(
                "JSON_SEARCH", -1, json_doc, one_or_all, search_str, escape_char, path
            )


@cython.cclass
class JSON_SET(SQLFunction):
    """Represents the `JSON_SET(json_doc, path, val[, path, val] ...)` function.

    MySQL description:
    - Inserts or updates data in a JSON document and returns the result. Returns NULL
      if json_doc or path is NULL, or if path, when given, does not locate an object.
      Otherwise, an error occurs if the json_doc argument is not a valid JSON document
      or any path argument is not a valid path expression or contains a * or ** wildcard.
    - The path-value pairs are evaluated left to right. The document produced by evaluating
      one pair becomes the new value against which the next pair is evaluated.
    """

    def __init__(self, json_doc, *path_val_pairs):
        """The `JSON_SET(json_doc, path, val[, path, val] ...)` function.

        MySQL description:
        - Inserts or updates data in a JSON document and returns the result. Returns NULL
          if json_doc or path is NULL, or if path, when given, does not locate an object.
          Otherwise, an error occurs if the json_doc argument is not a valid JSON document
          or any path argument is not a valid path expression or contains a * or ** wildcard.
        - The path-value pairs are evaluated left to right. The document produced by evaluating
          one pair becomes the new value against which the next pair is evaluated.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '{ "a": 1, "b": [2, 3]}'

        sqlfunc.JSON_SET(js, "$.a", 10, "$.c", "[true, false]")
        # Escape output: "JSON_SET('{ "a": 1, "b": [2, 3]}','$.a',10,'$.c','[true, false]')"
        # Expect result: '{"a": 10, "b": [2, 3], "c": "[true, false]"}'
        ```
        """
        _validate_args_paris(path_val_pairs, "JSON_SET", "path_val_pairs")
        super().__init__("JSON_SET", -1, json_doc, *path_val_pairs)


@cython.cclass
class JSON_TYPE(SQLFunction):
    """Represents the `JSON_TYPE(json_val)` function.

    MySQL description:
    - Returns a utf8mb4 string indicating the type of a JSON value.
      This can be an object, an array, or a scalar type.
    """

    def __init__(self, json_val):
        """The `JSON_TYPE(json_val)` function.

        MySQL description:
        - Returns a utf8mb4 string indicating the type of a JSON value.
          This can be an object, an array, or a scalar type.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        js = '{"a": [10, true]}'

        sqlfunc.JSON_TYPE(sqlfunc.JSON_EXTRACT(js, "$.a"))
        # Escape output: "JSON_TYPE(JSON_EXTRACT('{"a": [10, true]}','$.a'))"
        # Expect result: 'ARRAY'

        sqlfunc.JSON_TYPE(sqlfunc.JSON_EXTRACT(js, "$.a[0]"))
        # Escape output: "JSON_TYPE(JSON_EXTRACT('{"a": [10, true]}','$.a[0]'))"
        # Expect result: 'INTEGER'

        sqlfunc.JSON_TYPE(sqlfunc.JSON_EXTRACT(js, "$.a[1]"))
        # Escape output: "JSON_TYPE(JSON_EXTRACT('{"a": [10, true]}','$.a[1]'))"
        # Expect result: 'BOOLEAN'
        """
        super().__init__("JSON_TYPE", 1, json_val)


@cython.cclass
class JSON_UNQUOTE(SQLFunction):
    """Represents the `JSON_UNQUOTE(json_val)` function.

    MySQL description:
    - Unquotes JSON value and returns the result as a utf8mb4 string. Returns NULL if
      the argument is NULL. An error occurs if the value starts and ends with double
      quotes but is not a valid JSON string literal.
    """

    def __init__(self, json_val):
        """The `JSON_UNQUOTE(json_val)` function.

        MySQL description:
        - Unquotes JSON value and returns the result as a utf8mb4 string. Returns NULL if
          the argument is NULL. An error occurs if the value starts and ends with double
          quotes but is not a valid JSON string literal.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_UNQUOTE('"abc"')
        # Escape output: "JSON_UNQUOTE('\"abc\"')"
        # Expect result: 'abc'

        sqlfunc.JSON_UNQUOTE(sqlfunc.JSON_QUOTE("abc"))
        # Escape output: "JSON_UNQUOTE(JSON_QUOTE('abc'))"
        # Expect result: 'abc'
        ```
        """
        super().__init__("JSON_UNQUOTE", 1, json_val)


@cython.cclass
class JSON_VALID(SQLFunction):
    """Represents the `JSON_VALID(val)` function.

    MySQL description:
    - Returns 0 or 1 to indicate whether a value is valid JSON.
    - Returns NULL if the argument is NULL.
    """

    def __init__(self, val):
        """The `JSON_VALID(val)` function.

        MySQL description:
        - Returns 0 or 1 to indicate whether a value is valid JSON.
        - Returns NULL if the argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.JSON_VALID('{"a": 1}')
        # Escape output: "JSON_VALID('{"a": 1}')"
        # Expect result: 1

        sqlfunc.JSON_VALID('hello')
        # Escape output: "JSON_VALID('hello')"
        # Expect result: 0
        ```
        """
        super().__init__("JSON_VALID", 1, val)


# Functions: L -------------------------------------------------------------------------------------------------------
@cython.cclass
class LAST_DAY(SQLFunction):
    """Represents the `LAST_DAY(date)` function.

    MySQL description:
    - Takes a date or datetime value and returns the corresponding value
      for the last day of the month.
    - Returns NULL if the argument is invalid or NULL.
    """

    def __init__(self, date):
        """The `LAST_DAY(date)` function.

        MySQL description:
        - Takes a date or datetime value and returns the corresponding value
          for the last day of the month.
        - Returns NULL if the argument is invalid or NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.LAST_DAY("2003-02-05")
        # Escape output: "LAST_DAY('2003-02-05')"
        # Expect result: '2003-02-28'

        sqlfunc.LAST_DAY(datetime.date(2004, 2, 5))
        # Escape output: "LAST_DAY('2004-02-05')"
        # Expect result: '2004-02-29'

        sqlfunc.LAST_DAY("2004-01-01 01:01:01")
        # Escape output: "LAST_DAY('2004-01-01 01:01:01')"
        # Expect result: '2004-01-31'
        ```
        """
        super().__init__("LAST_DAY", 1, date)


@cython.cclass
class LEAST(SQLFunction):
    """Represents the `LEAST(value1, value2, ...)` function.

    MySQL description:
    - With two or more arguments, returns the smallest (minimum-valued) argument.
      The arguments are compared using the following rules:
    - If any argument is NULL, the result is NULL. No comparison is needed.
    - If all arguments are integer-valued, they are compared as integers.
    - If at least one argument is double precision, they are compared as double-precision
      values. Otherwise, if at least one argument is a DECIMAL value, they are compared
      as DECIMAL values.
    - If the arguments comprise a mix of numbers and strings, they are compared as strings.
    - If any argument is a nonbinary (character) string, the arguments are compared as
      nonbinary strings.
    - In all other cases, the arguments are compared as binary strings.
    """

    def __init__(self, *values):
        """The `LEAST(value1, value2, ...)` function.

        MySQL description:
        - With two or more arguments, returns the smallest (minimum-valued) argument.
          The arguments are compared using the following rules:
        - If any argument is NULL, the result is NULL. No comparison is needed.
        - If all arguments are integer-valued, they are compared as integers.
        - If at least one argument is double precision, they are compared as double-precision
          values. Otherwise, if at least one argument is a DECIMAL value, they are compared
          as DECIMAL values.
        - If the arguments comprise a mix of numbers and strings, they are compared as strings.
        - If any argument is a nonbinary (character) string, the arguments are compared as
          nonbinary strings.
        - In all other cases, the arguments are compared as binary strings.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LEAST(2, 0)
        # Escape output: "LEAST(2,0)"
        # Expect result: 0

        sqlfunc.LEAST("B", "A", "C")
        # Escape output: "LEAST('B','A','C')"
        # Expect result: 'A'
        ```
        """
        super().__init__("LEAST", -1, *values)


@cython.cclass
class LEFT(SQLFunction):
    """Represents the `LEFT(string, length)` function.

    MySQL description:
    - Returns the leftmost len characters from the string str.
    - Returns NULL if any argument is NULL.
    """

    def __init__(self, string, length):
        """The `LEFT(string, length)` function.

        MySQL description:
        - Returns the leftmost len characters from the string str.
        - Returns NULL if any argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LEFT("foobarbar", 5)
        # Escape output: "LEFT('foobarbar',5)"
        # Expect result: 'fooba'
        ```
        """
        super().__init__("LEFT", 2, string, length)


@cython.cclass
class LENGTH(SQLFunction):
    """Represents the `LENGTH(string)` function.

    MySQL description:
    - Returns the length of the string str, measured in bytes. A multibyte
      character counts as multiple bytes. This means that for a string
      containing five 2-byte characters, LENGTH() returns 10, whereas
      CHAR_LENGTH() returns 5.
    - Returns NULL if str is NULL.
    """

    def __init__(self, string):
        """The `LENGTH(string)` function.

        MySQL description:
        - Returns the length of the string str, measured in bytes. A multibyte
          character counts as multiple bytes. This means that for a string
          containing five 2-byte characters, LENGTH() returns 10, whereas
          CHAR_LENGTH() returns 5.
        - Returns NULL if str is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LENGTH("text")
        # Escape output: "LENGTH('text')"
        # Expect result: 4
        ```
        """
        super().__init__("LENGTH", 1, string)


@cython.cclass
class LineString(SQLFunction):
    """Represents the `LineString(pt [, pt] ...)` function.

    MySQL description:
    - Constructs a LineString value from a number of Point or WKB Point arguments.
    - If the number of arguments is less than two, the return value is NULL.
    """

    def __init__(self, *pt):
        """The `LineString(pt [, pt] ...)` function.

        MySQL description:
        - Constructs a LineString value from the specified WKT (Well-Known Text) format
          representation. The LineString is a collection of points that are joined by
          line segments.
        - Returns NULL if any argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LineString(sqlfunc.Point(1, 1), sqlfunc.Point(2, 2))
        # Escape output: "LineString(Point(1,1),Point(2,2))"
        # Expect result: 'LINESTRING(1 1, 2 2)'
        ```
        """
        super().__init__("LineString", -1, *pt)


@cython.cclass
class LN(SQLFunction):
    """Represents the `LN(X)` function.

    MySQL description:
    - Returns the natural logarithm of X; that is, the base-e logarithm of
      X. If X is less than or equal to 0.0E0, the function returns NULL and
      a warning “Invalid argument for logarithm” is reported.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `LN(X)` function.

        MySQL description:
        - Returns the natural logarithm of X; that is, the base-e logarithm of
          X. If X is less than or equal to 0.0E0, the function returns NULL and
          a warning “Invalid argument for logarithm” is reported.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LN(2)
        # Escape output: "LN(2)"
        # Expect result: 0.69314718055995

        sqlfunc.LN("-2")
        # Escape output: "LN('-2')"
        # Expect result: NULL
        ```
        """
        super().__init__("LN", 1, X)


@cython.cclass
class LOAD_FILE(SQLFunction):
    """Represents the `LOAD_FILE(file_name)` function.

    MySQL description:
    - Reads the file and returns the file contents as a string. To use this function,
      the file must be located on the server host, you must specify the full path name
      to the file, and you must have the FILE privilege. The file must be readable by
      the server and its size less than max_allowed_packet bytes. If the secure_file_priv
      system variable is set to a nonempty directory name, the file to be loaded must be
      located in that directory.
    - If the file does not exist or cannot be read because one of the preceding conditions
      is not satisfied, the function returns NULL.
    """

    def __init__(self, file_name):
        """The `LOAD_FILE(file_name)` function.

        MySQL description:
        - Reads the file and returns the file contents as a string. To use this function,
          the file must be located on the server host, you must specify the full path name
          to the file, and you must have the FILE privilege. The file must be readable by
          the server and its size less than max_allowed_packet bytes. If the secure_file_priv
          system variable is set to a nonempty directory name, the file to be loaded must be
          located in that directory.
        - If the file does not exist or cannot be read because one of the preceding conditions
          is not satisfied, the function returns NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LOAD_FILE("/tmp/picture")
        # Escape output: "LOAD_FILE('/tmp/picture')"
        ```
        """
        super().__init__("LOAD_FILE", 1, file_name)


@cython.cclass
class LOCALTIME(SQLFunction):
    """Represents the `LOCALTIME([fps])` function.

    MySQL description:
    - Returns the current date and time as a value in 'YYYY-MM-DD hh:mm:ss' or
      YYYYMMDDhhmmss format, depending on whether the function is used in string
      or numeric context. The value is expressed in the session time zone.
    - If the fsp argument is given to specify a fractional seconds precision from
      0 to 6, the return value includes a fractional seconds part of that many digits.
    """

    def __init__(self, fsp: Any | Sentinel = IGNORED):
        """The `LOCALTIME([fps])` function.

        MySQL description:
        - Returns the current date and time as a value in 'YYYY-MM-DD hh:mm:ss' or
          YYYYMMDDhhmmss format, depending on whether the function is used in string
          or numeric context. The value is expressed in the session time zone.
        - If the fsp argument is given to specify a fractional seconds precision from
          0 to 6, the return value includes a fractional seconds part of that many digits.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LOCALTIME()
        # Escape output: "LOCALTIME()"
        # Expect result: '2021-12-25 19:25:37'

        sqlfunc.LOCALTIME(3)
        # Escape output: "LOCALTIME(3)"
        # Expect result: '2021-12-25 19:25:37.840'
        ```
        """
        if fsp is IGNORED:
            super().__init__("LOCALTIME", 0)
        else:
            super().__init__("LOCALTIME", 1, fsp)


@cython.cclass
class LOCATE(SQLFunction):
    """Represents the `LOCATE(substr, string[, pos])` function.

    MySQL description:
    - The first syntax returns the position of the first occurrence of substring
      substr in string str. The second syntax returns the position of the first
      occurrence of substring substr in string str, starting at position pos.
    - Returns 0 if substr is not in str. Returns NULL if any argument is NULL.
    """

    def __init__(self, substr, string, pos: Any | Sentinel = IGNORED):
        """The `LOCATE(substr, string[, pos])` function.

        MySQL description:
        - The first syntax returns the position of the first occurrence of substring
          substr in string str. The second syntax returns the position of the first
          occurrence of substring substr in string str, starting at position pos.
        - Returns 0 if substr is not in str. Returns NULL if any argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LOCATE("bar", "foobarbar")
        # Escape output: "LOCATE('bar','foobarbar')"
        # Expect result: 4

        sqlfunc.LOCATE("bar", "foobarbar", 5)
        # Escape output: "LOCATE('bar','foobarbar',5)"
        # Expect result: 7
        ```
        """
        if pos is IGNORED:
            super().__init__("LOCATE", 2, substr, string)
        else:
            super().__init__("LOCATE", 3, substr, string, pos)


@cython.cclass
class LOG(SQLFunction):
    """Represents the `LOG(X[, B])` function.

    MySQL description:
    - If called with one parameter, this function returns the natural logarithm
      of X. If X is less than or equal to 0.0E0, the function returns NULL and
      a warning “Invalid argument for logarithm” is reported.
    - If called with two parameters, this function returns the logarithm of X to
      the base B. If X is less than or equal to 0, or if B is less than or equal
      to 1, then NULL is returned.
    - Returns NULL if X or B is NULL.
    """

    def __init__(self, X, B: Any | Sentinel = IGNORED):
        """The `LOG(X[, B])` function.

        MySQL description:
        - If called with one parameter, this function returns the natural logarithm
          of X. If X is less than or equal to 0.0E0, the function returns NULL and
          a warning “Invalid argument for logarithm” is reported.
        - If called with two parameters, this function returns the logarithm of X to
          the base B. If X is less than or equal to 0, or if B is less than or equal
          to 1, then NULL is returned.
        - Returns NULL if X or B is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LOG(2)
        # Escape output: "LOG(2)"
        # Expect result: 0.69314718055995

        sqlfunc.LOG(2, 10)
        # Escape output: "LOG(10,100)"
        # Expect result: 2
        ```
        """
        if B is IGNORED:
            super().__init__("LOG", 1, X)
        else:
            super().__init__("LOG", 2, X, B)


@cython.cclass
class LOG2(SQLFunction):
    """Represents the `LOG2(X)` function.

    MySQL description:
    - Returns the base-2 logarithm of X. If X is less than or equal to 0.0E0,
      the function returns NULL and a warning “Invalid argument for logarithm”
      is reported.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `LOG2(X)` function.

        MySQL description:
        - Returns the base-2 logarithm of X. If X is less than or equal to 0.0E0,
          the function returns NULL and a warning “Invalid argument for logarithm”
          is reported.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LOG2(65536)
        # Escape output: "LOG2(65536)"
        # Expect result: 16

        sqlfunc.LOG2("-100")
        # Escape output: "LOG2('-100')"
        # Expect result: NULL
        ```
        """
        super().__init__("LOG2", 1, X)


@cython.cclass
class LOG10(SQLFunction):
    """Represents the `LOG10(X)` function.

    MySQL description:
    - Returns the base-10 logarithm of X. If X is less than or equal to 0.0E0,
      the function returns NULL and a warning “Invalid argument for logarithm”
      is reported.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `LOG10(X)` function.

        MySQL description:
        - Returns the base-10 logarithm of X. If X is less than or equal to 0.0E0,
          the function returns NULL and a warning “Invalid argument for logarithm”
          is reported.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LOG10(2)
        # Escape output: "LOG10(2)"
        # Expect result: 0.30102999566398

        sqlfunc.LOG10("100")
        # Escape output: "LOG10('100')"
        # Expect result: 2
        ```
        """
        super().__init__("LOG10", 1, X)


@cython.cclass
class LOWER(SQLFunction):
    """Represents the `LOWER(string)` function.

    MySQL description:
    - Returns the string str with all characters changed to lowercase
      according to the current character set mapping.
    - Returns NULL if str is NULL. The default character set is utf8mb4.
    """

    def __init__(self, string):
        """The `LOWER(string)` function.

        MySQL description:
        - Returns the string str with all characters changed to lowercase
          according to the current character set mapping.
        - Returns NULL if str is NULL. The default character set is utf8mb4.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LOWER("QUADRATICALLY")
        # Escape output: "LOWER('QUADRATICALLY')"
        # Expect result: 'quadratically'
        ```
        """
        super().__init__("LOWER", 1, string)


@cython.cclass
class LPAD(SQLFunction):
    """Represents the `LPAD(string, length, pad_string)` function.

    MySQL description:
    - Returns the string str, left-padded with the string padstr to a length of
      len characters. If str is longer than len, the return value is shortened
      to len characters.
    """

    def __init__(self, string, length, pad_string):
        """The `LPAD(string, length, pad_string)` function.

        MySQL description:
        - Returns the string str, left-padded with the string padstr to a length of
          len characters. If str is longer than len, the return value is shortened
          to len characters.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LPAD("hi", 4, "?")
        # Escape output: "LPAD('hi',4,'?')"
        # Expect result: '??hi'

        sqlfunc.LPAD("hi", 1, "?")
        # Escape output: "LPAD('hi',1,'?')"
        # Expect result: 'h'
        ```
        """
        super().__init__("LPAD", 3, string, length, pad_string)


@cython.cclass
class LTRIM(SQLFunction):
    """Represents the `LTRIM(string)` function.

    MySQL description:
    - Returns the string str with leading space characters removed.
    - Returns NULL if str is NULL.
    """

    def __init__(self, string):
        """The `LTRIM(string)` function.

        MySQL description:
        - Returns the string str with leading space characters removed.
        - Returns NULL if str is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.LTRIM("  barbar")
        # Escape output: "LTRIM('  barbar')"
        # Expect result: 'barbar'
        ```
        """
        super().__init__("LTRIM", 1, string)


# Functions: M -------------------------------------------------------------------------------------------------------
@cython.cclass
class MAKE_SET(SQLFunction):
    """Represents the `MAKE_SET(bits, str1, str2, ...)` function.

    MySQL description:
    - Returns a set value (a string containing substrings separated by ',' characters)
      consisting of the strings that have the corresponding bit in bits set. str1
      corresponds to bit 0, str2 to bit 1, and so on.
    - NULL values in str1, str2, ... are not appended to the result.
    """

    def __init__(self, bits, *strings):
        """The `MAKE_SET(bits, str1, str2, ...)` function.

        MySQL description:
        - Returns a set value (a string containing substrings separated by ',' characters)
          consisting of the strings that have the corresponding bit in bits set. str1
          corresponds to bit 0, str2 to bit 1, and so on.
        - NULL values in str1, str2, ... are not appended to the result.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.MAKE_SET(1, "a", "b", "c")
        # Escape output: "MAKE_SET(1,'a','b','c')"
        # Expect result: 'a'

        sqlfunc.MAKE_SET(1 | 4, "hello", "nice", "world")
        # Escape output: "MAKE_SET(5,'hello','nice','world')"
        # Expect result: 'hello,world'

        sqlfunc.MAKE_SET(1 | 4, "hello", "nice", None, "world")
        # Escape output: "MAKE_SET(5,'hello','nice',NULL,'world')"
        # Expect result: 'hello'

        sqlfunc.MAKE_SET(0, "a", "b", "c")
        # Escape output: "MAKE_SET(0,'a','b','c')"
        # Expect result: ''
        ```
        """
        super().__init__("MAKE_SET", -1, bits, *strings)


@cython.cclass
class MAKEDATE(SQLFunction):
    """Represents the `MAKEDATE(year, dayofyear) function.

    MySQL description:
    - Returns a date, given year and day-of-year values. dayofyear must
      be greater than 0 or the result is NULL.
    - The result is also NULL if either argument is NULL.
    """

    def __init__(self, year, dayofyear):
        """The `MAKEDATE(year, dayofyear)` function.

        MySQL description:
        - Returns a date, given year and day-of-year values. dayofyear must
          be greater than 0 or the result is NULL.
        - The result is also NULL if either argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.MAKEDATE(2011, 31)
        # Escape output: "MAKEDATE(2011,31)"
        # Expect result: '2011-01-31'

        sqlfunc.MAKEDATE(2011, 0)
        # Escape output: "MAKEDATE(2011,0)"
        # Expect result: NULL
        ```
        """
        super().__init__("MAKEDATE", 2, year, dayofyear)


@cython.cclass
class MAKETIME(SQLFunction):
    """Represents the `MAKETIME(hour, minute, second)` function.

    MySQL description:
    - Returns a time value calculated from the hour, minute, and second
      arguments. Returns NULL if any of its arguments are NULL.
    - The second argument can have a fractional part.
    """

    def __init__(self, hour, minute, second):
        """The `MAKETIME(hour, minute, second)` function.

        MySQL description:
        - Returns a time value calculated from the hour, minute, and second
          arguments. Returns NULL if any of its arguments are NULL.
        - The second argument can have a fractional part.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.MAKETIME(12, 15, 30)
        # Escape output: "MAKETIME(12,15,30)"
        # Expect result: '12:15:30'

        sqlfunc.MAKETIME(12, 15, 30.123)
        # Escape output: "MAKETIME(12,15,30.123)"
        # Expect result: '12:15:30.123'
        ```
        """
        super().__init__("MAKETIME", 3, hour, minute, second)


@cython.cclass
class MBRContains(SQLFunction):
    """Represents the `MBRContains(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether the minimum bounding rectangle
      of g1 contains the minimum bounding rectangle of g2.
    - This tests the opposite relationship as MBRWithin().
    """

    def __init__(self, g1, g2):
        """The `MBRContains(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether the minimum bounding rectangle
          of g1 contains the minimum bounding rectangle of g2.
        - This tests the opposite relationship as MBRWithin().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1 = "Polygon((0 0,0 3,3 3,3 0,0 0))"
        g2 = "Polygon((1 1,1 2,2 2,2 1,1 1))"
        g3 = "Polygon((5 5,5 10,10 10,10 5,5 5))"
        p1, p2 = "Point(1 1)", "Point(3 3)"

        sqlfunc.MBRContains(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2))
        # Escape output: "MBRContains(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 1

        sqlfunc.MBRContains(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g3))
        # Escape output: "MBRContains(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))"
        # Expect result: 0

        sqlfunc.MBRContains(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(p1))
        # Escape output: "MBRContains(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Point(1 1)'))"
        # Expect result: 1

        sqlfunc.MBRContains(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(p2))
        # Escape output: "MBRContains(ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'),ST_GeomFromText('Point(3 3)'))"
        # Expect result: 0
        ```
        """
        super().__init__("MBRContains", 2, g1, g2)


@cython.cclass
class MBRCoveredBy(SQLFunction):
    """Represents the `MBRCoveredBy(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether the minimum bounding rectangle of g1
      is covered by the minimum bounding rectangle of g2.
    - This tests the opposite relationship as MBRCovers().
    """

    def __init__(self, g1, g2):
        """The `MBRCoveredBy(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether the minimum bounding rectangle of g1
          is covered by the minimum bounding rectangle of g2.
        - This tests the opposite relationship as MBRCovers().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Point(1 1)"

        sqlfunc.MBRCoveredBy(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2))
        # Escape output: "MBRCoveredBy(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Point(1 1)'))"
        # Expect result: 0

        sqlfunc.MBRCoveredBy(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(g1))
        # Escape output: "MBRCoveredBy(ST_GeomFromText('Point(1 1)'),ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'))"
        # Expect result: 1
        ```
        """
        super().__init__("MBRCoveredBy", 2, g1, g2)


@cython.cclass
class MBRCovers(SQLFunction):
    """Represents the `MBRCovers(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether the minimum bounding rectangle of g1
      covers the minimum bounding rectangle of g2.
    - This tests the opposite relationship as MBRCoveredBy().
    """

    def __init__(self, g1, g2):
        """The `MBRCovers(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether the minimum bounding rectangle of g1
          covers the minimum bounding rectangle of g2.
        - This tests the opposite relationship as MBRCoveredBy().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Point(1 1)"

        sqlfunc.MBRCovers(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2))
        # Escape output: "MBRCovers(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Point(1 1)'))"
        # Expect result: 1

        sqlfunc.MBRCovers(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(g1))
        # Escape output: "MBRCovers(ST_GeomFromText('Point(1 1)'),ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'))"
        # Expect result: 0
        ```
        """
        super().__init__("MBRCovers", 2, g1, g2)


@cython.cclass
class MBRDisjoint(SQLFunction):
    """Represents the `MBRDisjoint(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether the minimum bounding rectangles
      of the two geometries g1 and g2 are disjoint (do not intersect).
    """

    def __init__(self, g1, g2):
        """The `MBRDisjoint(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether the minimum bounding rectangles
          of the two geometries g1 and g2 are disjoint (do not intersect).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        g3, g4 = "Polygon((0 0,0 5,5 5,5 0,0 0))", "Polygon((5 5,5 10,10 10,10 5,5 5))"

        sqlfunc.MBRDisjoint(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g4))
        # Escape output: "MBRDisjoint(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))"
        # Expect result: 1

        sqlfunc.MBRDisjoint(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(g4))
        # Escape output: "MBRDisjoint(ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))"
        # Expect result: 1

        sqlfunc.MBRDisjoint(sqlfunc.ST_GeomFromText(g3), sqlfunc.ST_GeomFromText(g4))
        # Escape output: "MBRDisjoint(ST_GeomFromText('Polygon((0 0,0 5,5 5,5 0,0 0))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))"
        # Expect result: 0

        sqlfunc.MBRDisjoint(sqlfunc.ST_GeomFromText(g4), sqlfunc.ST_GeomFromText(g4))
        # Escape output: "MBRDisjoint(ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))"
        # Expect result: 0
        ```
        """
        super().__init__("MBRDisjoint", 2, g1, g2)


@cython.cclass
class MBREquals(SQLFunction):
    """Represents the `MBREquals(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether the minimum bounding rectangles
      of the two geometries g1 and g2 are the same.
    """

    def __init__(self, g1, g2):
        """The `MBREquals(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether the minimum bounding rectangles
          of the two geometries g1 and g2 are the same.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.MBREquals(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g1))
        # Escape output: "MBREquals(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'))"
        # Expect result: 1

        sqlfunc.MBREquals(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2))
        # Escape output: "MBREquals(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 0
        ```
        """
        super().__init__("MBREquals", 2, g1, g2)


@cython.cclass
class MBRIntersects(SQLFunction):
    """Represents the `MBRIntersects(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether the minimum bounding
      rectangles of the two geometries g1 and g2 intersect.
    """

    def __init__(self, g1, g2):
        """The `MBRIntersects(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether the minimum bounding
          rectangles of the two geometries g1 and g2 intersect.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        g3, g4 = "Polygon((0 0,0 5,5 5,5 0,0 0))", "Polygon((5 5,5 10,10 10,10 5,5 5))"

        sqlfunc.MBRIntersects(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2))
        # Escape output: "MBRIntersects(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 1

        sqlfunc.MBRIntersects(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g3))
        # Escape output: "MBRIntersects(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((0 0,0 5,5 5,5 0,0 0))'))"
        # Expect result: 1

        sqlfunc.MBRIntersects(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g4))
        # Escape output: "MBRIntersects(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))"
        # Expect result: 0
        ```
        """
        super().__init__("MBRIntersects", 2, g1, g2)


@cython.cclass
class MBROverlaps(SQLFunction):
    """Represents the `MBROverlaps(g1, g2)` function.

    MySQL description:
    - Two geometries spatially overlap if they intersect and their intersection
      results in a geometry of the same dimension but not equal to either of the
      given geometries.
    - This function returns 1 or 0 to indicate whether the minimum bounding
      rectangles of the two geometries g1 and g2 overlap.
    """

    def __init__(self, g1, g2):
        """The `MBROverlaps(g1, g2)` function.

        MySQL description:
        - Two geometries spatially overlap if they intersect and their intersection
          results in a geometry of the same dimension but not equal to either of the
          given geometries.
        - This function returns 1 or 0 to indicate whether the minimum bounding
          rectangles of the two geometries g1 and g2 overlap.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.MBROverlaps(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2))
        # Escape output: "MBROverlaps(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 0
        ```
        """
        super().__init__("MBROverlaps", 2, g1, g2)


@cython.cclass
class MBRTouches(SQLFunction):
    """Represents the `MBRTouches(g1, g2)` function.

    MySQL description:
    - Two geometries spatially touch if their interiors do not intersect, but
      the boundary of one of the geometries intersects either the boundary or
      the interior of the other.
    - This function returns 1 or 0 to indicate whether the minimum bounding
      rectangles of the two geometries g1 and g2 touch.
    """

    def __init__(self, g1, g2):
        """The `MBRTouches(g1, g2)` function.

        MySQL description:
        - Two geometries spatially touch if their interiors do not intersect, but
          the boundary of one of the geometries intersects either the boundary or
          the interior of the other.
        - This function returns 1 or 0 to indicate whether the minimum bounding
          rectangles of the two geometries g1 and g2 touch.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.MBRTouches(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2))
        # Escape output: "MBRTouches(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 0
        ```
        """
        super().__init__("MBRTouches", 2, g1, g2)


@cython.cclass
class MBRWithin(SQLFunction):
    """Represents the `MBRWithin(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether the minimum bounding rectangle of g1
      is within the minimum bounding rectangle of g2. This tests the opposite
      relationship as MBRContains().
    """

    def __init__(self, g1, g2):
        """The `MBRWithin(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether the minimum bounding rectangle of g1
          is within the minimum bounding rectangle of g2. This tests the opposite
          relationship as MBRContains().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.MBRWithin(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2))
        # Escape output: "MBRWithin(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 0

        sqlfunc.MBRWithin(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(g1))
        # Escape output: "MBRWithin(ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'),ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'))"
        # Expect result: 1
        ```
        """
        super().__init__("MBRWithin", 2, g1, g2)


@cython.cclass
class MD5(SQLFunction):
    """Represents the `MD5(string)` function.

    MySQL description:
    - Calculates an MD5 128-bit checksum for the string. The value is
      returned as a string of 32 hexadecimal digits, or NULL if the
      argument was NULL. The return value can, for example, be used as
      a hash key.
    - The return value is a string in the connection character set. If
      FIPS mode is enabled, MD5() returns NULL
    """

    def __init__(self, string):
        """The `MD5(string)` function.

        MySQL description:
        - Calculates an MD5 128-bit checksum for the string. The value is
          returned as a string of 32 hexadecimal digits, or NULL if the
          argument was NULL. The return value can, for example, be used as
          a hash key.
        - The return value is a string in the connection character set. If
          FIPS mode is enabled, MD5() returns NULL

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.MD5("testing")
        # Escape output: "MD5('testing')"
        # Expect result: 'ae2b1fca515949e5d54fb22b8ed95575'
        ```
        """
        super().__init__("MD5", 1, string)


@cython.cclass
class MICROSECOND(SQLFunction):
    """Represents the `MICROSECOND(expr)` function.

    MySQL description:
    - Returns the microseconds from the time or datetime expression expr as
      a number in the range from 0 to 999999.
    - Returns NULL if expr is NULL.
    """

    def __init__(self, expr):
        """The `MICROSECOND(expr)` function.

        MySQL description:
        - Returns the microseconds from the time or datetime expression expr as
          a number in the range from 0 to 999999.
        - Returns NULL if expr is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.MICROSECOND("12:00:00.123456")
        # Escape output: "MICROSECOND('12:00:00.123456')"
        # Expect result: 123456

        sqlfunc.MICROSECOND(datetime.time(23, 59, 59, 10))
        # Escape output: "MICROSECOND('23:59:59.000010')"
        # Expect result: 10
        ```
        """
        super().__init__("MICROSECOND", 1, expr)


@cython.cclass
class MINUTE(SQLFunction):
    """Represents the `MINUTE(expr)` function.

    MySQL description:
    - Returns the minute for time or datetime expression expr as a number in
      the range from 0 to 59.
    - Returns NULL if expr is NULL.
    """

    def __init__(self, expr):
        """The `MINUTE(expr)` function.

        MySQL description:
        - Returns the minute for time or datetime expression expr as a number in
          the range from 0 to 59.
        - Returns NULL if expr is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.MINUTE("2008-02-03 10:05:03")
        # Escape output: "MINUTE('2008-02-03 10:05:03')"
        # Expect result: 5

        sqlfunc.MINUTE(datetime.time(23, 59, 59))
        # Escape output: "MINUTE('23:59:59')"
        # Expect result: 59
        ```
        """
        super().__init__("MINUTE", 1, expr)


@cython.cclass
class MOD(SQLFunction):
    """Represents the `MOD(N, M)` function.

    MySQL description:
    - Modulo operation. Returns the remainder of N divided by M.
    - Returns NULL if M or N is NULL.
    """

    def __init__(self, N, M):
        """The `MOD(N, M)` function.

        MySQL description:
        - Modulo operation. Returns the remainder of N divided by M.
        - Returns NULL if M or N is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.MOD(234, 10)
        # Escape output: "MOD(234,10)"
        # Expect result: 4

        sqlfunc.MOD("29", "9")
        # Escape output: "MOD('29','9')"
        # Expect result: 2
        ```
        """
        super().__init__("MOD", 2, N, M)


@cython.cclass
class MONTH(SQLFunction):
    """Represents the `MONTH(date)` function.

    MySQL description:
    - Returns the month for date, in the range 1 to 12 for January
      to December, or 0 for dates such as '0000-00-00' or '2008-00-00'
      that have a zero month part.
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `MONTH(date)` function.

        MySQL description:
        - Returns the month for date, in the range 1 to 12 for January
          to December, or 0 for dates such as '0000-00-00' or '2008-00-00'
          that have a zero month part.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.MONTH("2008-02-03")
        # Escape output: "MONTH('2008-02-03')"
        # Expect result: 2

        sqlfunc.MONTH(datetime.datetime(2010, 12, 31))
        # Escape output: "MONTH('2010-12-31 00:00:00')"
        # Expect result: 12
        ```
        """
        super().__init__("MONTH", 1, date)


@cython.cclass
class MONTHNAME(SQLFunction):
    """Represents the `MONTHNAME(date)` function.

    MySQL description:
    - Returns the full name of the month for date. The language used for the
      name is controlled by the value of the lc_time_names system variable
      [link](https://dev.mysql.com/doc/refman/8.4/en/locale-support.html).
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `MONTHNAME(date)` function.

        MySQL description:
        - Returns the full name of the month for date. The language used for the
          name is controlled by the value of the lc_time_names system variable
          [link](https://dev.mysql.com/doc/refman/8.4/en/locale-support.html).
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.MONTHNAME("2008-02-03")
        # Escape output: "MONTHNAME('2008-02-03')"
        # Expect result: 'February'

        sqlfunc.MONTHNAME(datetime.datetime(2010, 12, 31))
        # Escape output: "MONTHNAME('2010-12-31 00:00:00')"
        # Expect result: 'December'
        ```
        """
        super().__init__("MONTHNAME", 1, date)


@cython.cclass
class MultiLineString(SQLFunction):
    """Represents the `MultiLineString(ls [, ls] ...)` function.

    MySQL description:
    - Constructs a MultiLineString value using LineString or WKB LineString arguments.
    """

    def __init__(self, *ls):
        """The `MultiLineString(ls [, ls] ...)` function.

        MySQL description:
        - Constructs a MultiLineString value using LineString or WKB LineString arguments.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.MultiLineString(
            sqlfunc.LineString(sqlfunc.Point(1, 1), sqlfunc.Point(2, 2))
        )
        # Escape output: "MultiLineString(LineString(Point(1,1),Point(2,2)))"
        # Expect result: 'MULTILINESTRING((1 1, 2 2))'
        ```
        """
        super().__init__("MultiLineString", -1, *ls)


@cython.cclass
class MultiPoint(SQLFunction):
    """Represents the `MultiPoint(pt [, pt] ...)` function.

    MySQL description:
    - Constructs a MultiPoint value using Point or WKB Point arguments.
    """

    def __init__(self, *pt):
        """The `MultiPoint(pt [, pt] ...)` function.

        MySQL description:
        - Constructs a MultiPoint value using Point or WKB Point arguments.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.MultiPoint(sqlfunc.Point(1, 1), sqlfunc.Point(2, 2))
        # Escape output: "MultiPoint(Point(1,1),Point(2,2))"
        # Expect result: 'MULTIPOINT(1 1, 2 2)'
        ```
        """
        super().__init__("MultiPoint", -1, *pt)


@cython.cclass
class MultiPolygon(SQLFunction):
    """Represents the `MultiPolygon(poly [, poly] ...)` function.

    MySQL description:
    - Constructs a MultiPolygon value from a set of Polygon or WKB Polygon arguments.
    """

    def __init__(self, *poly):
        """The `MultiPolygon(poly [, poly] ...)` function.

        MySQL description:
        - Constructs a MultiPolygon value from a set of Polygon or WKB Polygon arguments.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.MultiPolygon(
            sqlfunc.Polygon(
                sqlfunc.LineString(
                    sqlfunc.Point(0, 0),
                    sqlfunc.Point(5, 0),
                    sqlfunc.Point(5, 5),
                    sqlfunc.Point(0, 5),
                    sqlfunc.Point(0, 0)
                )
            )
        )
        # Escape output: "MultiPolygon(Polygon(LineString(Point(0,0),Point(5,0),Point(5,5),Point(0,5),Point(0,0))))"
        # Expect result: 'MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)))'
        ```
        """
        super().__init__("MultiPolygon", -1, *poly)


# Functions: N -------------------------------------------------------------------------------------------------------
@cython.cclass
class NOT_IN(SQLFunction):
    """Represents the `NOT IN (value,...)` function.

    MySQL description:
    - Returns 1 (true) if the proceeding expr is not equal to any
      of the values in the NOT IN() list, else returns 0 (false).
    """

    def __init__(self, *values):
        """The `IN (value,...)` function.

        MySQL description:
        - Returns 1 (true) if the proceeding expr is not equal to any
          of the values in the NOT IN() list, else returns 0 (false).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.NOT_IN(0, 3, 5, 7)
        # Escape output: "NOT IN(0,3,5,7)"

        sqlfunc.NOT_IN(*["a", "b", "c"])
        # Escape output: "NOT IN('a','b','c')"
        """
        super().__init__("NOT IN", -1, *values)


@cython.cclass
class NOW(SQLFunction):
    """Represents the `NOW([fsp])` function.

    MySQL description:
    - Returns the current date and time as a value in 'YYYY-MM-DD hh:mm:ss' or
      YYYYMMDDhhmmss format, depending on whether the function is used in string
      or numeric context. The value is expressed in the session time zone.
    - If the fsp argument is given to specify a fractional seconds precision from
      0 to 6, the return value includes a fractional seconds part of that many digits.
    """

    def __init__(self, fsp: Any | Sentinel = IGNORED):
        """The `NOW([fps])` function.

        MySQL description:
        - Returns the current date and time as a value in 'YYYY-MM-DD hh:mm:ss' or
          YYYYMMDDhhmmss format, depending on whether the function is used in string
          or numeric context. The value is expressed in the session time zone.
        - If the fsp argument is given to specify a fractional seconds precision from
          0 to 6, the return value includes a fractional seconds part of that many digits.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.NOW()
        # Escape output: "NOW()"
        # Expect result: '2021-12-25 19:25:37'

        sqlfunc.NOW(3)
        # Escape output: "NOW(3)"
        # Expect result: '2021-12-25 19:25:37.840'
        ```
        """
        if fsp is IGNORED:
            super().__init__("NOW", 0)
        else:
            super().__init__("NOW", 1, fsp)


@cython.cclass
class NULLIF(SQLFunction):
    """Represents the `NULLIF(expr1, expr2)` function.

    MySQL description:
    - Returns NULL if expr1 = expr2 is true, otherwise returns expr1.
    - This is the same as CASE WHEN expr1 = expr2 THEN NULL ELSE expr1 END.
    """

    def __init__(self, expr1, expr2):
        """The `NULLIF(expr1, expr2)` function.

        MySQL description:
        - Returns NULL if expr1 = expr2 is true, otherwise returns expr1.
        - This is the same as CASE WHEN expr1 = expr2 THEN NULL ELSE expr1 END.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.NULLIF(1, 1)
        # Escape output: "NULLIF(1,1)"
        # Expect result: NULL

        sqlfunc.NULLIF(1, 2)
        # Escape output: "NULLIF(1,2)"
        # Expect result: 1
        """
        super().__init__("NULLIF", 2, expr1, expr2)


# Functions: O -------------------------------------------------------------------------------------------------------
@cython.cclass
class OCT(SQLFunction):
    """Represents the `OCT(N)` function.

    MySQL description:
    - Returns a string representation of the octal value of N, where N
      is a longlong (BIGINT) number. This is equivalent to CONV(N,10,8).
    - Returns NULL if N is NULL.
    """

    def __init__(self, N):
        """The `OCT(N)` function.

        MySQL description:
        - Returns a string representation of the octal value of N, where N
          is a longlong (BIGINT) number. This is equivalent to CONV(N,10,8).
        - Returns NULL if N is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.OCT(12)
        # Escape output: "OCT(12)"
        # Expect result: '14'

        sqlfunc.OCT("10")
        # Escape output: "OCT('10')"
        # Expect result: '12'
        ```
        """
        super().__init__("OCT", 1, N)


@cython.cclass
class ORD(SQLFunction):
    """Represents the `ORD(string)` function.

    MySQL description:
    - If the leftmost character of the string str is a multibyte character,
      returns the code for that character, calculated from the numeric values
      of its constituent bytes.
    - If the leftmost character is not a multibyte character, ORD() returns
      the same value as the ASCII() function. The function returns NULL if str
      is NULL.
    """

    def __init__(self, string):
        """The `ORD(string)` function.

        MySQL description:
        - If the leftmost character of the string str is a multibyte character,
          returns the code for that character, calculated from the numeric values
          of its constituent bytes.
        - If the leftmost character is not a multibyte character, ORD() returns
          the same value as the ASCII() function. The function returns NULL if str
          is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ORD('2')
        # Escape output: "ORD('2')"
        # Expect result: 50
        ```
        """
        super().__init__("ORD", 1, string)


# Functions: P -------------------------------------------------------------------------------------------------------
@cython.cclass
class PERIOD_ADD(SQLFunction):
    """Represents the `PERIOD_ADD(P,N)` function.

    MySQL description:
    - Adds N months to period P (in the format YYMM or YYYYMM).
      Returns a value in the format YYYYMM.
    - This function returns NULL if P or N is NULL.
    """

    def __init__(self, P, N):
        """The `PERIOD_ADD(P,N)` function.

        MySQL description:
        - Adds N months to period P (in the format YYMM or YYYYMM).
          Returns a value in the format YYYYMM.
        - This function returns NULL if P or N is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.PERIOD_ADD(200801, 2)
        # Escape output: "PERIOD_ADD(200801,2)"
        # Expect result: 200803

        sqlfunc.PERIOD_ADD("200801", "2")
        # Escape output: "PERIOD_ADD('200801','2')"
        # Expect result: 200803
        ```
        """
        super().__init__("PERIOD_ADD", 2, P, N)


@cython.cclass
class PERIOD_DIFF(SQLFunction):
    """Represents the `PERIOD_DIFF(P1,P2)` function.

    MySQL description:
    - Returns the number of months between periods P1 and P2.
      P1 and P2 should be in the format YYMM or YYYYMM. Note
      that the period arguments P1 and P2 are not date values.
    - This function returns NULL if P1 or P2 is NULL.
    """

    def __init__(self, P1, P2):
        """The `PERIOD_DIFF(P1,P2)` function.

        MySQL description:
        - Returns the number of months between periods P1 and P2.
          P1 and P2 should be in the format YYMM or YYYYMM. Note
          that the period arguments P1 and P2 are not date values.
        - This function returns NULL if P1 or P2 is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.PERIOD_DIFF(200802, 200703)
        # Escape output: "PERIOD_DIFF(200802,200703)"
        # Expect result: 11

        sqlfunc.PERIOD_DIFF("200802", "200703")
        # Escape output: "PERIOD_DIFF('200802','200703')"
        # Expect result: 1
        ```
        """
        super().__init__("PERIOD_DIFF", 2, P1, P2)


@cython.cclass
class PI(SQLFunction):
    """Represents the `PI()` function.

    MySQL description:
    - Returns the value of π (pi). The default number of decimal places
      displayed is seven, but MySQL uses the full double-precision value
      internally.
    - Because the return value of this function is a double-precision value,
      its exact representation may vary between platforms or implementations.
      This also applies to any expressions making use of PI().
    """

    def __init__(self):
        """The `PI()` function.

        MySQL description:
        - Returns the value of π (pi). The default number of decimal places
          displayed is seven, but MySQL uses the full double-precision value
          internally.
        - Because the return value of this function is a double-precision value,
          its exact representation may vary between platforms or implementations.
          This also applies to any expressions making use of PI().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.PI()
        # Escape output: "PI()"
        # Expect result: 3.141593
        ```
        """
        super().__init__("PI", 0)


@cython.cclass
class Point(SQLFunction):
    """Represents the `Point(x, y)` function.

    MySQL description:
    - Constructs a Point value using the given X and Y coordinates.
    - Returns NULL if any argument is NULL.
    """

    def __init__(self, x, y):
        """The `Point(x, y)` function.

        MySQL description:
        - Constructs a Point value using the given X and Y coordinates.
        - Returns NULL if any argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.Point(1, 1)
        # Escape output: "Point(1,1)"
        # Expect result: 'Point(1 1)'
        ```
        """
        super().__init__("Point", 2, x, y)


@cython.cclass
class Polygon(SQLFunction):
    """Represents the `Polygon(ls [, ls] ...)` function.

    MySQL description:
    - Constructs a Polygon value from a number of LineString or WKB
      LineString arguments. If any argument does not represent a
      LinearRing (that is, not a closed and simple LineString), the
      return value is NULL.
    """

    def __init__(self, *ls):
        """The `Polygon(ls [, ls] ...)` function.

        MySQL description:
        - Constructs a Polygon value from a number of LineString or WKB
          LineString arguments. If any argument does not represent a
          LinearRing (that is, not a closed and simple LineString), the
          return value is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.Polygon(
            sqlfunc.LineString(
                sqlfunc.Point(0, 0),
                sqlfunc.Point(5, 0),
                sqlfunc.Point(5, 5),
                sqlfunc.Point(0, 5),
                sqlfunc.Point(0, 0)
            )
        )
        # Escape output: "Polygon(LineString(Point(0,0),Point(5,0),Point(5,5),Point(0,5),Point(0,0)))"
        # Expect result: 'POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))'
        ```
        """
        super().__init__("Polygon", -1, *ls)


@cython.cclass
class POW(SQLFunction):
    """Represents the `POW(X, Y)` function.

    MySQL description:
    - Returns the value of X raised to the power of Y.
    - Returns NULL if X or Y is NULL.
    """

    def __init__(self, X, Y):
        """The `POW(X, Y)` function.

        MySQL description:
        - Returns the value of X raised to the power of Y.
        - Returns NULL if X or Y is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.POW(2, 2)
        # Escape output: "POW(2,2)"
        # Expect result: 4

        sqlfunc.POW("2", "-2")
        # Escape output: "POW('2','-2')"
        # Expect result: 0.25
        ```
        """
        super().__init__("POW", 2, X, Y)


@cython.cclass
class PS_CURRENT_THREAD_ID(SQLFunction):
    """Represents the `PS_CURRENT_THREAD_ID()` function.

    MySQL description:
    - Returns a BIGINT UNSIGNED value representing the Performance Schema
      thread ID assigned to the current connection.
    - The thread ID return value is a value of the type given in the
      THREAD_ID column of Performance Schema tables.
    - Performance Schema configuration affects PS_CURRENT_THREAD_ID() the
      same way as for PS_THREAD_ID(). For details, see the description of
      that function.
    """

    def __init__(self):
        """The `PS_CURRENT_THREAD_ID()` function.

        MySQL description:
        - Returns a BIGINT UNSIGNED value representing the Performance Schema
          thread ID assigned to the current connection.
        - The thread ID return value is a value of the type given in the
          THREAD_ID column of Performance Schema tables.
        - Performance Schema configuration affects PS_CURRENT_THREAD_ID() the
          same way as for PS_THREAD_ID(). For details, see the description of
          that function.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.PS_CURRENT_THREAD_ID()
        # Escape output: "PS_CURRENT_THREAD_ID()"
        # Expect result: 1
        ```
        """
        super().__init__("PS_CURRENT_THREAD_ID", 0)


@cython.cclass
class PS_THREAD_ID(SQLFunction):
    """Represents the `PS_THREAD_ID(connection_id)` function.

    MySQL description:
    - Given a connection ID, returns a BIGINT UNSIGNED value representing the
      Performance Schema thread ID assigned to the connection ID, or NULL if
      no thread ID exists for the connection ID. The latter can occur for threads
      that are not instrumented, or if connection_id is NULL.
    - The connection ID argument is a value of the type given in the PROCESSLIST_ID
      column of the Performance Schema threads table or the Id column of SHOW PROCESSLIST
      output.
    - The thread ID return value is a value of the type given in the THREAD_ID column
      of Performance Schema tables.
    """

    def __init__(self, connection_id):
        """The `PS_THREAD_ID(connection_id)` function.

        MySQL description:
        - Given a connection ID, returns a BIGINT UNSIGNED value representing the
          Performance Schema thread ID assigned to the connection ID, or NULL if
          no thread ID exists for the connection ID. The latter can occur for threads
          that are not instrumented, or if connection_id is NULL.
        - The connection ID argument is a value of the type given in the PROCESSLIST_ID
          column of the Performance Schema threads table or the Id column of SHOW PROCESSLIST
          output.
        - The thread ID return value is a value of the type given in the THREAD_ID column
          of Performance Schema tables.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.PS_THREAD_ID(sqlfunc.CONNECTION_ID())
        # Escape output: "PS_THREAD_ID(CONNECTION_ID())"
        ```
        """
        super().__init__("PS_THREAD_ID", 1, connection_id)


# Functions: Q -------------------------------------------------------------------------------------------------------
@cython.cclass
class QUARTER(SQLFunction):
    """Represents the `QUARTER(date)` function.

    MySQL description:
    - Returns the quarter of the year for date, in the range 1 to 4.
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `QUARTER(date)` function.

        MySQL description:
        - Returns the quarter of the year for date, in the range 1 to 4.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.QUARTER("2008-04-01")
        # Escape output: "QUARTER('2008-04-01')"
        # Expect result: 2

        sqlfunc.QUARTER(datetime.datetime(2010, 12, 31))
        # Escape output: "QUARTER('2010-12-31 00:00:00')"
        # Expect result: 4
        ```
        """
        super().__init__("QUARTER", 1, date)


@cython.cclass
class QUOTE(SQLFunction):
    """Represents the `QUOTE(string)` function.

    MySQL description:
    - Quotes a string to produce a result that can be used as a properly escaped
      data value in an SQL statement. The string is returned enclosed by single
      quotation marks and with each instance of backslash (\), single quote ('),
      ASCII NUL, and Control+Z preceded by a backslash.
    - If the argument is NULL, the return value is the word “NULL” without enclosing
      single quotation marks.
    """

    def __init__(self, string):
        """The `QUOTE(string)` function.

        MySQL description:
        - Quotes a string to produce a result that can be used as a properly escaped
          data value in an SQL statement. The string is returned enclosed by single
          quotation marks and with each instance of backslash (\), single quote ('),
          ASCII NUL, and Control+Z preceded by a backslash.
        - If the argument is NULL, the return value is the word “NULL” without enclosing
          single quotation marks.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.QUOTE("It's a beautiful day")
        # Escape output: "QUOTE('It\\'s a beautiful day')"
        # Expect result: "'It\\'s a beautiful day'"

        sqlfunc.QUOTE(None)
        # Escape output: "QUOTE(NULL)"
        # Expect result: 'NULL'
        """
        super().__init__("QUOTE", 1, string)


# Functions: R -------------------------------------------------------------------------------------------------------
@cython.cclass
class RADIANS(SQLFunction):
    """Represents the `RADIANS(X)` function.

    MySQL description:
    - Returns the argument X, converted from degrees to radians.
      (Note that π radians equals 180 degrees.)
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `RADIANS(X)` function.

        MySQL description:
        - Returns the argument X, converted from degrees to radians.
          (Note that π radians equals 180 degrees.)
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RADIANS(90)
        # Escape output: "RADIANS(90)"
        # Expect result: 1.5707963267948966

        sqlfunc.RADIANS("180")
        # Escape output: "RADIANS('180')"
        # Expect result: 3.141592653589793
        ```
        """
        super().__init__("RADIANS", 1, X)


@cython.cclass
class RAND(SQLFunction):
    """Represents the `RAND([N])` function.

    MySQL description:
    - Returns a random floating-point value v in the range 0 <= v < 1.0.
    - If an integer argument N is specified, it is used as the seed value.
      With a constant initializer argument, the seed is initialized once
      when the statement is prepared, prior to execution. With a nonconstant
      initializer argument (such as a column name), the seed is initialized
      with the value for each invocation of RAND(). One implication of this
      behavior is that for equal argument values, RAND(N) returns the same
      value each time, and thus produces a repeatable sequence of column values.
    - To obtain a random integer R in the range i <= R < j, use the custom
      SQLFunction 'RANDINT', which is implemented by the following expression:
      FLOOR(i + RAND() * (j - i)).
    """

    def __init__(self, N: Any | Sentinel = IGNORED):
        """The `RAND([N])` function.

        MySQL description:
        - Returns a random floating-point value v in the range 0 <= v < 1.0.
        - If an integer argument N is specified, it is used as the seed value.
          With a constant initializer argument, the seed is initialized once
          when the statement is prepared, prior to execution. With a nonconstant
          initializer argument (such as a column name), the seed is initialized
          with the value for each invocation of RAND(). One implication of this
          behavior is that for equal argument values, RAND(N) returns the same
          value each time, and thus produces a repeatable sequence of column values.
        - To obtain a random integer R in the range i <= R < j, use the custom
          SQLFunction 'RANDINT', which is implemented by the following expression:
          FLOOR(i + RAND() * (j - i)).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RAND()
        # Escape output: "RAND()"
        # Expect result: 0.6794153085718074

        sqlfunc.RAND(1)
        # Escape output: "RAND(1)"
        # Expect result: 0.40540353712197724
        """
        if N is IGNORED:
            super().__init__("RAND", 0)
        else:
            super().__init__("RAND", 1, N)


@cython.cclass
class RANDOM_BYTES(SQLFunction):
    """Represents the `RANDOM_BYTES(length)` function.

    MySQL description:
    - This function returns a binary string of len random bytes generated using
      the random number generator of the SSL library. Permitted values of len
      range from 1 to 1024. For values outside that range, an error occurs.
      Returns NULL if len is NULL.
    - RANDOM_BYTES() can be used to provide the initialization vector for the
      AES_DECRYPT() and AES_ENCRYPT() functions. For use in that context, len
      must be at least 16. Larger values are permitted, but bytes in excess of
      16 are ignored.
    - RANDOM_BYTES() generates a random value, which makes its result nondeterministic.
      Consequently, statements that use this function are unsafe for statement-based
      replication.
    - If RANDOM_BYTES() is invoked from within the mysql client, binary strings display
      using hexadecimal notation, depending on the value of the --binary-as-hex.
    """

    def __init__(self, length):
        """The `RANDOM_BYTES(length)` function.

        MySQL description:
        - This function returns a binary string of len random bytes generated using
          the random number generator of the SSL library. Permitted values of len
          range from 1 to 1024. For values outside that range, an error occurs.
          Returns NULL if len is NULL.
        - RANDOM_BYTES() can be used to provide the initialization vector for the
          AES_DECRYPT() and AES_ENCRYPT() functions. For use in that context, len
          must be at least 16. Larger values are permitted, but bytes in excess of
          16 are ignored.
        - RANDOM_BYTES() generates a random value, which makes its result nondeterministic.
          Consequently, statements that use this function are unsafe for statement-based
          replication.
        - If RANDOM_BYTES() is invoked from within the mysql client, binary strings display
          using hexadecimal notation, depending on the value of the --binary-as-hex.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RANDOM_BYTES(16)
        # Escape output: "RANDOM_BYTES(16)"
        # Expect result: b'\x18\x00\xed}\xcb\xb1\xd6^\xbd\x9egR\x18p\x8f\xb5'
        ```
        """
        super().__init__("RANDOM_BYTES", 1, length)


@cython.cclass
class RANK(SQLFunction):
    """Represents the `RANK()` function.

    MySQL description:
    - Returns the rank of the current row within its partition, with gaps.
      Peers are considered ties and receive the same rank. This function
      does not assign consecutive ranks to peer groups if groups of size
      greater than one exist; the result is noncontiguous rank numbers.
    - This function should be used with ORDER BY to sort partition rows
      into the desired order. Without ORDER BY, all rows are peers.
    """

    def __init__(self):
        """The `RANK()` function.

        MySQL description:
        - Returns the rank of the current row within its partition, with gaps.
          Peers are considered ties and receive the same rank. This function
          does not assign consecutive ranks to peer groups if groups of size
          greater than one exist; the result is noncontiguous rank numbers.
        - This function should be used with ORDER BY to sort partition rows
          into the desired order. Without ORDER BY, all rows are peers.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RANK()
        # Escape output: "RANK()"
        ```
        """
        super().__init__("RANK", 0)


@cython.cclass
class REGEXP_INSTR(SQLFunction):
    """Represents the `REGEXP_INSTR(expr, pat[, pos[, occurrence[, return_option[, match_type]]]])` function.

    MySQL description:
    - Returns the starting index of the substring of the string expr that matches the
      regular expression specified by the pattern pat, 0 if there is no match. If expr
      or pat is NULL, the return value is NULL. Character indexes begin at 1.
    - `pos`: The position in expr at which to start the search. If omitted, the default is 1.
    - `occurrence`: Which occurrence of a match to search for. If omitted, the default is 1.
    = `return_option`: Which type of position to return. If this value is 0, REGEXP_INSTR()
      returns the position of the matched substring's first character. If this value is 1,
      REGEXP_INSTR() returns the position following the matched substring. If omitted, the
      default is 0.
    - `match_type`: A string that specifies how to perform matching. The meaning is as
      described for REGEXP_LIKE().
    """

    def __init__(
        self,
        expr,
        pat,
        pos: int | Sentinel = IGNORED,
        occurrence: int | Sentinel = IGNORED,
        return_option: int | Sentinel = IGNORED,
        match_type: str | Sentinel = IGNORED,
    ):
        """The `REGEXP_INSTR(expr, pat[, pos[, occurrence[, return_option[, match_type]]]])` function.

        MySQL description:
        - Returns the starting index of the substring of the string expr that matches the
          regular expression specified by the pattern pat, 0 if there is no match. If expr
          or pat is NULL, the return value is NULL. Character indexes begin at 1.
        - `pos`: The position in expr at which to start the search. If omitted, the default is 1.
        - `occurrence`: Which occurrence of a match to search for. If omitted, the default is 1.
        = `return_option`: Which type of position to return. If this value is 0, REGEXP_INSTR()
          returns the position of the matched substring's first character. If this value is 1,
          REGEXP_INSTR() returns the position following the matched substring. If omitted, the
          default is 0.
        - `match_type`: A string that specifies how to perform matching. The meaning is as
          described for REGEXP_LIKE().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.REGEXP_INSTR("dog cat dog", "dog")
        # Escape output: "REGEXP_INSTR('dog cat dog','dog')"
        # Expect result: 1

        sqlfunc.REGEXP_INSTR("dog cat dog", "dog", 2)
        # Escape output: "REGEXP_INSTR('dog cat dog','dog',2)"
        # Expect result: 9

        sqlfunc.REGEXP_INSTR("dog cat dog", "dog", 1, 2)
        # Escape output: "REGEXP_INSTR('dog cat dog','dog',1,2)"
        # Expect result: 9

        sqlfunc.REGEXP_INSTR("dog cat dog", "dog", 1, 1, 1)
        # Escape output: "REGEXP_INSTR('dog cat dog','dog',1,1,1)"
        # Expect result: 4

        sqlfunc.REGEXP_INSTR("dog cat dog", "Dog", 1, 1, 0, "i")
        # Escape output: "REGEXP_INSTR('dog cat dog','Dog',1,1,0,'i')"
        # Expect result: 1
        ```
        """
        _fn = "REGEXP_INSTR"
        if pos is IGNORED:
            super().__init__(_fn, 2, expr, pat)
        elif occurrence is IGNORED:
            super().__init__(_fn, 3, expr, pat, pos)
        elif return_option is IGNORED:
            super().__init__(_fn, 4, expr, pat, pos, occurrence)
        elif match_type is IGNORED:
            super().__init__(_fn, 5, expr, pat, pos, occurrence, return_option)
        else:
            super().__init__(
                _fn, 6, expr, pat, pos, occurrence, return_option, match_type
            )


@cython.cclass
class REGEXP_LIKE(SQLFunction):
    """Represents the `REGEXP_LIKE(expr, pat[, match_type])` function.

    MySQL description:
    - Returns 1 if the string expr matches the regular expression specified by the
      pattern pat, 0 otherwise. If expr or pat is NULL, the return value is NULL.
    - The pattern can be an extended regular expression, the syntax for which is
      discussed in Regular Expression Syntax. The pattern need not be a literal
      string. For example, it can be specified as a string expression or table column.
    - The optional match_type argument is a string that may contain any or all the
      following characters specifying how to perform matching: `'c'` Case-sensitive
      matching; `'i'` Case-insensitive matching; `'m'` Multiple-line mode, recognize
      line terminators within the string (The default behavior is to match line
      terminators only at the start and end of the string expression); `'n'` the '.'
      character matches line terminators (The default is for '.' matching to stop at
      the end of a line); `'u'` unix-only line endings, only the newline character is
      recognized as a line ending by the ., ^, and $ match operators.
    - If characters specifying contradictory options are specified within match_type,
      the rightmost one takes precedence.
    """

    def __init__(self, expr, pat, match_type: str | Sentinel = IGNORED):
        """The `REGEXP_LIKE(expr, pat[, match_type])` function.

        MySQL description:
        - Returns 1 if the string expr matches the regular expression specified by the
          pattern pat, 0 otherwise. If expr or pat is NULL, the return value is NULL.
        - The pattern can be an extended regular expression, the syntax for which is
          discussed in Regular Expression Syntax. The pattern need not be a literal
          string. For example, it can be specified as a string expression or table column.
        - The optional match_type argument is a string that may contain any or all the
          following characters specifying how to perform matching: `'c'` Case-sensitive
          matching; `'i'` Case-insensitive matching; `'m'` Multiple-line mode, recognize
          line terminators within the string (The default behavior is to match line
          terminators only at the start and end of the string expression); `'n'` the '.'
          character matches line terminators (The default is for '.' matching to stop at
          the end of a line); `'u'` unix-only line endings, only the newline character is
          recognized as a line ending by the ., ^, and $ match operators.
        - If characters specifying contradictory options are specified within match_type,
          the rightmost one takes precedence.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.REGEXP_LIKE("CamelCase", "CAMELCASE")
        # Escape output: "REGEXP_LIKE('CamelCase','CAMELCASE')"
        # Expect result: 1

        sqlfunc.REGEXP_LIKE("CamelCase", "CAMELCASE", "c")
        # Escape output: "REGEXP_LIKE('CamelCase','CAMELCASE','c')"
        # Expect result: 0

        sqlfunc.REGEXP_LIKE("CamelCase", "CAMELCASE", "imnu")
        # Escape output: "REGEXP_LIKE('CamelCase','CAMELCASE','imnu')"
        # Expect result: 1
        ```
        """
        if match_type is IGNORED:
            super().__init__("REGEXP_LIKE", 2, expr, pat)
        else:
            super().__init__("REGEXP_LIKE", 3, expr, pat, match_type)


@cython.cclass
class REGEXP_REPLACE(SQLFunction):
    """Represents the `REGEXP_REPLACE(expr, pat, repl[, pos[, occurrence[, match_type]]])` function.

    MySQL description:
    - Replaces occurrences in the string expr that match the regular expression
      specified by the pattern pat with the replacement string repl, and returns
      the resulting string. If expr, pat, or repl is NULL, the return value is NULL.
    - `pos`: The position in expr at which to start the search. If omitted, the default is 1.
    - `occurrence`: Which occurrence of a match to replace. If omitted, the default
      is 0 (which means “replace all occurrences”).
    - `match_type`: A string that specifies how to perform matching. The meaning is as
      described for REGEXP_LIKE().
    """

    def __init__(
        self,
        expr,
        pat,
        repl,
        pos: int | Sentinel = IGNORED,
        occurrence: int | Sentinel = IGNORED,
        match_type: str | Sentinel = IGNORED,
    ):
        """The `REGEXP_REPLACE(expr, pat, repl[, pos[, occurrence[, match_type]]])` function.

        MySQL description:
        - Replaces occurrences in the string expr that match the regular expression
          specified by the pattern pat with the replacement string repl, and returns
          the resulting string. If expr, pat, or repl is NULL, the return value is NULL.
        - `pos`: The position in expr at which to start the search. If omitted, the default is 1.
        - `occurrence`: Which occurrence of a match to replace. If omitted, the default
          is 0 (which means “replace all occurrences”).
        - `match_type`: A string that specifies how to perform matching. The meaning is as
          described for REGEXP_LIKE().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.REGEXP_REPLACE("a b c", "b", "X")
        # Escape output: "REGEXP_REPLACE('a b c','b','X')"
        # Expect result: 'a X c'

        sqlfunc.REGEXP_REPLACE("a b b c", "b", "X", 4)
        # Escape output: "REGEXP_REPLACE('a b b c','b','X',4)"
        # Expect result: 'a b X c'

        sqlfunc.REGEXP_REPLACE("a b b b c", "b", "X", 1, 2)
        # Escape output: "REGEXP_REPLACE('a b b b c','b','X',1,2)"
        # Expect result: 'a b X b c'

        sqlfunc.REGEXP_REPLACE("a b b b c", "B", "X", 1, 2, "i")
        # Escape output: "REGEXP_REPLACE('a b b b c','B','X',1,2,'i')"
        # Expect result: 'a b X b c'
        ```
        """
        _fn = "REGEXP_REPLACE"
        if pos is IGNORED:
            super().__init__(_fn, 3, expr, pat, repl)
        elif occurrence is IGNORED:
            super().__init__(_fn, 4, expr, pat, repl, pos)
        elif match_type is IGNORED:
            super().__init__(_fn, 5, expr, pat, repl, pos, occurrence)
        else:
            super().__init__(_fn, 6, expr, pat, repl, pos, occurrence, match_type)


@cython.cclass
class REGEXP_SUBSTR(SQLFunction):
    """Represents the `REGEXP_SUBSTR(expr, pat[, pos[, occurrence[, match_type]]])` function.

    MySQL description:
    - Returns the substring of the string expr that matches the regular expression
      specified by the pattern pat, NULL if there is no match. If expr or pat is
      NULL, the return value is NULL.
    - `pos`: The position in expr at which to start the search. If omitted, the default is 1.
    - `occurrence`: Which occurrence of a match to search for. If omitted, the default is 1.
    - `match_type`: A string that specifies how to perform matching. The meaning is
      as described for REGEXP_LIKE().
    - The result returned by this function uses the character set and collation of the
      expression searched for matches.
    """

    def __init__(
        self,
        expr,
        pat,
        pos: int | Sentinel = IGNORED,
        occurrence: int | Sentinel = IGNORED,
        match_type: str | Sentinel = IGNORED,
    ):
        """The `REGEXP_SUBSTR(expr, pat[, pos[, occurrence[, match_type]]])` function.

        MySQL description:
        - Returns the substring of the string expr that matches the regular expression
          specified by the pattern pat, NULL if there is no match. If expr or pat is
          NULL, the return value is NULL.
        - `pos`: The position in expr at which to start the search. If omitted, the default is 1.
        - `occurrence`: Which occurrence of a match to search for. If omitted, the default is 1.
        - `match_type`: A string that specifies how to perform matching. The meaning is
          as described for REGEXP_LIKE().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.REGEXP_SUBSTR("abc def ghi", "[a-z]+")
        # Escape output: "REGEXP_SUBSTR('abc def ghi','[a-z]+')"
        # Expect result: 'abc'

        sqlfunc.REGEXP_SUBSTR("abc def ghi", "[a-z]+", 4)
        # Escape output: "REGEXP_SUBSTR('abc def ghi','[a-z]+',4)"
        # Expect result: 'def'

        sqlfunc.REGEXP_SUBSTR("abc def ghi", "[a-z]+", 1, 2)
        # Escape output: "REGEXP_SUBSTR('abc def ghi','[a-z]+',1,2)"
        # Expect result: 'def'

        sqlfunc.REGEXP_SUBSTR("abc def ghi", "[A-Z]+", 1, 2, "i")
        # Escape output: "REGEXP_SUBSTR('abc def ghi','[A-Z]+',1,2,'i')"
        # Expect result: 'def'
        ```
        """
        _fn = "REGEXP_SUBSTR"
        if pos is IGNORED:
            super().__init__(_fn, 2, expr, pat)
        elif occurrence is IGNORED:
            super().__init__(_fn, 3, expr, pat, pos)
        elif match_type is IGNORED:
            super().__init__(_fn, 4, expr, pat, pos, occurrence)
        else:
            super().__init__(_fn, 5, expr, pat, pos, occurrence, match_type)


@cython.cclass
class RELEASE_ALL_LOCKS(SQLFunction):
    """Represents the `RELEASE_ALL_LOCKS()` function.

    MySQL description:
    - Releases all named locks held by the current session and
      returns the number of locks released (0 if there were none)
    - This function is unsafe for statement-based replication. A warning is
      logged if you use this function when binlog_format is set to STATEMENT.
    """

    def __init__(self):
        """The `RELEASE_ALL_LOCKS()` function.

        MySQL description:
        - Releases all named locks held by the current session and
          returns the number of locks released (0 if there were none)
        - This function is unsafe for statement-based replication. A warning is
          logged if you use this function when binlog_format is set to STATEMENT.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RELEASE_ALL_LOCKS()
        # Escape output: "RELEASE_ALL_LOCKS()"
        ```
        """
        super().__init__("RELEASE_ALL_LOCKS", 0)


@cython.cclass
class RELEASE_LOCK(SQLFunction):
    """Represents the `RELEASE_LOCK(string)` function.

    MySQL description:
    - Releases the lock named by the string str that was obtained with GET_LOCK().
      Returns 1 if the lock was released, 0 if the lock was not established by this
      thread (in which case the lock is not released), and NULL if the named lock
      did not exist. The lock does not exist if it was never obtained by a call to
      GET_LOCK() or if it has previously been released.
    """

    def __init__(self, string):
        """The `RELEASE_LOCK(string)` function.

        MySQL description:
        - Releases the lock named by the string str that was obtained with GET_LOCK().
          Returns 1 if the lock was released, 0 if the lock was not established by this
          thread (in which case the lock is not released), and NULL if the named lock
          did not exist. The lock does not exist if it was never obtained by a call to
          GET_LOCK() or if it has previously been released.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RELEASE_LOCK("lock1")
        # Escape output: "RELEASE_LOCK('lock1')"
        ```
        """
        super().__init__("RELEASE_LOCK", 1, string)


@cython.cclass
class REPEAT(SQLFunction):
    """Represents the `REPEAT(string, count)` function.

    MySQL description:
    - Returns a string consisting of the string str repeated count times.
      If count is less than 1, returns an empty string.
    - Returns NULL if str or count is NULL.
    """

    def __init__(self, string, count):
        """The `REPEAT(string, count)` function.

        MySQL description:
        - Returns a string consisting of the string str repeated count times.
          If count is less than 1, returns an empty string.
        - Returns NULL if str or count is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.REPEAT("MySQL", 3)
        # Escape output: "REPEAT('MySQL',3)"
        # Expect result: 'MySQLMySQLMySQL'

        sqlfunc.REPEAT("MySQL", 0)
        # Escape output: "REPEAT('MySQL',0)"
        # Expect result: ''
        ```
        """
        super().__init__("REPEAT", 2, string, count)


@cython.cclass
class REPLACE(SQLFunction):
    """Represents the `REPLACE(string,from_str,to_str)` function.

    MySQL description:
    - Returns the string str with all occurrences of the string
      from_str replaced by the string to_str. REPLACE() performs
      a case-sensitive match when searching for from_str.
    - This function is multibyte safe. It returns NULL if any of
      its arguments are NULL.
    """

    def __init__(self, string, from_str, to_str):
        """The `REPLACE(string,from_str,to_str)` function.

        MySQL description:
        - Returns the string str with all occurrences of the string
          from_str replaced by the string to_str. REPLACE() performs
          a case-sensitive match when searching for from_str.
        - This function is multibyte safe. It returns NULL if any of
          its arguments are NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.REPLACE("www.mysql.com", "w", "Ww")
        # Escape output: "REPLACE('www.mysql.com','w','Ww')"
        # Expect result: 'WwWwWw.mysql.com'
        ```
        """
        super().__init__("REPLACE", 3, string, from_str, to_str)


@cython.cclass
class REVERSE(SQLFunction):
    """Represents the `REVERSE(string)` function.

    MySQL description:
    - Returns the string str with the order of the characters reversed.
    - Returns NULL if str is NULL.
    """

    def __init__(self, string):
        """The `REVERSE(string)` function.

        MySQL description:
        - Returns the string str with the order of the characters reversed.
        - Returns NULL if str is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.REVERSE("abc")
        # Escape output: "REVERSE('abc')"
        # Expect result: 'cba'
        ```
        """
        super().__init__("REVERSE", 1, string)


@cython.cclass
class RIGHT(SQLFunction):
    """Represents the `RIGHT(string, length)` function.

    MySQL description:
    - Returns the rightmost len characters from the string str.
    - Returns NULL if any argument is NULL.
    """

    def __init__(self, string, length):
        """The `RIGHT(string, length)` function.

        MySQL description:
        - Returns the rightmost len characters from the string str.
        - Returns NULL if any argument is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RIGHT("foobarbar", 4)
        # Escape output: "RIGHT('foobarbar',4)"
        # Expect result: 'rbar'
        ```
        """
        super().__init__("RIGHT", 2, string, length)


@cython.cclass
class ROLES_GRAPHML(SQLFunction):
    """Represents the `ROLES_GRAPHML()` function.

    MySQL description:
    - Returns a utf8mb3 string containing a GraphML document representing
      memory role subgraphs. The ROLE_ADMIN privilege (or the deprecated
      SUPER privilege) is required to see content in the <graphml> element.
    - Otherwise, the result shows only an empty element
    """

    def __init__(self):
        """The `ROLES_GRAPHML()` function.

        MySQL description:
        - Returns a utf8mb3 string containing a GraphML document representing
          memory role subgraphs. The ROLE_ADMIN privilege (or the deprecated
          SUPER privilege) is required to see content in the <graphml> element.
        - Otherwise, the result shows only an empty element

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ROLES_GRAPHML()
        # Escape output: "ROLES_GRAPHML()"
        # Expect result: '<?xml version="1.0" encoding="UTF-8"?><graphml />'
        ```
        """
        super().__init__("ROLES_GRAPHML", 0)


@cython.cclass
class ROUND(SQLFunction):
    """Represents the `ROUND(X [, D])` function.

    MySQL description:
    - Rounds the argument X to D decimal places. The rounding algorithm depends
      on the data type of X. D defaults to 0 if not specified. D can be negative
      to cause D digits left of the decimal point of the value X to become zero.
      The maximum absolute value for D is 30; any digits in excess of 30 (or -30)
      are truncated.
    - If X or D is NULL, the function returns NULL.
    """

    def __init__(self, X, D: Any | Sentinel = IGNORED):
        """The `ROUND(X [, D])` function.

        MySQL description:
        - Rounds the argument X to D decimal places. The rounding algorithm depends
          on the data type of X. D defaults to 0 if not specified. D can be negative
          to cause D digits left of the decimal point of the value X to become zero.
          The maximum absolute value for D is 30; any digits in excess of 30 (or -30)
          are truncated.
        - If X or D is NULL, the function returns NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ROUND(-1.23)
        # Escape output: "ROUND(-1.23)"
        # Expect result: -1

        sqlfunc.ROUND(1.298, 1)
        # Escape output: "ROUND(1.298,1)"
        # Expect result: 1.3
        ```
        """
        if D is IGNORED:
            super().__init__("ROUND", 1, X)
        else:
            super().__init__("ROUND", 2, X, D)


@cython.cclass
class ROW_COUNT(SQLFunction):
    """Represents the `ROW_COUNT()` function.

    MySQL description:
    - DDL statements: 0. This applies to statements such as CREATE TABLE or DROP TABLE.
    - DML statements other than SELECT: The number of affected rows. This applies to
      statements such as UPDATE, INSERT, or DELETE (as before), but now also to statements
      such as ALTER TABLE and LOAD DATA.
    - SELECT: -1 if the statement returns a result set, or the number of rows “affected”
      if it does not. For example, for SELECT * FROM t1, ROW_COUNT() returns -1. For
      SELECT * FROM t1 INTO OUTFILE 'file_name', ROW_COUNT() returns the number of rows
      written to the file.
    - SIGNAL statements: 0.
    """

    def __init__(self):
        """The `ROW_COUNT()` function.

        MySQL description:
        - DDL statements: 0. This applies to statements such as CREATE TABLE or DROP TABLE.
        - DML statements other than SELECT: The number of affected rows. This applies to
          statements such as UPDATE, INSERT, or DELETE (as before), but now also to statements
          such as ALTER TABLE and LOAD DATA.
        - SELECT: -1 if the statement returns a result set, or the number of rows “affected”
          if it does not. For example, for SELECT * FROM t1, ROW_COUNT() returns -1. For
          SELECT * FROM t1 INTO OUTFILE 'file_name', ROW_COUNT() returns the number of rows
          written to the file.
        - SIGNAL statements: 0.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ROW_COUNT()
        # Escape output: "ROW_COUNT()"
        ```
        """
        super().__init__("ROW_COUNT", 0)


@cython.cclass
class ROW_NUMBER(SQLFunction):
    """Represents the `ROW_NUMBER()` function.

    MySQL description:
    - Returns the number of the current row within its partition.
      Rows numbers range from 1 to the number of partition rows.
    - ORDER BY affects the order in which rows are numbered. Without
      ORDER BY, row numbering is nondeterministic.
    - ROW_NUMBER() assigns peers different row numbers. To assign peers
      the same value, use RANK() or DENSE_RANK(). For an example, see
      the RANK() function description.
    """

    def __init__(self):
        """The `ROW_NUMBER()` function.

        MySQL description:
        - Returns the number of the current row within its partition.
          Rows numbers range from 1 to the number of partition rows.
        - ORDER BY affects the order in which rows are numbered. Without
          ORDER BY, row numbering is nondeterministic.
        - ROW_NUMBER() assigns peers different row numbers. To assign peers
          the same value, use RANK() or DENSE_RANK(). For an example, see
          the RANK() function description.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ROW_NUMBER()
        # Escape output: "ROW_NUMBER()"
        ```
        """
        super().__init__("ROW_NUMBER", 0)


@cython.cclass
class RPAD(SQLFunction):
    """Represents the `RPAD(string, length, padstr)` function.

    MySQL description:
    - Returns the string str, right-padded with the string padstr to a length of len
      characters. If str is longer than len, the return value is shortened to len
      characters.
    - If str, padstr, or len is NULL, the function returns NULL.
    """

    def __init__(self, string, length, padstr):
        """The `RPAD(string, length, padstr)` function.

        MySQL description:
        - Returns the string str, right-padded with the string padstr to a length of len
          characters. If str is longer than len, the return value is shortened to len
          characters.
        - If str, padstr, or len is NULL, the function returns NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RPAD("hi", 5, "?")
        # Escape output: "RPAD('hi',5,'?')"
        # Expect result: 'hi???'

        sqlfunc.RPAD("hi", 1, "?")
        # Escape output: "RPAD('hi',1,'?')"
        # Expect result: 'h'
        ```
        """
        super().__init__("RPAD", 3, string, length, padstr)


@cython.cclass
class RTRIM(SQLFunction):
    """Represents the `RTRIM(string)` function.

    MySQL description:
    - Returns the string str with trailing space characters removed.
    - Returns NULL if str is NULL.
    """

    def __init__(self, string):
        """The `RTRIM(string)` function.

        MySQL description:
        - Returns the string str with trailing space characters removed.
        - Returns NULL if str is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.RTRIM("barbar   ")
        # Escape output: "RTRIM('barbar   ')"
        # Expect result: 'barbar'
        ```
        """
        super().__init__("RTRIM", 1, string)


# Functions: S -------------------------------------------------------------------------------------------------------
@cython.cclass
class SEC_TO_TIME(SQLFunction):
    """Represents the `SEC_TO_TIME(seconds)` function.

    MySQL description:
    - Returns the seconds argument, converted to hours, minutes, and seconds,
      as a TIME value. The range of the result is constrained to that of the
      TIME data type. A warning occurs if the argument corresponds to a value
      outside that range.
    - The function returns NULL if seconds is NULL.
    """

    def __init__(self, seconds):
        """The `SEC_TO_TIME(seconds)` function.

        MySQL description:
        - Returns the seconds argument, converted to hours, minutes, and seconds,
          as a TIME value. The range of the result is constrained to that of the
          TIME data type. A warning occurs if the argument corresponds to a value
          outside that range.
        - The function returns NULL if seconds is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SEC_TO_TIME(2378)
        # Escape output: "SEC_TO_TIME(2378)"
        # Expect result: '00:39:38'

        sqlfunc.SEC_TO_TIME("2378")
        # Escape output: "SEC_TO_TIME('2378')"
        # Expect result: '00:39:38'
        ```
        """
        super().__init__("SEC_TO_TIME", 1, seconds)


@cython.cclass
class SECOND(SQLFunction):
    """Represents the `SECOND(time)` function.

    MySQL description:
    - Returns the second for time, in the range 0 to 59.
    - Returns NULL if time is NULL.
    """

    def __init__(self, time):
        """The `SECOND(time)` function.

        MySQL description:
        - Returns the second for time, in the range 0 to 59.
        - Returns NULL if time is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.SECOND("10:05:03")
        # Escape output: "SECOND('10:05:03')"
        # Expect result: 3

        sqlfunc.SECOND(datetime.datetime(2021, 12, 25, 19, 25, 37))
        # Escape output: "SECOND('2021-12-25 19:25:37')"
        # Expect result: 37
        ```
        """
        super().__init__("SECOND", 1, time)


@cython.cclass
class SHA1(SQLFunction):
    """Represents the `SHA1(string)` function.

    MySQL description:
    - Calculates an SHA-1 160-bit checksum for the string, as described in
      RFC 3174 (Secure Hash Algorithm). The value is returned as a string
      of 40 hexadecimal digits, or NULL if the argument is NULL. One of the
      possible uses for this function is as a hash key.
    - SHA() is synonymous with SHA1().
    """

    def __init__(self, string):
        """The `SHA1(string)` function.

        MySQL description:
        - Calculates an SHA-1 160-bit checksum for the string, as described in RFC 3174.
        - Returns NULL if the argument was NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SHA1("abc")
        # Escape output: "SHA1('abc')"
        # Expect result: 'a9993e364706816aba3e25717850c26c9cd0d89d'
        ```
        """
        super().__init__("SHA1", 1, string)


@cython.cclass
class SHA2(SQLFunction):
    """Represents the `SHA2(string, hash_length)` function.

    MySQL description:
    - Calculates the SHA-2 family of hash functions (SHA-224, SHA-256, SHA-384,
      and SHA-512). The first argument is the plaintext string to be hashed.
      The second argument indicates the desired bit length of the result, which
      must have a value of 224, 256, 384, 512, or 0 (which is equivalent to 256).
    - If either argument is NULL or the hash length is not one of the permitted
      values, the return value is NULL. Otherwise, the function result is a hash
      value containing the desired number of bits. See the notes at the beginning
      of this section about storing hash values efficiently.
    """

    def __init__(self, string, hash_length: cython.int):
        """The `SHA2(string, hash_length)` function.

        MySQL description:
        - Calculates the SHA-2 family of hash functions (SHA-224, SHA-256, SHA-384,
          and SHA-512). The first argument is the plaintext string to be hashed.
          The second argument indicates the desired bit length of the result, which
          must have a value of 224, 256, 384, 512, or 0 (which is equivalent to 256).
        - If either argument is NULL or the hash length is not one of the permitted
          values, the return value is NULL. Otherwise, the function result is a hash
          value containing the desired number of bits. See the notes at the beginning
          of this section about storing hash values efficiently.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SHA2("abc", 224)
        # Escape output: "SHA2('abc',224)"
        # Expect result: '23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7'

        sqlfunc.SHA2("abc", 256)
        # Escape output: "SHA2('abc',256)"
        # Expect result: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
        ```
        """
        if hash_length not in (224, 256, 384, 512, 0):
            raise errors.SQLFunctionError(
                "SQL function SHA2 argument 'hash_length' must be one of %s, "
                "instead got '%d'." % hash_length
            )
        super().__init__("SHA2", 2, string, hash_length)


@cython.cclass
class SIGN(SQLFunction):
    """Represents the `SIGN(X)` function.

    MySQL description:
    - Returns the sign of the argument as -1, 0, or 1, depending
      on whether X is negative, zero, or positive.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `SIGN(X)` function.

        MySQL description:
        - Returns the sign of the argument as -1, 0, or 1, depending
          on whether X is negative, zero, or positive.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SIGN(-32)
        # Escape output: "SIGN(-32)"
        # Expect result: -1

        sqlfunc.SIGN("0")
        # Escape output: "SIGN('0')"
        # Expect result: 0

        sqlfunc.SIGN(234)
        # Escape output: "SIGN(234)"
        # Expect result: 1
        ```
        """
        super().__init__("SIGN", 1, X)


@cython.cclass
class SIN(SQLFunction):
    """Represents the `SIN(X)` function.

    MySQL description:
    - Returns the sine of X, where X is given in radians.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `SIN(X)` function.

        MySQL description:
        - Returns the sine of X, where X is given in radians.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SIN(1)
        # Escape output: "SIN(1)"
        # Expect result: 0.8414709848078965

        sqlfunc.SIN(sqlfunc.PI())
        # Escape output: "SIN(PI())"
        # Expect result: 1.2246467991473532e-16
        ```
        """
        super().__init__("SIN", 1, X)


@cython.cclass
class SLEEP(SQLFunction):
    """Represents the `SLEEP(duration)` function.

    MySQL description:
    - Sleeps (pauses) for the number of seconds given by the duration
      argument, then returns 0. The duration may have a fractional part.
      If the argument is NULL or negative, SLEEP() produces a warning,
      or an error in strict SQL mode.
    - When sleep returns normally (without interruption), it returns 0.
    - When SLEEP() is the only thing invoked by a query that is interrupted,
      it returns 1 and the query itself returns no error. This is true whether
      the query is killed or times out.
    - When SLEEP() is only part of a query that is interrupted, the query
      returns an error.
    """

    def __init__(self, duration):
        """The `SLEEP(duration)` function.

        MySQL description:
        - Sleeps (pauses) for the number of seconds given by the duration
          argument, then returns 0. The duration may have a fractional part.
          If the argument is NULL or negative, SLEEP() produces a warning,
          or an error in strict SQL mode.
        - When sleep returns normally (without interruption), it returns 0.
        - When SLEEP() is the only thing invoked by a query that is interrupted,
          it returns 1 and the query itself returns no error. This is true whether
          the query is killed or times out.
        - When SLEEP() is only part of a query that is interrupted, the query
          returns an error.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SLEEP(5)
        # Escape output: "SLEEP(5)"
        ```
        """
        super().__init__("SLEEP", 1, duration)


@cython.cclass
class SOUNDEX(SQLFunction):
    """Represents the `SOUNDEX(string)` function.

    MySQL description:
    - Returns a soundex string from str, or NULL if str is NULL. Two strings that
      sound almost the same should have identical soundex strings. A standard
      soundex string is four characters long, but the SOUNDEX() function returns
      an arbitrarily long string. You can use SUBSTRING() on the result to get a
      standard soundex string.
    - All nonalphabetic characters in str are ignored.
    - All international alphabetic characters outside the A-Z range are treated
      as vowels.
    """

    def __init__(self, string):
        """The `SOUNDEX(string)` function.

        MySQL description:
        - Returns a soundex string from str, or NULL if str is NULL. Two strings that
          sound almost the same should have identical soundex strings. A standard
          soundex string is four characters long, but the SOUNDEX() function returns
          an arbitrarily long string. You can use SUBSTRING() on the result to get a
          standard soundex string.
        - All nonalphabetic characters in str are ignored.
        - All international alphabetic characters outside the A-Z range are treated
          as vowels.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SOUNDEX("Hello")
        # Escape output: "SOUNDEX('Hello')"
        # Expect result: 'H400'

        sqlfunc.SOUNDEX("Quadratically")
        # Escape output: "SOUNDEX('Quadratically')"
        # Expect result: 'Q36324'
        ```
        """
        super().__init__("SOUNDEX", 1, string)


@cython.cclass
class SPACE(SQLFunction):
    """Represents the `SPACE(N)` function.

    MySQL description:
    - Returns a string consisting of N space characters.
    - Returns NULL if N is NULL.
    """

    def __init__(self, N):
        """The `SPACE(N)` function.

        MySQL description:
        - Returns a string consisting of N space characters.
        - Returns NULL if N is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SPACE(6)
        # Escape output: "SPACE(6)"
        # Expect result: '      '
        ```
        """
        super().__init__("SPACE", 1, N)


@cython.cclass
class SQRT(SQLFunction):
    """Represents the `SQRT(X)` function.

    MySQL description:
    - Returns the square root of a nonnegative number X.
    - If X is NULL, the function returns NULL.
    """

    def __init__(self, X):
        """The `SQRT(X)` function.

        MySQL description:
        - Returns the square root of a nonnegative number X.
        - If X is NULL, the function returns NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SQRT(4)
        # Escape output: "SQRT(4)"
        # Expect result: 2

        sqlfunc.SQRT("20")
        # Escape output: "SQRT('20')"
        # Expect result: 4.47213595499958
        ```
        """
        super().__init__("SQRT", 1, X)


@cython.cclass
class ST_Area(SQLFunction):
    """Represents the `ST_Area({poly|mpoly}) function.

    MySQL description:
    - Returns a double-precision number indicating the area of the
      Polygon or MultiPolygon argument, as measured in its spatial
      reference system.
    - If the geometry is geometrically invalid, either the result is
      an undefined area (that is, it can be any number), or an error
      occurs.
    - If the geometry is valid but is not a Polygon or MultiPolygon
      object, an ER_UNEXPECTED_GEOMETRY_TYPE error occurs.
    - If the geometry is a valid Polygon in a Cartesian SRS, the result
      is the Cartesian area of the polygon.
    - If the geometry is a valid MultiPolygon in a Cartesian SRS, the
      result is the sum of the Cartesian area of the polygons.
    - If the geometry is a valid Polygon in a geographic SRS, the result
      is the geodetic area of the polygon in that SRS, in square meters.
    - If the geometry is a valid MultiPolygon in a geographic SRS, the
      result is the sum of geodetic area of the polygons in that SRS,
      in square meters.
    - If an area computation results in +inf, an ER_DATA_OUT_OF_RANGE error occurs.
    - If the geometry has a geographic SRS with a longitude or latitude
      that is out of range, an error occurs:
    - If a longitude value is not in the range (-180, 180], an
      ER_GEOMETRY_PARAM_LONGITUDE_OUT_OF_RANGE error occurs.
    - If a latitude value is not in the range [-90, 90], an
      ER_GEOMETRY_PARAM_LATITUDE_OUT_OF_RANGE error occurs.
    - Ranges shown are in degrees. The exact range limits deviate
      slightly due to floating-point arithmetic.
    """

    def __init__(self, poly):
        """The `ST_Area({poly|mpoly})` function.

        MySQL description:
        - Returns a double-precision number indicating the area of the
          Polygon or MultiPolygon argument, as measured in its spatial
          reference system.
        - If the geometry is geometrically invalid, either the result is
          an undefined area (that is, it can be any number), or an error
          occurs.
        - If the geometry is valid but is not a Polygon or MultiPolygon
          object, an ER_UNEXPECTED_GEOMETRY_TYPE error occurs.
        - If the geometry is a valid Polygon in a Cartesian SRS, the result
          is the Cartesian area of the polygon.
        - If the geometry is a valid MultiPolygon in a Cartesian SRS, the
          result is the sum of the Cartesian area of the polygons.
        - If the geometry is a valid Polygon in a geographic SRS, the result
          is the geodetic area of the polygon in that SRS, in square meters.
        - If the geometry is a valid MultiPolygon in a geographic SRS, the
          result is the sum of geodetic area of the polygons in that SRS,
          in square meters.
        - If an area computation results in +inf, an ER_DATA_OUT_OF_RANGE error occurs.
        - If the geometry has a geographic SRS with a longitude or latitude
          that is out of range, an error occurs:
        - If a longitude value is not in the range (-180, 180], an
          ER_GEOMETRY_PARAM_LONGITUDE_OUT_OF_RANGE error occurs.
        - If a latitude value is not in the range [-90, 90], an
          ER_GEOMETRY_PARAM_LATITUDE_OUT_OF_RANGE error occurs.
        - Ranges shown are in degrees. The exact range limits deviate
          slightly due to floating-point arithmetic.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "Polygon((0 0,0 3,3 0,0 0),(1 1,1 2,2 1,1 1))"
        sqlfunc.ST_Area(sqlfunc.ST_GeomFromText(poly))
        # Escape output: "ST_Area(ST_GeomFromText('Polygon((0 0,0 3,3 0,0 0),(1 1,1 2,2 1,1 1))'))"
        # Expect result: 4

        multipoly = "MultiPolygon(((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1)))"
        sqlfunc.ST_Area(sqlfunc.ST_GeomFromText(multipoly))
        # Escape output: "ST_Area(ST_GeomFromText('MultiPolygon(((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1)))'))"
        # Expect result: 8
        ```
        """
        super().__init__("ST_Area", 1, poly)


@cython.cclass
class ST_AsBinary(SQLFunction):
    """Represents the `ST_AsBinary(g [, options]), ST_AsWKB(g [, options])` function.

    MySQL description:
    - Converts a value in internal geometry format to its WKB
      representation and returns the binary result.
    - The function return value has geographic coordinates (latitude,
      longitude) in the order specified by the spatial reference system
      that applies to the geometry argument. An optional options argument
      may be given to override the default axis order.
    """

    def __init__(self, g, options: Any | Sentinel = IGNORED):
        """The `ST_AsBinary(g [, options]), ST_AsWKB(g [, options])` function.

        MySQL description:
        - Converts a value in internal geometry format to its WKB
          representation and returns the binary result.
        - The function return value has geographic coordinates (latitude,
          longitude) in the order specified by the spatial reference system
          that applies to the geometry argument. An optional options argument
          may be given to override the default axis order.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(0 5,5 10,10 15)"

        sqlfunc.ST_AsBinary(sqlfunc.ST_GeomFromText(ls))
        # Escape output: "ST_AsBinary(ST_GeomFromText('LineString(0 5,5 10,10 15)'))"

        sqlfunc.ST_AsBinary(sqlfunc.ST_GeomFromText(ls), "axis-order=long-lat")
        # Escape output: "ST_AsBinary(ST_GeomFromText('LineString(0 5,5 10,10 15)'),'axis-order=long-lat')"
        ```
        """
        if options is IGNORED:
            super().__init__("ST_AsBinary", 1, g)
        else:
            super().__init__("ST_AsBinary", 2, g, options)


@cython.cclass
class ST_AsWKB(SQLFunction):
    """Represents the `ST_AsWKB(g [, options]), ST_AsWKB(g [, options])` function.

    MySQL description:
    - Converts a value in internal geometry format to its WKB
      representation and returns the binary result.
    - The function return value has geographic coordinates (latitude,
      longitude) in the order specified by the spatial reference system
      that applies to the geometry argument. An optional options argument
      may be given to override the default axis order.
    """

    def __init__(self, g, options: Any | Sentinel = IGNORED):
        """The `ST_AsWKB(g [, options]), ST_AsWKB(g [, options])` function.

        MySQL description:
        - Converts a value in internal geometry format to its WKB
          representation and returns the binary result.
        - The function return value has geographic coordinates (latitude,
          longitude) in the order specified by the spatial reference system
          that applies to the geometry argument. An optional options argument
          may be given to override the default axis order.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(0 5,5 10,10 15)"

        sqlfunc.ST_AsWKB(sqlfunc.ST_GeomFromText(ls))
        # Escape output: "ST_AsWKB(ST_GeomFromText('LineString(0 5,5 10,10 15)'))"

        sqlfunc.ST_AST_AsWKBsBinary(sqlfunc.ST_GeomFromText(ls), "axis-order=long-lat")
        # Escape output: "ST_AsWKB(ST_GeomFromText('LineString(0 5,5 10,10 15)'),'axis-order=long-lat')"
        ```
        """
        if options is IGNORED:
            super().__init__("ST_AsWKB", 1, g)
        else:
            super().__init__("ST_AsWKB", 2, g, options)


@cython.cclass
class ST_AsGeoJSON(SQLFunction):
    """Represents the `ST_AsGeoJSON(g [, max_dec_digits [, options]])` function.

    MySQL description:
    - Generates a GeoJSON object from the geometry g. The object string
      has the connection character set and collation.
    - If any argument is NULL, the return value is NULL. If any non-NULL
      argument is invalid, an error occurs.
    - max_dec_digits, if specified, limits the number of decimal digits
      for coordinates and causes rounding of output. If not specified, this
      argument defaults to its maximum value of 232 - 1. The minimum is 0.
    - options, if specified, is a bitmask. The following table shows the
      permitted flag values. If the geometry argument has an SRID of 0, no
      CRS object is produced even for those flag values that request one.
    """

    def __init__(
        self,
        g,
        max_dec_digits: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_AsGeoJSON(g [, max_dec_digits [, options]])` function.

        MySQL description:
        - Generates a GeoJSON object from the geometry g. The object string
          has the connection character set and collation.
        - If any argument is NULL, the return value is NULL. If any non-NULL
          argument is invalid, an error occurs.
        - max_dec_digits, if specified, limits the number of decimal digits
          for coordinates and causes rounding of output. If not specified, this
          argument defaults to its maximum value of 232 - 1. The minimum is 0.
        - options, if specified, is a bitmask. The following table shows the
          permitted flag values. If the geometry argument has an SRID of 0, no
          CRS object is produced even for those flag values that request one.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p = "POINT(11.11111 12.22222)"

        sqlfunc.ST_AsGeoJSON(sqlfunc.ST_GeomFromText(p))
        # Escape output: "ST_AsGeoJSON(ST_GeomFromText('POINT(11.11111 12.22222)'))"
        # Expect result: '{"type": "Point", "coordinates": [11.11111, 12.22222]}'

        sqlfunc.ST_AsGeoJSON(sqlfunc.ST_GeomFromText(p), 2)
        # Escape output: "ST_AsGeoJSON(ST_GeomFromText('POINT(11.11111 12.22222)'),2)"
        # Expect result: '{"type": "Point", "coordinates": [11.11, 12.22]}'
        ```
        """
        if max_dec_digits is IGNORED:
            super().__init__("ST_AsGeoJSON", 1, g)
        elif options is IGNORED:
            super().__init__("ST_AsGeoJSON", 2, g, max_dec_digits)
        else:
            super().__init__("ST_AsGeoJSON", 3, g, max_dec_digits, options)


@cython.cclass
class ST_AsText(SQLFunction):
    """Represents the `ST_AsText(g [, options])` function.

    MySQL description:
    - Converts a value in internal geometry format to its WKT representation
      and returns the string result.
    - The function return value has geographic coordinates (latitude, longitude)
      in the order specified by the spatial reference system that applies to the
      geometry argument. An optional options argument may be given to override
      the default axis order.
    """

    def __init__(self, g, options: Any | Sentinel = IGNORED):
        """The `ST_AsText(g [, options])` function.

        MySQL description:
        - Converts a value in internal geometry format to its WKT representation
          and returns the string result.
        - The function return value has geographic coordinates (latitude, longitude)
          in the order specified by the spatial reference system that applies to the
          geometry argument. An optional options argument may be given to override
          the default axis order.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls, multi_p = "LineString(1 1,2 2,3 3)", "MultiPoint(0 0,1 2,2 3)"

        sqlfunc.ST_AsText(sqlfunc.ST_GeomFromText(ls))
        # Escape output: "ST_AsText(ST_GeomFromText('LineString(1 1,2 2,3 3)'))"
        # Expect result: 'LINESTRING(1 1,2 2,3 3)'

        sqlfunc.ST_AsText(sqlfunc.ST_GeomFromText(multi_p))
        # Escape output: "ST_AsText(ST_GeomFromText('MultiPoint(0 0,1 2,2 3)'))"
        # Expect result: 'MULTIPOINT((0 0),(1 2),(2 3))'
        ```
        """
        if options is IGNORED:
            super().__init__("ST_AsText", 1, g)
        else:
            super().__init__("ST_AsText", 2, g, options)


@cython.cclass
class ST_AsWKT(SQLFunction):
    """Represents the `ST_AsWKT(g [, options])` function.

    MySQL description:
    - Converts a value in internal geometry format to its WKT representation
      and returns the string result.
    - The function return value has geographic coordinates (latitude, longitude)
      in the order specified by the spatial reference system that applies to the
      geometry argument. An optional options argument may be given to override
      the default axis order.
    """

    def __init__(self, g, options: Any | Sentinel = IGNORED):
        """The `ST_AsWKT(g [, options])` function.

        MySQL description:
        - Converts a value in internal geometry format to its WKT representation
          and returns the string result.
        - The function return value has geographic coordinates (latitude, longitude)
          in the order specified by the spatial reference system that applies to the
          geometry argument. An optional options argument may be given to override
          the default axis order.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls, multi_p = "LineString(1 1,2 2,3 3)", "MultiPoint(0 0,1 2,2 3)"

        sqlfunc.ST_AsWKT(sqlfunc.ST_GeomFromText(ls))
        # Escape output: "ST_AsWKT(ST_GeomFromText('LineString(1 1,2 2,3 3)'))"
        # Expect result: 'LINESTRING(1 1,2 2,3 3)'

        sqlfunc.ST_AsWKT(sqlfunc.ST_GeomFromText(multi_p))
        # Escape output: "ST_AsWKT(ST_GeomFromText('MultiPoint(0 0,1 2,2 3)'))"
        # Expect result: 'MULTIPOINT((0 0),(1 2),(2 3))'
        ```
        """
        if options is IGNORED:
            super().__init__("ST_AsWKT", 1, g)
        else:
            super().__init__("ST_AsWKT", 2, g, options)


@cython.cclass
class ST_Buffer(SQLFunction):
    """Represents the `ST_Buffer(g, d [, strategy1 [, strategy2 [, strategy3]]])` function.

    MySQL description:
    - Returns a geometry that represents all points whose distance from the geometry
      value g is less than or equal to a distance of d. The result is in the same SRS
      as the geometry argument.
    """

    def __init__(
        self,
        g,
        d,
        strategy1: Any | Sentinel = IGNORED,
        strategy2: Any | Sentinel = IGNORED,
        strategy3: Any | Sentinel = IGNORED,
    ):
        """The `ST_Buffer(g, d [, strategy1 [, strategy2 [, strategy3]]])` function.

        MySQL description:
        - Returns a geometry that represents all points whose distance from the geometry
          value g is less than or equal to a distance of d. The result is in the same SRS
          as the geometry argument.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ST_AsText(sqlfunc.ST_Buffer(sqlfunc.ST_GeomFromText("POINT(0 0)"), 0))
        # Escape output: "ST_AsText(ST_Buffer(ST_GeomFromText('POINT(0 0)'),0))"
        # Expect result: 'POINT(0 0)'

        sqlfunc.ST_AsText(sqlfunc.ST_Buffer(sqlfunc.ST_GeomFromText("POINT(0 0)"), 2, sqlfunc.ST_Buffer_Strategy("point_square")))
        # Escape output: "ST_AsText(ST_Buffer(ST_GeomFromText('POINT(0 0)'),2,ST_Buffer_Strategy('point_square')))"
        # Expect result: 'POLYGON((-2 -2,2 -2,2 2,-2 2,-2 -2))'
        ```
        """
        if strategy1 is IGNORED:
            super().__init__("ST_Buffer", 2, g, d)
        elif strategy2 is IGNORED:
            super().__init__("ST_Buffer", 3, g, d, strategy1)
        elif strategy3 is IGNORED:
            super().__init__("ST_Buffer", 4, g, d, strategy1, strategy2)
        else:
            super().__init__("ST_Buffer", 5, g, d, strategy1, strategy2, strategy3)


@cython.cclass
class ST_Buffer_Strategy(SQLFunction):
    """Represents the `ST_Buffer_Strategy(strategy [, points_per_circle])` function.

    MySQL description:
    - This function returns a strategy byte string for use with ST_Buffer()
      to influence buffer computation.
    - The first argument must be a string indicating a strategy option: for point
      strategies, permitted values are 'point_circle' and 'point_square'; for join
      strategies, permitted values are 'join_round' and 'join_miter'; for end
      strategies, permitted values are 'end_round' and 'end_flat'.
    - If the first argument is 'point_circle', 'join_round', 'join_miter', or 'end_round',
      the points_per_circle argument must be given as a positive numeric value. The maximum
      points_per_circle value is the value of the max_points_in_geometry system variable.
    """

    def __init__(self, strategy: str, points_per_circle: Any | Sentinel = IGNORED):
        """The `ST_Buffer_Strategy(strategy [, points_per_circle])` function.

        MySQL description:
        - This function returns a strategy byte string for use with ST_Buffer()
          to influence buffer computation.
        - The first argument must be a string indicating a strategy option: for point
          strategies, permitted values are 'point_circle' and 'point_square'; for join
          strategies, permitted values are 'join_round' and 'join_miter'; for end
          strategies, permitted values are 'end_round' and 'end_flat'.
        - If the first argument is 'point_circle', 'join_round', 'join_miter', or 'end_round',
          the points_per_circle argument must be given as a positive numeric value. The maximum
          points_per_circle value is the value of the max_points_in_geometry system variable.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ST_Buffer_Strategy("end_flat")
        # Escape output: "ST_Buffer_Strategy('end_flat')"

        sqlfunc.ST_Buffer_Strategy("join_round", 10)
        # Escape output: "ST_Buffer_Strategy('join_round',10)"
        ```
        """
        if points_per_circle is IGNORED:
            super().__init__("ST_Buffer_Strategy", 1, strategy)
        else:
            super().__init__("ST_Buffer_Strategy", 2, strategy, points_per_circle)


@cython.cclass
class ST_Centroid(SQLFunction):
    """Represents the `ST_Centroid({poly|mpoly})` function.

    MySQL description:
    - Returns the mathematical centroid for the Polygon or MultiPolygon argument
      as a Point. The result is not guaranteed to be on the MultiPolygon.
    - This function processes geometry collections by computing the centroid point
      for components of highest dimension in the collection. Such components are
      extracted and made into a single MultiPolygon, MultiLineString, or MultiPoint
      for centroid computation.
    - ST_Centroid() handles its arguments as described in the introduction to this
      section, with these exceptions: The return value is NULL for the additional
      condition that the argument is an empty geometry collection; if the geometry
      has an SRID value for a geographic spatial reference system (SRS), an
      ER_NOT_IMPLEMENTED_FOR_GEOGRAPHIC_SRS error occurs.
    """

    def __init__(self, poly):
        """The `ST_Centroid({poly|mpoly})` function.

        MySQL description:
        - Returns the mathematical centroid for the Polygon or MultiPolygon argument
          as a Point. The result is not guaranteed to be on the MultiPolygon.
        - This function processes geometry collections by computing the centroid point
          for components of highest dimension in the collection. Such components are
          extracted and made into a single MultiPolygon, MultiLineString, or MultiPoint
          for centroid computation.
        - ST_Centroid() handles its arguments as described in the introduction to this
          section, with these exceptions: The return value is NULL for the additional
          condition that the argument is an empty geometry collection; if the geometry
          has an SRID value for a geographic spatial reference system (SRS), an
          ER_NOT_IMPLEMENTED_FOR_GEOGRAPHIC_SRS error occurs.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "Polygon((0 0,10 0,10 10,0 10,0 0),(5 5,7 5,7 7,5 7,5 5))"

        sqlfunc.ST_Centroid(sqlfunc.ST_GeomFromText(poly))
        # Escape output: "ST_Centroid(ST_GeomFromText('Polygon((0 0,10 0,10 10,0 10,0 0),(5 5,7 5,7 7,5 7,5 5))'))"
        # Expect result: 'POINT(4.95833333333333 4.95833333333333)'
        ```
        """
        super().__init__("ST_Centroid", 1, poly)


@cython.cclass
class ST_Contains(SQLFunction):
    """Represents the `ST_Contains(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether g1 completely contains g2
      (this means that g1 and g2 must not intersect). This relationship
      is the inverse of that tested by ST_Within().
    - ST_Contains() handles its arguments as described in the
      introduction to this section.
    """

    def __init__(self, g1, g2):
        """The `ST_Contains(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether g1 completely contains g2
          (this means that g1 and g2 must not intersect). This relationship
          is the inverse of that tested by ST_Within().
        - ST_Contains() handles its arguments as described in the
          introduction to this section.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "POLYGON((0 0,0 3,3 3,3 0,0 0))"
        p1, p2, p3 = "POINT(1 1)", "POINT(3 3)", "POINT(5 5)"

        sqlfunc.ST_Contains(sqlfunc.ST_GeomFromText(poly), sqlfunc.ST_GeomFromText(p1))
        # Escape output: "ST_Contains(ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('POINT(1 1)'))"
        # Expect result: 1

        sqlfunc.ST_Contains(sqlfunc.ST_GeomFromText(poly), sqlfunc.ST_GeomFromText(p2))
        # Escape output: "ST_Contains(ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('POINT(3 3)'))"
        # Expect result: 1

        sqlfunc.ST_Contains(sqlfunc.ST_GeomFromText(poly), sqlfunc.ST_GeomFromText(p3))
        # Escape output: "ST_Contains(ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('POINT(5 5)'))"
        # Expect result: 0
        ```
        """
        super().__init__("ST_Contains", 2, g1, g2)


@cython.cclass
class ST_ConvexHull(SQLFunction):
    """Represents the `ST_ConvexHull(g)` function.

    MySQL description:
    - Returns a geometry that represents the convex hull of
      the geometry value g.
    - This function computes a geometry's convex hull by first
      checking whether its vertex points are colinear. The function
      returns a linear hull if so, a polygon hull otherwise. This
      function processes geometry collections by extracting all
      vertex points of all components of the collection, creating
      a MultiPoint value from them, and computing its convex hull.
    """

    def __init__(self, g):
        """The `ST_ConvexHull(g)` function.

        MySQL description:
        - Returns a geometry that represents the convex hull of
          the geometry value g.
        - This function computes a geometry's convex hull by first
          checking whether its vertex points are colinear. The function
          returns a linear hull if so, a polygon hull otherwise. This
          function processes geometry collections by extracting all
          vertex points of all components of the collection, creating
          a MultiPoint value from them, and computing its convex hull.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        multi_p = "MULTIPOINT(5 0,25 0,15 10,15 25)"

        sqlfunc.ST_ConvexHull(sqlfunc.ST_GeomFromText(multi_p))
        # Escape output: "ST_ConvexHull(ST_GeomFromText('MULTIPOINT(5 0,25 0,15 10,15 25)'))"
        # Expect result: 'POLYGON((5 0, 25 0, 15 25, 5 0))'
        ```
        """
        super().__init__("ST_ConvexHull", 1, g)


@cython.cclass
class ST_Crosses(SQLFunction):
    """Represents the `ST_Crosses(g1, g2)` function.

    MySQL description:
    - Two geometries spatially cross if their spatial relation
      has the following properties:
    - Unless g1 and g2 are both of dimension 1: g1 crosses g2 if
      the interior of g2 has points in common with the interior
      of g1, but g2 does not cover the entire interior of g1.
    - If both g1 and g2 are of dimension 1: If the lines cross each
      other in a finite number of points (that is, no common line
      segments, only single points in common).
    - This function returns 1 or 0 to indicate whether g1 spatially
      crosses g2.
    """

    def __init__(self, g1, g2):
        """The `ST_Crosses(g1, g2)` function.

        MySQL description:
        - Two geometries spatially cross if their spatial relation
          has the following properties:
        - Unless g1 and g2 are both of dimension 1: g1 crosses g2 if
          the interior of g2 has points in common with the interior
          of g1, but g2 does not cover the entire interior of g1.
        - If both g1 and g2 are of dimension 1: If the lines cross each
          other in a finite number of points (that is, no common line
          segments, only single points in common).
        - This function returns 1 or 0 to indicate whether g1 spatially
          crosses g2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LINESTRING(0 0, 10 10)", "LINESTRING(0 10, 10 0)"

        sqlfunc.ST_Crosses(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls2))
        # Escape output: "ST_Crosses(ST_GeomFromText('LINESTRING(0 0, 10 10)'),ST_GeomFromText('LINESTRING(0 10, 10 0)'))"
        # Expect result: 1
        ```
        """
        super().__init__("ST_Crosses", 2, g1, g2)


@cython.cclass
class ST_Difference(SQLFunction):
    """Represents the `ST_Difference(g1, g2)` function.

    MySQL description:
    - Returns a geometry that represents the point set difference of the
      geometry values g1 and g2. The result is in the same SRS as the
      geometry arguments.
    - ST_Difference() permits arguments in either a Cartesian or a geographic
      SRS, and handles its arguments as described in the introduction to this
      section.
    """

    def __init__(self, g1, g2):
        """The `ST_Difference(g1, g2)` function.

        MySQL description:
        - Returns a geometry that represents the point set difference of the
          geometry values g1 and g2. The result is in the same SRS as the
          geometry arguments.
        - ST_Difference() permits arguments in either a Cartesian or a geographic
          SRS, and handles its arguments as described in the introduction to this
          section.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p1, p2 = "Point(1 1)", "Point(2 2)"

        sqlfunc.ST_Difference(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2))
        # Escape output: "ST_Difference(ST_GeomFromText('Point(1 1)'),ST_GeomFromText('Point(2 2)'))"
        # Expect result: 'POINT(1 1)'
        ```
        """
        super().__init__("ST_Difference", 2, g1, g2)


@cython.cclass
class ST_Dimension(SQLFunction):
    """Represents the `ST_Dimension(g)` function.

    MySQL description:
    - Returns the inherent dimension of the geometry value g. The
      dimension can be -1, 0, 1, or 2.
    - The meaning of these values, refer to [link](https://dev.mysql.com/doc/refman/8.4/en/gis-class-geometry.html).
    """

    def __init__(self, g):
        """The `ST_Dimension(g)` function.

        MySQL description:
        - Returns the inherent dimension of the geometry value g. The
          dimension can be -1, 0, 1, or 2.
        - The meaning of these values, refer to [link](https://dev.mysql.com/doc/refman/8.4/en/gis-class-geometry.html).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(1 1,2 2)"

        sqlfunc.ST_Dimension(sqlfunc.ST_GeomFromText(ls)))
        # Escape output: "ST_Dimension(ST_GeomFromText('LineString(1 1,2 2)'))"
        # Expect result: 1
        ```
        """
        super().__init__("ST_Dimension", 1, g)


@cython.cclass
class ST_Disjoint(SQLFunction):
    """Represents the `ST_Disjoint(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether g1 is spatially disjoint
      from (does not intersect) g2.
    """

    def __init__(self, g1, g2):
        """The `ST_Disjoint(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether g1 is spatially disjoint
          from (does not intersect) g2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly1, poly2 = "POLYGON((0 0,0 3,3 3,3 0,0 0))", "POLYGON((4 4,4 6,6 6,6 4,4 4))"

        sqlfunc.ST_Disjoint(sqlfunc.ST_GeomFromText(poly1), sqlfunc.ST_GeomFromText(poly2))
        # Escape output: "ST_Disjoint(ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('POLYGON((4 4,4 6,6 6,6 4,4 4))'))"
        # Expect result: 1
        ```
        """
        super().__init__("ST_Disjoint", 2, g1, g2)


@cython.cclass
class ST_Distance(SQLFunction):
    """Represents the `ST_Distance(g1, g2 [, unit])` function.

    MySQL description:
    - Returns the distance between g1 and g2, measured in the length unit of the
      spatial reference system (SRS) of the geometry arguments, or in the unit of
      the optional unit argument if that is specified.
    - This function processes geometry collections by returning the shortest distance
      among all combinations of the components of the two geometry arguments.
    """

    def __init__(self, g1, g2, unit: Any | Sentinel = IGNORED):
        """The `ST_Distance(g1, g2 [, unit])` function.

        MySQL description:
        - Returns the distance between g1 and g2, measured in the length unit of the
          spatial reference system (SRS) of the geometry arguments, or in the unit of
          the optional unit argument if that is specified.
        - This function processes geometry collections by returning the shortest distance
          among all combinations of the components of the two geometry arguments.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p1, p2 = "POINT(1 1)", "POINT(2 2)"

        sqlfunc.ST_Distance(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2))
        # Escape output: "ST_Distance(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(2 2)'))"
        # Expect result: 1.4142135623730951

        sqlfunc.ST_Distance(sqlfunc.ST_GeomFromText(p1, 4326), sqlfunc.ST_GeomFromText(p2, 4326), "foot")
        # Escape output: "ST_Distance(ST_GeomFromText('POINT(1 1)',4326),ST_GeomFromText('POINT(2 2)',4326),'foot')"
        # Expect result: 514679.7439273146
        ```
        """
        if unit is IGNORED:
            super().__init__("ST_Distance", 2, g1, g2)
        else:
            super().__init__("ST_Distance", 3, g1, g2, unit)


@cython.cclass
class ST_Distance_Sphere(SQLFunction):
    """Represents the `ST_Distance_Sphere(g1, g2 [, radius])` function.

    MySQL description:
    - Returns the minimum spherical distance between Point or MultiPoint
      arguments on a sphere, in meters. (For general-purpose distance
      calculations, see the ST_Distance() function.) The optional radius
      argument should be given in meters.
    - If both geometry parameters are valid Cartesian Point or MultiPoint
      values in SRID 0, the return value is shortest distance between the
      two geometries on a sphere with the provided radius. If omitted, the
      default radius is 6,370,986 meters, Point X and Y coordinates are
      interpreted as longitude and latitude, respectively, in degrees.
    - If both geometry parameters are valid Point or MultiPoint values in
      a geographic spatial reference system (SRS), the return value is the
      shortest distance between the two geometries on a sphere with the
      provided radius. If omitted, the default radius is equal to the mean
      radius, defined as (2a+b)/3, where a is the semi-major axis and b is
      the semi-minor axis of the SRS.
    """

    def __init__(self, g1, g2, radius: Any | Sentinel = IGNORED):
        """The `ST_Distance_Sphere(g1, g2 [, radius])` function.

        MySQL description:
        - Returns the minimum spherical distance between Point or MultiPoint
          arguments on a sphere, in meters. (For general-purpose distance
          calculations, see the ST_Distance() function.) The optional radius
          argument should be given in meters.
        - If both geometry parameters are valid Cartesian Point or MultiPoint
          values in SRID 0, the return value is shortest distance between the
          two geometries on a sphere with the provided radius. If omitted, the
          default radius is 6,370,986 meters, Point X and Y coordinates are
          interpreted as longitude and latitude, respectively, in degrees.
        - If both geometry parameters are valid Point or MultiPoint values in
          a geographic spatial reference system (SRS), the return value is the
          shortest distance between the two geometries on a sphere with the
          provided radius. If omitted, the default radius is equal to the mean
          radius, defined as (2a+b)/3, where a is the semi-major axis and b is
          the semi-minor axis of the SRS.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p1, p2 = "POINT(0 0)", "POINT(180 0)"

        sqlfunc.ST_Distance_Sphere(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2))
        # Escape output: "ST_Distance_Sphere(ST_GeomFromText('POINT(0 0)'),ST_GeomFromText('POINT(180 0)'))"
        # Expect result: 20015042.813723423
        ```
        """
        if radius is IGNORED:
            super().__init__("ST_Distance_Sphere", 2, g1, g2)
        else:
            super().__init__("ST_Distance_Sphere", 3, g1, g2, radius)


@cython.cclass
class ST_EndPoint(SQLFunction):
    """Represents the `ST_EndPoint(ls)` function.

    MySQL description:
    - Returns the Point that is the endpoint of the LineString value ls.
    """

    def __init__(self, ls):
        """The `ST_EndPoint(ls)` function.

        MySQL description:
        - Returns the Point that is the endpoint of the LineString value ls.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(1 1,2 2,3 3)"

        sqlfunc.ST_EndPoint(sqlfunc.ST_GeomFromText(ls))
        # Escape output: "ST_EndPoint(ST_GeomFromText('LineString(1 1,2 2,3 3)'))"
        # Expect result: 'POINT(3 3)'
        ```
        """
        super().__init__("ST_EndPoint", 1, ls)


@cython.cclass
class ST_Envelope(SQLFunction):
    """Represents the `ST_Envelope(g)` function.

    MySQL description:
    - Returns the minimum bounding rectangle (MBR) for the geometry value g.
      The result is returned as a Polygon value that is defined by the corner
      points of the bounding box: `POLYGON((MINX MINY, MAXX MINY, MAXX MAXY, MINX MAXY, MINX MINY))`
    - If the argument is a point or a vertical or horizontal line segment,
      ST_Envelope() returns the point or the line segment as its MBR rather
      than returning an invalid polygon.
    """

    def __init__(self, g):
        """The `ST_Envelope(g)` function.

        MySQL description:
        - Returns the minimum bounding rectangle (MBR) for the geometry value g.
          The result is returned as a Polygon value that is defined by the corner
          points of the bounding box: `POLYGON((MINX MINY, MAXX MINY, MAXX MAXY, MINX MAXY, MINX MINY))`
        - If the argument is a point or a vertical or horizontal line segment,
          ST_Envelope() returns the point or the line segment as its MBR rather
          than returning an invalid polygon.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LineString(1 1,2 2)", "LineString(1 1,1 2)"

        sqlfunc.ST_Envelope(sqlfunc.ST_GeomFromText(ls1))
        # Escape output: "ST_Envelope(ST_GeomFromText('LineString(1 1,2 2)'))"
        # Expect result: 'POLYGON((1 1,2 1,2 2,1 2,1 1))'

        sqlfunc.ST_Envelope(sqlfunc.ST_GeomFromText(ls2))
        # Escape output: "ST_Envelope(ST_GeomFromText('LineString(1 1,1 2)'))"
        # Expect result: 'LINESTRING(1 1,1 2)'
        ```
        """
        super().__init__("ST_Envelope", 1, g)


@cython.cclass
class ST_Equals(SQLFunction):
    """Represents the `ST_Equals(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether g1 is spatially equal to g2.
    - ST_Equals() handles its arguments as described in the introduction to this
      section, except that it does not return NULL for empty geometry arguments.
    """

    def __init__(self, g1, g2):
        """The `ST_Equals(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether g1 is spatially equal to g2.
        - ST_Equals() handles its arguments as described in the introduction to this
          section, except that it does not return NULL for empty geometry arguments.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p1, p2 = "POINT(1 1)", "POINT(2 2)"

        sqlfunc.ST_Equals(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p1))
        # Escape output: "ST_Equals(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(1 1)'))"
        # Expect result: 1

        sqlfunc.ST_Equals(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2))
        # Escape output: "ST_Equals(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(2 2)'))"
        # Expect result: 0
        ```
        """
        super().__init__("ST_Equals", 2, g1, g2)


@cython.cclass
class ST_ExteriorRing(SQLFunction):
    """Represents the `ST_ExteriorRing(poly)` function.

    MySQL description:
    - Returns the exterior ring of the Polygon value poly as a LineString.
    """

    def __init__(self, poly):
        """The `ST_ExteriorRing(poly)` function.

        MySQL description:
        - Returns the exterior ring of the Polygon value poly as a LineString.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.ST_ExteriorRing(sqlfunc.ST_GeomFromText(poly))
        # Escape output: "ST_ExteriorRing(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 'LINESTRING(0 0, 0 3, 3 3, 3 0, 0 0)'
        ```
        """
        super().__init__("ST_ExteriorRing", 1, poly)


@cython.cclass
class ST_FrechetDistance(SQLFunction):
    """Represents the `ST_FrechetDistance(g1, g2 [, unit])` function.

    MySQL description:
    - Returns the discrete Fréchet distance between two geometries, reflecting
      how similar the geometries are. The result is a double-precision number
      measured in the length unit of the spatial reference system (SRS) of the
      geometry arguments, or in the length unit of the unit argument if that
      argument is given.
    - This function implements the discrete Fréchet distance, which means it is
      restricted to distances between the points of the geometries. For example,
      given two LineString arguments, only the points explicitly mentioned in
      the geometries are considered. Points on the line segments between these
      points are not considered.
    """

    def __init__(self, g1, g2, unit: Any | Sentinel = IGNORED):
        """The `ST_FrechetDistance(g1, g2 [, unit])` function.

        MySQL description:
        - Returns the discrete Fréchet distance between two geometries, reflecting
          how similar the geometries are. The result is a double-precision number
          measured in the length unit of the spatial reference system (SRS) of the
          geometry arguments, or in the length unit of the unit argument if that
          argument is given.
        - This function implements the discrete Fréchet distance, which means it is
          restricted to distances between the points of the geometries. For example,
          given two LineString arguments, only the points explicitly mentioned in
          the geometries are considered. Points on the line segments between these
          points are not considered.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LINESTRING(0 0,0 5,5 5)", "LINESTRING(0 1,0 6,3 3,5 6)"

        sqlfunc.ST_FrechetDistance(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls2))
        # Escape output: "ST_FrechetDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)'))"
        # Expect result: 2.8284271247461903

        sqlfunc.ST_FrechetDistance(sqlfunc.ST_GeomFromText(ls1, 4326), sqlfunc.ST_GeomFromText(ls2, 4326), "foot")
        # Escape output: "ST_FrechetDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)',4326),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)',4326),'foot')"
        # Expect result: 1028284.7767115477
        ```
        """
        if unit is IGNORED:
            super().__init__("ST_FrechetDistance", 2, g1, g2)
        else:
            super().__init__("ST_FrechetDistance", 3, g1, g2, unit)


@cython.cclass
class ST_GeoHash(SQLFunction):
    """Represents the `ST_GeoHash(longitude, latitude, max_length)
    or ST_GeoHash(point, max_length) function.

    MySQL description:
    - Returns a geohash string in the connection character set and collation.
    - For the first syntax, the longitude must be a number in the range
      [-180, 180], and the latitude must be a number in the range [-90, 90].
    - For the second syntax, a POINT value is required, where the X and Y
      coordinates are in the valid ranges for longitude and latitude,
      respectively.
    - The resulting string is no longer than max_length characters, which
      has an upper limit of 100. The string might be shorter than max_length
      characters because the algorithm that creates the geohash value continues
      until it has created a string that is either an exact representation of
      the location or max_length characters, whichever comes first.
    """

    def __init__(self, arg1, arg2, arg3: Any | Sentinel = IGNORED):
        """The `ST_GeoHash(longitude, latitude, max_length)
        or ST_GeoHash(point, max_length)` function.

        MySQL description:
        - Returns a geohash string in the connection character set and collation.
        - For the first syntax, the longitude must be a number in the range
          [-180, 180], and the latitude must be a number in the range [-90, 90].
        - For the second syntax, a POINT value is required, where the X and Y
          coordinates are in the valid ranges for longitude and latitude,
          respectively.
        - The resulting string is no longer than max_length characters, which
          has an upper limit of 100. The string might be shorter than max_length
          characters because the algorithm that creates the geohash value continues
          until it has created a string that is either an exact representation of
          the location or max_length characters, whichever comes first.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ST_GeoHash(180, 0, 10)
        # Escape output: "ST_GeoHash(180,0,10)"
        # Expect result: 'xbpbpbpbpb'

        sqlfunc.ST_GeoHash(sqlfunc.Point(180, 0), 10)
        # Escape output: "ST_GeoHash(Point(180,0),10)"
        # Expect result: 'xbpbpbpbpb'
        ```
        """
        if arg3 is IGNORED:
            super().__init__("ST_GeoHash", 2, arg1, arg2)
        else:
            super().__init__("ST_GeoHash", 3, arg1, arg2, arg3)


@cython.cclass
class ST_GeomCollFromText(SQLFunction):
    """Represents the `ST_GeomCollFromText(wkt [, srid [, options]])` function.

    MySQL description:
    - Constructs a GeometryCollection value using its WKT representation and SRID.
    """

    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_GeomCollFromText(wkt [, srid [, options]])` function.

        MySQL description:
        - Constructs a GeometryCollection value using its WKT representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        multi_ls = "MULTILINESTRING((10 10, 11 11), (9 9, 10 10))"

        sqlfunc.ST_GeomCollFromText(multi_ls)
        # Escape output: "ST_GeomCollFromText('MULTILINESTRING((10 10, 11 11), (9 9, 10 10))')"
        # Expect result: 'MULTILINESTRING((10 10, 11 11), (9 9, 10 10))'

        sqlfunc.ST_GeomCollFromText(multi_ls, 4326)
        # Escape output: "ST_GeomCollFromText('MULTILINESTRING((10 10, 11 11), (9 9, 10 10))',4326)"
        # Expect result: 'MULTILINESTRING((10 10, 11 11), (9 9, 10 10)) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_GeomCollFromText", 1, wkt)
        elif options is IGNORED:
            super().__init__("ST_GeomCollFromText", 2, wkt, srid)
        else:
            super().__init__("ST_GeomCollFromText", 3, wkt, srid, options)


@cython.cclass
class ST_GeomCollFromWKB(SQLFunction):
    """Represents the `ST_GeomCollFromWKB(wkb [, srid [, options]])` function.

    MySQL description:
    - Constructs a GeometryCollection value using its WKB representation and SRID.
    """

    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_GeomCollFromWKB(wkb [, srid [, options]])` function.

        MySQL description:
        - Constructs a GeometryCollection value using its WKB representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        multi_ls = "MULTILINESTRING((10 10, 11 11), (9 9, 10 10))"

        sqlfunc.ST_GeomCollFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_GeomCollFromText(multi_ls)))
        # Escape output: "ST_GeomCollFromWKB(ST_AsBinary(ST_GeomCollFromText('MULTILINESTRING((10 10, 11 11), (9 9, 10 10))')))"
        # Expect result: 'MULTILINESTRING((10 10, 11 11), (9 9, 10 10))'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_GeomCollFromWKB", 1, wkb)
        elif options is IGNORED:
            super().__init__("ST_GeomCollFromWKB", 2, wkb, srid)
        else:
            super().__init__("ST_GeomCollFromWKB", 3, wkb, srid, options)


@cython.cclass
class ST_GeometryN(SQLFunction):
    """Represents the `ST_GeometryN(gc, N)` function.

    MySQL description:
    - Returns the N-th geometry in the GeometryCollection value gc.
      Geometries are numbered beginning with 1.
    """

    def __init__(self, gc, N):
        """The `ST_GeometryN(gc, N)` function.

        MySQL description:
        - Returns the N-th geometry in the GeometryCollection value gc.
          Geometries are numbered beginning with 1.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        gc = "GeometryCollection(Point(1 1),LineString(2 2, 3 3))"

        sqlfunc.ST_GeometryN(sqlfunc.ST_GeomFromText(gc), 1)
        # Escape output: "ST_GeometryN(ST_GeomFromText('GeometryCollection(Point(1 1),LineString(2 2, 3 3))'),1)"
        # Expect result: 'POINT(1 1)'
        ```
        """
        super().__init__("ST_GeometryN", 2, gc, N)


@cython.cclass
class ST_GeometryType(SQLFunction):
    """Represents the `ST_GeometryType(g)` function.

    MySQL description:
    - Returns a binary string indicating the name of the geometry type
      of which the geometry instance g is a member. The name corresponds
      to one of the instantiable Geometry subclasses.
    """

    def __init__(self, g):
        """The `ST_GeometryType(g)` function.

        MySQL description:
        - Returns a binary string indicating the name of the geometry type
          of which the geometry instance g is a member. The name corresponds
          to one of the instantiable Geometry subclasses.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ST_GeometryType(sqlfunc.ST_GeomFromText('POINT(1 1)'))
        # Escape output: "ST_GeometryType(ST_GeomFromText('POINT(1 1)'))"
        # Expect result: 'POINT'
        ```
        """
        super().__init__("ST_GeometryType", 1, g)


@cython.cclass
class ST_GeomFromGeoJSON(SQLFunction):
    """Represents the `ST_GeomFromGeoJSON(string [, options [, srid]])` function.

    MySQL description:
    - Parses a string str representing a GeoJSON object and returns a geometry.
    - If any argument is NULL, the return value is NULL. If any non-NULL
      argument is invalid, an error occurs.
    - options, if given, describes how to handle GeoJSON documents that
      contain geometries with coordinate dimensions higher than 2. option value of
      1 reject the document and produce an error. This is the default if options
      is not specified; options values of 2, 3, and 4 currently produce the same
      effect, accept the document and strip off the coordinates for higher
      coordinate dimensions.
    - The srid argument, if given, must be a 32-bit unsigned integer. If not given,
      the geometry return value has an SRID of 4326.
    """

    def __init__(
        self,
        string,
        options: Any | Sentinel = IGNORED,
        srid: Any | Sentinel = IGNORED,
    ):
        """The `ST_GeomFromGeoJSON(string [, options [, srid]])` function.

        MySQL description:
        - Parses a string str representing a GeoJSON object and returns a geometry.
        - If any argument is NULL, the return value is NULL. If any non-NULL
          argument is invalid, an error occurs.
        - options, if given, describes how to handle GeoJSON documents that
          contain geometries with coordinate dimensions higher than 2. option value of
          1 reject the document and produce an error. This is the default if options
          is not specified; options values of 2, 3, and 4 currently produce the same
          effect, accept the document and strip off the coordinates for higher
          coordinate dimensions.
        - The srid argument, if given, must be a 32-bit unsigned integer. If not given,
          the geometry return value has an SRID of 4326.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        geojson = '{ "type": "Point", "coordinates": [102.0, 0.0]}'

        sqlfunc.ST_GeomFromGeoJSON(geojson)
        # Escape output: "ST_GeomFromGeoJSON('{ \"type\": \"Point\", \"coordinates\": [102.0, 0.0]}')"
        # Expect result: 'POINT(102 0) | 4326'

        sqlfunc.ST_GeomFromGeoJSON(geojson, 2)
        # Escape output: "ST_GeomFromGeoJSON('{ \"type\": \"Point\", \"coordinates\": [102.0, 0.0]}',2)"
        # Expect result: 'POINT(102 0) | 4326'
        ```
        """
        if options is IGNORED:
            super().__init__("ST_GeomFromGeoJSON", 1, string)
        elif srid is IGNORED:
            super().__init__("ST_GeomFromGeoJSON", 2, string, options)
        else:
            super().__init__("ST_GeomFromGeoJSON", 3, string, options, srid)


@cython.cclass
class ST_GeomFromText(SQLFunction):
    """Represents the `ST_GeomFromText(wkt [, srid [, options]])` function.

    MySQL description:
    - Constructs a geometry value of any type using its WKT representation and SRID.
    """

    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_GeomFromText(wkt [, srid [, options]])` function.

        MySQL description:
        - Constructs a geometry value of any type using its WKT representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p1, p2 = "POINT(10 20)", "POINT(-73.935242 40.730610)"

        sqlfunc.ST_GeomFromText(p1)
        # Escape output: "ST_GeomFromText('POINT(10 20)')"
        # Expect result: 'POINT(10 20)'

        sqlfunc.ST_GeomFromText(p2, 4326)
        # Escape output: "ST_GeomFromText('POINT(-73.935242 40.730610)',4326)"
        # Expect result: 'POINT(40.73061 -73.935242) | 4326'

        sqlfunc.ST_GeomFromText(p1, 0, "axis-order=lat-long")
        # Escape output: "ST_GeomFromText('POINT(10 20)',0,'axis-order=lat-long')"
        # Expect result: 'POINT(10 20)'
        """
        if srid is IGNORED:
            super().__init__("ST_GeomFromText", 1, wkt)
        elif options is IGNORED:
            super().__init__("ST_GeomFromText", 2, wkt, srid)
        else:
            super().__init__("ST_GeomFromText", 3, wkt, srid, options)


@cython.cclass
class ST_GeomFromWKB(SQLFunction):
    """Represents the `ST_GeomFromWKB(wkb [, srid [, options]])` function.

    MySQL description:
    - Constructs a geometry value of any type using its WKB representation and SRID.
    """

    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_GeomFromWKB(wkb [, srid [, options]])` function.

        MySQL description:
        - Constructs a geometry value of any type using its WKB representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p = "POINT(10 20)"

        sqlfunc.ST_GeomFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_GeomFromText(p)))
        # Escape output: "ST_GeomFromWKB(ST_AsBinary(ST_GeomFromText('POINT(10 20)')))"
        # Expect result: 'POINT(10 20)'

        sqlfunc.ST_GeomFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_GeomFromText(p)), 4326)
        # Escape output: "ST_GeomFromWKB(ST_AsBinary(ST_GeomFromText('POINT(10 20)')),4326)"
        # Expect result: 'POINT(10 20) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_GeomFromWKB", 1, wkb)
        elif options is IGNORED:
            super().__init__("ST_GeomFromWKB", 2, wkb, srid)
        else:
            super().__init__("ST_GeomFromWKB", 3, wkb, srid, options)


@cython.cclass
class ST_HausdorffDistance(SQLFunction):
    """Represents the `ST_HausdorffDistance(g1, g2 [, unit])` function.

    MySQL description:
    - Returns the discrete Hausdorff distance between two geometries, reflecting
      how similar the geometries are. The result is a double-precision number
      measured in the length unit of the spatial reference system (SRS) of the
      geometry arguments, or in the length unit of the unit argument if that
      argument is given.
    - This function implements the discrete Hausdorff distance, which means it
      is restricted to distances between the points of the geometries. For example,
      given two LineString arguments, only the points explicitly mentioned in the
      geometries are considered. Points on the line segments between these points
      are not considered.
    """

    def __init__(self, g1, g2, unit: Any | Sentinel = IGNORED):
        """The `ST_HausdorffDistance(g1, g2 [, unit])` function.

        MySQL description:
        - Returns the discrete Hausdorff distance between two geometries, reflecting
          how similar the geometries are. The result is a double-precision number
          measured in the length unit of the spatial reference system (SRS) of the
          geometry arguments, or in the length unit of the unit argument if that
          argument is given.
        - This function implements the discrete Hausdorff distance, which means it
          is restricted to distances between the points of the geometries. For example,
          given two LineString arguments, only the points explicitly mentioned in the
          geometries are considered. Points on the line segments between these points
          are not considered.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LINESTRING(0 0,0 5,5 5)", "LINESTRING(0 1,0 6,3 3,5 6)"

        sqlfunc.ST_HausdorffDistance(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls2))
        # Escape output: "ST_HausdorffDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)'))"
        # Expect result: 1

        sqlfunc.ST_HausdorffDistance(sqlfunc.ST_GeomFromText(ls1, 4326), sqlfunc.ST_GeomFromText(ls2, 4326))
        # Escape output: "ST_HausdorffDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)',4326),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)',4326))"
        # Expect result: 111319.49079326246

        sqlfunc.ST_HausdorffDistance(sqlfunc.ST_GeomFromText(ls1, 4326), sqlfunc.ST_GeomFromText(ls2, 4326), "foot")
        # Escape output: "ST_HausdorffDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)',4326),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)',4326),'foot')"
        # Expect result: 365221.4264870815
        ```
        """
        if unit is IGNORED:
            super().__init__("ST_HausdorffDistance", 2, g1, g2)
        else:
            super().__init__("ST_HausdorffDistance", 3, g1, g2, unit)


@cython.cclass
class ST_InteriorRingN(SQLFunction):
    """Represents the `ST_InteriorRingN(poly, N)` function.

    MySQL description:
    - Returns the N-th interior ring for the Polygon value poly as
      a LineString. Rings are numbered beginning with 1.
    """

    def __init__(self, poly, N):
        """The `ST_InteriorRingN(poly, N)` function.

        MySQL description:
        - Returns the N-th interior ring for the Polygon value poly as
          a LineString. Rings are numbered beginning with 1.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.ST_InteriorRingN(sqlfunc.ST_GeomFromText(poly), 1)
        # Escape output: "ST_InteriorRingN(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))'),1)"
        # Expect result: 'LINESTRING(1 1, 1 2, 2 2, 2 1, 1 1)'
        ```
        """
        super().__init__("ST_InteriorRingN", 2, poly, N)


@cython.cclass
class ST_Intersection(SQLFunction):
    """Represents the `ST_Intersection(g1, g2)` function.

    MySQL description:
    - Returns a geometry that represents the point set intersection
      of the geometry values g1 and g2. The result is in the same
      SRS as the geometry arguments.
    """

    def __init__(self, g1, g2):
        """The `ST_Intersection(g1, g2)` function.

        MySQL description:
        - Returns a geometry that represents the point set intersection
          of the geometry values g1 and g2. The result is in the same
          SRS as the geometry arguments.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LineString(1 1, 3 3)", "LineString(1 3, 3 1)"

        sqlfunc.ST_Intersection(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls1))
        # Escape output: "ST_Intersection(ST_GeomFromText('LineString(1 1, 3 3)'),ST_GeomFromText('LineString(1 1, 3 3)'))"
        # Expect result: 'LINESTRING(1 1, 3 3)'
        ```
        """
        super().__init__("ST_Intersection", 2, g1, g2)


@cython.cclass
class ST_Intersects(SQLFunction):
    """Represents the `ST_Intersects(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether g1 spatially intersects g2.
    """

    def __init__(self, g1, g2):
        """The `ST_Intersects(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether g1 spatially intersects g2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p1, p2 = "POINT(1 1)", "POINT(2 2)"

        sqlfunc.ST_Intersects(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p1))
        # Escape output: "ST_Intersects(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(1 1)'))"
        # Expect result: 1

        sqlfunc.ST_Intersects(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2))
        # Escape output: "ST_Intersects(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(2 2)'))"
        # Expect result: 0
        ```
        """
        super().__init__("ST_Intersects", 2, g1, g2)


@cython.cclass
class ST_IsClosed(SQLFunction):
    """Represents the `ST_IsClosed(ls)` function.

    MySQL description:
    - For a LineString value ls, ST_IsClosed() returns 1 if ls is closed
      (that is, its ST_StartPoint() and ST_EndPoint() values are the same).
    - For a MultiLineString value ls, ST_IsClosed() returns 1 if ls is
      closed (that is, the ST_StartPoint() and ST_EndPoint() values are
      the same for each LineString in ls).
    - ST_IsClosed() returns 0 if ls is not closed, and NULL if ls is NULL.
    """

    def __init__(self, ls):
        """The `ST_IsClosed(ls)` function.

        MySQL description:
        - For a LineString value ls, ST_IsClosed() returns 1 if ls is closed
          (that is, its ST_StartPoint() and ST_EndPoint() values are the same).
        - For a MultiLineString value ls, ST_IsClosed() returns 1 if ls is
          closed (that is, the ST_StartPoint() and ST_EndPoint() values are
          the same for each LineString in ls).
        - ST_IsClosed() returns 0 if ls is not closed, and NULL if ls is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LineString(1 1,2 2,3 3,2 2)", "LineString(1 1,2 2,3 3,1 1)"

        sqlfunc.ST_IsClosed(sqlfunc.ST_GeomFromText(ls1))
        # Escape output: "ST_IsClosed(ST_GeomFromText('LineString(1 1,2 2,3 3,2 2)'))"
        # Expect result: 0

        sqlfunc.ST_IsClosed(sqlfunc.ST_GeomFromText(ls2))
        # Escape output: "ST_IsClosed(ST_GeomFromText('LineString(1 1,2 2,3 3,1 1)'))"
        # Expect result: 1
        ```
        """
        super().__init__("ST_IsClosed", 1, ls)


@cython.cclass
class ST_IsEmpty(SQLFunction):
    """Represents the `ST_IsEmpty(g)` function.

    MySQL description:
    - This function is a placeholder that returns 1 for an empty
      geometry collection value or 0 otherwise.
    - The only valid empty geometry is represented in the form of
      an empty geometry collection value. MySQL does not support
      GIS EMPTY values such as POINT EMPTY.
    """

    def __init__(self, g):
        """The `ST_IsEmpty(g)` function.

        MySQL description:
        - This function is a placeholder that returns 1 for an empty
          geometry collection value or 0 otherwise.
        - The only valid empty geometry is represented in the form of
          an empty geometry collection value. MySQL does not support
          GIS EMPTY values such as POINT EMPTY.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ST_IsEmpty(sqlfunc.ST_GeomFromText('POINT(1 1)'))
        # Escape output: "ST_IsEmpty(ST_GeomFromText('POINT(1 1)'))"
        # Expect result: 0

        sqlfunc.ST_IsEmpty(sqlfunc.ST_GeomFromText('GEOMETRYCOLLECTION EMPTY'))
        # Escape output: "ST_IsEmpty(ST_GeomFromText('GEOMETRYCOLLECTION EMPTY'))"
        # Expect result: 1
        ```
        """
        super().__init__("ST_IsEmpty", 1, g)


@cython.cclass
class ST_IsSimple(SQLFunction):
    """Represents the `ST_IsSimple(g)` function.

    MySQL description:
    - Returns 1 if the geometry value g is simple according to the
      ISO SQL/MM Part 3: Spatial standard. ST_IsSimple() returns 0
      if the argument is not simple.
    - The descriptions of the instantiable geometric classes given
      under Section 13.4.2, “The OpenGIS Geometry Model” include the
      specific conditions that cause class instances to be classified
      as not simple.
    - If the geometry has a geographic SRS with a longitude or latitude
      that is out of range, an error occurs: if a longitude value is not
      in the range (-180, 180], an ER_GEOMETRY_PARAM_LONGITUDE_OUT_OF_RANGE
      error occurs; If a latitude value is not in the range (-90, 90), an
      ER_GEOMETRY_PARAM_LATITUDE_OUT_OF_RANGE error occurs.
    - Ranges shown are in degrees. The exact range limits deviate slightly
      due to floating-point arithmetic.
    """

    def __init__(self, g):
        """The `ST_IsSimple(g)` function.

        MySQL description:
        - Returns 1 if the geometry value g is simple according to the
          ISO SQL/MM Part 3: Spatial standard. ST_IsSimple() returns 0
          if the argument is not simple.
        - The descriptions of the instantiable geometric classes given
          under Section 13.4.2, “The OpenGIS Geometry Model” include the
          specific conditions that cause class instances to be classified
          as not simple.
        - If the geometry has a geographic SRS with a longitude or latitude
          that is out of range, an error occurs: if a longitude value is not
          in the range (-180, 180], an ER_GEOMETRY_PARAM_LONGITUDE_OUT_OF_RANGE
          error occurs; If a latitude value is not in the range (-90, 90), an
          ER_GEOMETRY_PARAM_LATITUDE_OUT_OF_RANGE error occurs.
        - Ranges shown are in degrees. The exact range limits deviate slightly
          due to floating-point arithmetic.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1 = "LINESTRING(0 0,0 5,5 5)"

        sqlfunc.ST_IsSimple(sqlfunc.ST_GeomFromText(ls1))
        # Escape output: "ST_IsSimple(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'))"
        # Expect result: 1
        ```
        """
        super().__init__("ST_IsSimple", 1, g)


@cython.cclass
class ST_IsValid(SQLFunction):
    """Represents the `ST_IsValid(g)` function.

    MySQL description:
    - Returns 1 if the argument is geometrically valid, 0 if the argument is not
      geometrically valid. Geometry validity is defined by the OGC specification.
    - The only valid empty geometry is represented in the form of an empty geometry
      collection value. ST_IsValid() returns 1 in this case. MySQL does not support
      GIS EMPTY values such as POINT EMPTY.
    - If the geometry has a geographic SRS with a longitude or latitude that is
      out of range, an error occurs: if a longitude value is not in the range (-180, 180],
      an ER_GEOMETRY_PARAM_LONGITUDE_OUT_OF_RANGE error occurs; if a latitude value
      is not in the range (-90, 90), an ER_GEOMETRY_PARAM_LATITUDE_OUT_OF_RANGE error occurs.
    """

    def __init__(self, g):
        """The `ST_IsValid(g)` function.

        MySQL description:
        - Returns 1 if the argument is geometrically valid, 0 if the argument is not
          geometrically valid. Geometry validity is defined by the OGC specification.
        - The only valid empty geometry is represented in the form of an empty geometry
          collection value. ST_IsValid() returns 1 in this case. MySQL does not support
          GIS EMPTY values such as POINT EMPTY.
        - If the geometry has a geographic SRS with a longitude or latitude that is
          out of range, an error occurs: if a longitude value is not in the range (-180, 180],
          an ER_GEOMETRY_PARAM_LONGITUDE_OUT_OF_RANGE error occurs; if a latitude value
          is not in the range (-90, 90), an ER_GEOMETRY_PARAM_LATITUDE_OUT_OF_RANGE error occurs.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LINESTRING(0 0,-0.00 0,0.0 0)", "LINESTRING(0 0, 1 1)"

        sqlfunc.ST_IsValid(sqlfunc.ST_GeomFromText(ls1))
        # Escape output: "ST_IsValid(ST_GeomFromText('LINESTRING(0 0,-0.00 0,0.0 0)'))"
        # Expect result: 0

        sqlfunc.ST_IsValid(sqlfunc.ST_GeomFromText(ls2))
        # Escape output: "ST_IsValid(ST_GeomFromText('LINESTRING(0 0, 1 1)'))"
        # Expect result: 1
        ```
        """
        super().__init__("ST_IsValid", 1, g)


@cython.cclass
class ST_LatFromGeoHash(SQLFunction):
    """Represents the `ST_LatFromGeoHash(geohash_str)` function.

    MySQL description:
    - Returns the latitude from a geohash string value, as a double-precision
      number in the range [-90, 90].
    - The ST_LatFromGeoHash() decoding function reads no more than 433 characters
      from the geohash_str argument. That represents the upper limit on information
      in the internal representation of coordinate values. Characters past the 433rd
      are ignored, even if they are otherwise illegal and produce an error.
    """

    def __init__(self, geohash_str):
        """The `ST_LatFromGeoHash(geohash_str)` function.

        MySQL description:
        - Returns the latitude from a geohash string value, as a double-precision
          number in the range [-90, 90].
        - The ST_LatFromGeoHash() decoding function reads no more than 433 characters
          from the geohash_str argument. That represents the upper limit on information
          in the internal representation of coordinate values. Characters past the 433rd
          are ignored, even if they are otherwise illegal and produce an error.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ST_LatFromGeoHash(sqlfunc.ST_GeoHash(45, -20, 10))
        # Escape output: "ST_LatFromGeoHash(ST_GeoHash(45,-20,10))"
        # Expect result: -20
        ```
        """
        super().__init__("ST_LatFromGeoHash", 1, geohash_str)


@cython.cclass
class ST_Latitude(SQLFunction):
    """Represents the `ST_Latitude(p [, new_latitude_val])` function.

    MySQL description:
    - With a single argument representing a valid Point object p that
      has a geographic spatial reference system (SRS), ST_Latitude()
      returns the latitude value of p as a double-precision number.
    - With the optional second argument representing a valid latitude
      value, ST_Latitude() returns a Point object like the first argument
      with its latitude equal to the second argument.
    """

    def __init__(self, p, new_latitude_val: Any | Sentinel = IGNORED):
        """The `ST_Latitude(p [, new_latitude_val])` function.

        MySQL description:
        - With a single argument representing a valid Point object p that
          has a geographic spatial reference system (SRS), ST_Latitude()
          returns the latitude value of p as a double-precision number.
        - With the optional second argument representing a valid latitude
          value, ST_Latitude() returns a Point object like the first argument
          with its latitude equal to the second argument.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p = "POINT(45 90)"

        sqlfunc.ST_Latitude(sqlfunc.ST_GeomFromText(p, 4326))
        # Escape output: "ST_Latitude(ST_GeomFromText('POINT(45 90)',4326))"
        # Expect result: 45

        sqlfunc.ST_Latitude(sqlfunc.ST_GeomFromText(p, 4326), 10)
        # Escape output: "ST_Latitude(ST_GeomFromText('POINT(45 90)',4326),10)"
        # Expect result: 'POINT(90 10) | 4326'
        ```
        """
        if new_latitude_val is IGNORED:
            super().__init__("ST_Latitude", 1, p)
        else:
            super().__init__("ST_Latitude", 2, p, new_latitude_val)


@cython.cclass
class ST_Length(SQLFunction):
    """Represents the `ST_Length(ls [, unit])` function.

    MySQL description:
    - Returns a double-precision number indicating the length of the LineString
      or MultiLineString value ls in its associated spatial reference system.
      The length of a MultiLineString value is equal to the sum of the lengths
      of its elements.
    """

    def __init__(self, ls, unit: Any | Sentinel = IGNORED):
        """The `ST_Length(ls [, unit])` function.

        MySQL description:
        - Returns a double-precision number indicating the length of the LineString
          or MultiLineString value ls in its associated spatial reference system.
          The length of a MultiLineString value is equal to the sum of the lengths
          of its elements.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls, multi_ls = "LineString(1 1,2 2,3 3)", "MultiLineString((1 1,2 2,3 3),(4 4,5 5))"

        sqlfunc.ST_Length(sqlfunc.ST_GeomFromText(ls))
        # Escape output: "ST_Length(ST_GeomFromText('LineString(1 1,2 2,3 3)'))"
        # Expect result: 2.8284271247461903

        sqlfunc.ST_Length(sqlfunc.ST_GeomFromText(ls, 4326), "foot")
        # Escape output: "ST_Length(ST_GeomFromText('LineString(1 1,2 2,3 3)',4326),'foot')"
        # Expect result: 1029205.9131247795

        sqlfunc.ST_Length(sqlfunc.ST_GeomFromText(multi_ls))
        # Escape output: "ST_Length(ST_GeomFromText('MultiLineString((1 1,2 2,3 3),(4 4,5 5))'))"
        # Expect result: 4.242640687119286
        ```
        """
        if unit is IGNORED:
            super().__init__("ST_Length", 1, ls)
        else:
            super().__init__("ST_Length", 2, ls, unit)


@cython.cclass
class ST_LineFromText(SQLFunction):
    """Represents the `ST_LineFromText(wkt [, srid [, options]])` function.

    MySQL description:
    - Constructs a LineString value using its WKT representation and SRID.
    """

    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_LineFromText(wkt [, srid [, options]])` function.

        MySQL description:
        - Constructs a LineString value using its WKT representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LINESTRING(0 0,0 5,5 5)"

        sqlfunc.ST_LineFromText(ls)
        # Escape output: "ST_LineFromText('LINESTRING(0 0,0 5,5 5)')"
        # Expect result: 'LINESTRING(0 0, 0 5, 5 5)'

        sqlfunc.ST_LineFromText(ls, 4326)
        # Escape output: "ST_LineFromText('LINESTRING(0 0,0 5,5 5)',4326)"
        # Expect result: 'LINESTRING(0 0, 5 0, 5 5) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_LineFromText", 1, wkt)
        elif options is IGNORED:
            super().__init__("ST_LineFromText", 2, wkt, srid)
        else:
            super().__init__("ST_LineFromText", 3, wkt, srid, options)


@cython.cclass
class ST_LineFromWKB(SQLFunction):
    """Represents the `ST_LineFromWKB(wkb [, srid [, options]])` function.

    MySQL description:
    - Constructs a LineString value using its WKB representation and SRID.
    """

    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_LineFromWKB(wkb [, srid [, options]])` function.

        MySQL description:
        - Constructs a LineString value using its WKB representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LINESTRING(0 0,0 5,5 5)"

        sqlfunc.ST_LineFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_LineFromText(ls)))
        # Escape output: "ST_LineFromWKB(ST_AsBinary(ST_LineFromText('LINESTRING(0 0,0 5,5 5)')))"
        # Expect result: 'LINESTRING(0 0, 0 5, 5 5)'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_LineFromWKB", 1, wkb)
        elif options is IGNORED:
            super().__init__("ST_LineFromWKB", 2, wkb, srid)
        else:
            super().__init__("ST_LineFromWKB", 3, wkb, srid, options)


@cython.cclass
class ST_LineInterpolatePoint(SQLFunction):
    """Represents the `ST_LineInterpolatePoint(ls, fractional_distance)` function.

    MySQL description:
    - This function takes a LineString geometry and a fractional distance
      in the range [0.0, 1.0] and returns the Point along the LineString
      at the given fraction of the distance from its start point to its
      endpoint. It can be used to answer questions such as which Point
      lies halfway along the road described by the geometry argument.
    - The function is implemented for LineString geometries in all spatial
      reference systems, both Cartesian and geographic.
    - If the fractional_distance argument is 1.0, the result may not be
      exactly the last point of the LineString argument but a point close
      to it due to numerical inaccuracies in approximate-value computations.
    """

    def __init__(self, ls, fractional_distance):
        """The `ST_LineInterpolatePoint(ls, fractional_distance)` function.

        MySQL description:
        - This function takes a LineString geometry and a fractional distance
          in the range [0.0, 1.0] and returns the Point along the LineString
          at the given fraction of the distance from its start point to its
          endpoint. It can be used to answer questions such as which Point
          lies halfway along the road described by the geometry argument.
        - The function is implemented for LineString geometries in all spatial
          reference systems, both Cartesian and geographic.
        - If the fractional_distance argument is 1.0, the result may not be
          exactly the last point of the LineString argument but a point close
          to it due to numerical inaccuracies in approximate-value computations.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LINESTRING(0 0,0 5,5 5)"

        sqlfunc.ST_LineInterpolatePoint(sqlfunc.ST_GeomFromText(ls), 0.5)
        # Escape output: "ST_LineInterpolatePoint(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),0.5)"
        # Expect result: 'POINT(0 5)'
        ```
        """
        super().__init__("ST_LineInterpolatePoint", 2, ls, fractional_distance)


@cython.cclass
class ST_LineInterpolatePoints(SQLFunction):
    """Represents the `ST_LineInterpolatePoints(ls, fractional_distance)` function.

    MySQL description:
    - This function takes a LineString geometry and a fractional distance
      in the range (0.0, 1.0] and returns the MultiPoint consisting of the
      LineString start point, plus Point values along the LineString at each
      fraction of the distance from its start point to its endpoint. It can
      be used to answer questions such as which Point values lie every 10%
      of the way along the road described by the geometry argument.
    - The function is implemented for LineString geometries in all spatial
      reference systems, both Cartesian and geographic.
    - If the fractional_distance argument divides 1.0 with zero remainder
      the result may not contain the last point of the LineString argument
      but a point close to it due to numerical inaccuracies in approximate-value
      computations.
    """

    def __init__(self, ls, fractional_distance):
        """The `ST_LineInterpolatePoints(ls, fractional_distance)` function.

        MySQL description:
        - This function takes a LineString geometry and a fractional distance
          in the range (0.0, 1.0] and returns the MultiPoint consisting of the
          LineString start point, plus Point values along the LineString at each
          fraction of the distance from its start point to its endpoint. It can
          be used to answer questions such as which Point values lie every 10%
          of the way along the road described by the geometry argument.
        - The function is implemented for LineString geometries in all spatial
          reference systems, both Cartesian and geographic.
        - If the fractional_distance argument divides 1.0 with zero remainder
          the result may not contain the last point of the LineString argument
          but a point close to it due to numerical inaccuracies in approximate-value
          computations.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LINESTRING(0 0,0 5,5 5)"

        sqlfunc.ST_LineInterpolatePoints(sqlfunc.ST_GeomFromText(ls), 0.5)
        # Escape output: "ST_LineInterpolatePoints(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),0.5)"
        # Expect result: 'MULTIPOINT(0 5, 5 5)'

        sqlfunc.ST_LineInterpolatePoints(sqlfunc.ST_GeomFromText(ls), 0.25)
        # Escape output: "ST_LineInterpolatePoints(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),0.25)"
        # Expect result: 'MULTIPOINT(0 2.5, 0 5, 2.5 5, 5 5)'
        ```
        """
        super().__init__("ST_LineInterpolatePoints", 2, ls, fractional_distance)


@cython.cclass
class ST_LongFromGeoHash(SQLFunction):
    """Represents the `ST_LongFromGeoHash(geohash_str)` function.

    MySQL description:
    - Returns the longitude from a geohash string value, as
      a double-precision number in the range [-180, 180].
    """

    def __init__(self, geohash_str):
        """The `ST_LongFromGeoHash(geohash_str)` function.

        MySQL description:
        - Returns the longitude from a geohash string value, as
          a double-precision number in the range [-180, 180].

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ST_LongFromGeoHash(sqlfunc.ST_GeoHash(45, -20, 10))
        # Escape output: "ST_LongFromGeoHash(ST_GeoHash(45,-20,10))"
        # Expect result: 45
        ```
        """
        super().__init__("ST_LongFromGeoHash", 1, geohash_str)


@cython.cclass
class ST_Longitude(SQLFunction):
    """Represents the `ST_Longitude(p [, new_longitude_val])` function.

    MySQL description:
    - With a single argument representing a valid Point object p that
      has a geographic spatial reference system (SRS), ST_Longitude()
      returns the longitude value of p as a double-precision number.
    - With the optional second argument representing a valid longitude
      value, ST_Longitude() returns a Point object like the first argument
      with its longitude equal to the second argument.
    """

    def __init__(self, p, new_longitude_val: Any | Sentinel = IGNORED):
        """The `ST_Longitude(p [, new_longitude_val])` function.

        MySQL description:
        - With a single argument representing a valid Point object p that
          has a geographic spatial reference system (SRS), ST_Longitude()
          returns the longitude value of p as a double-precision number.
        - With the optional second argument representing a valid longitude
          value, ST_Longitude() returns a Point object like the first argument
          with its longitude equal to the second argument.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        p = "POINT(45 90)"

        sqlfunc.ST_Longitude(sqlfunc.ST_GeomFromText(p, 4326))
        # Escape output: "ST_Longitude(ST_GeomFromText('POINT(45 90)',4326))"
        # Expect result: 90

        sqlfunc.ST_Longitude(sqlfunc.ST_GeomFromText(p, 4326), 10)
        # Escape output: "ST_Longitude(ST_GeomFromText('POINT(45 90)',4326),10)"
        # Expect result: 'POINT(10 45) | 4326'
        ```
        """
        if new_longitude_val is IGNORED:
            super().__init__("ST_Longitude", 1, p)
        else:
            super().__init__("ST_Longitude", 2, p, new_longitude_val)


@cython.cclass
class ST_MakeEnvelope(SQLFunction):
    """Represents the `ST_MakeEnvelope(pt1, pt2)` function.

    MySQL description:
    - Returns the rectangle that forms the envelope around two
      points, as a Point, LineString, or Polygon.
    - Calculations are done using the Cartesian coordinate system
      rather than on a sphere, spheroid, or on earth.
    """

    def __init__(self, pt1, pt2):
        """The `ST_MakeEnvelope(pt1, pt2)` function.

        MySQL description:
        - Returns the rectangle that forms the envelope around two
          points, as a Point, LineString, or Polygon.
        - Calculations are done using the Cartesian coordinate system
          rather than on a sphere, spheroid, or on earth.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt1, pt2 = "POINT(0 0)", "POINT(1 1)"

        sqlfunc.ST_MakeEnvelope(sqlfunc.ST_GeomFromText(pt1), sqlfunc.ST_GeomFromText(pt2))
        # Escape output: "ST_MakeEnvelope(ST_GeomFromText('POINT(0 0)'),ST_GeomFromText('POINT(1 1)'))"
        # Expect result: 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'
        ```
        """
        super().__init__("ST_MakeEnvelope", 2, pt1, pt2)


@cython.cclass
class ST_MLineFromText(SQLFunction):
    """Represents the `ST_MLineFromText(wkt [, srid [, options]])` function.

    MySQL description:
    - Constructs a MultiLineString value using its WKT representation and SRID.
    """

    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_MLineFromText(wkt [, srid [, options]])` function.

        MySQL description:
        - Constructs a MultiLineString value using its WKT representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))"

        sqlfunc.ST_MLineFromText(ls)
        # Escape output: "ST_MLineFromText('MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))')"
        # Expect result: 'MULTILINESTRING((0 0, 0 5, 5 5), (1 1, 2 2))'

        sqlfunc.ST_MLineFromText(ls, 4326)
        # Escape output: "ST_MLineFromText('MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))',4326)"
        # Expect result: 'MULTILINESTRING((0 0, 5 0, 5 5), (1 1, 2 2)) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_MLineFromText", 1, wkt)
        elif options is IGNORED:
            super().__init__("ST_MLineFromText", 2, wkt, srid)
        else:
            super().__init__("ST_MLineFromText", 3, wkt, srid, options)


@cython.cclass
class ST_MLineFromWKB(SQLFunction):
    """Represents the `ST_MLineFromWKB(wkb [, srid [, options]])` function.

    MySQL description:
    - Constructs a MultiLineString value using its WKB representation and SRID.
    """

    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_MLineFromWKB(wkb [, srid [, options]])` function.

        MySQL description:
        - Constructs a MultiLineString value using its WKB representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))"

        sqlfunc.ST_MLineFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MLineFromText(ls)))
        # Escape output: "ST_MLineFromWKB(ST_AsBinary(ST_MLineFromText('MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))')))"
        # Expect result: 'MULTILINESTRING((0 0, 0 5, 5 5), (1 1, 2 2))'

        sqlfunc.ST_MLineFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MLineFromText(ls)), 4326)
        # Escape output: "ST_MLineFromWKB(ST_AsBinary(ST_MLineFromText('MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))'),4326)"
        # Expect result: 'MULTILINESTRING((0 0, 5 0, 5 5), (1 1, 2 2)) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_MLineFromWKB", 1, wkb)
        elif options is IGNORED:
            super().__init__("ST_MLineFromWKB", 2, wkb, srid)
        else:
            super().__init__("ST_MLineFromWKB", 3, wkb, srid, options)


@cython.cclass
class ST_MPointFromText(SQLFunction):
    """Represents the `ST_MPointFromText(wkt [, srid [, options]])` function.

    MySQL description:
    - Constructs a MultiPoint value using its WKT representation and SRID.
    """

    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_MPointFromText(wkt [, srid [, options]])` function.

        MySQL description:
        - Constructs a MultiPoint value using its WKT representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt = "MULTIPOINT(0 0,1 1)"

        sqlfunc.ST_MPointFromText(pt)
        # Escape output: "ST_MPointFromText('MULTIPOINT(0 0,1 1)')"
        # Expect result: 'MULTIPOINT(0 0, 1 1)'

        sqlfunc.ST_MPointFromText(pt, 4326)
        # Escape output: "ST_MPointFromText('MULTIPOINT(0 0,1 1)',4326)"
        # Expect result: 'MULTIPOINT(0 0, 1 1) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_MPointFromText", 1, wkt)
        elif options is IGNORED:
            super().__init__("ST_MPointFromText", 2, wkt, srid)
        else:
            super().__init__("ST_MPointFromText", 3, wkt, srid, options)


@cython.cclass
class ST_MPointFromWKB(SQLFunction):
    """Represents the `ST_MPointFromWKB(wkb [, srid [, options]])` function.

    MySQL description:
    - Constructs a MultiPoint value using its WKB representation and SRID.
    """

    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_MPointFromWKB(wkb [, srid [, options]])` function.

        MySQL description:
        - Constructs a MultiPoint value using its WKB representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt = "MULTIPOINT(0 0,1 1)"

        sqlfunc.ST_MPointFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MPointFromText(pt)))
        # Escape output: "ST_MPointFromWKB(ST_AsBinary(ST_MPointFromText('MULTIPOINT(0 0,1 1)')))"
        # Expect result: 'MULTIPOINT(0 0, 1 1)'

        sqlfunc.ST_MPointFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MPointFromText(pt)), 4326)
        # Escape output: "ST_MPointFromWKB(ST_AsBinary(ST_MPointFromText('MULTIPOINT(0 0,1 1)')),4326)"
        # Expect result: 'MULTIPOINT(0 0, 1 1) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_MPointFromWKB", 1, wkb)
        elif options is IGNORED:
            super().__init__("ST_MPointFromWKB", 2, wkb, srid)
        else:
            super().__init__("ST_MPointFromWKB", 3, wkb, srid, options)


@cython.cclass
class ST_MPolyFromText(SQLFunction):
    """Represents the `ST_MPolyFromText(wkt [, srid [, options]])` function.

    MySQL description:
    - Constructs a MultiPolygon value using its WKT representation and SRID.
    """

    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_MPolyFromText(wkt [, srid [, options]])` function.

        MySQL description:
        - Constructs a MultiPolygon value using its WKT representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))"

        sqlfunc.ST_MPolyFromText(poly)
        # Escape output: "ST_MPolyFromText('MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))')"
        # Expect result: 'MULTIPOLYGON(((0 0, 0 5, 5 5, 0 0)), ((1 1, 2 2, 2 1, 1 1)))'

        sqlfunc.ST_MPolyFromText(poly, 4326)
        # Escape output: "ST_MPolyFromText('MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))',4326)"
        # Expect result: 'MULTIPOLYGON(((0 0, 5 0, 5 5, 0 0)), ((1 1, 2 2, 1 2, 1 1))) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_MPolyFromText", 1, wkt)
        elif options is IGNORED:
            super().__init__("ST_MPolyFromText", 2, wkt, srid)
        else:
            super().__init__("ST_MPolyFromText", 3, wkt, srid, options)


@cython.cclass
class ST_MPolyFromWKB(SQLFunction):
    """Represents the `ST_MPolyFromWKB(wkb [, srid [, options]])` function.

    - Constructs a MultiPolygon value using its WKB representation and SRID.
    """

    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_MPolyFromWKB(wkb [, srid [, options]])` function.

        MySQL description:
        - Constructs a MultiPolygon value using its WKB representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))"

        sqlfunc.ST_MPolyFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MPolyFromText(poly)))
        # Escape output: "ST_MPolyFromWKB(ST_AsBinary(ST_MPolyFromText('MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))')))"
        # Expect result: 'MULTIPOLYGON(((0 0, 0 5, 5 5, 0 0)), ((1 1, 2 2, 2 1, 1 1)))'

        sqlfunc.ST_MPolyFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MPolyFromText(poly)), 4326)
        # Escape output: "ST_MPolyFromWKB(ST_AsBinary(ST_MPolyFromText('MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))')),4326)"
        # Expect result: 'MULTIPOLYGON(((0 0, 5 0, 5 5, 0 0)), ((1 1, 2 2, 1 2, 1 1))) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_MPolyFromWKB", 1, wkb)
        elif options is IGNORED:
            super().__init__("ST_MPolyFromWKB", 2, wkb, srid)
        else:
            super().__init__("ST_MPolyFromWKB", 3, wkb, srid, options)


@cython.cclass
class ST_NumGeometries(SQLFunction):
    """Represents the `ST_NumGeometries(gc)` function.

    MySQL description:
    - Returns the number of geometries in the GeometryCollection value gc.
    """

    def __init__(self, gc):
        """The `ST_NumGeometries(gc)` function.

        MySQL description:
        - Returns the number of geometries in the GeometryCollection value gc.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        gc = "GeometryCollection(Point(1 1),LineString(2 2, 3 3))"

        sqlfunc.ST_NumGeometries(sqlfunc.ST_GeomFromText(gc))
        # Escape output: "ST_NumGeometries(ST_GeomFromText('GeometryCollection(Point(1 1),LineString(2 2, 3 3))'))"
        # Expect result: 2
        ```
        """
        super().__init__("ST_NumGeometries", 1, gc)


@cython.cclass
class ST_NumInteriorRing(SQLFunction):
    """Represents the `ST_NumInteriorRing(poly)` function.

    MySQL description:
    - Returns the number of interior rings in the Polygon value poly.
    """

    def __init__(self, poly):
        """The `ST_NumInteriorRing(poly)` function.

        MySQL description:
        - Returns the number of interior rings in the Polygon value poly.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.ST_NumInteriorRing(sqlfunc.ST_GeomFromText(poly))
        # Escape output: "ST_NumInteriorRing(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 1
        ```
        """
        super().__init__("ST_NumInteriorRing", 1, poly)


@cython.cclass
class ST_NumPoints(SQLFunction):
    """Represents the `ST_NumPoints(ls)` function.

    MySQL description:
    - Returns the number of Point objects in the LineString value ls.
    """

    def __init__(self, ls):
        """The `ST_NumPoints(ls)` function.

        MySQL description:
        - Returns the number of Point objects in the LineString value ls.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(1 1,2 2,3 3)"

        sqlfunc.ST_NumPoints(sqlfunc.ST_GeomFromText(ls))
        # Escape output: "ST_NumPoints(ST_GeomFromText('LineString(1 1,2 2,3 3)'))"
        # Expect result: 3
        ```
        """
        super().__init__("ST_NumPoints", 1, ls)


@cython.cclass
class ST_Overlaps(SQLFunction):
    """Represents the `ST_Overlaps(g1, g2)` function.

    MySQL description:
    - Two geometries spatially overlap if they intersect and their
      intersection results in a geometry of the same dimension but
      not equal to either of the given geometries.
    - This function returns 1 or 0 to indicate whether g1 spatially
      overlaps g2.
    """

    def __init__(self, g1, g2):
        """The `ST_Overlaps(g1, g2)` function.

        MySQL description:
        - Two geometries spatially overlap if they intersect and their
          intersection results in a geometry of the same dimension but
          not equal to either of the given geometries.
        - This function returns 1 or 0 to indicate whether g1 spatially
          overlaps g2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly1, poly2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.ST_Overlaps(sqlfunc.ST_GeomFromText(poly1), sqlfunc.ST_GeomFromText(poly2))
        # Escape output: "ST_Overlaps(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 0
        ```
        """
        super().__init__("ST_Overlaps", 2, g1, g2)


@cython.cclass
class ST_PointAtDistance(SQLFunction):
    """Represents the `ST_PointAtDistance(ls, distance)` function.

    MySQL description:
    - This function takes a LineString geometry and a distance in the range
      [0.0, ST_Length(ls)] measured in the unit of the spatial reference
      system (SRS) of the LineString, and returns the Point along the LineString
      at that distance from its start point. It can be used to answer questions
      such as which Point value is 400 meters from the start of the road described
      by the geometry argument.
    - The function is implemented for LineString geometries in all spatial
      reference systems, both Cartesian and geographic.
    """

    def __init__(self, ls, distance):
        """The `ST_PointAtDistance(ls, distance)` function.

        MySQL description:
        - This function takes a LineString geometry and a distance in the range
          [0.0, ST_Length(ls)] measured in the unit of the spatial reference
          system (SRS) of the LineString, and returns the Point along the LineString
          at that distance from its start point. It can be used to answer questions
          such as which Point value is 400 meters from the start of the road described
          by the geometry argument.
        - The function is implemented for LineString geometries in all spatial
          reference systems, both Cartesian and geographic.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(0 0,0 5,5 5)"

        sqlfunc.ST_PointAtDistance(sqlfunc.ST_GeomFromText(ls), 2.5)
        # Escape output: "ST_PointAtDistance(ST_GeomFromText('LineString(0 0,0 5,5 5)'),2.5)"
        # Expect result: 'POINT(0 2.5)'
        ```
        """
        super().__init__("ST_PointAtDistance", 2, ls, distance)


@cython.cclass
class ST_PointFromGeoHash(SQLFunction):
    """Represents the `ST_PointFromGeoHash(geohash_str, srid)` function.

    MySQL description:
    - Returns a POINT value containing the decoded geohash value,
      given a geohash string value.
    - The X and Y coordinates of the point are the longitude in
      the range [-180, 180] and the latitude in the range [-90, 90],
      respectively.
    - The srid argument is an 32-bit unsigned integer.
    """

    def __init__(self, geohash_str, srid):
        """The `ST_PointFromGeoHash(geohash_str, srid)` function.

        MySQL description:
        - Returns a POINT value containing the decoded geohash value,
          given a geohash string value.
        - The X and Y coordinates of the point are the longitude in
          the range [-180, 180] and the latitude in the range [-90, 90],
          respectively.
        - The srid argument is an 32-bit unsigned integer.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.ST_PointFromGeoHash(sqlfunc.ST_GeoHash(45, -20, 10), 0)
        # Escape output: "ST_PointFromGeoHash(ST_GeoHash(45,-20,10),0)"
        # Expect result: 'POINT(45 -20)'
        ```
        """
        super().__init__("ST_PointFromGeoHash", 2, geohash_str, srid)


@cython.cclass
class ST_PointFromText(SQLFunction):
    """Represents the `ST_PointFromText(wkt [, srid [, options]])` function.

    MySQL description:
    - Constructs a Point value using its WKT representation and SRID.
    """

    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_PointFromText(wkt [, srid [, options]])` function.

        MySQL description:
        - Constructs a Point value using its WKT representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt = "POINT(1 1)"

        sqlfunc.ST_PointFromText(pt)
        # Escape output: "ST_PointFromText('POINT(1 1)')"
        # Expect result: 'POINT(1 1)'

        sqlfunc.ST_PointFromText(pt, 4326)
        # Escape output: "ST_PointFromText('POINT(1 1)',4326)"
        # Expect result: 'POINT(1 1) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_PointFromText", 1, wkt)
        elif options is IGNORED:
            super().__init__("ST_PointFromText", 2, wkt, srid)
        else:
            super().__init__("ST_PointFromText", 3, wkt, srid, options)


@cython.cclass
class ST_PointFromWKB(SQLFunction):
    """Represents the `ST_PointFromWKB(wkb [, srid [, options]])` function.

    MySQL description:
    - Constructs a Point value using its WKB representation and SRID.
    """

    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_PointFromWKB(wkb [, srid [, options]])` function.

        MySQL description:
        - Constructs a Point value using its WKB representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt = "POINT(1 1)"

        sqlfunc.ST_PointFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_PointFromText(pt)))
        # Escape output: "ST_PointFromWKB(ST_AsBinary(ST_PointFromText('POINT(1 1)')))"
        # Expect result: 'POINT(1 1)'

        sqlfunc.ST_PointFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_PointFromText(pt)), 4326)
        # Escape output: "ST_PointFromWKB(ST_AsBinary(ST_PointFromText('POINT(1 1)')),4326)"
        # Expect result: 'POINT(1 1) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_PointFromWKB", 1, wkb)
        elif options is IGNORED:
            super().__init__("ST_PointFromWKB", 2, wkb, srid)
        else:
            super().__init__("ST_PointFromWKB", 3, wkb, srid, options)


@cython.cclass
class ST_PointN(SQLFunction):
    """Represents the `ST_PointN(ls, N)` function.

    MySQL description:
    - Returns the N-th Point in the Linestring value ls.
      Points are numbered beginning with 1.
    """

    def __init__(self, ls, N):
        """The `ST_PointN(ls, N)` function.

        MySQL description:
        - Returns the N-th Point in the Linestring value ls.
          Points are numbered beginning with 1.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(1 1,2 2,3 3)"

        sqlfunc.ST_PointN(sqlfunc.ST_GeomFromText(ls), 2)
        # Escape output: "ST_PointN(ST_GeomFromText('LineString(1 1,2 2,3 3)'),2)"
        # Expect result: 'POINT(2 2)'
        ```
        """
        super().__init__("ST_PointN", 2, ls, N)


@cython.cclass
class ST_PolyFromText(SQLFunction):
    """Represents the `ST_PolyFromText(wkt [, srid [, options]])` function.

    MySQL description:
    - Constructs a Polygon value using its WKT representation and SRID.
    """

    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_PolyFromText(wkt [, srid [, options]])` function.

        MySQL description:
        - Constructs a Polygon value using its WKT representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "POLYGON((0 0,0 3,3 3,3 0,0 0))"

        sqlfunc.ST_PolyFromText(poly)
        # Escape output: "ST_PolyFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))')"
        # Expect result: 'POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))'

        sqlfunc.ST_PolyFromText(poly, 4326)
        # Escape output: "ST_PolyFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))',4326)"
        # Expect result: 'POLYGON((0 0, 3 0, 3 3, 0 3, 0 0)) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_PolyFromText", 1, wkt)
        elif options is IGNORED:
            super().__init__("ST_PolyFromText", 2, wkt, srid)
        else:
            super().__init__("ST_PolyFromText", 3, wkt, srid, options)


@cython.cclass
class ST_PolyFromWKB(SQLFunction):
    """Represents the `ST_PolyFromWKB(wkb [, srid [, options]])` function.

    MySQL description:
    - Constructs a Polygon value using its WKB representation and SRID.
    """

    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ):
        """The `ST_PolyFromWKB(wkb [, srid [, options]])` function.

        MySQL description:
        - Constructs a Polygon value using its WKB representation and SRID.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "POLYGON((0 0,0 3,3 3,3 0,0 0))"

        sqlfunc.ST_PolyFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_PolyFromText(poly)))
        # Escape output: "ST_PolyFromWKB(ST_AsBinary(ST_PolyFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))')))"
        # Expect result: 'POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))'

        sqlfunc.ST_PolyFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_PolyFromText(poly)), 4326)
        # Escape output: "ST_PolyFromWKB(ST_AsBinary(ST_PolyFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))')),4326)"
        # Expect result: 'POLYGON((0 0, 3 0, 3 3, 0 3, 0 0)) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_PolyFromWKB", 1, wkb)
        elif options is IGNORED:
            super().__init__("ST_PolyFromWKB", 2, wkb, srid)
        else:
            super().__init__("ST_PolyFromWKB", 3, wkb, srid, options)


@cython.cclass
class ST_Simplify(SQLFunction):
    """Represents the `ST_Simplify(g, max_distance)` function.

    MySQL description:
    - Simplifies a geometry using the Douglas-Peucker algorithm and returns
      a simplified value of the same type.
    - The geometry may be any geometry type, although the Douglas-Peucker algorithm
      may not actually process every type. A geometry collection is processed by
      giving its components one by one to the simplification algorithm, and the
      returned geometries are put into a geometry collection as result.
    - The max_distance argument is the distance (in units of the input coordinates)
      of a vertex to other segments to be removed. Vertices within this distance of
      the simplified linestring are removed.
    - According to Boost.Geometry, geometries might become invalid as a result of
      the simplification process, and the process might create self-intersections.
      To check the validity of the result, pass it to ST_IsValid().
    """

    def __init__(self, g, max_distance):
        """The `ST_Simplify(g, max_distance)` function.

        MySQL description:
        - Simplifies a geometry using the Douglas-Peucker algorithm and returns
          a simplified value of the same type.
        - The geometry may be any geometry type, although the Douglas-Peucker algorithm
          may not actually process every type. A geometry collection is processed by
          giving its components one by one to the simplification algorithm, and the
          returned geometries are put into a geometry collection as result.
        - The max_distance argument is the distance (in units of the input coordinates)
          of a vertex to other segments to be removed. Vertices within this distance of
          the simplified linestring are removed.
        - According to Boost.Geometry, geometries might become invalid as a result of
          the simplification process, and the process might create self-intersections.
          To check the validity of the result, pass it to ST_IsValid().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LINESTRING(0 0,0 1,1 1,1 2,2 2,2 3,3 3)"

        sqlfunc.ST_Simplify(sqlfunc.ST_GeomFromText(ls), 0.5)
        # Escape output: "ST_Simplify(ST_GeomFromText('LINESTRING(0 0,0 1,1 1,1 2,2 2,2 3,3 3)'),0.5)"
        # Expect result: 'LINESTRING(0 0, 0 1, 1 1, 2 3, 3 3)'

        sqlfunc.ST_Simplify(sqlfunc.ST_GeomFromText(ls), 1.0)
        # Escape output: "ST_Simplify(ST_GeomFromText('LINESTRING(0 0,0 1,1 1,1 2,2 2,2 3,3 3)'),1.0)"
        # Expect result: 'LINESTRING(0 0, 3 3)'
        ```
        """
        super().__init__("ST_Simplify", 2, g, max_distance)


@cython.cclass
class ST_SRID(SQLFunction):
    """Represents the `ST_SRID(g [, srid])` function.

    MySQL description:
    - With a single argument representing a valid geometry object g, ST_SRID()
      returns an integer indicating the ID of the spatial reference system (SRS)
      associated with g.
    - With the optional second argument representing a valid SRID value, ST_SRID()
      returns an object with the same type as its first argument with an SRID value
      equal to the second argument. This only sets the SRID value of the object;
      it does not perform any transformation of coordinate values.
    - ST_SRID() handles its arguments as described in the introduction to this
      section, with this exception:
    - For the single-argument syntax, ST_SRID() returns the geometry SRID even
      if it refers to an undefined SRS. An ER_SRS_NOT_FOUND error does not occur.
    """

    def __init__(self, g, srid: Any | Sentinel = IGNORED):
        """The `ST_SRID(g [, srid])` function.

        MySQL description:
        - With a single argument representing a valid geometry object g, ST_SRID()
          returns an integer indicating the ID of the spatial reference system (SRS)
          associated with g.
        - With the optional second argument representing a valid SRID value, ST_SRID()
          returns an object with the same type as its first argument with an SRID value
          equal to the second argument. This only sets the SRID value of the object;
          it does not perform any transformation of coordinate values.
        - ST_SRID() handles its arguments as described in the introduction to this
          section, with this exception:
        - For the single-argument syntax, ST_SRID() returns the geometry SRID even
          if it refers to an undefined SRS. An ER_SRS_NOT_FOUND error does not occur.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(1 1,2 2)"

        sqlfunc.ST_SRID(sqlfunc.ST_GeomFromText(ls, 0))
        # Escape output: "ST_SRID(ST_GeomFromText('LineString(1 1,2 2)',0))"
        # Expect result: 0

        sqlfunc.ST_SRID(sqlfunc.ST_GeomFromText(ls, 0), 4326)
        # Escape output: "ST_SRID(ST_GeomFromText('LineString(1 1,2 2)',0),4326)"
        # Expect result: 'LINESTRING(1 1, 2 2) | 4326'
        ```
        """
        if srid is IGNORED:
            super().__init__("ST_SRID", 1, g)
        else:
            super().__init__("ST_SRID", 2, g, srid)


@cython.cclass
class ST_StartPoint(SQLFunction):
    """Represents the `ST_StartPoint(ls)` function.

    MySQL description:
    - Returns the Point that is the start point of the LineString value ls.
    """

    def __init__(self, ls):
        """The `ST_StartPoint(ls)` function.

        MySQL description:
        - Returns the Point that is the start point of the LineString value ls.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls = "LineString(1 1,2 2,3 3)"

        sqlfunc.ST_StartPoint(sqlfunc.ST_GeomFromText(ls))
        # Escape output: "ST_StartPoint(ST_GeomFromText('LineString(1 1,2 2,3 3)'))"
        # Expect result: 'POINT(1 1)'
        ```
        """
        super().__init__("ST_StartPoint", 1, ls)


@cython.cclass
class ST_SwapXY(SQLFunction):
    """Represents the `ST_SwapXY(g)` function.

    MySQL description:
    - Accepts an argument in internal geometry format, swaps the
      X and Y values of each coordinate pair within the geometry,
      and returns the result.
    """

    def __init__(self, g):
        """The `ST_SwapXY(g)` function.

        MySQL description:
        - Accepts an argument in internal geometry format, swaps the
          X and Y values of each coordinate pair within the geometry,
          and returns the result.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt = "LINESTRING(0 5,5 10,10 15)"

        sqlfunc.ST_SwapXY(sqlfunc.ST_GeomFromText(pt))
        # Escape output: "ST_SwapXY(ST_GeomFromText('LINESTRING(0 5,5 10,10 15)'))"
        # Expect result: 'LINESTRING(5 0, 10 5, 15 10)'
        ```
        """
        super().__init__("ST_SwapXY", 1, g)


@cython.cclass
class ST_SymDifference(SQLFunction):
    """Represents the `ST_SymDifference(g1, g2)` function.

    MySQL description:
    - Returns a geometry that represents the point set symmetric difference
      of the geometry values g1 and g2.
    """

    def __init__(self, g1, g2):
        """The `ST_SymDifference(g1, g2)` function.

        MySQL description:
        - Returns a geometry that represents the point set symmetric difference
          of the geometry values g1 and g2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        multi_p1, multi_p2 = "MULTIPOINT(5 0,15 10,15 25)", "MULTIPOINT(1 1,15 10,15 25)"

        sqlfunc.ST_SymDifference(sqlfunc.ST_GeomFromText(multi_p1), sqlfunc.ST_GeomFromText(multi_p2))
        # Escape output: "ST_SymDifference(ST_GeomFromText('MULTIPOINT(5 0,15 10,15 25)'),ST_GeomFromText('MULTIPOINT(1 1,15 10,15 25)'))"
        # Expect result: 'MULTIPOINT(5 0, 1 1)'
        ```
        """
        super().__init__("ST_SymDifference", 2, g1, g2)


@cython.cclass
class ST_Touches(SQLFunction):
    """Represents the `ST_Touches(g1, g2)` function.

    MySQL description:
    - Two geometries spatially touch if their interiors do not intersect,
      but the boundary of one of the geometries intersects either the
      boundary or the interior of the other.
    - This function returns 1 or 0 to indicate whether g1 spatially touches g2.
    """

    def __init__(self, g1, g2):
        """The `ST_Touches(g1, g2)` function.

        MySQL description:
        - Two geometries spatially touch if their interiors do not intersect,
          but the boundary of one of the geometries intersects either the
          boundary or the interior of the other.
        - This function returns 1 or 0 to indicate whether g1 spatially touches g2.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly1, poly2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"

        sqlfunc.ST_Touches(sqlfunc.ST_GeomFromText(poly1), sqlfunc.ST_GeomFromText(poly2))
        # Escape output: "ST_Touches(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))"
        # Expect result: 0
        ```
        """
        super().__init__("ST_Touches", 2, g1, g2)


@cython.cclass
class ST_Transform(SQLFunction):
    """Represents the `ST_Transform(g, target_srid)` function.

    MySQL description:
    - Transforms a geometry from one spatial reference system (SRS) to another.
      The return value is a geometry of the same type as the input geometry
      with all coordinates transformed to the target SRID, target_srid.
    """

    def __init__(self, g, target_srid):
        """The `ST_Transform(g, target_srid)` function.

        MySQL description:
        - Transforms a geometry from one spatial reference system (SRS) to another.
          The return value is a geometry of the same type as the input geometry
          with all coordinates transformed to the target SRID, target_srid.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt = "POINT(1 1)"

        sqlfunc.ST_Transform(sqlfunc.ST_GeomFromText(pt, 4230), 4326)
        # Escape output: "ST_Transform(ST_GeomFromText('POINT(1 1)',4230),4326)"
        # Expect result: 'POINT(0.999461838965145 0.998543480230371) | 4326'
        ```
        """
        super().__init__("ST_Transform", 2, g, target_srid)


@cython.cclass
class ST_Union(SQLFunction):
    """Represents the `ST_Union(g1, g2)` function.

    MySQL description:
    - Returns a geometry that represents the point set union of the geometry
      values g1 and g2. The result is in the same SRS as the geometry arguments.
    """

    def __init__(self, g1, g2):
        """The `ST_Union(g1, g2)` function.

        MySQL description:
        - Returns a geometry that represents the point set union of the geometry
          values g1 and g2. The result is in the same SRS as the geometry arguments.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LineString(1 1, 3 3)", "LineString(1 3, 3 1)"

        sqlfunc.ST_Union(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls2))
        # Escape output: "ST_Union(ST_GeomFromText('LineString(1 1, 3 3)'),ST_GeomFromText('LineString(1 3, 3 1)'))"
        # Expect result: 'MULTILINESTRING((1 1, 3 3), (1 3, 3 1))'
        ```
        """
        super().__init__("ST_Union", 2, g1, g2)


@cython.cclass
class ST_Validate(SQLFunction):
    """Represents the `ST_Validate(g)` function.

    MySQL description:
    - Validates a geometry according to the OGC specification. A geometry can
      be syntactically well-formed (WKB value plus SRID) but geometrically
      invalid. For example, this polygon is geometrically invalid:
      POLYGON((0 0, 0 0, 0 0, 0 0, 0 0))
    - ST_Validate() returns the geometry if it is syntactically well-formed and
      is geometrically valid, NULL if the argument is not syntactically well-formed
      or is not geometrically valid or is NULL.
    - ST_Validate() can be used to filter out invalid geometry data, although
      at a cost. For applications that require more precise results not tainted
      by invalid data, this penalty may be worthwhile.
    - If the geometry argument is valid, it is returned as is, except that if an
      input Polygon or MultiPolygon has clockwise rings, those rings are reversed
      before checking for validity. If the geometry is valid, the value with the
      reversed rings is returned.
    - The only valid empty geometry is represented in the form of an empty geometry
      collection value. ST_Validate() returns it directly without further checks in
      this case.
    """

    def __init__(self, g):
        """The `ST_Validate(g)` function.

        MySQL description:
        - Validates a geometry according to the OGC specification. A geometry can
          be syntactically well-formed (WKB value plus SRID) but geometrically
          invalid. For example, this polygon is geometrically invalid:
          POLYGON((0 0, 0 0, 0 0, 0 0, 0 0))
        - ST_Validate() returns the geometry if it is syntactically well-formed and
          is geometrically valid, NULL if the argument is not syntactically well-formed
          or is not geometrically valid or is NULL.
        - ST_Validate() can be used to filter out invalid geometry data, although
          at a cost. For applications that require more precise results not tainted
          by invalid data, this penalty may be worthwhile.
        - If the geometry argument is valid, it is returned as is, except that if an
          input Polygon or MultiPolygon has clockwise rings, those rings are reversed
          before checking for validity. If the geometry is valid, the value with the
          reversed rings is returned.
        - The only valid empty geometry is represented in the form of an empty geometry
          collection value. ST_Validate() returns it directly without further checks in
          this case.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        ls1, ls2 = "LINESTRING(0 0)", "LINESTRING(0 0, 1 1)"

        sqlfunc.ST_Validate(sqlfunc.ST_GeomFromText(ls1))
        # Escape output: "ST_Validate(ST_GeomFromText('LINESTRING(0 0)'))"
        # Expect result: NULL

        sqlfunc.ST_Validate(sqlfunc.ST_GeomFromText(ls2))
        # Escape output: "ST_Validate(ST_GeomFromText('LINESTRING(0 0, 1 1)'))"
        # Expect result: 'LINESTRING(0 0, 1 1)'
        ```
        """
        super().__init__("ST_Validate", 1, g)


@cython.cclass
class ST_Within(SQLFunction):
    """Represents the `ST_Within(g1, g2)` function.

    MySQL description:
    - Returns 1 or 0 to indicate whether g1 is spatially within g2.
      This tests the opposite relationship as ST_Contains().
    """

    def __init__(self, g1, g2):
        """The `ST_Within(g1, g2)` function.

        MySQL description:
        - Returns 1 or 0 to indicate whether g1 is spatially within g2.
          This tests the opposite relationship as ST_Contains().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        poly = "POLYGON((0 0,0 3,3 3,3 0,0 0))"
        p1, p2, p3 = "POINT(1 1)", "POINT(3 3)", "POINT(5 5)"

        sqlfunc.ST_Within(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(poly))
        # Escape output: "ST_Within(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'))"
        # Expect result: 1

        sqlfunc.ST_Within(sqlfunc.ST_GeomFromText(p2), sqlfunc.ST_GeomFromText(poly))
        # Escape output: "ST_Within(ST_GeomFromText('POINT(3 3)'),ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'))"
        # Expect result: 0

        sqlfunc.ST_Within(sqlfunc.ST_GeomFromText(p3), sqlfunc.ST_GeomFromText(poly))
        # Escape output: "ST_Within(ST_GeomFromText('POINT(5 5)'),ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'))"
        # Expect result: 0
        ```
        """
        super().__init__("ST_Within", 2, g1, g2)


@cython.cclass
class ST_X(SQLFunction):
    """Represents the `ST_X(p [, new_x_val])` function.

    MySQL description:
    - With a single argument representing a valid Point object p, ST_X() returns
      the X-coordinate value of p as a double-precision number. The X coordinate
      is considered to refer to the axis that appears first in the Point spatial
      reference system (SRS) definition.
    - With the optional second argument, ST_X() returns a Point object like the
      first argument with its X coordinate equal to the second argument. If the
      Point object has a geographic SRS, the second argument must be in the proper
      range for longitude or latitude values.
    """

    def __init__(self, p, new_x_val: Any | Sentinel = IGNORED):
        """The `ST_X(p [, new_x_val])` function.

        MySQL description:
        - With a single argument representing a valid Point object p, ST_X() returns
          the X-coordinate value of p as a double-precision number. The X coordinate
          is considered to refer to the axis that appears first in the Point spatial
          reference system (SRS) definition.
        - With the optional second argument, ST_X() returns a Point object like the
          first argument with its X coordinate equal to the second argument. If the
          Point object has a geographic SRS, the second argument must be in the proper
          range for longitude or latitude values.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt = "POINT(56.7 53.34)"

        sqlfunc.ST_X(sqlfunc.ST_GeomFromText(pt))
        # Escape output: "ST_X(ST_GeomFromText('POINT(56.7 53.34)'))"
        # Expect result: 56.7

        sqlfunc.ST_X(sqlfunc.ST_GeomFromText(pt), 10.5)
        # Escape output: "ST_X(ST_GeomFromText('POINT(56.7 53.34)'),10.5)"
        # Expect result: 'POINT(10.5 53.34)'
        ```
        """
        if new_x_val is IGNORED:
            super().__init__("ST_X", 1, p)
        else:
            super().__init__("ST_X", 2, p, new_x_val)


@cython.cclass
class ST_Y(SQLFunction):
    """Represents the `ST_Y(p [, new_y_val])` function.

    MySQL description:
    = With a single argument representing a valid Point object p, ST_Y() returns
      the Y-coordinate value of p as a double-precision number.The Y coordinate
      is considered to refer to the axis that appears second in the Point spatial
      reference system (SRS) definition.
    - With the optional second argument, ST_Y() returns a Point object like the
      first argument with its Y coordinate equal to the second argument. If the
      Point object has a geographic SRS, the second argument must be in the proper
      range for longitude or latitude values.
    """

    def __init__(self, p, new_y_val: Any | Sentinel = IGNORED):
        """The `ST_Y(p [, new_y_val])` function.

        MySQL description:
        = With a single argument representing a valid Point object p, ST_Y() returns
          the Y-coordinate value of p as a double-precision number.The Y coordinate
          is considered to refer to the axis that appears second in the Point spatial
          reference system (SRS) definition.
        - With the optional second argument, ST_Y() returns a Point object like the
          first argument with its Y coordinate equal to the second argument. If the
          Point object has a geographic SRS, the second argument must be in the proper
          range for longitude or latitude values.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        pt = "POINT(56.7 53.34)"

        sqlfunc.ST_Y(sqlfunc.ST_GeomFromText(pt))
        # Escape output: "ST_Y(ST_GeomFromText('POINT(56.7 53.34)'))"
        # Expect result: 53.34

        sqlfunc.ST_Y(sqlfunc.ST_GeomFromText(pt), 10.5)
        # Escape output: "ST_Y(ST_GeomFromText('POINT(56.7 53.34)'),10.5)"
        # Expect result: 'POINT(56.7 10.5)'
        ```
        """
        if new_y_val is IGNORED:
            super().__init__("ST_Y", 1, p)
        else:
            super().__init__("ST_Y", 2, p, new_y_val)


@cython.cclass
class STR_TO_DATE(SQLFunction):
    """Represents the `STR_TO_DATE(string,format)` function.

    MySQL description:
    - This is the inverse of the DATE_FORMAT() function. It takes a string
      str and a format string format. STR_TO_DATE() returns a DATETIME value
      if the format string contains both date and time parts, or a DATE or
      TIME value if the string contains only date or time parts. If str or
      format is NULL, the function returns NULL. If the date, time, or datetime
      value extracted from str cannot be parsed according to the rules followed
      by the server, STR_TO_DATE() returns NULL and produces a warning.
    - The server scans str attempting to match format to it. The format string
      can contain literal characters and format specifiers beginning with %.
      Literal characters in format must match literally in str. Format specifiers
      in format must match a date or time part in str.
    """

    def __init__(self, string, format):
        """The `STR_TO_DATE(string,format)` function.

        MySQL description:
        - This is the inverse of the DATE_FORMAT() function. It takes a string
          str and a format string format. STR_TO_DATE() returns a DATETIME value
          if the format string contains both date and time parts, or a DATE or
          TIME value if the string contains only date or time parts. If str or
          format is NULL, the function returns NULL. If the date, time, or datetime
          value extracted from str cannot be parsed according to the rules followed
          by the server, STR_TO_DATE() returns NULL and produces a warning.
        - The server scans str attempting to match format to it. The format string
          can contain literal characters and format specifiers beginning with %.
          Literal characters in format must match literally in str. Format specifiers
          in format must match a date or time part in str.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.STR_TO_DATE("01,5,2013", "%d,%m,%Y")
        # Escape output: "STR_TO_DATE('01,5,2013','%d,%m,%Y')"
        # Expect result: '2013-05-01'

        sqlfunc.STR_TO_DATE("May 1, 2013", "%M %d,%Y")
        # Escape output: "STR_TO_DATE('May 1, 2013','%M %d,%Y')"
        # Expect result: '2013-05-01'
        ```
        """
        super().__init__("STR_TO_DATE", 2, string, format)


@cython.cclass
class STRCMP(SQLFunction):
    """Represents the `STRCMP(expr1,expr2)` function.

    MySQL description:
    - returns 0 if the strings are the same, -1 if the first argument is smaller
      than the second according to the current sort order, and NULL if either
      argument is NULL. It returns 1 otherwise.
    """

    def __init__(self, expr1, expr2):
        """The `STRCMP(expr1,expr2)` function.

        MySQL description:
        - returns 0 if the strings are the same, -1 if the first argument is smaller
          than the second according to the current sort order, and NULL if either
          argument is NULL. It returns 1 otherwise.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.STRCMP("text", "text2")
        # Escape output: "STRCMP('text','text2')"
        # Expect result: -1

        sqlfunc.STRCMP("text2", "text")
        # Escape output: "STRCMP('text2','text')"
        # Expect result: 1

        sqlfunc.STRCMP("text", "text")
        # Escape output: "STRCMP('text','text')"
        # Expect result: 0
        ```
        """
        super().__init__("STRCMP", 2, expr1, expr2)


@cython.cclass
class SUBDATE(SQLFunction):
    """Represents the `SUBDATE(date, INTERVAL expr unit)` function.

    MySQL description:
    - When invoked with the INTERVAL form of the second argument,
      SUBDATE() is a synonym for DATE_SUB().
    - When invoked with the days form of the second argument, MySQL
      treats it as an integer number of days to be subtracted from expr.
    """

    def __init__(self, date, expr):
        """The `SUBDATE(date, INTERVAL expr unit)` function.

        MySQL description:
        - When invoked with the INTERVAL form of the second argument,
          SUBDATE() is a synonym for DATE_SUB().
        - When invoked with the days form of the second argument, MySQL
          treats it as an integer number of days to be subtracted from expr.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc, sqlintvl

        sqlfunc.SUBDATE("2008-01-02", 31)
        # Escape output: "SUBDATE('2008-01-02',31)"
        # Expect result: '2007-12-02'

        sqlfunc.SUBDATE(datetime.date(2008, 1, 2), sqlintvl.DAY(31))
        # Escape output: "SUBDATE('2008-01-02',INTERVAL 31 DAY)"
        # Expect result: '2007-12-02'

        sqlfunc.SUBDATE(datetime.datetime(2008, 1, 2), sqlintvl.DAY(31))
        # Escape output: "SUBDATE('2008-01-02 00:00:00',INTERVAL 31 DAY)"
        # Expect result: '2007-12-02'
        ```
        """
        super().__init__("SUBDATE", 2, date, expr)


@cython.cclass
class SUBSTRING(SQLFunction):
    """Represents the `SUBSTRING(string, pos [, length])` function.

    MySQL description:
    - The forms without a len argument return a substring from string str starting
      at position pos. The forms with a len argument return a substring len characters
      long from string str, starting at position pos. The forms that use FROM are
      standard SQL syntax.
    - It is also possible to use a negative value for pos. In this case, the beginning
      of the substring is pos characters from the end of the string, rather than the
      beginning. A negative value may be used for pos in any of the forms of this
      function. A value of 0 for pos returns an empty string.
    """

    def __init__(self, string, pos, length: Any | Sentinel = IGNORED):
        """The `SUBSTRING(string, pos [, length])` function.

        MySQL description:
        - The forms without a len argument return a substring from string str starting
          at position pos. The forms with a len argument return a substring len characters
          long from string str, starting at position pos. The forms that use FROM are
          standard SQL syntax.
        - It is also possible to use a negative value for pos. In this case, the beginning
          of the substring is pos characters from the end of the string, rather than the
          beginning. A negative value may be used for pos in any of the forms of this
          function. A value of 0 for pos returns an empty string.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SUBSTRING("Quadratically", 5)
        # Escape output: "SUBSTRING('Quadratically',5)"
        # Expect result: 'ratically'

        sqlfunc.SUBSTRING("Quadratically", 5, 3)
        # Escape output: "SUBSTRING('Quadratically',5,3)"
        # Expect result: 'rat'
        ```
        """
        if length is IGNORED:
            super().__init__("SUBSTRING", 2, string, pos)
        else:
            super().__init__("SUBSTRING", 3, string, pos, length)


@cython.cclass
class SUBSTRING_INDEX(SQLFunction):
    """Represents the `SUBSTRING_INDEX(str,delim,count)` function.

    MySQL description:
    - Returns the substring from string str before count occurrences of the delimiter
      delim. If count is positive, everything to the left of the final delimiter
      (counting from the left) is returned. If count is negative, everything to the
      right of the final delimiter (counting from the right) is returned.
    - SUBSTRING_INDEX() performs a case-sensitive match when searching for delim.
    """

    def __init__(self, string, delim, count):
        """The `SUBSTRING_INDEX(str,delim,count)` function.

        MySQL description:
        - Returns the substring from string str before count occurrences of the delimiter
          delim. If count is positive, everything to the left of the final delimiter
          (counting from the left) is returned. If count is negative, everything to the
          right of the final delimiter (counting from the right) is returned.
        - SUBSTRING_INDEX() performs a case-sensitive match when searching for delim.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SUBSTRING_INDEX("www.mysql.com", ".", 2)
        # Escape output: "SUBSTRING_INDEX('www.mysql.com','.',2)"
        # Expect result: 'www.mysql'

        sqlfunc.SUBSTRING_INDEX("www.mysql.com", ".", -2)
        # Escape output: "SUBSTRING_INDEX('www.mysql.com','.',-2)"
        # Expect result: 'mysql.com'
        ```
        """
        super().__init__("SUBSTRING_INDEX", 3, string, delim, count)


@cython.cclass
class SUBTIME(SQLFunction):
    """Represents the `SUBTIME(expr1,expr2)` function.

    MySQL description:
    - Returns expr1 - expr2 expressed as a value in the same format as expr1.
    - expr1 is a time or datetime expression, and expr2 is a time expression.
    """

    def __init__(self, expr1, expr2):
        """The `SUBTIME(expr1,expr2)` function.

        MySQL description:
        - Returns expr1 - expr2 expressed as a value in the same format as expr1.
        - expr1 is a time or datetime expression, and expr2 is a time expression.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.SUBTIME("2007-12-31 23:59:59.999999", "1 1:1:1.000002")
        # Escape output: "SUBTIME('2007-12-31 23:59:59.999999','1 1:1:1.000002')"
        # Expect result: '2007-12-30 22:58:58.999997'

        sqlfunc.SUBTIME("01:00:00.999999", "02:00:00.999998")
        # Escape output: "SUBTIME('01:00:00.999999','02:00:00.999998')"
        # Expect result: '-00:59:59.999999'

        sqlfunc.SUBTIME(datetime.time(1, 0, 0, 999999), "02:00:00.999998")
        # Escape output: "SUBTIME('01:00:00.999999','02:00:00.999998')"
        # Expect result: '-00:59:59.999999'
        ```
        """
        super().__init__("SUBTIME", 2, expr1, expr2)


@cython.cclass
class SYSDATE(SQLFunction):
    """Represents the `SYSDATE([fsp])` function.

    MySQL description:
    - Returns the current date and time as a value in 'YYYY-MM-DD hh:mm:ss'
      or YYYYMMDDhhmmss format, depending on whether the function is used
      in string or numeric context.
    - If the fsp argument is given to specify a fractional seconds precision
      from 0 to 6, the return value includes a fractional seconds part of
      that many digits.
    - SYSDATE() returns the time at which it executes. This differs from
      the behavior for NOW(), which returns a constant time that indicates
      the time at which the statement began to execute. (Within a stored
      function or trigger, NOW() returns the time at which the function
      or triggering statement began to execute.)
    """

    def __init__(self, fsp: Any | Sentinel = IGNORED):
        """The `SYSDATE([fsp])` function.

        MySQL description:
        - Returns the current date and time as a value in 'YYYY-MM-DD hh:mm:ss'
          or YYYYMMDDhhmmss format, depending on whether the function is used
          in string or numeric context.
        - If the fsp argument is given to specify a fractional seconds precision
          from 0 to 6, the return value includes a fractional seconds part of
          that many digits.
        - SYSDATE() returns the time at which it executes. This differs from
          the behavior for NOW(), which returns a constant time that indicates
          the time at which the statement began to execute. (Within a stored
          function or trigger, NOW() returns the time at which the function
          or triggering statement began to execute.)

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.SYSDATE()
        # Escape output: "SYSDATE()"
        # Expect result: '2025-01-02 15:42:43'

        sqlfunc.SYSDATE(3)
        # Escape output: "SYSDATE(3)"
        # Expect result: '2025-01-02 15:42:43.390'
        ```
        """
        if fsp is IGNORED:
            super().__init__("SYSDATE", 0)
        else:
            super().__init__("SYSDATE", 1, fsp)


# Functions: T -------------------------------------------------------------------------------------------------------
@cython.cclass
class TAN(SQLFunction):
    """Represents the `TAN(X)` function.

    MySQL description:
    - Returns the tangent of X, where X is given in radians.
    - Returns NULL if X is NULL.
    """

    def __init__(self, X):
        """The `TAN(X)` function.

        MySQL description:
        - Returns the tangent of X, where X is given in radians.
        - Returns NULL if X is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.TAN(sqlfunc.PI())
        # Escape output: "TAN(PI())"
        # Expect result: -1.2246467991473532e-16
        """
        super().__init__("TAN", 1, X)


@cython.cclass
class TIME(SQLFunction):
    """Represents the `TIME(expr)` function.

    MySQL description:
    - Extracts the time part of the time or datetime expression expr
      and returns it as a string. Returns NULL if expr is NULL.
    - This function is unsafe for statement-based replication. A warning
      is logged if you use this function when binlog_format is set to STATEMENT.
    """

    def __init__(self, expr):
        """The `TIME(expr)` function.

        MySQL description:
        - Extracts the time part of the time or datetime expression expr
          and returns it as a string. Returns NULL if expr is NULL.
        - This function is unsafe for statement-based replication. A warning
          is logged if you use this function when binlog_format is set to STATEMENT.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.TIME("2003-12-31 01:02:03")
        # Escape output: "TIME('2003-12-31 01:02:03')"
        # Expect result: '01:02:03'

        sqlfunc.TIME(datetime.datetime(2003, 12, 31, 1, 2, 3, 123))
        # Escape output: "TIME('2003-12-31 01:02:03.000123')"
        # Expect result: '01:02:03.000123'
        ```
        """
        super().__init__("TIME", 1, expr)


@cython.cclass
class TIME_FORMAT(SQLFunction):
    """Represents the `TIME_FORMAT(time, format)` function.

    MySQL description:
    - This is used like the DATE_FORMAT() function, but the format string
      may contain format specifiers only for hours, minutes, seconds, and
      microseconds. Other specifiers produce a NULL or 0. TIME_FORMAT()
      returns NULL if time or format is NULL.
    - If the time value contains an hour part that is greater than 23, the
      %H and %k hour format specifiers produce a value larger than the usual
      range of 0..23. The other hour format specifiers produce the hour value
      modulo 12.
    """

    def __init__(self, time, format):
        """The `TIME_FORMAT(time, format)` function.

        MySQL description:
        - This is used like the DATE_FORMAT() function, but the format string
          may contain format specifiers only for hours, minutes, seconds, and
          microseconds. Other specifiers produce a NULL or 0. TIME_FORMAT()
          returns NULL if time or format is NULL.
        - If the time value contains an hour part that is greater than 23, the
          %H and %k hour format specifiers produce a value larger than the usual
          range of 0..23. The other hour format specifiers produce the hour value
          modulo 12.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.TIME_FORMAT("100:00:00", "%H %k %h %I %l")
        # Escape output: "TIME_FORMAT('100:00:00','%H %k %h %I %l')"
        # Expect result: '100 100 04 04 4'

        sqlfunc.TIME_FORMAT(datetime.time(10, 0, 0), "%H %k %h %I %l")
        # Escape output: "TIME_FORMAT('10:00:00','%H %k %h %I %l')"
        # Expect result: '10 10 10 10 10'
        ```
        """
        super().__init__("TIME_FORMAT", 2, time, format)


@cython.cclass
class TIME_TO_SEC(SQLFunction):
    """Represents the `TIME_TO_SEC(time)` function.

    MySQL description:
    - Returns the time argument, converted to seconds.
    - Returns NULL if time is NULL.
    """

    def __init__(self, time):
        """The `TIME_TO_SEC(time)` function.

        MySQL description:
        - Returns the time argument, converted to seconds.
        - Returns NULL if time is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.TIME_TO_SEC("22:23:00")
        # Escape output: "TIME_TO_SEC('22:23:00')"
        # Expect result: 80580

        sqlfunc.TIME_TO_SEC(datetime.time(00, 39, 38))
        # Escape output: "TIME_TO_SEC('00:39:38')"
        # Expect result: 2378
        ```
        """
        super().__init__("TIME_TO_SEC", 1, time)


@cython.cclass
class TIMEDIFF(SQLFunction):
    """Represents the `TIMEDIFF(expr1, expr2)` function.

    MySQL description:
    - TIMEDIFF() returns expr1 - expr2 expressed as a time value. expr1 and
      expr2 are strings which are converted to TIME or DATETIME expressions;
      these must be of the same type following conversion. Returns NULL if
      expr1 or expr2 is NULL.
    - The result returned by TIMEDIFF() is limited to the range allowed
      for TIME values. Alternatively, you can use either of the functions
      TIMESTAMPDIFF() and UNIX_TIMESTAMP(), both of which return integers.
    """

    def __init__(self, expr1, expr2):
        """The `TIMEDIFF(expr1, expr2)` function.

        MySQL description:
        - TIMEDIFF() returns expr1 - expr2 expressed as a time value. expr1 and
          expr2 are strings which are converted to TIME or DATETIME expressions;
          these must be of the same type following conversion. Returns NULL if
          expr1 or expr2 is NULL.
        - The result returned by TIMEDIFF() is limited to the range allowed
          for TIME values. Alternatively, you can use either of the functions
          TIMESTAMPDIFF() and UNIX_TIMESTAMP(), both of which return integers.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.TIMEDIFF("2000-01-01 00:00:00", "2000-01-01 00:00:00.000001")
        # Escape output: "TIMEDIFF('2000-01-01 00:00:00','2000-01-01 00:00:00.000001')"
        # Expect result: '-00:00:00.000001'

        sqlfunc.TIMEDIFF(datetime.datetime(2008, 12, 31, 23, 59, 59, 1), datetime.datetime(2008, 12, 30, 1, 1, 1, 2))
        # Escape output: "TIMEDIFF('2008-12-31 23:59:59.000001','2008-12-30 01:01:01.000002')"
        # Expect result: '46:58:57.999999'
        ```
        """
        super().__init__("TIMEDIFF", 2, expr1, expr2)


@cython.cclass
class TIMESTAMP(SQLFunction):
    """Represents the `TIMESTAMP(expr1[, expr2])` function.

    MySQL description:
    - With a single argument, this function returns the date or datetime
      expression expr as a datetime value.
    - With two arguments, it adds the time expression expr2 to the date
      or datetime expression expr1 and returns the result as a datetime
      value.
    - Returns NULL if expr, expr1, or expr2 is NULL.
    """

    def __init__(self, expr1, expr2: Any | Sentinel = IGNORED):
        """The `TIMESTAMP(expr1[, expr2])` function.

        MySQL description:
        - With a single argument, this function returns the date or datetime
          expression expr as a datetime value.
        - With two arguments, it adds the time expression expr2 to the date
          or datetime expression expr1 and returns the result as a datetime
          value.
        - Returns NULL if expr, expr1, or expr2 is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.TIMESTAMP("2003-12-31")
        # Escape output: "TIMESTAMP('2003-12-31')"
        # Expect result: '2003-12-31 00:00:00'

        sqlfunc.TIMESTAMP(datetime.datetime(2003, 12, 31, 12), "12:00:00")
        # Escape output: "TIMESTAMP('2003-12-31 12:00:00','12:00:00')"
        # Expect result: '2004-01-01 00:00:00'
        ```
        """
        if expr2 is IGNORED:
            super().__init__("TIMESTAMP", 1, expr1)
        else:
            super().__init__("TIMESTAMP", 2, expr1, expr2)


@cython.cclass
class TIMESTAMPADD(SQLFunction):
    """Represents the `TIMESTAMPADD(unit,interval,datetime_expr)` function.

    MySQL description:
    - Adds the integer expression interval to the date or datetime expression
      datetime_expr. The unit for interval is given by the unit argument, which
      should be one of the following values: MICROSECOND (microseconds), SECOND,
      MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, or YEAR.
    - The unit value may be specified using one of keywords as shown, or with a
      prefix of SQL_TSI_. For example, DAY and SQL_TSI_DAY both are legal.
    - This function returns NULL if interval or datetime_expr is NULL.
    """

    def __init__(self, unit, interval, datetime_expr):
        """The `TIMESTAMPADD(unit,interval,datetime_expr)` function.

        MySQL description:
        - Adds the integer expression interval to the date or datetime expression
          datetime_expr. The unit for interval is given by the unit argument, which
          should be one of the following values: MICROSECOND (microseconds), SECOND,
          MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, or YEAR.
        - The unit value may be specified using one of keywords as shown, or with a
          prefix of SQL_TSI_. For example, DAY and SQL_TSI_DAY both are legal.
        - This function returns NULL if interval or datetime_expr is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc, sqlintvl

        sqlfunc.TIMESTAMPADD("MINUTE", 1, "2003-01-02")
        # Escape output: "TIMESTAMPADD(MINUTE,1,'2003-01-02')"
        # Expect result: '2003-01-02 00:01:00'

        sqlfunc.TIMESTAMPADD(sqlintvl.WEEK, 1, datetime.datetime(2003, 1, 2))
        # Escape output: "TIMESTAMPADD(WEEK,1,'2003-01-02 00:00:00')"
        # Expect result: '2003-01-09 00:00:00'
        ```
        """
        unit = RawText(_validate_interval_unit(unit, "TIMESTAMPADD"))
        super().__init__("TIMESTAMPADD", 3, unit, interval, datetime_expr)


@cython.cclass
class TIMESTAMPDIFF(SQLFunction):
    """Represents the `TIMESTAMPDIFF(unit,datetime_expr1,datetime_expr2)` function.

    MySQL description:
    - Returns datetime_expr2 - datetime_expr1, where datetime_expr1 and datetime_expr2
      are date or datetime expressions. One expression may be a date and the other a
      datetime; a date value is treated as a datetime having the time part '00:00:00'
      where necessary. The unit for the result (an integer) is given by the unit argument.
      The legal values for unit are the same as those listed in the description of the
      TIMESTAMPADD() function.
    - This function returns NULL if datetime_expr1 or datetime_expr2 is NULL.
    """

    def __init__(self, unit, datetime_expr1, datetime_expr2):
        """The `TIMESTAMPDIFF(unit,datetime_expr1,datetime_expr2)` function.

        MySQL description:
        - Returns datetime_expr2 - datetime_expr1, where datetime_expr1 and datetime_expr2
          are date or datetime expressions. One expression may be a date and the other a
          datetime; a date value is treated as a datetime having the time part '00:00:00'
          where necessary. The unit for the result (an integer) is given by the unit argument.
          The legal values for unit are the same as those listed in the description of the
          TIMESTAMPADD() function.
        - This function returns NULL if datetime_expr1 or datetime_expr2 is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc, sqlintvl

        sqlfunc.TIMESTAMPDIFF("MONTH", "2003-02-01", "2003-05-01")
        # Escape output: "TIMESTAMPDIFF(MONTH,'2003-02-01','2003-05-01')"
        # Expect result: 3

        sqlfunc.TIMESTAMPDIFF(sqlintvl.YEAR, datetime.date(2002, 5, 1), "2001-01-01")
        # Escape output: "TIMESTAMPDIFF(YEAR,'2002-05-01','2001-01-01')"
        # Expect result: -1

        sqlfunc.TIMESTAMPDIFF(sqlintvl.MINUTE, "2003-02-01", datetime.datetime(2003, 5, 1, 12, 5, 55))
        # Escape output: "TIMESTAMPDIFF(MINUTE,'2003-02-01','2003-05-01 12:05:55')"
        # Expect result: 128885
        ```
        """
        unit = RawText(_validate_interval_unit(unit, "TIMESTAMPDIFF"))
        super().__init__("TIMESTAMPDIFF", 3, unit, datetime_expr1, datetime_expr2)


@cython.cclass
class TO_DAYS(SQLFunction):
    """Represents the `TO_DAYS(date)` function.

    MySQL description:
    - Given a date date, returns a day number (the number of days since year 0).
      Returns NULL if date is NULL.
    - TO_DAYS() is not intended for use with values that precede the advent of
      the Gregorian calendar (1582), because it does not take into account the
      days that were lost when the calendar was changed. For dates before 1582
      (and possibly a later year in other locales), results from this function
      are not reliable.
    """

    def __init__(self, date):
        """The `TO_DAYS(date)` function.

        MySQL description:
        - Given a date date, returns a day number (the number of days since year 0).
          Returns NULL if date is NULL.
        - TO_DAYS() is not intended for use with values that precede the advent of
          the Gregorian calendar (1582), because it does not take into account the
          days that were lost when the calendar was changed. For dates before 1582
          (and possibly a later year in other locales), results from this function
          are not reliable.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.TO_DAYS(950501)
        # Escape output: "TO_DAYS(950501)"
        # Expect result: 728779

        sqlfunc.TO_DAYS("2007-10-07")
        # Escape output: "TO_DAYS('2007-10-07')"
        # Expect result: 733321

        sqlfunc.TO_DAYS(datetime.datetime(2007, 10, 7, 0, 0, 0))
        # Escape output: "TO_DAYS('2007-10-07 00:00:00')"
        # Expect result: 733321
        """
        super().__init__("TO_DAYS", 1, date)


@cython.cclass
class TO_SECONDS(SQLFunction):
    """Represents the `TO_SECONDS(expr)` function.

    MySQL description:
    - Given a date or datetime expr, returns the number of seconds since the year 0.
      If expr is not a valid date or datetime value (including NULL), it returns NULL.
    - TO_SECONDS() is not intended for use with values that precede the advent of the
      Gregorian calendar (1582), because it does not take into account the days that
      were lost when the calendar was changed. For dates before 1582 (and possibly a
      later year in other locales), results from this function are not reliable.
    """

    def __init__(self, expr):
        """The `TO_SECONDS(expr)` function.

        MySQL description:
        - Given a date or datetime expr, returns the number of seconds since the year 0.
          If expr is not a valid date or datetime value (including NULL), it returns NULL.
        - TO_SECONDS() is not intended for use with values that precede the advent of the
          Gregorian calendar (1582), because it does not take into account the days that
          were lost when the calendar was changed. For dates before 1582 (and possibly a
          later year in other locales), results from this function are not reliable.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.TO_SECONDS(950501)
        # Escape output: "TO_SECONDS(950501)"
        # Expect result: 62966505600

        sqlfunc.TO_SECONDS("2009-11-29")
        # Escape output: "TO_SECONDS('2009-11-29')"
        # Expect result: 63426672000

        sqlfunc.TO_SECONDS(datetime.datetime(2009, 11, 29, 13, 43, 32))
        # Escape output: "TO_SECONDS('2009-11-29 13:43:32')"
        # Expect result: 63426721412
        """
        super().__init__("TO_SECONDS", 1, expr)


@cython.cclass
class TRIM(SQLFunction):
    """Represents the `TRIM([{BOTH | LEADING | TRAILING} [remstr] FROM] string)` function.

    MySQL description:
    - Returns the string str with all remstr prefixes or suffixes removed. If none of
      the specifiers BOTH, LEADING, or TRAILING is given, BOTH is assumed. remstr is
      optional and, if not specified, spaces are removed.
    - This function is multibyte safe. It returns NULL if any of its arguments are NULL.
    """

    def __init__(
        self,
        string,
        remstr: Any | Sentinel = IGNORED,
        mode: str | Sentinel = IGNORED,
    ):
        """The `TRIM([{BOTH | LEADING | TRAILING} [remstr] FROM] string)` function.

        MySQL description:
        - Returns the string str with all remstr prefixes or suffixes removed. If none of
          the specifiers BOTH, LEADING, or TRAILING is given, BOTH is assumed. remstr is
          optional and, if not specified, spaces are removed.
        - This function is multibyte safe. It returns NULL if any of its arguments are NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.TRIM("  bar   ")
        # Escape output: "TRIM('  bar   ')"
        # Expect result: 'bar'

        sqlfunc.TRIM("xxxbarxxx", "x")
        # Escape output: "TRIM('x' FROM 'xxxbarxxx')"
        # Expect result: 'bar'

        sqlfunc.TRIM("xxxbarxxx", "x", "LEADING")
        # Escape output: "TRIM(LEADING 'x' FROM 'xxxbarxxx')"
        # Expect result: 'barxxx'
        ```
        """
        if remstr is IGNORED:
            super().__init__("TRIM", 1, string)
        elif mode is IGNORED:
            super().__init__("TRIM", 3, remstr, RawText("FROM"), string, sep=" ")
        elif isinstance(mode, str):
            _mode: str = mode
            if not set_contains(TRIM_MODES, _mode):
                _mode = _mode.upper()
                if not set_contains(TRIM_MODES, _mode):
                    raise errors.SQLFunctionError(
                        "SQL function TRIM 'mode' argument is invalid '%s'." % mode
                    )
            super().__init__(
                "TRIM", 4, RawText(_mode), remstr, RawText("FROM"), string, sep=" "
            )
        else:
            raise errors.SQLFunctionError(
                "SQL function TRIM 'mode' expects `<'str'>` type, "
                "instead got %s '%r'." % (type(mode).__name__, mode)
            )


@cython.cclass
class TRUNCATE(SQLFunction):
    """Represents the `TRUNCATE(X,D)` function.

    MySQL description:
    - Returns the number X, truncated to D decimal places. If D is 0,
      the result has no decimal point or fractional part. D can be
      negative to cause D digits left of the decimal point of the value
      X to become zero. If X or D is NULL, the function returns NULL.
    - All numbers are rounded toward zero. The data type returned by
      TRUNCATE() follows the same rules that determine the return type
      of the ROUND() function; for details, see the description for ROUND().
    """

    def __init__(self, X, D):
        """The `TRUNCATE(X,D)` function.

        MySQL description:
        - Returns the number X, truncated to D decimal places. If D is 0,
          the result has no decimal point or fractional part. D can be
          negative to cause D digits left of the decimal point of the value
          X to become zero. If X or D is NULL, the function returns NULL.
        - All numbers are rounded toward zero. The data type returned by
          TRUNCATE() follows the same rules that determine the return type
          of the ROUND() function; for details, see the description for ROUND().

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.TRUNCATE(1.223, 1)
        # Escape output: "TRUNCATE(1.223,1)"
        # Expect result: 1.2

        sqlfunc.TRUNCATE(1.999, 1)
        # Escape output: "TRUNCATE(1.999,1)"
        # Expect result: 1.9

        sqlfunc.TRUNCATE(122, -2)
        # Escape output: "TRUNCATE(122,-2)"
        # Expect result: 100

        sqlfunc.TRUNCATE("10.28", 0)
        # Escape output: "TRUNCATE('10.28',0)"
        # Expect result: 10
        """
        super().__init__("TRUNCATE", 2, X, D)


# Functions: U -------------------------------------------------------------------------------------------------------
@cython.cclass
class UNCOMPRESS(SQLFunction):
    """Represents the `UNCOMPRESS(string)` function.

    MySQL description:
    - Uncompresses a string compressed by the COMPRESS() function. If the argument
      is not a compressed value, the result is NULL; if string_to_uncompress is NULL,
      the result is also NULL. This function requires MySQL to have been compiled with
      a compression library such as zlib.
    - Return value is always NULL.
    """

    def __init__(self, string):
        """The `UNCOMPRESS(string)` function.

        MySQL description:
        - Uncompresses a string compressed by the COMPRESS() function. If the argument
          is not a compressed value, the result is NULL; if string_to_uncompress is NULL,
          the result is also NULL. This function requires MySQL to have been compiled with
          a compression library such as zlib.
        - Return value is always NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UNCOMPRESS(sqlfunc.COMPRESS("any string"))
        # Escape output: "UNCOMPRESS(COMPRESS('any string'))"
        # Expect result: 'any string'

        sqlfunc.UNCOMPRESS("any string")
        # Escape output: "UNCOMPRESS('any string')"
        # Expect result: NULL
        """
        super().__init__("UNCOMPRESS", 1, string)


@cython.cclass
class UNCOMPRESSED_LENGTH(SQLFunction):
    """Represents the `UNCOMPRESSED_LENGTH(compressed_string)` function.

    MySQL description:
    - Returns the length that the compressed string had before being compressed.
    - Returns NULL if compressed_string is NULL.
    """

    def __init__(self, compressed_string):
        """The `UNCOMPRESSED_LENGTH(compressed_string)` function.

        MySQL description:
        - Returns the length that the compressed string had before being compressed.
        - Returns NULL if compressed_string is NULL.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UNCOMPRESSED_LENGTH(sqlfunc.COMPRESS(sqlfunc.REPEAT('a',30)))
        # Escape output: "UNCOMPRESSED_LENGTH(COMPRESS(REPEAT('a',30)))"
        # Expect result: 30
        """
        super().__init__("UNCOMPRESSED_LENGTH", 1, compressed_string)


@cython.cclass
class UNHEX(SQLFunction):
    """Represents the `UNHEX(string)` function.

    MySQL description:
    - For a string argument str, UNHEX(string) interprets each pair
      of characters in the argument as a hexadecimal number and
      converts it to the byte represented by the number.
    - The return value is a binary string.
    """

    def __init__(self, string):
        """The `UNHEX(string)` function.

        MySQL description:
        - For a string argument str, UNHEX(string) interprets each pair
          of characters in the argument as a hexadecimal number and
          converts it to the byte represented by the number.
        - The return value is a binary string.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UNHEX("4D7953514C")
        # Escape output: "UNHEX('4D7953514C')"
        # Expect result: 'MySQL'

        sqlfunc.UNHEX(sqlfunc.HEX("string"))
        # Escape output: "UNHEX(HEX('string'))"
        # Expect result: 'string'

        sqlfunc.HEX(sqlfunc.UNHEX("1267"))
        # Escape output: "HEX(UNHEX('1267'))"
        # Expect result: '1267'
        ```
        """
        super().__init__("UNHEX", 1, string)


@cython.cclass
class UNIX_TIMESTAMP(SQLFunction):
    """Represents the `UNIX_TIMESTAMP([date])` function.

    MySQL description:
    - If UNIX_TIMESTAMP() is called with no date argument, it returns a Unix
      timestamp representing seconds since '1970-01-01 00:00:00' UTC.
    - If UNIX_TIMESTAMP() is called with a date argument, it returns the value
      of the argument as seconds since '1970-01-01 00:00:00' UTC. The server
      interprets date as a value in the session time zone and converts it to
      an internal Unix timestamp value in UTC. (Clients can set the session
      time zone as described in Section 7.1.15, “MySQL Server Time Zone Support”.)
      The date argument may be a DATE, DATETIME, or TIMESTAMP string, or a number
      in YYMMDD, YYMMDDhhmmss, YYYYMMDD, or YYYYMMDDhhmmss format. If the argument
      includes a time part, it may optionally include a fractional seconds part.
    - The return value is an integer if no argument is given or the argument does
      not include a fractional seconds part, or DECIMAL if an argument is given
      that includes a fractional seconds part.
    """

    def __init__(self, date: Any | Sentinel = IGNORED):
        """The `UNIX_TIMESTAMP([date])` function.

        MySQL description:
        - If UNIX_TIMESTAMP() is called with no date argument, it returns a Unix
          timestamp representing seconds since '1970-01-01 00:00:00' UTC.
        - If UNIX_TIMESTAMP() is called with a date argument, it returns the value
          of the argument as seconds since '1970-01-01 00:00:00' UTC. The server
          interprets date as a value in the session time zone and converts it to
          an internal Unix timestamp value in UTC. (Clients can set the session
          time zone as described in Section 7.1.15, “MySQL Server Time Zone Support”.)
          The date argument may be a DATE, DATETIME, or TIMESTAMP string, or a number
          in YYMMDD, YYMMDDhhmmss, YYYYMMDD, or YYYYMMDDhhmmss format. If the argument
          includes a time part, it may optionally include a fractional seconds part.
        - The return value is an integer if no argument is given or the argument does
          not include a fractional seconds part, or DECIMAL if an argument is given
          that includes a fractional seconds part.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.UNIX_TIMESTAMP()
        # Escape output: "UNIX_TIMESTAMP()"
        # Expect result: 1735680000

        sqlfunc.UNIX_TIMESTAMP("2015-11-13 10:20:19")
        # Escape output: "UNIX_TIMESTAMP('2015-11-13 10:20:19')"
        # Expect result: 1447381219

        sqlfunc.UNIX_TIMESTAMP(datetime.datetime(2015, 11, 13, 10, 20, 19, 12000))
        # Escape output: "UNIX_TIMESTAMP('2015-11-13 10:20:19.012000')"
        # Expect result: 1447381219.012000
        ```
        """
        if date is IGNORED:
            super().__init__("UNIX_TIMESTAMP", 0)
        else:
            super().__init__("UNIX_TIMESTAMP", 1, date)


@cython.cclass
class UPPER(SQLFunction):
    """Represents the `UPPER(string)` function.

    MySQL description:
    - Returns the string str with all characters changed to uppercase
      according to the current character set mapping.
    - Returns NULL if str is NULL. The default character set is utf8mb4.
    """

    def __init__(self, string):
        """The `UPPER(string)` function.

        MySQL description:
        - Returns the string str with all characters changed to uppercase
          according to the current character set mapping.
        - Returns NULL if str is NULL. The default character set is utf8mb4.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UPPER("hello")
        # Escape output: "UPPER('hello')"
        # Expect result: 'HELLO'
        ```
        """
        super().__init__("UPPER", 1, string)


@cython.cclass
class USER(SQLFunction):
    """Represents the `USER()` function.

    MySQL description:
    - Returns the current MySQL user name and host name as a string
      in the utf8mb3 character set.
    """

    def __init__(self):
        """The `USER()` function.

        MySQL description:
        - Returns the current MySQL user name and host name as a string
          in the utf8mb3 character set.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.USER()
        # Escape output: "USER()"
        # Expect result: 'root@localhost'
        """
        super().__init__("USER", 0)


@cython.cclass
class UTC_DATE(SQLFunction):
    """Represents the `UTC_DATE()` function.

    MySQL description:
    - Returns the current UTC date as a value in 'YYYY-MM-DD' or YYYYMMDD format,
      depending on whether the function is used in string or numeric context.
    """

    def __init__(self):
        """The `UTC_DATE()` function.

        MySQL description:
        - Returns the current UTC date as a value in 'YYYY-MM-DD' or YYYYMMDD format,
          depending on whether the function is used in string or numeric context.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UTC_DATE()
        # Escape output: "UTC_DATE()"
        # Expect result: '2023-11-13'
        ```
        """
        super().__init__("UTC_DATE", 0)


@cython.cclass
class UTC_TIME(SQLFunction):
    """Represents the `UTC_TIME([fsp])` function.

    MySQL description:
    - Returns the current UTC time as a value in 'hh:mm:ss' or hhmmss format,
      depending on whether the function is used in string or numeric context.
    - If the fsp argument is given to specify a fractional seconds precision
      from 0 to 6, the return value includes a fractional seconds part of
      that many digits.
    """

    def __init__(self, fsp: int | Sentinel = IGNORED):
        """The `UTC_TIME([fsp])` function.

        MySQL description:
        - Returns the current UTC time as a value in 'hh:mm:ss' or hhmmss format,
          depending on whether the function is used in string or numeric context.
        - If the fsp argument is given to specify a fractional seconds precision
          from 0 to 6, the return value includes a fractional seconds part of
          that many digits.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UTC_TIME()
        # Escape output: "UTC_TIME()"
        # Expect result: '10:20:19'

        sqlfunc.UTC_TIME(3)
        # Escape output: "UTC_TIME(3)"
        # Expect result: '10:20:19.012'
        ```
        """
        if fsp is IGNORED:
            super().__init__("UTC_TIME", 0)
        else:
            super().__init__("UTC_TIME", 1, fsp)


@cython.cclass
class UTC_TIMESTAMP(SQLFunction):
    """Represents the `UTC_TIMESTAMP([fsp])` function.

    MySQL description:
    - Returns the current UTC date and time as a value in 'YYYY-MM-DD hh:mm:ss'
      or YYYYMMDDhhmmss format, depending on whether the function is used in
      string or numeric context.
    - If the fsp argument is given to specify a fractional seconds precision
      from 0 to 6, the return value includes a fractional seconds part of
      that many digits.
    """

    def __init__(self, fsp: int | Sentinel = IGNORED):
        """The `UTC_TIMESTAMP([fsp])` function.

        MySQL description:
        - Returns the current UTC date and time as a value in 'YYYY-MM-DD hh:mm:ss'
          or YYYYMMDDhhmmss format, depending on whether the function is used in
          string or numeric context.
        - If the fsp argument is given to specify a fractional seconds precision
          from 0 to 6, the return value includes a fractional seconds part of
          that many digits.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UTC_TIMESTAMP()
        # Escape output: "UTC_TIMESTAMP()"
        # Expect result: '2023-11-13 10:20:19'

        sqlfunc.UTC_TIMESTAMP(3)
        # Escape output: "UTC_TIMESTAMP(3)"
        # Expect result: '2023-11-13 10:20:19.012'
        ```
        """
        if fsp is IGNORED:
            super().__init__("UTC_TIMESTAMP", 0)
        else:
            super().__init__("UTC_TIMESTAMP", 1, fsp)


@cython.cclass
class UUID(SQLFunction):
    """Represents the `UUID()` function.

    MySQL description:
    - Returns a Universal Unique Identifier (UUID) generated according to RFC 4122,
      [“A Universally Unique IDentifier (UUID) URN Namespace”](http://www.ietf.org/rfc/rfc4122.txt).
    - A UUID is designed as a number that is globally unique in space and time.
      Two calls to UUID() are expected to generate two different values, even if
      these calls are performed on two separate devices not connected to each other.
    """

    def __init__(self):
        """The `UUID()` function.

        MySQL description:
        - Returns a Universal Unique Identifier (UUID) generated according to RFC 4122,
          [“A Universally Unique IDentifier (UUID) URN Namespace”](http://www.ietf.org/rfc/rfc4122.txt).
        - A UUID is designed as a number that is globally unique in space and time.
          Two calls to UUID() are expected to generate two different values, even if
          these calls are performed on two separate devices not connected to each other.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UUID()
        # Escape output: "UUID()"
        # Expect result: '6ccd780c-baba-1026-9564-5b8c656024db'
        ```
        """
        super().__init__("UUID", 0)


@cython.cclass
class UUID_SHORT(SQLFunction):
    """Represents the `UUID_SHORT()` function.

    MySQL description:
    - Returns a “short” universal identifier as a 64-bit unsigned integer.
      Values returned by UUID_SHORT() differ from the string-format 128-bit
      identifiers returned by the UUID() function and have different uniqueness
      properties.
    - The value of UUID_SHORT() is guaranteed to be unique if the following
      conditions hold: the server_id value of the current server is between
      0 and 255 and is unique among your set of source and replica servers;
      you do not set back the system time for your server host between mysqld
      restarts; you invoke UUID_SHORT() on average fewer than 16 million times
      per second between mysqld restarts.
    """

    def __init__(self):
        """The `UUID_SHORT()` function.

        MySQL description:
        - Returns a “short” universal identifier as a 64-bit unsigned integer.
          Values returned by UUID_SHORT() differ from the string-format 128-bit
          identifiers returned by the UUID() function and have different uniqueness
          properties.
        - The value of UUID_SHORT() is guaranteed to be unique if the following
          conditions hold: the server_id value of the current server is between
          0 and 255 and is unique among your set of source and replica servers;
          you do not set back the system time for your server host between mysqld
          restarts; you invoke UUID_SHORT() on average fewer than 16 million times
          per second between mysqld restarts.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.UUID_SHORT()
        # Escape output: "UUID_SHORT()"
        # Expect result: 101158163524354048
        ```
        """
        super().__init__("UUID_SHORT", 0)


@cython.cclass
class UUID_TO_BIN(SQLFunction):
    """Represents the `UUID_TO_BIN(string_uuid[, swap_flag])` function.

    MySQL description:
    - Converts a string UUID to a binary UUID and returns the result. The
      return binary UUID is a VARBINARY(16) value.
    - If the UUID argument is NULL, the return value is NULL. If any argument
      is invalid, an error occurs.
    - For optional swap_flag argument, if swap_flag is 0, the two-argument form
      is equivalent to the one-argument form. The binary result is in the same
      order as the string argument. If swap_flag is 1, the format of the return
      value differs: The time-low and time-high parts (the first and third groups
      of hexadecimal digits, respectively) are swapped. This moves the more rapidly
      varying part to the right and can improve indexing efficiency if the result
      is stored in an indexed column.
    """

    def __init__(self, string_uuid, swap_flag: Any | Sentinel = IGNORED):
        """The `UUID_TO_BIN(string_uuid[, swap_flag])` function.

        MySQL description:
        - Converts a string UUID to a binary UUID and returns the result. The
          return binary UUID is a VARBINARY(16) value.
        - If the UUID argument is NULL, the return value is NULL. If any argument
          is invalid, an error occurs.
        - For optional swap_flag argument, if swap_flag is 0, the two-argument form
          is equivalent to the one-argument form. The binary result is in the same
          order as the string argument. If swap_flag is 1, the format of the return
          value differs: The time-low and time-high parts (the first and third groups
          of hexadecimal digits, respectively) are swapped. This moves the more rapidly
          varying part to the right and can improve indexing efficiency if the result
          is stored in an indexed column.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        uuid = "6ccd780c-baba-1026-9564-5b8c656024db"

        sqlfunc.HEX(sqlfunc.UUID_TO_BIN(uuid))
        # Escape output: "HEX(UUID_TO_BIN('6ccd780c-baba-1026-9564-5b8c656024db'))"
        # Expect result: '6CCD780CBABA102695645B8C656024DB'

        sqlfunc.HEX(sqlfunc.UUID_TO_BIN(uuid, 1))
        # Escape output: "HEX(UUID_TO_BIN('6ccd780c-baba-1026-9564-5b8c656024db',1))"
        # Expect result: '1026BABA6CCD780C95645B8C656024DB'
        ```
        """
        if swap_flag is IGNORED:
            super().__init__("UUID_TO_BIN", 1, string_uuid)
        else:
            super().__init__("UUID_TO_BIN", 2, string_uuid, swap_flag)


# Functions: V -------------------------------------------------------------------------------------------------------
@cython.cclass
class VALIDATE_PASSWORD_STRENGTH(SQLFunction):
    """Represents the `VALIDATE_PASSWORD_STRENGTH(string)` function.

    MySQL description:
    - Given an argument representing a plaintext password, this function
      returns an integer to indicate how strong the password is.
    - Returns NULL if the argument is NULL. The return value ranges from
      0 (weak) to 100 (strong).
    """

    def __init__(self, string):
        """The `VALIDATE_PASSWORD_STRENGTH(string)` function.

        MySQL description:
        - Given an argument representing a plaintext password, this function
          returns an integer to indicate how strong the password is.
        - Returns NULL if the argument is NULL. The return value ranges from
          0 (weak) to 100 (strong).

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.VALIDATE_PASSWORD_STRENGTH("password")
        # Escape output: "VALIDATE_PASSWORD_STRENGTH('password')"
        # Expect result: 50
        """
        super().__init__("VALIDATE_PASSWORD_STRENGTH", 1, string)


@cython.cclass
class VERSION(SQLFunction):
    """Represents the `VERSION()` function.

    MySQL description:
    - Returns a string that indicates the MySQL server version. The string
      uses the utf8mb3 character set. The value might have a suffix in
      addition to the version number.
    """

    def __init__(self):
        """The `VERSION()` function.

        MySQL description:
        - Returns a string that indicates the MySQL server version. The string
          uses the utf8mb3 character set. The value might have a suffix in
          addition to the version number.

        ## Example:
        ```python
        from sqlcycli import sqlfunc

        sqlfunc.VERSION()
        # Escape output: "VERSION()"
        # Expect result: '8.4.3-standard'
        """
        super().__init__("VERSION", 0)


# Functions: W -------------------------------------------------------------------------------------------------------
@cython.cclass
class WEEK(SQLFunction):
    """Represents the `WEEK(date[, mode])` function.

    MySQL description:
    - This function returns the week number for date. The two-argument form of WEEK()
      enables you to specify whether the week starts on Sunday or Monday and whether
      the return value should be in the range from 0 to 53 or from 1 to 53. If the mode
      argument is omitted, the value of the default_week_format system variable is used.
    - For a NULL date value, the function returns NULL.

    ```plaintext
    Mode  First  day of week  Range Week 1 is the first week
    0	 Sunday	        0-53  with a Sunday in this year
    1	 Monday	        0-53  with 4 or more days this year
    2	 Sunday	        1-53  with a Sunday in this year
    3	 Monday	        1-53  with 4 or more days this year
    4	 Sunday	        0-53  with 4 or more days this year
    5	 Monday	        0-53  with a Monday in this year
    6	 Sunday	        1-53  with 4 or more days this year
    7	 Monday	        1-53  with a Monday in this year
    ```
    """

    def __init__(self, date, mode: int | Sentinel = IGNORED):
        """The `WEEK(date[,mode])` function.

        MySQL description:
        - This function returns the week number for date. The two-argument form of WEEK()
          enables you to specify whether the week starts on Sunday or Monday and whether
          the return value should be in the range from 0 to 53 or from 1 to 53. If the mode
          argument is omitted, the value of the default_week_format system variable is used.
        - For a NULL date value, the function returns NULL.

        ```plaintext
        Mode  First  day of week  Range Week 1 is the first week
        0	 Sunday	        0-53  with a Sunday in this year
        1	 Monday	        0-53  with 4 or more days this year
        2	 Sunday	        1-53  with a Sunday in this year
        3	 Monday	        1-53  with 4 or more days this year
        4	 Sunday	        0-53  with 4 or more days this year
        5	 Monday	        0-53  with a Monday in this year
        6	 Sunday	        1-53  with 4 or more days this year
        7	 Monday	        1-53  with a Monday in this year
        ```

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.WEEK("2008-02-20")
        # Escape output: "WEEK('2008-02-20')"
        # Expect result: 7

        sqlfunc.WEEK(datetime.date(2008, 2, 20), 1)
        # Escape output: "WEEK('2008-02-20',1)"
        # Expect result: 8
        ```
        """
        if mode is IGNORED:
            super().__init__("WEEK", 1, date)
        else:
            super().__init__("WEEK", 2, date, mode)


@cython.cclass
class WEEKDAY(SQLFunction):
    """Represents the `WEEKDAY(date)` function.

    MySQL description:
    - Returns the weekday index for date (0 = Monday, 1 = Tuesday, … 6 = Sunday).
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `WEEKDAY(date)` function.

        MySQL description:
        - Returns the weekday index for date (0 = Monday, 1 = Tuesday, … 6 = Sunday).
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.WEEKDAY("2008-02-03 22:23:00")
        # Escape output: "WEEKDAY('2008-02-03 22:23:00')"
        # Expect result: 6

        sqlfunc.WEEKDAY(datetime.date(2007, 11, 6))
        # Escape output: "WEEKDAY('2007-11-06')"
        # Expect result: 1
        ```
        """
        super().__init__("WEEKDAY", 1, date)


@cython.cclass
class WEEKOFYEAR(SQLFunction):
    """Represents the `WEEKOFYEAR(date)` function.

    MySQL description:
    - Returns the calendar week of the date as a number in the range from 1 to 53.
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `WEEKOFYEAR(date)` function.

        MySQL description:
        - Returns the calendar week of the date as a number in the range from 1 to 53.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.WEEKOFYEAR("2008-02-20")
        # Escape output: "WEEKOFYEAR('2008-02-20')"
        # Expect result: 8

        sqlfunc.WEEKOFYEAR(datetime.date(2008, 2, 20))
        # Escape output: "WEEKOFYEAR('2008-02-20')"
        # Expect result: 8
        ```
        """
        super().__init__("WEEKOFYEAR", 1, date)


# Functions: Y -------------------------------------------------------------------------------------------------------
@cython.cclass
class YEAR(SQLFunction):
    """Represents the `YEAR(date)` function.

    MySQL description:
    - Returns the year for date, in the range 1000 to 9999, or 0 for the “zero” date.
    - Returns NULL if date is NULL.
    """

    def __init__(self, date):
        """The `YEAR(date)` function.

        MySQL description:
        - Returns the year for date, in the range 1000 to 9999, or 0 for the “zero” date.
        - Returns NULL if date is NULL.

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.YEAR("2008-02-20")
        # Escape output: "YEAR('2008-02-20')"
        # Expect result: 2008

        sqlfunc.YEAR(datetime.date(2008, 2, 20))
        # Escape output: "YEAR('2008-02-20')"
        # Expect result: 2008
        ```
        """
        super().__init__("YEAR", 1, date)


@cython.cclass
class YEARWEEK(SQLFunction):
    """Represents the `YEARWEEK(date[, mode])` function.

    MySQL description:
    - Returns year and week for a date. The year in the result may be different
      from the year in the date argument for the first and the last week of the
      year.
    - Returns NULL if date is NULL.

    ```plaintext
    Mode  First  day of week  Range Week 1 is the first week
    0	 Sunday	        0-53  with a Sunday in this year
    1	 Monday	        0-53  with 4 or more days this year
    2	 Sunday	        1-53  with a Sunday in this year
    3	 Monday	        1-53  with 4 or more days this year
    4	 Sunday	        0-53  with 4 or more days this year
    5	 Monday	        0-53  with a Monday in this year
    6	 Sunday	        1-53  with 4 or more days this year
    7	 Monday	        1-53  with a Monday in this year
    ```
    """

    def __init__(self, date, mode: int | Sentinel = IGNORED):
        """The `YEARWEEK(date[, mode])` function.

        MySQL description:
        - Returns year and week for a date. The year in the result may be different
          from the year in the date argument for the first and the last week of the
          year.
        - Returns NULL if date is NULL.

        ```plaintext
        Mode  First  day of week  Range Week 1 is the first week
        0	 Sunday	        0-53  with a Sunday in this year
        1	 Monday	        0-53  with 4 or more days this year
        2	 Sunday	        1-53  with a Sunday in this year
        3	 Monday	        1-53  with 4 or more days this year
        4	 Sunday	        0-53  with 4 or more days this year
        5	 Monday	        0-53  with a Monday in this year
        6	 Sunday	        1-53  with 4 or more days this year
        7	 Monday	        1-53  with a Monday in this year
        ```

        ## Example:
        ```python
        import datetime
        from sqlcycli import sqlfunc

        sqlfunc.YEARWEEK("2008-02-20")
        # Escape output: "YEARWEEK('2008-02-20')"
        # Expect result: '200807'

        sqlfunc.YEARWEEK(datetime.date(2008, 2, 20), 1)
        # Escape output: "YEARWEEK('2008-02-20',1)"
        # Expect result: '200808'
        ```
        """
        if mode is IGNORED:
            super().__init__("YEARWEEK", 1, date)
        else:
            super().__init__("YEARWEEK", 2, date, mode)
