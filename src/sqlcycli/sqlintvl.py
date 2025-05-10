# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

"""
This module provides a collection of classes to represents the MySQL temporal intervals
in a structured way. Each class corresponds to a distinct MySQL interval unit (e.g.,
YEAR, MONTH, DAY, HOUR_MINUTE, etc.), and all of them derive from a common base class
`<'SQLInterval'>`.

The interval classes in this module are more like a wrapper for the INTERVAL value
so the 'escape()' method in the 'sqlcycli' package can handle them correctly.
"""

# Cython imports
import cython

__all__ = [
    "SQLInterval",
    "MICROSECOND",
    "SECOND",
    "MINUTE",
    "HOUR",
    "DAY",
    "WEEK",
    "MONTH",
    "QUARTER",
    "YEAR",
    "SECOND_MICROSECOND",
    "MINUTE_MICROSECOND",
    "MINUTE_SECOND",
    "HOUR_MICROSECOND",
    "HOUR_SECOND",
    "HOUR_MINUTE",
    "DAY_MICROSECOND",
    "DAY_SECOND",
    "DAY_MINUTE",
    "DAY_HOUR",
    "YEAR_MONTH",
]


# Constant -----------------------------------------------------------------------------------------------------------
INTERVAL_UNITS: set[str] = {
    "MICROSECOND",
    "SECOND",
    "MINUTE",
    "HOUR",
    "DAY",
    "WEEK",
    "MONTH",
    "QUARTER",
    "YEAR",
    "SECOND_MICROSECOND",
    "MINUTE_MICROSECOND",
    "MINUTE_SECOND",
    "HOUR_MICROSECOND",
    "HOUR_SECOND",
    "HOUR_MINUTE",
    "DAY_MICROSECOND",
    "DAY_SECOND",
    "DAY_MINUTE",
    "DAY_HOUR",
    "YEAR_MONTH",
}


# Base class ---------------------------------------------------------------------------------------------------------
@cython.cclass
class SQLInterval:
    """Represents the base class for MySQL temporal interval."""

    _name: str
    _expr: object
    _hashcode: cython.Py_ssize_t

    def __init__(self, interval_name: str, expr: object):
        """The base class for MySQL temporal interval.

        :param interval_name `<'str'>`: The name of the MySQL temporal interval (e.g. 'YEAR', 'MONTH', 'DAY')
        :param expr `<'object'>`: The quantity of the interval.
        """
        self._name = interval_name
        self._expr = expr
        self._hashcode = -1

    @property
    def name(self) -> str:
        """The name of the MySQL temporal interval."""
        return self._name

    @property
    def expr(self) -> object:
        """The quantity of the MySQL temporal interval."""
        return self._expr

    @cython.ccall
    def syntax(self) -> str:
        """Generate the temporal interval syntax with the correct 
        placeholders for the expression (expr) `<'str'>`.
        """
        return "INTERVAL %s " + self._name

    def __repr__(self) -> str:
        return "<SQLInterval: INTERVAL %s %s>" % (str(self._expr), self._name)

    def __str__(self) -> str:
        return "INTERVAL %s %s" % (str(self._expr), self._name)

    def __hash__(self) -> int:
        if self._hashcode == -1:
            self._hashcode = hash((self.__class__.__name__, self._name, self._expr))
        return self._hashcode


# Intervals ----------------------------------------------------------------------------------------------------------
@cython.cclass
class MICROSECOND(SQLInterval):
    """Represents the MySQL 'MICROSECONDS' interval."""

    def __init__(self, expr):
        """The MySQL 'MICROSECONDS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.MICROSECOND(5)
        # Escape output: "INTERVAL 5 MICROSECOND"
        ```
        """
        super().__init__("MICROSECOND", expr)


@cython.cclass
class SECOND(SQLInterval):
    """Represents the MySQL 'SECONDS' interval."""

    def __init__(self, expr):
        """The MySQL 'SECONDS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.SECOND(5)
        # Escape output: "INTERVAL 5 SECOND"
        ```
        """
        super().__init__("SECOND", expr)


@cython.cclass
class MINUTE(SQLInterval):
    """Represents the MySQL 'MINUTES' interval."""

    def __init__(self, expr: object):
        """The MySQL 'MINUTES' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.MINUTE(5)
        # Escape output: "INTERVAL 5 MINUTE"
        ```
        """
        super().__init__("MINUTE", expr)


@cython.cclass
class HOUR(SQLInterval):
    """Represents the MySQL 'HOURS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'HOURS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.HOUR(5)
        # Escape output: "INTERVAL 5 HOUR"
        ```
        """
        super().__init__("HOUR", expr)


@cython.cclass
class DAY(SQLInterval):
    """Represents the MySQL 'DAYS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'DAYS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.DAY(5)
        # Escape output: "INTERVAL 5 DAY"
        ```
        """
        super().__init__("DAY", expr)


@cython.cclass
class WEEK(SQLInterval):
    """Represents the MySQL 'WEEKS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'WEEKS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.WEEK(5)
        # Escape output: "INTERVAL 5 WEEK"
        ```
        """
        super().__init__("WEEK", expr)


@cython.cclass
class MONTH(SQLInterval):
    """Represents the MySQL 'MONTHS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'MONTHS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.MONTH(5)
        # Escape output: "INTERVAL 5 MONTH"
        ```
        """
        super().__init__("MONTH", expr)


@cython.cclass
class QUARTER(SQLInterval):
    """Represents the MySQL 'QUARTERS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'QUARTERS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.QUARTER(5)
        # Escape output: "INTERVAL 5 QUARTER"
        ```
        """
        super().__init__("QUARTER", expr)


@cython.cclass
class YEAR(SQLInterval):
    """Represents the MySQL 'YEARS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'YEARS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.YEAR(5)
        # Escape output: "INTERVAL 5 YEAR"
        ```
        """
        super().__init__("YEAR", expr)


@cython.cclass
class SECOND_MICROSECOND(SQLInterval):
    """Represents the MySQL 'SS.US' interval."""

    def __init__(self, expr: object):
        """The MySQL 'SS.US' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.SECOND_MICROSECOND("01.000001")
        # Escape output: "INTERVAL '01.000001' SECOND_MICROSECOND"
        ```
        """
        super().__init__("SECOND_MICROSECOND", expr)


@cython.cclass
class MINUTE_MICROSECOND(SQLInterval):
    """Represents the MySQL 'MI:SS.US' interval."""

    def __init__(self, expr: object):
        """The MySQL 'MI:SS.US' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.MINUTE_MICROSECOND("01:01.000001")
        # Escape output: "INTERVAL '01:01.000001' MINUTE_MICROSECOND"
        ```
        """
        super().__init__("MINUTE_MICROSECOND", expr)


@cython.cclass
class MINUTE_SECOND(SQLInterval):
    """Represents the MySQL 'MI:SS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'MI:SS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.MINUTE_SECOND("01:01")
        # Escape output: "INTERVAL '01:01' MINUTE_SECOND"
        ```
        """
        super().__init__("MINUTE_SECOND", expr)


@cython.cclass
class HOUR_MICROSECOND(SQLInterval):
    """Represents the MySQL 'HH:MI:SS.US' interval."""

    def __init__(self, expr: object):
        """The MySQL 'HH:MI:SS.US' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.HOUR_MICROSECOND("01:01:01.000001")
        # Escape output: "INTERVAL '01:01:01.000001' HOUR_MICROSECOND"
        ```
        """
        super().__init__("HOUR_MICROSECOND", expr)


@cython.cclass
class HOUR_SECOND(SQLInterval):
    """Represents the MySQL 'HH:MI:SS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'HH:MI:SS' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.HOUR_SECOND("01:01:01")
        # Escape output: "INTERVAL '01:01:01' HOUR_SECOND"
        ```
        """
        super().__init__("HOUR_SECOND", expr)


@cython.cclass
class HOUR_MINUTE(SQLInterval):
    """Represents the MySQL 'HH:MI' interval."""

    def __init__(self, expr: object):
        """The MySQL 'HH:MI' interval.

        ```python
        from sqlcycli import sqlintvl

        sqlintvl.HOUR_MINUTE("01:01")
        # Escape output: "INTERVAL '01:01' HOUR_MINUTE"
        ```
        """
        super().__init__("HOUR_MINUTE", expr)


@cython.cclass
class DAY_MICROSECOND(SQLInterval):
    """Represents the MySQL 'DD HH:MI:SS.US' interval."""

    def __init__(self, expr: object):
        """The MySQL 'DD HH:MI:SS.US' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.DAY_MICROSECOND("01 01:01:01.000001")
        # Escape output: "INTERVAL '01 01:01:01.000001' DAY_MICROSECOND"
        ```
        """
        super().__init__("DAY_MICROSECOND", expr)


@cython.cclass
class DAY_SECOND(SQLInterval):
    """Represents the MySQL 'DD HH:MI:SS' interval."""

    def __init__(self, expr: object):
        """The MySQL 'DD HH:MI:SS' interval.

        ## Exmple:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.DAY_SECOND("01 01:01:01")
        # Escape output: "INTERVAL '01 01:01:01' DAY_SECOND"
        ```
        """
        super().__init__("DAY_SECOND", expr)


@cython.cclass
class DAY_MINUTE(SQLInterval):
    """Represents the MySQL 'DD HH:MI interval."""

    def __init__(self, expr: object):
        """The MySQL 'DD HH:MI interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.DAY_MINUTE("01 01:01")
        # Escape output: "INTERVAL '01 01:01' DAY_MINUTE"
        ```
        """
        super().__init__("DAY_MINUTE", expr)


@cython.cclass
class DAY_HOUR(SQLInterval):
    """Represents the MySQL 'DD HH' interval."""

    def __init__(self, expr: object):
        """The MySQL 'DD HH' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.DAY_HOUR("01 01")
        # Escape output: "INTERVAL '01 01' DAY_HOUR"
        ```
        """
        super().__init__("DAY_HOUR", expr)


@cython.cclass
class YEAR_MONTH(SQLInterval):
    """Represents the MySQL 'YY-MM' interval."""

    def __init__(self, expr: object):
        """The MySQL 'YY-MM' interval.

        ## Example:
        ```python
        from sqlcycli import sqlintvl

        sqlintvl.YEAR_MONTH("2-5")
        # Escape output: "INTERVAL '2-5' YEAR_MONTH"
        ```
        """
        super().__init__("YEAR_MONTH", expr)
