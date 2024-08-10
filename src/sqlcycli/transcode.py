# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.libc.math import isnormal, isfinite  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.list import PyList_AsTuple as list_to_tuple  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GET_ITEM as tuple_getitem  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_GET_SIZE as bytes_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Split as str_split  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as read_char  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Substring as str_substr  # type: ignore
from cython.cimports.sqlcycli.constants import _FIELD_TYPE  # type: ignore
from cython.cimports.sqlcycli import typeref  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from typing import Callable, Iterable
import datetime, numpy as np
from pandas import Series, DataFrame
from MySQLdb._mysql import string_literal
from orjson import loads, dumps, OPT_SERIALIZE_NUMPY
from sqlcycli.constants import _FIELD_TYPE
from sqlcycli import typeref, errors

__all__ = ["escape", "decode", "BIT", "JSON"]

# Constants -----------------------------------------------------------------------------------
# . translate table
# Used to translate Python string to literals.
STR_ESCAPE_TABLE: list = [chr(x) for x in range(128)]
STR_ESCAPE_TABLE[0] = "\\0"
STR_ESCAPE_TABLE[ord("\\")] = "\\\\"
STR_ESCAPE_TABLE[ord("\n")] = "\\n"
STR_ESCAPE_TABLE[ord("\r")] = "\\r"
STR_ESCAPE_TABLE[ord("\032")] = "\\Z"
STR_ESCAPE_TABLE[ord('"')] = '\\"'
STR_ESCAPE_TABLE[ord("'")] = "\\'"
# Used to translate 'orjson' serialized datetime64 to literal datetime format.
DT64_JSON_TABLE: list = [chr(x) for x in range(128)]
DT64_JSON_TABLE[ord("T")] = " "
DT64_JSON_TABLE[ord('"')] = "'"
DT64_JSON_TABLE[ord("[")] = "("
DT64_JSON_TABLE[ord("]")] = ")"
# . time units
# fmt: off
DAYS_BR_MONTH: cython.uint[13] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
US_DAY: cython.longlong = 86_400_000_000
US_HOUR: cython.longlong = 3_600_000_000
EPOCH_US: cython.longlong = 62_135_683_200_000_000
DT_MAX_US: cython.longlong = 315_537_983_999_999_999
DT_MIN_US: cython.longlong = 86_400_000_000
# fmt: on
# . datetime
US_FRACTION_CORRECTION: cython.uint[5] = [100000, 10000, 1000, 100, 10]
# . ndarray dtype
_arr: np.ndarray = np.array(None, dtype=object)
NDARRAY_DTYPE_OBJECT: cython.char = _arr.descr.kind  # 'O'
_arr: np.ndarray = np.array(1, dtype=np.bool_)
NDARRAY_DTYPE_BOOL: cython.char = _arr.descr.kind  # 'b'
_arr: np.ndarray = np.array(1.1, dtype=np.float64)
NDARRAY_DTYPE_FLOAT: cython.char = _arr.descr.kind  # 'f'
_arr: np.ndarray = np.array(1, dtype=np.int64)
NDARRAY_DTYPE_INT: cython.char = _arr.descr.kind  # 'i'
_arr: np.ndarray = np.array(1, dtype=np.uint64)
NDARRAY_DTYPE_UINT: cython.char = _arr.descr.kind  # 'u'
_arr: np.ndarray = np.array(1, dtype="datetime64[ns]")
NDARRAY_DTYPE_DT64: cython.char = _arr.descr.kind  # 'M'
_arr: np.ndarray = np.array(1, dtype="timedelta64[ns]")
NDARRAY_DTYPE_TD64: cython.char = _arr.descr.kind  # 'm'
_arr: np.ndarray = np.array(b"", dtype="S")
NDARRAY_DTYPE_BYTES: cython.char = _arr.descr.kind  # 'S'
_arr: np.ndarray = np.array("", dtype="U")
NDARRAY_DTYPE_UNICODE: cython.char = _arr.descr.kind  # 'U'
_arr: np.ndarray = None
# . functions
FN_ORJSON_LOADS: Callable = loads
FN_ORJSON_DUMPS: Callable = dumps
FN_ORJSON_OPT_NUMPY: object = OPT_SERIALIZE_NUMPY
FN_MYSQLCLI_STR2LIT: Callable = string_literal


# Orjson dumps --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _orjson_dumps(obj: object) -> str:
    """(cfunc) Serialize python object to JSON string `<'str'>`.

    Based on [orjson](https://github.com/ijl/orjson) `'dumps()'` function.
    """
    return decode_bytes_utf8(FN_ORJSON_DUMPS(obj))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _orjson_dumps_numpy(obj: object) -> str:
    """(cfunc) Serialize numpy.ndarray to JSON string `<'str'>`.

    Based on [orjson](https://github.com/ijl/orjson) `'dumps()'` function.
    """
    return decode_bytes_utf8(FN_ORJSON_DUMPS(obj, option=FN_ORJSON_OPT_NUMPY))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _bytes_to_literal(obj: object) -> object:
    """(cfunc) Escape bytes object to literal `<'bytes'>`.

    Based on [mysqlclient](https://github.com/PyMySQL/mysqlclient)
    `'string_literal()'` function.
    """
    return FN_MYSQLCLI_STR2LIT(obj)


@cython.cfunc
@cython.inline(True)
def _string_to_literal(obj: object, encoding: cython.pchar) -> str:
    """(cfunc) Escape string object to literal `<'str'>`.

    Based on [mysqlclient](https://github.com/PyMySQL/mysqlclient)
    `'string_literal()'` function.
    """
    return decode_bytes(FN_MYSQLCLI_STR2LIT(encode_str(obj, encoding)), encoding)  # type: ignore


# Custom types ================================================================================
@cython.cclass
class _CustomType:
    """The base class for custom type.

    Validation & escape (conversion) should only happens
    when executed by the 'escape()' function.
    """

    _value: object

    def __init__(self, value: object) -> None:
        """The base class for custom type.

        Validation & escape (conversion) should only happens
        when executed by the 'escape()' function.

        :param value: `<'object'>` The value.
        """
        self._value = value

    @property
    def value(self) -> object:
        """The value `<'object'>`."""
        return self._value

    def __repr__(self) -> str:
        return "<'%s' (value=%r)>" % (self.__class__.__name__, self._value)


@cython.cclass
class BIT(_CustomType):
    """Represents a value for MySQL BIT column. Act as a wrapper
    for the BIT value, so the 'escape()' function can identify and
    escape the value to the desired literal format.

    - Accepts raw bytes or integer value.
    - Validation & conversion only happens when executed by the 'escape()' function.
    """

    def __init__(self, value: bytes | int) -> None:
        """The value for MySQL BIT column. Act as a wrapper for the
        BIT value, so the 'escape()' function can identify and escape
        the value to the desired literal format.

        - Validation & conversion only happens when executed by the 'escape()' function.

        :param value: `<'bytes/int'>` The value for MySQL BIT column, accepts:
            - `<'bytes'>`: The raw bytes value, e.g. b'\\x01'.
            - `<'int'>`: An unsigned integer value, e.g. 1.
        """
        self._value = value


@cython.cclass
class JSON(_CustomType):
    """Represents a value for MySQL JSON column. Act as a wrapper
    for the JSON value, so the 'escape()' function can identify and
    escape the value to the desired literal format.

    - Accepts any objects that can be serialized to JSON format.
    - Do `NOT` pass already serialized JSON string to this class.
    - Validation & conversion only happens when called by the 'escape()' function.
    """

    def __init__(self, value: object) -> None:
        """The value for MySQL JSON column. Act as a wrapper for the
        JSON value, so the 'escape()' function can identify and escape
        the value to the desired literal format.

        - Do `NOT` pass already serialized JSON string to this class.
        - Validation & conversion only happens when called by the 'escape()' function.

        :param value: `<'object'>` The value for MySQL JSON column.
            - An objects that can be serialized to JSON format.
        """
        self._value = value


# Escape ======================================================================================
# Escape --------------------------------------------------------------------------------------
# . Basic types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_bool(data: object) -> str:
    """(cfunc) Escape boolean 'data' to literal `<'str'>.

    ### Example:
    >>> _escape_bool(True)
    >>> "1"  # str
    """
    return "1" if data else "0"


@cython.cfunc
@cython.inline(True)
def _escape_int(data: object) -> str:
    """(cfunc) Escape integer 'data' to literal `<'str'>.

    Since this function simply calls Python built-in `str()`,
    all actual integer escapes are written as 'str(data)' inline
    (with '# _escape_int' comment) for optimal performance.

    ### Example:
    >>> _escape_int(123)
    >>> "123"  # str
    """
    return str(data)


@cython.cfunc
@cython.inline(True)
def _escape_float(data: object) -> str:
    """(cfunc) Escape float 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_float(123.456)
    >>> "123.456"  # str
    """
    # For normal native Python float numbers, orjson 
    # performs faster than Python built-in `str()` 
    # function.
    if isnormal(data):
        return _orjson_dumps(data)
    # For other float values, we fallback to Python
    # built-in `str()` approach.
    return _escape_float64(data)


@cython.cfunc
@cython.inline(True)
def _escape_float64(data: object) -> str:
    """(cfunc) Escape numpy.float_ 'data' to literal `<'str'>`.

    :raises `<'ValueError'>`: If float value is invalid.

    ### Example:
    >>> _escape_float64(np.float64(123.456))
    >>> "123.456"  # str
    """
    # For numpy.float64, Python built-in `str()`
    # function performs faster than orjson for most small
    # float numbers (with less than 6 decimal places).
    if isfinite(data):
        return str(data)
    raise ValueError("Float value '%s' is not supported." % data)


@cython.cfunc
@cython.inline(True)
def _escape_str(data: object, encoding: cython.pchar) -> str:
    """(cfunc) Escape string 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_str("Hello, World!")
    >>> "'Hello, World!'"  # str
    """
    return _string_to_literal(data, encoding)


@cython.cfunc
@cython.inline(True)
def _escape_none(_) -> str:
    """(cfunc) Escape None 'data' to literal `<'str'>`.

    Since this function simply returns 'NULL', all
    actual None escapes return 'NULL' directly inline
    (with '# _escape_none' comment) for optimal
    performance.

    ### Example:
    >>> _escape_none(None)
    >>> "NULL"  # str
    """
    return "NULL"


# . Date&Time types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_datetime(data: object) -> str:
    """(cfunc) Escape datetime 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_datetime(datetime.datetime(2021, 1, 1, 12, 0, 0, 100))
    >>> "'2021-01-01 12:00:00.00100'"  # str
    """
    microsecond: cython.int = datetime.PyDateTime_DATE_GET_MICROSECOND(data)
    if microsecond == 0:
        return "'%04d-%02d-%02d %02d:%02d:%02d'" % (
            datetime.PyDateTime_GET_YEAR(data),
            datetime.PyDateTime_GET_MONTH(data),
            datetime.PyDateTime_GET_DAY(data),
            datetime.PyDateTime_DATE_GET_HOUR(data),
            datetime.PyDateTime_DATE_GET_MINUTE(data),
            datetime.PyDateTime_DATE_GET_SECOND(data),
        )
    else:
        return "'%04d-%02d-%02d %02d:%02d:%02d.%06d'" % (
            datetime.PyDateTime_GET_YEAR(data),
            datetime.PyDateTime_GET_MONTH(data),
            datetime.PyDateTime_GET_DAY(data),
            datetime.PyDateTime_DATE_GET_HOUR(data),
            datetime.PyDateTime_DATE_GET_MINUTE(data),
            datetime.PyDateTime_DATE_GET_SECOND(data),
            microsecond,
        )


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_datetime64(data: object) -> str:
    """(cfunc) Escape numpy.datetime64 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_datetime64(np.datetime64('2021-01-01T12:00:00.001'))
    >>> "'2021-01-01 12:00:00.00100'"  # str
    """
    # Add back epoch seconds
    microseconds: cython.longlong = dt64_to_microseconds(data) + EPOCH_US  # type: ignore
    microseconds = min(max(microseconds, DT_MIN_US), DT_MAX_US)
    # Calculate ymd
    ymd = ordinal_to_ymd(microseconds // US_DAY)  # type: ignore
    # Calculate hms
    hms = microseconds_to_hms(microseconds)  # type: ignore
    # Escape
    # fmt: off
    microsecond: cython.uint = hms.microsecond
    if microsecond == 0:
        return "'%04d-%02d-%02d %02d:%02d:%02d'" % (
            ymd.year, ymd.month, ymd.day, 
            hms.hour, hms.minute, hms.second,
        )
    else:
        return "'%04d-%02d-%02d %02d:%02d:%02d.%06d'" % (
            ymd.year, ymd.month, ymd.day, 
            hms.hour, hms.minute, hms.second, microsecond,
        )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def _escape_struct_time(data: object) -> str:
    """(cfunc) Escape struct_time 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_struct_time(time.localtime())
    >>> "'2021-01-01 12:00:00'"  # str
    """
    # fmt: off
    return _escape_datetime(datetime.datetime_new(
        data.tm_year, data.tm_mon, data.tm_mday,
        data.tm_hour, data.tm_min, data.tm_sec,
        0, None, 0) )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def _escape_date(data: object) -> str:
    """(cfunc) Escape date 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_date(datetime.date(2021, 1, 1))
    >>> "'2021-01-01'"  # str
    """
    return "'%04d-%02d-%02d'" % (
        datetime.PyDateTime_GET_YEAR(data),
        datetime.PyDateTime_GET_MONTH(data),
        datetime.PyDateTime_GET_DAY(data),
    )


@cython.cfunc
@cython.inline(True)
def _escape_time(data: object) -> str:
    """(cfunc) Escape time 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_time(datetime.time(12, 0, 0, 100))
    >>> "'12:00:00.00100'"  # str
    """
    microsecond: cython.int = datetime.PyDateTime_TIME_GET_MICROSECOND(data)
    if microsecond == 0:
        return "'%02d:%02d:%02d'" % (
            datetime.PyDateTime_TIME_GET_HOUR(data),
            datetime.PyDateTime_TIME_GET_MINUTE(data),
            datetime.PyDateTime_TIME_GET_SECOND(data),
        )
    else:
        return "'%02d:%02d:%02d.%06d'" % (
            datetime.PyDateTime_TIME_GET_HOUR(data),
            datetime.PyDateTime_TIME_GET_MINUTE(data),
            datetime.PyDateTime_TIME_GET_SECOND(data),
            microsecond,
        )


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_timedelta(data: object) -> str:
    """(cfunc) Escape timedelta 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_timedelta(
            datetime.timedelta(hours=12, minutes=0, seconds=0, microseconds=100))
    >>> "'12:00:00.000100'"  # str
    """
    # Get total seconds and microseconds
    seconds: cython.longlong = (
        datetime.PyDateTime_DELTA_GET_SECONDS(data)
        + datetime.PyDateTime_DELTA_GET_DAYS(data) * 86_400
    )
    microseconds: cython.int = datetime.PyDateTime_DELTA_GET_MICROSECONDS(data)

    # Positive timedelta
    if seconds >= 0:
        hours = seconds // 3_600
        seconds %= 3_600
        minutes = seconds // 60
        seconds %= 60
        if microseconds == 0:
            return "'%02d:%02d:%02d'" % (hours, minutes, seconds)
        else:
            return "'%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, microseconds)

    # Negative w/o microseconds
    elif microseconds == 0:
        seconds = abs(seconds)
        if seconds >= 3_600:
            hours = seconds // 3_600
            seconds %= 3_600
            minutes = seconds // 60
            seconds %= 60
            return "'-%02d:%02d:%02d'" % (hours, minutes, seconds)
        elif seconds >= 60:
            minutes = seconds // 60
            seconds %= 60
            return "'-00:%02d:%02d'" % (minutes, seconds)
        else:
            return "'-00:00:%02d'" % seconds

    # Negative w/t microseconds
    else:
        us: cython.longlong = abs(seconds * 1_000_000 + microseconds)
        if us >= US_HOUR:
            hours = us // US_HOUR
            us %= US_HOUR
            minutes = us // 60_000_000
            us %= 60_000_000
            seconds = us // 1_000_000
            us %= 1_000_000
            return "'-%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us)
        elif us >= 60_000_000:
            minutes = us // 60_000_000
            us %= 60_000_000
            seconds = us // 1_000_000
            us %= 1_000_000
            return "'-00:%02d:%02d.%06d'" % (minutes, seconds, us)
        elif us >= 1_000_000:
            seconds = us // 1_000_000
            us %= 1_000_000
            return "'-00:00:%02d.%06d'" % (seconds, us)
        else:
            return "'-00:00:00.%06d'" % us


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_timedelta64(data: object) -> str:
    """(cfunc) Escape numpy.timedelta64 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_timedelta64(np.timedelta64('12:00:00.000100'))
    >>> "'12:00:00.000100'"  # str
    """
    us: cython.longlong = td64_to_microseconds(data)  # type: ignore
    return _escape_timedelta64_fr_us(us)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_timedelta64_fr_us(us: cython.longlong) -> str:
    """(cfunc) Escape numpy.timedelta64 microseconds to literal `<'str'>`."""
    negate: cython.bint = us < 0
    us = abs(us)
    hours = us // US_HOUR
    us %= US_HOUR
    minutes = us // 60_000_000
    us %= 60_000_000
    seconds = us // 1_000_000
    us %= 1_000_000
    if us == 0:
        if negate:
            return "'-%02d:%02d:%02d'" % (hours, minutes, seconds)
        else:
            return "'%02d:%02d:%02d'" % (hours, minutes, seconds)
    else:
        if negate:
            return "'-%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us)
        else:
            return "'%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us)


# . Bytes types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_bytes(data: object) -> str:
    """(cfunc) Escape bytes 'data' to literal
    ('_binary' prefixed) `<'str'>`.

    ### Example:
    >>> _escape_bytes(b"Hello, World!")
    >>> "_binary'Hello, World!'"  # str
    """
    return "_binary" + decode_bytes_ascii(_bytes_to_literal(data))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_bytearray(data: object) -> str:
    """(cfunc) Escape bytearray 'data' to literal
    ('_binary' prefixed) `<'str'>`.

    ### Example:
    >>> _escape_bytearray(bytearray(b"Hello, World!"))
    >>> "_binary'Hello, World!'"  # str
    """
    return _escape_bytes(bytes(data))


@cython.cfunc
@cython.inline(True)
def _escape_memoryview(data: memoryview) -> str:
    """(cfunc) Escape memoryview 'data' to literal
    ('_binary' prefixed) `<'str'>`.

    ### Example:
    >>> _escape_memoryview(memoryview(b"Hello, World!"))
    >>> "_binary'Hello, World!'"  # str
    """
    return _escape_bytes(data.tobytes())


# . Numeric types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_decimal(data: object) -> str:
    """(cfunc) Escape decimal 'data' to literal `<'str'>`.

    Since this function simply calls Python built-in `str()`,
    all actual decimal escapes are written as 'str(data)' inline
    (with '# _escape_decimal' comment) for optimal performance.

    ### Example:
    >>> _escape_decimal(decimal.Decimal("123.456"))
    >>> "123.456"  # str
    """
    return str(data)


# . Custom types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_bit(data: BIT) -> str:
    """(cfunc) Escape BIT 'data' to literal `<'str'>`.

    ### Example (bytes):
    >>> _escape_bit(BIT(b'\\x01'))
    >>> '1'  # str

    ### Example (int):
    >>> _escape_bit(BIT(1))
    >>> '1'  # str
    """
    value: object = data._value
    dtype = type(value)
    # Raw bytes
    if dtype is bytes:
        b_val: bytes = value
    # Bytes-like
    elif dtype is bytearray or dtype is memoryview or dtype is typeref.BYTES_:
        b_val: bytes = bytes(value)
    # Integer
    else:
        # . validate int & escape
        try:
            i_val: cython.ulonglong = int(value)
            return str(i_val)  # _escape_int
        except Exception as err:
            raise ValueError("Invalid BIT value %r %s." % (value, type(value))) from err

    # . decode to int & esacpe (raw bytes)
    return str(_decode_bit(b_val, True))  # _escape_int


@cython.cfunc
@cython.inline(True)
def _escape_json(data: JSON, encoding: cython.pchar) -> str:
    """(cfunc) Escape JSON 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_json(JSON({"key": "value"}))
    >>> "'{\\"key\\":\\"value\\"}'"  # str
    """
    try:
        return decode_bytes(  # type: ignore
            _bytes_to_literal(FN_ORJSON_DUMPS(data._value)),
            encoding,
        )
    except Exception as err:
        raise ValueError(
            "Invalid JSON value %s\n%r." % (type(data._value), data._value)
        ) from err


# . Sequence types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_list(data: list, encoding: cython.pchar) -> str:
    """(cfunc) Escape list 'data' to literal `<'str'>`.

    ### Example (flat):
    >>> _escape_list(
        ["val1", 1, 1.1])
    >>> "('val1',1,1.1)"  # str

    ### Example (nested):
    >>> _escape_list(
        [["val1", 1, 1.1], ["val2", 2, 2.2]])
    >>> "('val1',1,1.1),('val2',2,2.2)"  # str
    """
    res = ",".join([_escape_common(i, encoding) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_tuple(data: tuple, encoding: cython.pchar) -> str:
    """(cfunc) Escape tuple 'data' to literal `<'str'>`.

    ### Example (flat):
    >>> _escape_tuple(
        ("val1", 1, 1.1))
    >>> "('val1',1,1.1)"  # str

    ### Example (nested):
    >>> _escape_tuple(
        (("val1", 1, 1.1), ("val2", 2, 2.2)))
    >>> "('val1',1,1.1),('val2',2,2.2)"  # str
    """
    res = ",".join([_escape_common(i, encoding) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_set(data: set, encoding: cython.pchar) -> str:
    """(cfunc) Escape set 'data' to literal `<'str'>`.

    ### Example (flat):
    >>> _escape_set(
        {"val1", 1, 1.1})
    >>> "('val1',1,1.1)"  # str

    ### Example (nested):
    >>> _escape_set(
        {("val1", 1, 1.1), ("val2", 2, 2.2)})
    >>> "('val1',1,1.1),('val2',2,2.2)"  # str
    """
    res = ",".join([_escape_common(i, encoding) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_frozenset(data: frozenset, encoding: cython.pchar) -> str:
    """(cfunc) Escape frozenset 'data' to literal `<'str'>`.

    ### Example (flat):
    >>> _escape_frozenset(frozenset(
        {"val1", 1, 1.1}))
    >>> "('val1',1,1.1)"  # str

    ### Example (nested):
    >>> _escape_frozenset(frozenset(
        {("val1", 1, 1.1), ("val2", 2, 2.2)}))
    >>> "('val1',1,1.1),('val2',2,2.2)"  # str
    """
    res = ",".join([_escape_common(i, encoding) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_sequence(data: Iterable, encoding: cython.pchar) -> str:
    """(cfunc) Escape sequence 'data' to literal `<'str'>`.

    ### Example (flat):
    >>> _escape_sequence(
        {"key1": "val1", "key2": 1, "key3": 1.1}.values())
    >>> "('val1',1,1.1)"  # str

    ### Example (nested):
    >>> _escape_sequence(
        {"key1": ["val1", 1, 1.1], "key2": ["val2", 2, 2.2]}.values())
    >>> "('val1',1,1.1),('val2',2,2.2)"  # str
    """
    res = ",".join([_escape_common(i, encoding) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_range(data: object) -> str:
    """(cfunc) Escape range 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_range(range(1, 4))
    >>> "(1,2,3)"  # str
    """
    return "(" + ",".join([str(i) for i in data]) + ")"  # _escape_int


# . Mapping types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_dict(data: dict, encoding: cython.pchar) -> str:
    """(cfunc) Escape dict 'data' to literal `<'str'>`.

    ### Example (flat):
    >>> _escape_dict(
        {"key1": "val1", "key2": 1, "key3": 1.1})
    >>> "('val1',1,1.1)"  # str

    ### Example (nested):
    >>> _escape_dict(
        {"key1": ["val1", 1, 1.1], "key2": ["val2", 2, 2.2]})
    >>> "('val1',1,1.1),('val2',2,2.2)"  # str
    """
    res = ",".join([_escape_common(i, encoding) for i in data.values()])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


# . Numpy ndarray - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_ndarray(data: np.ndarray, encoding: cython.pchar) -> str:
    """(cfunc) Escape numpy.ndarray 'data' to literal `<'str'>`.

    ### Example (1-dimension):
    >>> _escape_ndarray(np.array([1, 2, 3], dtype=np.int64))
    >>> "(1,2,3)"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray(np.array(
        [[1, 2, 3], [3, 4, 5]], dtype=np.int64))
    >>> "(1,2,3),(3,4,5)"  # str
    """
    # Get ndarray dtype
    dtype: cython.char = data.descr.kind

    # . ndarray[object]
    if dtype == NDARRAY_DTYPE_OBJECT:
        return _escape_ndarray_object(data, encoding)
    # . ndarray[float]
    if dtype == NDARRAY_DTYPE_FLOAT:
        return _escape_ndarray_float(data)
    # . ndarray[int]
    if dtype == NDARRAY_DTYPE_INT:
        return _escape_ndarray_int(data)
    # . ndarray[uint]
    if dtype == NDARRAY_DTYPE_UINT:
        return _escape_ndarray_int(data)
    # . ndarray[bool]
    if dtype == NDARRAY_DTYPE_BOOL:
        return _escape_ndarray_bool(data)
    # . ndarray[datetime64]
    if dtype == NDARRAY_DTYPE_DT64:
        return _escape_ndarray_dt64(data)
    # . ndarray[timedelta64]
    if dtype == NDARRAY_DTYPE_TD64:
        return _escape_ndarray_td64(data)
    # . ndarray[bytes]
    if dtype == NDARRAY_DTYPE_BYTES:
        return _escape_ndarray_bytes(data)
    # . ndarray[str]
    if dtype == NDARRAY_DTYPE_UNICODE:
        return _escape_ndarray_unicode(data, encoding)
    # # . invalid dtype
    raise TypeError("Unsupported <'numpy.ndarray'> dtype [%s]." % data.dtype)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_object(arr: np.ndarray, encoding: cython.pchar) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' to literal `<'str'>`.

    #### This function is for ndarray dtype `"O" (object)` only.

    ### Example (1-dimension):
    >>> _escape_ndarray_object(np.array([1, 1.23, "abc"], dtype="O"))
    >>> "(1,1.23,'abc')"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray_object(np.array(
        [[1, 1.23, "abc"], [2, 4.56, "def"]], dtype="O"))
    >>> "(1,1.23,'abc'),(2,4.56,'def')"  # str
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return "()"  # exit
        # . escape to literal
        # fmt: off
        l_i = [
            _escape_common(arr_getitem_1d(arr, i), encoding)  # type: ignore
            for i in range(s_i)
        ]
        # fmt: on
        return "(" + ",".join(l_i) + ")"
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return "()"  # exit
        # . escape to literal
        # fmt: off
        l_i = [
            "(" + ",".join([
                _escape_common(arr_getitem_2d(arr, i, j), encoding)  # type: ignore
                for j in range(s_j)]) + ")"
            for i in range(s_i)
        ]
        # fmt: on
        return ",".join(l_i)
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_float(arr: np.ndarray) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' to literal `<'str'>`.

    :raises `<'ValueError'>`: If float value is invalid (not finite).

    #### This function is for ndarray dtype `"f" (float)` only.

    ### Example (1-dimension):
    >>> _escape_ndarray_float(np.array([-1.1, 0.0, 1.1], dtype=float))
    >>> "(-1.1,0.0,1.1)"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray_float(np.array(
        [[-1.1, 0.0], [1.1, 2.2]], dtype=float))
    >>> "(-1.1,0.0),(1.1,2.2)"  # str
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return "()"  # exit
        # . check if value is finite
        if not is_arr_float_finite_1d(arr, s_i):  # type: ignore
            raise ValueError("Float value 'nan' & 'inf' not supported.")
        # . escape to literal
        res = _orjson_dumps_numpy(arr)
        return replace_bracket(res, 1)  # type: ignore
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return "()"  # exit
        # . check if value is finite
        if not is_arr_float_finite_2d(arr, s_i, s_j):  # type: ignore
            raise ValueError("Float value 'nan' & 'inf' not supported.")
        # . escape to literal
        res = _orjson_dumps_numpy(arr)
        if read_char(res, 1) == "[":
            res = str_substr(res, 1, str_len(res) - 1)
        return replace_bracket(res, -1)  # type: ignore
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_int(arr: np.ndarray) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' to literal `<'str'>`.

    #### This function is ndarray dtype `"i" (int)` and `"u" (uint)`.

    ### Example (1-dimension):
    >>> _escape_ndarray_int(np.array([-1, -2, -3], dtype=int))
    >>> "(-1,-2,-3)"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray_int(np.array(
        [[1, 2], [3, 4]], dtype=np.uint64))
    >>> "(1,2),(3,4)"  # str
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return "()"  # exit
        # . escape to literal
        res = _orjson_dumps_numpy(arr)
        return replace_bracket(res, 1)  # type: ignore
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return "()"  # exit
        # . escape to literal
        res = _orjson_dumps_numpy(arr)
        if read_char(res, 1) == "[":
            res = str_substr(res, 1, str_len(res) - 1)
        return replace_bracket(res, -1)  # type: ignore
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_bool(arr: np.ndarray) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' to literal `<'str'>`.

    #### This function is for ndarray dtype `"b" (bool)` only.

    ### Example (1-dimension):
    >>> _escape_ndarray_bool(np.array([True, False, True], dtype=bool))
    >>> "(1,0,1)"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray_bool(np.array(
        [[True, False], [False, True]], dtype=bool))
    >>> "(1,0),(0,1)"  # str
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return "()"  # exit
        # . escape to literal
        res = _orjson_dumps_numpy(np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64))
        return replace_bracket(res, 1)  # type: ignore
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return "()"  # exit
        # . escape to literal
        res = _orjson_dumps_numpy(np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64))
        if read_char(res, 1) == "[":
            res = str_substr(res, 1, str_len(res) - 1)
        return replace_bracket(res, -1)  # type: ignore
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_dt64(arr: np.ndarray) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' to literal `<'str'>`.

    #### This function is for ndarray dtype `"M" (datetime64)` only.

    ### Example (1-dimension):
    >>> _escape_ndarray_dt64(np.array([1, 2, 3], dtype="datetime64[s]"))
    >>> "('1970-01-01 00:00:01','1970-01-01 00:00:02','1970-01-01 00:00:03')"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray_dt64(np.array(
        [[1, 2], [3, 4]], dtype="datetime64[s]"))
    >>> "('1970-01-01 00:00:01','1970-01-01 00:00:02'),('1970-01-01 00:00:03','1970-01-01 00:00:04')"  # str
    """
    # Notes: 'orjson' returns '["1970-01-01T00:00:00",...,"2000-01-01T00:00:01"]',
    # so character ['"', "T", "[", "]"] should be replaced to comply with literal
    # datetime format.
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return "()"  # exit
        # . escape to literal
        res = _orjson_dumps_numpy(arr)
        return res.translate(DT64_JSON_TABLE)
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return "()"  # exit
        # . escape to literal
        res = _orjson_dumps_numpy(arr)
        if read_char(res, 1) == "[":
            res = str_substr(res, 1, str_len(res) - 1)
        return res.translate(DT64_JSON_TABLE)
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_ndarray_td64(arr: np.ndarray) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' to literal `<'str'>`.

    #### This function is for ndarray dtype `"m" (timedelta64)` only.

    ### Example (1-dimension):
    >>> _escape_ndarray_td64(np.array([-1, 0, 1], dtype="timedelta64[s]"))
    >>> "('-00:00:01','00:00:00','00:00:01')"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray_td64(np.array(
        [[-1, 0], [1, 2]], dtype="timedelta64[s]"))
    >>> "('-00:00:01','00:00:00'),('00:00:01','00:00:02')"  # str
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return "()"  # exit
        # . escape to literal
        unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
        l_i = []
        for i in range(s_i):
            us: cython.longlong = arr_getitem_1d_ll(arr, i)  # type: ignore
            us = nptime_to_microseconds(us, unit)  # type: ignore
            l_i.append(_escape_timedelta64_fr_us(us))
        return "(" + ",".join(l_i) + ")"
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return "()"  # exit
        # . escape to literal
        unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0, 0])
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
        l_i = []
        for i in range(s_i):
            l_j = []
            for j in range(s_j):
                us: cython.longlong = arr_getitem_2d_ll(arr, i, j)  # type: ignore
                us = nptime_to_microseconds(us, unit)  # type: ignore
                l_j.append(_escape_timedelta64_fr_us(us))
            l_i.append("(" + ",".join(l_j) + ")")
        return ",".join(l_i)
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_bytes(arr: np.ndarray) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' to literal `<'str'>`.

    #### This function is for ndarray dtype `"S" (bytes string)` only.

    ### Example (1-dimension):
    >>> _escape_ndarray_bytes(np.array([1, 2, 3], dtype="S"))
    >>> "(_binary'1',_binary'2',_binary'3')"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray_bytes(np.array(
        [[1, 2], [3, 4]], dtype="S"))
    >>> "(_binary'1',_binary'2'),(_binary'3',_binary'4')"  # str
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return "()"  # exit
        # . escape to literal
        l_i = [_escape_bytes(arr_getitem_1d(arr, i)) for i in range(s_i)]  # type: ignore
        return "(" + ",".join(l_i) + ")"
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return "()"  # exit
        # . escape to literal
        l_i = [
            "(" + ",".join([_escape_bytes(arr_getitem_2d(arr, i, j)) for j in range(s_j)]) + ")"  # type: ignore
            for i in range(s_i)
        ]
        return ",".join(l_i)
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_unicode(arr: np.ndarray, encoding: cython.pchar) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' to literal `<'str'>`.

    #### This function is for ndarray dtype `"U" (unicode string)` only.

    ### Example (1-dimension):
    >>> _escape_ndarray_unicode(np.array([1, 2, 3], dtype="U"))
    >>> "('1','2','3')"  # str

    ### Example (2-dimension):
    >>> _escape_ndarray_unicode(np.array(
        [[1, 2], [3, 4]], dtype="U"))
    >>> "('1','2'),('3','4')"  # str
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return "()"  # exit
        # . escape to literal
        # fmt: off
        l_i = [
            _escape_str(arr_getitem_1d(arr, i), encoding)  # type: ignore
            for i in range(s_i)
        ]
        # fmt: on
        return "(" + ",".join(l_i) + ")"
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return "()"  # exit
        # . escape to literal
        # fmt: off
        l_i = [
            "(" + ",".join([
                _escape_str(arr_getitem_2d(arr, i, j), encoding)  # type: ignore
                for j in range(s_j)])+ ")"
            for i in range(s_i)
        ]
        # fmt: on
        return ",".join(l_i)
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


# . Pandas Series - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_series(data: Series, encoding: cython.pchar) -> str:
    """(cfunc) Escape pandas.Series 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_series(pd.Series([1, 2, 3]))
    >>> "(1,2,3)"  # str
    """
    try:
        arr: np.ndarray = data.values
    except Exception as err:
        raise TypeError("Expects <'pandas.Series'>, got %s." % type(data)) from err
    return _escape_ndarray(arr, encoding)


# . Pandas DataFrame - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_dataframe(data: DataFrame, encoding: cython.pchar) -> str:
    """(cfunc) Escape pandas.DataFrame 'data' to literal `<'str'>`.

    ### Example:
    >>> _escape_dataframe(pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.1, 2.2, 3.3],
            "c": ["a", "b", "c"],
        }))
    >>> "(1,1.1,'a'),(2,2.2,'b'),(3,3.3,'c')"  # str
    """
    # Validate shape
    shape: tuple = data.shape
    width: cython.Py_ssize_t = shape[1]
    if width == 0:
        return "()"  # exit
    size: cython.Py_ssize_t = shape[0]
    if size == 0:
        return "()"  # exit
    # Escape DataFrame
    # fmt: off
    cols = [
        _escape_item_ndarray(col.values, encoding, False) 
        for _, col in data.items()
    ]
    # fmt: on
    rows = [
        "(" + ",".join([cython.cast(object, tuple_getitem(cols[j], i)) for j in range(width)]) + ")"  # type: ignore
        for i in range(size)
    ]
    return ",".join(rows)


# . Escape - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_common(data: object, encoding: cython.pchar) -> str:
    """(cfunc) Escape common 'data' to literal `<'str'>`."""
    # Get data type
    dtype = type(data)

    ##### Common Types #####
    # Basic Types
    # . <'str'>
    if dtype is str:
        return _escape_str(data, encoding)
    # . <'float'>
    if dtype is float:
        return _escape_float(data)
    # . <'int'>
    if dtype is int:
        return str(data)  # _escape_int
    # . <'bool'>
    if dtype is bool:
        return _escape_bool(data)
    # . <None>
    if dtype is typeref.NONE:
        return "NULL"  # _escape_none

    # Date&Time Types
    # . <'datetime.datetime'>
    if dtype is datetime.datetime:
        return _escape_datetime(data)
    # . <'datetime.date'>
    if dtype is datetime.date:
        return _escape_date(data)
    # . <'datetime.time'>
    if dtype is datetime.time:
        return _escape_time(data)
    # . <'datetime.timedelta'>
    if dtype is datetime.timedelta:
        return _escape_timedelta(data)

    # Numeric Types
    # . <'decimal.Decimal'>
    if dtype is typeref.DECIMAL:
        return str(data)  # _escape_decimal

    # Bytes Types
    # . <'bytes'>
    if dtype is bytes:
        return _escape_bytes(data)
    # . <'bytearray'>
    if dtype is bytearray:
        return _escape_bytearray(data)

    # Sequence Types
    # . <'tuple'>
    if dtype is tuple:
        return _escape_tuple(data, encoding)
    # . <'list'>
    if dtype is list:
        return _escape_list(data, encoding)
    # . <'set'>
    if dtype is set:
        return _escape_set(data, encoding)

    # Mapping Types
    # . <'dict'>
    if dtype is dict:
        return _escape_dict(data, encoding)

    ##### Uncommon Types #####
    return _escape_uncommon(data, encoding, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_uncommon(data: object, encoding: cython.pchar, dtype: type) -> str:
    """(cfunc) Escape uncommon 'data' to literal `<'str'>`."""
    ##### Uncommon Types #####
    # Basic Types
    # . <'numpy.float_'>
    if dtype is typeref.FLOAT64 or dtype is typeref.FLOAT32 or dtype is typeref.FLOAT16:
        return _escape_float64(data)
    # . <'numpy.int_'>
    if (
        dtype is typeref.INT64
        or dtype is typeref.INT32
        or dtype is typeref.INT16
        or dtype is typeref.INT8
    ):
        return str(data)  # _escape_int
    # . <'numpy.uint'>
    if (
        dtype is typeref.UINT64
        or dtype is typeref.UINT32
        or dtype is typeref.UINT16
        or dtype is typeref.UINT8
    ):
        return str(data)  # _escape_int
    # . <'numpy.bool_'>
    if dtype is typeref.BOOL_:
        return _escape_bool(data)

    # Date&Time Types
    # . <'pandas.Timestamp'>
    if dtype is typeref.TIMESTAMP:
        return _escape_datetime(data)
    # . <'pandas.Timedelta'>`
    if dtype is typeref.TIMEDELTA:
        return _escape_timedelta(data)
    # . <'numpy.datetime64'>
    if dtype is typeref.DATETIME64:
        return _escape_datetime64(data)
    # . <'numpy.timedelta64'>
    if dtype is typeref.TIMEDELTA64:
        return _escape_timedelta64(data)
    # . <'time.struct_time'>`
    if dtype is typeref.STRUCT_TIME:
        return _escape_struct_time(data)

    # Bytes Types
    # . <'memoryview'>
    if dtype is memoryview:
        return _escape_memoryview(data)
    # . <'numpy.bytes_'>
    if dtype is typeref.BYTES_:
        return _escape_bytes(data)

    # String Types:
    # . <'numpy.str_'>
    if dtype is typeref.STR_:
        return _escape_str(data, encoding)

    # Sequence Types
    # . <'frozenset'>
    if dtype is frozenset:
        return _escape_frozenset(data, encoding)
    # . <'range'>
    if dtype is range:
        return _escape_range(data)
    # . <'dict_keys'> & <'dict_values'>
    if dtype is typeref.DICT_VALUES or dtype is typeref.DICT_KEYS:
        return _escape_sequence(data, encoding)

    # Numpy Types
    # . <'numpy.ndarray'>
    if dtype is np.ndarray:
        return _escape_ndarray(data, encoding)
    # . <'numpy.record'>
    if dtype is typeref.RECORD:
        return _escape_sequence(data, encoding)

    # Pandas Types
    # . <'pandas.Series'> & <'pandas.DatetimeIndex'> & <'pandas.TimedeltaIndex'>
    if (
        dtype is typeref.SERIES
        or dtype is typeref.DATETIMEINDEX
        or dtype is typeref.TIMEDELTAINDEX
    ):
        return _escape_series(data, encoding)
    # . <'pandas.DataFrame'>
    if dtype is typeref.DATAFRAME:
        return _escape_dataframe(data, encoding)

    # Custom Types
    if dtype is BIT:
        return _escape_bit(data)
    if dtype is JSON:
        return _escape_json(data, encoding)

    # Cytimes Types
    if typeref.CYTIMES_AVAILABLE:
        # . <'cytimes.pydt'>
        if dtype is typeref.PYDT:
            return _escape_datetime(data.dt)
        # . <'cytimes.pddt'>
        if dtype is typeref.PDDT:
            return _escape_series(data.dt, encoding)

    ##### Subclass Types #####
    return _escape_subclass(data, encoding, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_subclass(data: object, encoding: cython.pchar, dtype: type) -> str:
    """(cfunc) Escape subclass 'data' to literal `<'str'>`."""
    ##### Subclass Types #####
    # Basic Types
    # . subclass of <'str'>
    if isinstance(data, str):
        return _escape_str(data, encoding)
    # . subclass of <'float'>
    if isinstance(data, float):
        return _escape_float(data)
    # . subclass of <'int'>
    if isinstance(data, int):
        return str(data)  # _escape_int
    # . subclass of <'bool'>
    if isinstance(data, bool):
        return _escape_bool(data)

    # Date&Time Types
    # . subclass of <'datetime.datetime'>
    if isinstance(data, datetime.datetime):
        return _escape_datetime(data)
    # . subclass of <'datetime.date'>
    if isinstance(data, datetime.date):
        return _escape_date(data)
    # . subclass of <'datetime.time'>
    if isinstance(data, datetime.time):
        return _escape_time(data)
    # . subclass of <'datetime.timedelta'>
    if isinstance(data, datetime.timedelta):
        return _escape_timedelta(data)

    # Numeric Types
    # . subclass of <'decimal.Decimal'>
    if isinstance(data, typeref.DECIMAL):
        return str(data)  # _escape_decimal

    # Bytes Types
    # . subclass of <'bytes'>
    if isinstance(data, bytes):
        return _escape_bytes(data)
    # . subclass of <'bytearray'>
    if isinstance(data, bytearray):
        return _escape_bytearray(data)

    # Sequence Types
    # . subclass of <'tuple'>
    if isinstance(data, tuple):
        return _escape_tuple(data, encoding)
    # . subclass of <'list'>
    if isinstance(data, list):
        return _escape_list(data, encoding)
    # . subclass of <'set'>
    if isinstance(data, set):
        return _escape_set(data, encoding)
    # . subclass of <'frozenset'>
    if isinstance(data, frozenset):
        return _escape_frozenset(data, encoding)

    # Mapping Types
    # . subclass of <'dict'>
    if isinstance(data, dict):
        return _escape_dict(data, encoding)

    # Invalid Data Type
    raise TypeError("Unsupported 'data' type %s." % dtype)


# Escape Item ---------------------------------------------------------------------------------
# . Sequence types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_list(
    data: list,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of list 'data' to
    sequence(s) of literals `<'tuple/list'>`.

    ### Example (many=False | flat):
    >>> _escape_item_list(
        ["val1", 1, 1.1], False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    ### Exmaple (many=False | nested):
    >>> _escape_item_list(
        [["val1", 1, 1.1], ["val2", 2, 2.2]], False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    ### Example (many=True | flat):
    >>> _escape_item_list(
        ["val1", 1, 1.1], True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    ### Example (many=True | nested):
    >>> _escape_item_list(
        [["val1", 1, 1.1], ["val2", 2, 2.2]], False)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    """
    if not many:
        return list_to_tuple([_escape_common(i, encoding) for i in data])
    else:
        return [_escape_item_common(i, encoding, False) for i in data]


@cython.cfunc
@cython.inline(True)
def _escape_item_tuple(
    data: tuple,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of tuple 'data' to
    sequence(s) of literals `<'tuple/list'>`.

    ### Example (many=False | flat):
    >>> _escape_item_list(
        ("val1", 1, 1.1), False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    ### Exmaple (many=False | nested):
    >>> _escape_item_list(
        (("val1", 1, 1.1), ("val2", 2, 2.2)), False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    ### Example (many=True | flat):
    >>> _escape_item_list(
        ("val1", 1, 1.1), True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    ### Example (many=True | nested):
    >>> _escape_item_list(
        (("val1", 1, 1.1), ("val2", 2, 2.2)), False)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    """
    if not many:
        return list_to_tuple([_escape_common(i, encoding) for i in data])
    else:
        return [_escape_item_common(i, encoding, False) for i in data]


@cython.cfunc
@cython.inline(True)
def _escape_item_set(
    data: set,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of set 'data' to
    sequence(s) of literals `<'tuple/list'>`.

    ### Example (many=False | flat):
    >>> _escape_item_set(
        {"val1", 1, 1.1}, False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    ### Exmaple (many=False | nested):
    >>> _escape_item_set(
        {("val1", 1, 1.1), ("val2", 2, 2.2)}, False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    ### Example (many=True | flat):
    >>> _escape_item_set(
        {"val1", 1, 1.1}, True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    ### Example (many=True | nested):
    >>> _escape_item_set(
        {("val1", 1, 1.1), ("val2", 2, 2.2)}, False)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    """
    if not many:
        return list_to_tuple([_escape_common(i, encoding) for i in data])
    else:
        return [_escape_item_common(i, encoding, False) for i in data]


@cython.cfunc
@cython.inline(True)
def _escape_item_frozenset(
    data: frozenset,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of frozenset 'data' to
    sequence(s) of literals `<'tuple/list'>`.

    ### Example (many=False | flat):
    >>> _escape_item_frozenset(frozenset(
        {"val1", 1, 1.1}), False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    ### Exmaple (many=False | nested):
    >>> _escape_item_frozenset(frozenset(
        {("val1", 1, 1.1), ("val2", 2, 2.2)}), False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    ### Example (many=True | flat):
    >>> _escape_item_frozenset(frozenset(
        {"val1", 1, 1.1}), True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    ### Example (many=True | nested):
    >>> _escape_item_frozenset(frozenset(
        {("val1", 1, 1.1), ("val2", 2, 2.2)}), False)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    """
    if not many:
        return list_to_tuple([_escape_common(i, encoding) for i in data])
    else:
        return [_escape_item_common(i, encoding, False) for i in data]


@cython.cfunc
@cython.inline(True)
def _escape_item_sequence(
    data: Iterable,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of sequence 'data' to
    sequence(s) of literals `<'tuple/list'>`.

    ### Exmaple (many=False | flat):
    >>> _escape_item_dict(
        {"key1": "val1", "key2": 1, "key3": 1.1}.values(), False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    ### Example (many=False | nested):
    >>> _escape_item_dict(
        {"key1": ["val1", 1, 1.1], "key2": ["val2", 2, 2.2]}.values(), False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    ### Exmaple (many=True | flat):
    >>> _escape_item_dict(
        {"key1": "val1", "key2": 1, "key3": 1.1}.values(), True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    ### Example (many=True | nested):
    >>> _escape_item_dict(
        {"key1": ["val1", 1, 1.1], "key2": ["val2", 2, 2.2]}.values(), True)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    """
    if not many:
        return list_to_tuple([_escape_common(i, encoding) for i in data])
    else:
        return [_escape_item_common(i, encoding, False) for i in data]


@cython.cfunc
@cython.inline(True)
def _escape_item_range(data: object, many: cython.bint) -> object:
    """(cfunc) Escape items of range 'data' to
    sequence of literals `<'tuple/list'>`.

    ### Example (many=False):
    >>> _escape_item_range(range(1, 4), False)
    >>> ("1", "2", "3")  # tuple[str]

    ### Example (many=True):
    >>> _escape_item_range(range(1, 4), True)
    >>> ["1", "2", "3"]  # list[str]
    """
    l = [str(i) for i in data]  # _escape_int
    return list_to_tuple(l) if not many else l


# . Mapping types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_dict(
    data: dict,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of dict 'data' to
    sequence(s) of literals `<'tuple/list'>`.

    ### Exmaple (many=False | flat):
    >>> _escape_item_dict(
        {"key1": "val1", "key2": 1, "key3": 1.1}, False)
    >>> ("'val1'", "1", "1.1")  # tuple[str]

    ### Example (many=False | nested):
    >>> _escape_item_dict(
        {"key1": ["val1", 1, 1.1], "key2": ["val2", 2, 2.2]}, False)
    >>> ("('val1',1,1.1)", "('val2',2,2.2)")  # tuple[str]

    ### Exmaple (many=True | flat):
    >>> _escape_item_dict(
        {"key1": "val1", "key2": 1, "key3": 1.1}, True)
    >>> ["'val1'", "1", "1.1"]  # list[str]

    ### Example (many=True | nested):
    >>> _escape_item_dict(
        {"key1": ["val1", 1, 1.1], "key2": ["val2", 2, 2.2]}, True)
    >>> [("'val1'", "1", "1.1"), ("'val2'", "2", "2.2")]  # list[tuple[str]]
    """
    if not many:
        return list_to_tuple([_escape_common(i, encoding) for i in data.values()])
    else:
        return [_escape_item_common(i, encoding, False) for i in data.values()]


# . Numpy ndarray - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray(
    data: np.ndarray,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of numpy.ndarray 'data' to
    sequence(s) of literals `<'tuple/list'>`.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray(np.array([1, 2, 3]), False)
    >>> ("1", "2", "3")  # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray(np.array([1, 2, 3]), True)
    >>> ["1", "2", "3"]  # list[tuple[str]]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray(np.array(
        [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]), True)
    >>> [("1.1", "2.2", "3.3"), ("4.4", "5.5", "6.6")]  # list[tuple[str]]
    """
    # Get ndarray dtype
    dtype: cython.char = data.descr.kind

    # . ndarray[object]
    if dtype == NDARRAY_DTYPE_OBJECT:
        return _escape_item_ndarray_object(data, encoding, many)
    # . ndarray[float]
    if dtype == NDARRAY_DTYPE_FLOAT:
        return _escape_item_ndarray_float(data, many)
    # . ndarray[int]
    if dtype == NDARRAY_DTYPE_INT:
        return _escape_item_ndarray_int(data, many)
    # . ndarray[uint]
    if dtype == NDARRAY_DTYPE_UINT:
        return _escape_item_ndarray_int(data, many)
    # . ndarray[bool]
    if dtype == NDARRAY_DTYPE_BOOL:
        return _escape_item_ndarray_bool(data, many)
    # . ndarray[datetime64]
    if dtype == NDARRAY_DTYPE_DT64:
        return _escape_item_ndarray_dt64(data, many)
    # . ndarray[timedelta64]
    if dtype == NDARRAY_DTYPE_TD64:
        return _escape_item_ndarray_td64(data, many)
    # . ndarray[bytes]
    if dtype == NDARRAY_DTYPE_BYTES:
        return _escape_item_ndarray_bytes(data, many)
    # . ndarray[str]
    if dtype == NDARRAY_DTYPE_UNICODE:
        return _escape_item_ndarray_unicode(data, encoding, many)
    # . invalid dtype
    raise TypeError("Unsupported <'numpy.ndarray'> dtype [%s]." % data.dtype)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_object(
    arr: np.ndarray,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of numpy.ndarray 'arr' to
    sequence(s) of literals `<'tuple/list'>`.

    #### This function is for ndarray dtype `"O" (object)` only.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray_object(np.array([1, 1.23, "abc"], dtype="O"), False)
    >>> ("1", "1.23", "'abc'")           # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray_object(np.array([1, 1.23, "abc"], dtype="O"), True)
    >>> ["1", "1.23", "'abc'"]  # list[str]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray_object(np.array(
        [[1, 1.23, "abc"], [2, 4.56, "def"]], dtype="O"))
    >>> [("1", "1.23", "'abc'"), ("2", "4.56", "'def'")]  # list[tupe[str]]
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return ()  # exit
        # . escape items to literals
        if not many:
            # fmt: off
            return list_to_tuple([
                _escape_common(arr_getitem_1d(arr, i), encoding)  # type: ignore
                for i in range(s_i)
            ])
        else:
            return [
                _escape_item_common(arr_getitem_1d(arr, i), encoding, False)  # type: ignore
                for i in range(s_i)
            ]
            # fmt: on
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return []  # exit
        # . escape items to literals
        # fmt: off
        return [
            list_to_tuple([
                _escape_common(arr_getitem_2d(arr, i, j), encoding)  # type: ignore
                for j in range(s_j)])
            for i in range(s_i)
        ]
        # fmt: on
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_float(arr: np.ndarray, many: cython.bint) -> object:
    """(cfunc) Escape items of numpy.ndarray arr to
    sequence(s) of literals `<'tuple/list'>`.

    :raises `<'ValueError'>`: If float value is invalid (not finite).

    #### This function is for ndarray dtype `"f" (float)` only.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray_float(np.array([-1.1, 0.0, 1.1], dtype=float), False)
    >>> ("-1.1", "0.0", "1.1")  # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray_float(np.array([-1.1, 0.0, 1.1], dtype=float), True)
    >>> ["-1.1", "0.0", "1.1"]  # list[str]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray_float(np.array(
        [[-1.1, 0.0], [1.1, 2.2]], dtype=float))
    >>> [("-1.1", "0.0"), ("1.1", "2.2")]  # list[tupe[str]]
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return ()  # exit
        # . check if value is finite
        if not is_arr_float_finite_1d(arr, s_i):  # type: ignore
            raise ValueError("Float value 'nan' & 'inf' is not supported.")
        # . escape items to literals
        res = _orjson_dumps_numpy(arr)
        l_i = str_split(str_substr(res, 1, str_len(res) - 1), ",", -1)
        return list_to_tuple(l_i) if not many else l_i
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return []  # exit
        # . check if value is finite
        if not is_arr_float_finite_2d(arr, s_i, s_j):  # type: ignore
            raise ValueError("Float value 'nan' & 'inf' is not supported.")
        # . escape items to literals
        res = _orjson_dumps_numpy(arr)
        l_i = str_split(str_substr(res, 2, str_len(res) - 2), "],[", -1)
        return [list_to_tuple(str_split(i, ",", -1)) for i in l_i]
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_int(arr: np.ndarray, many: cython.bint) -> object:
    """(cfunc) Escape items of numpy.ndarray 'arr' to
    sequence(s) of literals `<'tuple/list'>`.

    #### This function is ndarray dtype `"i" (int)` and `"u" (uint)`.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray_int(np.array([-1, 0, 1], dtype=int), False)
    >>> ("-1", "0", "1")  # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray_int(np.array([-1, 0, 1], dtype=int), True)
    >>> ["-1", "0", "1"]  # list[str]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray_int(np.array(
        [[0, 1], [2, 3]], dtype=np.uint))
    >>> [("0", "1"), ("2", "3")]  # list[tupe[str]]
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return ()  # exit
        # . escape items to literals
        res = _orjson_dumps_numpy(arr)
        l_i = str_split(str_substr(res, 1, str_len(res) - 1), ",", -1)
        return list_to_tuple(l_i) if not many else l_i
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return []  # exit
        # . escape items to literals
        res = _orjson_dumps_numpy(arr)
        l_i = str_split(str_substr(res, 2, str_len(res) - 2), "],[", -1)
        return [list_to_tuple(str_split(i, ",", -1)) for i in l_i]
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_bool(arr: np.ndarray, many: cython.bint) -> object:
    """(cfunc) Escape items of numpy.ndarray 'arr' to
    sequence(s) of literals `<'tuple/list'>`.

    #### This function is for ndarray dtype `"b" (bool)` only.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray_bool(np.array([True, False, True], dtype=bool), False)
    >>> ("1", "0", "1")  # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray_bool(np.array([True, False, True], dtype=bool), True)
    >>> ["1", "0", "1"]  # list[str]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray_bool(np.array(
        [[True, False], [False, True]], dtype=bool))
    >>> [("1", "0"), ("0", "1")]  # list[tupe[str]]
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return ()  # exit
        # . escape items to literals
        if not many:
            return list_to_tuple(["1" if arr_getitem_1d_bint(arr, i) else "0" for i in range(s_i)])  # type: ignore
        else:
            return ["1" if arr_getitem_1d_bint(arr, i) else "0" for i in range(s_i)]  # type: ignore
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return []  # exit
        # . escape items to literals
        return [
            list_to_tuple(["1" if arr_getitem_2d_bint(arr, i, j) else "0" for j in range(s_j)])  # type: ignore
            for i in range(s_i)
        ]
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_dt64(arr: np.ndarray, many: cython.bint) -> object:
    """(cfunc) Escape items of numpy.ndarray 'arr' to
    sequence(s) of literals `<'tuple/list'>`.

    #### This function is for ndarray dtype `"M" (datetime64)` only.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray_dt64(np.array([1, 2, 3], dtype="datetime64[s]"), False)
    >>> ("'1970-01-01 00:00:01'", "'1970-01-01 00:00:02'", "'1970-01-01 00:00:03'")  # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray_dt64(np.array([1, 2, 3], dtype="datetime64[s]"), True)
    >>> ["'1970-01-01 00:00:01'", "'1970-01-01 00:00:02'", "'1970-01-01 00:00:03'"]  # list[str]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray_dt64(np.array(
        [[1, 2], [3, 4]], dtype="datetime64[s]))
    >>> [
            ("'1970-01-01 00:00:01'", "'1970-01-01 00:00:02'"),
            ("'1970-01-01 00:00:03'", "'1970-01-01 00:00:04'"),
        ]  # list[tuple[str]]
    """
    # Notes: 'orjson' returns '["1970-01-01T00:00:00",...,"2000-01-01T00:00:01"]',
    # so character ['"', "T", "[", "]"] should be replaced to comply with literal
    # datetime format.
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return ()  # exit
        # . escape items to literals
        res = _orjson_dumps_numpy(arr)
        res = str_substr(res, 1, str_len(res) - 1)
        l_i = str_split(res.translate(DT64_JSON_TABLE), ",", -1)
        return list_to_tuple(l_i) if not many else l_i
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return []  # exit
        # . escape items to literals
        res = _orjson_dumps_numpy(arr)
        res = res.translate(DT64_JSON_TABLE)
        l_i = str_split(str_substr(res, 2, str_len(res) - 2), "),(", -1)
        return [list_to_tuple(str_split(i, ",", -1)) for i in l_i]
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_td64(arr: np.ndarray, many: cython.bint) -> object:
    """(cfunc) Escape items of numpy.ndarray 'arr' to
    sequence(s) of literals `<'tuple/list'>`.

    #### This function is for ndarray dtype `"m" (timedelta64)` only.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray_td64(np.array([-1, 0, 1], dtype="timedelta64[s]"), False)
    >>> ("'-00:00:01'", "'00:00:00'", "'00:00:01'")  # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray_td64(np.array([-1, 0, 1], dtype="timedelta64[s]"), True)
    >>> ["'-00:00:01'", "'00:00:00'", "'00:00:01'"]  # list[str]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray_td64(np.array(
        [[-1, 0], [1, 2]], dtype="timedelta64[s]"))
    >>> [("'-00:00:01'", "'00:00:00'"), ("'00:00:01'", "'00:00:02'")]  # list[tuple[str]]
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return ()  # exit
        # . escape items to literals
        unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
        l_i = []
        for i in range(s_i):
            us: cython.longlong = arr_getitem_1d_ll(arr, i)  # type: ignore
            us = nptime_to_microseconds(us, unit)  # type: ignore
            l_i.append(_escape_timedelta64_fr_us(us))
        return list_to_tuple(l_i) if not many else l_i
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return []  # exit
        # . escape items to literals
        unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0, 0])
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
        l_i = []
        for i in range(s_i):
            l_j = []
            for j in range(s_j):
                us: cython.longlong = arr_getitem_2d_ll(arr, i, j)  # type: ignore
                us = nptime_to_microseconds(us, unit)  # type: ignore
                l_j.append(_escape_timedelta64_fr_us(us))
            l_i.append(list_to_tuple(l_j))
        return l_i
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_bytes(arr: np.ndarray, many: cython.bint) -> object:
    """(cfunc) Escape items of numpy.ndarray 'arr' to
    sequence(s) of literals `<'tuple/list'>`.

    #### This function is for ndarray dtype `"S" (bytes string)` only.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray_bytes(np.array([1, 2, 3], dtype="S"), False)
    >>> ("_binary'1'", "_binary'2'", "_binary'3'")  # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray_bytes(np.array([1, 2, 3], dtype="S"), True)
    >>> ["_binary'1'", "_binary'2'", "_binary'3'"]  # list[str]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray_bytes(np.array(
        [[1, 2], [3, 4]], dtype="S"))
    >>> [("_binary'1'", "_binary'2'"), ("_binary'3'", "_binary'4'")]  # list[tuple[str]]
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return ()  # exit
        # . escape items to literals
        if not many:
            return list_to_tuple([_escape_bytes(arr_getitem_1d(arr, i)) for i in range(s_i)])  # type: ignore
        else:
            return [_escape_bytes(arr_getitem_1d(arr, i)) for i in range(s_i)]  # type: ignore
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return []  # exit
        # . escape items to literals
        return [
            list_to_tuple([_escape_bytes(arr_getitem_2d(arr, i, j)) for j in range(s_j)])  # type: ignore
            for i in range(s_i)
        ]
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_unicode(
    arr: np.ndarray,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of numpy.ndarray 'arr' to
    sequence(s) of literals `<'tuple/list'>`.

    #### This function is for ndarray dtype `"U" (unicode string)` only.

    ### Example (1-dimension | many=False)
    >>> _escape_item_ndarray_bytes(np.array([1, 2, 3], dtype="U"), False)
    >>> ("'1'", "'2'", "'3'")  # tuple[str]

    ### Example (1-dimension | many=True)
    >>> _escape_item_ndarray_bytes(np.array([1, 2, 3], dtype="U"), True)
    >>> ["'1'", "'2'", "'3'"]  # list[str]

    ### Example (2-dimension | many [ignored])
    >>> _escape_item_ndarray_bytes(np.array(
        [[1, 2], [3, 4]], dtype="U"))
    >>> [("'1'", "'2'"), ("'3'", "'4'")]  # list[tuple[str]]
    """
    ndim: cython.Py_ssize_t = arr.ndim
    shape = arr.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j = shape[0], 0
        # . empty ndarray
        if s_i == 0:
            return ()  # exit
        # . escape items to literals
        # fmt: off
        if not many:
            return list_to_tuple([
                _escape_str(arr_getitem_1d(arr, i), encoding)  # type: ignore
                for i in range(s_i)
            ]) 
        else:
            return [
                _escape_str(arr_getitem_1d(arr, i), encoding)  # type: ignore
                for i in range(s_i)
            ]
        # fmt: on
    # 2-dimensional
    if ndim == 2:
        s_i, s_j = shape[0], shape[1]
        # . empty ndarray
        if s_j == 0:
            return []  # exit
        # . escape items to literals
        # fmt: off
        return [
            list_to_tuple([
                _escape_str(arr_getitem_2d(arr, i, j), encoding)  # type: ignore
                for j in range(s_j)])
            for i in range(s_i)
        ]
        # fmt: on
    # invalid
    raise ValueError("Unsupported <'numpy.ndarray'> dimension: %d." % ndim)


# . Pandas Series - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_series(
    data: Series,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of pandas.Series 'data' to
    tuple of literals `<'tuple[str]'>`.

    ### Example:
    >>> _escape_item_series(pd.Series([1, 2, 3]))
    >>> ("1", "2", "3")  # tuple[str]
    """
    try:
        arr: np.ndarray = data.values
    except Exception as err:
        raise TypeError("Expects <'pandas.Series'>, got %s." % type(data)) from err
    return _escape_item_ndarray(arr, encoding, many)


# . Pandas DataFrame - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_dataframe(data: DataFrame, encoding: cython.pchar) -> list:
    """(cfunc) Escape items of pandas.DataFrame 'data' to
    sequences of literals `<'list[tuple[str]]'>`.

    ### Example:
    >>> _escape_item_dataframe(pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.1, 2.2, 3.3],
            "c": ["a", "b", "c"],
        }))
    >>> [
            ('1', '1.1', "'a'"),
            ('2', '2.2', "'b'"),
            ('3', '3.3', "'c'")
        ]  # list[tuple[str]]
    """
    # Validate shape
    shape: tuple = data.shape
    width: cython.Py_ssize_t = shape[1]
    if width == 0:
        return ()  # exit
    size: cython.Py_ssize_t = shape[0]
    if size == 0:
        return ()  # exit
    # Escape DataFrame
    # fmt: off
    cols = [
        _escape_item_ndarray(col.values, encoding, False) 
        for _, col in data.items()
    ]
    return [
        list_to_tuple([
            cython.cast(object, tuple_getitem(cols[j], i)) 
            for j in range(width)])
        for i in range(size)
    ]
    # fmt: on


# . Escape - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_common(
    data: object,
    encoding: cython.pchar,
    many: cython.bint,
) -> object:
    """(cfunc) Escape items of common 'data' to literal
    or sequence(s) of literals `<'str/tuple/list'>`."""
    # Get data type
    dtype = type(data)

    ##### Common Types #####
    # Basic Types
    # . <'str'>
    if dtype is str:
        return _escape_str(data, encoding)
    # . <'float'>
    if dtype is float:
        return _escape_float(data)
    # . <'int'>
    if dtype is int:
        return str(data)  # _escape_int
    # . <'bool'>
    if dtype is bool:
        return _escape_bool(data)
    # . <None>
    if dtype is typeref.NONE:
        return "NULL"  # _escape_none

    # Date&Time Types
    # . <'datetime.datetime'>
    if dtype is datetime.datetime:
        return _escape_datetime(data)
    # . <'datetime.date'>
    if dtype is datetime.date:
        return _escape_date(data)
    # . <'datetime.time'>
    if dtype is datetime.time:
        return _escape_time(data)
    # . <'datetime.timedelta'>
    if dtype is datetime.timedelta:
        return _escape_timedelta(data)

    # Numeric Types
    # . <'decimal.Decimal'>
    if dtype is typeref.DECIMAL:
        return str(data)  # _escape_decimal

    # Bytes Types
    # . <'bytes'>
    if dtype is bytes:
        return _escape_bytes(data)
    # . <'bytearray'>
    if dtype is bytearray:
        return _escape_bytearray(data)

    # Sequence Types
    # . <'list'>
    if dtype is list:
        return _escape_item_list(data, encoding, many)
    # . <'tuple'>
    if dtype is tuple:
        return _escape_item_tuple(data, encoding, many)
    # . <'set'>
    if dtype is set:
        return _escape_item_set(data, encoding, many)

    # Mapping Types
    # . <'dict'>
    if dtype is dict:
        return _escape_item_dict(data, encoding, many)

    ##### Uncommon Types #####
    return _escape_item_uncommon(data, encoding, many, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_item_uncommon(
    data: object,
    encoding: cython.pchar,
    many: cython.bint,
    dtype: type,
) -> object:
    """(cfunc) Escape items of uncommon 'data' to literal
    or sequence(s) of literals `<'str/tuple/list'>`."""
    ##### Uncommon Types #####
    # Basic Types
    # . <'numpy.float_'>
    if dtype is typeref.FLOAT64 or dtype is typeref.FLOAT32 or dtype is typeref.FLOAT16:
        return _escape_float64(data)
    # . <'numpy.int_'>
    if (
        dtype is typeref.INT64
        or dtype is typeref.INT32
        or dtype is typeref.INT16
        or dtype is typeref.INT8
    ):
        return str(data)  # _escape_int
    # . <'numpy.uint'>
    if (
        dtype is typeref.UINT64
        or dtype is typeref.UINT32
        or dtype is typeref.UINT16
        or dtype is typeref.UINT8
    ):
        return str(data)  # _escape_int
    # . <'numpy.bool_'>
    if dtype is typeref.BOOL_:
        return _escape_bool(data)

    # Date&Time Types
    # . <'pandas.Timestamp'>
    if dtype is typeref.TIMESTAMP:
        return _escape_datetime(data)
    # . <'pandas.Timedelta'>`
    if dtype is typeref.TIMEDELTA:
        return _escape_timedelta(data)
    # . <'numpy.datetime64'>
    if dtype is typeref.DATETIME64:
        return _escape_datetime64(data)
    # . <'numpy.timedelta64'>
    if dtype is typeref.TIMEDELTA64:
        return _escape_timedelta64(data)
    # . <'time.struct_time'>`
    if dtype is typeref.STRUCT_TIME:
        return _escape_struct_time(data)

    # Bytes Types
    # . <'memoryview'>
    if dtype is memoryview:
        return _escape_memoryview(data)
    # . <'numpy.bytes_'>
    if dtype is typeref.BYTES_:
        return _escape_bytes(data)

    # String Types:
    # . <'numpy.str_'>
    if dtype is typeref.STR_:
        return _escape_str(data, encoding)

    # Sequence Types
    # . <'frozenset'>
    if dtype is frozenset:
        return _escape_item_frozenset(data, encoding, many)
    # . <'range'>
    if dtype is range:
        return _escape_item_range(data, many)
    # . <'dict_keys'> & <'dict_values'>
    if dtype is typeref.DICT_VALUES or dtype is typeref.DICT_KEYS:
        return _escape_item_sequence(data, encoding, many)

    # Numpy Types
    # . <'numpy.ndarray'>
    if dtype is np.ndarray:
        return _escape_item_ndarray(data, encoding, many)
    # . <'numpy.record'>
    if dtype is typeref.RECORD:
        return _escape_item_sequence(data, encoding, many)

    # Pandas Types
    # . <'pandas.Series'> & <'pandas.DatetimeIndex'> & <'pandas.TimedeltaIndex'>
    if (
        dtype is typeref.SERIES
        or dtype is typeref.DATETIMEINDEX
        or dtype is typeref.TIMEDELTAINDEX
    ):
        return _escape_item_series(data, encoding, many)
    # . <'pandas.DataFrame'>
    if dtype is typeref.DATAFRAME:
        return _escape_item_dataframe(data, encoding)

    # Custom Types
    if dtype is BIT:
        return _escape_bit(data)
    if dtype is JSON:
        return _escape_json(data, encoding)

    # Cytimes Types
    if typeref.CYTIMES_AVAILABLE:
        # . <'cytimes.pydt'>
        if dtype is typeref.PYDT:
            return _escape_datetime(data.dt)
        # . <'cytimes.pddt'>
        if dtype is typeref.PDDT:
            return _escape_item_series(data.dt, encoding, many)

    ##### Subclass Types #####
    return _escape_item_subclass(data, encoding, many, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_item_subclass(
    data: object,
    encoding: cython.pchar,
    many: cython.bint,
    dtype: type,
) -> object:
    """(cfunc) Escape items of subclass 'data' to literal
    or sequence(s) of literals `<'str/tuple/list'>`."""
    ##### Subclass Types #####
    # Basic Types
    # . subclass of <'str'>
    if isinstance(data, str):
        return _escape_str(data, encoding)
    # . subclass of <'float'>
    if isinstance(data, float):
        return _escape_float(data)
    # . subclass of <'int'>
    if isinstance(data, int):
        return str(data)  # _escape_int
    # . subclass of <'bool'>
    if isinstance(data, bool):
        return _escape_bool(data)

    # Date&Time Types
    # . subclass of <'datetime.datetime'>
    if isinstance(data, datetime.datetime):
        return _escape_datetime(data)
    # . subclass of <'datetime.date'>
    if isinstance(data, datetime.date):
        return _escape_date(data)
    # . subclass of <'datetime.time'>
    if isinstance(data, datetime.time):
        return _escape_time(data)
    # . subclass of <'datetime.timedelta'>
    if isinstance(data, datetime.timedelta):
        return _escape_timedelta(data)

    # Numeric Types
    # . subclass of <'decimal.Decimal'>
    if isinstance(data, typeref.DECIMAL):
        return str(data)  # _escape_decimal

    # Bytes Types
    # . subclass of <'bytes'>
    if isinstance(data, bytes):
        return _escape_bytes(data)
    # . subclass of <'bytearray'>
    if isinstance(data, bytearray):
        return _escape_bytearray(data)

    # Sequence Types
    # . subclass of <'list'>
    if isinstance(data, list):
        return _escape_item_list(data, encoding, many)
    # . subclass of <'tuple'>
    if isinstance(data, tuple):
        return _escape_item_tuple(data, encoding, many)
    # . subclass of <'set'>
    if isinstance(data, set):
        return _escape_item_set(data, encoding, many)
    # . subclass of <'frozenset'>
    if isinstance(data, frozenset):
        return _escape_item_frozenset(data, encoding, many)

    # Mapping Types
    # . subclass of <'dict'>
    if isinstance(data, dict):
        return _escape_item_dict(data, encoding, many)

    # Invalid Data Type
    raise TypeError("Unsupported 'data' type %s." % dtype)


# Escape Function -----------------------------------------------------------------------------
@cython.ccall
def escape(
    data: object,
    encoding: cython.pchar,
    itemize: cython.bint = True,
    many: cython.bint = False,
) -> object:
    """Escape 'data' to formatable object(s) `<'str/tuple/list[str/tuple]'>`.

    ### Arguments
    :param data: `<'object'>` The object to escape, supports:
        - Python native:
          int, float, bool, str, None, datetime, date, time,
          timedelta, struct_time, bytes, bytearray, memoryview,
          Decimal, dict, list, tuple, set, frozenset, range.
        - Library [numpy](https://github.com/numpy/numpy):
          np.int_, np.uint, np.float_, np.bool_, np.bytes_,
          np.str_, np.datetime64, np.timedelta64, np.ndarray.
        - Library [pandas](https://github.com/pandas-dev/pandas):
          pd.Timestamp, pd.Timedelta, pd.DatetimeIndex,
          pd.TimedeltaIndex, pd.Series, pd.DataFrame.
        - Library [cytimes](https://github.com/AresJef/cyTimes):
          pydt, pddt.

    :param encoding `<'bytes'>`: The encoding for the data.

    :param itemize: `<'bool'>` Whether to escape each items of the 'data' individual. Defaults to `True`.
        - When 'itemize=True', the 'data' type determines how to escape.
            * 1. Sequence or Mapping (e.g. `list`, `tuple`, `dict`, etc)
              escapes to `<'tuple[str]'>`.
            * 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to
              `<'tuple[str]'>`.
            * 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes
              to `<'list[tuple[str]]'>`. Each tuple represents one row
              of the 'data' .
            * 4. Single object (such as `int`, `float`, `str`, etc) escapes
              to one literal string `<'str'>`.
        - When 'itemize=False', regardless of the 'data' type, all
          escapes to one single literal string `<'str'>`.

    :param many: `<'bool'>` Wheter to escape 'data' as multi-rows. Defaults to `False`.
        * When 'many=True', the argument 'itemize' is ignored.
        * 1. sequence and mapping (e.g. `list`, `tuple`, `dict`, etc)
          escapes to `<'list[str/tuple[str]]'>`. Each element represents
          one row of the 'data'.
        * 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to
          `<'list[str]'>`. Each element represents one row of the 'data'.
        * 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes
          to `<'list[tuple[str]]'>`. Each tuple represents one row
          of the 'data' .
        * 4. Single object (such as `int`, `float`, `str`, etc) escapes
          to one literal string `<'str'>`.

    ### Exceptions
    :raises `<'EscapeTypeError'>`: If any error occurs during escaping.

    ### Returns
    - If returns a <'str'>, it represents a single literal string.
      The 'sql' should only have one '%s' placeholder.
    - If returns a <'tuple'>, it represents a single row of literal
      strings. The 'sql' should have '%s' placeholders equal to the
      tuple length.
    - If returns a <'list'>, it represents multiple rows of literal
      strings. The 'sql' should have '%s' placeholders equal to the
      item count in each row.
    """
    try:
        if itemize or many:
            return _escape_item_common(data, encoding, many)
        else:
            return _escape_common(data, encoding)
    except Exception as err:
        raise errors.EscapeTypeError(
            "Failed to escape: %s\n%r\nError: %s" % (type(data), data, err)
        ) from err


# Decode ======================================================================================
@cython.cfunc
@cython.inline(True)
def _decode_string(
    value: bytes,
    encoding: cython.pchar,
    is_binary: cython.bint,
) -> object:
    """(cfunc) Decode a SRTING column value to `<'str/bytes'>`.

    Argument 'is_binary' determines whether to return the bytes
    value untouched or decode the value to `<'str'>`.

    ### Example (is_binary=True):
    >>> _decode_string(b'hello world', b'utf-8', True)
    >>> b'hello world'

    ### Example (is_binary=False):
    >>> _decode_string(b'hello world', b'utf-8', False)
    >>> 'hello world'
    """
    return value if is_binary else decode_bytes(value, encoding)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_bit(value: bytes, decode_bit: cython.bint) -> object:
    """(cfunc) Decode a BIT column value to `<'bytes/int'>`.

    Argument 'decode_bit' determines whether to return the bytes
    value untouched or decode the value to `<'int'>`.

    ### Example (decode_bit=True)
    >>> _decode_bit(b"\\x01", True)
    >>> 1

    ### Example (decode_bit=False)
    >>> _decode_bit(b"\\x01", False)
    >>> b'\\x01'
    """
    if not decode_bit:
        return value  # exit
    if bytes_len(value) < 8:
        value = value.rjust(8, b"\x00")
    return unpack_uint64_big_endian(value, 0)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_int(value: bytes) -> object:
    """(cfunc) Decode an INTEGER column value to `<'int'>`.

    ### Example:
    >>> _decode_int(b'-9223372036854775808')
    >>> -9223372036854775808
    """
    c_value: cython.pchar = value
    if c_value[0] == 45:  # negative "-" sign
        return chars_to_long(c_value)  # type: ignore
    else:
        return chars_to_ulong(c_value)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_float(value: bytes) -> object:
    """(cfunc) Decode an FLOAT column value to `<'float'>`.

    ### Example:
    >>> _decode_float(b'-3.141592653589793')
    >>> -3.141592653589793
    """
    return chars_to_double(value)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_decimal(value: bytes, use_decimal: cython.bint) -> object:
    """(cfunc) Decode a DECIMAL column value to `<'float/Decimal'>`.

    Argument 'use_decimal' determines whether to parse
    the value as `<'Decimal'>` object.

    ### Example (use_decimal=True):
    >>> _decode_decimal(b'-3.141592653589793')
    >>> Decimal('-3.141592653589793')

    ### Example (use_decimal=False):
    >>> _decode_decimal(b'-3.141592653589793')
    >>> -3.141592653589793
    """
    if use_decimal:
        return typeref.DECIMAL(decode_bytes_ascii(value))  # type: ignore
    else:
        return _decode_float(value)


@cython.cfunc
@cython.inline(True)
def _decode_datetime(value: bytes) -> object:
    """(cfunc) Decode a DATETIME or TIMESTAMP column value to `<'datetime.datetime'>`.

    ### Example:
    >>> _decode_datetime(b'2007-02-25 23:06:20')
    >>> datetime.datetime(2007, 2, 25, 23, 6, 20)

    ### Example (illegal value):
    >>> _decode_datetime(b'2007-02-31 23:06:20')
    >>> None
    """
    try:
        return datetime.datetime.fromisoformat(decode_bytes_ascii(value))  # type: ignore
    except Exception:
        return None


@cython.cfunc
@cython.inline(True)
def _decode_date(value: bytes) -> object:
    """(cfunc) Decode a DATE column value to `<'datetime.date'>`.

    ### Example:
    >>> _decode_date(b'2007-02-26')
    >>> datetime.date(2007, 2, 26)

    ### Example (illegal value):
    >>> _decode_date(b'2007-02-31')
    >>> None
    """
    try:
        return datetime.date.fromisoformat(decode_bytes_ascii(value))  # type: ignore
    except Exception:
        return None


@cython.cfunc
@cython.inline(True)
def _decode_timedelta(value: bytes) -> object:
    """(cfunc) Decode a TIME column value to `<'datetime.timedelta'>`.

    Note that MySQL always returns TIME columns as (+|-)HH:MM:SS, but
    can accept values as (+|-)DD HH:MM:SS. The latter format will not
    be parsed correctly by this function.

    ### Example:
    >>> _decode_timedelta(b'-25:06:17')
    >>> datetime.timedelta(-2, 83177)

    ### Example (illegal value):
    >>> _decode_timedelta(b'random crap')
    >>> None
    """
    # Empty value
    length: cython.Py_ssize_t = bytes_len(value)
    if length == 0:
        return None  # eixt
    c_value: cython.pchar = value

    # Parse negate and setup position
    ch: cython.Py_UCS4 = c_value[0]
    if ch == "-":
        negate: cython.int = -1
        start: cython.Py_ssize_t = 1
    else:
        negate: cython.int = 1
        start: cython.Py_ssize_t = 0
    idx: cython.Py_ssize_t = 1

    # Parse HH
    hh: cython.int = -1
    while idx < length:
        ch = c_value[idx]
        idx += 1
        if ch == ":":
            try:
                hh = slice_to_int(c_value, start, idx)  # type: ignore
            except Exception:
                return None  # exit: invalid HH
            start = idx
            break
    if hh < 0:
        return None  # exit: invalid HH

    # Parse MM
    mm: cython.int = -1
    while idx < length:
        ch = c_value[idx]
        idx += 1
        if ch == ":":
            try:
                mm = slice_to_int(c_value, start, idx)  # type: ignore
            except Exception:
                return None  # exit: invalid MM
            start = idx
            break
    if not 0 <= mm <= 59:
        return None  # exit: invalid MM

    # Parse SS and Fraction
    ss: cython.int = -1
    us: cython.int = 0
    while idx < length:
        ch = c_value[idx]
        idx += 1
        if ch == ".":
            # . parse SS
            try:
                ss = slice_to_int(c_value, start, idx)  # type: ignore
            except Exception:
                return None  # exit: invalid SS
            # . parse US
            try:
                us = parse_us_fraction(c_value, idx, length)  # type: ignore
            except Exception:
                return None  # exit: invalid us
            break
    # There is not fraction, and SS is the last component
    if ss == -1:
        try:
            ss = slice_to_int(c_value, start, idx)  # type: ignore
        except Exception:
            return None
    if not 0 <= ss <= 59:
        return None  # exit: invalid SS

    # Generate timedelta
    try:
        return datetime.timedelta_new(
            0, (negate * hh * 3600 + negate * mm * 60 + negate * ss), negate * us
        )
    except Exception:
        return None  # exit: value overflow


@cython.cfunc
@cython.inline(True)
def _decode_enum(value: bytes, encoding: cython.pchar) -> object:
    """(cfunc) Decode an ENUM column value to `<'str'>`.

    ### Example:
    >>> _decode_enum(b'enum1', b'utf-8')
    >>> 'enum1'
    """
    return decode_bytes(value, encoding)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_set(value: bytes, encoding: cython.pchar) -> object:
    """(cfunc) Decode a SET column value to `<'set'>`.

    ### Example:
    >>> _decode_set(b'val1,val2,val3', b'utf-8')
    >>> {'val1', 'val2', 'val3'}
    """
    return set(str_split(decode_bytes(value, encoding), ",", -1))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_json(
    value: bytes,
    encoding: cython.pchar,
    decode_json: cython.bint,
) -> object:
    """(cfunc) Decode a JSON column value to python `<'object'>`.

    Argument 'decode_json' determines whether to return the JSON
    string untouched or deserialize JSON to the corresponding
    python object.

    ### Example (decode_json=False):
    >>> _decode_json(b'{"a":1,"b":2,"c":3}', b'utf-8', False)
    >>> '{"a":1,"b":2,"c":3}'

    ### Example (decode_json=True):
    >>> _decode_json(b'{"a":1,"b":2,"c":3}', b'utf-8', True)
    >>> {"a": 1, "b": 2, "c": 3}
    """
    val = decode_bytes(value, encoding)  # type: ignore
    return FN_ORJSON_LOADS(val) if decode_json else val


# Decode Function -----------------------------------------------------------------------------
@cython.ccall
def decode(
    value: bytes,
    field_type: cython.uint,
    encoding: cython.pchar,
    is_binary: cython.bint,
    use_decimal: cython.bint,
    decode_bit: cython.bint,
    decode_json: cython.bint,
) -> object:
    """Decode MySQL column value in Python `<'object'>`.

    :param value `<'bytes'>`: The value of the column item to decode.
    :param field_type `<'int'>`: The field type of the column. Please refer to 'constants.FIELD_TYPE'
    :param encoding `<'bytes'>`: The encoding of the column.
    :param is_binary `<'bool'>`: Whether the column is binary data.
    :param use_decimal `<'bool'>`: Whether to use <'Decimal'> to represent DECIMAL column, `False` use <'float'>.
    :param decode_bit `<'bool'>`: Whether to decode BIT column to integer, `False` keep as original <'bytes'>.
    :param decode_json `<'bool'>`: Whether to deserialize JSON column, `False` keep as original JSON <'str'>.
    """
    # Char / Binary
    if field_type in (
        _FIELD_TYPE.STRING,  # CHAR / BINARY 254
        _FIELD_TYPE.VAR_STRING,  # VARCHAR / VARBINARY 253
        _FIELD_TYPE.VARCHAR,  # VARCHAR / VARBINARY 15
    ):
        return _decode_string(value, encoding, is_binary)
    # Integer
    if field_type in (
        _FIELD_TYPE.TINY,  # TINYINT 1
        _FIELD_TYPE.LONGLONG,  # BIGINT 8
        _FIELD_TYPE.LONG,  # INT 3
        _FIELD_TYPE.INT24,  # MEDIUMINT 9
        _FIELD_TYPE.SHORT,  # SMALLINT 2
    ):
        return _decode_int(value)
    # Decimal / Float
    if field_type in (
        _FIELD_TYPE.NEWDECIMAL,  # DECIMAL 246
        _FIELD_TYPE.DECIMAL,  # DECIMAL 0
    ):
        return _decode_decimal(value, use_decimal)
    if field_type in (
        _FIELD_TYPE.DOUBLE,  # DOUBLE 5
        _FIELD_TYPE.FLOAT,  # FLOAT 4
    ):
        return _decode_float(value)
    # DATETIME / TIMESTAMP
    if field_type in (
        _FIELD_TYPE.DATETIME,  # DATETIME 12
        _FIELD_TYPE.TIMESTAMP,  # TIMESTAMP 7
    ):
        return _decode_datetime(value)
    # DATE
    if field_type in (
        _FIELD_TYPE.DATE,  # DATE 10
        _FIELD_TYPE.NEWDATE,  # DATE 14
    ):
        return _decode_date(value)
    # TIME
    if field_type == _FIELD_TYPE.TIME:  # TIME 11
        return _decode_timedelta(value)
    # TEXT / BLOB
    if field_type in (
        _FIELD_TYPE.TINY_BLOB,  # TINYTEXT / TINYBLOB 249
        _FIELD_TYPE.BLOB,  # TEXT / BLOB 252
        _FIELD_TYPE.MEDIUM_BLOB,  # MEDIUMTEXT / MEDIUMBLOB 250
        _FIELD_TYPE.LONG_BLOB,  # LONGTEXT / LONGBLOB 251
    ):
        return _decode_string(value, encoding, is_binary)
    # BIT
    if field_type == _FIELD_TYPE.BIT:  # BIT 16
        return _decode_bit(value, decode_bit)
    # ENUM
    if field_type == _FIELD_TYPE.ENUM:  # ENUM 247
        return _decode_enum(value, encoding)
    # SET
    if field_type == _FIELD_TYPE.SET:  # SET 248
        return _decode_set(value, encoding)
    # JSON
    if field_type == _FIELD_TYPE.JSON:  # JSON 245
        return _decode_json(value, encoding, decode_json)
    # GEOMETRY
    if field_type == _FIELD_TYPE.GEOMETRY:  # GEOMETRY 255
        return _decode_string(value, encoding, is_binary)
    # YEAR
    if field_type == _FIELD_TYPE.YEAR:  # YEAR 13
        return _decode_int(value)
    # Unknown
    raise errors.DecodeTypeError("Unknown FIELD_TYPE: %d." % field_type)
