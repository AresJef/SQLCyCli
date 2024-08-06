# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.libc.math import isnormal, isfinite  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.list import PyList_AsTuple as list_to_tuple  # type: ignore
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

__all__ = ["escape", "decode"]

# Constants -----------------------------------------------------------------------------------
# . cytimes [library]
try:
    from cytimes import pddt, pydt

    CYTIMES_AVAILABLE: cython.bint = True
except ImportError:
    CYTIMES_AVAILABLE: cython.bint = False
# . translate table
# Used to translate Python string to literal string.
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
    """(cfunc) Serialize object using
    'orjson [https://github.com/ijl/orjson]' into JSON string `<'str'>`."""
    return decode_bytes_utf8(FN_ORJSON_DUMPS(obj))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _orjson_dumps_numpy(obj: object) -> str:
    """(cfunc) Serialize numpy.ndarray using
    'orjson [https://github.com/ijl/orjson]' into JSON string `<'str'>`."""
    return decode_bytes_utf8(FN_ORJSON_DUMPS(obj, option=FN_ORJSON_OPT_NUMPY))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _mysqlclient_literal(obj: object) -> str:
    """(cfunc) Escape `<'str'>` or `<'bytes'>` using
    'mysqlclient [https://github.com/PyMySQL/mysqlclient]' into literal string `<'str'>`.
    """
    return decode_bytes_utf8(FN_MYSQLCLI_STR2LIT(obj))  # type: ignore


# Escape ======================================================================================
# Escape --------------------------------------------------------------------------------------
# . Basic types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_bool(data: object) -> str:
    """(cfunc) Escape boolean 'data' into literal string `<'str'>.

    >>> _escape_bool(True)
    >>> "1"
    """
    return "1" if data else "0"


@cython.cfunc
@cython.inline(True)
def _escape_int(data: object) -> str:
    """(cfunc) Escape integer 'data' into literal string `<'str'>.

    >>> _escape_int(123)
    >>> "123"
    """
    return str(data)


@cython.cfunc
@cython.inline(True)
def _escape_float(data: object) -> str:
    """(cfunc) Escape float 'data' into literal string `<'str'>`.

    >>> _escape_float(123.456)
    >>> "123.456"
    """
    # For normal native Python float numbers, orjson performs
    # faster than Python built-in `str()` function.
    if isnormal(data):
        return _orjson_dumps(data)
    # For numpy float numbers such as np.float64, use
    # Python built-in `str()` function for escaping.
    return _escape_float64(data)


@cython.cfunc
@cython.inline(True)
def _escape_float64(data: object) -> str:
    """(cfunc) Escape numpy.float_ 'data' into literal string `<'str'>`.

    :raises `<'TypeError'>`: If float value is invalid.

    >>> _escape_float64(np.float64(123.456))
    >>> "123.456"
    """
    # For numpy.float64, Python built-in `str()`
    # function performs faster than orjson.
    if isfinite(data):
        return str(data)
    raise TypeError("float value '%s' is invalid." % data)


@cython.cfunc
@cython.inline(True)
def _escape_str(data: object) -> str:
    """(cfunc) Escape string 'data' into literal string `<'str'>`.

    >>> _escape_str("Hello, World!")
    >>> "'Hello, World!'"
    """
    return _mysqlclient_literal(data)


@cython.cfunc
@cython.inline(True)
def _escape_none(_) -> str:
    """(cfunc) Escape None 'data' into literal string `<'str'>`.

    >>> _escape_none(None)
    >>> "NULL"
    """
    return "NULL"


# . Date&Time types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_datetime(data: object) -> str:
    """(cfunc) Escape datetime 'data' into literal string `<'str'>`.

    >>> _escape_datetime(datetime.datetime(2021, 1, 1, 12, 0, 0, 100))
    >>> "'2021-01-01 12:00:00.00100'"
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
    """(cfunc) Escape numpy.datetime64 'data' into literal string `<'str'>`.

    >>> _escape_datetime64(np.datetime64('2021-01-01T12:00:00.001'))
    >>> "'2021-01-01 12:00:00.00100'"
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
    """(cfunc) Escape struct_time 'data' into literal string `<'str'>`.

    >>> _escape_struct_time(time.localtime())
    >>> "'2021-01-01 12:00:00'"
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
    """(cfunc) Escape date 'data' into literal string `<'str'>`.

    >>> _escape_date(datetime.date(2021, 1, 1))
    >>> "'2021-01-01'"
    """
    return "'%04d-%02d-%02d'" % (
        datetime.PyDateTime_GET_YEAR(data),
        datetime.PyDateTime_GET_MONTH(data),
        datetime.PyDateTime_GET_DAY(data),
    )


@cython.cfunc
@cython.inline(True)
def _escape_time(data: object) -> str:
    """(cfunc) Escape time 'data' into literal string `<'str'>`.

    >>> _escape_time(datetime.time(12, 0, 0, 100))
    >>> "'12:00:00.00100'"
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
    """(cfunc) Escape timedelta 'data' into literal string `<'str'>`.

    >>> _escape_timedelta(datetime.timedelta(hours=12, minutes=0, seconds=0, microseconds=100))
    >>> "'12:00:00.000100'"
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
    """(cfunc) Escape numpy.timedelta64 'data' into literal string `<'str'>`.

    >>> _escape_timedelta64(np.timedelta64('12:00:00.000100'))
    >>> "'12:00:00.000100'"
    """
    us: cython.longlong = td64_to_microseconds(data)  # type: ignore
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
    """(cfunc) Escape bytes 'data' into literal
    ('_binary' prefixed) string `<'str'>`.

    >>> _escape_bytes(b"Hello, World!")
    >>> "_binary'Hello, World!'"
    """
    res: str = decode_bytes_ascii(data)  # type: ignore
    return "_binary'" + res.translate(STR_ESCAPE_TABLE) + "'"


@cython.cfunc
@cython.inline(True)
def _escape_bytearray(data: object) -> str:
    """(cfunc) Escape bytearray 'data' into literal
    ('_binary' prefixed) string `<'str'>`.

    >>> _escape_bytearray(bytearray(b"Hello, World!"))
    >>> "_binary'Hello, World!'"
    """
    res: str = decode_bytearray_ascii(data)  # type: ignore
    return "_binary'" + res.translate(STR_ESCAPE_TABLE) + "'"


@cython.cfunc
@cython.inline(True)
def _escape_memoryview(data: memoryview) -> str:
    """(cfunc) Escape memoryview 'data' into literal
    ('_binary' prefixed) string `<'str'>`.

    >>> _escape_memoryview(memoryview(b"Hello, World!"))
    >>> "_binary'Hello, World!'"
    """
    return _escape_bytes(data.tobytes())


# . Numeric types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_decimal(data: object) -> str:
    """(cfunc) Escape decimal 'data' into literal string `<'str'>`.

    >>> _escape_decimal(decimal.Decimal("123.456"))
    >>> "123.456"
    """
    return str(data)


# . Mapping types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_dict(data: dict) -> str:
    """(cfunc) Escape dict 'data' into literal string `<'str'>`.

    >>> _escape_dict({"key1": "value1", "key2": "value2"})
    >>> "('value1','value2')"
    """
    res: str = ",".join([_escape_common(i) for i in data.values()])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


# . Sequence types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_list(data: list) -> str:
    """(cfunc) Escape list 'data' into literal string `<'str'>`.

    >>> _escape_list(["value1", "value2"])
    >>> "('value1','value2')"
    """
    res: str = ",".join([_escape_common(i) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_tuple(data: tuple) -> str:
    """(cfunc) Escape tuple 'data' into literal string `<'str'>`.

    >>> _escape_tuple(("value1", "value2"))
    >>> "('value1','value2')"
    """
    res: str = ",".join([_escape_common(i) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_set(data: set) -> str:
    """(cfunc) Escape set 'data' into literal string `<'str'>`.

    >>> _escape_set({"value1", "value2"})
    >>> "('value1','value2')"
    """
    res: str = ",".join([_escape_common(i) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_frozenset(data: frozenset) -> str:
    """(cfunc) Escape frozenset 'data' into literal string `<'str'>`.

    >>> _escape_frozenset(frozenset({"value1", "value2"}))
    >>> "('value1','value2')"
    """
    res: str = ",".join([_escape_common(i) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_sequence(data: Iterable) -> str:
    """(cfunc) Escape sequence 'data' into literal string `<'str'>`.

    >>> _escape_sequence({"key1": "value1", "key2": "value2"}.values())
    >>> "('value1','value2')"
    """
    res: str = ",".join([_escape_common(i) for i in data])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


# . Numpy ndarray - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_ndarray(data: np.ndarray) -> str:
    """(cfunc) Escape numpy.ndarray 'data' into literal string `<'str'>`.

    >>> _escape_ndarray(np.array([1, 2, 3]))
    >>> "(1,2,3)"
    """
    # Validate ndarray dimensions & size
    if data.ndim != 1:
        raise ValueError("only supports 1-dimensional <'numpy.ndarray'>.")
    size: cython.Py_ssize_t = data.shape[0]
    if size == 0:
        return "()"  # exit
    # Escape ndarray
    dtype: cython.char = data.descr.kind
    # . ndarray[object]
    if dtype == NDARRAY_DTYPE_OBJECT:
        return _escape_ndarray_object(data, size)
    # . ndarray[float]
    if dtype == NDARRAY_DTYPE_FLOAT:
        return _escape_ndarray_float(data, size)
    # . ndarray[int]
    if dtype == NDARRAY_DTYPE_INT:
        return _escape_ndarray_int(data, size)
    # . ndarray[uint]
    if dtype == NDARRAY_DTYPE_UINT:
        return _escape_ndarray_int(data, size)
    # . ndarray[bool]
    if dtype == NDARRAY_DTYPE_BOOL:
        return _escape_ndarray_bool(data, size)
    # . ndarray[datetime64]
    if dtype == NDARRAY_DTYPE_DT64:
        return _escape_ndarray_dt64(data, size)
    # . ndarray[timedelta64]
    if dtype == NDARRAY_DTYPE_TD64:
        return _escape_ndarray_td64(data, size)
    # . ndarray[bytes]
    if dtype == NDARRAY_DTYPE_BYTES:
        return _escape_ndarray_bytes(data, size)
    # . ndarray[str]
    if dtype == NDARRAY_DTYPE_UNICODE:
        return _escape_ndarray_unicode(data, size)
    # # . invalid dtype
    raise TypeError("unsupported <'numpy.ndarray'> dtype [%s]." % data.dtype)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_object(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' into literal string `<'str'>`.

    This function is specifically for ndarray with dtype of: "O" (object).

    >>> _escape_ndarray_object(np.array([1, 1.23, "abc"], dtype="O"), 3)
    >>> "(1,1.23,'abc')"
    """
    l = [_escape_common(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore
    return "(" + ",".join(l) + ")"


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_float(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' into literal string `<'str'>`.

    This function is specifically for ndarray with dtype of: "f" (float).

    :raises `<'TypeError'>`: If float value is invalid.

    >>> _escape_ndarray_float(np.array([1.23, 4.56, 7.89], dtype=np.float64), 3)
    >>> "(1.23,4.56,7.89)"
    """
    if not is_arr_double_finite(arr, size):  # type: ignore
        raise TypeError("float value such as 'nan' & 'inf' is invalid.")
    # Alternative approach:
    # str = _orjson_dumps(np.PyArray_ToList(arr))
    res: str = _orjson_dumps_numpy(arr)
    return replace_bracket(res)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_int(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' into literal string `<'str'>`.

    This function is specifically for ndarray with dtype of:
    "i" (int) and "u" (uint).

    >>> _escape_ndarray_int(np.array([-1, -2, -3], dtype=np.int64), 3)
    >>> "(-1,-2,-3)"

    >>> _escape_ndarray_int(np.array([1, 2, 3], dtype=np.uint64), 3)
    >>> "(1,2,3)"
    """
    # Alternative approach:
    # res: str = _orjson_dumps(np.PyArray_ToList(arr))
    res: str = _orjson_dumps_numpy(arr)
    return replace_bracket(res)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_bool(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' into literal string `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "b" (bool).

    >>> _escape_ndarray_bool(np.array([True, False, True], dtype=np.bool_), 3)
    >>> "(1,0,1)"
    """
    # Alternative approach:
    # l = ["1" if arr_getitem_1d_bint(arr, i) else "0" for i in range(size)]
    # return "(" + ",".join(l) + ")"
    res: str = _orjson_dumps_numpy(np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64))
    return replace_bracket(res)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_dt64(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' into literal string `<'str'>`.

    This function is specifically for ndarray with
    dtype of: "M" (datetime64).

    >>> _escape_ndarray_dt64(np.array([1, 2, 3], dtype="datetime64[s]"), 3)
    >>> "('1970-01-01 00:00:01','1970-01-01 00:00:02','1970-01-01 00:00:03')"
    """
    # Notes: This approach is faster than escaping each element individually.
    # 'orjson' returns '["1970-01-01T00:00:00",...,"2000-01-01T00:00:01"]',
    # so character ['"', "T", "[", "]"] will be replaced to comply with literal
    # datetime format.
    res: str = _orjson_dumps_numpy(arr)
    return res.translate(DT64_JSON_TABLE)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_ndarray_td64(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' into literal string `<'str'>`.

    This function is specifically for ndarray with
    dtype of: "m" (timedelta64).

    >>> _escape_ndarray_td64(np.array([1, 2, 3], dtype="timedelta64[s]"), 3)
    >>> "('00:00:01','00:00:02','00:00:03')"
    """
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    res: list = []
    for i in range(size):
        us: cython.longlong = arr_getitem_1d_long(arr, i)  # type: ignore
        us = nptime_to_microseconds(us, unit)  # type: ignore
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
                res.append("'-%02d:%02d:%02d'" % (hours, minutes, seconds))
            else:
                res.append("'%02d:%02d:%02d'" % (hours, minutes, seconds))
        else:
            if negate:
                res.append("'-%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us))
            else:
                res.append("'%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us))
    return "(" + ",".join(res) + ")"


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_bytes(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' into literal string `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "S" (bytes string).

    >>> _escape_ndarray_bytes(np.array([1, 2, 3], dtype="S"), 3)
    >>> "(_binary'1',_binary'2',_binary'3')"
    """
    l = [_escape_bytes(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore
    return "(" + ",".join(l) + ")"


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_unicode(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray 'arr' into literal string `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "S" (bytes string).

    >>> _escape_ndarray_unicode(np.array([1, 2, 3], dtype="U"), 3)
    >>> "('1','2','3')"
    """
    l = [_escape_str(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore
    return "(" + ",".join(l) + ")"


# . Pandas Series - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_series(data: Series) -> str:
    """(cfunc) Escape pandas.Series 'data' into literal string `<'str'>`.

    >>> _escape_series(pd.Series([1, 2, 3]))
    >>> "(1,2,3)"
    """
    try:
        arr: np.ndarray = data.values
    except Exception as err:
        raise TypeError("Expects <'pandas.Series'>, got %s." % type(data)) from err
    return _escape_ndarray(arr)


# . Pandas DataFrame - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_dataframe(data: DataFrame) -> str:
    """(cfunc) Escape pandas.DataFrame 'data' into literal string `<'str'>`.

    >>> _escape_dataframe(pdDataFrame(
            {"a": [1, 2, 3], "b": [1.1, 2.2, 3.3], "c": ["a", "b", "c"]})
        )
    >>> "(1,1.1,'a'),(2,2.2,'b'),(3,3.3,'c')"
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
    rows: list = []
    cols: list[list] = [
        _escape_item_dataframe_column(col, size) for _, col in data.items()
    ]
    for i in range(size):
        row: list = []
        for j in range(width):
            row.append(cols[j][i])
        rows.append("(" + ",".join(row) + ")")
    return ",".join(rows)


# . Escape - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_common(data: object) -> str:
    """(cfunc) Escape common 'data' into literal string `<'str'>`."""
    # Get data type
    dtype = type(data)

    ##### Common Types #####
    # Basic Types
    # . <'str'>
    if dtype is str:
        return _escape_str(data)
    # . <'float'>
    if dtype is float:
        return _escape_float(data)
    # . <'int'>
    if dtype is int:
        return _escape_int(data)
    # . <'bool'>
    if dtype is bool:
        return _escape_bool(data)
    # . <None>
    if dtype is typeref.NONE:
        return _escape_none(data)

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
        return _escape_decimal(data)

    # Bytes Types
    # . <'bytes'>
    if dtype is bytes:
        return _escape_bytes(data)

    # Mapping Types
    # . <'dict'>
    if dtype is dict:
        return _escape_dict(data)

    # Sequence Types
    # . <'list'>
    if dtype is list:
        return _escape_list(data)
    # . <'tuple'>
    if dtype is tuple:
        return _escape_tuple(data)

    ##### Uncommon Types #####
    return _escape_uncommon(data, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_uncommon(data: object, dtype: type) -> str:
    """(cfunc) Escape uncommon 'data' into literal string `<'str'>`."""
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
        return _escape_int(data)
    # . <'numpy.uint'>
    if (
        dtype is typeref.UINT64
        or dtype is typeref.UINT32
        or dtype is typeref.UINT16
        or dtype is typeref.UINT8
    ):
        return _escape_int(data)
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
    # . <'cytimes.pydt'>
    if CYTIMES_AVAILABLE and dtype is pydt:
        return _escape_datetime(data.dt)

    # Bytes Types
    # . <'bytearray'>
    if dtype is bytearray:
        return _escape_bytearray(data)
    # . <'memoryview'>
    if dtype is memoryview:
        return _escape_memoryview(data)
    # . <'numpy.bytes_'>
    if dtype is typeref.BYTES_:
        return _escape_bytes(data)

    # String Types:
    # . <'numpy.str_'>
    if dtype is typeref.STR_:
        return _escape_str(data)

    # Sequence Types
    # . <'set'>
    if dtype is set:
        return _escape_set(data)
    # . <'frozenset'>
    if dtype is frozenset:
        return _escape_frozenset(data)
    # . <'range'> & <'dict_keys'> & <'dict_values'>
    if dtype is range or dtype is typeref.DICT_KEYS or dtype is typeref.DICT_VALUES:
        return _escape_sequence(data)

    # Numpy Types
    # . <'numpy.ndarray'>
    if dtype is np.ndarray:
        return _escape_ndarray(data)
    # . <'numpy.record'>
    if dtype is typeref.RECORD:
        return _escape_sequence(data)

    # Pandas Types
    # . <'pandas.Series'> & <'pandas.DatetimeIndex'> & <'pandas.TimedeltaIndex'>
    if (
        dtype is typeref.SERIES
        or dtype is typeref.DATETIMEINDEX
        or dtype is typeref.TIMEDELTAINDEX
    ):
        return _escape_series(data)
    # . <'cytimes.pddt'>
    if CYTIMES_AVAILABLE and dtype is pddt:
        return _escape_series(data.dt)
    # . <'pandas.DataFrame'>
    if dtype is typeref.DATAFRAME:
        return _escape_dataframe(data)

    ##### Subclass Types #####
    return _escape_subclass(data, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_subclass(data: object, dtype: type) -> str:
    """(cfunc) Escape subclass 'data' into literal string `<'str'>`."""
    ##### Subclass Types #####
    # Basic Types
    # . subclass of <'str'>
    if isinstance(data, str):
        return _escape_str(data)
    # . subclass of <'float'>
    if isinstance(data, float):
        return _escape_float(data)
    # . subclass of <'int'>
    if isinstance(data, int):
        return _escape_int(data)
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
        return _escape_decimal(data)

    # Bytes Types
    # . subclass of <'bytes'>
    if isinstance(data, bytes):
        return _escape_bytes(data)
    # . subclass of <'bytearray'>
    if isinstance(data, bytearray):
        return _escape_bytearray(data)

    # Mapping Types
    # . subclass of <'dict'>
    if isinstance(data, dict):
        return _escape_dict(data)

    # Sequence Types
    # . subclass of <'list'>
    if isinstance(data, list):
        return _escape_list(data)
    # . subclass of <'tuple'>
    if isinstance(data, tuple):
        return _escape_tuple(data)
    # . subclass of <'set'>
    if isinstance(data, set):
        return _escape_set(data)
    # . subclass of <'frozenset'>
    if isinstance(data, frozenset):
        return _escape_frozenset(data)

    # Invalid Data Type
    raise TypeError("unsupported data type %s" % dtype)


# Escape Item ---------------------------------------------------------------------------------
# . Mapping types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_dict(data: dict) -> tuple:
    """(cfunc) Escape items of dict 'data' into
    tuple of literal strings `<'tuple[str]'>`.

    >>> _escape_item_dict({"key1": "value1", "key2": "value2"})
    >>> ("'value1'", "'value2'")
    """
    return list_to_tuple([_escape_common(i) for i in data.values()])


# . Sequence types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_list(data: list) -> tuple:
    """(cfunc) Escape items of list 'data' into
    tuple of literal strings `<'tuple[str]'>`.

    >>> _escape_item_list(["value1", "value2"])
    >>> ("'value1'", "'value2'")
    """
    return list_to_tuple([_escape_common(i) for i in data])


@cython.cfunc
@cython.inline(True)
def _escape_item_tuple(data: tuple) -> tuple:
    """(cfunc) Escape items of tuple 'data' into
    tuple of literal strings `<'tuple[str]'>`.

    >>> _escape_item_tuple(("value1", "value2"))
    >>> ("'value1'", "'value2'")
    """
    return list_to_tuple([_escape_common(i) for i in data])


@cython.cfunc
@cython.inline(True)
def _escape_item_set(data: set) -> tuple:
    """(cfunc) Escape items of set 'data' into
    tuple of literal strings `<'tuple[str]'>`.

    >>> _escape_item_set({"value1", "value2"})
    >>> ("'value1'", "'value2'")
    """
    return list_to_tuple([_escape_common(i) for i in data])


@cython.cfunc
@cython.inline(True)
def _escape_item_frozenset(data: frozenset) -> tuple:
    """(cfunc) Escape items of frozenset 'data' into
    tuple of literal strings `<'tuple[str]'>`.

    >>> _escape_item_frozenset(frozenset({"value1", "value2"}))
    >>> ("'value1'", "'value2'")
    """
    return list_to_tuple([_escape_common(i) for i in data])


@cython.cfunc
@cython.inline(True)
def _escape_item_sequence(data: Iterable) -> tuple:
    """(cfunc) Escape items of sequence 'data' into
    tuple of literal strings `<'tuple[str]'>`.

    >>> _escape_item_sequence({"key1": "value1", "key2": "value2"}.values())
    >>> ("'value1'", "'value2'")
    """
    return list_to_tuple([_escape_common(i) for i in data])


# . Numpy ndarray - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray(data: np.ndarray) -> tuple:
    """(cfunc) Escape items of numpy.ndarray 'data' into
    tuple of literal strings `<'tuple[str]'>`.

    >>> _escape_item_ndarray(np.array([1, 2, 3]))
    >>> ("1", "2", "3")
    """
    # Validate ndarray dimensions & size
    if data.ndim != 1:
        raise ValueError("only supports 1-dimensional <'numpy.ndarray'>.")
    size: cython.Py_ssize_t = data.shape[0]
    if size == 0:
        return ()  # exit
    # Escape ndarray
    dtype: cython.char = data.descr.kind
    # . ndarray[object]
    if dtype == NDARRAY_DTYPE_OBJECT:
        return list_to_tuple(_escape_item_ndarray_object(data, size))
    # . ndarray[float]
    if dtype == NDARRAY_DTYPE_FLOAT:
        return list_to_tuple(_escape_item_ndarray_float(data, size))
    # . ndarray[int]
    if dtype == NDARRAY_DTYPE_INT:
        return list_to_tuple(_escape_item_ndarray_int(data, size))
    # . ndarray[uint]
    if dtype == NDARRAY_DTYPE_UINT:
        return list_to_tuple(_escape_item_ndarray_int(data, size))
    # . ndarray[bool]
    if dtype == NDARRAY_DTYPE_BOOL:
        return list_to_tuple(_escape_item_ndarray_bool(data, size))
    # . ndarray[datetime64]
    if dtype == NDARRAY_DTYPE_DT64:
        return list_to_tuple(_escape_item_ndarray_dt64(data, size))
    # . ndarray[timedelta64]
    if dtype == NDARRAY_DTYPE_TD64:
        return list_to_tuple(_escape_item_ndarray_td64(data, size))
    # . ndarray[bytes]
    if dtype == NDARRAY_DTYPE_BYTES:
        return list_to_tuple(_escape_item_ndarray_bytes(data, size))
    # . ndarray[str]
    if dtype == NDARRAY_DTYPE_UNICODE:
        return list_to_tuple(_escape_item_ndarray_unicode(data, size))
    # . invalid dtype
    raise TypeError("unsupported <'numpy.ndarray'> dtype [%s]." % data.dtype)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_object(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of numpy.ndarray 'arr' into
    list of literal strings `<'list[str]'>`.

    This function is specifically for ndarray with dtype of: "O" (object).

    >>> _escape_item_ndarray_object(np.array([1, 1.23, "abc"], dtype="O"), 3)
    >>> ["1", "1.23", "'abc'"]

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    return [_escape_common(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_float(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of numpy.ndarray arr into
    list of literal strings `<'list[str]'>`.

    This function is specifically for ndarray with dtype of:
    "f" (float), "i" (int) and "u" (uint).

    :raises `<'TypeError'>`: If float value is invalid.

    >>> _escape_item_ndarray_float(np.array([1.23, 4.56, 7.89], dtype=np.float64), 3)
    >>> ["1.23", "4.56", "7.89"]

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    # Check if any item is not finite.
    if not is_arr_double_finite(arr, size):  # type: ignore
        raise TypeError("float value of 'nan' & 'inf' is invalid.")
    # Alternative approach:
    # res: str = _orjson_dumps(np.PyArray_ToList(arr))
    res: str = _orjson_dumps_numpy(arr)
    return str_split(str_substr(res, 1, str_len(res) - 1), ",", -1)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_int(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of numpy.ndarray 'arr' into
    list of literal strings `<'list[str]'>`.

    This function is specifically for ndarray with dtype of:
    "i" (int) and "u" (uint).

    >>> _escape_item_ndarray_int(np.array([-1, -2, -3], dtype=np.int64), 3)
    >>> ["-1", "-2", "-3"]

    >>> _escape_item_ndarray_int(np.array([1, 2, 3], dtype=np.uint64), 3)
    >>> ["1", "2", "3"]

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    # Alternative approach:
    # res: str = _orjson_dumps(np.PyArray_ToList(arr))
    res: str = _orjson_dumps_numpy(arr)
    return str_split(str_substr(res, 1, str_len(res) - 1), ",", -1)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_bool(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of numpy.ndarray 'arr' into
    list of literal strings `<'list[str]'>`.

    This function is specifically for ndarray with dtype of: "b" (bool).

    >>> _escape_item_ndarray_bool(np.array([True, False, True], dtype=np.bool_), 3)
    >>> ["1", "0", "1"]

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    return ["1" if arr_getitem_1d_bint(arr, i) else "0" for i in range(size)]  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_dt64(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of numpy.ndarray 'arr' into
    list of literal strings `<'list[str]'>`.

    This function is specifically for ndarray with dtype of: "M" (datetime64).

    >>> _escape_item_ndarray_dt64(np.array([1, 2, 3], dtype="datetime64[s]"), 3)
    >>> ("'1970-01-01 00:00:01'", "'1970-01-01 00:00:02'", "'1970-01-01 00:00:03'")

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    # Notes: This approach is faster than encoding each element individually.
    # 'orjson' returns '["1970-01-01T00:00:00",...,"2000-01-01T00:00:01"]',
    # so character ['"', "T"] will be replaced to comply with literal datetime
    # format, and then split by "," into sequence of literal strings.
    res: str = _orjson_dumps_numpy(arr)
    res = str_substr(res, 1, str_len(res) - 1)
    return str_split(res.translate(DT64_JSON_TABLE), ",", -1)


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_td64(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of numpy.ndarray 'arr' into
    list of literal strings `<'list[str]'>`.

    This function is specifically for ndarray with dtype of: "m" (timedelta64).

    >>> _escape_item_ndarray_td64(np.array([1, 2, 3], dtype="timedelta64[s]"), 3)
    >>> ("'00:00:01'", "'00:00:02'", "'00:00:03'")

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(arr[0])
    arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)
    res: list = []
    for i in range(size):
        us: cython.longlong = arr_getitem_1d_long(arr, i)  # type: ignore
        us = nptime_to_microseconds(us, unit)  # type: ignore
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
                res.append("'-%02d:%02d:%02d'" % (hours, minutes, seconds))
            else:
                res.append("'%02d:%02d:%02d'" % (hours, minutes, seconds))
        else:
            if negate:
                res.append("'-%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us))
            else:
                res.append("'%02d:%02d:%02d.%06d'" % (hours, minutes, seconds, us))
    return res


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_bytes(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of numpy.ndarray 'arr' into
    list of literal strings `<'list[str]'>`.

    This function is specifically for ndarray with dtype of: "S" (bytes string).

    >>> _escape_item_ndarray_bytes(np.array([1, 2, 3], dtype="S"), 3)
    >>> ("_binary'1'", "_binary'2'", "_binary'3'")

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    return [_escape_bytes(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_item_ndarray_unicode(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of numpy.ndarray 'arr' into
    list of literal strings `<'list[str]'>`.

    This function is specifically for ndarray with dtype of: "S" (bytes string).

    >>> _escape_item_ndarray_unicode(np.array([1, 2, 3], dtype="U"), 3)
    >>> ("'1'", "'2'", "'3'")

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    return [_escape_str(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore


# . Pandas Series - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_series(data: Series) -> tuple:
    """(cfunc) Escape items of pandas.Series 'data' into
    tuple of literal strings `<'tuple[str]'>`.

    >>> _escape_item_series(pd.Series([1, 2, 3]))
    >>> ("1", "2", "3")
    """
    try:
        arr: np.ndarray = data.values
    except Exception as err:
        raise TypeError("Expects <'pandas.Series'>, got %s." % type(data)) from err
    return _escape_item_ndarray(arr)


# . Pandas DataFrame - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_dataframe(data: DataFrame) -> list:
    """(cfunc) Escape items of pandas.DataFrame 'data' into
    list of tuple of literal strings `<'list[tuple[str]]'>`.

    >>> _escape_item_dataframe(pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3], "c": ["a", "b", "c"]}))
    >>> (('1', '1.1', "'a'"), ('2', '2.2', "'b'"), ('3', '3.3', "'c'"))

    ### Notice
    This function returns `LIST` instead of `TUPLE` to represent
    there are `multi-rows` in the escaped result.
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
    rows: list = []
    cols: list[list] = [
        _escape_item_dataframe_column(col, size) for _, col in data.items()
    ]
    for i in range(size):
        row: list = []
        for j in range(width):
            row.append(cols[j][i])
        rows.append(list_to_tuple(row))
    return rows


@cython.cfunc
@cython.inline(True)
def _escape_item_dataframe_column(data: Series, size: cython.Py_ssize_t) -> list:
    """(cfunc) Escape items of pandas.DataFrame column 'data' into
    list of literal strings `<'list[str]'>`.

    >>> _escape_item_dataframe_column(pd.Series([1, 2, 3]), 3)
    >>> ["1", "2", "3"]

    ### Notice
    This function returns `LIST` instead of `TUPLE`
    for performance optimization.
    """
    # Get values
    try:
        arr: np.ndarray = data.values
    except Exception as err:
        raise TypeError("Expects <'pandas.Series'>, got %s." % type(data)) from err
    # Escape ndarray
    dtype: cython.char = arr.descr.kind
    # . ndarray[object]
    if dtype == NDARRAY_DTYPE_OBJECT:
        return _escape_item_ndarray_object(arr, size)
    # . ndarray[float]
    if dtype == NDARRAY_DTYPE_FLOAT:
        return _escape_item_ndarray_float(arr, size)
    # . ndarray[int]
    if dtype == NDARRAY_DTYPE_INT:
        return _escape_item_ndarray_int(arr, size)
    # . ndarray[uint]
    if dtype == NDARRAY_DTYPE_UINT:
        return _escape_item_ndarray_int(arr, size)
    # . ndarray[bool]
    if dtype == NDARRAY_DTYPE_BOOL:
        return _escape_item_ndarray_bool(arr, size)
    # . ndarray[datetime64]
    if dtype == NDARRAY_DTYPE_DT64:
        return _escape_item_ndarray_dt64(arr, size)
    # . ndarray[timedelta64]
    if dtype == NDARRAY_DTYPE_TD64:
        return _escape_item_ndarray_td64(arr, size)
    # . ndarray[bytes]
    if dtype == NDARRAY_DTYPE_BYTES:
        return _escape_item_ndarray_bytes(arr, size)
    # . ndarray[str]
    if dtype == NDARRAY_DTYPE_UNICODE:
        return _escape_item_ndarray_unicode(arr, size)
    # . invalid dtype
    raise TypeError("unsupported <'Series'> dtype [%s]." % data.dtype)


# . Escape - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_common(data: object) -> object:
    """(cfunc) Escape items of common 'data' into
    formatable object `<'str/tuple'>`."""
    # Get data type
    dtype = type(data)

    ##### Common Types #####
    # Basic Types
    # . <'str'>
    if dtype is str:
        return _escape_str(data)
    # . <'float'>
    if dtype is float:
        return _escape_float(data)
    # . <'int'>
    if dtype is int:
        return _escape_int(data)
    # . <'bool'>
    if dtype is bool:
        return _escape_bool(data)
    # . <None>
    if dtype is typeref.NONE:
        return _escape_none(data)

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
        return _escape_decimal(data)

    # Bytes Types
    # . <'bytes'>
    if dtype is bytes:
        return _escape_bytes(data)

    # Mapping Types
    # . <'dict'>
    if dtype is dict:
        return _escape_item_dict(data)

    # Sequence Types
    # . <'list'>
    if dtype is list:
        return _escape_item_list(data)
    # . <'tuple'>
    if dtype is tuple:
        return _escape_item_tuple(data)

    ##### Uncommon Types #####
    return _escape_item_uncommon(data, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_item_uncommon(data: object, dtype: type) -> object:
    """(cfunc) Escape items of uncommon 'data' into
    formatable object `<'str/tuple'>`."""
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
        return _escape_int(data)
    # . <'numpy.uint'>
    if (
        dtype is typeref.UINT64
        or dtype is typeref.UINT32
        or dtype is typeref.UINT16
        or dtype is typeref.UINT8
    ):
        return _escape_int(data)
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
    # . <'cytimes.pydt'>
    if CYTIMES_AVAILABLE and dtype is pydt:
        return _escape_datetime(data.dt)

    # Bytes Types
    # . <'bytearray'>
    if dtype is bytearray:
        return _escape_bytearray(data)
    # . <'memoryview'>
    if dtype is memoryview:
        return _escape_memoryview(data)
    # . <'numpy.bytes_'>
    if dtype is typeref.BYTES_:
        return _escape_bytes(data)

    # String Types:
    # . <'numpy.str_'>
    if dtype is typeref.STR_:
        return _escape_str(data)

    # Sequence Types
    # . <'set'>
    if dtype is set:
        return _escape_item_set(data)
    # . <'frozenset'>
    if dtype is frozenset:
        return _escape_item_frozenset(data)
    # . <'range'> & <'dict_keys'> & <'dict_values'>
    if dtype is range or dtype is typeref.DICT_KEYS or dtype is typeref.DICT_VALUES:
        return _escape_item_sequence(data)

    # Numpy Types
    # . <'numpy.ndarray'>
    if dtype is np.ndarray:
        return _escape_item_ndarray(data)
    # . <'numpy.record'>
    if dtype is typeref.RECORD:
        return _escape_item_sequence(data)

    # Pandas Types
    # . <'pandas.Series'> & <'pandas.DatetimeIndex'> & <'pandas.TimedeltaIndex'>
    if (
        dtype is typeref.SERIES
        or dtype is typeref.DATETIMEINDEX
        or dtype is typeref.TIMEDELTAINDEX
    ):
        return _escape_item_series(data)
    # . <'cytimes.pddt'>
    if CYTIMES_AVAILABLE and dtype is pddt:
        return _escape_item_series(data.dt)
    # . <'pandas.DataFrame'>
    if dtype is typeref.DATAFRAME:
        return _escape_item_dataframe(data)

    ##### Subclass Types #####
    return _escape_item_subclass(data, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_item_subclass(data: object, dtype: type) -> object:
    """(cfunc) Escape items of subclass 'data' into
    formatable object `<'str/tuple'>`."""
    ##### Subclass Types #####
    # Basic Types
    # . subclass of <'str'>
    if isinstance(data, str):
        return _escape_str(data)
    # . subclass of <'float'>
    if isinstance(data, float):
        return _escape_float(data)
    # . subclass of <'int'>
    if isinstance(data, int):
        return _escape_int(data)
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
        return _escape_decimal(data)

    # Bytes Types
    # . subclass of <'bytes'>
    if isinstance(data, bytes):
        return _escape_bytes(data)
    # . subclass of <'bytearray'>
    if isinstance(data, bytearray):
        return _escape_bytearray(data)

    # Mapping Types
    # . subclass of <'dict'>
    if isinstance(data, dict):
        return _escape_item_dict(data)

    # Sequence Types
    # . subclass of <'list'>
    if isinstance(data, list):
        return _escape_item_list(data)
    # . subclass of <'tuple'>
    if isinstance(data, tuple):
        return _escape_item_tuple(data)
    # . subclass of <'set'>
    if isinstance(data, set):
        return _escape_item_set(data)
    # . subclass of <'frozenset'>
    if isinstance(data, frozenset):
        return _escape_item_frozenset(data)

    # Numpy Types
    # . subclass of <'numpy.ndarray'>
    if isinstance(data, np.ndarray):
        return _escape_item_ndarray(data)

    # Invalid Data Type
    raise TypeError("unsupported data type %s" % dtype)


# Escape Function -----------------------------------------------------------------------------
@cython.ccall
def escape(
    data: object,
    many: cython.bint = False,
    itemize: cython.bint = True,
) -> object:
    """Escape 'data' into formatable object(s) `<'str/tuple/list[str/tuple]'>`.

    ### Arguments
    :param data: `<'object'>` The object to escape, supports:
        - Python native: int, float, bool, str, None, datetime, date,
          time, timedelta, struct_time, bytes, bytearray, memoryview,
          Decimal, dict, list, tuple, set, frozenset.
        - Library numpy: np.int_, np.uint, np.float_, np.bool_, np.bytes_,
          np.str_, np.datetime64, np.timedelta64, np.ndarray.
        - Library pandas: pd.Timestamp, pd.Timedelta, pd.DatetimeIndex,
          pd.TimedeltaIndex, pd.Series, pd.DataFrame.

    :param many: `<'bool'>` Wheter to escape 'data' into multi-rows. Defaults to `False`.
        - When 'many=True', the argument 'itemize' is ignored.
            * 1. 'list' and 'tuple' will be escaped into list of str or tuple[str],
              depends on type of the sequence items. Each item represents one row
              of the 'data' `<'list[str/tuple[str]]'>`.
            * 2. 'DataFrame' will be escaped into list of tuple[str]. Each tuple
              represents one row of the 'data' `<'list[tuple[str]]'>`.
            * 3. All other sequences or mappings 'data' will be escaped into tuple
              of literal strings `<'tuple[str]'>`.
            * 4. Single 'data' will be escaped into one literal string `<'str'>`.
        - When 'many=False', the argument 'itemize' determines how to escape the 'data'.

    :param itemize: `<'bool'>` Whether to escape each items of the 'data' individual. Defaults to `True`.
        - When 'itemize=True', the 'data' type determines how to escape.
            * 1. 'DataFrame' will be escaped into list of tuple[str]. Each tuple
              represents one row of the 'data' `<'list[tuple[str]]'>`.
            * 2. All sequences or mappings 'data' will be escaped into tuple of
              literal strings `<'tuple[str]'>`. This includes 'list', 'tuple',
              'set', 'frozenset', 'dict', 'np.ndarray', 'pd.Series', etc.
            * 3. Single 'data' will be escaped into one literal string `<'str'>`.
        - When 'itemize=False', regardless of the 'data' type, it will be escaped into
          one single literal string `<'str'>`.

    ### Exceptions
    :raises `<'EscapeTypeError'>`: If any error occurs during escaping.

    ### Returns:
    - If returns a <'str'>, it represents a single literal string.
      The 'sql' should only have one '%s' placeholder.
    - If returns a <'tuple'>, it represents a single row of literal
      strings. The 'sql' should have '%s' placeholders equal to the
      tuple length.
    - If returns a <'list'>, it represents multiple rows of literal
      strings. The 'sql' should have '%s' placeholders equal to the
      item count in each row.

    ### Example (list or tuple)
    >>> escape([(1, 2), (3, 4)], many=True)
    >>> [("1", "2"), ("3", "4")]  # list[tuple[str]]
        # many=True, [optional] itemize=True
    >>> escape([(1, 2), (3, 4)], many=False, itemize=True)
    >>> ("(1,2)", "(3,4)")        # tuple[str]
        # many=False, itemize=True
    >>> escape([(1, 2), (3, 4)], many=False, itemize=False)
    >>> "(1,2),(3,4)"             # str
        # many=False, itemize=False

    ### Example (set [sequence])
    >>> escape({(1, 2), (3, 4)}, many=True)
    >>> ("(1,2)", "(3,4)")        # tuple[str]
        # many=True, [optional] itemize=True
    >>> escape({(1, 2), (3, 4)}, many=False, itemize=True)
    >>> ("(1,2)", "(3,4)")        # tuple[str]
        # many=False, itemize=True
    >>> escape({(1, 2), (3, 4)}, many=False, itemize=False)
    >>> "(1,2),(3,4)"             # str
        # many=False, itemize=False

    ### Example (DataFrame)
    >>> escape(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), many=True)
    >>> [('1', '3'), ('2', '4')]  # list[tuple[str]]
        # many=True, [optional] itemize=True
    >>> escape(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), many=False, itemize=True)
    >>> [('1', '3'), ('2', '4')]  # list[tuple[str]]
        # many=False, itemize=True
    >>> escape(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), many=False, itemize=False)
    >>> "(1,3),(2,4)"             # str
        # many=False, itemize=False

    ### Example (single item)
    >>> escape(1, many=True)
    >>> "1"                       # str
        # many=True, [optional] itemize=True
    >>> escape(1, many=False, itemize=True)
    >>> "1"                       # str
        # many=False, itemize=True
    >>> escape(1, many=False, itemize=False)
    >>> "1"                       # str
        # many=False, itemize=False
    """
    try:
        if many:
            dtype = type(data)
            if dtype is list or dtype is tuple:
                return [_escape_item_common(i) for i in data]
            else:
                return _escape_item_common(data)
        elif itemize:
            return _escape_item_common(data)
        else:
            return _escape_common(data)
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
    """(cfunc) Decode a SRTING related column value into `<'bytes'>` or `<'str'>`.

    Argument 'is_binary' determines whether to return the bytes
    value untouched or decode the value into `<'str'>`.
    """
    return value if is_binary else decode_bytes(value, encoding)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_int(value: bytes) -> object:
    """(cfunc) Decode an INTEGER column value into `<'int'>`.

    >>> _decode_int(b'-9223372036854775808')
        -9223372036854775808
    """
    c_value: cython.pchar = value
    if c_value[0] == 45:  # negative "-" sign
        return chars_to_long(c_value)  # type: ignore
    else:
        return chars_to_ulong(c_value)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_float(value: bytes) -> object:
    """(cfunc) Decode an FLOAT column value into `<'float'>`.

    >>> _decode_float(b'-3.141592653589793')
        -3.141592653589793
    """
    return chars_to_double(value)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_decimal(value: bytes, use_decimal: cython.bint) -> object:
    """(cfunc) Decode a DECIMAL column value into `<'Decimal'>` or `<'float'>`.

    Argument 'use_decimal' determines whether to Parse
    the value into a `<'Decimal'>` object.
    """
    if use_decimal:
        return typeref.DECIMAL(decode_bytes_ascii(value))  # type: ignore
    else:
        return _decode_float(value)


@cython.cfunc
@cython.inline(True)
def _decode_datetime(value: bytes) -> object:
    """(cfunc) Decode a DATETIME or TIMESTAMP column value into `<'datetime.datetime'>`.

    >>> _decode_datetime(b'2007-02-25 23:06:20')
        datetime.datetime(2007, 2, 25, 23, 6, 20)

    Illegal values are returned as None:
    >>> _decode_datetime(b'2007-02-31T23:06:20')
        None
    """
    try:
        return datetime.datetime.fromisoformat(decode_bytes_ascii(value))  # type: ignore
    except Exception:
        return None


@cython.cfunc
@cython.inline(True)
def _decode_date(value: bytes) -> object:
    """(cfunc) Decode a DATE column value into `<'datetime.date'>`.

    >>> _decode_date(b'2007-02-26')
        datetime.date(2007, 2, 26)

    Illegal values are returned as None:
    >>> _decode_date(b'2007-02-31')
        None
    """
    try:
        return datetime.date.fromisoformat(decode_bytes_ascii(value))  # type: ignore
    except Exception:
        return None


@cython.cfunc
@cython.inline(True)
def _decode_timedelta(value: bytes) -> object:
    """(cfunc) Decode a TIME column value into `<'datetime.timedelta'>`.

    >>> _decode_timedelta(b'-25:06:17')
        datetime.timedelta(-2, 83177)

    Illegal values are returned as None:
    >>> _decode_timedelta(b'random crap')
        None

    Note that MySQL always returns TIME columns as (+|-)HH:MM:SS, but
    can accept values as (+|-)DD HH:MM:SS. The latter format will not
    be parsed correctly by this function.
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
    """(cfunc) Decode an ENUM column value into `<'str'>`."""
    return decode_bytes(value, encoding)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_set(value: bytes, encoding: cython.pchar) -> object:
    """(cfunc) Decode a SET column value into `<'set'>`."""
    return set(str_split(decode_bytes(value, encoding), ",", -1))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _decode_json(
    value: bytes,
    encoding: cython.pchar,
    decode_json: cython.bint,
) -> object:
    """(cfunc) Decode a JSON column value into python `<'object'>`."""
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
    decode_json: cython.bint,
) -> object:
    """Decode MySQL column value into Python `<'object'>`.

    :param value `<'bytes'>`: The value of the MySQL item to decode.
    :param field_type `<'int'>`: The field type of the MySQL item.
    :param encoding `<'bytes'>: The encoding of the MySQL item.
    :param is_binary `<'bool'>`: Whether the MySQL item is binary data.
    :param use_decimal `<'bool'>`: Whether to use `<'Decimal'>` for DECIMAL column, `False` decode as <'float'>`.
    :param decode_json `<'bool'>`: Whether to deserialize JSON column item, `False` decode as json `<'str'>`.
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
    # BIT / GEOMETRY
    if field_type in (
        _FIELD_TYPE.BIT,  # BIT 16
        _FIELD_TYPE.GEOMETRY,  # GEOMETRY 255
    ):
        return _decode_string(value, encoding, is_binary)
    # ENUM
    if field_type == _FIELD_TYPE.ENUM:  # ENUM 247
        return _decode_enum(value, encoding)
    # SET
    if field_type == _FIELD_TYPE.SET:  # SET 248
        return _decode_set(value, encoding)
    # JSON
    if field_type == _FIELD_TYPE.JSON:  # JSON 245
        return _decode_json(value, encoding, decode_json)
    # YEAR
    if field_type == _FIELD_TYPE.YEAR:  # YEAR 13
        return _decode_int(value)
    # Unknown
    raise errors.DecodeFieldTypeError("Unknown MySQL data FIELD_TYPE: %s." % field_type)
