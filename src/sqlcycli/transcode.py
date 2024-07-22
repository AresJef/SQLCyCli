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

__all__ = ["escape_item", "encode_item", "decode_item"]

# Constants -----------------------------------------------------------------------------------
# . translate table
# Used to translate Python string to MySQL literal string.
STR_ESCAPE_TABLE: list = [chr(x) for x in range(128)]
STR_ESCAPE_TABLE[0] = "\\0"
STR_ESCAPE_TABLE[ord("\\")] = "\\\\"
STR_ESCAPE_TABLE[ord("\n")] = "\\n"
STR_ESCAPE_TABLE[ord("\r")] = "\\r"
STR_ESCAPE_TABLE[ord("\032")] = "\\Z"
STR_ESCAPE_TABLE[ord('"')] = '\\"'
STR_ESCAPE_TABLE[ord("'")] = "\\'"
# Used to translate 'orjson' serialized datetime64 to MySQL datetime format.
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
def _orjson_dumps(value: object) -> str:
    """(cfunc) Serialize object using
    'orjson [https://github.com/ijl/orjson]' into JSON `<'str'>`."""
    return decode_bytes_utf8(FN_ORJSON_DUMPS(value))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _orjson_dumps_numpy(value: object) -> str:
    """(cfunc) Serialize numpy.ndarray using
    'orjson [https://github.com/ijl/orjson]' into JSON `<'str'>`."""
    return decode_bytes_utf8(FN_ORJSON_DUMPS(value, option=FN_ORJSON_OPT_NUMPY))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _mysqlclient_literal(value: object) -> str:
    """(cfunc) Escape `<'str'>` or `<'bytes'>` using
    'mysqlclient [https://github.com/PyMySQL/mysqlclient]' into MySQL `<'str'>`."""
    return decode_bytes_utf8(FN_MYSQLCLI_STR2LIT(value))  # type: ignore


# Escaper -------------------------------------------------------------------------------------
# . Basic types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_bool(value: object) -> str:
    """(cfunc) Escape boolean value into MySQL string `<'str'>."""
    return "1" if value else "0"


@cython.cfunc
@cython.inline(True)
def _escape_int(value: object) -> str:
    """(cfunc) Escape integer value into MySQL string `<'str'>."""
    return str(value)


@cython.cfunc
@cython.inline(True)
def _escape_float(value: object) -> str:
    """(cfunc) Escape float value into MySQL string `<'str'>`."""
    # For normal native Python float numbers, orjson performs
    # faster than Python built-in `str()` function.
    if isnormal(value):
        return _orjson_dumps(value)
    # For numpy float numbers such as np.float64, use
    # Python built-in `str()` function for escaping.
    return _escape_float64(value)


@cython.cfunc
@cython.inline(True)
def _escape_float64(value: object) -> str:
    """(cfunc) Escape numpy.float_ value into MySQL string `<'str'>`."""
    # For numpy.float64, Python built-in `str()`
    # function performs faster than orjson.
    if isfinite(value):
        return str(value)
    raise TypeError("float value '%s' can not be used with MySQL." % value)


@cython.cfunc
@cython.inline(True)
def _escape_str(value: object) -> str:
    """(cfunc) Escape string value into MySQL string `<'str'>`."""
    return _mysqlclient_literal(value)


@cython.cfunc
@cython.inline(True)
def _escape_none(_) -> str:
    """(cfunc) Escape None value into MySQL string `<'str'>`."""
    return "NULL"


# . Date&Time types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_datetime(value: object) -> str:
    """(cfunc) Escape datetime value into MySQL string `<'str'>`."""
    microsecond: cython.int = datetime.PyDateTime_DATE_GET_MICROSECOND(value)
    if microsecond == 0:
        return "'%04d-%02d-%02d %02d:%02d:%02d'" % (
            datetime.PyDateTime_GET_YEAR(value),
            datetime.PyDateTime_GET_MONTH(value),
            datetime.PyDateTime_GET_DAY(value),
            datetime.PyDateTime_DATE_GET_HOUR(value),
            datetime.PyDateTime_DATE_GET_MINUTE(value),
            datetime.PyDateTime_DATE_GET_SECOND(value),
        )
    else:
        return "'%04d-%02d-%02d %02d:%02d:%02d.%06d'" % (
            datetime.PyDateTime_GET_YEAR(value),
            datetime.PyDateTime_GET_MONTH(value),
            datetime.PyDateTime_GET_DAY(value),
            datetime.PyDateTime_DATE_GET_HOUR(value),
            datetime.PyDateTime_DATE_GET_MINUTE(value),
            datetime.PyDateTime_DATE_GET_SECOND(value),
            microsecond,
        )


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_datetime64(value: object) -> str:
    """(cfunc) Escape numpy.datetime64 value into MySQL string `<'str'>`."""
    # Add back epoch seconds
    microseconds: cython.longlong = dt64_to_microseconds(value) + EPOCH_US  # type: ignore
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
def _escape_struct_time(value: object) -> str:
    """(cfunc) Escape struct_time value into MySQL string `<'str'>`."""
    # fmt: off
    return _escape_datetime(datetime.datetime_new(
        value.tm_year, value.tm_mon, value.tm_mday,
        value.tm_hour, value.tm_min, value.tm_sec,
        0, None, 0) )
    # fmt: on


@cython.cfunc
@cython.inline(True)
def _escape_date(value: object) -> str:
    """(cfunc) Escape date value into MySQL string `<'str'>`."""
    return "'%04d-%02d-%02d'" % (
        datetime.PyDateTime_GET_YEAR(value),
        datetime.PyDateTime_GET_MONTH(value),
        datetime.PyDateTime_GET_DAY(value),
    )


@cython.cfunc
@cython.inline(True)
def _escape_time(value: object) -> str:
    """(cfunc) Escape time value into MySQL string `<'str'>`."""
    microsecond: cython.int = datetime.PyDateTime_TIME_GET_MICROSECOND(value)
    if microsecond == 0:
        return "'%02d:%02d:%02d'" % (
            datetime.PyDateTime_TIME_GET_HOUR(value),
            datetime.PyDateTime_TIME_GET_MINUTE(value),
            datetime.PyDateTime_TIME_GET_SECOND(value),
        )
    else:
        return "'%02d:%02d:%02d.%06d'" % (
            datetime.PyDateTime_TIME_GET_HOUR(value),
            datetime.PyDateTime_TIME_GET_MINUTE(value),
            datetime.PyDateTime_TIME_GET_SECOND(value),
            microsecond,
        )


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_timedelta(value: object) -> str:
    """(cfunc) Escape timedelta value into MySQL string `<'str'>`."""
    # Get total seconds and microseconds
    seconds: cython.longlong = (
        datetime.PyDateTime_DELTA_GET_SECONDS(value)
        + datetime.PyDateTime_DELTA_GET_DAYS(value) * 86_400
    )
    microseconds: cython.int = datetime.PyDateTime_DELTA_GET_MICROSECONDS(value)

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
def _escape_timedelta64(value: object) -> str:
    """(cfunc) Escape numpy.timedelta64 value into MySQL string `<'str'>`."""
    us: cython.longlong = td64_to_microseconds(value)  # type: ignore
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
def _escape_bytes(value: object) -> str:
    """(cfunc) Escape bytes value into MySQL bytes string `<'str'>`."""
    res: str = decode_bytes_ascii(value)  # type: ignore
    return "_binary'" + res.translate(STR_ESCAPE_TABLE) + "'"


@cython.cfunc
@cython.inline(True)
def _escape_bytearray(value: object) -> str:
    """(cfunc) Escape bytearray value into MySQL bytes string `<'str'>`."""
    res: str = decode_bytearray_ascii(value)  # type: ignore
    return "_binary'" + res.translate(STR_ESCAPE_TABLE) + "'"


@cython.cfunc
@cython.inline(True)
def _escape_memoryview(value: memoryview) -> str:
    """(cfunc) Escape memoryview value into MySQL bytes string `<'str'>`."""
    return _escape_bytes(value.tobytes())


# . Numeric types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_decimal(value: object) -> str:
    """(cfunc) Escape decimal value into MySQL string `<'str'>`."""
    return str(value)


# . Mapping types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_dict(value: dict) -> str:
    """(cfunc) Escape dict value into MySQL string `<'str'>`."""
    res: str = ",".join([_escape_item_common(i) for i in value.values()])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


# . Sequence types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_list(value: list) -> str:
    """(cfunc) Escape list value into MySQL string `<'str'>`."""
    res: str = ",".join([_escape_item_common(i) for i in value])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_tuple(value: tuple) -> str:
    """(cfunc) Escape tuple value into MySQL string `<'str'>`."""
    res: str = ",".join([_escape_item_common(i) for i in value])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_set(value: set) -> str:
    """(cfunc) Escape set value into MySQL string `<'str'>`."""
    res: str = ",".join([_escape_item_common(i) for i in value])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_frozenset(value: frozenset) -> str:
    """(cfunc) Escape frozenset value into MySQL string `<'str'>`."""
    res: str = ",".join([_escape_item_common(i) for i in value])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


@cython.cfunc
@cython.inline(True)
def _escape_sequence(value: Iterable) -> str:
    """(cfunc) Escape sequence value into MySQL string `<'str'>`."""
    res: str = ",".join([_escape_item_common(i) for i in value])
    return res if read_char(res, 0) == "(" else "(" + res + ")"


# . Numpy ndarray - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_ndarray(value: np.ndarray) -> str:
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`."""
    # Validate ndarray dimensions & size
    if value.ndim != 1:
        raise ValueError("only supports 1-dimensional <'numpy.ndarray'>.")
    size: cython.Py_ssize_t = value.shape[0]
    if size == 0:
        return "()"  # exit
    # Escape ndarray
    dtype: cython.char = value.descr.kind
    # . ndarray[object]
    if dtype == NDARRAY_DTYPE_OBJECT:
        return _escape_ndarray_object(value, size)
    # . ndarray[float]
    if dtype == NDARRAY_DTYPE_FLOAT:
        return _escape_ndarray_float(value, size)
    # . ndarray[int]
    if dtype == NDARRAY_DTYPE_INT:
        return _escape_ndarray_int(value, size)
    # . ndarray[uint]
    if dtype == NDARRAY_DTYPE_UINT:
        return _escape_ndarray_int(value, size)
    # . ndarray[bool]
    if dtype == NDARRAY_DTYPE_BOOL:
        return _escape_ndarray_bool(value, size)
    # . ndarray[datetime64]
    if dtype == NDARRAY_DTYPE_DT64:
        return _escape_ndarray_dt64(value, size)
    # . ndarray[timedelta64]
    if dtype == NDARRAY_DTYPE_TD64:
        return _escape_ndarray_td64(value, size)
    # . ndarray[bytes]
    if dtype == NDARRAY_DTYPE_BYTES:
        return _escape_ndarray_bytes(value, size)
    # . ndarray[str]
    if dtype == NDARRAY_DTYPE_UNICODE:
        return _escape_ndarray_unicode(value, size)
    # # . invalid dtype
    raise TypeError("unsupported <'numpy.ndarray'> dtype [%s]." % value.dtype)


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_object(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`.

    This function is specifically for ndarray with dtype of: "O" (object).
    """
    l = [_escape_item_common(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore
    return "(" + ",".join(l) + ")"


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_float(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`.

    This function is specifically for ndarray with dtype of: "f" (float).
    """
    if not is_arr_double_finite(arr, size):  # type: ignore
        raise TypeError("float value of 'nan' & 'inf' can not be used with MySQL.")
    # For size < 30, serialize as list is faster.
    if size < 30:
        res: str = _orjson_dumps(np.PyArray_ToList(arr))
    # For larger size, serialize as ndarray is faster.
    else:
        res: str = _orjson_dumps_numpy(arr)
    return replace_bracket(res)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_int(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`.

    This function is specifically for ndarray with dtype of:
    "i" (int) and "u" (uint).
    """
    # For size < 60, serialize as list is faster.
    if size < 60:
        res: str = _orjson_dumps(np.PyArray_ToList(arr))
    # For larger size, serialize as ndarray is faster.
    else:
        res: str = _orjson_dumps_numpy(arr)
    return replace_bracket(res)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_bool(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "b" (bool).
    """
    # For size < 60, serialize as list and Python built-in 'join()' is faster.
    if size < 60:
        l = ["1" if arr_getitem_1d_bint(arr, i) else "0" for i in range(size)]  # type: ignore
        return "(" + ",".join(l) + ")"
    # For larger size, serialize as ndarray is faster.
    else:
        res: str = _orjson_dumps_numpy(np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64))
    return replace_bracket(res)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_dt64(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`.

    This function is specifically for ndarray with
    dtype of: "M" (datetime64).
    """
    # Notes: This approach is faster than escaping each element individually.
    # 'orjson' returns '["1970-01-01T00:00:00",...,"2000-01-01T00:00:01"]',
    # so character ['"', "T", "[", "]"] will be replaced to comply with MySQL
    # datetime format.
    res: str = _orjson_dumps_numpy(arr)
    return res.translate(DT64_JSON_TABLE)


@cython.cfunc
@cython.inline(True)
@cython.cdivision(True)
def _escape_ndarray_td64(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`.

    This function is specifically for ndarray with
    dtype of: "m" (timedelta64).
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
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "S" (bytes string).
    """
    l = [_escape_bytes(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore
    return "(" + ",".join(l) + ")"


@cython.cfunc
@cython.inline(True)
def _escape_ndarray_unicode(arr: np.ndarray, size: cython.Py_ssize_t) -> str:
    """(cfunc) Escape numpy.ndarray value into MySQL string `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "S" (bytes string).
    """
    l = [_escape_str(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore
    return "(" + ",".join(l) + ")"


# . Pandas Series - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_series(value: Series) -> str:
    """(cfunc) Escape pandas.Series value into MySQL string `<'str'>`."""
    try:
        arr: np.ndarray = value.values
    except Exception as err:
        raise TypeError("Expects <'pandas.Series'>, got %s." % type(value)) from err
    return _escape_ndarray(arr)


# . Pandas DataFrame - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_dataframe(value: DataFrame) -> str:
    """(cfunc) Escape pandas.DataFrame value into MySQL strings `<'str'>`."""
    # Validate shape
    shape: tuple = value.shape
    width: cython.Py_ssize_t = shape[1]
    if width == 0:
        return "()"  # exit
    size: cython.Py_ssize_t = shape[0]
    if size == 0:
        return "()"  # exit

    # Escape DataFrame
    rows: list = []
    cols: list[list] = [_encode_dataframe_column(col, size) for _, col in value.items()]
    for i in range(size):
        row: list = []
        for j in range(width):
            row.append(cols[j][i])
        rows.append("(" + ",".join(row) + ")")
    return ",".join(rows)


# . Escape - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _escape_item_common(value: object) -> str:
    """(cfunc) Escape common item value into MySQL string `<'str'>`."""
    # Get data type
    dtype = type(value)

    ##### Common Types #####
    # Basic Types
    # . <'str'>
    if dtype is str:
        return _escape_str(value)
    # . <'float'>
    if dtype is float:
        return _escape_float(value)
    # . <'int'>
    if dtype is int:
        return _escape_int(value)
    # . <'bool'>
    if dtype is bool:
        return _escape_bool(value)
    # . <None>
    if dtype is typeref.NONE:
        return _escape_none(value)

    # Date&Time Types
    # . <'datetime.datetime'>
    if dtype is datetime.datetime:
        return _escape_datetime(value)
    # . <'datetime.date'>
    if dtype is datetime.date:
        return _escape_date(value)
    # . <'datetime.time'>
    if dtype is datetime.time:
        return _escape_time(value)
    # . <'datetime.timedelta'>
    if dtype is datetime.timedelta:
        return _escape_timedelta(value)

    # Numeric Types
    # . <'decimal.Decimal'>
    if dtype is typeref.DECIMAL:
        return _escape_decimal(value)

    # Bytes Types
    # . <'bytes'>
    if dtype is bytes:
        return _escape_bytes(value)

    # Mapping Types
    # . <'dict'>
    if dtype is dict:
        return _escape_dict(value)

    # Sequence Types
    # . <'list'>
    if dtype is list:
        return _escape_list(value)
    # . <'tuple'>
    if dtype is tuple:
        return _escape_tuple(value)

    ##### Uncommon Types #####
    return _escape_item_uncommon(value, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_item_uncommon(value: object, dtype: type) -> str:
    """(cfunc) Escape uncommon item value into MySQL string `<'str'>`."""
    ##### Uncommon Types #####
    # Basic Types
    # . <'numpy.float_'>
    if dtype is typeref.FLOAT64 or dtype is typeref.FLOAT32 or dtype is typeref.FLOAT16:
        return _escape_float64(value)
    # . <'numpy.int_'>
    if (
        dtype is typeref.INT64
        or dtype is typeref.INT32
        or dtype is typeref.INT16
        or dtype is typeref.INT8
    ):
        return _escape_int(value)
    # . <'numpy.uint'>
    if (
        dtype is typeref.UINT64
        or dtype is typeref.UINT32
        or dtype is typeref.UINT16
        or dtype is typeref.UINT8
    ):
        return _escape_int(value)
    # . <'numpy.bool_'>
    if dtype is typeref.BOOL_:
        return _escape_bool(value)

    # Date&Time Types
    # . <'pandas.Timestamp'>
    if dtype is typeref.TIMESTAMP:
        return _escape_datetime(value)
    # . <'pandas.Timedelta'>`
    if dtype is typeref.TIMEDELTA:
        return _escape_timedelta(value)
    # . <'numpy.datetime64'>
    if dtype is typeref.DATETIME64:
        return _escape_datetime64(value)
    # . <'numpy.timedelta64'>
    if dtype is typeref.TIMEDELTA64:
        return _escape_timedelta64(value)
    # . <'time.struct_time'>`
    if dtype is typeref.STRUCT_TIME:
        return _escape_struct_time(value)
    # . <'cytimes.pydt'>
    if dtype is typeref.PYDT:
        return _escape_datetime(value.dt)

    # Bytes Types
    # . <'bytearray'>
    if dtype is bytearray:
        return _escape_bytearray(value)
    # . <'memoryview'>
    if dtype is memoryview:
        return _escape_memoryview(value)
    # . <'numpy.bytes_'>
    if dtype is typeref.BYTES_:
        return _escape_bytes(value)

    # String Types:
    # . <'numpy.str_'>
    if dtype is typeref.STR_:
        return _escape_str(value)

    # Sequence Types
    # . <'set'>
    if dtype is set:
        return _escape_set(value)
    # . <'frozenset'>
    if dtype is frozenset:
        return _escape_frozenset(value)
    # . <'range'> & <'dict_keys'> & <'dict_values'>
    if dtype is range or dtype is typeref.DICT_KEYS or dtype is typeref.DICT_VALUES:
        return _escape_sequence(value)

    # Numpy Types
    # . <'numpy.ndarray'>
    if dtype is np.ndarray:
        return _escape_ndarray(value)
    # . <'numpy.record'>
    if dtype is typeref.RECORD:
        return _escape_sequence(value)

    # Pandas Types
    # . <'pandas.Series'> & <'pandas.DatetimeIndex'> & <'pandas.TimedeltaIndex'>
    if (
        dtype is typeref.SERIES
        or dtype is typeref.DATETIMEINDEX
        or dtype is typeref.TIMEDELTAINDEX
    ):
        return _escape_series(value)
    # . <'cytimes.pddt'>
    if dtype is typeref.PDDT:
        return _escape_series(value.dt)
    # . <'pandas.DataFrame'>
    if dtype is typeref.DATAFRAME:
        return _escape_dataframe(value)

    ##### Subclass Types #####
    return _escape_item_subclass(value, dtype)


@cython.cfunc
@cython.inline(True)
def _escape_item_subclass(value: object, dtype: type) -> str:
    """(cfunc) Escape subclass item value into MySQL string `<'str'>`."""
    ##### Subclass Types #####
    # Basic Types
    # . subclass of <'str'>
    if isinstance(value, str):
        return _escape_str(value)
    # . subclass of <'float'>
    if isinstance(value, float):
        return _escape_float(value)
    # . subclass of <'int'>
    if isinstance(value, int):
        return _escape_int(value)
    # . subclass of <'bool'>
    if isinstance(value, bool):
        return _escape_bool(value)

    # Date&Time Types
    # . subclass of <'datetime.datetime'>
    if isinstance(value, datetime.datetime):
        return _escape_datetime(value)
    # . subclass of <'datetime.date'>
    if isinstance(value, datetime.date):
        return _escape_date(value)
    # . subclass of <'datetime.time'>
    if isinstance(value, datetime.time):
        return _escape_time(value)
    # . subclass of <'datetime.timedelta'>
    if isinstance(value, datetime.timedelta):
        return _escape_timedelta(value)

    # Numeric Types
    # . subclass of <'decimal.Decimal'>
    if isinstance(value, typeref.DECIMAL):
        return _escape_decimal(value)

    # Bytes Types
    # . subclass of <'bytes'>
    if isinstance(value, bytes):
        return _escape_bytes(value)
    # . subclass of <'bytearray'>
    if isinstance(value, bytearray):
        return _escape_bytearray(value)

    # Mapping Types
    # . subclass of <'dict'>
    if isinstance(value, dict):
        return _escape_dict(value)

    # Sequence Types
    # . subclass of <'list'>
    if isinstance(value, list):
        return _escape_list(value)
    # . subclass of <'tuple'>
    if isinstance(value, tuple):
        return _escape_tuple(value)
    # . subclass of <'set'>
    if isinstance(value, set):
        return _escape_set(value)
    # . subclass of <'frozenset'>
    if isinstance(value, frozenset):
        return _escape_frozenset(value)

    # Invalid Data Type
    raise TypeError("unsupported data type %s" % dtype)


@cython.ccall
def escape_item(value: object) -> str:
    """Escape item value into MySQL string `<'str'>`."""
    try:
        return _escape_item_common(value)
    except Exception as err:
        raise errors.EscapeTypeError(
            "Failed to escape %s: %s" % (type(value), err)
        ) from err


# Encoder -------------------------------------------------------------------------------------
# . Mapping types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _encode_dict(value: dict) -> tuple:
    """(cfunc) Encode dict value into dictionary of MySQL strings `<'tuple'>`."""
    return list_to_tuple([_encode_item_common(v) for v in value.values()])


# . Sequence types - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _encode_list(value: list) -> tuple:
    """(cfunc) Encode list value into tuple of MySQL strings `<'tuple'>`."""
    return list_to_tuple([_encode_item_common(i) for i in value])


@cython.cfunc
@cython.inline(True)
def _encode_tuple(value: tuple) -> tuple:
    """(cfunc) Encode tuple value into tuple of MySQL strings `<'tuple'>`."""
    return list_to_tuple([_encode_item_common(i) for i in value])


@cython.cfunc
@cython.inline(True)
def _encode_set(value: set) -> tuple:
    """(cfunc) Encode set value into tuple of MySQL strings `<'tuple'>`."""
    return list_to_tuple([_encode_item_common(i) for i in value])


@cython.cfunc
@cython.inline(True)
def _encode_frozenset(value: frozenset) -> tuple:
    """(cfunc) Encode set value into tuple of MySQL strings `<'tuple'>`."""
    return list_to_tuple([_encode_item_common(i) for i in value])


@cython.cfunc
@cython.inline(True)
def _encode_sequence(value: Iterable) -> tuple:
    """(cfunc) Encode sequence value into tuple of MySQL strings `<'tuple'>`."""
    return list_to_tuple([_encode_item_common(i) for i in value])


# . Numpy ndarray - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _encode_ndarray(value: np.ndarray) -> tuple:
    """(cfunc) Encode numpy.ndarray value into tuple of MySQL strings `<'tuple'>`."""
    # Validate ndarray dimensions & size
    if value.ndim != 1:
        raise ValueError("only supports 1-dimensional <'numpy.ndarray'>.")
    size: cython.Py_ssize_t = value.shape[0]
    if size == 0:
        return ()  # exit
    # Encode ndarray
    dtype: cython.char = value.descr.kind
    # . ndarray[object]
    if dtype == NDARRAY_DTYPE_OBJECT:
        return list_to_tuple(_encode_ndarray_object(value, size))
    # . ndarray[float]
    if dtype == NDARRAY_DTYPE_FLOAT:
        return list_to_tuple(_encode_ndarray_float(value, size))
    # . ndarray[int]
    if dtype == NDARRAY_DTYPE_INT:
        return list_to_tuple(_encode_ndarray_int(value, size))
    # . ndarray[uint]
    if dtype == NDARRAY_DTYPE_UINT:
        return list_to_tuple(_encode_ndarray_int(value, size))
    # . ndarray[bool]
    if dtype == NDARRAY_DTYPE_BOOL:
        return list_to_tuple(_encode_ndarray_bool(value, size))
    # . ndarray[datetime64]
    if dtype == NDARRAY_DTYPE_DT64:
        return list_to_tuple(_encode_ndarray_dt64(value, size))
    # . ndarray[timedelta64]
    if dtype == NDARRAY_DTYPE_TD64:
        return list_to_tuple(_encode_ndarray_td64(value, size))
    # . ndarray[bytes]
    if dtype == NDARRAY_DTYPE_BYTES:
        return list_to_tuple(_encode_ndarray_bytes(value, size))
    # . ndarray[str]
    if dtype == NDARRAY_DTYPE_UNICODE:
        return list_to_tuple(_encode_ndarray_unicode(value, size))
    # . invalid dtype
    raise TypeError("unsupported <'numpy.ndarray'> dtype [%s]." % value.dtype)


@cython.cfunc
@cython.inline(True)
def _encode_ndarray_object(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode numpy.ndarray value into `list` of MySQL strings `<'list'>`.

    This function is specifically for ndarray with dtype of: "O" (object).

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
    """
    return [_encode_item_common(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore


@cython.cfunc
@cython.inline(True)
def _encode_ndarray_float(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode numpy.ndarray value into `list` of MySQL strings `<'list'>`.

    This function is specifically for ndarray with dtype of:
    "f" (float), "i" (int) and "u" (uint).

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
    """
    # Check if any item is not finite.
    if not is_arr_double_finite(arr, size):  # type: ignore
        raise TypeError("float value of 'nan' & 'inf' can not be used with MySQL.")
    # For size < 30, serialize as list is faster.
    if size < 30:
        res: str = _orjson_dumps(np.PyArray_ToList(arr))
    # For larger size, serialize as ndarray is faster.
    else:
        res: str = _orjson_dumps_numpy(arr)
    return str_split(str_substr(res, 1, str_len(res) - 1), ",", -1)


@cython.cfunc
@cython.inline(True)
def _encode_ndarray_int(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode numpy.ndarray value into `list` of MySQL strings `<'list'>`.

    This function is specifically for ndarray with dtype of:
    "i" (int) and "u" (uint).

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
    """
    # For size < 60, serialize as list is faster.
    if size < 60:
        res: str = _orjson_dumps(np.PyArray_ToList(arr))
    # For larger size, serialize as ndarray is faster.
    else:
        res: str = _orjson_dumps_numpy(arr)
    return str_split(str_substr(res, 1, str_len(res) - 1), ",", -1)


@cython.cfunc
@cython.inline(True)
def _encode_ndarray_bool(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode numpy.ndarray value into `list` of MySQL strings `<'list'>`.

    This function is specifically for ndarray with dtype of: "b" (bool).

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
    """
    return ["1" if arr_getitem_1d_bint(arr, i) else "0" for i in range(size)]  # type: ignore


@cython.cfunc
@cython.inline(True)
def _encode_ndarray_dt64(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode numpy.ndarray value into `list` of MySQL strings `<'list'>`.

    This function is specifically for ndarray with dtype of: "M" (datetime64).

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
    """
    # Notes: This approach is faster than encoding each element individually.
    # 'orjson' returns '["1970-01-01T00:00:00",...,"2000-01-01T00:00:01"]',
    # so character ['"', "T"] will be replaced to comply with MySQL datetime
    # format, and then split by "," into sequence of MySQL strings.
    res: str = _orjson_dumps_numpy(arr)
    res = str_substr(res, 1, str_len(res) - 1)
    return str_split(res.translate(DT64_JSON_TABLE), ",", -1)


@cython.cfunc
@cython.inline(True)
def _encode_ndarray_td64(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode numpy.ndarray value into `list` of MySQL strings `<'list'>`.

    This function is specifically for ndarray with dtype of: "m" (timedelta64).

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
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
def _encode_ndarray_bytes(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode numpy.ndarray value into `list` of MySQL strings `<'list'>`.

    This function is specifically for ndarray with dtype of: "S" (bytes string).

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
    """
    return [_escape_bytes(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore


@cython.cfunc
@cython.inline(True)
def _encode_ndarray_unicode(arr: np.ndarray, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode numpy.ndarray value into `list` of MySQL strings `<'list'>`.

    This function is specifically for ndarray with dtype of: "S" (bytes string).

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
    """
    return [_escape_str(arr_getitem_1d(arr, i)) for i in range(size)]  # type: ignore


# . Pandas Series - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _encode_series(value: Series) -> tuple:
    """(cfunc) Encode pandas.Series value into tuple of MySQL strings `<'tuple'>`."""
    try:
        arr: np.ndarray = value.values
    except Exception as err:
        raise TypeError("Expects <'pandas.Series'>, got %s." % type(value)) from err
    return _encode_ndarray(arr)


# . Pandas DataFrame - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _encode_dataframe(value: DataFrame) -> tuple:
    """(cfunc) Encode pandas.DataFrame value into tuple of MySQL strings `<'tuple'>`."""
    # Validate shape
    shape: tuple = value.shape
    width: cython.Py_ssize_t = shape[1]
    if width == 0:
        return ()  # exit
    size: cython.Py_ssize_t = shape[0]
    if size == 0:
        return ()  # exit

    # Encode DataFrame
    rows: list = []
    cols: list[list] = [_encode_dataframe_column(col, size) for _, col in value.items()]
    for i in range(size):
        row: list = []
        for j in range(width):
            row.append(cols[j][i])
        rows.append(list_to_tuple(row))
    return list_to_tuple(rows)


@cython.cfunc
@cython.inline(True)
def _encode_dataframe_column(value: Series, size: cython.Py_ssize_t) -> list:
    """(cfunc) Encode pandas.DataFrame column into list of MySQL strings `<'list'>`.

    ### Notice
    This function returns `LIST` instead of `TUPLE` for performance
    reason, please convert it to <'tuple'> for final result.
    """
    # Get values
    try:
        arr: np.ndarray = value.values
    except Exception as err:
        raise TypeError("Expects <'pandas.Series'>, got %s." % type(value)) from err
    # Encode ndarray
    dtype: cython.char = arr.descr.kind
    # . ndarray[object]
    if dtype == NDARRAY_DTYPE_OBJECT:
        return _encode_ndarray_object(arr, size)
    # . ndarray[float]
    if dtype == NDARRAY_DTYPE_FLOAT:
        return _encode_ndarray_float(arr, size)
    # . ndarray[int]
    if dtype == NDARRAY_DTYPE_INT:
        return _encode_ndarray_int(arr, size)
    # . ndarray[uint]
    if dtype == NDARRAY_DTYPE_UINT:
        return _encode_ndarray_int(arr, size)
    # . ndarray[bool]
    if dtype == NDARRAY_DTYPE_BOOL:
        return _encode_ndarray_bool(arr, size)
    # . ndarray[datetime64]
    if dtype == NDARRAY_DTYPE_DT64:
        return _encode_ndarray_dt64(arr, size)
    # . ndarray[timedelta64]
    if dtype == NDARRAY_DTYPE_TD64:
        return _encode_ndarray_td64(arr, size)
    # . ndarray[bytes]
    if dtype == NDARRAY_DTYPE_BYTES:
        return _encode_ndarray_bytes(arr, size)
    # . ndarray[str]
    if dtype == NDARRAY_DTYPE_UNICODE:
        return _encode_ndarray_unicode(arr, size)
    # . invalid dtype
    raise TypeError("unsupported <'Series'> dtype [%s]." % value.dtype)


# . Encode - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.cfunc
@cython.inline(True)
def _encode_item_common(value: object) -> object:
    """(cfunc) Encode common item value into string
    formatable object `<'str/tuple'>`."""
    # Get data type
    dtype = type(value)

    ##### Common Types #####
    # Basic Types
    # . <'str'>
    if dtype is str:
        return _escape_str(value)
    # . <'float'>
    if dtype is float:
        return _escape_float(value)
    # . <'int'>
    if dtype is int:
        return _escape_int(value)
    # . <'bool'>
    if dtype is bool:
        return _escape_bool(value)
    # . <None>
    if dtype is typeref.NONE:
        return _escape_none(value)

    # Date&Time Types
    # . <'datetime.datetime'>
    if dtype is datetime.datetime:
        return _escape_datetime(value)
    # . <'datetime.date'>
    if dtype is datetime.date:
        return _escape_date(value)
    # . <'datetime.time'>
    if dtype is datetime.time:
        return _escape_time(value)
    # . <'datetime.timedelta'>
    if dtype is datetime.timedelta:
        return _escape_timedelta(value)

    # Numeric Types
    # . <'decimal.Decimal'>
    if dtype is typeref.DECIMAL:
        return _escape_decimal(value)

    # Bytes Types
    # . <'bytes'>
    if dtype is bytes:
        return _escape_bytes(value)

    # Mapping Types
    # . <'dict'>
    if dtype is dict:
        return _encode_dict(value)

    # Sequence Types
    # . <'list'>
    if dtype is list:
        return _encode_list(value)
    # . <'tuple'>
    if dtype is tuple:
        return _encode_tuple(value)

    ##### Uncommon Types #####
    return _encode_item_uncommon(value, dtype)


@cython.cfunc
@cython.inline(True)
def _encode_item_uncommon(value: object, dtype: type) -> object:
    """(cfunc) Encode uncommon item value into string
    formatable object `<'str/tuple'>`."""
    ##### Uncommon Types #####
    # Basic Types
    # . <'numpy.float_'>
    if dtype is typeref.FLOAT64 or dtype is typeref.FLOAT32 or dtype is typeref.FLOAT16:
        return _escape_float64(value)
    # . <'numpy.int_'>
    if (
        dtype is typeref.INT64
        or dtype is typeref.INT32
        or dtype is typeref.INT16
        or dtype is typeref.INT8
    ):
        return _escape_int(value)
    # . <'numpy.uint'>
    if (
        dtype is typeref.UINT64
        or dtype is typeref.UINT32
        or dtype is typeref.UINT16
        or dtype is typeref.UINT8
    ):
        return _escape_int(value)
    # . <'numpy.bool_'>
    if dtype is typeref.BOOL_:
        return _escape_bool(value)

    # Date&Time Types
    # . <'pandas.Timestamp'>
    if dtype is typeref.TIMESTAMP:
        return _escape_datetime(value)
    # . <'pandas.Timedelta'>`
    if dtype is typeref.TIMEDELTA:
        return _escape_timedelta(value)
    # . <'numpy.datetime64'>
    if dtype is typeref.DATETIME64:
        return _escape_datetime64(value)
    # . <'numpy.timedelta64'>
    if dtype is typeref.TIMEDELTA64:
        return _escape_timedelta64(value)
    # . <'time.struct_time'>`
    if dtype is typeref.STRUCT_TIME:
        return _escape_struct_time(value)
    # . <'cytimes.pydt'>
    if dtype is typeref.PYDT:
        return _escape_datetime(value.dt)

    # Bytes Types
    # . <'bytearray'>
    if dtype is bytearray:
        return _escape_bytearray(value)
    # . <'memoryview'>
    if dtype is memoryview:
        return _escape_memoryview(value)
    # . <'numpy.bytes_'>
    if dtype is typeref.BYTES_:
        return _escape_bytes(value)

    # String Types:
    # . <'numpy.str_'>
    if dtype is typeref.STR_:
        return _escape_str(value)

    # Sequence Types
    # . <'set'>
    if dtype is set:
        return _encode_set(value)
    # . <'frozenset'>
    if dtype is frozenset:
        return _encode_frozenset(value)
    # . <'range'> & <'dict_keys'> & <'dict_values'>
    if dtype is range or dtype is typeref.DICT_KEYS or dtype is typeref.DICT_VALUES:
        return _encode_sequence(value)

    # Numpy Types
    # . <'numpy.ndarray'>
    if dtype is np.ndarray:
        return _encode_ndarray(value)
    # . <'numpy.record'>
    if dtype is typeref.RECORD:
        return _encode_sequence(value)

    # Pandas Types
    # . <'pandas.Series'> & <'pandas.DatetimeIndex'> & <'pandas.TimedeltaIndex'>
    if (
        dtype is typeref.SERIES
        or dtype is typeref.DATETIMEINDEX
        or dtype is typeref.TIMEDELTAINDEX
    ):
        return _encode_series(value)
    # . <'cytimes.pddt'>
    if dtype is typeref.PDDT:
        return _encode_series(value.dt)
    # . <'pandas.DataFrame'>
    if dtype is typeref.DATAFRAME:
        return _encode_dataframe(value)

    ##### Subclass Types #####
    return _encode_item_subclass(value, dtype)


@cython.cfunc
@cython.inline(True)
def _encode_item_subclass(value: object, dtype: type) -> object:
    """(cfunc) Encode common item value into string
    formatable object `<'str/tuple'>`."""
    ##### Subclass Types #####
    # Basic Types
    # . subclass of <'str'>
    if isinstance(value, str):
        return _escape_str(value)
    # . subclass of <'float'>
    if isinstance(value, float):
        return _escape_float(value)
    # . subclass of <'int'>
    if isinstance(value, int):
        return _escape_int(value)
    # . subclass of <'bool'>
    if isinstance(value, bool):
        return _escape_bool(value)

    # Date&Time Types
    # . subclass of <'datetime.datetime'>
    if isinstance(value, datetime.datetime):
        return _escape_datetime(value)
    # . subclass of <'datetime.date'>
    if isinstance(value, datetime.date):
        return _escape_date(value)
    # . subclass of <'datetime.time'>
    if isinstance(value, datetime.time):
        return _escape_time(value)
    # . subclass of <'datetime.timedelta'>
    if isinstance(value, datetime.timedelta):
        return _escape_timedelta(value)

    # Numeric Types
    # . subclass of <'decimal.Decimal'>
    if isinstance(value, typeref.DECIMAL):
        return _escape_decimal(value)

    # Bytes Types
    # . subclass of <'bytes'>
    if isinstance(value, bytes):
        return _escape_bytes(value)
    # . subclass of <'bytearray'>
    if isinstance(value, bytearray):
        return _escape_bytearray(value)

    # Mapping Types
    # . subclass of <'dict'>
    if isinstance(value, dict):
        return _encode_dict(value)

    # Sequence Types
    # . subclass of <'list'>
    if isinstance(value, list):
        return _encode_list(value)
    # . subclass of <'tuple'>
    if isinstance(value, tuple):
        return _encode_tuple(value)
    # . subclass of <'set'>
    if isinstance(value, set):
        return _encode_set(value)
    # . subclass of <'frozenset'>
    if isinstance(value, frozenset):
        return _encode_frozenset(value)

    # Numpy Types
    # . subclass of <'numpy.ndarray'>
    if isinstance(value, np.ndarray):
        return _encode_ndarray(value)

    # Invalid Data Type
    raise TypeError("unsupported data type %s" % dtype)


@cython.ccall
def encode_item(value: object) -> object:
    """(cfunc) Encode common item value into string
    formatable object `<'str/tuple'>`."""
    try:
        return _encode_item_common(value)
    except Exception as err:
        raise errors.EncodeTypeError(
            "Failed to encode %s: %s" % (type(value), err)
        ) from err


# Decoder -------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _decode_string(
    value: bytes, encoding: cython.pchar, is_binary: cython.bint
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


# . Dncode - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@cython.ccall
def decode_item(
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
