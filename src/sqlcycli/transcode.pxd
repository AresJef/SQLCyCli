# cython: language_level=3
cimport cython
from libc cimport math
from libc.stdlib cimport strtoll, strtoull, strtold
from cpython.bytes cimport PyBytes_Size as bytes_len
from cpython.bytes cimport PyBytes_AsString as bytes_to_chars
from cpython.unicode cimport PyUnicode_Translate, PyUnicode_AsEncodedString
from cpython.unicode cimport PyUnicode_Decode, PyUnicode_DecodeUTF8, PyUnicode_DecodeASCII
cimport numpy as np

# Constants
cdef:
    # orjson options
    object OPT_SERIALIZE_NUMPY
    # . translate table
    list STR_ESCAPE_TABLE
    list DT64_JSON_TABLE
    list BRACKET_TABLE
    # . ndarray dtype
    char NDARRAY_OBJECT
    char NDARRAY_INT
    char NDARRAY_UINT
    char NDARRAY_FLOAT
    char NDARRAY_BOOL
    char NDARRAY_DT64
    char NDARRAY_TD64
    char NDARRAY_BYTES
    char NDARRAY_UNICODE
    # . calendar
    int[13] DAYS_BR_MONTH
    # . date
    int ORDINAL_MAX
    # . datetime
    #: EPOCH (1970-01-01)
    long long EPOCH_YEAR
    long long EPOCH_MONTH
    long long EPOCH_DAY
    long long EPOCH_HOUR
    long long EPOCH_MINUTE
    long long EPOCH_SECOND
    long long EPOCH_MILLISECOND
    long long EPOCH_MICROSECOND
    #: fraction correction
    int[5] US_FRAC_CORRECTION
    # . conversion for seconds
    long long SS_MINUTE
    long long SS_HOUR
    long long SS_DAY
    # . conversion for milliseconds
    long long MS_SECOND
    long long MS_MINUTE
    long long MS_HOUR
    long long MS_DAY
    # . conversion for microseconds
    long long US_MILLISECOND
    long long US_SECOND
    long long US_MINUTE
    long long US_HOUR
    long long US_DAY
    # . conversion for nanoseconds
    long long NS_MICROSECOND
    long long NS_MILLISECOND
    long long NS_SECOND
    long long NS_MINUTE
    long long NS_HOUR
    long long NS_DAY
    # . conversion for timedelta64
    double TD64_YY_DAY
    long long TD64_YY_SECOND
    long long TD64_YY_MILLISECOND
    long long TD64_YY_MICROSECOND
    long long TD64_YY_NANOSECOND
    double TD64_MM_DAY
    long long TD64_MM_SECOND
    long long TD64_MM_MILLISECOND
    long long TD64_MM_MICROSECOND
    long long TD64_MM_NANOSECOND

# Struct
ctypedef struct ymd:
    unsigned int year
    unsigned int month
    unsigned int day

ctypedef struct hmsf:
    unsigned int hour
    unsigned int minute
    unsigned int second
    unsigned int microsecond

# Utils: math
cdef inline long long math_mod(long long num, long long factor):
    """Computes the modulo of a number by the factor, handling
    negative numbers accoring to Python's modulo semantics `<'int'>`.

    Equivalent to:
    >>> num % factor
    """
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.math_mod()'.")

    cdef:
        bint neg_f = factor < 0
        long long r
    
    with cython.cdivision(True):
        r = num % factor
        if r != 0:
            if not neg_f:
                if r < 0:
                    r += factor
            else:
                if r > 0:
                    r += factor
    return r

cdef inline long long math_round_div(long long num, long long factor):
    """Divides a number by the factor and rounds the result to
    the nearest integer (half away from zero), handling negative
    numbers accoring to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> round(num / factor, 0)
    """
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.math_round_div()'.")

    cdef:
        bint neg_f = factor < 0
        long long abs_f = -factor if neg_f else factor
        long long q, r, abs_r
    
    with cython.cdivision(True):
        q = num // factor
        r = num % factor
        abs_r = -r if r < 0 else r
        if abs_r * 2 >= abs_f:
            if (not neg_f and num >= 0) or (neg_f and num < 0):
                q += 1
            else:
                q -= 1
    return q

cdef inline long long math_ceil_div(long long num, long long factor):
    """Divides a number by the factor and rounds the result up
    to the nearest integer, handling negative numbers accoring
    to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> math.ceil(num / factor)
    """
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.math_ceil_div()'.")

    cdef long long q, r
    with cython.cdivision(True):
        q = num // factor
        r = num % factor
        if r != 0:
            if factor > 0:
                if num > 0:
                    q += 1
            else:
                if num < 0:
                    q += 1
    return q

cdef inline long long math_floor_div(long long num, long long factor):
    """Divides a number by the factor and rounds the result
    down to the nearest integer, handling negative numbers
    accoring to Python's division semantics `<'int'>`.

    Equivalent to:
    >>> math.floor(num / factor)
    """
    if factor == 0:
        raise ZeroDivisionError("division by zero for 'utils.math_floor_div()'.")

    cdef long long q, r
    with cython.cdivision(True):
        q = num // factor
        r = num % factor
        if r != 0:
            if factor > 0:
                if num < 0:
                    q -= 1
            else:
                if num > 0:
                    q -= 1
    return q

# Utils: string
cdef inline bytes encode_str(object obj, char* encoding):
    """Encode string to bytes using 'encoding' with
    'surrogateescape' error handling `<'bytes'>`."""
    return PyUnicode_AsEncodedString(obj, encoding, b"surrogateescape")

cdef inline str decode_bytes(object data, char* encoding):
    """Decode bytes to string using 'encoding' with "surrogateescape" error handling `<'str'>`."""
    return PyUnicode_Decode(bytes_to_chars(data), bytes_len(data), encoding, b"surrogateescape")

cdef inline str decode_bytes_utf8(object data):
    """Decode bytes to string using 'utf-8' encoding
    with 'surrogateescape' error handling `<'str'>`."""
    return PyUnicode_DecodeUTF8(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str decode_bytes_ascii(object data):
    """Decode bytes to string using 'ascii' encoding
    with 'surrogateescape' error handling `<'str'>`."""
    return PyUnicode_DecodeASCII(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str translate_str(object value, list table):
    """Translate string with the given 'table' `<'str'>`."""
    return PyUnicode_Translate(value, table, NULL)

cdef inline char* slice_to_chars(char* data, Py_ssize_t start, Py_ssize_t end):
    """Slice 'data' from 'start' to 'end' `<'char*'>`."""
    if end - start < 1:
        raise ValueError("Invalid slice size: [start]%d -> [end]%d." % (start, end))
    return data[start:end]

cdef inline long long slice_to_int(char* data, Py_ssize_t start, Py_ssize_t end):
    """Slice 'data' from 'start' to 'end', and convert to `<'long long'>`."""
    return strtoll(slice_to_chars(data, start, end), NULL, 10)

cdef inline long long chars_to_ll(char* data):
    """Convert 'data' to `<'long long'>`."""
    return strtoll(data, NULL, 10)

cdef inline unsigned long long chars_to_ull(char* data):
    """Convert 'data' to `<'unsigned long long'>`."""
    return strtoull(data, NULL, 10)

cdef inline long double chars_to_ld(char* data):
    """Convert 'data' to `<'long double'>`."""
    return strtold(data, NULL)

# Utils: Unpack unsigned integer
cdef inline unsigned long long unpack_uint64_big_endian(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 64-bit integer from 'data'
    at the given 'pos' in big-endian order `<'int'>`.
    """
    cdef:
        unsigned long long v0 = <unsigned char> data[pos + 7]
        unsigned long long v1 = <unsigned char> data[pos + 6]
        unsigned long long v2 = <unsigned char> data[pos + 5]
        unsigned long long v3 = <unsigned char> data[pos + 4]
        unsigned long long v4 = <unsigned char> data[pos + 3]
        unsigned long long v5 = <unsigned char> data[pos + 2]
        unsigned long long v6 = <unsigned char> data[pos + 1]
        unsigned long long v7 = <unsigned char> data[pos]
        unsigned long long res = (
            v0 | (v1 << 8) | (v2 << 16) | (v3 << 24) | (v4 << 32)
            | (v5 << 40) | (v6 << 48) | (v7 << 56)
        )
    return res

# Utils: date&time
cdef inline bint is_leap_year(int year) except -1:
    """Check if the passed in 'year' is a leap year `<'bool'>`."""
    if year < 1:
        return False
    if year % 4 == 0:
        if year % 400 == 0:
            return True
        if year % 100 != 0:
            return True
    return False

cdef inline unsigned int days_bf_month(int year, int month) except -1:
    """Compute the number of days between the 1st day of the
    'year' and the 1st day of the 'month' `<'int'>`.
    """
    # January
    if month <= 1:
        return 0
    # February
    if month == 2:
        return 31
    # Rest
    cdef int days = DAYS_BR_MONTH[month - 1] if month < 12 else 334
    if is_leap_year(year):
        days += 1
    return days

cdef inline ymd ymd_fr_ordinal(int val) except *:
    """Create 'struct:ymd' from Gregorian ordinal days `<'struct:ymd'>`."""
    # n is a 1-based index, starting at 1-Jan-1.  The pattern of leap years
    # repeats exactly every 400 years.  The basic strategy is to find the
    # closest 400-year boundary at or before n, then work with the offset
    # from that boundary to n.  Life is much clearer if we subtract 1 from
    # n first -- then the values of n at 400-year boundaries are exactly
    # those divisible by _DI400Y:
    cdef:
        int n, n400, n100, n4, n1
        int yy, mm, days_bf

    n = min(max(val, 1), ORDINAL_MAX) - 1
    n400 = n // 146_097
    n -= n400 * 146_097

    # Now n is the (non-negative) offset, in days, from January 1 of year, to
    # the desired date.  Now compute how many 100-year cycles precede n.
    # Note that it's possible for n100 to equal 4!  In that case 4 full
    # 100-year cycles precede the desired day, which implies the desired
    # day is December 31 at the end of a 400-year cycle.
    n100 = n // 36_524
    n -= n100 * 36_524

    # Now compute how many 4-year cycles precede it.
    n4 = n // 1_461
    n -= n4 * 1_461

    # And now how many single years.  Again n1 can be 4, and again meaning
    # that the desired day is December 31 at the end of the 4-year cycle.
    n1 = n // 365
    n -= n1 * 365

    # We now know the year and the offset from January 1st.  Leap years are
    # tricky, because they can be century years.  The basic rule is that a
    # leap year is a year divisible by 4, unless it's a century year --
    # unless it's divisible by 400.  So the first thing to determine is
    # whether year is divisible by 4.  If not, then we're done -- the answer
    # is December 31 at the end of the year.
    yy = n400 * 400 + n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        return ymd(yy, 12, 31)  # exit: Last day of the year
    yy += 1

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    mm = (n + 50) >> 5
    days_bf = days_bf_month(yy, mm)
    if days_bf > n:
        mm -= 1
        days_bf = days_bf_month(yy, mm)
    return ymd(yy, mm, n - days_bf + 1)

cdef inline hmsf hmsf_fr_us(unsigned long long val) except *:
    """Create 'struct:hmsf' from microseconds (int) `<'struct:hmsf'>`.
    
    Notice that the orgin of the microseconds must be 0,
    and `NOT` the Unix Epoch (1970-01-01 00:00:00).
    """
    if val <= 0:
        return hmsf(0, 0, 0, 0)

    val = math_mod(val, US_DAY)
    cdef int hh = math_floor_div(val, US_HOUR)
    val -= hh * US_HOUR
    cdef int mi = math_floor_div(val, US_MINUTE)
    val -= mi * US_MINUTE
    cdef long long ss = math_floor_div(val, US_SECOND)
    return hmsf(hh, mi, ss, val - ss * US_SECOND)

cdef inline int parse_us_fraction(char* data, Py_ssize_t start, Py_ssize_t end):
    """Parse microsecond fraction from 'start' to 'end' of the 'data' `<'int'>`."""
    # Validate fraction
    cdef Py_ssize_t size = end - start
    if size > 6:
        size = 6
    elif size < 1:
        raise ValueError("Invalid microsecond fraction.")
    # Slice & Convert to int
    cdef int res = strtoll(data[start:start+size], NULL, 10)
    # Adjust fraction
    if size < 6:
        res *= US_FRAC_CORRECTION[size - 1]
    return res

# Utils: ndarray 1-dimensional
cdef inline object arr_getitem_1d(np.ndarray arr, np.npy_intp i):
    """Get item from 1-dimensional numpy ndarray as `<'object'>`."""
    cdef void* itemptr = <void*> np.PyArray_GETPTR1(arr, i)
    cdef object item = np.PyArray_GETITEM(arr, itemptr)
    return item

cdef inline bint arr_getitem_1d_bint(np.ndarray arr, np.npy_intp i) except -1:
    """Get item from 1-dimensional numpy ndarray as `<'bint'>`."""
    cdef char* item = <char*> np.PyArray_GETPTR1(arr, i)
    return item[0]

cdef inline long long arr_getitem_1d_ll(np.ndarray arr, np.npy_intp i):
    """Get item from 1-dimensional numpy ndarray as `<'long long'>`."""
    cdef long long* item = <long long*> np.PyArray_GETPTR1(arr, i)
    return item[0]

cdef inline bint is_arr_float_finite_1d(np.ndarray arr, np.npy_intp s_i) except -1:
    """Check if all items for 1-dimensional ndarray[double] is finite `<'bool'>`."""
    cdef:
        int npy_type = np.PyArray_TYPE(arr)
        double* d_ptr
        float* f_ptr
        np.npy_intp i
    # Check float64
    if npy_type == np.NPY_TYPES.NPY_FLOAT64:
        d_ptr = <double*> np.PyArray_DATA(arr)
        for i in range(s_i):
            if not math.isfinite(d_ptr[i]):
                return False
        return True
    # Cast: float16 -> float32
    if npy_type == np.NPY_TYPES.NPY_FLOAT16:
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        npy_type = np.NPY_TYPES.NPY_FLOAT32
    # Check float32
    if npy_type == np.NPY_TYPES.NPY_FLOAT32:
        f_ptr = <float*> np.PyArray_DATA(arr)
        for i in range(s_i):
            if not math.isfinite(f_ptr[i]):
                return False
        return True
    # Invalid dtype
    raise TypeError("Unsupported <'np.ndarray'> float dtype: %s." % arr.dtype)

# Utils: ndarray 2-dimensional
cdef inline object arr_getitem_2d(np.ndarray arr, np.npy_intp i, np.npy_intp j):
    """Get item from 2-dimensional numpy ndarray as `<'object'>`."""
    cdef void* itemptr = <void*> np.PyArray_GETPTR2(arr, i, j)
    cdef object item = np.PyArray_GETITEM(arr, itemptr)
    return item

cdef inline bint arr_getitem_2d_bint(np.ndarray arr, np.npy_intp i, np.npy_intp j) except -1:
    """Get item from 2-dimensional numpy ndarray as `<'bint'>`."""
    cdef char* item = <char*> np.PyArray_GETPTR2(arr, i, j)
    return item[0]

cdef inline long long arr_getitem_2d_ll(np.ndarray arr, np.npy_intp i, np.npy_intp j):
    """Get item from 2-dimensional numpy ndarray as `<'long long'>`."""
    cdef long long* item = <long long*> np.PyArray_GETPTR2(arr, i, j)
    return item[0]

cdef inline bint is_arr_float_finite_2d(np.ndarray arr, np.npy_intp s_i, np.npy_intp s_j) except -1:
    """Check if all items for 2-dimensional ndarray[double] is finite `<'bool'>`."""
    cdef:
        int npy_type = np.PyArray_TYPE(arr)
        double* d_ptr
        float* f_ptr
        np.npy_intp i, j
    # Check float64
    if npy_type == np.NPY_TYPES.NPY_FLOAT64:
        d_ptr = <double*> np.PyArray_DATA(arr)
        for i in range(s_i):
            for j in range(s_j):
                if not math.isfinite(d_ptr[i * s_j + j]):
                    return False
        return True
    # Cast: float16 -> float32
    if npy_type == np.NPY_TYPES.NPY_FLOAT16:
        arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        npy_type = np.NPY_TYPES.NPY_FLOAT32
    # Check float32
    if npy_type == np.NPY_TYPES.NPY_FLOAT32:
        f_ptr = <float*> np.PyArray_DATA(arr)
        for i in range(s_i):
            for j in range(s_j):
                if not math.isfinite(f_ptr[i * s_j + j]):
                    return False
        return True
    # Invalid dtype
    raise TypeError("Unsupported <'np.ndarray'> float dtype: %s." % arr.dtype)

# Utils: Numpy share
cdef inline str map_nptime_unit_int2str(int unit):
    """Map numpy datetime64/timedelta64 unit from integer
    to the corresponding string representation `<'str'>`."""
    # Common units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return "ns"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return "us"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return "ms"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return "s"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return "m"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return "h"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return "D"

    # Uncommon units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return "ps"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return "fs"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return "as"
    # if unit == np.NPY_DATETIMEUNIT.NPY_FR_B:
    #     return "B"

    # Unsupported unit
    raise ValueError("unknown datetime unit '%d'." % unit)

cdef inline bint _raise_dt64_as_int64_unit_error(str reso, int unit, bint is_dt64=True) except -1:
    """(internal) Raise unsupported unit for dt/td_as_int*() function."""
    obj_type = "datetime64" if is_dt64 else "timedelta64"
    try:
        unit_str = map_nptime_unit_int2str(unit)
    except Exception as err:
        raise ValueError(
            "cannot cast %s to an integer under '%s' resolution.\n"
            "%s with datetime unit '%d' is not support∫ed."
            % (obj_type, reso, obj_type, unit)
        ) from err
    else:
        raise ValueError(
            "cannot cast %s[%s] to an integer under '%s' resolution.\n"
            "%s with datetime unit '%s' is not supported."
            % (obj_type, unit_str, reso, obj_type, unit_str)
        )

# Utils: Numpy datetime64
cdef inline bint is_dt64(object obj) except -1:
    """Check if an object is an instance of np.datetime64 `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, np.datetime64)
    """
    return np.is_datetime64_object(obj)

cdef inline bint validate_dt64(object obj) except -1:
    """Validate if an object is an instance of np.datetime64,
    and raises `TypeError` if not."""
    if not np.is_datetime64_object(obj):
        raise TypeError("expects 'np.datetime64', got %s." % type(obj))
    return True

cdef inline np.npy_int64 dt64_as_int64_us(object dt64):
    """Cast np.datetime64 to int64 under 'us' (microsecond) resolution `<'int'>`.
    
    Equivalent to:
    >>> dt64.astype("datetime64[us]").astype("int64")
    """
    # Get unit & value
    validate_dt64(dt64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(dt64)
    cdef np.npy_int64 val = np.get_datetime64_value(dt64)
    # Conversion
    return dt64_val_as_int64_us(val, unit)

cdef inline np.npy_int64 dt64_val_as_int64_us(np.npy_int64 val, np.NPY_DATETIMEUNIT unit):
    """Cast np.datetime64 value to int64 under 'us' (microsecond) resolution `<'int'>`."""
    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * US_MILLISECOND
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * US_SECOND
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * US_MINUTE
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * US_DAY
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _dt64_W_as_int64_D(val, US_DAY)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _dt64_M_as_int64_D(val, US_DAY)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _dt64_Y_as_int64_D(val, US_DAY)
    # . unsupported
    _raise_dt64_as_int64_unit_error("us", unit)

cdef inline np.npy_int64 _dt64_Y_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1):
    """(internal) Convert the value of np.datetime64[Y] to int64 under 'D' resolution `<'int'>`."""
    cdef:
        np.npy_int64 year = val + EPOCH_YEAR
        # Compute leap years
        np.npy_int64 y_1 = year - 1
        np.npy_int64 leaps = (
            (y_1 // 4 - 1970 // 4)
            - (y_1 // 100 - 1970 // 100)
            + (y_1 // 400 - 1970 // 400)
        )
    return (val * 365 + leaps) * factor

cdef inline np.npy_int64 _dt64_M_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1):
    """(internal) Convert the value of np.datetime64[M] to int64 under 'D' resolution `<'int'>`."""
    cdef:
        np.npy_int64 year_ep = val // 12
        np.npy_int64 year = year_ep + EPOCH_YEAR
        np.npy_int64 month = val % 12 + 1
        # Compute leap years
        np.npy_int64 y_1 = year - 1
        np.npy_int64 leaps = (
            (y_1 // 4 - 1970 // 4)
            - (y_1 // 100 - 1970 // 100)
            + (y_1 // 400 - 1970 // 400)
        )
    return (year_ep * 365 + leaps + days_bf_month(year, month)) * factor

cdef inline np.npy_int64 _dt64_W_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1):
    """(internal) Convert the value of np.datetime64[W] to int64 under 'D' resolution `<'int'>`."""
    return val * 7 * factor

# Utils: Numpy timedelta64
cdef inline bint is_td64(object obj) except -1:
    """Check if an object is an instance of np.timedelta64 `<'bool'>`.

    Equivalent to:
    >>> isinstance(obj, np.timedelta64)
    """
    return np.is_timedelta64_object(obj)

cdef inline bint validate_td64(object obj) except -1:
    """Validate if an object is an instance of np.timedelta64,
    and raises `TypeError` if not."""
    if not np.is_timedelta64_object(obj):
        raise TypeError("expects 'np.timedelta64', got %s." % type(obj))
    return True

cdef inline np.npy_int64 td64_as_int64_us(object td64):
    """Cast np.timedelta64 to int64 under 'us' (microsecond) resolution `<'int'>`.
    
    Equivalent to:
    >>> td64.astype("timedelta64[D]").astype("int64")
    """
    # Get unit & value
    validate_td64(td64)
    cdef np.NPY_DATETIMEUNIT unit = np.get_datetime64_unit(td64)
    cdef np.npy_int64 val = np.get_timedelta64_value(td64)
    # Conversion
    return td64_val_as_int64_us(val, unit)

cdef inline np.npy_int64 td64_val_as_int64_us(np.npy_int64 val, np.NPY_DATETIMEUNIT unit):
    """Cast np.timedelta64 value to int64 under 'us' (microsecond) resolution `<'int'>`."""
    # Conversion
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return val // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return val
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return val * US_MILLISECOND
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return val * US_SECOND
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return val * US_MINUTE
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return val * US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return val * US_DAY
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:  # week
        return _td64_W_as_int64_D(val, US_DAY)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:  # month
        return _td64_M_as_int64_D(val, US_DAY)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:  # year
        return _td64_Y_as_int64_D(val, US_DAY)
    # . unsupported unit
    _raise_dt64_as_int64_unit_error("us", unit, False)

cdef inline np.npy_int64 _td64_Y_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1):
    """(internal) Convert the value of np.timedelta[Y] to int64 under 'D' resolution `<'int'>`."""
    # Average number of days in a year: 365.2425
    # We use integer arithmetic by scaling to avoid floating-point inaccuracies.
    # Multiply by 3652425 and divide by 10000 to represent 365.2425 days/year.
    cdef np.npy_int64 days
    if factor == 1:  # day
        return val * 3_652_425 // 10_000  # val * 365.2425
    if factor == 24:  # hour
        return val * 876_582 // 100  # val * 8765.82 (365.2425 * 24)
    if factor == 1_440:  # minute
        return val * 5_259_492 // 10  # val * 525949.2 (365.2425 * 1440)
    if factor == SS_DAY:  # second
        return val * TD64_YY_SECOND
    if factor == MS_DAY:  # millisecond
        return val * TD64_YY_MILLISECOND
    if factor == US_DAY:  # microsecond
        return val * TD64_YY_MICROSECOND
    if factor == NS_DAY:  # nanosecond
        return val * TD64_YY_NANOSECOND
    raise AssertionError("unsupported factor '%d' for timedelta unit 'Y' conversion." % factor)

cdef inline np.npy_int64 _td64_M_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1):
    """(internal) Convert the value of np.timedelta[M] to int64 under 'D' resolution `<'int'>`."""
    # Average number of days in a month: 30.436875 (365.2425 / 12)
    # We use integer arithmetic by scaling to avoid floating-point inaccuracies.
    # Multiply by 30436875 and divide by 1000000 to represent 30.436875 days/month.
    cdef np.npy_int64 days
    if factor == 1: #  day
        return val * 30_436_875 // 1_000_000  # val * 30.436875
    if factor == 24: #  hour
        return val * 730_485 // 1_000  # val * 730.485 (30.436875 * 24)
    if factor == 1_440:  # minute
        return val * 438_291 // 10  # val * 43829.1 (30.436875 * 1440)
    if factor == SS_DAY:  # second
        return val * TD64_MM_SECOND
    if factor == MS_DAY:  # millisecond
        return val * TD64_MM_MILLISECOND
    if factor == US_DAY:  # microsecond
        return val * TD64_MM_MICROSECOND
    if factor == NS_DAY:  # nanosecond
        return val * TD64_MM_NANOSECOND
    raise AssertionError("unsupported factor '%d' for timedelta unit 'M' conversion." % factor)

cdef inline np.npy_int64 _td64_W_as_int64_D(np.npy_int64 val, np.npy_int64 factor=1, np.npy_int64 offset=0):
    """(internal) Convert the value of np.timedelta[W] to int64 under 'D' resolution `<'int'>`."""
    return val * 7 * factor + offset

# Custom types
cdef class _CustomType:
    cdef object _value

cdef class BIT(_CustomType):
    pass

cdef class JSON(_CustomType):
    pass

# Escape
cpdef object escape(object value, bint many=?, bint itemize=?)

# Decode
cpdef object decode(
    bytes value, unsigned int field_type, char* encoding, 
    bint is_binary, bint use_decimal, bint decode_bit, bint decode_json)
