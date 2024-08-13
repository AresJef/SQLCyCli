# cython: language_level=3
cimport cython
from libc.math cimport isfinite
from libc.stdlib cimport strtoll, strtoull, strtold
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.bytes cimport PyBytes_GET_SIZE as bytes_len
from cpython.bytes cimport PyBytes_AS_STRING as bytes_to_chars
from cpython.unicode cimport PyUnicode_Decode, PyUnicode_DecodeUTF8
from cpython.unicode cimport PyUnicode_Replace, PyUnicode_AsEncodedString
cimport numpy as np
from numpy cimport PyArray_TYPE, PyArray_Cast, PyArray_DATA
from numpy cimport PyArray_GETITEM, PyArray_GETPTR1, PyArray_GETPTR2

# Constants
cdef:
    # . translate table
    list STR_ESCAPE_TABLE, DT64_JSON_TABLE
    # . time units
    unsigned int[13] DAYS_BR_MONTH
    long long US_DAY, US_HOUR, EPOCH_US, DT_MAX_US, DT_MIN_US
    # . datetime
    unsigned int[5] US_FRACTION_CORRECTION
    # . ndarray dtype
    np.ndarray _arr
    char NDARRAY_DTYPE_OBJECT, NDARRAY_DTYPE_BOOL, NDARRAY_DTYPE_FLOAT
    char NDARRAY_DTYPE_INT, NDARRAY_DTYPE_UINT
    char NDARRAY_DTYPE_DT64, NDARRAY_DTYPE_TD64
    char NDARRAY_DTYPE_BYTES, NDARRAY_DTYPE_UNICODE
    # . functions
    object FN_ORJSON_LOADS, FN_ORJSON_DUMPS, FN_ORJSON_OPT_NUMPY, FN_MYSQLCLI_STR2LIT

# Struct
ctypedef struct ymd:
    unsigned int year
    unsigned int month
    unsigned int day

ctypedef struct hms:
    unsigned int hour
    unsigned int minute
    unsigned int second
    unsigned int microsecond

# Utils
cdef inline object encode_str(object obj, char* encoding):
    """Encode string to bytes using 'encoding' with
    'surrogateescape' error handling `<'bytes'>`."""
    return PyUnicode_AsEncodedString(obj, encoding, "surrogateescape")

cdef inline str decode_bytes(object value, char* encoding):
    """Decode bytes to string using 'encoding' with "surrogateescape" error handling `<'str'>`."""
    cdef char* chs = bytes_to_chars(value)
    return PyUnicode_Decode(chs, bytes_len(value), encoding, "surrogateescape")

cdef inline str decode_bytes_ascii(object value):
    """Decode bytes to string using 'ascii' encoding
    with 'surrogateescape' error handling `<'str'>`."""
    cdef char* chs = bytes_to_chars(value)
    return PyUnicode_Decode(chs, bytes_len(value), "ascii", "surrogateescape")

cdef inline str decode_bytes_utf8(object value):
    """Decode bytes to string using 'utf-8' encoding
    with 'surrogateescape' error handling `<'str'>`."""
    cdef char* chs = bytes_to_chars(value)
    return PyUnicode_DecodeUTF8(chs, bytes_len(value), "surrogateescape")

cdef inline str replace_bracket(str value, Py_ssize_t maxcount):
    """Replace '[' and ']' with '(' and ')' respectively `<'str'>`."""
    return PyUnicode_Replace(PyUnicode_Replace(
        value, "[", "(", maxcount
        ), "]", ")", maxcount
    )

cdef inline bint is_leapyear(unsigned int year) except -1:
    """Determine whether the given 'year' is a leap year `<'bool'>`."""
    if year == 0:
        return False
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

cdef inline unsigned int days_bf_month(unsigned int year, unsigned int month):
    """Calculate the number of days between the 1st day
    of the given 'year' and the 1st day of the 'month' `<'int'>`."""
    cdef unsigned int days
    if month <= 2:
        return 31 if month == 2 else 0
    days = DAYS_BR_MONTH[min(month, 12) -1]
    if is_leapyear(year):
        days += 1
    return days

cdef inline ymd ordinal_to_ymd(int ordinal) except *:
    """Convert ordinal to YMD `<'struct:ymd'>`."""
    # n is a 1-based index, starting at 1-Jan-1.  The pattern of leap years
    # repeats exactly every 400 years.  The basic strategy is to find the
    # closest 400-year boundary at or before n, then work with the offset
    # from that boundary to n.  Life is much clearer if we subtract 1 from
    # n first -- then the values of n at 400-year boundaries are exactly
    # those divisible by _DI400Y:
    cdef:
        unsigned int n = min(max(ordinal, 1), 3_652_059) - 1
        unsigned int n400 = n // 146_097
        unsigned int year, month, days_bf
        unsigned int n100, n4, n1
    n = n % 146_097
    year = n400 * 400 + 1

    # Now n is the (non-negative) offset, in days, from January 1 of year, to
    # the desired date.  Now compute how many 100-year cycles precede n.
    # Note that it's possible for n100 to equal 4!  In that case 4 full
    # 100-year cycles precede the desired day, which implies the desired
    # day is December 31 at the end of a 400-year cycle.
    n100 = n // 36_524
    n = n % 36_524

    # Now compute how many 4-year cycles precede it.
    n4 = n // 1_461
    n = n % 1_461

    # And now how many single years.  Again n1 can be 4, and again meaning
    # that the desired day is December 31 at the end of the 4-year cycle.
    n1 = n // 365
    n = n % 365

    # We now know the year and the offset from January 1st.  Leap years are
    # tricky, because they can be century years.  The basic rule is that a
    # leap year is a year divisible by 4, unless it's a century year --
    # unless it's divisible by 400.  So the first thing to determine is
    # whether year is divisible by 4.  If not, then we're done -- the answer
    # is December 31 at the end of the year.
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        return ymd(year - 1, 12, 31)  # type: ignore

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    month = (n + 50) >> 5
    days_bf = days_bf_month(year, month)
    if days_bf > n:
        month -= 1
        days_bf = days_bf_month(year, month)
    n = n - days_bf + 1
    return ymd(year, month, n)  # type: ignore

@cython.cdivision(True)
cdef inline hms microseconds_to_hms(long long microseconds) except *:
    """Convert microseconds to HMS `<'struct:hms'>`."""
    cdef unsigned int hour, minute, second, microsecond
    if microseconds <= 0:
        return hms(0, 0, 0, 0)  # exit

    microseconds = microseconds % US_DAY
    hour = microseconds // US_HOUR
    microseconds = microseconds % US_HOUR
    minute = microseconds // 60_000_000
    microseconds = microseconds % 60_000_000
    second = microseconds // 1_000_000
    microsecond = microseconds % 1_000_000
    return hms(hour, minute, second, microsecond)

cdef inline object arr_getitem_1d(np.ndarray arr, np.npy_intp i):
    """Get item from 1-dimensional numpy ndarray as `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR1(arr, i)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

cdef inline bint arr_getitem_1d_bint(np.ndarray arr, np.npy_intp i) except -1:
    """Get item from 1-dimensional numpy ndarray as `<'bint'>`."""
    cdef char* item = <char*>PyArray_GETPTR1(arr, i)
    return item[0]

cdef inline long long arr_getitem_1d_ll(np.ndarray arr, np.npy_intp i):
    """Get item from 1-dimensional numpy ndarray as `<'long long'>`."""
    cdef long long* item = <long long*>PyArray_GETPTR1(arr, i)
    return item[0]

cdef inline bint is_arr_float_finite_1d(np.ndarray arr, np.npy_intp s_i) except -1:
    """Check if all items for 1-dimensional ndarray[double] is finite `<'bool'>`."""
    cdef:
        int npy_type = PyArray_TYPE(arr)
        double* d_ptr
        float* f_ptr
    # Check float64
    if npy_type == np.NPY_TYPES.NPY_FLOAT64:
        d_ptr = <double*> PyArray_DATA(arr)
        for i in range(s_i):
            if not isfinite(d_ptr[i]):
                return False
        return True
    # Cast: float16 -> float32
    if npy_type == np.NPY_TYPES.NPY_FLOAT16:
        arr = PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        npy_type = np.NPY_TYPES.NPY_FLOAT32
    # Check float32
    if npy_type == np.NPY_TYPES.NPY_FLOAT32:
        f_ptr = <float*> PyArray_DATA(arr)
        for i in range(s_i):
            if not isfinite(f_ptr[i]):
                return False
        return True
    # Invalid dtype
    raise TypeError("Unsupported <'np.ndarray'> float dtype: %s." % arr.dtype)

cdef inline object arr_getitem_2d(np.ndarray arr, np.npy_intp i, np.npy_intp j):
    """Get item from 2-dimensional numpy ndarray as `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR2(arr, i, j)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

cdef inline bint arr_getitem_2d_bint(np.ndarray arr, np.npy_intp i, np.npy_intp j) except -1:
    """Get item from 2-dimensional numpy ndarray as `<'bint'>`."""
    cdef char* item = <char*>PyArray_GETPTR2(arr, i, j)
    return item[0]

cdef inline long long arr_getitem_2d_ll(np.ndarray arr, np.npy_intp i, np.npy_intp j):
    """Get item from 2-dimensional numpy ndarray as `<'long long'>`."""
    cdef long long* item = <long long*>PyArray_GETPTR2(arr, i, j)
    return item[0]

cdef inline bint is_arr_float_finite_2d(np.ndarray arr, np.npy_intp s_i, np.npy_intp s_j) except -1:
    """Check if all items for 2-dimensional ndarray[double] is finite `<'bool'>`."""
    cdef:
        int npy_type = PyArray_TYPE(arr)
        double* d_ptr
        float* f_ptr
    # Check float64
    if npy_type == np.NPY_TYPES.NPY_FLOAT64:
        d_ptr = <double*> PyArray_DATA(arr)
        for i in range(s_i):
            for j in range(s_j):
                if not isfinite(d_ptr[i * s_j + j]):
                    return False
        return True
    # Cast: float16 -> float32
    if npy_type == np.NPY_TYPES.NPY_FLOAT16:
        arr = PyArray_Cast(arr, np.NPY_TYPES.NPY_FLOAT32)
        npy_type = np.NPY_TYPES.NPY_FLOAT32
    # Check float32
    if npy_type == np.NPY_TYPES.NPY_FLOAT32:
        f_ptr = <float*> PyArray_DATA(arr)
        for i in range(s_i):
            for j in range(s_j):
                if not isfinite(f_ptr[i * s_j + j]):
                    return False
        return True
    # Invalid dtype
    raise TypeError("Unsupported <'np.ndarray'> float dtype: %s." % arr.dtype)

cdef inline long long nptime_to_microseconds(np.npy_datetime value, np.NPY_DATETIMEUNIT unit):
    """Convert numpy.datetime64/timedelta64 value into microseconds `<'long long'>`."""
    # . common units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:  # nanosecond
        return value // 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:  # microsecond
        return value
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:  # millisecond
        return value * 1_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:  # second
        return value * 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:  # minute
        return value * 60_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:  # hour
        return value * US_HOUR
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:  # day
        return value * US_DAY
    # . uncommon units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:  # picosecond
        return value // 1_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:  # femtosecond
        return value // 1_000_000_000
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:  # attosecond
        return value // 1_000_000_000 // 1_000
    # . unsupported unit
    raise ValueError(
        "Unsupported <'numpy.datetime64/timedelta64'> time unit "
        "to perform conversion: %d." % unit
    )

cdef inline long long dt64_to_microseconds(object dt64):
    """Convert numpy.datetime64 to microseconds `<'long long'>`."""
    return nptime_to_microseconds(np.get_datetime64_value(dt64), np.get_datetime64_unit(dt64))

cdef inline long long td64_to_microseconds(object td64):
    """Convert numpy.timedelta64 to total microseconds `<'long long'>`."""
    return nptime_to_microseconds(np.get_timedelta64_value(td64), np.get_datetime64_unit(td64))

cdef inline char* slice_to_chars(char* data, Py_ssize_t start, Py_ssize_t size):
    """Slice data `<'char*'>` from 'start' to 'start + size' `<'char*'>`."""
    return bytes_to_chars(PyBytes_FromStringAndSize(data + start, size))

cdef inline long long slice_to_int(char* data, Py_ssize_t start, Py_ssize_t end):
    """Slice data `<'char*'>` from 'start' to 'end', and convert to `<'long long'>`."""
    # Validate integer
    cdef Py_ssize_t size = end - start
    if size < 1:
        raise ValueError("Invalid integer from slice.")
    # Slice & Convert to long long
    return strtoll(slice_to_chars(data, start, size), NULL, 10)

cdef inline int parse_us_fraction(char* data, Py_ssize_t start, Py_ssize_t end):
    """Parse microsecond fraction from 'data' `<'char*'>` from 'start' to 'end' `<'int'>`."""
    # Validate fraction
    cdef Py_ssize_t size = end - start
    if size > 6:
        size = 6
    elif size < 1:
        raise ValueError("Invalid microsecond fraction.")
    # Slice & Convert to int
    cdef int res = strtoll(slice_to_chars(data, start, size), NULL, 10)
    # Adjust fraction
    if size < 6:
        res *= US_FRACTION_CORRECTION[size - 1]
    return res

cdef inline long long chars_to_ll(char* data):
    """Convert 'data' `<'char*'>` to `<'long long'>`."""
    return strtoll(data, NULL, 10)

cdef inline unsigned long long chars_to_ull(char* data):
    """Convert 'data' `<'char*'>` to `<'unsigned long long'>`."""
    return strtoull(data, NULL, 10)

cdef inline long double chars_to_double(char* data):
    """Convert 'data' `<'char*'>` to `<'long double'>`."""
    return strtold(data, NULL)

cdef inline unsigned long long unpack_uint64_big_endian(char* data, unsigned long long pos):
    """Read (unpack) unsigned 64-bit integer from 'data' at givent 'pos' `<'int'>`.
    
    Note: The data is assumed to be in big-endian format.
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

# Custom types
cdef class _CustomType:
    cdef object _value

cdef class BIT(_CustomType):
    pass

cdef class JSON(_CustomType):
    pass

# Escape
cpdef object escape(object value, char* encoding, bint itemize=?, bint many=?)

# Decode
cpdef object decode(
    bytes value, unsigned int field_type, char* encoding, bint is_binary,
    bint use_decimal, bint decode_bit, bint decode_json)
