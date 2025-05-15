# cython: language_level=3

from libc.string cimport strchr
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.bytes cimport PyBytes_Size as bytes_len
from cpython.bytes cimport PyBytes_AsString as bytes_to_chars
from cpython.unicode cimport PyUnicode_AsEncodedString, PyUnicode_Decode
from cpython.unicode cimport PyUnicode_DecodeUTF8, PyUnicode_DecodeASCII, PyUnicode_DecodeLatin1
from sqlcycli.charset cimport Charset

# Constants
cdef:
    str DEFAULT_USER
    str DEFUALT_CHARSET
    int MAX_CONNECT_TIMEOUT
    int DEFALUT_MAX_ALLOWED_PACKET
    int MAXIMUM_MAX_ALLOWED_PACKET
    unsigned int MAX_PACKET_LENGTH
    #: Max statement size which :meth:`executemany` generates.
    #: Max size of allowed statement is max_allowed_packet - packet_header_size.
    #: Default value of max_allowed_packet is 1048576.
    unsigned int MAX_STATEMENT_LENGTH
    #: Regular expression for :meth:`Cursor.executemany`.
    #: executemany only supports simple bulk insert.
    #: You can use it to load large dataset.
    object INSERT_VALUES_RE
    #: Regular expression for server version.
    object SERVER_VERSION_RE

    # The following values are for the first byte
    # value of MySQL length encoded integer.
    unsigned char NULL_COLUMN  # 251
    unsigned char UNSIGNED_CHAR_COLUMN  # 251
    unsigned char UNSIGNED_SHORT_COLUMN  # 252
    unsigned char UNSIGNED_INT24_COLUMN  # 253
    unsigned char UNSIGNED_INT64_COLUMN  # 254

# Utils: Connection
cdef bytes gen_connect_attrs(list attrs)
cdef bytes DEFAULT_CONNECT_ATTRS
cdef str validate_arg_str(object arg, str arg_name, str default)
cdef object validate_arg_uint(object arg, str arg_name, unsigned int min_val, unsigned int max_val)
cdef bytes validate_arg_bytes(object arg, str arg_name, char* encoding, str default)
cdef Charset validate_charset(object charset, object collation, str default_charset)
cdef int validate_autocommit(object auto) except -2
cdef int validate_max_allowed_packet(object max_allowed_packet, int default, int maximum)
cdef str validate_sql_mode(object sql_mode)
cdef object validate_ssl(object ssl)

# Utils: string
cdef inline bytes encode_str(object obj, char* encoding):
    """Encode string to bytes using the 'encoding' with
    'surrogateescape' error handling `<'bytes'>`."""
    return PyUnicode_AsEncodedString(obj, encoding, b"surrogateescape")

cdef inline str decode_bytes(object data, char* encoding):
    """Decode bytes to string using the 'encoding' with
    "surrogateescape" error handling `<'str'>`."""
    return PyUnicode_Decode(bytes_to_chars(data), bytes_len(data), encoding, b"surrogateescape")

cdef inline str decode_bytes_utf8(object data):
    """Decode bytes to string using 'utf-8' encoding with
    'surrogateescape' error handling `<'str'>`."""
    return PyUnicode_DecodeUTF8(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str decode_bytes_ascii(object data):
    """Decode bytes to string using 'ascii' encoding with
    'surrogateescape' error handling `<'str'>`."""
    return PyUnicode_DecodeASCII(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str decode_bytes_latin1(object data):
    """Decode bytes to string using 'latin1' encoding with
    'surrogateescape' error handling `<'str'>`."""
    return PyUnicode_DecodeLatin1(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline Py_ssize_t find_null_term(char* data, Py_ssize_t pos) except -2:
    """Find the next NULL-terminated starting from 'pos' in the data `<'int'>`."""
    cdef const char* ptr = data + pos
    loc = strchr(ptr, 0)
    if loc is NULL:
        return -1
    return loc - ptr + pos

# Utils: Pack custom
cdef inline bytes pack_I24B(unsigned int i, unsigned char j):
    """Pack 'I[24bit]B' in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<I", i)[:3] + struct.pack("<B", b)
    """
    cdef char buffer[4]
    buffer[0] = i & 0xFF
    buffer[1] = (i >> 8) & 0xFF
    buffer[2] = (i >> 16) & 0xFF
    buffer[3] = j
    return PyBytes_FromStringAndSize(buffer, 4)

cdef inline bytes pack_IB(unsigned int i, unsigned char j):
    """Pack 'IB' in little-endian order to `<'bytes'>`.

    Equivalent to: 
    >>> struct.pack("<IB", i, j)
    """
    cdef char buffer[5]
    buffer[0] = i & 0xFF
    buffer[1] = (i >> 8) & 0xFF
    buffer[2] = (i >> 16) & 0xFF
    buffer[3] = (i >> 24) & 0xFF
    buffer[4] = j
    return PyBytes_FromStringAndSize(buffer, 5)

cdef inline bytes pack_IIB23s(unsigned int i, unsigned int j, unsigned char k):
    """Pack 'IIB23s' in little-endian order to `<'bytes'>`.

    Equivalent to: 
    >>> struct.pack("<IIB23s", i, j, k, b"")
    """
    cdef char buffer[32]
    buffer[0] = i & 0xFF
    buffer[1] = (i >> 8) & 0xFF
    buffer[2] = (i >> 16) & 0xFF
    buffer[3] = (i >> 24) & 0xFF
    buffer[4] = j & 0xFF
    buffer[5] = (j >> 8) & 0xFF
    buffer[6] = (j >> 16) & 0xFF
    buffer[7] = (j >> 24) & 0xFF
    buffer[8] = k
    for i in range(23):
        buffer[i + 9] = 0
    return PyBytes_FromStringAndSize(buffer, 32)

cdef inline bytes gen_length_encoded_integer(unsigned long long i):
    """Generate 'Length Coded Integer' for MySQL `<'bytes'>."""
    # https://dev.mysql.com/doc/internals/en/integer.html#packet-Protocol::LengthEncodedInteger
    cdef char buffer[9]
    # value 251 is reserved for NULL, so only 0-250, 252-254
    # are used as the first byte of a length-encoded integer.
    if i < UNSIGNED_CHAR_COLUMN:  # 251
        buffer[0] = i & 0xFF
        return PyBytes_FromStringAndSize(buffer, 1)
    elif i < 65_536:  # 1 << 16
        buffer[0] = UNSIGNED_SHORT_COLUMN  # 252
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 3)
    elif i < 16_777_216: # 1 << 24
        buffer[0] = UNSIGNED_INT24_COLUMN  # 253
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        buffer[3] = (i >> 16) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 4)
    else: # 1 << 64
        buffer[0] = UNSIGNED_INT64_COLUMN  # 254
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        buffer[3] = (i >> 16) & 0xFF
        buffer[4] = (i >> 24) & 0xFF
        buffer[5] = (i >> 32) & 0xFF
        buffer[6] = (i >> 40) & 0xFF
        buffer[7] = (i >> 48) & 0xFF
        buffer[8] = (i >> 56) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 9)

# Utils: Pack unsigned integers
cdef inline bytes pack_uint8(unsigned int value):
    """Pack `UNSIGNED` 8-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<B", value)
    """
    cdef char buffer[1]
    buffer[0] = value & 0xFF
    return PyBytes_FromStringAndSize(buffer, 1)

cdef inline bytes pack_uint16(unsigned int value):
    """Pack `UNSIGNED` 16-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<H", value)
    """
    cdef char buffer[2]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 2)

cdef inline bytes pack_uint24(unsigned int value):
    """Pack `UNSIGNED` 24-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<I", value)[:3]
    """
    cdef char buffer[3]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 3)

cdef inline bytes pack_uint32(unsigned long long value):
    """Pack `UNSIGNED` 32-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<I", value)
    """
    cdef char buffer[4]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 4)

cdef inline bytes pack_uint64(unsigned long long value):
    """Pack `UNSIGNED` 64-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<Q", value)
    """
    cdef char buffer[8]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    buffer[4] = (value >> 32) & 0xFF
    buffer[5] = (value >> 40) & 0xFF
    buffer[6] = (value >> 48) & 0xFF
    buffer[7] = (value >> 56) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 8)

# Utils: Unpack unsigned integer
cdef inline unsigned char unpack_uint8(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 8-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    return <unsigned char> data[pos]

cdef inline unsigned short unpack_uint16(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 16-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned short p0 = <unsigned char> data[pos]
        unsigned short p1 = <unsigned char> data[pos + 1]
        unsigned short res = p0 | (p1 << 8)
    return res

cdef inline unsigned int unpack_uint24(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 24-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned int p0 = <unsigned char> data[pos]
        unsigned int p1 = <unsigned char> data[pos + 1]
        unsigned int p2 = <unsigned char> data[pos + 2]
        unsigned int res = p0 | (p1 << 8) | (p2 << 16)
    return res

cdef inline unsigned int unpack_uint32(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 32-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned int p0 = <unsigned char> data[pos]
        unsigned int p1 = <unsigned char> data[pos + 1]
        unsigned int p2 = <unsigned char> data[pos + 2]
        unsigned int p3 = <unsigned char> data[pos + 3]
        unsigned int res = p0 | (p1 << 8) | (p2 << 16) | (p3 << 24)
    return res

cdef inline unsigned long long unpack_uint64(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 64-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned long long p0 = <unsigned char> data[pos]
        unsigned long long p1 = <unsigned char> data[pos + 1]
        unsigned long long p2 = <unsigned char> data[pos + 2]
        unsigned long long p3 = <unsigned char> data[pos + 3]
        unsigned long long p4 = <unsigned char> data[pos + 4]
        unsigned long long p5 = <unsigned char> data[pos + 5]
        unsigned long long p6 = <unsigned char> data[pos + 6]
        unsigned long long p7 = <unsigned char> data[pos + 7]
        unsigned long long res = (
            p0 | (p1 << 8) | (p2 << 16) | (p3 << 24) | (p4 << 32)
            | (p5 << 40) | (p6 << 48) | (p7 << 56) )
    return res

# Utils: Pack signed integer
cdef inline bytes pack_int8(int value):
    """Pack `SIGNED` 8-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<b", value)
    """
    cdef char buffer[1]
    buffer[0] = value & 0xFF
    return PyBytes_FromStringAndSize(buffer, 1)

cdef inline bytes pack_int16(int value):
    """Pack `SIGNED` 16-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<h", value)
    """
    cdef char buffer[2]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 2)

cdef inline bytes pack_int24(int value):
    """Pack `SIGNED` 24-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<i", value)[:3]
    """
    cdef char buffer[3]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 3)

cdef inline bytes pack_int32(long long value):
    """Pack `SIGNED` 32-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<i", value)
    """
    cdef char buffer[4]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 4)

cdef inline bytes pack_int64(long long value):
    """Pack `SIGNED` 64-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<q", value)
    """
    cdef char buffer[8]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    buffer[4] = (value >> 32) & 0xFF
    buffer[5] = (value >> 40) & 0xFF
    buffer[6] = (value >> 48) & 0xFF
    buffer[7] = (value >> 56) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 8)

# Utils: Unpack signed integer
cdef inline signed char unpack_int8(char* data, Py_ssize_t pos):
    """Unpack (read) `SIGNED` 8-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    return <signed char> data[pos]

cdef inline short unpack_int16(char* data, Py_ssize_t pos):
    """Unpack (read) `SIGNED` 16-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    return <short> unpack_uint16(data, pos)

cdef inline int unpack_int24(char* data, Py_ssize_t pos):
    """Unpack (read) `SIGNED` 24-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef int res = <int> unpack_uint24(data, pos)
    return res if res < 0x800000 else res - 0x1000000

cdef inline int unpack_int32(char* data, Py_ssize_t pos):
    """Unpack (read) `SIGNED` 32-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    return <int> unpack_uint32(data, pos)

cdef inline long long unpack_int64(char* data, Py_ssize_t pos):
    """Unpack (read) `SIGNED` 64-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    return <long long> unpack_uint64(data, pos)


