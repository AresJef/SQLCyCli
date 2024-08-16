# cython: language_level=3
from libc.string cimport strchr
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_AsEncodedString
from sqlcycli.charset cimport Charset

# Utils: string
cdef inline object encode_str(object obj, char* encoding):
    """Encode string to bytes using 'encoding' with
    'surrogateescape' error handling `<'bytes'>`."""
    return PyUnicode_AsEncodedString(obj, encoding, "surrogateescape")

cdef inline bytes gen_length_encoded_integer(unsigned long long i):
    """Generate 'Length Coded Integer' for MySQL `<'bytes'>."""
    # https://dev.mysql.com/doc/internals/en/integer.html#packet-Protocol::LengthEncodedInteger
    cdef char buffer[9]
    if i < 0xFB:  # 251 (1 byte)
        buffer[0] = i & 0xFF
        return PyBytes_FromStringAndSize(buffer, 1)
    elif i < 65_536:  # 1 << 16 (2 bytes)
        buffer[0] = 0xFC  # 252
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 3)
    elif i < 16_777_216: # 1 << 24 (3 bytes)
        buffer[0] = 0xFD  # 253
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        buffer[3] = (i >> 16) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 4)
    else: # 1 << 64 (8 bytes)
        buffer[0] = 0xFE  # 254
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        buffer[3] = (i >> 16) & 0xFF
        buffer[4] = (i >> 24) & 0xFF
        buffer[5] = (i >> 32) & 0xFF
        buffer[6] = (i >> 40) & 0xFF
        buffer[7] = (i >> 48) & 0xFF
        buffer[8] = (i >> 56) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 9)

cdef inline Py_ssize_t find_null_term(char* data, Py_ssize_t pos) except -2:
    """Find the next NULL-terminated string in the data `<'int'>`."""
    cdef const char* ptr = data + pos
    loc = strchr(ptr, 0)
    if loc is NULL:
        return -1
    return loc - ptr + pos

# Utils: Pack custom
cdef inline bytes pack_I24B(unsigned int i, unsigned char j):
    """Pack 'I[24bit]B' in little-endian order `<'bytes'>`.
    
    Equivalent to 'struct.pack("<I[int24]B", i, j)'
    """
    cdef char buffer[4]
    buffer[0] = i & 0xFF
    buffer[1] = (i >> 8) & 0xFF
    buffer[2] = (i >> 16) & 0xFF
    buffer[3] = j
    return PyBytes_FromStringAndSize(buffer, 4)

cdef inline bytes pack_IB(unsigned int i, unsigned char j):
    """Pack 'IB' in little-endian order `<'bytes'>`.

    Equivalent to 'struct.pack("<IB", i, j)'
    """
    cdef char buffer[5]
    buffer[0] = i & 0xFF
    buffer[1] = (i >> 8) & 0xFF
    buffer[2] = (i >> 16) & 0xFF
    buffer[3] = (i >> 24) & 0xFF
    buffer[4] = j
    return PyBytes_FromStringAndSize(buffer, 5)

cdef inline bytes pack_IIB23s(unsigned int i, unsigned int j, unsigned char k):
    """Pack 'IIB23s' in little-endian order `<'bytes'>`.

    Equivalent to 'struct.pack("<IIB23s", i, j, k, b"")'
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

# Utils: Pack unsigned integers
cdef inline bytes pack_uint8(unsigned int value):
    """Pack unsigned 8-bit integer in little-endian order `<'bytes'>`."""
    cdef char buffer[1]
    buffer[0] = value & 0xFF
    return PyBytes_FromStringAndSize(buffer, 1)

cdef inline bytes pack_uint16(unsigned int value):
    """Pack unsigned 16-bit integer in little-endian order `<'bytes'>`."""
    cdef char buffer[2]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 2)

cdef inline bytes pack_uint24(unsigned int value):
    """Pack unsigned 24-bit integer in little-endian order `<'bytes'>`."""
    cdef char buffer[3]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 3)

cdef inline bytes pack_uint32(unsigned long long value):
    """Pack unsigned 32-bit integer in little-endian order `<'bytes'>`."""
    cdef char buffer[4]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 4)

cdef inline bytes pack_uint64(unsigned long long value):
    """Pack unsigned 64-bit integer in little-endian order `<'bytes'>`."""
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
cdef inline unsigned char unpack_uint8(char* data, unsigned long long pos):
    """Unpack (read) unsigned 8-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    return <unsigned char> data[pos]

cdef inline unsigned short unpack_uint16(char* data, unsigned long long pos):
    """Unpack (read) unsigned 16-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned short p0 = <unsigned char> data[pos]
        unsigned short p1 = <unsigned char> data[pos + 1]
        unsigned short res = p0 | (p1 << 8)
    return res

cdef inline unsigned int unpack_uint24(char* data, unsigned long long pos):
    """Unpack (read) unsigned 24-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned int p0 = <unsigned char> data[pos]
        unsigned int p1 = <unsigned char> data[pos + 1]
        unsigned int p2 = <unsigned char> data[pos + 2]
        unsigned int res = p0 | (p1 << 8) | (p2 << 16)
    return res

cdef inline unsigned int unpack_uint32(char* data, unsigned long long pos):
    """Unpack (read) unsigned 32-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned int p0 = <unsigned char> data[pos]
        unsigned int p1 = <unsigned char> data[pos + 1]
        unsigned int p2 = <unsigned char> data[pos + 2]
        unsigned int p3 = <unsigned char> data[pos + 3]
        unsigned int res = p0 | (p1 << 8) | (p2 << 16) | (p3 << 24)
    return res

cdef inline unsigned long long unpack_uint64(char* data, unsigned long long pos):
    """Unpack (read) unsigned 64-bit integer from 'data'
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
    """Pack signed 8-bit integer in little-endian order `<'bytes'>`."""
    cdef char buffer[1]
    buffer[0] = value & 0xFF
    return PyBytes_FromStringAndSize(buffer, 1)

cdef inline bytes pack_int16(int value):
    """Pack signed 16-bit integer in little-endian order `<'bytes'>`."""
    cdef char buffer[2]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 2)

cdef inline bytes pack_int24(int value):
    """Pack signed 24-bit integer in little-endian order `<'bytes'>`."""
    cdef char buffer[3]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 3)

cdef inline bytes pack_int32(long long value):
    """Pack signed 32-bit integer in little-endian order `<'bytes'>`."""
    cdef char buffer[4]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 4)

cdef inline bytes pack_int64(long long value):
    """Pack signed 64-bit integer in little-endian order `<'bytes'>`."""
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
cdef inline signed char unpack_int8(char* data, unsigned long long pos):
    """Read (unpack) signed 8-bit integer from 'data' at givent 'pos' `<'int'>`."""
    return <signed char> data[pos]

cdef inline short unpack_int16(char* data, unsigned long long pos):
    """Read (unpack) signed 16-bit integer from 'data' at givent 'pos' `<'int'>`."""
    return <short> unpack_uint16(data, pos)

cdef inline int unpack_int24(char* data, unsigned long long pos):
    """Read (unpack) signed 24-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef int res = <int> unpack_uint24(data, pos)
    return res if res < 0x800000 else res - 0x1000000

cdef inline int unpack_int32(char* data, unsigned long long pos):
    """Read (unpack) signed 32-bit integer from 'data' at givent 'pos' `<'int'>`."""
    return <int> unpack_uint32(data, pos)

cdef inline long long unpack_int64(char* data, unsigned long long pos):
    """Read (unpack) signed 64-bit integer from 'data' at givent 'pos' `<'int'>`."""
    return <long long> unpack_uint64(data, pos)

# Utils: Connection
cdef bytes gen_connect_attrs(list attrs)
cdef str validate_arg_str(object arg, str arg_name, str default)
cdef object validate_arg_uint(object arg, str arg_name, unsigned int min_val, unsigned int max_val)
cdef bytes validate_arg_bytes(object arg, str arg_name, char* encoding, str default)
cdef Charset validate_charset(object charset, object collation, str default_charset)
cdef int validate_autocommit(object auto) except -2
cdef int validate_max_allowed_packet(object max_allowed_packet, int default, int maximum)
cdef str validate_sql_mode(object sql_mode, char* encoding)
cdef object validate_cursor(object cursor, object cursor_class)
cdef object validate_ssl(object ssl)
