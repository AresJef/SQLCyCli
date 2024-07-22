# cython: language_level=3
from libc.string cimport strchr
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.limits cimport INT_MAX, UINT_MAX, LLONG_MAX, ULLONG_MAX

# Constants
cdef:
    unsigned int NULL_COLUMN, UNSIGNED_CHAR_COLUMN, UNSIGNED_SHORT_COLUMN
    unsigned int UNSIGNED_INT24_COLUMN, UNSIGNED_INT64_COLUMN

# Utils
cdef inline Py_ssize_t find_null_term(char* data, Py_ssize_t pos) except -2:
    """Find the next NULL-terminated string in the data `<'int'>`."""
    cdef const char* ptr = data + pos
    loc = strchr(ptr, 0)
    if loc is NULL:
        return -1
    return loc - ptr + pos

# Pack unsigned integer
cdef inline bytes pack_uint8(unsigned int value):
    """Pack unsigned 8-bit integer 'value' into 1-bytes `<'bytes'>`."""
    cdef:
        unsigned char v0 = value & 0xFF
        char data[1]
    data[0] = v0
    return PyBytes_FromStringAndSize(data, 1)

cdef inline bytes pack_uint16(unsigned int value):
    """Pack unsigned 16-bit integer 'value' into 2-bytes `<'bytes'>`."""
    cdef: 
        unsigned char v0 = value & 0xFF
        unsigned char v1 = (value >> 8) & 0xFF
        char data[2]
    data[0], data[1] = v0, v1
    return PyBytes_FromStringAndSize(data, 2)

cdef inline bytes pack_uint24(unsigned int value):
    """Pack unsigned 24-bit integer 'value' into 3-bytes `<'bytes'>`."""
    cdef: 
        unsigned char v0 = value & 0xFF
        unsigned char v1 = (value >> 8) & 0xFF
        unsigned char v2 = (value >> 16) & 0xFF
        char data[3]
    data[0], data[1], data[2] = v0, v1, v2
    return PyBytes_FromStringAndSize(data, 3)

cdef inline bytes pack_uint32(unsigned long long value):
    """Pack unsigned 32-bit integer 'value' into 4-bytes `<'bytes'>`."""
    cdef:
        unsigned char v0 = value & 0xFF
        unsigned char v1 = (value >> 8) & 0xFF
        unsigned char v2 = (value >> 16) & 0xFF
        unsigned char v3 = (value >> 24) & 0xFF
        char data[4]
    data[0], data[1], data[2], data[3] = v0, v1, v2, v3
    return PyBytes_FromStringAndSize(data, 4)

cdef inline bytes pack_uint64(unsigned long long value):
    """Pack unsigned 64-bit integer 'value' into 8-bytes `<'bytes'>`."""
    cdef:
        unsigned char v0 = value & 0xFF
        unsigned char v1 = (value >> 8) & 0xFF
        unsigned char v2 = (value >> 16) & 0xFF
        unsigned char v3 = (value >> 24) & 0xFF
        unsigned char v4 = (value >> 32) & 0xFF
        unsigned char v5 = (value >> 40) & 0xFF
        unsigned char v6 = (value >> 48) & 0xFF
        unsigned char v7 = (value >> 56) & 0xFF
        char data[8]
    data[0], data[1], data[2], data[3] = v0, v1, v2, v3
    data[4], data[5], data[6], data[7] = v4, v5, v6, v7
    return PyBytes_FromStringAndSize(data, 8)

# Unpack unsigned integer
cdef inline unsigned char unpack_uint8(char* data, unsigned long long pos):
    """Read (unpack) unsigned 8-bit integer from 'data' at givent 'pos' `<'int'>`."""
    return <unsigned char> data[pos]

cdef inline unsigned int unpack_uint16(char* data, unsigned long long pos):
    """Read (unpack) unsigned 16-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef: 
        unsigned int v0 = <unsigned char> data[pos]
        unsigned int v1 = <unsigned char> data[pos + 1]
    return v0 | (v1 << 8)

cdef inline unsigned int unpack_uint24(char* data, unsigned long long pos):
    """Read (unpack) unsigned 24-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef: 
        unsigned int v0 = <unsigned char> data[pos]
        unsigned int v1 = <unsigned char> data[pos + 1] 
        unsigned int v2 = <unsigned char> data[pos + 2]
    return v0 | (v1 << 8) | (v2 << 16)

cdef inline unsigned long long unpack_uint32(char* data, unsigned long long pos):
    """Read (unpack) unsigned 32-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef: 
        unsigned long long v0 = <unsigned char> data[pos]
        unsigned long long v1 = <unsigned char> data[pos + 1] 
        unsigned long long v2 = <unsigned char> data[pos + 2] 
        unsigned long long v3 = <unsigned char> data[pos + 3] 
    return v0 | (v1 << 8) | (v2 << 16) | (v3 << 24)

cdef inline unsigned long long unpack_uint64(char* data, unsigned long long pos):
    """Read (unpack) unsigned 64-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef: 
        unsigned long long v0 = <unsigned char> data[pos]
        unsigned long long v1 = <unsigned char> data[pos + 1] 
        unsigned long long v2 = <unsigned char> data[pos + 2] 
        unsigned long long v3 = <unsigned char> data[pos + 3] 
        unsigned long long v4 = <unsigned char> data[pos + 4] 
        unsigned long long v5 = <unsigned char> data[pos + 5] 
        unsigned long long v6 = <unsigned char> data[pos + 6] 
        unsigned long long v7 = <unsigned char> data[pos + 7]
    return v0 | (v1 << 8) | (v2 << 16) | (v3 << 24) | (v4 << 32) | (v5 << 40) | (v6 << 48) | (v7 << 56)

# Pack signed integer
cdef inline bytes pack_int8(int value):
    """Pack signed 8-bit integer 'value' into 1-bytes `<'bytes'>`."""
    cdef:
        signed char v0 = value & 0xFF
        char data[1]
    data[0] = v0
    return PyBytes_FromStringAndSize(data, 1)

cdef inline bytes pack_int16(int value):
    """Pack signed 16-bit integer 'value' into 2-bytes `<'bytes'>`."""
    cdef: 
        signed char v0 = value & 0xFF
        signed char v1 = (value >> 8) & 0xFF
        char data[2]
    data[0], data[1] = v0, v1
    return PyBytes_FromStringAndSize(data, 2)

cdef inline bytes pack_int24(int value):
    """Pack signed 24-bit integer 'value' into 3-bytes `<'bytes'>`."""
    cdef: 
        signed char v0 = value & 0xFF
        signed char v1 = (value >> 8) & 0xFF
        signed char v2 = (value >> 16) & 0xFF
        char data[3]
    data[0], data[1], data[2] = v0, v1, v2
    return PyBytes_FromStringAndSize(data, 3)

cdef inline bytes pack_int32(long long value):
    """Pack signed 32-bit integer 'value' into 4-bytes `<'bytes'>`."""
    cdef:
        signed char v0 = value & 0xFF
        signed char v1 = (value >> 8) & 0xFF
        signed char v2 = (value >> 16) & 0xFF
        signed char v3 = (value >> 24) & 0xFF
        char data[4]
    data[0], data[1], data[2], data[3] = v0, v1, v2, v3
    return PyBytes_FromStringAndSize(data, 4)

cdef inline bytes pack_int64(long long value):
    """Pack signed 64-bit integer 'value' into 8-bytes `<'bytes'>`."""
    cdef:
        signed char v0 = value & 0xFF
        signed char v1 = (value >> 8) & 0xFF
        signed char v2 = (value >> 16) & 0xFF
        signed char v3 = (value >> 24) & 0xFF
        signed char v4 = (value >> 32) & 0xFF
        signed char v5 = (value >> 40) & 0xFF
        signed char v6 = (value >> 48) & 0xFF
        signed char v7 = (value >> 56) & 0xFF
        char data[8]
    data[0], data[1], data[2], data[3] = v0, v1, v2, v3
    data[4], data[5], data[6], data[7] = v4, v5, v6, v7
    return PyBytes_FromStringAndSize(data, 8)

# Unpack signed integer
cdef inline signed char unpack_int8(char* data, unsigned long long pos):
    """Read (unpack) signed 8-bit integer from 'data' at givent 'pos' `<'int'>`."""
    return <signed char> data[pos]

cdef inline int unpack_int16(char* data, unsigned long long pos):
    """Read (unpack) signed 16-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef: 
        unsigned int v0 = <unsigned char> data[pos]
        unsigned int v1 = <unsigned char> data[pos + 1]
        int res = v0 | (v1 << 8)
    return res if res < 0x8000 else res - 0x10000

cdef inline int unpack_int24(char* data, unsigned long long pos):
    """Read (unpack) signed 24-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef: 
        unsigned int v0 = <unsigned char> data[pos]
        unsigned int v1 = <unsigned char> data[pos + 1] 
        unsigned int v2 = <unsigned char> data[pos + 2]
        int res = v0 | (v1 << 8) | (v2 << 16)
    return res if res < 0x800000 else res - 0x1000000

cdef inline long long unpack_int32(char* data, unsigned long long pos):
    """Read (unpack) signed 32-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef: 
        unsigned long long v0 = <unsigned char> data[pos]
        unsigned long long v1 = <unsigned char> data[pos + 1] 
        unsigned long long v2 = <unsigned char> data[pos + 2] 
        unsigned long long v3 = <unsigned char> data[pos + 3] 
        long long res = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24)
    return res - UINT_MAX - 1 if res > INT_MAX else res

cdef inline long long unpack_int64(char* data, unsigned long long pos):
    """Read (unpack) signed 64-bit integer from 'data' at givent 'pos' `<'int'>`."""
    cdef: 
        unsigned long long v0 = <unsigned char> data[pos]
        unsigned long long v1 = <unsigned char> data[pos + 1] 
        unsigned long long v2 = <unsigned char> data[pos + 2] 
        unsigned long long v3 = <unsigned char> data[pos + 3] 
        unsigned long long v4 = <unsigned char> data[pos + 4] 
        unsigned long long v5 = <unsigned char> data[pos + 5] 
        unsigned long long v6 = <unsigned char> data[pos + 6] 
        unsigned long long v7 = <unsigned char> data[pos + 7]
        long long res = (
            v0 | (v1 << 8) | (v2 << 16) | (v3 << 24) | (v4 << 32) 
            | (v5 << 40) | (v6 << 48) | (v7 << 56) )
    return res - ULLONG_MAX - 1 if res > LLONG_MAX else res

# MySQL Packet
cdef class MysqlPacket:
    cdef:
        # . raw data
        bytes _data
        char* _data_c
        char* _encoding
        unsigned long long _size, _pos
        # . packet data
        unsigned long long _affected_rows, _insert_id
        int _server_status
        unsigned int _warning_count
        bint _has_next
        bytes _message, _filename
        bytes _plugin_name, _salt
    # Read Data
    cdef inline bytes read_all_data(self)
    cdef inline bytes read(self, unsigned long long size)
    cdef inline bytes read_remains(self)
    cdef inline unsigned long long read_length_encoded_integer(self)
    cdef inline bytes read_length_encoded_string(self)
    cdef inline unsigned int _read_uint8(self)
    cdef inline unsigned int _read_uint16(self)
    cdef inline unsigned int _read_uint24(self)
    cdef inline unsigned long long _read_uint32(self)
    cdef inline unsigned long long _read_uint64(self)
    # Read Packet
    cdef inline bint is_ok_packet(self) except -1
    cpdef bint read_ok_packet(self) except -1
    cdef inline bint is_load_local_packet(self) except -1
    cpdef bint read_load_local_packet(self) except -1
    cdef inline bint is_eof_packet(self) except -1
    cpdef bint read_eof_packet(self) except -1
    cdef inline bint is_auth_switch_request(self) except -1
    cpdef bint read_auth_switch_request(self) except -1
    cdef inline bint is_extra_auth_data(self) except -1
    cdef inline bint is_resultset_packet(self) except -1
    cdef inline bint is_error_packet(self) except -1
    # Curosr
    cdef inline bint advance(self, unsigned long long length) except -1
    cdef inline bint rewind(self, unsigned long long position) except -1
    # Error
    cpdef bint check_error(self) except -1
    cpdef bint raise_for_error(self) except -1

cdef class FieldDescriptorPacket(MysqlPacket):
    cdef:
        # . packet data
        bytes _catalog
        str _db, _table, _table_org, _column, _column_org
        unsigned int _charsetnr
        unsigned long long _length
        unsigned int _type_code, _flags, _scale
        bint _is_binary
    # Read Packet
    cpdef tuple description(self)
    cdef inline unsigned long long _get_column_length(self)
