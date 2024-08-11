# cython: language_level=3
from libc.string cimport strchr
from libc.limits cimport INT_MAX, UINT_MAX, LLONG_MAX, ULLONG_MAX
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.bytes cimport PyBytes_GET_SIZE as bytes_len
from cpython.unicode cimport PyUnicode_AsUTF8String
from cpython.unicode cimport PyUnicode_GET_LENGTH as str_len
from cpython.unicode cimport PyUnicode_ReadChar as read_char
from cpython.unicode cimport PyUnicode_Substring as str_substr
from cpython.unicode cimport PyUnicode_AsEncodedString as str_encode
from sqlcycli.transcode cimport escape
from sqlcycli._ssl cimport is_ssl, is_ssl_ctx
from sqlcycli.charset cimport Charset, _charsets
from sqlcycli import errors

# Utils: c-string 
cdef inline object encode_str(object obj, char* encoding):
    """Encode string to bytes using 'encoding' with 
    'surrogateescape' error handling `<'bytes'>`."""
    return str_encode(obj, encoding, "surrogateescape")

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
cdef inline bytes gen_connect_attrs(list attrs):
    """Generate connection attributes for MySQL `<'bytes'>`."""
    cdef bytearray arr = bytearray()
    cdef bytes attr
    for i in attrs:
        attr = PyUnicode_AsUTF8String(i)
        arr += gen_length_encoded_integer(bytes_len(attr))
        arr += attr
    return bytes(arr)

# Utils: Connection validator
cdef inline str validate_arg_str(object arg, str arg_name, str default):
    """Validate if the string related argument is valid `<'str'>`."""
    # Argument is None
    if arg is None:
        return default
    # Argument is a string
    if isinstance(arg, str):
        return arg if str_len(arg) > 0 else default
    # Invalid data type
    raise errors.InvalidConnectionArgsError(
        "Invalid '%s' argument: %r, expects <'str'> instead of %s."
        % (arg_name, arg, type(arg))
    )

cdef inline object validate_arg_uint(object arg, str arg_name, unsigned int min_value, unsigned int max_value):
    """Validate if the unsigned integer related argument is valid `<'unsigned int'>`."""
    # Argument is None
    if arg is None:
        return None
    # Argument is an integer
    try:
        value = int(arg)
    except Exception as err:
        raise errors.InvalidConnectionArgsError(
            "Invalid '%s' argument: %r, expects <'int'> instead of %s."
            % (arg_name, arg, type(arg))
        ) from err
    if not min_value <= value <= max_value:
        raise errors.InvalidConnectionArgsError(
            "Invalid '%s' argument: '%s', must be between %d and %d."
            % (arg_name, value, min_value, max_value)
        )
    return value

cdef inline bytes validate_arg_bytes(object arg, str arg_name, char* encoding, str default):
    """Validate if the bytes related argument is valid `<'bytes'>`."""
    # Argument is None
    if arg is None:
        if default is not None:
            return str_encode(default, encoding, NULL)
        return None
    # Argument is a string
    if isinstance(arg, str):
        if str_len(arg) > 0:
            return str_encode(arg, encoding, NULL)
        if default is not None:
            return str_encode(default, encoding, NULL)
        return None
    # Argument is bytes
    if isinstance(arg, bytes):
        if bytes_len(arg) > 0:
            return arg
        if default is not None:
            return str_encode(default, encoding, NULL)
        return None
    # Invalid data type
    raise errors.InvalidConnectionArgsError(
        "Invalid '%s' argument: %r, expects <'str/bytes'> instead of %s."
        % (arg_name, arg, type(arg))
    )

cdef inline Charset validate_charset(object charset, object collation, str default_charset):
    """Validate if the 'charset' & 'collation' arguments are valid `<'Charset'>`."""
    cdef:
        str _charset = validate_arg_str(charset, "charset", default_charset)
        str _collation = validate_arg_str(collation, "collation", None)
    if _collation is None:
        return _charsets.by_name(_charset)
    else:
        return _charsets.by_name_n_collation(_charset, _collation)

cdef inline int validate_autocommit(object autocommit) except -2:
    """Validate if the 'autocommit' argument is valid `<'int'>`.
    Returns -1 if use server default, or 1/0 as autocommit value.
    """
    if autocommit is None:
        return -1
    return 1 if bool(autocommit) else 0

cdef inline unsigned int validate_max_allowed_packet(object max_allowed_packet, int default, int maximum):
    """Validate if the 'max_allowed_packet' argument is valid `<'int'>`."""
    # Argument is None
    if max_allowed_packet is None:
        return default
    # Argument is an integer
    cdef long long value
    if isinstance(max_allowed_packet, int):
        try:
            value = max_allowed_packet
        except Exception as err:
            raise errors.InvalidConnectionArgsError(
                "Invalid 'max_allowed_packet' argument: %s, "
                "expects <'str/int'> instead of %s."
                % (max_allowed_packet, type(max_allowed_packet))
            ) from err
        # Validate value
        if not 0 < value <= maximum:
            raise errors.InvalidConnectionArgsError(
                "Invalid 'max_allowed_packet' argument: "
                "'%s', must be between 1 and %s."
                % (value, maximum)
            )
        return value
    # Argument is not a string
    if not isinstance(max_allowed_packet, str):
        raise errors.InvalidConnectionArgsError(
            "Invalid 'max_allowed_packet' argument: %r, "
            "expects <'str/int'> instead of %s."
            % (max_allowed_packet, type(max_allowed_packet))
        )
    # Parse value from string
    cdef Py_ssize_t length = str_len(max_allowed_packet)
    cdef unsigned int mult
    cdef Py_UCS4 ch
    ch = read_char(max_allowed_packet, length - 1)
    # . skip M/K/G[B] suffix
    if ch in ("B", "b"):
        length -= 1
        ch = read_char(max_allowed_packet, length - 1)
    # . K/KiB suffix
    if ch in ("K", "k"):
        length -= 1
        mult = 1_024
    # . M/MiB suffix
    elif ch in ("M", "m"):
        length -= 1
        mult = 1_048_576
    # . G/GiB suffix
    elif ch in ("G", "g"):
        length -= 1
        mult = 1_073_741_824
    # . no suffix
    else:
        mult = 1
    # . parse to integer
    if length < 1:
        raise errors.InvalidConnectionArgsError(
            "Invalid 'max_allowed_packet' argument: '%s' %s."
            % (max_allowed_packet, type(max_allowed_packet))
        )
    # Parse integer
    try:
        value = int(str_substr(max_allowed_packet, 0, length))
    except Exception as err:
        raise errors.InvalidConnectionArgsError(
            "Invalid 'max_allowed_packet' argument: '%s' %s."
            % (max_allowed_packet, type(max_allowed_packet))
        ) from err
    value *= mult
    # Validate value
    if not 0 < value <= maximum:
        raise errors.InvalidConnectionArgsError(
            "Invalid 'max_allowed_packet' argument: "
            "'%s' (%s), must be between 1 and %s."
            % (max_allowed_packet, value, maximum)
        )
    return value

cdef inline str validate_sql_mode(object sql_mode, char* encoding):
    """Validate if the 'sql_mode' argument is valid `<'str'>`."""
    # Argument is None
    if sql_mode is None:
        return None
    # Argument is a string
    if isinstance(sql_mode, str):
        if str_len(sql_mode) > 0:
            return escape(sql_mode, encoding, False, False)
        return None
    # Invalid data type
    raise errors.InvalidConnectionArgsError(
        "Invalid 'sql_mode' argument: %r, expects <'str'> instead of %s." 
        % (sql_mode, type(sql_mode))
    )

cdef inline object validate_cursor(object cursor, object cursor_class):
    """Validate if the 'cursor' argument is valid `<'type[Cursor]'>`."""
    if cursor is None:
        return cursor_class
    if type(cursor) is not type or not issubclass(cursor, cursor_class):
        raise errors.InvalidConnectionArgsError(
            "Invalid 'cursor' %r, must be type of %r." % (cursor, cursor_class)
        )
    return cursor

cdef inline object validate_ssl(object ssl):
    """Validate if the 'ssl' argument is valid `<'ssl.SSLContext'>`."""
    if ssl is None:
        return None
    if is_ssl(ssl):
        return ssl.context if ssl else None
    if is_ssl_ctx(ssl):
        return ssl
    raise errors.InvalidConnectionArgsError(
        "Invalid 'ssl' argument: %r, expects <'SSL/SSLContext'> "
        "instead of %s." % (ssl, type(ssl))
    )
