# cython: language_level=3
from libc.string cimport strchr
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.bytes cimport PyBytes_GET_SIZE as bytes_len
from cpython.unicode cimport PyUnicode_AsUTF8String
from cpython.unicode cimport PyUnicode_GET_LENGTH as str_len
from cpython.unicode cimport PyUnicode_ReadChar as read_char
from cpython.unicode cimport PyUnicode_Substring as str_substr
from cpython.unicode cimport PyUnicode_AsEncodedString as str_encode
from sqlcycli._auth cimport AuthPlugin
from sqlcycli.transcode cimport escape_item
from sqlcycli._ssl cimport is_ssl, is_ssl_ctx
from sqlcycli.charset cimport Charset, _charsets
from sqlcycli.protocol cimport MysqlPacket, FieldDescriptorPacket
from sqlcycli import errors

# Constant
cdef:
    str DEFAULT_USER, DEFUALT_CHARSET
    int MAX_CONNECT_TIMEOUT
    int DEFALUT_MAX_ALLOWED_PACKET, MAXIMUM_MAX_ALLOWED_PACKET
    bytes DEFAULT_CONNECT_ATTRS
    unsigned int MAX_PACKET_LENGTH
    object INSERT_VALUES_RE, SERVER_VERSION_RE

# Utils
cdef inline object encode_str(object obj, char* encoding):
    """Encode string to bytes using 'encoding' with 
    'surrogateescape' error handling `<'bytes'>`."""
    return str_encode(obj, encoding, "surrogateescape")

cdef inline bytes gen_length_encoded_integer(unsigned long long i):
    """Generate 'Length Coded Integer' for MySQL `<'bytes'>."""
    # https://dev.mysql.com/doc/internals/en/integer.html#packet-Protocol::LengthEncodedInteger
    cdef char output[9]
    if i < 0xFB:  # 251 (1 byte)
        output[0] = <char> i
        return PyBytes_FromStringAndSize(output, 1)
    elif i < 65_536:  # 1 << 16 (3 bytes)
        output[0] = 0xFC  # 252
        output[1] = i & 0xFF  # 255
        output[2] = i >> 8
        return PyBytes_FromStringAndSize(output, 3)
    elif i < 16_777_216: # 1 << 24 (4 bytes)
        output[0] = 0xFD  # 0xFD
        output[1] = i & 0xFF  # 0xFF
        output[2] = (i >> 8) & 0xFF  # 0xFF
        output[3] = i >> 16
        return PyBytes_FromStringAndSize(output, 4)
    else: # 1 << 64 (9 bytes)
        output[0] = 0xFE  # 254
        output[1] = i & 0xFF  # 255
        output[2] = (i >> 8) & 0xFF  # 255
        output[3] = (i >> 16) & 0xFF  # 255
        output[4] = (i >> 24) & 0xFF  # 255
        output[5] = (i >> 32) & 0xFF  # 255
        output[6] = (i >> 40) & 0xFF  # 255
        output[7] = (i >> 48) & 0xFF  # 255
        output[8] = i >> 56
        return PyBytes_FromStringAndSize(output, 9)

cdef inline bytes gen_connect_attrs(list attrs):
    """Generate connection attributes for MySQL `<'bytes'>`."""
    cdef bytearray arr = bytearray()
    cdef bytes attr
    for i in attrs:
        attr = PyUnicode_AsUTF8String(i)
        arr += gen_length_encoded_integer(bytes_len(attr))
        arr += attr
    return bytes(arr)

cdef inline Py_ssize_t find_null_term(char* data, Py_ssize_t pos) except -2:
    """Find the next NULL-terminated string in the data `<'int'>`."""
    cdef const char* ptr = data + pos
    loc = strchr(ptr, 0)
    if loc is NULL:
        return -1
    return loc - ptr + pos

# Validator
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
        % (arg_name, arg, type(arg)) )

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
            % (arg_name, arg, type(arg)) ) from err
    if not min_value <= value <= max_value:
        raise errors.InvalidConnectionArgsError(
            "Invalid '%s' argument: '%s', must be between %d and %d." 
            % (arg_name, value, min_value, max_value) )
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
        % (arg_name, arg, type(arg)) )

cdef inline Charset validate_charset(object charset, object collation):
    """Validate if the 'charset' & 'collation' arguments are valid `<'Charset'>`."""
    cdef:
        str _charset = validate_arg_str(charset, "charset", DEFUALT_CHARSET)
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

cdef inline unsigned int validate_max_allowed_packet(object max_allowed_packet) except -1:
    """Validate if the 'max_allowed_packet' argument is valid `<'int'>`."""
    # Argument is None
    if max_allowed_packet is None:
        return DEFALUT_MAX_ALLOWED_PACKET
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
        if not 0 < value <= MAXIMUM_MAX_ALLOWED_PACKET:
            raise errors.InvalidConnectionArgsError(
                "Invalid 'max_allowed_packet' argument: "
                "'%s', must be between 1 and %s."
                % (value, MAXIMUM_MAX_ALLOWED_PACKET)
            )
        return value
    # Argument is not a string
    if not isinstance(max_allowed_packet, str):
        raise errors.InvalidConnectionArgsError(
            "Invalid 'max_allowed_packet' argument: %r, "
            "expects <'str/int'> instead of %s." 
            % (max_allowed_packet, type(max_allowed_packet)) )
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
            % (max_allowed_packet, type(max_allowed_packet)) )
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
    if not 0 < value <= MAXIMUM_MAX_ALLOWED_PACKET:
        raise errors.InvalidConnectionArgsError(
            "Invalid 'max_allowed_packet' argument: "
            "'%s' (%s), must be between 1 and %s."
            % (max_allowed_packet, value, MAXIMUM_MAX_ALLOWED_PACKET)
        )
    return value

cdef inline str validate_sql_mode(object sql_mode):
    """Validate if the 'sql_mode' argument is valid `<'str'>`."""
    # Argument is None
    if sql_mode is None:
        return None
    # Argument is a string
    if isinstance(sql_mode, str):
        return escape_item(sql_mode) if str_len(sql_mode) > 0 else None
    # Invalid data type
    raise errors.InvalidConnectionArgsError(
        "Invalid 'sql_mode' argument: %r, expects <'str'> instead of %s." 
        % (sql_mode, type(sql_mode)) )

cdef inline object validate_cursor(object cursor):
    """Validate if the 'cursor' argument is valid `<'type[Cursor]'>`."""
    if cursor is None:
        return Cursor
    if type(cursor) is not type or not issubclass(cursor, Cursor):
        raise errors.InvalidConnectionArgsError(
            "Invalid 'cursor' argument: %r, "
            "must be a class type of <'mysqlcycle.Cursor'>." % cursor
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
        "instead of %s." % (ssl, type(ssl)) )

# Pack
cdef inline bytes pack_I24B(unsigned int i, unsigned j):
    """Manually pack 'struct.pack("<I[int24]B", i, j)' `<'bytes'>`"""
    cdef:
        unsigned char v0 = i & 0xFF
        unsigned char v1 = (i >> 8) & 0xFF
        unsigned char v2 = (i >> 16) & 0xFF
        unsigned char v3 = j & 0xFF
        char data[4]
    data[0], data[1], data[2], data[3] = v0, v1, v2, v3
    return PyBytes_FromStringAndSize(data, 4)

cdef inline bytes pack_IB(unsigned int i, unsigned j):
    """Manually pack 'struct.pack("<I[int24]B", i, j)' `<'bytes'>`"""
    cdef:
        unsigned char v0 = i & 0xFF
        unsigned char v1 = (i >> 8) & 0xFF
        unsigned char v2 = (i >> 16) & 0xFF
        unsigned char v3 = (i >> 24) & 0xFF
        unsigned char v4 = j & 0xFF
        char data[5]
    data[0], data[1], data[2], data[3], data[4] = v0, v1, v2, v3, v4
    return PyBytes_FromStringAndSize(data, 5)

cdef inline bytes pack_IIB23s(unsigned int i, unsigned int j, unsigned int k):
    """Manually pack 'struct.pack("<IIB23s", i, j, k, b"")' `<'bytes'>`"""
    cdef:
        unsigned char v0 = i & 0xFF
        unsigned char v1 = (i >> 8) & 0xFF
        unsigned char v2 = (i >> 16) & 0xFF
        unsigned char v3 = (i >> 24) & 0xFF
        unsigned char v4 = j & 0xFF
        unsigned char v5 = (j >> 8) & 0xFF
        unsigned char v6 = (j >> 16) & 0xFF
        unsigned char v7 = (j >> 24) & 0xFF
        unsigned char v8 = k & 0xFF
        char data[32]
    data[0], data[1], data[2], data[3] = v0, v1, v2, v3
    data[4], data[5], data[6], data[7] = v4, v5, v6, v7
    data[8] = v8
    for i in range(23):
        data[i + 9] = 0
    return PyBytes_FromStringAndSize(data, 32)

# Result
cdef class MysqlResult:
    cdef:
        # Connection
        BaseConnection _conn
        bint _use_decimal, _decode_json
        # Packet data
        unsigned long long affected_rows, insert_id
        int server_status
        unsigned int warning_count
        bint has_next
        bytes message
        # Field data
        unsigned long long field_count
        tuple fields, rows
        # Unbuffered
        bint unbuffered_active
    # Methods
    cpdef bint read(self) except -1
    cpdef bint init_unbuffered_query(self) except -1
    cdef inline bint _read_ok_packet(self, MysqlPacket pkt) except -1
    cdef inline bint _read_load_local_packet(self, MysqlPacket pkt) except -1
    cdef inline bint _read_eof_packet(self, MysqlPacket pkt) except -1
    cdef inline bint _read_result_packet(self, MysqlPacket pkt) except -1
    cdef inline bint _read_result_packet_fields(self, MysqlPacket pkt) except -1
    cdef inline tuple _read_result_packet_row(self, MysqlPacket pkt)
    cdef inline tuple _read_result_packet_row_unbuffered(self)
    cdef inline bint _drain_result_packet_unbuffered(self) except -1

# Cursor
cdef class Cursor:
    cdef:
        bint _unbuffered
        BaseConnection _conn
        char* _encoding_c
        bytes _executed_sql
        MysqlResult _result
        unsigned long long _field_count
        tuple _fields, _rows
        unsigned long long _affected_rows, _row_idx, _row_size
        unsigned long long _insert_id
        unsigned int _warning_count
    # Init
    cdef inline bint _init_setup(self, BaseConnection conn, bint unbuffered) except -1
    # Write
    cpdef unsigned long long execute(self, str sql, object args=?, bint force_many=?, bint itemize=?)
    cpdef tuple callproc(self, str procname, object args)
    cpdef str mogrify(self, str sql, object args=?, bint itemize=?)
    cpdef object escape_args(self, object args, bint itemize=?)
    cpdef bytes encode_sql(self, str sql)
    cdef inline unsigned long long _query_str(self, str sql)
    cdef inline unsigned long long _query_bytes(self, bytes sql)
    cdef inline str _format(self, str sql, object args)
    # Read
    cdef inline tuple _fetchone_row(self)
    cdef inline dict _fetchone_dict(self)
    cdef inline object _fetchone_df(self)
    cdef inline tuple _fetch_row(self, unsigned long long size=?)
    cdef inline tuple _fetch_dict(self, unsigned long long size=?)
    cdef inline object _fetch_df(self, unsigned long long size=?)
    cpdef bint scroll(self, long long value, object mode=?) except -1
    cpdef bint next_set(self) except -1
    cdef inline tuple _next_row(self)
    cpdef tuple columns(self)
    cdef inline bint _read_result(self) except -1
    cdef inline bint _clear_result(self) except -1
    cdef inline unsigned long long _get_row_size(self)
    cdef inline bint _has_more_rows(self) except -1
    cdef inline bint _verify_executed(self) except -1
    cdef inline bint _verify_connected(self) except -1
    # Close
    cpdef bint close(self) except -1
    cpdef bint closed(self) except -1

# Connection
cdef class CursorManager:
    cdef:
        BaseConnection _conn
        object _cur_type
        Cursor _cur
    cdef inline bint _close(self) except -1

cdef class TransactionManager(CursorManager):
    cdef inline bint _close_connection(self) except -1

cdef class BaseConnection:
    cdef:
        # Basic
        str _host
        object _port
        bytes _user, _password, _database
        # Charset
        str _charset, _collation
        unsigned int _charset_id
        bytes _encoding
        char* _encoding_c
        # Timeouts
        object _connect_timeout
        object _read_timeout
        object _write_timeout
        object _wait_timeout
        # Client
        str _bind_address, _unix_socket
        int _autocommit_mode
        bint _local_infile
        unsigned int _max_allowed_packet
        str _sql_mode, _init_command
        object _cursor
        unsigned int _client_flag
        bytes _connect_attrs
        # SSL
        object _ssl_ctx
        # Auth
        AuthPlugin _auth_plugin
        bytes _server_public_key
        # Decode
        bint _use_decimal, _decode_json
        # Internal
        # . server
        int _server_protocol_version
        str _server_info
        tuple _server_version
        int _server_version_major
        str _server_vendor
        long long _server_thred_id
        bytes _server_salt
        int _server_status
        long long _server_capabilities
        str _server_auth_plugin_name
        # . client
        double _last_used_time
        bint _closed, _secure
        str _host_info, _close_reason
        # . query
        MysqlResult _result
        unsigned int _next_seq_id
        # . transport
        object _reader, _writer
    # Init
    cdef inline bint _init_charset(self, Charset charset) except -1
    cdef inline bint _init_client_flag(self, unsigned int client_flag) except -1
    cdef inline bint _init_connect_attrs(self, object program_name) except -1
    cdef inline bint _init_internal(self) except -1
    # Cursor
    cpdef CursorManager cursor(self, object cursor=?)
    cpdef TransactionManager transaction(self, object cursor=?)
    cdef inline bint _set_use_time(self) except -1
    # Query
    cpdef unsigned long long query(self, str sql, bint unbuffered=?)
    cpdef bint begin(self) except -1
    cpdef bint commit(self) except -1
    cpdef bint rollback(self) except -1
    cpdef bint set_charset(self, object charset, object collation=?) except -1
    cpdef bint set_read_timeout(self, object timeout) except -1
    cpdef unsigned int get_read_timeout(self)
    cpdef bint set_write_timeout(self, object timeout) except -1
    cpdef unsigned int get_write_timeout(self)
    cpdef bint set_wait_timeout(self, object timeout) except -1
    cpdef unsigned int get_wait_timeout(self)
    cdef inline bint _set_timeout(self, str name, object timeout) except -1
    cdef inline unsigned int _get_timeout(self, str name, bint session)
    cpdef bint get_autocommit(self) except -1
    cpdef bint set_autocommit(self, bint auto) except -1
    cpdef tuple show_warnings(self)
    cpdef bint select_database(self, str db) except -1
    cpdef unsigned long long get_affected_rows(self)
    cpdef unsigned long long get_insert_id(self)
    cpdef bint get_transaction_status(self) except -1
    cpdef tuple get_server_version(self)
    cpdef str get_server_vendor(self)
    cpdef bint set_use_decimal(self, bint value) except -1
    cpdef bint set_decode_json(self, bint value) except -1
    cpdef object escape_args(self, object args, bint itemize=?)
    cpdef bytes encode_sql(self, str sql)
    # Connect / Close
    cpdef bint connect(self) except -1
    cdef inline bint _connect(self) except -1
    cpdef bint close(self) except -1
    cpdef bint force_close(self) except -1
    cdef inline bint _close_with_reason(self, str reason) except -1
    cpdef bint closed(self) except -1
    cpdef bint kill(self, int thread_id) except -1
    cpdef bint ping(self, bint reconnect=?) except -1
    cdef inline bint _setup_server_information(self) except -1
    cdef inline bint _request_authentication(self) except -1
    cdef inline MysqlPacket _process_authentication(self, MysqlPacket auth_pkt)
    cdef inline MysqlPacket _process_auth_caching_sha2(self, MysqlPacket pkt)
    cdef inline MysqlPacket _process_auth_sha256(self, MysqlPacket pkt)
    cdef inline MysqlPacket _process_auth_send_data(self, bytes data)
    cdef inline bint _verify_connected(self) except -1
    # Write
    cdef inline bint _execute_command(self, unsigned int command, bytes sql) except -1
    cdef inline bint _write_packet(self, bytes payload) except -1
    cdef inline bint _write_bytes(self, bytes data) except -1
    # Read
    cpdef unsigned long long next_result(self, bint unbuffered=?)
    cdef inline MysqlPacket _read_ok_packet(self)
    cdef inline unsigned long long _read_query_result(self, bint unbuffered)
    cdef inline MysqlPacket _read_packet(self)
    cdef inline FieldDescriptorPacket _read_field_descriptor_packet(self)
    cdef inline bytes _read_packet_buffer(self)
    cdef inline bytes _read_bytes(self, unsigned int size)
