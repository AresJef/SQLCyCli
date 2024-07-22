# cython: language_level=3
from sqlcycli.charset cimport Charset
from sqlcycli._auth cimport AuthPlugin
from sqlcycli.protocol cimport MysqlPacket
from sqlcycli import errors

# Validator
cdef inline object validate_cursor(object cursor):
    """Validate if the 'cursor' argument is valid `<'type[Cursor]'>`."""
    if cursor is None:
        return Cursor
    if type(cursor) is not type or not issubclass(cursor, Cursor):
        raise errors.InvalidConnectionArgsError(
            "Invalid 'cursor' argument: %r, "
            "must be a class type of <'aio.Cursor'>." % cursor
        )
    return cursor

# Result
cdef class MysqlResult:
    cdef:
        # Connection
        BaseConnection _conn
        object _local_file
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
    cdef inline bint _read_ok_packet(self, MysqlPacket pkt) except -1
    cdef inline bint _read_eof_packet(self, MysqlPacket pkt) except -1
    cdef inline tuple _read_result_packet_row(self, MysqlPacket pkt)

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
    cpdef str mogrify(self, str sql, object args=?, bint itemize=?)
    cpdef object escape_args(self, object args, bint itemize=?)
    cpdef bytes encode_sql(self, str sql)
    cdef inline str _format(self, str sql, object args)
    # Read
    cpdef tuple columns(self)
    cdef inline bint _read_result(self) except -1
    cdef inline bint _clear_result(self) except -1
    cdef inline unsigned long long _get_row_size(self)
    cdef inline bint _has_more_rows(self) except -1
    cdef inline bint _verify_executed(self) except -1
    cdef inline bint _verify_connected(self) except -1
    # Close
    cpdef bint force_close(self) except -1
    cpdef bint closed(self) except -1

# Connection
cdef class CursorManager:
    cdef:
        BaseConnection _conn
        object _cur_type
        Cursor _cur

cdef class TransactionManager(CursorManager):
    pass
    
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
        # . loop
        object _loop
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
    cpdef bint get_autocommit(self) except -1
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
    cpdef bint force_close(self) except -1
    cdef inline bint _close_with_reason(self, str reason) except -1
    cpdef bint closed(self) except -1
    cdef inline bint _verify_connected(self) except -1
    # Write
    cdef inline bint _write_packet(self, bytes payload) except -1
    cdef inline bint _write_bytes(self, bytes data) except -1
