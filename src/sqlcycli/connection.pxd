# cython: language_level=3
from sqlcycli.charset cimport Charset
from sqlcycli._auth cimport AuthPlugin
from sqlcycli.protocol cimport MysqlPacket, FieldDescriptorPacket

# Result
cdef class MysqlResult:
    cdef:
        # Connection
        BaseConnection _conn
        # Packet data
        unsigned long long _affected_rows
        unsigned long long _insert_id
        int _server_status
        unsigned int _warning_count
        bint _has_next
        bytes _message
        # Field data
        unsigned long long _field_count
        tuple _fields
        tuple _rows
        # Unbuffered
        bint _unbuffered_active
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
        unsigned long long _arraysize
        MysqlResult _result
        unsigned long long _field_count
        tuple _fields
        tuple _rows
        tuple _columns
        unsigned long long _affected_rows
        unsigned long long _row_idx
        unsigned long long _row_size
        unsigned long long _insert_id
        unsigned int _warning_count
    # Init
    cdef inline bint _setup(self, BaseConnection conn, bint unbuffered) except -1
    # Write
    cpdef unsigned long long execute(self, str sql, object args=?, bint many=?, bint itemize=?)
    cpdef unsigned long long executemany(self, str sql, object args=?)
    cpdef object callproc(self, str procname, object args)
    cpdef str mogrify(self, str sql, object args=?, bint many=?, bint itemize=?)
    cdef inline unsigned long long _query_str(self, str sql)
    cdef inline unsigned long long _query_bytes(self, bytes sql)
    cdef inline str _format(self, str sql, object args)
    # Read
    cdef inline tuple _fetchone_tuple(self)
    cdef inline dict _fetchone_dict(self)
    cdef inline object _fetchone_df(self)
    cdef inline tuple _fetchmany_tuple(self, unsigned long long size)
    cdef inline tuple _fetchmany_dict(self, unsigned long long size)
    cdef inline object _fetchmany_df(self, unsigned long long size)
    cdef inline tuple _fetchall_tuple(self)
    cdef inline tuple _fetchall_dict(self)
    cdef inline object _fetchall_df(self)
    cdef inline dict _convert_row_to_dict(self, tuple row, tuple cols, unsigned long long field_count)
    cpdef bint scroll(self, long long value, str mode=?) except -1
    cpdef bint nextset(self) except -1
    cdef inline tuple _next_row_unbuffered(self)
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

cdef class DictCursor(Cursor):
    pass

cdef class DfCursor(Cursor):
    pass

cdef class SSCursor(Cursor):
    pass

cdef class SSDictCursor(DictCursor):
    pass

cdef class SSDfCursor(DfCursor):
    pass

# Connection
cdef class CursorManager:
    cdef:
        BaseConnection _conn
        object _cur_type
        Cursor _cur
        bint _closed
    cdef inline Cursor _acquire(self)
    cdef inline bint _close(self) except -1

cdef class TransactionManager(CursorManager):
    pass

cdef class BaseConnection:
    cdef:
        # Basic
        str _host
        object _port
        bytes _user
        bytes _password
        bytes _database
        # Charset
        str _charset
        str _collation
        unsigned int _charset_id
        bytes _encoding
        char* _encoding_c
        bint _charset_changed
        # Timeouts
        object _connect_timeout
        object _read_timeout
        bint _read_timeout_changed
        object _write_timeout
        bint _write_timeout_changed
        object _wait_timeout
        bint _wait_timeout_changed
        object _interactive_timeout
        bint _interactive_timeout_changed
        object _lock_wait_timeout
        bint _lock_wait_timeout_changed
        object _execution_timeout
        bint _execution_timeout_changed
        # Client
        str _bind_address
        str _unix_socket
        int _autocommit_mode
        bint _local_infile
        unsigned int _max_allowed_packet
        str _sql_mode
        str _init_command
        object _cursor
        unsigned int _client_flag
        bytes _connect_attrs
        # SSL
        object _ssl_ctx
        # Auth
        AuthPlugin _auth_plugin
        bytes _server_public_key
        # Decode
        bint _use_decimal
        bint _decode_bit
        bint _decode_json
        # Internal
        # . server
        int _server_protocol_version
        str _server_info
        tuple _server_version
        int _server_version_major
        str _server_vendor
        long long _server_thread_id
        bytes _server_salt
        int _server_status
        long long _server_capabilities
        str _server_auth_plugin_name
        # . client
        double _last_used_time
        bint _secure
        str _close_reason
        # . query
        MysqlResult _result
        unsigned int _next_seq_id
        # . transport
        object _reader
        object _writer
    # Setup
    cdef inline bint _setup_charset(self, Charset charset) except -1
    cdef inline bint _setup_client_flag(self, unsigned int client_flag) except -1
    cdef inline bint _setup_connect_attrs(self, object program_name) except -1
    cdef inline bint _setup_internal(self) except -1
    # Cursor
    cpdef Cursor cursor(self, object cursor=?)
    cpdef TransactionManager transaction(self, object cursor=?)
    cdef inline type _validate_cursor(self, object cursor)
    # Query
    cpdef unsigned long long query(self, str sql, bint unbuffered=?)
    cpdef bint begin(self) except -1
    cpdef bint start(self) except -1
    cpdef bint commit(self) except -1
    cpdef bint rollback(self) except -1
    cpdef bint create_savepoint(self, str identifier) except -1
    cpdef bint rollback_savepoint(self, str identifier) except -1
    cpdef bint release_savepoint(self, str identifier) except -1
    cpdef tuple show_warnings(self)
    cpdef bint select_database(self, str db) except -1
    cpdef object escape_args(self, object args, bint many=?, bint itemize=?)
    cpdef bytes encode_sql(self, str sql)
    # . client
    cpdef bint set_charset(self, str charset, object collation=?) except -1
    cpdef bint get_autocommit(self) except -1
    cpdef bint set_autocommit(self, bint value) except -1
    # . timeout
    cpdef bint set_read_timeout(self, object timeout) except -1
    cpdef unsigned int get_read_timeout(self)
    cpdef bint set_write_timeout(self, object timeout) except -1
    cpdef unsigned int get_write_timeout(self)
    cpdef bint set_wait_timeout(self, object timeout) except -1
    cpdef unsigned int get_wait_timeout(self)
    cpdef bint set_interactive_timeout(self, object timeout) except -1
    cpdef unsigned int get_interactive_timeout(self)
    cpdef bint set_lock_wait_timeout(self, object timeout) except -1
    cpdef unsigned int get_lock_wait_timeout(self)
    cpdef bint set_execution_timeout(self, object timeout) except -1
    cpdef unsigned int get_execution_timeout(self)
    cdef inline bint _set_timeout(self, str name, object timeout) except -1
    cdef inline unsigned int _get_timeout(self, str name, bint session)
    # . server
    cpdef tuple get_server_version(self)
    cpdef str get_server_vendor(self)
    # . status
    cpdef unsigned long long get_affected_rows(self)
    cpdef unsigned long long get_insert_id(self)
    cpdef bint get_transaction_status(self) except -1
    # . decode
    cpdef bint set_use_decimal(self, bint value) except -1
    cpdef bint set_decode_bit(self, bint value) except -1
    cpdef bint set_decode_json(self, bint value) except -1
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
    cdef inline bint _set_use_time(self) except -1
    # Read
    cpdef unsigned long long next_result(self, bint unbuffered=?)
    cdef inline MysqlPacket _read_ok_packet(self)
    cdef inline unsigned long long _read_query_result(self, bint unbuffered)
    cdef inline MysqlPacket _read_packet(self)
    cdef inline FieldDescriptorPacket _read_field_descriptor_packet(self)
    cdef inline bytes _read_packet_buffer(self)
    cdef inline bytes _read_bytes(self, unsigned int size)
