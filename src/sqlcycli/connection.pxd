# cython: language_level=3
from sqlcycli.charset cimport Charset
from sqlcycli._auth cimport AuthPlugin
from sqlcycli.protocol cimport MysqlPacket, FieldDescriptorPacket

# Constant
cdef:
    str DEFAULT_USER, DEFUALT_CHARSET
    int MAX_CONNECT_TIMEOUT
    int DEFALUT_MAX_ALLOWED_PACKET, MAXIMUM_MAX_ALLOWED_PACKET
    bytes DEFAULT_CONNECT_ATTRS
    unsigned int MAX_PACKET_LENGTH
    object INSERT_VALUES_RE, SERVER_VERSION_RE

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
