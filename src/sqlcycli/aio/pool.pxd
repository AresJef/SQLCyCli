# cython: language_level=3
from sqlcycli.charset cimport Charset
from sqlcycli._auth cimport AuthPlugin
from sqlcycli.aio.connection cimport BaseConnection

# Connection
cdef class PoolConnection(BaseConnection):
    cdef:
        Py_ssize_t _pool_id
        bint _close_scheduled
    # Pool
    cpdef bint schedule_close(self) except -1

# Pool
cdef class PoolConnectionManager:
    cdef:
        Pool _pool
        PoolConnection _conn
        bint _closed

cdef class Pool:
    cdef:
        # Pool
        # . counting
        unsigned int _acqr
        unsigned int _free
        # . internal
        unsigned int _min_size
        unsigned int _max_size
        long long _recycle
        Py_ssize_t _id
        object _free_conns
        set _used_conns
        object _loop
        object _condition
        bint _closing
        bint _closed
        # . server
        int _server_protocol_version
        str _server_info
        tuple _server_version
        int _server_version_major
        str _server_vendor
        str _server_auth_plugin_name
        # Connection args
        # . basic
        str _host
        object _port
        bytes _user
        bytes _password
        bytes _database
        # . charset
        Charset _charset
        # . timeouts
        object _connect_timeout
        object _read_timeout
        object _write_timeout
        object _wait_timeout
        # . client
        str _bind_address
        str _unix_socket
        object _autocommit_mode
        bint _autocommit
        object _local_infile
        object _max_allowed_packet
        str _sql_mode
        str _init_command
        object _cursor
        object _client_flag
        str _program_name
        # . ssl
        object _ssl_ctx
        # . auth
        AuthPlugin _auth_plugin
        bytes _server_public_key
        # . decode
        bint _use_decimal
        bint _decode_bit
        bint _decode_json
        
    # Setup
    cdef inline bint _setup(self, object min_size, object max_size, object recycle, object loop) except -1
    # Pool
    cpdef unsigned int get_free(self)
    cpdef unsigned int get_used(self)
    cpdef unsigned int get_total(self)
    cpdef bint set_min_size(self, unsigned int size) except -1
    cpdef bint set_recycle(self, object size) except -1
    cdef inline PoolConnection _get_free_conn(self)
    cdef inline bint _add_free_conn(self, object conn) except -1
    cpdef bint set_autocommit(self, bint value) except -1
    cpdef bint set_use_decimal(self, bint value) except -1
    cpdef bint set_decode_bit(self, bint value) except -1
    cpdef bint set_decode_json(self, bint value) except -1
    # Acquire / Fill / Release
    cpdef PoolConnectionManager acquire(self)
    # Close
    cpdef bint terminate(self) except -1
    cpdef bint closed(self) except -1
    cdef inline bint _verify_open(self) except -1
  