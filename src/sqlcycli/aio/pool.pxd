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

cdef class Pool:
    cdef:
        # Connection args
        # . basic
        str _host
        object _port
        bytes _user, _password, _database
        # . charset
        Charset _charset
        # . timeouts
        object _connect_timeout
        object _read_timeout
        object _write_timeout
        object _wait_timeout
        # . client
        str _bind_address, _unix_socket
        object _autocommit_mode
        bint _autocommit_value
        object _local_infile, _max_allowed_packet
        str _sql_mode, _init_command
        object _cursor, _client_flag
        str _program_name
        # . ssl
        object _ssl_ctx
        # . auth
        AuthPlugin _auth_plugin
        bytes _server_public_key
        # . decode
        bint _use_decimal, _decode_json
        # . server
        int _server_protocol_version
        str _server_info
        tuple _server_version
        int _server_version_major
        str _server_vendor
        str _server_auth_plugin_name
        # Pool
        unsigned int _min_size, _max_size
        Py_ssize_t _recycle, _id
        unsigned int _acqr, _free, _used
        object _free_conn
        set _used_conn
        object _loop, _cond
        bint _closing, _closed
    # Pool
    cpdef unsigned int get_free(self) except -1
    cpdef unsigned int get_used(self) except -1
    cpdef unsigned int get_total(self) except -1
    cpdef bint set_min_size(self, unsigned int value) except -1
    cpdef bint set_recycle(self, unsigned int value) except -1
    cpdef PoolConnectionManager acquire(self)
    # Close
    cpdef bint terminate(self) except -1
    cpdef bint closed(self) except -1
    cdef inline bint _verify_open(self) except -1
  