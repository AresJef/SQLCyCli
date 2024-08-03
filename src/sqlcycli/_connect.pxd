# cython: language_level=3

# Connection
cdef class ConnectionManager:
    cdef:
        # . connection
        object _conn, _aconn
        # . arguments
        object _cursor, _args
    # Sync
    cdef inline object _sync_cursor(self, object cursor)
    # Async
    cdef inline object _async_cursor(self, object cursor)

cpdef ConnectionManager connect(
    object host=?,
    object port=?,
    object user=?,
    object password=?,
    object database=?,
    object charset=?,
    object collation=?,
    object connect_timeout=?,
    object read_timeout=?,
    object write_timeout=?,
    object wait_timeout=?,
    object bind_address=?,
    object unix_socket=?,
    object autocommit=?,
    object local_infile=?,
    object max_allowed_packet=?,
    object sql_mode=?,
    object init_command=?,
    object cursor=?,
    object client_flag=?,
    object program_name=?,
    object option_file=?,
    object ssl=?,
    object auth_plugin=?,
    object server_public_key=?,
    object use_decimal=?,
    object decode_json=?,
)
