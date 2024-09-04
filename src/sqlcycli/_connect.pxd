# cython: language_level=3

# Utils
cdef object _validate_sync_cursor(object cursor)
cdef object _validate_async_cursor(object cursor)

# Connection
cdef class ConnectionManager:
    cdef:
        # . connection
        object _conn_sync
        object _conn_async
        # . arguments
        dict _kwargs
        object _cursor
        object _loop

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
    object loop=?,
)

# Pool
cdef class PoolManager:
    cdef:
        # . pool
        object _pool
        # . arguments
        dict _kwargs
        object _cursor

cpdef PoolManager create_pool(
    object host=?,
    object port=?,
    object user=?,
    object password=?,
    object database=?,
    object min_size=?,
    object max_size=?,
    object recycle=?,
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
    object loop=?,
)
