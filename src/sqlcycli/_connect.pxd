# cython: language_level=3
from sqlcycli cimport connection as sync_conn
from sqlcycli.aio cimport connection as async_conn, pool as aio_pool

# Connection
cdef class ConnectionManager:
    cdef:
        # . connection
        sync_conn.BaseConnection _conn_sync
        async_conn.BaseConnection _conn_async
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
    object interactive_timeout=?,
    object lock_wait_timeout=?,
    object execution_timeout=?,
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
    object decode_bit=?,
    object decode_json=?,
    object loop=?,
)

# Pool
cdef class PoolManager:
    cdef:
        # . pool
        aio_pool.Pool _pool
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
    object interactive_timeout=?,
    object lock_wait_timeout=?,
    object execution_timeout=?,
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
    object decode_bit=?,
    object decode_json=?,
)
