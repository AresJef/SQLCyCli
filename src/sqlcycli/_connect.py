# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython

# Python imports
from os import PathLike
from typing import Generator, Any
from asyncio import AbstractEventLoop
from sqlcycli._ssl import SSL
from sqlcycli._auth import AuthPlugin
from sqlcycli._optionfile import OptionFile
from sqlcycli.connection import (
    Cursor,
    DictCursor,
    DfCursor,
    SSCursor,
    SSDictCursor,
    SSDfCursor,
    BaseConnection,
    Connection,
)
from sqlcycli import aio, errors

__all__ = ["connect", "ConnectionManager", "create_pool", "PoolManager"]


# Utils ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _validate_sync_cursor(cursor: Any) -> object:
    """(cfunc) Validate and map the given 'cursor'
    argument to the correct sync `<'Cursor'>`."""
    if cursor is None:
        return None
    if type(cursor) is type:
        if issubclass(cursor, Cursor):
            return cursor
        if cursor is aio.Cursor:
            return Cursor
        if cursor is aio.DictCursor:
            return DictCursor
        if cursor is aio.DfCursor:
            return DfCursor
        if cursor is aio.SSCursor:
            return SSCursor
        if cursor is aio.SSDictCursor:
            return SSDictCursor
        if cursor is aio.SSDfCursor:
            return SSDfCursor
    raise errors.InvalidConnectionArgsError(
        "Invalid 'cursor' argument: %r. "
        "Must be type [class] of %r." % (cursor, Cursor)
    )


@cython.cfunc
@cython.inline(True)
def _validate_async_cursor(cursor: Any) -> object:
    """(cfunc) Validate and map the given 'cursor'
    argument to the correct async `<'aio.Cursor'>`."""
    if cursor is None:
        return None
    if type(cursor) is type:
        if issubclass(cursor, aio.Cursor):
            return cursor
        if cursor is Cursor:
            return aio.Cursor
        if cursor is DictCursor:
            return aio.DictCursor
        if cursor is DfCursor:
            return aio.DfCursor
        if cursor is SSCursor:
            return aio.SSCursor
        if cursor is SSDictCursor:
            return aio.SSDictCursor
        if cursor is SSDfCursor:
            return aio.SSDfCursor
    raise errors.InvalidConnectionArgsError(
        "Invalid 'cursor' argument: %r. "
        "Must be type [class] of %r." % (cursor, aio.Cursor)
    )


# Connection ----------------------------------------------------------------------------------
@cython.cclass
class ConnectionManager:
    """The Context Manager for both `sync` and `async` Connection."""

    # . connection
    _conn_sync: BaseConnection
    _conn_async: aio.BaseConnection
    # . arguments
    _kwargs: dict[str, Any]
    _cursor: type[Cursor | aio.Cursor]
    _loop: AbstractEventLoop

    def __init__(
        self,
        kwargs: dict[str, Any],
        cursor: type[Cursor | aio.Cursor] | None,
        loop: AbstractEventLoop | None,
    ) -> None:
        """The Context Manager for both `sync` and `async` Connection.

        For information about the arguments, please
        refer to the 'connect()' function.
        """
        # Connection
        self._conn_sync = None
        self._conn_async = None
        # Arguments
        self._kwargs = kwargs
        self._cursor = cursor
        self._loop = loop

    # Sync --------------------------------------------------------------------------------------
    def __enter__(self) -> BaseConnection:
        conn = Connection(
            cursor=_validate_sync_cursor(self._cursor),
            **self._kwargs,
        )
        conn.connect()
        self._conn_sync = conn
        return self._conn_sync

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn_sync.close()
        self._conn_sync = None

    # Async -------------------------------------------------------------------------------------
    async def _acquire_async_conn(self) -> aio.BaseConnection:
        """(internal) Acquire an `async` connection `<'BaseConnection'>`."""
        conn = aio.Connection(
            cursor=_validate_async_cursor(self._cursor),
            loop=self._loop,
            **self._kwargs,
        )
        await conn.connect()
        return conn

    def __await__(self) -> Generator[Any, Any, aio.BaseConnection]:
        return self._acquire_async_conn().__await__()

    async def __aenter__(self) -> aio.BaseConnection:
        self._conn_async = await self._acquire_async_conn()
        return self._conn_async

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._conn_async.close()
        self._conn_async = None

    def __del__(self):
        if self._conn_sync is not None:
            self._conn_sync.force_close()
            self._conn_sync = None
        if self._conn_async is not None:
            self._conn_async.force_close()
            self._conn_async = None


@cython.ccall
def connect(
    host: str | None = "localhost",
    port: int | Any = 3306,
    user: str | bytes | None = None,
    password: str | bytes | None = None,
    database: str | bytes | None = None,
    *,
    charset: str | None = "utf8mb4",
    collation: str | None = None,
    connect_timeout: int | Any = 5,
    read_timeout: int | None = None,
    write_timeout: int | None = None,
    wait_timeout: int | None = None,
    bind_address: str | None = None,
    unix_socket: str | None = None,
    autocommit: bool | None = False,
    local_infile: bool = False,
    max_allowed_packet: int | str | None = None,
    sql_mode: str | None = None,
    init_command: str | None = None,
    cursor: type[Cursor] | None = Cursor,
    client_flag: int | Any = 0,
    program_name: str | None = None,
    option_file: str | bytes | PathLike | OptionFile | None = None,
    ssl: SSL | object | None = None,
    auth_plugin: dict[str | bytes, type] | AuthPlugin | None = None,
    server_public_key: bytes | None = None,
    use_decimal: bool = False,
    decode_json: bool = False,
    loop: AbstractEventLoop | None = None,
) -> ConnectionManager:
    """Connect to the server and acquire a `sync` or `async`
    connection through context manager `<'ConnectionManager'>`.

    :param host: `<'str/None'>` The host of the server. Defaults to `'localhost'`.
    :param port: `<'int'>` The port of the server. Defaults to `3306`.
    :param user: `<'str/bytes/None'>` The username to login as. Defaults to `None`.
    :param password: `<'str/bytes/None'>` The password for login authentication. Defaults to `None`.
    :param database: `<'str/bytes/None'>` The default database to use by the connection. Defaults to `None`.
    :param charset: `<'str/None'>` The character set for the connection. Defaults to `'utf8mb4'`.
    :param collation: `<'str/None'>` The collation for the connection. Defaults to `None`.
    :param connect_timeout: `<'int'>` Timeout in seconds for establishing the connection. Defaults to `5`.
    :param read_timeout: `<'int/None>` Set connection (SESSION) 'net_read_timeout'. Defaults to `None` (use GLOBAL settings).
    :param write_timeout: `<'int/None>` Set connection (SESSION) 'net_write_timeout'. Defaults to `None` (use GLOBAL settings).
    :param wait_timeout: `<'int/None>` Set connection (SESSION) 'wait_timeout'. Defaults to `None` (use GLOBAL settings).
    :param bind_address: `<'str/None'>` The interface from which to connect to the host. Accept both hostname or IP address. Defaults to `None`.
    :param unix_socket: `<'str/None'>` The unix socket for establishing connection rather than TCP/IP. Defaults to `None`.
    :param autocommit: `<'bool/None'>` The autocommit mode for the connection. `None` means use server default. Defaults to `False`.
    :param local_infile: `<'bool'>` Enable/Disable LOAD DATA LOCAL command. Defaults to `False`.
    :param max_allowed_packet: `<'int/str/None'>` The max size of packet sent to server in bytes. Defaults to `None` (16MB).
    :param sql_mode: `<'str/None'>` The default SQL_MODE for the connection. Defaults to `None`.
    :param init_command: `<'str/None'>` The initial SQL statement to run when connection is established. Defaults to `None`.
    :param cursor: `<'type[Cursor]'>` The default cursor type (class) to use. Defaults to `<'Cursor'>`.
    :param client_flag: `<'int'>` Custom flags to sent to server, see 'constants.CLIENT'. Defaults to `0`.
    :param program_name: `<'str/None'>` The program name for the connection. Defaults to `None`.
    :param option_file: `<'OptionFile/PathLike/None>` The MySQL option file to load connection parameters. Defaults to `None`.
        - Recommand use <'OptionFile'> to load MySQL option file.
        - If passed str/bytes/PathLike argument, it will be automatically converted
            to <'OptionFile'>, with option group defaults to 'client'.

    :param ssl: `<'SSL/ssl.SSLContext/None'>` The SSL configuration for the connection. Defaults to `None`.
        - Supports both <'SSL'> or pre-configured <'ssl.SSLContext'> object.

    :param auth_plugin: `<'AuthPlugin/dict/None'>` The authentication plugins handlers. Defaults to `None`.
        - Recommand use <'AuthPlugin'> to setup MySQL authentication plugin handlers.
        - If passed dict argument, it will be automatically converted to <'AuthPlugin'>.

    :param server_public_key: `<'bytes/None'>` The public key for the server authentication. Defaults to `None`.
    :param use_decimal: `<'bool'>` If `True` use <'decimal.Decimal'> to represent DECIMAL column data, else use <'float'>. Defaults to `False`.
    :param decode_json: `<'bool'>` If `True` decode JSON column data, else keep as original json string. Defaults to `False`.
    :param loop: `<'AbstractEventLoop/None'>` The event loop for the `async` connection. Defaults to `None`.
        - Only applicable for `async` connection. `sync` connection will ignore this argument.

    ### Example (sync):
    >>> with connect("localhost", 3306, "root", "password") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    ### Example (async):
    >>> async with connect("localhost", 3306, "root", "password") as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
    """
    return ConnectionManager(
        {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": charset,
            "collation": collation,
            "connect_timeout": connect_timeout,
            "read_timeout": read_timeout,
            "write_timeout": write_timeout,
            "wait_timeout": wait_timeout,
            "bind_address": bind_address,
            "unix_socket": unix_socket,
            "autocommit": autocommit,
            "local_infile": local_infile,
            "max_allowed_packet": max_allowed_packet,
            "sql_mode": sql_mode,
            "init_command": init_command,
            "client_flag": client_flag,
            "program_name": program_name,
            "option_file": option_file,
            "ssl": ssl,
            "auth_plugin": auth_plugin,
            "server_public_key": server_public_key,
            "use_decimal": use_decimal,
            "decode_json": decode_json,
        },
        cursor,
        loop,
    )


# Pool ----------------------------------------------------------------------------------------
@cython.cclass
class PoolManager:
    """The Context Manager for Pool."""

    # . pool
    _pool: aio.Pool
    # . arguments
    _kwargs: dict[str, Any]
    _cursor: type[Cursor | aio.Cursor]

    def __init__(
        self,
        kwargs: dict[str, Any],
        cursor: type[Cursor | aio.Cursor] | None,
    ) -> None:
        """The Context Manager for Pool.

        For information about the arguments, please
        refer to the 'create_pool()' function.
        """
        # Pool
        self._pool = None
        # Arguments
        self._kwargs = kwargs
        self._cursor = cursor

    # Sync --------------------------------------------------------------------------------------
    def __enter__(self) -> aio.Pool:
        pool = aio.Pool(cursor=_validate_async_cursor(self._cursor), **self._kwargs)
        self._pool = pool
        return self._pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.terminate()
        self._pool = None

    # Async -------------------------------------------------------------------------------------
    async def _create_and_fill_pool(self) -> aio.Pool:
        """(internal) Create a pool and fill free connections `<'Pool'>`."""
        pool = aio.Pool(cursor=_validate_async_cursor(self._cursor), **self._kwargs)
        await pool.fill(-1)
        return pool

    def __await__(self) -> Generator[Any, Any, aio.Pool]:
        return self._create_and_fill_pool().__await__()

    async def __aenter__(self) -> aio.Pool:
        self._pool = await self._create_and_fill_pool()
        return self._pool

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._pool.close()
        self._pool = None

    def __del__(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None


@cython.ccall
def create_pool(
    host: str | None = "localhost",
    port: int | Any = 3306,
    user: str | bytes | None = None,
    password: str | bytes | None = None,
    database: str | bytes | None = None,
    min_size: int | Any = 0,
    max_size: int | Any = 10,
    recycle: int | None = None,
    *,
    charset: str | None = "utf8mb4",
    collation: str | None = None,
    connect_timeout: int | Any = 5,
    read_timeout: int | None = None,
    write_timeout: int | None = None,
    wait_timeout: int | None = None,
    bind_address: str | None = None,
    unix_socket: str | None = None,
    autocommit: bool | None = False,
    local_infile: bool = False,
    max_allowed_packet: int | str | None = None,
    sql_mode: str | None = None,
    init_command: str | None = None,
    cursor: type[Cursor] | None = Cursor,
    client_flag: int | Any = 0,
    program_name: str | None = None,
    option_file: str | bytes | PathLike | OptionFile | None = None,
    ssl: SSL | object | None = None,
    auth_plugin: dict[str | bytes, type] | AuthPlugin | None = None,
    server_public_key: bytes | None = None,
    use_decimal: bool = False,
    decode_json: bool = False,
    loop: AbstractEventLoop | None = None,
) -> PoolManager:
    """Create a connection pool to manage and maintain `async`
    connections through context manager `<'PoolManager'>`.

    #### Pool args:
    :param min_size: `<'int'>` The minimum number of active connections to maintain. Defaults to `0`.
    :param max_size: `<'int'>` The maximum number of active connections to maintain. Defaults to `10`.
    :param recycle: `<'int/None'>` The recycle time in seconds. Defaults to `None`.
        - If set to positive integer, the pool will automatically close
            and remove any connections idling more than the 'recycle' time.
        - If 'recycle=None' (Default), recycling is disabled.
    :param loop: `<'AbstractEventLoop/None'>` The event loop for the pool connections. Defaults to `None`.

    #### Connection args:
    :param host: `<'str/None'>` The host of the server. Defaults to `'localhost'`.
    :param port: `<'int'>` The port of the server. Defaults to `3306`.
    :param user: `<'str/bytes/None'>` The username to login as. Defaults to `None`.
    :param password: `<'str/bytes/None'>` The password for login authentication. Defaults to `None`.
    :param database: `<'str/bytes/None'>` The default database to use by the connection. Defaults to `None`.
    :param charset: `<'str/None'>` The character set for the connection. Defaults to `'utf8mb4'`.
    :param collation: `<'str/None'>` The collation for the connection. Defaults to `None`.
    :param connect_timeout: `<'int'>` Timeout in seconds for establishing the connection. Defaults to `5`.
    :param read_timeout: `<'int/None>` Set connection (SESSION) 'net_read_timeout'. Defaults to `None` (use GLOBAL settings).
    :param write_timeout: `<'int/None>` Set connection (SESSION) 'net_write_timeout'. Defaults to `None` (use GLOBAL settings).
    :param wait_timeout: `<'int/None>` Set connection (SESSION) 'wait_timeout'. Defaults to `None` (use GLOBAL settings).
    :param bind_address: `<'str/None'>` The interface from which to connect to the host. Accept both hostname or IP address. Defaults to `None`.
    :param unix_socket: `<'str/None'>` The unix socket for establishing connection rather than TCP/IP. Defaults to `None`.
    :param autocommit: `<'bool/None'>` The autocommit mode for the connection. `None` means use server default. Defaults to `False`.
    :param local_infile: `<'bool'>` Enable/Disable LOAD DATA LOCAL command. Defaults to `False`.
    :param max_allowed_packet: `<'int/str/None'>` The max size of packet sent to server in bytes. Defaults to `None` (16MB).
    :param sql_mode: `<'str/None'>` The default SQL_MODE for the connection. Defaults to `None`.
    :param init_command: `<'str/None'>` The initial SQL statement to run when connection is established. Defaults to `None`.
    :param cursor: `<'type[Cursor]/None'>` The default cursor type (class) to use. Defaults to `<'Cursor'>`.
    :param client_flag: `<'int'>` Custom flags to sent to server, see 'constants.CLIENT'. Defaults to `0`.
    :param program_name: `<'str/None'>` The program name for the connection. Defaults to `None`.
    :param option_file: `<'OptionFile/PathLike/None>` The MySQL option file to load connection parameters. Defaults to `None`.
        - Recommand use <'OptionFile'> to load MySQL option file.
        - If passed str/bytes/PathLike argument, it will be automatically converted
            to <'OptionFile'>, with option group defaults to 'client'.

    :param ssl: `<'SSL/ssl.SSLContext/None'>` The SSL configuration for the connection. Defaults to `None`.
        - Supports both <'SSL'> or pre-configured <'ssl.SSLContext'> object.

    :param auth_plugin: `<'AuthPlugin/dict/None'>` The authentication plugins handlers. Defaults to `None`.
        - Recommand use <'AuthPlugin'> to setup MySQL authentication plugin handlers.
        - If passed dict argument, it will be automatically converted to <'AuthPlugin'>.

    :param server_public_key: `<'bytes/None'>` The public key for the server authentication. Defaults to `None`.
    :param use_decimal: `<'bool'>` If `True` use <'decimal.Decimal'> to represent DECIMAL column data, else use <'float'>. Defaults to `False`.
    :param decode_json: `<'bool'>` If `True` decode JSON column data, else keep as original json string. Defaults to `False`.
    """
    return PoolManager(
        {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "min_size": min_size,
            "max_size": max_size,
            "recycle": recycle,
            "charset": charset,
            "collation": collation,
            "connect_timeout": connect_timeout,
            "read_timeout": read_timeout,
            "write_timeout": write_timeout,
            "wait_timeout": wait_timeout,
            "bind_address": bind_address,
            "unix_socket": unix_socket,
            "autocommit": autocommit,
            "local_infile": local_infile,
            "max_allowed_packet": max_allowed_packet,
            "sql_mode": sql_mode,
            "init_command": init_command,
            "client_flag": client_flag,
            "program_name": program_name,
            "option_file": option_file,
            "ssl": ssl,
            "auth_plugin": auth_plugin,
            "server_public_key": server_public_key,
            "use_decimal": use_decimal,
            "decode_json": decode_json,
            "loop": loop,
        },
        cursor,
    )
