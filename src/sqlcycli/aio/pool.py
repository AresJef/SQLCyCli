# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False
from __future__ import annotations

# Cython imports
import cython
from cython.cimports.libc.limits import UINT_MAX  # type: ignore
from cython.cimports.cpython.time import time as unix_time  # type: ignore
from cython.cimports.cpython.set import PySet_Add as set_add  # type: ignore
from cython.cimports.cpython.set import PySet_Clear as set_clear  # type: ignore
from cython.cimports.cpython.set import PySet_GET_SIZE as set_len  # type: ignore
from cython.cimports.cpython.set import PySet_Discard as set_discard  # type: ignore
from cython.cimports.sqlcycli._ssl import SSL  # type: ignore
from cython.cimports.sqlcycli.charset import Charset  # type: ignore
from cython.cimports.sqlcycli._auth import AuthPlugin  # type: ignore
from cython.cimports.sqlcycli.transcode import escape  # type: ignore
from cython.cimports.sqlcycli._optionfile import OptionFile  # type: ignore
from cython.cimports.sqlcycli.aio import connection as async_conn  # type: ignore
from cython.cimports.sqlcycli import connection as sync_conn, utils  # type: ignore

# Python imports
from os import PathLike
from collections import deque
from typing import Literal, Generator, Any
from asyncio import AbstractEventLoop, Condition
from asyncio import gather as _gather, get_event_loop as _get_event_loop
from pandas import DataFrame
from sqlcycli._ssl import SSL
from sqlcycli.charset import Charset
from sqlcycli._auth import AuthPlugin
from sqlcycli.transcode import escape
from sqlcycli._optionfile import OptionFile
from sqlcycli.aio import connection as async_conn
from sqlcycli import connection as sync_conn, utils, errors

__all__ = [
    "PoolConnection",
    "PoolSyncConnection",
    "PoolConnectionManager",
    "PoolTransactionManager",
    "Pool",
]


# Utils ---------------------------------------------------------------------------------------
@cython.ccall
def validate_sync_cursor(cursor: object) -> type:
    """Validate and map the given 'cursor' to the corresponding
    `sync` cursor type `<'type[Cursor]/None'>`.
    """
    if cursor is None:
        return None
    if type(cursor) is type:
        if issubclass(cursor, sync_conn.Cursor):
            return cursor
        if cursor is tuple:
            return sync_conn.Cursor
        if cursor is dict:
            return sync_conn.DictCursor
        if cursor is DataFrame:
            return sync_conn.DfCursor
        if cursor is async_conn.Cursor:
            return sync_conn.Cursor
        if cursor is async_conn.DictCursor:
            return sync_conn.DictCursor
        if cursor is async_conn.DfCursor:
            return sync_conn.DfCursor
        if cursor is async_conn.SSCursor:
            return sync_conn.SSCursor
        if cursor is async_conn.SSDictCursor:
            return sync_conn.SSDictCursor
        if cursor is async_conn.SSDfCursor:
            return sync_conn.SSDfCursor
    raise errors.InvalidConnectionArgsError(
        "Invalid 'cursor' argument: %r.\n"
        "Expects type (subclass) of %r." % (cursor, sync_conn.Cursor)
    )


@cython.ccall
def validate_async_cursor(cursor: object) -> type:
    """Validate and map the given 'cursor' to the corresponding
    `async` cursor type `<'type[Cursor]/None'>`.
    """
    if cursor is None:
        return None
    if type(cursor) is type:
        if issubclass(cursor, async_conn.Cursor):
            return cursor
        if cursor is tuple:
            return async_conn.Cursor
        if cursor is dict:
            return async_conn.DictCursor
        if cursor is DataFrame:
            return async_conn.DfCursor
        if cursor is sync_conn.Cursor:
            return async_conn.Cursor
        if cursor is sync_conn.DictCursor:
            return async_conn.DictCursor
        if cursor is sync_conn.DfCursor:
            return async_conn.DfCursor
        if cursor is sync_conn.SSCursor:
            return async_conn.SSCursor
        if cursor is sync_conn.SSDictCursor:
            return async_conn.SSDictCursor
        if cursor is sync_conn.SSDfCursor:
            return async_conn.SSDfCursor
    raise errors.InvalidConnectionArgsError(
        "Invalid 'cursor' argument: %r.\n"
        "Expects type (subclass) of %r." % (cursor, async_conn.Cursor)
    )


# Pool Connection -----------------------------------------------------------------------------
@cython.cclass
class PoolConnection(async_conn.BaseConnection):
    """Represents the `async` connection to the server managed by a `Pool`.

    This class serves as the connection object managed by a pool only.
    It does not perform argument validations during initialization.
    Such validations are delegated to the class `<'aio.Pool'>`.

    ### Please do `NOT` create an instance of this class directly.
    """

    # . pool
    _pool_id: cython.Py_ssize_t
    _close_scheduled: cython.bint

    def __init__(
        self,
        pool_id: cython.Py_ssize_t,
        host: str,
        port: int,
        user: bytes | None,
        password: bytes,
        database: bytes | None,
        charset: Charset,
        connect_timeout: int,
        read_timeout: int | None,
        write_timeout: int | None,
        wait_timeout: int | None,
        interactive_timeout: int | None,
        lock_wait_timeout: int | None,
        execution_timeout: int | None,
        bind_address: str | None,
        unix_socket: str | None,
        autocommit_mode: cython.int,
        local_infile: cython.bint,
        max_allowed_packet: cython.uint,
        sql_mode: str | None,
        init_command: str | None,
        cursor: type[async_conn.Cursor],
        client_flag: cython.uint,
        program_name: str | None,
        ssl_ctx: object | None,
        auth_plugin: AuthPlugin | None,
        server_public_key: bytes | None,
        use_decimal: cython.bint,
        decode_bit: cython.bint,
        decode_json: cython.bint,
        loop: AbstractEventLoop,
    ):
        """The `async` connection to the server managed by a `Pool`.

        This class serves as the connection object managed by a pool only.
        It does not perform argument validations during initialization.
        Such validations are delegated to the class `<'aio.Pool'>`.

        ### Please do `NOT` create an instance of this class directly.

        :param pool_id `<'int'>`: The unique identifier of the pool.
        """
        # . pool
        self._pool_id = pool_id
        self._close_scheduled = False
        # . internal
        self._setup_internal()
        # . basic
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        # . charset
        self._setup_charset(charset)
        # . timeouts
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._wait_timeout = wait_timeout
        self._interactive_timeout = interactive_timeout
        self._lock_wait_timeout = lock_wait_timeout
        self._execution_timeout = execution_timeout
        # . client
        self._bind_address = bind_address
        self._unix_socket = unix_socket
        self._autocommit_mode = autocommit_mode
        self._local_infile = local_infile
        self._max_allowed_packet = max_allowed_packet
        self._sql_mode = sql_mode
        self._init_command = init_command
        self._cursor = cursor
        self._setup_client_flag(client_flag)
        self._setup_connect_attrs(program_name)
        # . ssl
        self._ssl_ctx = ssl_ctx
        # . auth
        self._auth_plugin = auth_plugin
        self._server_public_key = server_public_key
        # . decode
        self._use_decimal = use_decimal
        self._decode_bit = decode_bit
        self._decode_json = decode_json
        # . loop
        self._loop = loop

    # Property --------------------------------------------------------------------------------
    # . pool
    @property
    def close_scheduled(self) -> bool:
        """Whether the connection is scheduled to be closed
        when 'release()' back to the pool `<'bool'>`."""
        return self._close_scheduled

    # Pool ------------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def schedule_close(self) -> cython.bint:
        """After calling the method, this connection is scheduled
        to be closed when 'release()' back to the pool.
        """
        self._close_scheduled = True


@cython.cclass
class PoolSyncConnection(sync_conn.BaseConnection):
    """Represents the `sync` connection to the server managed by a `Pool`.

    This class serves as the connection object managed by a pool only.
    It does not perform argument validations during initialization.
    Such validations are delegated to the class `<'aio.Pool'>`.

    ### Please do `NOT` create an instance of this class directly.
    """

    # . pool
    _pool_id: cython.Py_ssize_t
    _close_scheduled: cython.bint

    def __init__(
        self,
        pool_id: cython.Py_ssize_t,
        host: str,
        port: int,
        user: bytes | None,
        password: bytes,
        database: bytes | None,
        charset: Charset,
        connect_timeout: int,
        read_timeout: int | None,
        write_timeout: int | None,
        wait_timeout: int | None,
        interactive_timeout: int | None,
        lock_wait_timeout: int | None,
        execution_timeout: int | None,
        bind_address: str | None,
        unix_socket: str | None,
        autocommit_mode: cython.int,
        local_infile: cython.bint,
        max_allowed_packet: cython.uint,
        sql_mode: str | None,
        init_command: str | None,
        cursor: type[sync_conn.Cursor],
        client_flag: cython.uint,
        program_name: str | None,
        ssl_ctx: object | None,
        auth_plugin: AuthPlugin | None,
        server_public_key: bytes | None,
        use_decimal: cython.bint,
        decode_bit: cython.bint,
        decode_json: cython.bint,
    ):
        """The `sync` connection to the server managed by a `Pool`.

        This class serves as the connection object managed by a pool only.
        It does not perform argument validations during initialization.
        Such validations are delegated to the class `<'aio.Pool'>`.

        ### Please do `NOT` create an instance of this class directly.

        :param pool_id `<'int'>`: The unique identifier of the pool.
        """
        # . pool
        self._pool_id = pool_id
        self._close_scheduled = False
        # . internal
        self._setup_internal()
        # . basic
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        # . charset
        self._setup_charset(charset)
        # . timeouts
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._wait_timeout = wait_timeout
        self._interactive_timeout = interactive_timeout
        self._lock_wait_timeout = lock_wait_timeout
        self._execution_timeout = execution_timeout
        # . client
        self._bind_address = bind_address
        self._unix_socket = unix_socket
        self._autocommit_mode = autocommit_mode
        self._local_infile = local_infile
        self._max_allowed_packet = max_allowed_packet
        self._sql_mode = sql_mode
        self._init_command = init_command
        self._cursor = cursor
        self._setup_client_flag(client_flag)
        self._setup_connect_attrs(program_name)
        # . ssl
        self._ssl_ctx = ssl_ctx
        # . auth
        self._auth_plugin = auth_plugin
        self._server_public_key = server_public_key
        # . decode
        self._use_decimal = use_decimal
        self._decode_bit = decode_bit
        self._decode_json = decode_json

    # Property --------------------------------------------------------------------------------
    # . pool
    @property
    def close_scheduled(self) -> bool:
        """Whether the connection is scheduled to be closed
        when 'release()' back to the pool `<'bool'>`."""
        return self._close_scheduled

    # Pool ------------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def schedule_close(self) -> cython.bint:
        """After calling the method, this connection is scheduled
        to be closed when 'release()' back to the pool.
        """
        self._close_scheduled = True


# Pool ----------------------------------------------------------------------------------------
@cython.cclass
class PoolConnectionManager:
    """The Context Manager for a PoolConnection."""

    _pool: Pool
    _sync_conn: PoolSyncConnection
    _async_conn: PoolConnection

    def __init__(self, pool: Pool) -> None:
        """The Context Manager for a PoolConnection.

        :param pool `<'Pool'>`: The pool to manage the connection.
        """
        self._pool = pool
        self._sync_conn = None
        self._async_conn = None

    # Sync ---------------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _acquire_sync_conn(self) -> PoolSyncConnection:
        """(internal) Acquire a `sync` connection from the pool `<'PoolSyncConnection'>`."""
        try:
            conn = self._pool._acquire_sync_conn()
        except:  # noqa
            self._release_sync_conn(False)
            raise
        if conn is None:
            self._pool._verify_open()
            raise errors.PoolClosedError(
                0, "Failed to acquire connection from the pool for unknown reason."
            )
        return conn

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _release_sync_conn(self, close: cython.bint) -> cython.bint:
        """(internal) Release the `sync` connection back to the pool."""
        conn: PoolSyncConnection = self._sync_conn
        if conn is not None:
            if close:
                conn.schedule_close()
            self._pool._release_sync_conn(conn)
            self._sync_conn = None
        return True

    def __enter__(self) -> PoolSyncConnection:
        self._sync_conn = self._acquire_sync_conn()
        return self._sync_conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._release_sync_conn(False)

    # Async --------------------------------------------------------------------------------------
    async def _acquire_async_conn(self) -> PoolConnection:
        """(internal) Acquire `async`` connection from the pool `<'PoolConnection'>`."""
        try:
            conn = await self._pool._acquire_async_conn()
        except:  # noqa
            await self._release_async_conn(False)
            raise
        if conn is None:
            self._pool._verify_open()
            raise errors.PoolClosedError(
                0, "Failed to acquire connection from the pool for unknown reason."
            )
        return conn

    async def _release_async_conn(self, close: cython.bint) -> None:
        """(internal) Release the `async` connection back to the pool.

        :param close `<'bool'>`: Whether to close the connection at Pool release.
        """
        conn: PoolConnection = self._async_conn
        if conn is not None:
            if close:
                conn.schedule_close()
            await self._pool._release_async_conn(conn)
            self._async_conn = None
        return None

    def __await__(self) -> Generator[Any, Any, PoolConnection]:
        return self._acquire_async_conn().__await__()

    async def __aenter__(self) -> PoolConnection:
        self._async_conn = await self._acquire_async_conn()
        return self._async_conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._release_async_conn(False)

    # Special ------------------------------------------------------------------------------------
    def __del__(self):
        self._sync_conn = None
        self._async_conn = None
        self._pool = None


@cython.cclass
class PoolTransactionManager(PoolConnectionManager):
    """The Context Manager for a PoolConnection in transaction mode."""

    def __init__(self, pool: Pool):
        """The Context Manager for a PoolConnection.

        :param pool `<'Pool'>`: The pool to manage the connection.
        """
        self._pool = pool
        self._sync_conn = None
        self._async_conn = None

    # Sync ---------------------------------------------------------------------------------------
    def __enter__(self) -> PoolSyncConnection:
        self._sync_conn = self._acquire_sync_conn()
        try:
            self._sync_conn.begin()
        except:  # noqa
            self._release_sync_conn(False)
            raise
        return self._sync_conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Encounter error
        if exc_val is not None:
            self._release_sync_conn(True)
            raise exc_val

        # Try commit transaction
        try:
            # exit: commit successfully
            self._sync_conn.commit()
            self._release_sync_conn(False)
        except:
            # fail to commit
            self._release_sync_conn(True)
            raise

    # Async --------------------------------------------------------------------------------------
    async def __aenter__(self) -> PoolConnection:
        self._async_conn = await self._acquire_async_conn()
        try:
            await self._async_conn.begin()
        except:  # noqa
            await self._release_async_conn(False)
            raise
        return self._async_conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Encounter error
        if exc_val is not None:
            await self._release_async_conn(True)
            raise exc_val

        # Try commit transaction
        try:
            # exit: commit successfully
            await self._async_conn.commit()
            await self._release_async_conn(False)
        except:  # noqa
            # fail to commit
            await self._release_async_conn(True)
            raise


@cython.cclass
class Pool:
    """Represents the pool that manages and maintains connections to the server."""

    # Sync Connection
    _sync_conn: PoolSyncConnection
    # Pool
    # . counting
    _acqr: cython.uint
    _free: cython.uint
    # . internal
    _min_size: cython.uint
    _max_size: cython.uint
    _recycle: cython.longlong
    _id: cython.Py_ssize_t
    _free_conns: deque[PoolConnection]
    _used_conns: set[PoolConnection]
    _loop: AbstractEventLoop
    _condition: Condition
    _closing: cython.bint
    _closed: cython.bint
    # . server
    _server_protocol_version: cython.int
    _server_info: str
    _server_version: tuple[int]
    _server_version_major: cython.int
    _server_vendor: str
    _server_auth_plugin_name: str
    # Connection
    # . basic
    _host: str
    _port: object  # uint
    _user: bytes
    _password: bytes
    _database: bytes
    # . charset
    _charset: Charset
    # . timeouts
    _connect_timeout: object  # uint
    _read_timeout: object  # uint | None
    _write_timeout: object  # uint | None
    _wait_timeout: object  # uint | None
    _interactive_timeout: object  # uint | None
    _lock_wait_timeout: object  # uint | None
    _execution_timeout: object  # uint | None
    # . client
    _bind_address: str
    _unix_socket: str
    _autocommit_mode: object  # int
    _autocommit: cython.bint
    _local_infile: object  # bool
    _max_allowed_packet: object  # uint
    _sql_mode: str
    _init_command: str
    _cursor: type[async_conn.Cursor]
    _sync_cursor: type[sync_conn.Cursor]
    _client_flag: object  # uint
    _program_name: str
    # . ssl
    _ssl_ctx: object  # ssl.SSLContext
    # . auth
    _auth_plugin: AuthPlugin
    _server_public_key: bytes
    # . decode
    _use_decimal: cython.bint  # bool
    _decode_bit: cython.bint  # bool
    _decode_json: cython.bint  # bool

    def __init__(
        self,
        host: str | None = "localhost",
        port: int = 3306,
        user: str | bytes | None = None,
        password: str | bytes | None = None,
        database: str | bytes | None = None,
        min_size: cython.int = 0,
        max_size: cython.int = 10,
        recycle: int | None = None,
        *,
        charset: str | None = "utf8mb4",
        collation: str | None = None,
        connect_timeout: int = 5,
        read_timeout: int | None = None,
        write_timeout: int | None = None,
        wait_timeout: int | None = None,
        interactive_timeout: int | None = None,
        lock_wait_timeout: int | None = None,
        execution_timeout: int | None = None,
        bind_address: str | None = None,
        unix_socket: str | None = None,
        autocommit: bool | None = False,
        local_infile: cython.bint = False,
        max_allowed_packet: int | str | None = None,
        sql_mode: str | None = None,
        init_command: str | None = None,
        cursor: (
            type[async_conn.Cursor | tuple | dict | DataFrame] | None
        ) = async_conn.Cursor,
        client_flag: int = 0,
        program_name: str | None = None,
        option_file: str | bytes | PathLike | OptionFile | None = None,
        ssl: SSL | object | None = None,
        auth_plugin: dict[str | bytes, type] | AuthPlugin | None = None,
        server_public_key: bytes | None = None,
        use_decimal: cython.bint = False,
        decode_bit: cython.bint = False,
        decode_json: cython.bint = False,
    ):
        """The pool that manages and maintains connections to the server.

        :param min_size `<'int'>`: The minimum number of active `async` connections to maintain. Defaults to `0`.
        :param max_size `<'int'>`: The maximum number of active `async` connections to maintain. Defaults to `10`.
        :param recycle `<'int/None'>`: The recycle time in seconds. Defaults to `None`.
            - If set to positive integer, the pool will automatically close
              and remove any connections idling more than the 'recycle' time.
            - If 'recycle=None' (Default), recycling is disabled.

        :param host `<'str/None'>`: The host of the server. Defaults to `'localhost'`.
        :param port `<'int'>`: The port of the server. Defaults to `3306`.
        :param user `<'str/bytes/None'>`: The username to login as. Defaults to `None`.
        :param password `<'str/bytes/None'>`: The password for login authentication. Defaults to `None`.
        :param database `<'str/bytes/None'>`: The default database to use by the connection. Defaults to `None`.
        :param charset `<'str/None'>`: The character set for the connection. Defaults to `'utf8mb4'`.
        :param collation `<'str/None'>`: The collation for the connection. Defaults to `None`.
        :param connect_timeout `<'int'>`: Set timeout (in seconds) for establishing the connection. Defaults to `5`.
        :param read_timeout `<'int/None>`: Set SESSION 'net_read_timeout' (in seconds). Defaults to `None` (use GLOBAL settings).
        :param write_timeout `<'int/None>`: Set SESSION 'net_write_timeout' (in seconds). Defaults to `None` (use GLOBAL settings).
        :param wait_timeout `<'int/None>`: Set SESSION 'wait_timeout' (in seconds). Defaults to `None` (use GLOBAL settings).
        :param interactive_timeout `<'int/None>`: Set SESSION 'interactive_timeout' (in seconds). Defaults to `None` (use GLOBAL settings).
        :param lock_wait_timeout `<'int/None>`: Set SESSION 'innodb_lock_wait_timeout' (in seconds). Defaults to `None` (use GLOBAL settings).
        :param execution_timeout `<'int/None>`: Set SESSION 'max_execution_time' (in milliseconds). Defaults to `None` (use GLOBAL settings).
        :param bind_address `<'str/None'>`: The interface from which to connect to the host. Accept both hostname or IP address. Defaults to `None`.
        :param unix_socket `<'str/None'>`: The unix socket for establishing connection rather than TCP/IP. Defaults to `None`.
        :param autocommit `<'bool/None'>`: The autocommit mode for the connection. `None` means use server default. Defaults to `False`.
        :param local_infile `<'bool'>`: Enable/Disable LOAD DATA LOCAL command. Defaults to `False`.
        :param max_allowed_packet `<'int/str/None'>`: The max size of packet sent to server in bytes. Defaults to `None` (16MB).
        :param sql_mode `<'str/None'>`: The default SQL_MODE for the connection. Defaults to `None`.
        :param init_command `<'str/None'>`: The initial SQL statement to run when connection is established. Defaults to `None`.
        :param cursor `<'type[Cursor]/None'>`: The default cursor type (class) to use. Defaults to `<'Cursor'>`.
            Also accepts: 'tuple' => 'Cursor' / 'dict' => 'DictCursor' / 'DataFrame' => 'DfCursor'.

        :param client_flag `<'int'>`: Custom flags to sent to server, see 'constants.CLIENT'. Defaults to `0`.
        :param program_name `<'str/None'>`: The program name for the connection. Defaults to `None`.
        :param option_file `<'OptionFile/PathLike/None>`: The MySQL option file to load connection parameters. Defaults to `None`.
            - Recommand use <'OptionFile'> to load MySQL option file.
            - If passed str/bytes/PathLike argument, it will be automatically converted
              to <'OptionFile'>, with option group defaults to 'client'.

        :param ssl `<'SSL/ssl.SSLContext/None'>`: The SSL configuration for the connection. Defaults to `None`.
            - Supports both <'SSL'> or pre-configured <'ssl.SSLContext'> object.

        :param auth_plugin `<'AuthPlugin/dict/None'>`: The authentication plugins handlers. Defaults to `None`.
            - Recommand use <'AuthPlugin'> to setup MySQL authentication plugin handlers.
            - If passed dict argument, it will be automatically converted to <'AuthPlugin'>.

        :param server_public_key `<'bytes/None'>`: The public key for the server authentication. Defaults to `None`.
        :param use_decimal `<'bool'>`: If `True` use <'Decimal'> to represent DECIMAL column data, else use <'float'>. Defaults to `False`.
        :param decode_bit `<'bool'>`: If `True` decode BIT column data to <'int'>, else keep as original bytes. Defaults to `False`.
        :param decode_json `<'bool'>`: If `True` deserialize JSON column data, else keep as original json string. Defaults to `False`.
        """
        # Sync Connection
        self._sync_conn = None

        # Pool Args
        self._setup(min_size, max_size, recycle)

        # Connection Args
        # . option file
        if option_file is not None:
            if isinstance(option_file, OptionFile):
                _opt: OptionFile = option_file
            elif isinstance(option_file, (str, bytes, PathLike)):
                _opt: OptionFile = OptionFile(option_file)
            else:
                raise errors.InvalidOptionFileError(
                    "Invalid 'option_file' argument: %r.\n"
                    "Please use <class 'OptionFile'> from the package "
                    "to load MySQL option file." % option_file
                )
            if _opt._host is not None:
                host = _opt._host
            if _opt._port >= 0:
                port = _opt._port
            if _opt._user is not None:
                user = _opt._user
            if _opt._password is not None:
                password = _opt._password
            if _opt._database is not None:
                database = _opt._database
            if _opt._charset is not None:
                charset = _opt._charset
            if _opt._bind_address is not None:
                bind_address = _opt._bind_address
            if _opt._unix_socket is not None:
                unix_socket = _opt._unix_socket
            if _opt._max_allowed_packet is not None:
                max_allowed_packet = _opt._max_allowed_packet
            if _opt._ssl is not None:
                ssl = _opt._ssl

        # fmt: off
        # . charset
        self._charset = utils.validate_charset(charset, collation, utils.DEFUALT_CHARSET)
        encoding: cython.pchar = self._charset._encoding_c
        # . basic
        self._host = utils.validate_arg_str(host, "host", "localhost")
        self._port = utils.validate_arg_uint(port, "port", 1, 65_535)
        self._user = utils.validate_arg_bytes(user, "user", encoding, utils.DEFAULT_USER)
        self._password = utils.validate_arg_bytes(password, "password", b"latin1", "")
        self._database = utils.validate_arg_bytes(database, "database", encoding, None)
        # . timeouts
        self._connect_timeout = utils.validate_arg_uint(
            connect_timeout, "connect_timeout", 1, utils.MAX_CONNECT_TIMEOUT)
        self._read_timeout = utils.validate_arg_uint(read_timeout, "read_timeout", 1, UINT_MAX)
        self._write_timeout = utils.validate_arg_uint(write_timeout, "write_timeout", 1, UINT_MAX)
        self._wait_timeout = utils.validate_arg_uint(wait_timeout, "wait_timeout", 1, UINT_MAX)
        self._interactive_timeout = utils.validate_arg_uint(interactive_timeout, "interactive_timeout", 1, UINT_MAX)
        self._lock_wait_timeout = utils.validate_arg_uint(lock_wait_timeout, "lock_wait_timeout", 1, UINT_MAX)
        self._execution_timeout = utils.validate_arg_uint(execution_timeout, "execution_timeout", 1, UINT_MAX)
        # . client
        self._bind_address = utils.validate_arg_str(bind_address, "bind_address", None)
        self._unix_socket = utils.validate_arg_str(unix_socket, "unix_socket", None)
        self._autocommit_mode = utils.validate_autocommit(autocommit)
        self._local_infile = local_infile
        self._max_allowed_packet = utils.validate_max_allowed_packet(
            max_allowed_packet, utils.DEFALUT_MAX_ALLOWED_PACKET, utils.MAXIMUM_MAX_ALLOWED_PACKET)
        self._sql_mode = utils.validate_sql_mode(sql_mode)
        self._init_command = utils.validate_arg_str(init_command, "init_command", None)
        self._cursor = validate_async_cursor(cursor)
        self._sync_cursor = validate_sync_cursor(cursor)
        self._client_flag = utils.validate_arg_uint(client_flag, "client_flag", 0, UINT_MAX)
        self._program_name = utils.validate_arg_str(program_name, "program_name", None)
        # . ssl
        self._ssl_ctx = utils.validate_ssl(ssl)
        # fmt: on
        # . auth
        if auth_plugin is not None:
            if isinstance(auth_plugin, AuthPlugin):
                self._auth_plugin = auth_plugin
            elif isinstance(auth_plugin, dict):
                self._auth_plugin = AuthPlugin(auth_plugin)
            else:
                raise errors.InvalidAuthPluginError(
                    "Invalid 'auth_plugin' argument: %r.\n"
                    "Please use <class 'AuthPlugin'> from the package "
                    "to setup MySQL authentication plugin handlers." % auth_plugin
                )
        else:
            self._auth_plugin = None
        self._server_public_key = utils.validate_arg_bytes(
            server_public_key, "server_public_key", b"ascii", None
        )
        # . decode
        self._use_decimal = use_decimal
        self._decode_bit = decode_bit
        self._decode_json = decode_json

    # Setup -----------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _setup(
        self,
        min_size: cython.int,
        max_size: cython.int,
        recycle: object,
    ) -> cython.bint:
        """(internal) Setup the pool.

        :param min_size `<'int'>`: The minimum number of active `async` connections to maintain.
        :param max_size `<'int'>`: The maximum number of active `async` connections to maintain.
        :param recycle `<'int/None'>`: The recycle time in seconds.
        """
        # . counting
        self._acqr = 0
        self._free = 0
        # . internal
        if min_size < 0:
            raise errors.InvalidPoolArgsError(
                "Invalid minimum pool size '%s', must be greater than 0." % min_size
            )
        self._min_size = min_size
        if max_size < max(1, min_size):
            raise errors.InvalidPoolArgsError(
                "Invalid maximum pool size %s', must be greater than %d."
                % (max_size, max(1, min_size))
            )
        self._max_size = max_size
        if recycle is None:
            self._recycle = -1
        else:
            try:
                self._recycle = int(recycle)
            except Exception as err:
                raise errors.InvalidPoolArgsError(
                    "Invalid recycle time: %r." % recycle
                ) from err
            if self._recycle < -1:
                self._recycle = -1
        self._id = id(self)
        self._free_conns = deque(maxlen=self._max_size)
        self._used_conns = set()
        self._loop = None
        self._condition = Condition()
        self._closing = False
        self._closed = True
        # . server
        self._server_protocol_version = -1
        self._server_info = None
        self._server_version = None
        self._server_version_major = -1
        self._server_vendor = None
        self._server_auth_plugin_name = None

    # Property --------------------------------------------------------------------------------
    # . client
    @property
    def host(self) -> str:
        """The host of the server `<'str'>`."""
        return self._host

    @property
    def port(self) -> int:
        """The port of the server `<'int'>`."""
        return self._port

    @property
    def user(self) -> str | None:
        """The username to login as `<'str/None'>`."""
        if self._user is None:
            return None
        return utils.decode_bytes(self._user, self._charset._encoding_c)

    @property
    def password(self) -> str:
        """The password for login authentication. `<'str'>`."""
        return utils.decode_bytes_latin1(self._password)

    @property
    def database(self) -> str | None:
        """The default database to use by the connections. `<'str/None'>`."""
        if self._database is None:
            return None
        return utils.decode_bytes(self._database, self._charset._encoding_c)

    @property
    def charset(self) -> str:
        """The 'CHARSET' of the connections `<'str'>`."""
        return self._charset._name

    @property
    def collation(self) -> str:
        """The 'COLLATION' of the connections `<'str'>`."""
        return self._charset._collation

    @property
    def encoding(self) -> str:
        """The 'encoding' of the connections `<'str'>`."""
        return utils.decode_bytes_ascii(self._charset._encoding)

    @property
    def connect_timeout(self) -> int:
        """Timeout in seconds for establishing the connections `<'int'>`."""
        return self._connect_timeout

    @property
    def bind_address(self) -> str | None:
        """The interface from which to connect to the host `<'str/None'>`."""
        return self._bind_address

    @property
    def unix_socket(self) -> str | None:
        """The unix socket (rather than TCP/IP) for establishing the connections `<'str/None'>`."""
        return self._unix_socket

    @property
    def autocommit(self) -> bool | None:
        """The default 'autocommit' mode of the pool. All acquired
        connections will comply to this setting `<'bool/None'>`.

        `None` means pool is not connected and use server default.
        """
        if not self.closed():
            return self._autocommit
        if self._autocommit_mode == -1:
            return None
        return bool(self._autocommit_mode)

    @property
    def local_infile(self) -> bool:
        """Whether LOAD DATA LOCAL command is enabled `<'bool'>`."""
        return self._local_infile

    @property
    def max_allowed_packet(self) -> int:
        """The max size of packet sent to server in bytes `<'int'>`."""
        return self._max_allowed_packet

    @property
    def sql_mode(self) -> str | None:
        """The default SQL_MODE for the connections `<'str/None'>`."""
        return self._sql_mode

    @property
    def init_command(self) -> str | None:
        """The 'init_command' to be executed after connecting `<'str/None'>`."""
        return self._init_command

    @property
    def client_flag(self) -> int:
        """The initial SQL statement to run when connections are established `<'str/None'>`."""
        return self._client_flag

    @property
    def ssl(self) -> object | None:
        """The 'ssl.SSLContext' for the connections `<'SSLContext/None'>`."""
        return self._ssl_ctx

    @property
    def auth_plugin(self) -> AuthPlugin | None:
        """The authentication plugins handlers `<'AuthPlugin/None'>`."""
        return self._auth_plugin

    # . server
    @property
    def protocol_version(self) -> int | None:
        """The protocol version of the server `<'int/None'>`."""
        if self._server_protocol_version == -1:
            return None
        return self._server_protocol_version

    @property
    def server_info(self) -> str | None:
        """The server information (name & version) `<'str/None'>`."""
        return self._server_info

    @property
    def server_version(self) -> tuple[int] | None:
        """The server version `<'tuple[int]/None'>`.
        >>> (8, 0, 23)  # example
        """
        return self._server_version

    @property
    def server_version_major(self) -> int | None:
        """The server major version `<'int/None'>`.
        >>> 8  # example
        """
        if self._server_version_major == -1:
            return None
        return self._server_version_major

    @property
    def server_vendor(self) -> Literal["mysql", "mariadb"] | None:
        """The name of the server vendor (database type) `<'str/None'>`."""
        return self._server_vendor

    @property
    def server_auth_plugin_name(self) -> str | None:
        """The authentication plugin name of the server `<'str/None'>`."""
        return self._server_auth_plugin_name

    # . decode
    @property
    def use_decimal(self) -> bool:
        """The default decode behavior of the pool: whether to
        use <'DECIMAL'> to represent DECIMAL column data `<'bool'>`.

        If `False`, use <'float'> instead. All acquired
        connections will comply to this settings.
        """
        return self._use_decimal

    @property
    def decode_bit(self) -> bool:
        """The default decode behavior of the pool: whether
        to decode BIT column data to integer `<'bool'>`.

        If `False`, keep as the original bytes. All acq
        uired connections will comply to this settings.
        """
        return self._decode_bit

    @property
    def decode_json(self) -> bool:
        """The default decode behavior of the pool: whether
        to deserialize JSON column data `<'bool'>`.

        If `False`, keep as the original JSON string. All
        acquired connections will comply to this settings.
        """
        return self._decode_json

    # . pool
    @property
    def free(self) -> int:
        """The number of free `async` connections in the pool `<'int'>`."""
        return self._free

    @property
    def used(self) -> int:
        """The number of `async` connections that are in use `<'int'>`."""
        return self._get_used()

    @property
    def total(self) -> int:
        """The total number of `async` connections in the pool `<'int'>`."""
        return self._get_total()

    @property
    def min_size(self) -> int:
        """The minimum number of active `async` connections to maintain `<'int'>`."""
        return self._min_size

    @property
    def max_size(self) -> int:
        """The maximum number of active `async` connections to maintain `<'int'>`."""
        return self._max_size

    @property
    def recycle(self) -> int | None:
        """The recycle time in seconds `<int/None>`.

        Any connections idling more than the 'recycle' time
        will be closed and removed from the pool.
        """
        return None if self._recycle == -1 else self._recycle

    # Pool ------------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_min_size(self, size: cython.uint) -> cython.bint:
        """Change the minimum number of active `async` connections to maintain by the pool.

        :param size `<'int'>`: New minium pool size.
        """
        if size > self._max_size:
            raise errors.InvalidPoolArgsError(
                "Minimum pool size '%d' must be less than maximum pool size '%d'."
                % (size, self._max_size)
            )
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_recycle(self, recycle: int | None) -> cython.bint:
        """Change the recycle time.

        :param recycle `<'int/None'>`: New recycle time in seconds.
            - If set to any positive integer, the pool will automatically close
              and remove any connections idling more than the 'recycle' time.
            - Set to `None` to disable recycling.
        """
        if recycle is None:
            self._recycle = -1
        else:
            try:
                self._recycle = int(recycle)
            except Exception as err:
                raise errors.InvalidPoolArgsError(
                    "Invalid recycle time: %r." % recycle
                ) from err
            if self._recycle < -1:
                self._recycle = -1
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_autocommit(self, value: cython.bint) -> cython.bint:
        """Set the default 'autocommit' mode of the pool. All acquired
        connections will comply to this setting.

        :param value `<'bool'>`: Enable/Disable autocommit.
            - `True` to operate in autocommit (non-transactional) mode.
            - `False` to operate in manual commit (transactional) mode.
        """
        self._autocommit_mode = int(value)
        if not self.closed():
            self._autocommit = value
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_use_decimal(self, value: cython.bint) -> cython.bint:
        """Set the default decode behavior of the pool: whether to
        use `<'DECIMAL'> to represent DECIMAL column data. All
        acquired connections will comply to this setting.

        :param value `<'bool'>`: True to use <'DECIMAL>', else <'float'>.
        """
        self._use_decimal = value
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_decode_bit(self, value: cython.bint) -> cython.bint:
        """Set the default decode behavior of the pool: whether to
        decode BIT column data to integer. All acquired connections
        will comply to this settings.

        :param value `<'bool'>`: True to decode BIT column, else keep as original bytes.
        """
        self._decode_bit = value
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_decode_json(self, value: cython.bint) -> cython.bint:
        """Set the default decode behavior of the pool: whether to
        deserialize JSON column data. All acquired connections will
        comply to this settings.

        :param value `<'bool'>`: True to deserialize JSON column, else keep as original JSON string.
        """
        self._decode_json = value
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _add_free_conn(self, conn: PoolConnection) -> cython.bint:
        """(internal) Add back a free `async` connection, and update counting."""
        self._free_conns.append(conn)
        self._free += 1
        return True

    @cython.cfunc
    @cython.inline(True)
    def _get_free_conn(self) -> PoolConnection:
        """(internal) Get a free `async` connection, and update counting `<'PoolConnection/None'>`."""
        try:
            conn: PoolConnection = self._free_conns.popleft()
            if self._free > 0:
                self._free -= 1
            return conn
        except:  # noqa
            self._free = 0
            return None

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_free(self) -> cython.uint:
        """(internal) Get the number of free `async` connections in the pool `<'int'>`."""
        return self._free

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_used(self) -> cython.uint:
        """(internal) Get the number of `async` connections that are in use `<'int'>`."""
        if self._used_conns is None:
            return 0
        return set_len(self._used_conns)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_total(self) -> cython.uint:
        """(internal) Get the total number of `async` connections in the pool `<'int'>`."""
        return self._acqr + self._free + self._get_used()

    @cython.cfunc
    @cython.inline(True)
    def _get_loop(self) -> object:
        """(internal) Get async event loop `<'AbstractEventLoop'>`'"""
        if self._loop is None:
            self._loop = _get_event_loop()
        return self._loop

    # Acquire / Transaction / Fill / Release --------------------------------------------------
    @cython.ccall
    def acquire(self) -> PoolConnectionManager:
        """Acquire a free connection from the pool through context manager `<'PoolConnectionManager'>`.

        ## Notice
        - To maintain consistency, connection 'autocommit', 'used_decimal',
          'decode_bit' and 'decode_json' will be reset to pool settings
          when acquired.
        - If user changes 'charset', 'read_timeout', 'write_timeout',
          'wait_timeout', 'interactive_timeout', 'lock_wait_timeout'
          and 'exeuction_timeout' through connection built-in 'set_*()'
          methods, these settings will also be reset to pool defaults
          at release.
        - For connection consistency, other changes made on the connection
          [SESSION] (especially through SQL queries), please manually call
          'conn.schedule_close()' method before releasing back to the pool.

        ## Example (sync):
        >>> with pool.acquire() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM table")

        ## Example (async):
        >>> async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT * FROM table")

        ## Example (direct async - NOT recommended):
        >>> conn = await pool.acquire()
            async with conn.cursor() as cur:
                await cur.execute("SELECT * FROM table")
            await pool.release(conn)  # manual release
        """
        return PoolConnectionManager(self)

    @cython.ccall
    def transaction(self) -> PoolTransactionManager:
        """Acquire a free connection from the pool in `TRANSACTION` mode
        through context manager `<'PoolTransactionManager'>`.

        ## Explanation
        By acquiring connection through this method, the following happens:
        - 1. Acquire a free connection from the Pool.
        - 2. Use the connection to `BEGIN` a transaction.
        - 3. Returns the connection.
        - 4a. If catches ANY exceptions during the transaction, close and release the connection by the Pool.
        - 4b. If the transaction executed successfully, execute `COMMIT` and release the connection back to the Pool.

        ## Notice
        - Do `NOT` use 'conn.transaction()' to restart a TRANSACTION,
          only use the 'conn.cursor()' method to acquire the cursor.
        - To maintain consistency, connection 'autocommit', 'used_decimal',
          'decode_bit' and 'decode_json' will be reset to pool settings
          when acquired.
        - If user changes 'charset', 'read_timeout', 'write_timeout',
          'wait_timeout', 'interactive_timeout', 'lock_wait_timeout'
          and 'exeuction_timeout' through connection built-in 'set_*()'
          methods, these settings will also be reset to pool defaults
          at release.
        - For connection consistency, other changes made on the connection
          [SESSION] (especially through SQL queries), please manually call
          'conn.schedule_close()' method before releasing back to the pool.

        ## Example (sync):
        >>> with pool.transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO tb (id, name) VALUES (1, 'test')")
            # Equivalent to:
            BEGIN;
            INSERT INTO tb (id, name) VALUES (1, 'test');
            COMMIT;

        ## Example (async):
        >>> async with pool.transaction() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("INSERT INTO tb (id, name) VALUES (1, 'test')")
            # Equivalent to:
            BEGIN;
            INSERT INTO tb (id, name) VALUES (1, 'test');
            COMMIT;
        """
        return PoolTransactionManager(self)

    async def fill(self, num: cython.int = 1) -> None:
        """Fill the pool with specific number of new `async` connections.

        :param num `<'int'>`: The number of new `async` connections to fill. Defaults to `1`.
            - If the number plus the existing `async` connections exceeds the
              maximum pool size, the pool will only fill to the 'max_size' limit.
            - If 'num=-1', the pool will fill to the 'min_size'.
        """
        if num == 0 or num < -1 or self._closing:
            return None

        async with self._condition:
            total: cython.uint = self._get_total()
            if total >= self._max_size:
                return None

            n: cython.uint
            if num == -1:
                if self._free >= self._min_size:
                    return None
                n = self._min_size - self._free
            else:
                n = num
                n = min(n, self._max_size - total)

            await _gather(*[self._fill_async_new() for _ in range(n)])
            self._condition.notify()

    @cython.ccall
    def release(self, conn: PoolConnection | PoolSyncConnection) -> object:
        """Release a connection back to the pool `<'Future'>`.
        This method is `NOT` needed if the connection is acquired through the
        'acquire()' or 'transaction()' methods. The context manager will
        automatically release the connection back to the pool.

        :param conn `<'PoolConnection/PoolSyncConnection'>`: The pool connection to release.
        :raises `<'PoolReleaseError'>`: If the connection does not belong to this pool.

        ## Notice
        - For `async` connection, please await for the result: `await pool.release(conn)`.
        - To maintain consistency, connection 'autocommit', 'used_decimal',
          'decode_bit' and 'decode_json' will be reset to pool settings
          when acquired.
        - If user changes 'charset', 'read_timeout', 'write_timeout',
          'wait_timeout', 'interactive_timeout', 'lock_wait_timeout'
          and 'exeuction_timeout' through connection built-in 'set_*()'
          methods, these settings will also be reset to pool defaults
          at release.
        - For connection consistency, other changes made on the connection
          [SESSION] (especially through SQL queries), please manually call
          'conn.schedule_close()' method before releasing back to the pool.
        """
        # Async connection
        loop: AbstractEventLoop = self._get_loop()
        if isinstance(conn, PoolConnection):
            return loop.create_task(self._release_async_conn(conn))

        # Sync connection
        if isinstance(conn, PoolSyncConnection):
            self._release_sync_conn(conn)
            fut = loop.create_future()
            fut.set_result(None)
            return fut

        # Invalid connection
        raise errors.PoolReleaseError(
            "Invalid connection: %s %r.\n"
            "Must be an instance of <'PoolConnection/PoolSyncConnection'>."
            % (type(conn), conn)
        )

    # (Sync) Acquire / Release ----------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _acquire_sync_conn(self) -> PoolSyncConnection:
        """(internal) Acquired `sync` connection `<'PoolSyncConnection/None'>`.

        ### This method returns `None` only when the pool is closing.
        """
        # Pool is closing
        if self._closing:
            return None

        # Create new connection
        conn: PoolSyncConnection = self._sync_conn
        if conn is None:
            conn = PoolSyncConnection(
                self._id,
                self._host,
                self._port,
                self._user,
                self._password,
                self._database,
                self._charset,
                self._connect_timeout,
                self._read_timeout,
                self._write_timeout,
                self._wait_timeout,
                self._interactive_timeout,
                self._lock_wait_timeout,
                self._execution_timeout,
                self._bind_address,
                self._unix_socket,
                self._autocommit_mode if self.closed() else int(self._autocommit),
                self._local_infile,
                self._max_allowed_packet,
                self._sql_mode,
                self._init_command,
                self._sync_cursor,
                self._client_flag,
                self._program_name,
                self._ssl_ctx,
                self._auth_plugin,
                self._server_public_key,
                self._use_decimal,
                self._decode_bit,
                self._decode_json,
            )
            conn.connect()
            # . initial setup
            if self._closed:
                self._autocommit = conn.get_autocommit()
                self._server_protocol_version = conn._server_protocol_version
                self._server_info = conn._server_info
                self._server_version = conn.get_server_version()
                self._server_version_major = conn._server_version_major
                self._server_vendor = conn.get_server_vendor()
                self._server_auth_plugin_name = conn._server_auth_plugin_name
                self._closed = False  # change state
            # . assign connection
            self._sync_conn = conn

        # Check connection
        else:
            # . connection already closed
            if conn.closed():
                self._sync_conn = None
                return self._acquire_sync_conn()
            # . should be recycled
            idle_time: cython.double = unix_time() - conn._last_used_time
            if self._recycle != -1 and idle_time > self._recycle:
                conn.close()
                self._sync_conn = None
                return self._acquire_sync_conn()
            # . ping connection
            if idle_time > 60:
                conn.ping(True)
            # . reset autocommit
            if conn.get_autocommit() != self._autocommit:
                conn.set_autocommit(self._autocommit)
            # . reset decode
            conn._use_decimal = self._use_decimal
            conn._decode_bit = self._decode_bit
            conn._decode_json = self._decode_json

        # return connection
        return conn

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _release_sync_conn(self, conn: PoolSyncConnection) -> cython.bint:
        """(internal) Release the `sync` connection back to the pool.

        ### If the connection belongs to the pool, this method does not raise any error.
        """
        # Validate pool identity
        if conn._pool_id != self._id:
            raise errors.PoolReleaseError(
                "Cannot release connection that does not belong to the pool."
            )

        # Scheduled close / in transaction / closing
        if conn._close_scheduled or conn.get_transaction_status() or self._closing:
            conn.close()
            return self._close_sync_conn()

        # Reset connection
        # . charset
        if conn._charset_changed:
            try:
                conn.set_charset(self._charset._name, self._charset._collation)
                conn._charset_changed = False
            except Exception:
                conn.close()
                return self._close_sync_conn()
        # . read timeout
        if conn._read_timeout_changed:
            try:
                conn.set_read_timeout(None)
            except Exception:
                conn.close()
                return self._close_sync_conn()
        # . write timeout
        if conn._write_timeout_changed:
            try:
                conn.set_write_timeout(None)
            except Exception:
                conn.close()
                return self._close_sync_conn()
        # . wait timeout
        if conn._wait_timeout_changed:
            try:
                conn.set_wait_timeout(None)
            except Exception:
                conn.close()
                return self._close_sync_conn()
        # . interactive timeout
        if conn._interactive_timeout_changed:
            try:
                conn.set_interactive_timeout(None)
            except Exception:
                conn.close()
                return self._close_sync_conn()
        # . lock wait timeout
        if conn._lock_wait_timeout_changed:
            try:
                conn.set_lock_wait_timeout(None)
            except Exception:
                conn.close()
                return self._close_sync_conn()
        # . execution timeout
        if conn._execution_timeout_changed:
            try:
                conn.set_execution_timeout(None)
            except Exception:
                conn.close()
                return self._close_sync_conn()

        # Reclaim connection (normally should not happen)
        if self._sync_conn is None:
            self._sync_conn = conn

        return True

    # (Async) Acquire / Release ---------------------------------------------------------------
    async def _acquire_async_conn(self) -> PoolConnection | None:
        """(internal) Acquire a `async` connection from the pool `<'PoolConnection/None'>`.

        ### This method returns `None` only when the pool is closing.
        """
        async with self._condition:
            while True:
                # Fill to minimum size
                await self._fill_async_min()

                # Acquire free connection
                conn = await self._acquire_async_free()
                if conn is not None:
                    return conn

                # Acquire new connection
                conn = await self._acquire_async_new()
                if conn is not None:
                    return conn

                # Pool is closing
                if self._closing:
                    # eixt: pool is closing, return None
                    return await self.wait_for_closed()

                # Wait for notification
                await self._condition.wait()

    async def _acquire_async_free(self) -> PoolConnection | None:
        """(internal) Acquire an existing free `async` connection from the pool `<'PoolConnection/None'>`.

        This method only tries to acquire existing free connections
        in FIFO (First-In-First-Out) order. Recycling and socket
        'eof' checks are performed here, and invalid connections
        are closed and removed. Also, 'autocommit', 'used_decimal',
        'decode_bit' and 'decode_json' will be reset to pool settings.

        ### This method returns `None` if no free connection available.
        """
        while self._free > 0 and not self._closing:
            conn = self._get_free_conn()
            if conn is None:
                return None
            # Connection already closed
            if conn.closed():
                continue
            # Should be recycled
            if (
                self._recycle != -1
                and unix_time() - conn._last_used_time > self._recycle
            ):
                await conn.close()
                continue
            # Invalid connection
            reader = conn._reader
            if reader._eof or reader.exception() is not None:
                await conn.close()
                continue
            # Reset connection
            # . autocommit
            if conn.get_autocommit() != self._autocommit:
                try:
                    await conn.set_autocommit(self._autocommit)
                except Exception:
                    await conn.close()
                    continue
            # . decode
            conn._use_decimal = self._use_decimal
            conn._decode_bit = self._decode_bit
            conn._decode_json = self._decode_json
            # Track in-use
            set_add(self._used_conns, conn)
            return conn

        # No free available
        return None

    async def _acquire_async_new(self) -> PoolConnection | None:
        """(internal) Create and acquire a new `async` connection from the pool `<'PoolConnection/None'>`.

        This method creates a new connection (if max limit not reached)
        and adds it to the pool. The connection is then acquired and
        returned to the caller.

        ### This method returns `None` if pool max limit is reached.
        """
        await self._fill_async_new()
        return await self._acquire_async_free()

    async def _fill_async_min(self) -> None:
        """(internal) Fill the pool with new `async` connections up to 'min_size'."""
        if self._min_size == 0 or self._closing:
            return None

        total: cython.uint = self._get_total()
        if total < self._min_size:
            await _gather(
                *[self._fill_async_new() for _ in range(self._min_size - total)]
            )
            self._condition.notify()

    async def _fill_async_new(self) -> None:
        """(internal) Fill the pool with one new `async` connection."""
        # Limit by pool size or is closing
        if self._get_total() >= self._max_size or self._closing:
            return None

        # Create new connection
        self._acqr += 1
        conn: PoolConnection = None
        try:
            # . connect
            conn = PoolConnection(
                self._id,
                self._host,
                self._port,
                self._user,
                self._password,
                self._database,
                self._charset,
                self._connect_timeout,
                self._read_timeout,
                self._write_timeout,
                self._wait_timeout,
                self._interactive_timeout,
                self._lock_wait_timeout,
                self._execution_timeout,
                self._bind_address,
                self._unix_socket,
                self._autocommit_mode if self.closed() else int(self._autocommit),
                self._local_infile,
                self._max_allowed_packet,
                self._sql_mode,
                self._init_command,
                self._cursor,
                self._client_flag,
                self._program_name,
                self._ssl_ctx,
                self._auth_plugin,
                self._server_public_key,
                self._use_decimal,
                self._decode_bit,
                self._decode_json,
                self._get_loop(),
            )
            await conn.connect()
            # . initial setup
            if self._closed:
                self._autocommit = conn.get_autocommit()
                self._server_protocol_version = conn._server_protocol_version
                self._server_info = conn._server_info
                self._server_version = conn.get_server_version()
                self._server_version_major = conn._server_version_major
                self._server_vendor = conn.get_server_vendor()
                self._server_auth_plugin_name = conn._server_auth_plugin_name
                self._closed = False  # change state
            # . add to pool
            self._add_free_conn(conn)
        except:  # noqa
            if conn is not None:
                await conn.close()
            raise
        finally:
            self._acqr -= 1

    async def _release_async_conn(self, conn: PoolConnection) -> None:
        """(internal) Release an `async` connection back to the pool.

        ### If the connection belongs to the pool, this method does not raise any error.
        """

        async def notify() -> None:
            """Notify the pool that a connection has been released."""
            async with self._condition:
                self._condition.notify()

        # Validate pool identity
        if conn._pool_id != self._id:
            raise errors.PoolReleaseError(
                "Cannot release connection that does not belong to the pool."
            )

        # Remove from 'used_conns'
        set_discard(self._used_conns, conn)

        # Connection already closed
        if conn.closed():
            return await notify()

        # Scheduled close / in transaction / closing
        if conn._close_scheduled or conn.get_transaction_status() or self._closing:
            await conn.close()
            return await notify()

        # Reset connection
        # . charset
        if conn._charset_changed:
            try:
                await conn.set_charset(self._charset._name, self._charset._collation)
                conn._charset_changed = False
            except Exception:
                await conn.close()
                return await notify()
        # . read timeout
        if conn._read_timeout_changed:
            try:
                await conn.set_read_timeout(None)
            except Exception:
                await conn.close()
                return await notify()
        # . write timeout
        if conn._write_timeout_changed:
            try:
                await conn.set_write_timeout(None)
            except Exception:
                await conn.close()
                return await notify()
        # . wait timeout
        if conn._wait_timeout_changed:
            try:
                await conn.set_wait_timeout(None)
            except Exception:
                await conn.close()
                return await notify()
        # . interactive timeout
        if conn._interactive_timeout_changed:
            try:
                await conn.set_interactive_timeout(None)
            except Exception:
                await conn.close()
                return await notify()
        # . lock wait timeout
        if conn._lock_wait_timeout_changed:
            try:
                await conn.set_lock_wait_timeout(None)
            except Exception:
                await conn.close()
                return await notify()
        # . execution timeout
        if conn._execution_timeout_changed:
            try:
                await conn.set_execution_timeout(None)
            except Exception:
                await conn.close()
                return await notify()

        # Back to pool
        self._add_free_conn(conn)
        return await notify()

    # Close -----------------------------------------------------------------------------------
    @cython.ccall
    def close(self) -> object:
        """Close the pool, and returns 'Pool.wait_for_closed()' `<'Future'>`.

        - Set pool 'closing' flag to `True`, so no free/new connections
          can be acquired, and all in-use connections will be closed at
          release.
        - Returns the 'Future' object wraps the 'Pool.wait_for_closed()'
          coroutine, which will close all free connections and `WAIT`
          for the return and release of all the in-use connections'.
        - If user only acquires `sync` connection and the pool is empty,
          the 'Future' object can be ignored.
        - If user acquires any `async` connections, please await of the 'Future'
          object or manual await 'Pool.wait_for_closed()' method.

        ### This method does not raise any error.

        ## Example (await closure):
        >>> async with pool.acquire() as conn:
                # some SQL queries
                ...
            await pool.close()  # closing & await for closure

        ## Example (manual closure):
        >>> async with pool.acquire() as conn:
                # some SQL queries
                ...
            pool.close()  # set 'closing' flag
            # some more task
            ...
            await pool.wait_for_closed()  # manual closure
        """
        loop: AbstractEventLoop = self._get_loop()

        # Pool already closed
        if self.closed():
            fut = loop.create_future()
            fut.set_result(None)
            return fut

        # Close sync connection
        self._close_sync_conn()

        # Set closing flag
        self._closing = True

        # Wait for closed (Future)
        return loop.create_task(self.wait_for_closed())

    async def wait_for_closed(self) -> None:
        """Wait for the pool to be closed.

        - Close all free connections.
        - Wait for all in-use connections to return and be released (closed).

        ### This method should only be called after 'Pool.close()'.

        :raises `<'PoolNotClosedError'>`: If called before 'Pool.close()'. Otherwise, this method does not raise any error.
        """
        # Pool already closed
        if self.closed():
            return None
        if not self._closing:
            raise errors.PoolNotClosedError(
                "'Pool.wait_for_closed()' should only be "
                "called after 'Pool.close()'."
            )

        async with self._condition:
            # Close all free connection
            while True:
                conn = self._get_free_conn()
                if conn is None:
                    break
                await conn.close()

            # Wait for used connection
            while self._get_total() > 0:
                await self._condition.wait()

            # Set closed flag
            self._closed = True

            # Notify
            self._condition.notify()

    async def clear(self) -> None:
        """Clear (close) existing free connections in the pool.

        ### This method does not affect in-use connections.
        """
        self._close_sync_conn()
        async with self._condition:
            while True:
                conn = self._get_free_conn()
                if conn is None:
                    break
                await conn.close()
            self._condition.notify()

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def terminate(self) -> cython.bint:
        """Terminate the pool immediately.

        - Set 'closing' flag to `True, so no free/new connections can be acquired.
        - Close all existing free and in-use connections immediately.

        ### This method does not raise any error.
        """
        # Pool already closed
        if self.closed():
            return True

        # Close sync connection
        self._close_sync_conn()

        # Set closing flag
        self._closing = True

        # Force close all free
        conn: PoolConnection
        while True:
            conn = self._get_free_conn()
            if conn is None:
                break
            conn.force_close()

        # Force close all used
        if self._get_used() > 0:
            for conn in self._used_conns:
                conn.force_close()
            set_clear(self._used_conns)

        # Set closed flag
        self._closed = True

        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def closed(self) -> cython.bint:
        """Represents the pool state: whether is closed `<'bool'>`."""
        return self._closed and self._sync_conn is None

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _close_sync_conn(self) -> cython.bint:
        """(internal) Close sync connection."""
        conn: PoolSyncConnection = self._sync_conn
        if conn is not None:
            conn.close()
            self._sync_conn = None
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_open(self) -> cython.bint:
        """(internal) Verify the pool is not closed.

        :raises `<'PoolClosedError'>`: If pool is already closed.
        """
        if self._closing or self.closed():
            raise errors.PoolClosedError(0, "Pool is closed.")
        return True

    # Query -----------------------------------------------------------------------------------
    @cython.ccall
    def escape_args(
        self,
        args: Any,
        many: cython.bint = False,
        itemize: cython.bint = True,
    ) -> object:
        """Escape 'args' to formatable object(s) `<'str/tuple/list[str/tuple]'>`.

        :param args `<'object'>`: The object to escape, supports:
            - Python native:
              int, float, bool, str, None, datetime, date, time,
              timedelta, struct_time, bytes, bytearray, memoryview,
              Decimal, dict, list, tuple, set, frozenset, range.
            - Library [numpy](https://github.com/numpy/numpy):
              np.int, np.uint, np.float, np.bool, np.bytes,
              np.str, np.datetime64, np.timedelta64, np.ndarray.
            - Library [pandas](https://github.com/pandas-dev/pandas):
              pd.Timestamp, pd.Timedelta, pd.DatetimeIndex,
              pd.TimedeltaIndex, pd.Series, pd.DataFrame.
            - Library [cytimes](https://github.com/AresJef/cyTimes):
              pydt, pddt.

        :param many `<'bool'>`: Wheter to escape 'args' as multi-rows. Defaults to `False`.
            * When 'many=True', the argument 'itemize' is ignored.
            * 1. sequence and mapping (e.g. `list`, `tuple`, `dict`, etc)
              escapes to `<'list[str/tuple[str]]'>`. Each element represents
              one row of the 'args'.
            * 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to
              `<'list[str]'>`. Each element represents one row of the 'args'.
            * 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes
              to `<'list[tuple[str]]'>`. Each tuple represents one row
              of the 'args' .
            * 4. Single object (such as `int`, `float`, `str`, etc) escapes
              to one literal string `<'str'>`.

        :param itemize `<'bool'>`: Whether to escape each items of the 'args' individual. Defaults to `True`.
            - When 'itemize=True', the 'args' type determines how to escape.
                * 1. Sequence or Mapping (e.g. `list`, `tuple`, `dict`, etc)
                  escapes to `<'tuple[str]'>`.
                * 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to
                  `<'tuple[str]'>`.
                * 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes
                  to `<'list[tuple[str]]'>`. Each tuple represents one row
                  of the 'args' .
                * 4. Single object (such as `int`, `float`, `str`, etc) escapes
                  to one literal string `<'str'>`.
            - When 'itemize=False', regardless of the 'args' type, all
              escapes to one single literal string `<'str'>`.

        :raises `<'EscapeTypeError'>`: If any error occurs during escaping.

        ## Returns
        - If returns a `<'str'>`, it represents a single literal string.
          The 'sql' should only have one '%s' placeholder.
        - If returns a `<'tuple'>`, it represents a single row of literal
          strings. The 'sql' should have '%s' placeholders equal to the
          tuple length.
        - If returns a `<'list'>`, it represents multiple rows of literal
          string(s). The 'sql' should have '%s' placeholders equal to the
          item count in each row.
        """
        return escape(args, many, itemize)

    # Special Methods -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            "<%s(free=%d, used=%d, total=%d, min_size=%d, max_size=%d, recycle=%s)>"
            % (
                self.__class__.__name__,
                self._free,
                self._get_used(),
                self._get_total(),
                self._min_size,
                self._max_size,
                None if self._recycle == -1 else self._recycle,
            )
        )

    def __enter__(self) -> Pool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    async def __aenter__(self) -> Pool:
        try:
            await self.fill(-1)
            return self
        except:  # noqa
            self.terminate()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __del__(self):
        self.terminate()
