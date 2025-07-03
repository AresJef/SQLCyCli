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


# Cursor --------------------------------------------------------------------------------------
@cython.ccall
def validate_sync_cursor(cursor: object) -> type:
    """Validate and map a cursor identifier to its [sync] cursor class. `<'type[Cursor]/None'>`."""
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
    """Validate and map a cursor identifier to its [async] cursor class. `<'type[Cursor]/None'>`."""
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
    """Represents the [async] socket connection to the server managed by a pool.

    - This class serves as the connection managed by a pool only.
      It does not perform argument validations during initialization.
      Such validations are delegated to the `<'aio.Pool'>` class.
    - Please do `NOT` instanciate this class directly.
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
        """The [async] socket connection to the server managed by a pool.

        - This class serves as the connection managed by a pool only.
          It does not perform argument validations during initialization.
          Such validations are delegated to the `<'aio.Pool'>` class.
        - Please do `NOT` instanciate this class directly.

        :param pool_id `<'int'>`: The unique identifier of the pool.

        - For details about other parameters, please refer to `sqlcycli.aio.connection.BaseConnection`.
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
    @property
    def close_scheduled(self) -> bool:
        """Check if the connection is scheduled for closure when released by the pool `<'bool'>`."""
        return self._close_scheduled

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def schedule_close(self) -> cython.bint:
        """Schedule the connection for closure (instead of reuse) when released by the pool."""
        self._close_scheduled = True


@cython.cclass
class PoolSyncConnection(sync_conn.BaseConnection):
    """Represents the [sync] socket connection to the server managed by a pool.

    - This class serves as the connection managed by a pool only.
      It does not perform argument validations during initialization.
      Such validations are delegated to the `<'aio.Pool'>` class.
    - Please do `NOT` instanciate this class directly.
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
        """The [sync] socket connection to the server managed by a pool.

        - This class serves as the connection managed by a pool only.
          It does not perform argument validations during initialization.
          Such validations are delegated to the `<'aio.Pool'>` class.
        - Please do `NOT` instanciate this class directly.

        :param pool_id `<'int'>`: The unique identifier of the pool.

        - For details about other parameters, please refer to `sqlcycli.connection.BaseConnection`.
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
    @property
    def close_scheduled(self) -> bool:
        """Check if the connection is scheduled for closure when released by the pool `<'bool'>`."""
        return self._close_scheduled

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def schedule_close(self) -> cython.bint:
        """Schedule the connection for closure (instead of reuse) when released by the pool."""
        self._close_scheduled = True


# Pool Connection Manager ---------------------------------------------------------------------
@cython.cclass
class PoolConnectionManager:
    """The context manager for acquiring and releasing
    a free [sync/async] connection from the pool.
    """

    _pool: Pool
    _sync_conn: PoolSyncConnection
    _async_conn: PoolConnection

    def __init__(self, pool: Pool) -> None:
        """The context manager for acquiring and releasing
        a free [sync/async] connection from the pool.

        :param pool `<'Pool'>`: The pool managing the connections.
        """
        self._pool = pool
        self._sync_conn = None
        self._async_conn = None

    # Sync ---------------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _acquire_sync_conn(self) -> PoolSyncConnection:
        """(internal) Acquire a [sync] connection from the pool `<'PoolSyncConnection'>`.

        :returns `<'PoolSyncConnection'>`: An active [sync] connection managed by the pool.
        """
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
        """(internal) Release the [sync] connection back to the pool.

        :param close `<'bool'>`: Whether to schedule the connection
            for closure instead of reuse.
        """
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
        """(internal) Acquire an [async] connection from the pool `<'PoolSyncConnection'>`.

        :returns `<'PoolSyncConnection'>`: An active [async] connection managed by the pool.
        """
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
        """(internal) Release the [async] connection back to the pool.

        :param close `<'bool'>`: Whether to schedule the connection
            for closure instead of reuse.
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
    """The context manager for acquiring and releasing a free
    [sync/async] connection from the pool in TRANSACTION mode.

    ## On enter
    - 1. Acquire a connection from the pool.
    - 2. Calls `BEGIN` on the connection

    ## On exit
    - If no exception occurs: calls `COMMIT` and releases the connection back to the pool for reuse.
    - If an exception occurs: Schedules the connection for closure and releases it back to the pool.
    """

    def __init__(self, pool: Pool):
        """The context manager for acquiring and releasing a free
        [sync/async] connection from the pool in TRANSACTION mode.

        ## On enter
        - 1. Acquire a connection from the pool.
        - 2. Calls `BEGIN` on the connection

        ## On exit
        - If no exception occurs: calls `COMMIT` and releases the connection back to the pool for reuse.
        - If an exception occurs: Schedules the connection for closure and releases it back to the pool.

        :param pool `<'Pool'>`: The pool managing the connections.
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


# Pool ----------------------------------------------------------------------------------------
@cython.cclass
class Pool:
    """Represents the pool that manages and maintains both the
    synchronize and asynchronize connections to the server.
    """

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
        """The pool that manages and maintains both the synchronize and asynchronize connections to the server.

        :param min_size `<'int'>`: The minimum number of [async] connections to maintain. Defaults to `0`.
        :param max_size `<'int'>`: The maximum number of [async] connections to maintain. Defaults to `10`.
        :param recycle `<'int/None'>`: The connection recycle time in seconds. Defaults to `None`.
            When set to a positive integer, the pool will automatically close
            and remove any connections idling more than the `recycle` time.
            Any other values disables the recycling feature.

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
        :param cursor `<'type[Cursor]/None'>`: The default cursor class (type) to use. Defaults to `<'Cursor'>`.
            Determines the data type of the fetched result set.
            Also accepts: 1. `tuple` => `Cursor`; 2. `dict` => `DictCursor`; 3. `DataFrame` => `DfCursor`;

        :param client_flag `<'int'>`: Custom flags to sent to server, see 'constants.CLIENT'. Defaults to `0`.
        :param program_name `<'str/None'>`: The program name for the connection. Defaults to `None`.
        :param option_file `<'OptionFile/PathLike/None>`: The MySQL option file to load connection parameters. Defaults to `None`.
            - Recommand use <'OptionFile'> to load MySQL option file.
            - If passed str/bytes/PathLike argument, it will be automatically converted
              to <'OptionFile'>, with option group defaults to 'client'.

        :param ssl `<'SSL/ssl.SSLContext/None'>`: The SSL configuration for the connection. Defaults to `None`.
            - Supports both `sqlcycli.SSL` or pre-configured `ssl.SSLContext` object.

        :param auth_plugin `<'AuthPlugin/dict/None'>`: The authentication plugins handlers. Defaults to `None`.
            - Recommand use <'AuthPlugin'> to setup MySQL authentication plugin handlers.
            - If passed dict argument, it will be automatically converted to <'AuthPlugin'>.

        :param server_public_key `<'bytes/None'>`: The public key for the server authentication. Defaults to `None`.
        :param use_decimal `<'bool'>`: DECIMAL columns are decoded as `decimal.Decimal` if `True`, else as `float`. Defaults to `False`.
        :param decode_bit `<'bool'>`: BIT columns are decoded as `int` if `True`, else kept as the original `bytes`. Defaults to `False`.
        :param decode_json `<'bool'>`: JSON columns are deserialized if `True`, else kept as the original JSON string. Defaults to `False`.
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
        recycle: int | None,
    ) -> cython.bint:
        """(internal) Setup the pool.

        :param min_size `<'int'>`: The minimum number of [async] connections to maintain.
        :param max_size `<'int'>`: The maximum number of [async] connections to maintain.
        :param recycle `<'int/None'>`: The connection recycle time in seconds.
        """
        # . counting
        self._acqr = 0
        self._free = 0
        # . internal
        if min_size < 0:
            raise errors.InvalidPoolArgsError(
                "Invalid minimum pool size '%d', must be a postive integer." % min_size
            )
        self._min_size = min_size
        if max_size < max(1, min_size):
            raise errors.InvalidPoolArgsError(
                "Invalid maximum pool size '%d', must be greater than %d."
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
        """The CHARACTER SET of the connections `<'str'>`."""
        return self._charset._name

    @property
    def collation(self) -> str:
        """The COLLATION of the connections `<'str'>`."""
        return self._charset._collation

    @property
    def encoding(self) -> str:
        """The encoding of the connections `<'str'>`."""
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
        """The default autocommit mode of the pool `<'bool/None'>`.

        - All connections acquired from the pool will comply to this setting.

        :returns `<'bool/None'>`:
        - `True`: if the connections operate in autocommit (non-transactional) mode.
        - `False`: if the connections operate in manual commit (transactional) mode.
        - `None`: means pool is not connected yet and use the server default.
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
        """The maximum size of packet sent to server in bytes `<'int'>`."""
        return self._max_allowed_packet

    @property
    def sql_mode(self) -> str | None:
        """The default SQL_MODE for the connections `<'str/None'>`."""
        return self._sql_mode

    @property
    def init_command(self) -> str | None:
        """The initial SQL statement to run when connections are established `<'str/None'>`."""
        return self._init_command

    @property
    def client_flag(self) -> int:
        """The latest client flag of the connections `<'int'>`."""
        return self._client_flag

    @property
    def ssl(self) -> object | None:
        """The `ssl.SSLContext` of the connections `<'SSLContext/None'>`."""
        return self._ssl_ctx

    @property
    def auth_plugin(self) -> AuthPlugin | None:
        """The authentication plugin handlers `<'AuthPlugin/None'>`."""
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
        >>> (8, 0, 23)  # example"""
        return self._server_version

    @property
    def server_version_major(self) -> int | None:
        """The major server version `<'int/None'>`.
        >>> 8  # example"""
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
        """Determine whether DECIMAL columns are decoded as `decimal.Decimal` `<'bool'>`.

        DECIMAL columns are decoded as `decimal.Decimal` if `True`, else as `float`.

        - All connections acquired from the pool will comply to this setting.
        """
        return self._use_decimal

    @property
    def decode_bit(self) -> bool:
        """Determine whether BIT columns are decoded as integers `<'bool'>`.

        BIT columns are decoded as `int` if `True`, else kept as the original `bytes`.

        - All connections acquired from the pool will comply to this setting.
        """
        return self._decode_bit

    @property
    def decode_json(self) -> bool:
        """Determine whether JSON columns are deserialized `<'bool'>`.

        JSON columns are deserialized if `True`, else kept as the original JSON string.

        - All connections acquired from the pool will comply to this setting.
        """
        return self._decode_json

    # . pool
    @property
    def free(self) -> int:
        """Number of free [async] connections in the pool `<'int'>`."""
        return self._free

    @property
    def used(self) -> int:
        """Number of in-use [async] connections in the pool `<'int'>`."""
        return self._get_used()

    @property
    def total(self) -> int:
        """Total number of [async] connections in the pool `<'int'>`."""
        return self._get_total()

    @property
    def min_size(self) -> int:
        """Minimum number of [async] connections the pool will maintain `<'int'>`."""
        return self._min_size

    @property
    def max_size(self) -> int:
        """Maximum number of [async] connections the pool will maintain `<'int'>`."""
        return self._max_size

    @property
    def recycle(self) -> int | None:
        """The conneciton recycle time in seconds `<int/None>`.

        Any connections idling more than the `recycle` time
        will be closed and removed from the pool. Value of
        `None` means the recycling feature is disabled.
        """
        return None if self._recycle == -1 else self._recycle

    # Pool ------------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_min_size(self, size: cython.uint) -> cython.bint:
        """Change the minimum number of [async] connections the pool maintains.

        :param size `<'int'>`: The new minimum pool size.
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
        """Change the connection recycle time.

        :param recycle `<'int/None'>`: New connection recycle time in seconds.

            When set to a positive integer, the pool will automatically close
            and remove any connections idling more than the `recycle` time.
            Any other values disables the recycling feature.
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
        """Set the default autocommit mode of the pool.

        - All connections acquired from the pool will comply to this setting.

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
        """Set the decoding behavior for DECIMAL columns.

        - All connections acquired from the pool will comply to this setting.

        :param value `<'bool'>`: DECIMAL columns are decoded as
            `decimal.Decimal` if `True`, else as `float`.
        """
        self._use_decimal = value
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_decode_bit(self, value: cython.bint) -> cython.bint:
        """Set the decoding behavior for BIT columns.

        - All connections acquired from the pool will comply to this setting.

        :param value `<'bool'>`: BIT columns are decoded as `int`
            if `True`, else kept as the original `bytes`.
        """
        self._decode_bit = value
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_decode_json(self, value: cython.bint) -> cython.bint:
        """Set the decoding behavior for JSON columns.

        - All connections acquired from the pool will comply to this setting.

        :param value `<'bool'>`: JSON columns are deserialized if
            `True`, else kept as the original JSON string.
        """
        self._decode_json = value
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _add_free_conn(self, conn: PoolConnection) -> cython.bint:
        """(internal) Add a free [async] connection back to
        the pool, and update the counting.

        :param conn `<'PoolConnection'>`: The [async] connection instance.
        """
        self._free_conns.append(conn)
        self._free += 1
        return True

    @cython.cfunc
    @cython.inline(True)
    def _get_free_conn(self) -> PoolConnection:
        """(internal) Get a free [async] connection from the pool,
        and update the counting `<'PoolConnection/None'>`.

        :returns `<'PoolConnection'>`: The [async] connection instance,
            or `None` when no more free connection is available.
        """
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
        """(internal) Get the number of free [async] connections in the pool `<'int'>`."""
        return self._free

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_used(self) -> cython.uint:
        """(internal) Get the number of in-use [async] connections in the pool `<'int'>`."""
        if self._used_conns is None:
            return 0
        return set_len(self._used_conns)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_total(self) -> cython.uint:
        """(internal) Get the total number of [async] connections in the pool `<'int'>`."""
        return self._acqr + self._free + self._get_used()

    @cython.cfunc
    @cython.inline(True)
    def _get_loop(self) -> object:
        """(internal) Get the async event loop `<'AbstractEventLoop'>`'"""
        if self._loop is None:
            self._loop = _get_event_loop()
        return self._loop

    # Acquire / Transaction / Fill / Release --------------------------------------------------
    @cython.ccall
    def acquire(self) -> PoolConnectionManager:
        """Acquire a free connection from the pool through context manager `<'PoolConnectionManager'>`.

        ## Notice
        - **On Acquisition**: The following session settings are reset to the pool's defaults:
          `autocommit`, `used_decimal`, `decode_bit`, `decode_json`.
        - **At Release**: Any changes made vis `set_*()` methods (e.g. `set_charset()`,
          `set_read_timeout()`, etc.) will be reverted back to the pool defaults.
        - **Consistency**: Any other session-level changes (e.g. via SQL statements) will
          break the pool connection consistency. Please call `Connection.schedule_close()`
          before exiting the context.

        ## Example (sync):
        >>> with pool.acquire() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM table")

        ## Example (async):
        >>> async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT * FROM table")
        """
        return PoolConnectionManager(self)

    @cython.ccall
    def transaction(self) -> PoolTransactionManager:
        """Acquire a free connection from the pool in TRANSACTION mode
        through context manager `<'PoolTransactionManager'>`.

        ## On enter
        - 1. Acquire a free connection from the pool.
        - 2. Calls `BEGIN` on the connection

        ## On exit
        - If no exception occurs: calls `COMMIT` and releases the connection back to the pool for reuse.
        - If an exception occurs: Schedules the connection for closure and releases it back to the pool.

        ## Notice
        - **On Acquisition**: The following session settings are reset to the pool's defaults:
          `autocommit`, `used_decimal`, `decode_bit`, `decode_json`.
        - **At Release**: Any changes made vis `set_*()` methods (e.g. `set_charset()`,
          `set_read_timeout()`, etc.) will be reverted back to the pool defaults.
        - **Consistency**: Any other session-level changes (e.g. via SQL statements) will
          break the pool connection consistency. Please call `Connection.schedule_close()`
          before exiting the context.

        ## Example (sync):
        >>> with pool.transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO tb (id, name) VALUES (1, 'test')")

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
        """Fill the pool with new [async] connections.

        :param num `<'int'>`: Number of new [async] connections to create. Defaults to `1`.

            - If 'num' plus the total [async] connections in the pool exceeds the
              maximum pool size, only fills up to the `Pool.max_size` limit.
            - If 'num=-1', fills up to the `Pool.min_size` limit.
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
        """Release a connection back to the pool `<'Task[None]'>`.

        - Use this method `ONLY` when you directly acquired a connection without the context manager.
        - Connections obtained via context manager are released automatically on exits.

        :param conn `<'PoolConnection/PoolSyncConnection'>`: The pool [sync/async] connection to release.

        :returns `<'Task[None]'>`: An `asyncio.Task` that resolves once the connection is released.

            - For a [sync] connection, the returned `Task` can be ignored,
              as the connection is released immediately.
            - For an [async] connection, the returned `Task` must be awaited
              to ensure the connection is properly handled.

        :raises `<'PoolReleaseError'>`: If the connection does not belong to the pool.

        ## Example (sync):
        >>> pool.release(sync_conn)  # immediate release

        ## Example (async):
        >>> await pool.release(async_conn)  # 'await' for release
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
        """(internal) Acquire the [sync] connection maintained by the pool `<'PoolSyncConnection/None'>`.

        :returns `<'PoolSyncConnection/None'>`: Returns the existing active [sync]
            connection if available and within the recycle policy; otherwise,
            establishes a new one. If the pool is closing, returns `None`.
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
        """(internal) Release the [sync] connection back to the pool.

        :param conn `<'PoolSyncConnection'>`: The [sync] connection to release.
        :raises `<'PoolReleaseError'>`: If the connection does not belong to the pool.
        """
        # Validate pool identity
        if conn is not self._sync_conn:
            raise errors.PoolReleaseError(
                "Cannot release connection that does not belong to the pool."
            )

        # Scheduled close / in transaction / closing
        if conn._close_scheduled or conn.get_transaction_status() or self._closing:
            return self._close_sync_conn()

        # Reset connection
        # . charset
        if conn._charset_changed:
            try:
                conn.set_charset(self._charset._name, self._charset._collation)
                conn._charset_changed = False
            except Exception:
                return self._close_sync_conn()
        # . read timeout
        if conn._read_timeout_changed:
            try:
                conn.set_read_timeout(None)
            except Exception:
                return self._close_sync_conn()
        # . write timeout
        if conn._write_timeout_changed:
            try:
                conn.set_write_timeout(None)
            except Exception:
                return self._close_sync_conn()
        # . wait timeout
        if conn._wait_timeout_changed:
            try:
                conn.set_wait_timeout(None)
            except Exception:
                return self._close_sync_conn()
        # . interactive timeout
        if conn._interactive_timeout_changed:
            try:
                conn.set_interactive_timeout(None)
            except Exception:
                return self._close_sync_conn()
        # . lock wait timeout
        if conn._lock_wait_timeout_changed:
            try:
                conn.set_lock_wait_timeout(None)
            except Exception:
                return self._close_sync_conn()
        # . execution timeout
        if conn._execution_timeout_changed:
            try:
                conn.set_execution_timeout(None)
            except Exception:
                return self._close_sync_conn()

        # Finished
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _close_sync_conn(self) -> cython.bint:
        """(internal) Close the [sync] connection maintained by the pool."""
        if self._sync_conn is not None:
            self._sync_conn.close()
            self._sync_conn = None
        return True

    # (Async) Acquire / Release ---------------------------------------------------------------
    async def _acquire_async_conn(self) -> PoolConnection | None:
        """(internal) Acquire an [async] free connection maintained by the pool `<'PoolConnection/None'>`.

        :returns `<'PoolConnection/None'>`: Returns a free [async] connection
            if available; otherwise, fills the pool with new connections
            and waits for a free connection to become available.
            If the pool is closing, returns `None`.
        """
        async with self._condition:
            while True:
                # Fill to minimum size
                await self._fill_async_min()

                # Acquire free connection
                conn = await self._acquire_async_conn_free()
                if conn is not None:
                    return conn

                # Acquire new connection
                await self._fill_async_new()
                conn = await self._acquire_async_conn_free()
                if conn is not None:
                    return conn

                # Pool is closing
                if self._closing:
                    # eixt: pool is closing, return None
                    return await self.wait_for_closure()

                # Wait for notification
                await self._condition.wait()

    async def _acquire_async_conn_free(self) -> PoolConnection | None:
        """(internal) Acquire an existing free [async] connection from the pool `<'PoolConnection/None'>`.

        :returns `<'PoolConnection/None'>`: The existing free [async] connection
            in FIFO order, or `None` if free connection is not available.
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

    async def _fill_async_min(self) -> None:
        """(internal) Fill the pool with new [async] connections up to `Pool.min_size` limit."""
        if self._min_size == 0 or self._closing:
            return None

        total: cython.uint = self._get_total()
        if total < self._min_size:
            await _gather(
                *[self._fill_async_new() for _ in range(self._min_size - total)]
            )
            self._condition.notify()

    async def _fill_async_new(self) -> None:
        """(internal) Fill the pool with one extra new connection (limited by `Pool.max_size`)."""
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
        """(internal) Release an [async] connection back to the pool.

        :param conn `<'PoolConnection'>`: The [async] connection to release.
        :raises `<'PoolReleaseError'>`: If the connection does not belong to the pool.
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

        # Add back to pool
        self._add_free_conn(conn)
        return await notify()

    # Close -----------------------------------------------------------------------------------
    @cython.ccall
    def close(self) -> object:
        """Initiate the shutdown of the connection pool `<'Task[None]'>`.

        This method will:
        - 1. Prevent any further connection acquisitions.
        - 2. Immediately closes and removes the free [sync] connection.
        - 3. Creates an `asyncio.Task` that closes and removes all [async] connections in the pool.

        :returns `<'Task[None]'>`: An `asyncio.Task` that resolves once:

            - All free [async] connections are closed and removed.
            - Waited for all in-use [async] connections to be returned, closed and removed.

        ## Notice
        - In purely synchronous environment, the returned `Task` can be ignored,
          as the [sync] connection is always closed and remvoed immediately.
        - In asynchronous environment, please `await` for the returned `Task`
          to ensure a proper shutdown.
        - This method does not raise any error.

        ## Example (sync):
        >>> with pool.acquire() as conn:
                # some SQL queries
                ...
            pool.close()  # immediate closure

        ## Example (async):
        >>> async with pool.acquire() as conn:
                # some SQL queries
                ...
            await pool.close()  # 'await' for closure
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

        # Check for [async] connections status
        if self._get_total() == 0:  # all closed
            # Set closed flag
            self._closed = True
            fut = loop.create_future()
            fut.set_result(None)
            return fut  # exit

        # Create the 'wait_for_closure' Task
        return loop.create_task(self.wait_for_closure())

    async def wait_for_closure(self) -> None:
        """Wait until the connection pool has fully closed all its connections.

        This coroutine will:
        - 1. Close and remove any remaining [async] free connections.
        - 2. Wait for all in-use [async] connections to be returned, closed and removed.

        :raises `<'PoolNotClosedError'>`: Raises error if called before `Pool.close()`.
            Otherwise, this method does not raise any error.

        ## Notice
        - Only call this method after `Pool.close()` has been invoked.

        ## Example (async):
        >>> async with pool.acquire() as conn:
                # some SQL queries
                ...
            pool.close()  # initiate closure
            await pool.wait_for_closure()  # 'await' for closure
        """
        # Pool already closed
        if self.closed():
            return None
        if not self._closing:
            raise errors.PoolNotClosedError(
                "`Pool.wait_for_closure()` should only be "
                "called after `Pool.close()` is invoked."
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
        """Clear all the free [async] connections in the pool.

        This method closes and removes all the idling free [async] 
        connections in the pool, without affecting in-use ones.
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
        """Immediately shutdown the connection pool and force-close all connections.

        This method will:
        - 1. Prevent any further connection acquisitions.
        - 2. Force-close and remove all free [sync & async] connnections.
        - 3. Force-close and remove all in-use [async] connnections.

        ## Notice
        - Use this method only when an immediate, unconditional shutdown is needed.
        - This method does not raise any error.
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
        """Check if the pool is closed `<'bool'>`."""
        return self._closed and self._sync_conn is None

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_open(self) -> cython.bint:
        """(internal) Verify the pool is open and accepts connection acquisitions.

        :raises `<'PoolClosedError'>`: If pool has been closed.
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
        """Prepare and escape arguments for SQL binding `<'str/tuple/list[str/tuple]'>`.

        :param args `<'Any'>`: Arguments to escape, supports:

            - **Python built-ins**:
                int, float, bool, str, None, datetime, date, time,
                timedelta, struct_time, bytes, bytearray, memoryview,
                Decimal, dict, list, tuple, set, frozenset, range
            - **Library [numpy](https://github.com/numpy/numpy)**:
                np.int, np.uint, np.float, np.bool, np.bytes,
                np.str, np.datetime64, np.timedelta64, np.ndarray
            - **Library [pandas](https://github.com/pandas-dev/pandas)**:
                pd.Timestamp, pd.Timedelta, pd.DatetimeIndex,
                pd.TimedeltaIndex, pd.Series, pd.DataFrame
            - **Library [cytimes](https://github.com/AresJef/cyTimes)**:
                cytimes.Pydt, cytimes.Pddt

        :param many `<'bool'>`: Whether the 'args' is multi-row data. Defaults to `False`.
        - `many=False`: The 'itemize' parameter determines how to escape the 'args'.
        - `many=True`: The 'itemize' parameter is ignored, and the 'args' data type determines how escape is done.
            - 1. Sequence or Mapping (e.g. `list`, `tuple`, `dict`, etc) escapes to `<'list[str]'>`.
            - 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to `<'list[str]'>`.
            - 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes to `<'list[tuple[str]]'>`.
            - 4. Single object (such as `int`, `float`, `str`, etc) escapes to one literal string `<'str'>`.

        :param itemize `<'bool'>`: Whether to escape items of the 'args' individually. Defaults to `True`.
        - `itemize=False`: Always escapes to one single literal string `<'str'>`, regardless of the 'args' type.
        - `itemize=True`: The 'args' data type determines how escape is done.
            - 1. Sequence or Mapping (e.g. `list`, `tuple`, `dict`, etc) escapes to `<'tuple[str]'>`.
            - 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to `<'tuple[str]'>`.
            - 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes to `<'list[tuple[str]]'>`.
            - 4. Single object (such as `int`, `float`, `str`, etc) escapes to one literal string `<'str'>`.

        :returns `<'str/tuple/list'>`:
        - If returns `<'str'>`, it represents a single literal string.
        - If returns `<'tuple'>`, it represents a single row of literal strings.
        - If returns `<'list'>`, it represents multiple rows of literal strings.

        :raises `<'EscapeTypeError'>`: If escape fails due to unsupported type.
        """
        return escape(args, many, itemize)

    # Special Methods -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            "<%s (free=%d, used=%d, total=%d, min_size=%d, max_size=%d, recycle=%s)>"
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
