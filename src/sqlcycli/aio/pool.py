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
from cython.cimports.sqlcycli import connection as sync_conn  # type: ignore
from cython.cimports.sqlcycli.aio.connection import validate_cursor  # type: ignore
from cython.cimports.sqlcycli.aio.connection import BaseConnection, Cursor  # type: ignore
from cython.cimports.sqlcycli._ssl import SSL  # type: ignore
from cython.cimports.sqlcycli.charset import Charset  # type: ignore
from cython.cimports.sqlcycli._auth import AuthPlugin  # type: ignore
from cython.cimports.sqlcycli._optionfile import OptionFile  # type: ignore
from cython.cimports.sqlcycli.transcode import decode_bytes  # type: ignore

# Python imports
from os import PathLike
from collections import deque
from typing import Literal, Generator, Any
from asyncio import gather, get_event_loop
from asyncio import AbstractEventLoop, Condition, Future
from sqlcycli import connection as sync_conn
from sqlcycli.aio.connection import BaseConnection, Cursor
from sqlcycli._ssl import SSL
from sqlcycli.charset import Charset
from sqlcycli._auth import AuthPlugin
from sqlcycli._optionfile import OptionFile
from sqlcycli import errors

__all__ = ["PoolConnection", "PoolConnectionManager", "Pool"]


# Connection ----------------------------------------------------------------------------------
@cython.cclass
class PoolConnection(BaseConnection):
    """Represents a connection to the MySQL Server managed by a pool.
    It is used to communicate with the server and execute SQL queries.
    """

    # . pool
    _pool_id: cython.Py_ssize_t
    _close_scheduled: cython.bint

    # Init ------------------------------------------------------------------------------------
    def __init__(
        self,
        pool_id: int,
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
        bind_address: str | None,
        unix_socket: str | None,
        autocommit_mode: cython.int,
        local_infile: cython.bint,
        max_allowed_packet: cython.uint,
        sql_mode: str | None,
        init_command: str | None,
        cursor: type[Cursor],
        client_flag: cython.uint,
        program_name: str | None,
        ssl_ctx: object | None,
        auth_plugin: AuthPlugin | None,
        server_public_key: bytes | None,
        use_decimal: cython.bint,
        decode_json: cython.bint,
        loop: AbstractEventLoop,
    ):
        # . pool
        self._pool_id = pool_id
        self._close_scheduled = False
        # . basic
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        # . charset
        self._init_charset(charset)
        # . timeouts
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._wait_timeout = wait_timeout
        # . client
        self._bind_address = bind_address
        self._unix_socket = unix_socket
        self._autocommit_mode = autocommit_mode
        self._local_infile = local_infile
        self._max_allowed_packet = max_allowed_packet
        self._sql_mode = sql_mode
        self._init_command = init_command
        self._cursor = cursor
        self._init_client_flag(client_flag)
        self._init_connect_attrs(program_name)
        # . ssl
        self._ssl_ctx = ssl_ctx
        # . auth
        self._auth_plugin = auth_plugin
        self._server_public_key = server_public_key
        # . decode
        self._use_decimal = use_decimal
        self._decode_json = decode_json
        # . internal
        self._init_internal()
        # . loop
        self._loop = loop

    # Property --------------------------------------------------------------------------------
    # . pool
    @property
    def close_scheduled(self) -> bool:
        return self._close_scheduled

    # Pool ------------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def schedule_close(self) -> cython.bint:
        self._close_scheduled = True


# Pool ----------------------------------------------------------------------------------------
@cython.cclass
class PoolConnectionManager:
    _pool: Pool
    _conn: PoolConnection

    def __init__(self, pool: Pool) -> None:
        self._pool = pool
        self._conn = None

    async def _acquire(self) -> PoolConnection:
        try:
            conn = await self._pool._acquire_conn()
            if conn is None:
                self._pool._verify_open()
                raise errors.PoolClosedError(
                    0, "Failed to acquire connection for unknown reason."
                )
        except:  # noqa
            self._conn = None
            self._pool = None
            raise
        self._conn = conn
        return self._conn

    def __await__(self) -> Generator[Any, Any, PoolConnection]:
        return self._pool._acquire_conn().__await__()

    async def __aenter__(self) -> PoolConnection:
        try:
            conn = await self._pool._acquire_conn()
            if conn is None:
                self._pool._verify_open()
                raise errors.PoolClosedError(
                    0, "Failed to acquire connection for unknown reason."
                )
        except:  # noqa
            self._conn = None
            self._pool = None
            raise
        self._conn = conn
        return self._conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._pool.release(self._conn)
        finally:
            self._conn = None
            self._pool = None

    def __del__(self):
        self._conn = None
        self._pool = None


@cython.cclass
class Pool:
    """Represents a pool that manages and maintains connections
    of the MySQL Server. Increases reusability of established
    connections and reduces the overhead for creating new ones.
    """

    # Connection args
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
    # . client
    _bind_address: str
    _unix_socket: str
    _autocommit_mode: object  # int
    _autocommit_value: cython.bint
    _local_infile: object  # bool
    _max_allowed_packet: object  # uint
    _sql_mode: str
    _init_command: str
    _cursor: type[Cursor]
    _client_flag: object  # uint
    _program_name: str
    # . ssl
    _ssl_ctx: object  # ssl.SSLContext
    # . auth
    _auth_plugin: AuthPlugin
    _server_public_key: bytes
    # . decode
    _use_decimal: cython.bint  # bool
    _decode_json: cython.bint  # bool
    # . server
    _server_protocol_version: cython.int
    _server_info: str
    _server_version: tuple[int]
    _server_version_major: cython.int
    _server_vendor: str
    _server_auth_plugin_name: str
    # Pool
    _min_size: cython.uint
    _max_size: cython.uint
    _recycle: cython.Py_ssize_t
    _id: cython.Py_ssize_t
    _acqr: cython.uint
    _free: cython.uint
    _free_conn: deque[PoolConnection]
    _used: cython.uint
    _used_conn: set[PoolConnection]
    _loop: AbstractEventLoop
    _cond: Condition
    _closing: cython.bint
    _closed: cython.bint

    # Init ------------------------------------------------------------------------------------
    def __init__(
        self,
        host: str | None = "localhost",
        port: int = 3306,
        user: str | bytes | None = None,
        password: str | bytes | None = None,
        database: str | bytes | None = None,
        min_size: int = 0,
        max_size: int = 10,
        recycle: int | None = None,
        *,
        charset: str | None = "utf8mb4",
        collation: str | None = None,
        connect_timeout: int = 10,
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
        cursor: type[Cursor] = Cursor,
        client_flag: int = 0,
        program_name: str | None = None,
        option_file: str | bytes | PathLike | OptionFile | None = None,
        ssl: SSL | object | None = None,
        auth_plugin: dict[str | bytes, type] | AuthPlugin | None = None,
        server_public_key: bytes | None = None,
        use_decimal: bool = False,
        decode_json: bool = False,
        loop: AbstractEventLoop | None = None,
    ):
        # Connection args
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
        self._charset = sync_conn.validate_charset(charset, collation)
        encoding_c: cython.pchar = self._charset._encoding_c
        # . basic
        self._host = sync_conn.validate_arg_str(host, "host", "localhost")
        self._port = sync_conn.validate_arg_uint(port, "port", 1, 65_535)
        self._user = sync_conn.validate_arg_bytes(user, "user", encoding_c, sync_conn.DEFAULT_USER)
        self._password = sync_conn.validate_arg_bytes(password, "password", "latin1", "")
        self._database = sync_conn.validate_arg_bytes(database, "database", encoding_c, None)
        # . timeouts
        self._connect_timeout = sync_conn.validate_arg_uint(connect_timeout, "connect_timeout", 1, sync_conn.MAX_CONNECT_TIMEOUT)
        self._read_timeout = sync_conn.validate_arg_uint(read_timeout, "read_timeout", 1, UINT_MAX)
        self._write_timeout = sync_conn.validate_arg_uint(write_timeout, "write_timeout", 1, UINT_MAX)
        self._wait_timeout = sync_conn.validate_arg_uint(wait_timeout, "wait_timeout", 1, UINT_MAX)
        # . client
        self._bind_address = sync_conn.validate_arg_str(bind_address, "bind_address", None)
        self._unix_socket = sync_conn.validate_arg_str(unix_socket, "unix_socket", None)
        self._autocommit_mode = sync_conn.validate_autocommit(autocommit)
        self._local_infile = bool(local_infile)
        self._max_allowed_packet = sync_conn.validate_max_allowed_packet(max_allowed_packet)
        self._sql_mode = sync_conn.validate_sql_mode(sql_mode)
        self._init_command = sync_conn.validate_arg_str(init_command, "init_command", None)
        self._cursor = validate_cursor(cursor)
        self._client_flag = sync_conn.validate_arg_uint(client_flag, "client_flag", 0, UINT_MAX)
        self._program_name = sync_conn.validate_arg_str(program_name, "program_name", None)
        # . ssl
        self._ssl_ctx = sync_conn.validate_ssl(ssl)
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
        self._server_public_key = sync_conn.validate_arg_bytes(server_public_key, "server_public_key", "ascii", None)
        # fmt: on
        # . decode
        self._use_decimal = bool(use_decimal)
        self._decode_json = bool(decode_json)
        # . server
        self._server_protocol_version = -1
        self._server_info = None
        self._server_version = None
        self._server_version_major = -1
        self._server_vendor = None
        self._server_auth_plugin_name = None

        # Pool
        # . min_size
        try:
            self._min_size = min_size
        except Exception as err:
            raise errors.InvalidPoolArgsError(
                "Invalid minimum pool size '%s'." % min_size
            ) from err

        # . max_size
        try:
            self._max_size = max_size
        except Exception as err:
            raise errors.InvalidPoolArgsError(
                "Invalid maximum pool size %s'." % max_size
            ) from err
        if self._max_size < 1:
            raise errors.InvalidPoolArgsError("Maximum pool size must be >= 1.")
        elif self._max_size < self._min_size:
            raise errors.InvalidPoolArgsError(
                "Maximum pool size must be >= %d (minimum pool size)." % self._min_size
            )

        # . recycle
        if recycle is None:
            self._recycle = -1
        else:
            try:
                self._recycle = recycle
            except Exception as err:
                raise errors.InvalidPoolArgsError(
                    "Invalid recycle time: %r." % recycle
                ) from err
            if self._recycle < -1:
                self._recycle = -1

        # . internal
        self._id = id(self)
        self._acqr = 0
        self._free_conn = deque(maxlen=self._max_size)
        self._used_conn = set()
        if loop is None or not isinstance(loop, AbstractEventLoop):
            self._loop = get_event_loop()
        else:
            self._loop = loop
        self._cond = Condition()
        self._closing = False
        self._closed = True

    # Property --------------------------------------------------------------------------------
    # . client
    @property
    def host(self) -> str:
        """The 'host' of the database server `<'str'>`."""
        return self._host

    @property
    def port(self) -> int:
        """The 'port' of the database server `<'int'>`."""
        return self._port

    @property
    def user(self) -> str | None:
        """The 'user' to login as `<'str/None'>`."""
        if self._user is None:
            return None
        return decode_bytes(self._user, self._charset._encoding_c)

    @property
    def password(self) -> str:
        """The 'password' for user authentication `<'str'>`."""
        return decode_bytes(self._password, "latin1")

    @property
    def database(self) -> str | None:
        """The 'database' to be used by the client `<'str/None'>`."""
        if self._database is None:
            return None
        return decode_bytes(self._database, self._charset._encoding_c)

    @property
    def charset(self) -> str:
        """The 'charactor set' to be used by the client `<'str'>`."""
        return self._charset._name

    @property
    def collation(self) -> str:
        """The 'collation' to be used by the client `<'str'>`."""
        return self._charset._collation

    @property
    def encoding(self) -> str:
        """The 'encoding' to be used by the client `<'str'>`."""
        return decode_bytes(self._charset._encoding, "ascii")

    @property
    def connect_timeout(self) -> int:
        """The timeout in seconds for establishing the connection `<'int'>`."""
        return self._connect_timeout

    @property
    def bind_address(self) -> str | None:
        """The 'bind_address' from which to connect to the host `<'str/None'>`."""
        return self._bind_address

    @property
    def unix_socket(self) -> str | None:
        """The 'unix_socket' to be used rather than TCP/IP `<'str/None'>`."""
        return self._unix_socket

    @property
    def autocommit(self) -> bool | None:
        """The 'autocommit' mode `<'bool/None'>`.
        None means use server default.
        """
        return None if self._closed else self._autocommit_value

    @property
    def local_infile(self) -> bool:
        """Whether enable the use of LOAD DATA LOCAL command `<'bool'>`."""
        return self._local_infile

    @property
    def max_allowed_packet(self) -> int:
        """The max size of packet sent to server in bytes `<'int'>`."""
        return self._max_allowed_packet

    @property
    def sql_mode(self) -> str | None:
        """The default 'sql_mode' to be used `<'str/None'>`."""
        return self._sql_mode

    @property
    def init_command(self) -> str | None:
        """The 'init_command' to be executed after connecting `<'str/None'>`."""
        return self._init_command

    @property
    def client_flag(self) -> int:
        """The 'client_flag' of the connection `<'int'>`."""
        return self._client_flag

    @property
    def ssl(self) -> object | None:
        """The 'ssl.SSLContext' to be used for secure connection `<'SSLContext/None'>`."""
        return self._ssl_ctx

    @property
    def auth_plugin(self) -> AuthPlugin | None:
        """The 'auth_plugin' to be used for authentication `<'AuthPlugin/None'>`."""
        return self._auth_plugin

    # . decode
    @property
    def use_decimal(self) -> bool:
        """Whether to convert DECIMAL to Decimal object `<'bool'>`.
        If `False`, DECIMAL will be converted to float.
        """
        return self._use_decimal

    @property
    def decode_json(self) -> bool:
        """Whether to decode JSON data `<'bool'>`.
        If `False`, JSON will be returned as string.
        """
        return self._decode_json

    # . server
    @property
    def protocol_version(self) -> int | None:
        """The protocol version of the server `<'int/None'>`."""
        if self._server_protocol_version != -1:
            return self._server_protocol_version
        else:
            return None

    @property
    def server_info(self) -> str | None:
        """The server information `<'str/None'>`."""
        return self._server_info

    @property
    def server_version(self) -> tuple[int] | None:
        """The server version `<'tuple[int]/None'>`."""
        return self._server_version

    @property
    def server_version_major(self) -> int | None:
        """The server major version `<'int/None'>`."""
        if self._server_version_major != -1:
            return self._server_version_major
        else:
            return None

    @property
    def server_vendor(self) -> Literal["mysql", "mariadb"] | None:
        """The server vendor (database type) `<'str/None'>`."""
        return self._server_vendor

    @property
    def server_auth_plugin_name(self) -> str | None:
        """The server authentication plugin name `<'str/None'>`."""
        return self._server_auth_plugin_name

    # . pool
    @property
    def free(self) -> int:
        return self.get_free()

    @property
    def used(self) -> int:
        return self.get_used()

    @property
    def total(self) -> int:
        return self.get_total()

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def recycle(self) -> int | None:
        return None if self._recycle == -1 else self._recycle

    # Pool ------------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_free(self) -> cython.uint:
        if self._free_conn is None:
            return 0
        return len(self._free_conn)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_used(self) -> cython.uint:
        if self._used_conn is None:
            return 0
        return set_len(self._used_conn)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_total(self) -> cython.uint:
        return self._acqr + self.get_free() + self.get_used()

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_min_size(self, value: cython.uint) -> cython.bint:
        if value > self._max_size:
            raise errors.InvalidPoolArgsError(
                "Minimum pool size must be greater than maximum pool size '%d'."
                % self._max_size
            )
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_recycle(self, value: cython.uint) -> cython.bint:
        self._recycle = value
        return True

    # Acquire / Fill / Release ----------------------------------------------------------------
    @cython.ccall
    def acquire(self) -> PoolConnectionManager:
        return PoolConnectionManager(self)

    async def _acquire_conn(self) -> PoolConnection:
        async with self._cond:
            while True:
                # Fill minimum
                await self._fill_min()

                # Acquire free connection
                conn = await self._acquire_free()
                if conn is not None:
                    return conn

                # Acquire new connection
                conn = await self._acquire_new()
                if conn is not None:
                    return conn

                # Pool is closing
                if self._closing:
                    return await self.wait_for_closed()

                # Wait for notification
                await self._cond.wait()

    async def _acquire_free(self) -> PoolConnection | None:
        while self.get_free() > 0 and not self._closing:
            # Get connection from the pool
            conn: PoolConnection = self._free_conn.popleft()

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
            if reader._eof or reader.at_eof() or reader.exception() is not None:
                await conn.close()
                continue

            # Track in-use
            set_add(self._used_conn, conn)
            return conn

        # No free available
        return None

    async def _acquire_new(self) -> PoolConnection | None:
        await self._fill_new()
        return await self._acquire_free()

    async def _fill_new(self) -> None:
        # Limit by pool size or is closing
        if self.get_total() >= self._max_size or self._closing:
            return None

        # Create new connection
        self._acqr += 1
        conn: PoolConnection = None
        try:
            # . create
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
                self._wait_timeout,
                self._wait_timeout,
                self._bind_address,
                self._unix_socket,
                self._autocommit_mode,
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
                self._decode_json,
                self._loop,
            )
            await conn.connect()
            if self._closed:
                self._autocommit_value = conn.get_autocommit()
                self._server_protocol_version = conn._server_protocol_version
                self._server_info = conn._server_info
                self._server_version = conn.get_server_version()
                self._server_version_major = conn._server_version_major
                self._server_vendor = conn.get_server_vendor()
                self._server_auth_plugin_name = conn._server_auth_plugin_name
                self._closed = False
            # . add to pool
            self._free_conn.append(conn)
        except:  # noqa
            if conn is not None:
                await conn.close()
            raise
        finally:
            self._acqr -= 1

    async def _fill_min(self) -> None:
        if self._min_size == 0:
            return None

        total: cython.uint = self.get_total()
        if total < self._min_size:
            await gather(*[self._fill_new() for _ in range(self._min_size - total)])
            self._cond.notify()

    async def fill(self, num: cython.int = 1) -> None:
        if num == 0 or num < -1 or self._closing:
            return None

        async with self._cond:
            total: cython.uint = self.get_total()
            if total >= self._max_size:
                return None

            n: cython.uint
            if num == -1:
                free: cython.uint = self.get_free()
                if free >= self._min_size:
                    return None
                n = self._min_size - free
            else:
                n = num
                n = min(n, self._max_size - total)

            await gather(*[self._fill_new() for _ in range(n)])
            self._cond.notify()

    async def release(self, conn: PoolConnection) -> None:
        # Validate identity
        if conn._pool_id != self._id:
            raise errors.PoolReleaseError(
                "Cannot release connection that does not belong to the pool."
            )

        # Remove from 'used_conn' pool
        set_discard(self._used_conn, conn)

        # Pool is closing
        if self._closing:
            if not conn.closed():
                await conn.close()
            await self._notify()
            return None

        # Connection already closed
        if conn.closed():
            return None

        # Close scheduled or in transaction
        if conn._close_scheduled or conn.get_transaction_status():
            await conn.close()
            return None

        # Back to free pool
        # . reset autocommit
        if conn.get_autocommit() != self._autocommit_value:
            await conn.set_autocommit(self._autocommit_value)
        # . reset decode
        if conn._use_decimal != self._use_decimal:
            conn.set_use_decimal(self._use_decimal)
        if conn._decode_json != self._decode_json:
            conn.set_decode_json(self._decode_json)
        # . back to pool
        self._free_conn.append(conn)
        await self._notify()

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify()

    # Close -----------------------------------------------------------------------------------
    def close(self) -> Future:
        # Pool already closed
        if self._closed:
            fut = self._loop.create_future()
            fut.set_result(None)
            return fut

        # Set closing flag
        self._closing = True
        return self._loop.create_task(self.wait_for_closed())

    async def wait_for_closed(self) -> None:
        # Pool already closed
        if self._closed:
            return None
        if not self._closing:
            raise errors.PoolNotClosedError(
                "Pool.wait_for_closed() should only be called after Pool.close()."
            )

        async with self._cond:
            # Close all free connection
            await self._close_free()

            # Wait for used connection
            while self.get_total() > 0:
                await self._cond.wait()

            # Notify
            self._cond.notify()

        self._closed = True

    async def _close_free(self) -> None:
        # Close all free connection
        while self.get_free() > 0:
            conn = self._free_conn.popleft()
            await conn.close()

    async def clear(self) -> None:
        async with self._cond:
            await self._close_free()
            self._cond.notify()

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def terminate(self) -> cython.bint:
        # Pool already closed
        if self._closed:
            return True
        self._closing = True

        # Force close all free
        while self.get_free() > 0:
            conn = self._free_conn.popleft()
            conn.force_close()

        # Force close all used
        if self.get_used() > 0:
            for conn in self._used_conn:
                conn.force_close()
            set_clear(self._used_conn)

        self._closed = True
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def closed(self) -> cython.bint:
        return self._closed

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_open(self) -> cython.bint:
        if self._closed:
            raise errors.PoolClosedError(0, "Pool is closed.")
        return True

    # Special Methods -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            "<%s(free=%d, used=%d, total=%d, min_size=%d, max_size=%d, recycle=%s)>"
            % (
                self.__class__.__name__,
                self.get_free(),
                self.get_used(),
                self.get_total(),
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
        self.terminate()

    def __del__(self):
        self.terminate()
