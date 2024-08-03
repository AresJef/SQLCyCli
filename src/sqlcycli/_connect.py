# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython

# Python imports
from os import PathLike
from typing import Generator, Any
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


# Connection ----------------------------------------------------------------------------------
@cython.cclass
class ConnectionManager:
    """The Context Manager for both `sync` and `async` Connections."""

    # . connection
    _conn: BaseConnection
    _aconn: aio.BaseConnection
    # . arguments
    _cursor: type[Cursor | aio.Cursor]
    _args: dict[str, Any]

    def __init__(
        self,
        host: str | None = "localhost",
        port: int = 3306,
        user: str | bytes | None = None,
        password: str | bytes | None = None,
        database: str | bytes | None = None,
        *,
        charset: str | None = "utf8mb4",
        collation: str | None = None,
        connect_timeout: int = 5,
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
    ) -> None:
        """The Context Manager for both `sync` and `async` Connections.

        For information about the arguments, please refer
        to `<'sqlcycli.Connection'>` class.
        """
        # Connection
        self._conn = None
        self._aconn = None
        # Arguments
        self._cursor = cursor
        self._args = {
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
        }

    # Sync --------------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _sync_cursor(self, cursor: Any) -> object:
        """(cfunc) Validate and map `sync` `<'Cursor'>`."""
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
            "Invalid 'cursor' %r, must be type of %r." % (cursor, Cursor)
        )

    def __enter__(self) -> BaseConnection:
        conn = Connection(cursor=self._sync_cursor(self._cursor), **self._args)
        conn.connect()
        self._conn = conn
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        self._conn = None

    # Async -------------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _async_cursor(self, cursor: Any) -> object:
        """(cfunc) Validate and map `async` `<'Cursor'>`."""
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
            "Invalid 'cursor' %r, must be type of %r." % (cursor, aio.Cursor)
        )

    async def _acquire_aconn(self) -> aio.BaseConnection:
        """(internal) Acquire an `async` `<'BaseConnection'>`."""
        conn = aio.Connection(cursor=self._async_cursor(self._cursor), **self._args)
        await conn.connect()
        return conn

    def __await__(self) -> Generator[Any, Any, aio.BaseConnection]:
        return self._acquire_aconn().__await__()

    async def __aenter__(self) -> aio.BaseConnection:
        self._aconn = await self._acquire_aconn()
        return self._aconn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._aconn.close()
        self._aconn = None

    def __del__(self):
        if self._conn is not None:
            self._conn.force_close()
            self._conn = None
        if self._aconn is not None:
            self._aconn.force_close()
            self._aconn = None


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
    cursor: type[Cursor] = Cursor,
    client_flag: int | Any = 0,
    program_name: str | None = None,
    option_file: str | bytes | PathLike | OptionFile | None = None,
    ssl: SSL | object | None = None,
    auth_plugin: dict[str | bytes, type] | AuthPlugin | None = None,
    server_public_key: bytes | None = None,
    use_decimal: bool = False,
    decode_json: bool = False,
) -> ConnectionManager:
    """Connect to the server and acquire a `sync` or `async`
    connection through context manager `<'ConnectionManager'>`.

    For information about the arguments, please refer
    to `<'sqlcycli.Connection'>` class.

    ### Example (sync):
    >>> with connect("localhost") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    ### Example (async):
    >>> async with connect("localhost") as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
    """
    return ConnectionManager(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset=charset,
        collation=collation,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        wait_timeout=wait_timeout,
        bind_address=bind_address,
        unix_socket=unix_socket,
        autocommit=autocommit,
        local_infile=local_infile,
        max_allowed_packet=max_allowed_packet,
        sql_mode=sql_mode,
        init_command=init_command,
        cursor=cursor,
        client_flag=client_flag,
        program_name=program_name,
        option_file=option_file,
        ssl=ssl,
        auth_plugin=auth_plugin,
        server_public_key=server_public_key,
        use_decimal=use_decimal,
        decode_json=decode_json,
    )
