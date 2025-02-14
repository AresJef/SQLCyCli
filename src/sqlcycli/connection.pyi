from os import PathLike
from asyncio import AbstractEventLoop
from typing_extensions import Self
from typing import Any, Literal, Generator
import pandas as pd
from sqlcycli._ssl import SSL
from sqlcycli._auth import AuthPlugin
from sqlcycli._optionfile import OptionFile
from sqlcycli.protocol import FieldDescriptorPacket

# Result --------------------------------------------------------------------------------------
class MysqlResult:
    def __init__(self, conn: BaseConnection): ...
    # Property
    @property
    def affected_rows(self) -> int: ...
    @property
    def insert_id(self) -> int: ...
    @property
    def server_status(self) -> int: ...
    @property
    def warning_count(self) -> int: ...
    @property
    def message(self) -> bytes | None: ...
    @property
    def field_count(self) -> int: ...
    @property
    def fields(self) -> tuple[FieldDescriptorPacket] | None: ...
    # Read
    def read(self) -> bool: ...
    def init_unbuffered_query(self) -> bool: ...

# Cursor --------------------------------------------------------------------------------------
class Cursor:
    def __init__(self, conn: BaseConnection): ...
    # Property
    @property
    def executed_sql(self) -> str | None: ...
    @property
    def field_count(self) -> int: ...
    @property
    def fields(self) -> tuple[FieldDescriptorPacket] | None: ...
    @property
    def insert_id(self) -> int: ...
    @property
    def affected_rows(self) -> int: ...
    @property
    def warning_count(self) -> int: ...
    @property
    def lastrowid(self) -> int: ...
    @property
    def rowcount(self) -> int: ...
    @property
    def rownumber(self) -> int: ...
    @property
    def description(self) -> tuple[tuple] | None: ...
    @property
    def arraysize(self) -> int: ...
    @arraysize.setter
    def arraysize(self, value: int) -> None: ...
    # Write
    def execute(
        self,
        sql: str,
        args: Any = None,
        itemize: bool = True,
        many: bool = False,
    ) -> int: ...
    def executemany(self, sql: str, args: Any = None) -> int: ...
    def callproc(self, procname: str, args: tuple | list) -> tuple | list: ...
    def mogrify(
        self,
        sql: str,
        args: Any = None,
        itemize: bool = True,
        many: bool = False,
    ) -> str: ...
    # Read
    def fetchone(self) -> tuple | None: ...
    def fetchmany(self, size: int = 1) -> tuple[tuple]: ...
    def fetchall(self) -> tuple[tuple]: ...
    def scroll(
        self,
        value: int,
        mode: Literal["relative", "absolute"] = "relative",
    ) -> bool: ...
    def nextset(self) -> bool: ...
    def columns(self) -> tuple[str] | None: ...
    # Compliance
    def setinputsizes(self, *args): ...
    def setoutputsizes(self, *args): ...
    # Close
    def close(self) -> bool: ...
    def closed(self) -> bool: ...
    # Special
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> tuple: ...

class DictCursor(Cursor):
    # Read
    def fetchone(self) -> dict | None: ...
    def fetchmany(self, size: int = 1) -> tuple[dict]: ...
    def fetchall(self) -> tuple[dict]: ...
    # Special
    def __next__(self) -> dict: ...

class DfCursor(Cursor):
    # Read
    def fetchone(self) -> pd.DataFrame | None: ...
    def fetchmany(self, size: int = 1) -> pd.DataFrame: ...
    def fetchall(self) -> pd.DataFrame: ...
    # Special
    def __next__(self) -> pd.DataFrame: ...

class SSCursor(Cursor): ...
class SSDictCursor(DictCursor): ...
class SSDfCursor(DfCursor): ...

# Connection ----------------------------------------------------------------------------------
class CursorManager:
    def __init__(self, conn: BaseConnection, cursor: type[Cursor]): ...
    def __enter__(self) -> Cursor: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...

class TransactionManager(CursorManager):
    def __init__(self, conn: BaseConnection, cursor: type[Cursor]): ...

class BaseConnection:
    # Property
    @property
    def host(self) -> str: ...
    @property
    def port(self) -> int: ...
    @property
    def user(self) -> str | None: ...
    @property
    def password(self) -> str: ...
    @property
    def database(self) -> str | None: ...
    @property
    def charset(self) -> str: ...
    @property
    def collation(self) -> str: ...
    @property
    def encoding(self) -> str: ...
    @property
    def connect_timeout(self) -> int: ...
    @property
    def bind_address(self) -> str | None: ...
    @property
    def unix_socket(self) -> str | None: ...
    @property
    def autocommit(self) -> bool | None: ...
    @property
    def local_infile(self) -> bool: ...
    @property
    def max_allowed_packet(self) -> int: ...
    @property
    def sql_mode(self) -> str | None: ...
    @property
    def init_command(self) -> str | None: ...
    @property
    def client_flag(self) -> int: ...
    @property
    def ssl(self) -> object | None: ...
    @property
    def auth_plugin(self) -> AuthPlugin | None: ...
    @property
    def thread_id(self) -> int | None: ...
    @property
    def protocol_version(self) -> int | None: ...
    @property
    def server_info(self) -> str | None: ...
    @property
    def server_version(self) -> tuple[int] | None: ...
    @property
    def server_version_major(self) -> int | None: ...
    @property
    def server_vendor(self) -> Literal["mysql", "mariadb"] | None: ...
    @property
    def server_status(self) -> int | None: ...
    @property
    def server_capabilites(self) -> int | None: ...
    @property
    def server_auth_plugin_name(self) -> str | None: ...
    @property
    def affected_rows(self) -> int: ...
    @property
    def insert_id(self) -> int: ...
    @property
    def transaction_status(self) -> bool | None: ...
    @property
    def use_decimal(self) -> bool: ...
    @property
    def decode_bit(self) -> bool: ...
    @property
    def decode_json(self) -> bool: ...
    # Cursor
    def cursor(self, cursor: type[Cursor] | None = None) -> CursorManager: ...
    def transaction(self, cursor: type[Cursor] | None = None) -> TransactionManager: ...
    # Query
    def query(self, sql: str, unbuffered: bool = False) -> int: ...
    def begin(self) -> bool: ...
    def start(self) -> bool: ...
    def commit(self) -> bool: ...
    def rollback(self) -> bool: ...
    def create_savepoint(self, identifier: str) -> bool: ...
    def rollback_savepoint(self, identifier: str) -> bool: ...
    def release_savepoint(self, identifier: str) -> bool: ...
    def kill(self, thread_id: int) -> bool: ...
    def show_warnings(self) -> tuple[tuple]: ...
    def select_database(self, db: str) -> bool: ...
    def escape_args(
        self,
        args: Any,
        itemize: bool = True,
        many: bool = False,
    ) -> str | tuple[str | tuple[str]] | list[str | tuple[str]]: ...
    def encode_sql(self, sql: str) -> bytes: ...
    # . client
    def set_charset(
        self,
        charset: str,
        collation: str | None = None,
    ) -> bool: ...
    def get_autocommit(self) -> bool: ...
    def set_autocommit(self, value: bool) -> bool: ...
    # . timeouts
    def set_read_timeout(self, value: int | None) -> bool: ...
    def get_read_timeout(self) -> int: ...
    def set_write_timeout(self, value: int | None) -> bool: ...
    def get_write_timeout(self) -> int: ...
    def set_wait_timeout(self, value: int | None) -> bool: ...
    def get_wait_timeout(self) -> int: ...
    def set_interactive_timeout(self, value: int | None) -> bool: ...
    def get_interactive_timeout(self) -> int: ...
    def set_lock_wait_timeout(self, value: int | None) -> bool: ...
    def get_lock_wait_timeout(self) -> int: ...
    def set_execution_timeout(self, value: int | None) -> bool: ...
    def get_execution_timeout(self) -> int: ...
    # . server
    def get_server_version(self) -> tuple[int] | None: ...
    def get_server_vendor(self) -> str | None: ...
    def get_affected_rows(self) -> int: ...
    def get_insert_id(self) -> int: ...
    def get_transaction_status(self) -> bool: ...
    # . decode
    def set_use_decimal(self, value: bool) -> bool: ...
    def set_decode_bit(self, value: bool) -> bool: ...
    def set_decode_json(self, value: bool) -> bool: ...
    # Connect / Close
    def connect(self) -> bool: ...
    def close(self) -> bool: ...
    def force_close(self) -> bool: ...
    def closed(self) -> bool: ...
    def ping(self, reconnect: bool = True) -> bool: ...
    # Read
    def next_result(self, unbuffered: bool = False) -> int: ...
    # Special methods
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...

class Connection(BaseConnection):
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
        interactive_timeout: int | None = None,
        lock_wait_timeout: int | None = None,
        execution_timeout: int | None = None,
        bind_address: str | None = None,
        unix_socket: str | None = None,
        autocommit: bool | None = False,
        local_infile: bool = False,
        max_allowed_packet: int | str | None = None,
        sql_mode: str | None = None,
        init_command: str | None = None,
        cursor: type[Cursor] | None = Cursor,
        client_flag: int = 0,
        program_name: str | None = None,
        option_file: str | bytes | PathLike | OptionFile | None = None,
        ssl: SSL | object | None = None,
        auth_plugin: dict[str | bytes, type] | AuthPlugin | None = None,
        server_public_key: bytes | None = None,
        use_decimal: bool = False,
        decode_bit: bool = False,
        decode_json: bool = False,
    ): ...
