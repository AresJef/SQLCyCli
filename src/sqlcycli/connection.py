# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False
from __future__ import annotations

# Cython imports
import cython
from cython.cimports.cpython.time import time as unix_time  # type: ignore
from cython.cimports.libc.limits import UINT_MAX, ULLONG_MAX  # type: ignore
from cython.cimports.cpython.list import PyList_AsTuple as list_to_tuple  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GET_SIZE as tuple_len  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GetSlice as tuple_slice  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GetItem as tuple_getitem  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Split as str_split  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_GET_SIZE as bytes_len  # type: ignore
from cython.cimports.sqlcycli._ssl import SSL  # type: ignore
from cython.cimports.sqlcycli.charset import Charset  # type: ignore
from cython.cimports.sqlcycli._auth import AuthPlugin  # type: ignore
from cython.cimports.sqlcycli._optionfile import OptionFile  # type: ignore
from cython.cimports.sqlcycli.constants import _CLIENT, _COMMAND, _SERVER_STATUS  # type: ignore
from cython.cimports.sqlcycli.protocol import MysqlPacket, FieldDescriptorPacket  # type: ignore
from cython.cimports.sqlcycli.protocol import pack_int32, pack_uint8, unpack_uint8, unpack_uint16, unpack_uint32  # type: ignore
from cython.cimports.sqlcycli.transcode import escape_item, encode_item, decode_item, decode_bytes, decode_bytes_utf8  # type: ignore
from cython.cimports.sqlcycli import _auth, typeref  # type: ignore

# Python imports
from typing import Literal
from io import BufferedReader
from os import PathLike, getpid
import socket, errno, warnings, re
from pandas import DataFrame
from sqlcycli._ssl import SSL
from sqlcycli.charset import Charset
from sqlcycli._auth import AuthPlugin
from sqlcycli._optionfile import OptionFile
from sqlcycli.protocol import MysqlPacket, FieldDescriptorPacket
from sqlcycli.transcode import escape_item, encode_item, decode_item
from sqlcycli.constants import _CLIENT, _COMMAND, _SERVER_STATUS, CR, ER
from sqlcycli import _auth, typeref, errors

__all__ = [
    "MysqlResult",
    "Cursor",
    "DictCursor",
    "DfCursor",
    "SSCursor",
    "SSDictCursor",
    "SSDfCursor",
    "CursorManager",
    "TransactionManager",
    "BaseConnection",
    "Connection",
]


# Constant ------------------------------------------------------------------------------------
try:
    import getpass

    DEFAULT_USER: str = getpass.getuser()
    del getpass
except (ImportError, KeyError):
    # KeyError occurs when there's no entry in OS database for a current user.
    DEFAULT_USER: str = None
DEFUALT_CHARSET: str = "utf8mb4"
MAX_CONNECT_TIMEOUT: cython.int = 31_536_000  # 1 year
DEFALUT_MAX_ALLOWED_PACKET: cython.int = 16_777_216  # 16MB
MAXIMUM_MAX_ALLOWED_PACKET: cython.int = 1_073_741_824  # 1GB
# fmt: off
DEFAULT_CONNECT_ATTRS: bytes = gen_connect_attrs(  # type: ignore
    ["_client_name", "sqlcycli", "_client_version", "0.0.0"] )
# fmt: on
MAX_PACKET_LENGTH: cython.uint = 2**24 - 1
#: Max statement size which :meth:`executemany` generates.
#: Max size of allowed statement is max_allowed_packet - packet_header_size.
#: Default value of max_allowed_packet is 1048576.
MAX_STATEMENT_LENGTH: cython.uint = 1024000

#: Regular expression for :meth:`Cursor.executemany`.
#: executemany only supports simple bulk insert.
#: You can use it to load large dataset.
INSERT_VALUES_RE: re.Pattern = re.compile(
    r"\s*((?:INSERT|REPLACE)\b.+\bVALUES?\s*)"  # prefix: INSERT INTO ... VALUES
    + r"(\(\s*(?:%s|%\(.+\)s)\s*(?:,\s*(?:%s|%\(.+\)s)\s*)*\))"  # values: (%s, %s, ...)
    + r"(\s*(?:AS\s.*?)?\s*(?:ON DUPLICATE.*)?);?\s*\Z",  # suffix: AS ... ON DUPLICATE ...
    re.IGNORECASE | re.DOTALL,
)

#: Regular expression for server version.
SERVER_VERSION_RE: re.Pattern = re.compile(r".*?(\d+)\.(\d+)\.(\d+).*?")


# Result --------------------------------------------------------------------------------------
@cython.cclass
class MysqlResult:
    # Connection
    _conn: BaseConnection
    _use_decimal: cython.bint
    _decode_json: cython.bint
    # Packet data
    affected_rows: cython.ulonglong  # Value of 18446744073709551615 means None
    insert_id: cython.ulonglong  # Value of 18446744073709551615 means None
    server_status: cython.int  # Value of -1 means None
    warning_count: cython.uint
    has_next: cython.bint
    message: bytes
    # Field data
    field_count: cython.ulonglong
    fields: tuple[FieldDescriptorPacket]
    rows: tuple[tuple]
    # Unbuffered
    unbuffered_active: cython.bint

    def __init__(self, conn: BaseConnection) -> None:
        # Connection
        self._conn = conn
        self._use_decimal = conn._use_decimal
        self._decode_json = conn._decode_json
        # Packet data
        self.affected_rows = 0
        self.insert_id = 0
        self.server_status = -1
        self.warning_count = 0
        self.has_next = False
        self.message = None
        # Field data
        self.field_count = 0
        self.fields = None
        self.rows = None
        self.unbuffered_active = False

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def read(self) -> cython.bint:
        try:
            pkt = self._conn._read_packet()
            if pkt.read_ok_packet():
                self._read_ok_packet(pkt)
            elif pkt.read_load_local_packet():
                self._read_load_local_packet(pkt)
            else:
                self._read_result_packet(pkt)
        finally:
            self._conn = None
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def init_unbuffered_query(self) -> cython.bint:
        """
        :raise OperationalError: If the connection to the server is lost.
        :raise InternalError:
        """
        self.unbuffered_active = True
        try:
            pkt = self._conn._read_packet()
            if pkt.read_ok_packet():
                self._read_ok_packet(pkt)
                self.unbuffered_active = False
                self._conn = None
            elif pkt.read_load_local_packet():
                self._read_load_local_packet(pkt)
                self.unbuffered_active = False
                self._conn = None
            else:
                self._read_result_packet_fields(pkt)
                # Apparently, MySQLdb picks this number because it's the maximum
                # value of a 64bit unsigned integer. Since we're emulating MySQLdb,
                # we set it to this instead of None, which would be preferred.
                self.affected_rows = ULLONG_MAX  # 18446744073709551615
        except:  # noqa
            self._conn = None
            raise
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_ok_packet(self, pkt: MysqlPacket) -> cython.bint:
        self.affected_rows = pkt._affected_rows
        self.insert_id = pkt._insert_id
        self.server_status = pkt._server_status
        self.warning_count = pkt._warning_count
        self.has_next = pkt._has_next
        self.message = pkt._message
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_load_local_packet(self, pkt: MysqlPacket) -> cython.bint:
        if not self._conn._local_infile:
            raise RuntimeError(
                "**WARNING**: Received LOAD_LOCAL packet but option 'local_infile=False'."
            )
        try:
            self._conn._verify_connected()
            try:
                with open(pkt._filename, "rb") as file:
                    size = min(
                        self._conn._max_allowed_packet, 16 * 1024
                    )  # 16KB is efficient enough
                    while True:
                        chunk: bytes = file.read(size)
                        if not chunk:
                            break
                        self._conn._write_packet(chunk)
            except OSError:
                raise errors.LocalFileNotFoundError(
                    ER.FILE_NOT_FOUND,
                    "Cannot find local file at: '%s'." % pkt._filename,
                )
            finally:
                if not self._conn.closed():
                    # send the empty packet to signify we are done sending data
                    self._conn._write_packet(b"")
        except:  # noqa
            self._conn._read_packet()
            raise

        pkt = self._conn._read_packet()
        if not pkt.read_ok_packet():
            # pragma: no cover - upstream induced protocol error
            raise errors.CommandOutOfSyncError(
                CR.CR_COMMANDS_OUT_OF_SYNC, "Commands Out of Sync."
            )
        self._read_ok_packet(pkt)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_eof_packet(self, pkt: MysqlPacket) -> cython.bint:
        # TODO: Support CLIENT.DEPRECATE_EOF
        # 1) Add DEPRECATE_EOF to CAPABILITIES
        # 2) Mask CAPABILITIES with server_capabilities
        # 3) if server_capabilities & CLIENT.DEPRECATE_EOF:
        #    use OKPacketWrapper instead of EOFPacketWrapper
        self.warning_count = pkt._warning_count
        self.has_next = pkt._has_next
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_result_packet(self, pkt: MysqlPacket) -> cython.bint:
        # Fields
        self._read_result_packet_fields(pkt)
        # Rows
        rows = []
        affected_rows: cython.ulonglong = 0
        while True:
            pkt = self._conn._read_packet()
            if pkt.read_eof_packet():
                self._read_eof_packet(pkt)
                self._conn = None  # release reference to kill cyclic reference.
                break
            rows.append(self._read_result_packet_row(pkt))
            affected_rows += 1
        self.affected_rows = affected_rows
        self.rows = list_to_tuple(rows)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_result_packet_fields(self, pkt: MysqlPacket) -> cython.bint:
        self.field_count = pkt.read_length_encoded_integer()
        self.fields = list_to_tuple(
            [
                self._conn._read_field_descriptor_packet()
                for _ in range(self.field_count)
            ]
        )
        eof_packet = self._conn._read_packet()
        assert eof_packet.is_eof_packet(), "Protocol error, expecting EOF"
        return True

    @cython.cfunc
    @cython.inline(True)
    def _read_result_packet_row(self, pkt: MysqlPacket) -> tuple:
        # Settings
        encoding: cython.pchar = self._conn._encoding_c
        use_decimal: cython.bint = self._use_decimal
        decode_json: cython.bint = self._decode_json
        # Read data
        row: list = []
        field: FieldDescriptorPacket
        for field in self.fields:
            try:
                value: bytes = pkt.read_length_encoded_string()
            except IndexError:  # MysqlPacketCursorError
                # No more columns in this row
                break
            if value is not None:
                data = decode_item(
                    value,
                    field._type_code,
                    encoding,
                    field._is_binary,
                    use_decimal,
                    decode_json,
                )
            else:
                data = None
            row.append(data)
        # Return data
        return list_to_tuple(row)

    @cython.cfunc
    @cython.inline(True)
    def _read_result_packet_row_unbuffered(self) -> tuple:
        # Check if in an active query
        if not self.unbuffered_active:
            return None
        # EOF
        pkt = self._conn._read_packet()
        if pkt.read_eof_packet():
            self._read_eof_packet(pkt)
            self.unbuffered_active = False
            self.rows = None
            self._conn = None
            return None
        # Read row
        row = self._read_result_packet_row(pkt)
        self.affected_rows = 1
        self.rows = (row,)  # rows should tuple of row for compatibility.
        return row

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _drain_result_packet_unbuffered(self) -> cython.bint:
        # After much reading on the protocol, it appears that there is,
        # in fact, no way to stop from sending all the data after
        # executing a query, so we just spin, and wait for an EOF packet.
        while self.unbuffered_active:
            try:
                pkt = self._conn._read_packet()
            except errors.OperationalTimeoutError:
                # if the query timed out we can simply ignore this error
                self.unbuffered_active = False
                self._conn = None
                return True
            if pkt.read_eof_packet():
                self._read_eof_packet(pkt)
                self.unbuffered_active = False
                self._conn = None  # release reference to kill cyclic reference.
        return True

    def __del__(self):
        if self.unbuffered_active:
            self._drain_result_packet_unbuffered()


# Cursor --------------------------------------------------------------------------------------
# . buffered
@cython.cclass
class Cursor:
    _unbuffered: cython.bint  # Determines whether is SSCursor
    _conn: BaseConnection
    _encoding_c: cython.pchar
    _executed_sql: bytes
    _result: MysqlResult
    _field_count: cython.ulonglong
    _fields: tuple[FieldDescriptorPacket]
    _rows: tuple[tuple]
    _affected_rows: cython.ulonglong
    _row_idx: cython.ulonglong
    _row_size: cython.ulonglong
    _insert_id: cython.ulonglong
    _warning_count: cython.uint

    def __init__(self, conn: BaseConnection) -> None:
        self._init_setup(conn, False)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _init_setup(
        self,
        conn: BaseConnection,
        unbuffered: cython.bint,
    ) -> cython.bint:
        self._unbuffered = unbuffered  # Determines whether is SSCursor
        self._conn = conn
        self._encoding_c = conn._encoding_c
        self._executed_sql = None
        self._clear_result()
        return True

    # Property --------------------------------------------------------------------------------
    @property
    def executed_sql(self) -> bytes | None:
        """The last 'sql' executed by the cursor `<'bytes/None'>`.

        Notice the 'sql' is in <'bytes'> type, with all arguments (if any)
        escaped and encoded in the proper format for the server.
        """
        return self._executed_sql

    @property
    def field_count(self) -> int:
        return self._field_count

    @property
    def insert_id(self) -> int:
        return self._insert_id

    @property
    def affected_rows(self) -> int:
        return self._affected_rows

    @property
    def row_number(self) -> int:
        return self._row_idx

    @property
    def warning_count(self) -> int:
        return self._warning_count

    # Write -----------------------------------------------------------------------------------
    @cython.ccall
    def execute(
        self,
        sql: str,
        args: object = None,
        force_many: cython.bint = False,
        itemize: cython.bint = True,
    ) -> cython.ulonglong:
        """Execute a query.

        :param query: Query to execute.
        :type query: str

        :param args: Parameters used with query. (optional)
        :type args: tuple, list or dict

        :return: Number of affected rows.
        :rtype: int

        If args is a list or tuple, %s can be used as a placeholder in the query.
        If args is a dict, %(name)s can be used as a placeholder in the query.
        """

        """Run several data against one query.

        :param query: Query to execute.
        :type query: str

        :param args: Sequence of sequences or mappings. It is used as parameter.
        :type args: tuple or list

        :return: Number of rows affected, if any.
        :rtype: int or None

        This method improves performance on multiple-row INSERT and
        REPLACE. Otherwise it is equivalent to looping over args with
        execute().
        """
        # Single query: no args
        if args is None:
            return self._query_str(sql)

        # Escape args
        if force_many:
            itemize = True
        args = self.escape_args(args, itemize)

        # Single query: one arg
        if not itemize or type(args) is str:
            return self._query_str(self._format(sql, args))

        # Single query: empty args
        _args: tuple = args
        if tuple_len(_args) == 0:
            return self._query_str(self._format(sql, ()))

        # Single query: one row
        if not force_many and type(_args[0]) is str:
            return self._query_str(self._format(sql, _args))

        # Many-query: row by row
        rows: cython.ulonglong = 0
        m = INSERT_VALUES_RE.match(sql)
        if m is None:
            for arg in _args:
                rows += self._query_str(self._format(sql, arg))
            self._affected_rows = rows
            return rows

        # Bulk INSERT/REPLACE
        self._verify_connected()
        conn: BaseConnection = self._conn
        # . split query
        cmp: tuple = m.groups()
        pfix: str
        values: str
        sfix: str
        pfix, values, sfix = cmp
        # . query prefix
        pfix = self._format(pfix, ())
        prefix: bytes = conn.encode_sql(pfix)  # INSERT INTO ... VALUES
        # . query values
        values = values.rstrip()  # (%s, %s, ...)
        # . query suffix
        if sfix is not None:
            suffix: bytes = conn.encode_sql(sfix)  # AS ... ON DUPLICATE ...
        else:
            suffix: bytes = b""
        # . execute query
        args_iter = iter(_args)
        val: bytes = conn.encode_sql(self._format(values, next(args_iter)))
        val_len: cython.uint = bytes_len(val)
        stmt: bytearray = bytearray(prefix)
        stmt += val
        fix_len: cython.uint = bytes_len(prefix) + bytes_len(suffix)
        sql_len: cython.uint = fix_len + val_len
        for arg in args_iter:
            val = conn.encode_sql(self._format(values, arg))
            val_len = bytes_len(val)
            sql_len += val_len + 1
            if sql_len <= MAX_STATEMENT_LENGTH:
                stmt += b","
            else:
                rows += self._query_bytes(bytes(stmt + suffix))
                stmt = bytearray(prefix)
                sql_len = fix_len + val_len
            stmt += val
        rows += self._query_bytes(bytes(stmt + suffix))
        self._affected_rows = rows
        return rows

    @cython.ccall
    def callproc(self, procname: str, args: tuple | list) -> tuple:
        """Execute stored procedure procname with args.

        :param procname: Name of procedure to execute on server.
        :type procname: str

        :param args: Sequence of parameters to use with procedure.
        :type args: tuple or list

        Returns the original args.

        Compatibility warning: PEP-249 specifies that any modified
        parameters must be returned. This is currently impossible
        as they are only available by storing them in a server
        variable and then retrieved by a query. Since stored
        procedures return zero or more result sets, there is no
        reliable way to get at OUT or INOUT parameters via callproc.
        The server variables are named @_procname_n, where procname
        is the parameter above and n is the position of the parameter
        (from zero). Once all result sets generated by the procedure
        have been fetched, you can issue a SELECT @_procname_0, ...
        query using .execute() to get any OUT or INOUT values.

        Compatibility warning: The act of calling a stored procedure
        itself creates an empty result set. This appears after any
        result sets generated by the procedure. This is non-standard
        behavior with respect to the DB-API. Be sure to use next_set()
        to advance through all result sets; otherwise you may get
        disconnected.
        """
        # Validate & Escape 'args'
        if args is None:
            _args: tuple = ()
        else:
            items = encode_item(args)
            if not isinstance(items, tuple):
                raise errors.InvalidCursorArgsError(
                    "Invalid 'args' for 'callproc()' method, "
                    "expects <'tuple/list'> instead of %s." % type(args)
                )
            _args: tuple = items

        # Set arguments
        count: cython.Py_ssize_t = len(_args)
        if count > 0:
            fmt: str = f"@_{procname}_%d=%s"
            sql: str = "SET" + ",".join(
                [self._format(fmt, (idx, arg)) for idx, arg in enumerate(_args)]
            )
            self._query_str(sql)
            self.next_set()

        # Call procedures
        # fmt: off
        sql: str = "CALL %s(%s)" % (
            procname,
            ",".join([self._format("@_%s_%d", (procname, i)) for i in range(count)]),
        )
        # fmt: on
        self._query_str(sql)
        return _args

    @cython.ccall
    def mogrify(
        self,
        sql: str,
        args: object = None,
        itemize: cython.bint = True,
    ) -> str:
        if args is None:
            return sql
        return self._format(sql, self.escape_args(args, itemize))

    @cython.ccall
    def escape_args(self, args: object, itemize: cython.bint = True) -> object:
        """Escape object into string `<'str/tuple'>`."""
        return encode_item(args) if itemize else escape_item(args)

    @cython.ccall
    def encode_sql(self, sql: str) -> bytes:
        return encode_str(sql, self._encoding_c)  # type: ignore

    @cython.cfunc
    @cython.inline(True)
    def _query_str(self, sql: str) -> cython.ulonglong:
        return self._query_bytes(self.encode_sql(sql))

    @cython.cfunc
    @cython.inline(True)
    def _query_bytes(self, sql: bytes) -> cython.ulonglong:
        while self.next_set():
            pass
        self._verify_connected()
        self._clear_result()
        self._conn._execute_command(_COMMAND.COM_QUERY, sql)
        rows = self._conn._read_query_result(self._unbuffered)
        self._read_result()
        self._executed_sql = sql
        return rows

    @cython.cfunc
    @cython.inline(True)
    def _format(self, sql: str, args: object) -> str:
        try:
            return sql % args
        except Exception as err:
            raise errors.InvalidCursorArgsError(
                "Failed to format query:\n'%s'"
                "\nWith arguments: %s\n %s"
                "\nError: %s" % (sql, type(args), args, err)
            ) from err

    # Read ------------------------------------------------------------------------------------
    # . fetchone
    def fetchone(self) -> tuple | None:
        return self._fetchone_row()

    @cython.cfunc
    @cython.inline(True)
    def _fetchone_row(self) -> tuple:
        # Verify executed
        self._verify_executed()

        # Buffered
        if not self._unbuffered:
            # No more rows
            if not self._has_more_rows():
                return None  # exit: no rows
            row_i = tuple_getitem(self._rows, self._row_idx)
            self._row_idx += 1
            return cython.cast(tuple, row_i)  # exit: one row

        # Unbuffered
        else:
            row = self._next_row()
            if row is None:
                self._warning_count = self._result.warning_count
                return None  # exit: no rows
            return row  # exit: one row

    @cython.cfunc
    @cython.inline(True)
    def _fetchone_dict(self) -> dict:
        # Fetch row
        row = self._fetchone_row()
        if row is None:
            return None
        # No fields
        cols = self.columns()
        if cols is None:
            return None  # eixt: not column names
        # Generate dict
        return {cols[i]: row[i] for i in range(self._field_count)}

    @cython.cfunc
    @cython.inline(True)
    def _fetchone_df(self) -> object:
        # Fetch row
        row = self._fetchone_row()
        if row is None:
            return None
        # No fields
        cols = self.columns()
        if cols is None:
            return None
        # Generate DataFrame
        return typeref.DATAFRAME([row], columns=cols)

    # . fetch (all, many)
    def fetch(self, size: int = 0) -> tuple[tuple]:
        return self._fetch_row(size)

    @cython.cfunc
    @cython.inline(True)
    def _fetch_row(self, size: cython.ulonglong = 0) -> tuple[tuple]:
        # Verify executed
        self._verify_executed()

        # Buffered
        if not self._unbuffered:
            # No more rows
            if not self._has_more_rows():
                return ()  # exit: no rows
            # . fetch all
            row_size: cython.ulonglong = self._row_size  # row size already calcuated
            if size == 0:
                if self._row_idx == 0:
                    self._row_idx = row_size
                    return self._rows  # exit: all rows
                end: cython.ulonglong = row_size
            # . fetch many
            else:
                end: cython.ulonglong = min(self._row_idx + size, row_size)
            idx: cython.ulonglong = self._row_idx
            self._row_idx = end
            return tuple_slice(self._rows, idx, end)  # exit: multi-rows

        # Unbuffered
        else:
            rows: list = []
            # . fetch all
            if size == 0:
                while True:
                    row = self._next_row()
                    if row is None:
                        self._warning_count = self._result.warning_count
                        break
                    else:
                        rows.append(row)
            # . fetch many
            else:
                for _ in range(size):
                    row = self._next_row()
                    if row is None:
                        self._warning_count = self._result.warning_count
                        break
                    else:
                        rows.append(row)
            return list_to_tuple(rows)  # exit: multi-rows

    @cython.cfunc
    @cython.inline(True)
    def _fetch_dict(self, size: cython.ulonglong = 0) -> tuple[dict]:
        # Fetch rows
        rows = self._fetch_row(size)
        if not rows:
            return ()  # exit: no more rows
        # No fields
        cols = self.columns()
        if cols is None:
            return ()  # eixt: not column names
        # Generate tuple[dict]
        field_count: cython.ulonglong = self._field_count
        return list_to_tuple(
            [{cols[i]: r[i] for i in range(field_count)} for r in rows]
        )

    @cython.cfunc
    @cython.inline(True)
    def _fetch_df(self, size: cython.ulonglong = 0) -> object:
        # Fetch rows
        rows = self._fetch_row(size)
        if not rows:
            return None  # exit: no more rows
        # No fields
        cols = self.columns()
        if cols is None:
            return None  # eixt: not column names
        # Generate DataFrame
        return typeref.DATAFRAME(rows, columns=cols)

    # . scroll
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def scroll(
        self,
        value: cython.longlong,
        mode: Literal["relative", "absolute"] = "relative",
    ) -> cython.bint:
        # Validate
        self._verify_executed()

        # Buffered
        if not self._unbuffered:
            row_size: cython.ulonglong = self._get_row_size()
            if row_size == 0:
                self._row_idx = 0
                return False
            # Scroll cursor
            idx: cython.ulonglong
            if mode == "relative":
                if value < 0:
                    idx = -value
                    if idx > row_size:
                        raise errors.InvalidCursorIndexError(
                            "Cursor index cannot be negative: %d."
                            % (self._row_idx + value)
                        )
                idx = self._row_idx + value
            elif mode == "absolute":
                if value < 0:
                    raise errors.InvalidCursorIndexError(
                        "Cursor index cannot be negative: %d." % value
                    )
                idx = value
            else:
                raise errors.InvalidCursorArgsError("Inavlid scroll mode %r." % mode)
            if idx >= row_size:
                raise errors.InvalidCursorIndexError(
                    "Cursor index '%d' out of range: 0~%d." % (idx, row_size)
                )
            self._row_idx = idx
            return True

        # Unbuffered
        if value < 0:
            raise errors.InvalidCursorIndexError(
                "Backwards scrolling not supported by <'%s'>." % self.__class__.__name__
            )
        val: cython.ulonglong = value
        if mode == "relative":
            val = value
        elif mode == "absolute":
            if val < self._row_idx:
                raise errors.InvalidCursorIndexError(
                    "Backwards scrolling not supported by <'%s'>."
                    % self.__class__.__name__
                )
            val -= self._row_idx
        else:
            raise errors.InvalidCursorArgsError("Inavlid scroll mode %r." % mode)
        for _ in range(val):
            if self._next_row() is None:
                break
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def next_set(self) -> cython.bint:
        """Get the next set of query result."""
        self._verify_connected()
        conn: BaseConnection = self._conn
        if self._result is None or self._result is not conn._result:
            return False
        if not self._result.has_next:
            return False
        self._clear_result()
        conn.next_result(self._unbuffered)
        self._read_result()
        return True

    @cython.cfunc
    @cython.inline(True)
    def _next_row(self) -> tuple:
        row = self._result._read_result_packet_row_unbuffered()
        if row is not None:
            self._row_idx += 1
        return row

    @cython.ccall
    def columns(self) -> tuple[str]:
        """Get the 'column' names for each fields of the sql query result `<'tuple/None'>`."""
        # Sql returns no fields
        if self._field_count == 0:
            return None
        # Construct columns
        cols: list = []
        field: FieldDescriptorPacket
        for field in self._fields:
            col: str = field._column
            if col in cols:
                col = field._table + "." + col
            cols.append(col)
        return list_to_tuple(cols)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_result(self) -> cython.bint:
        result: MysqlResult = self._conn._result
        self._result = result
        self._field_count = result.field_count
        self._fields = result.fields
        self._rows = result.rows
        self._affected_rows = result.affected_rows
        self._insert_id = result.insert_id
        self._warning_count = result.warning_count
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _clear_result(self) -> cython.bint:
        self._result = None
        self._field_count = 0
        self._fields = None
        self._rows = None
        self._affected_rows = 0
        self._row_idx = 0
        self._row_size = 0
        self._insert_id = 0
        self._warning_count = 0
        return True

    @cython.cfunc
    @cython.inline(True)
    def _get_row_size(self) -> cython.ulonglong:
        if self._row_size == 0:
            if self._rows is None:
                return 0
            self._row_size = tuple_len(self._rows)
        return self._row_size

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _has_more_rows(self) -> cython.bint:
        row_size: cython.ulonglong = self._get_row_size()
        if row_size == 0 or self._row_idx >= row_size:
            return False  # exit: no more rows
        else:
            return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_executed(self) -> cython.bint:
        if self._executed_sql is None:
            raise errors.CursorNotExecutedError(0, "Please execute a 'sql' first.")
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_connected(self) -> cython.bint:
        "(cfunc) Verify if the connection is connected."
        if self.closed():
            raise errors.CursorClosedError(0, "Cursor is closed.")
        return True

    # Close -----------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def close(self) -> cython.bint:
        """Closing a cursor just exhausts all remaining data."""
        if self.closed():
            return True
        try:
            if (
                self._unbuffered
                and self._result is not None
                and self._result is self._conn._result
            ):
                self._result._drain_result_packet_unbuffered()
            while self.next_set():
                pass
        finally:
            self._conn = None
            self._clear_result()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def closed(self) -> cython.bint:
        """The cursor status, whether is closed `<'bool'>`."""
        return self._conn is None

    # Special methods -------------------------------------------------------------------------
    def __enter__(self) -> Cursor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self) -> Cursor:
        return self

    def __next__(self) -> tuple:
        row = self._fetchone_row()
        if row is None:
            raise StopIteration
        return row

    def __del__(self):
        self.close()


@cython.cclass
class DictCursor(Cursor):
    # . fetchone
    def fetchone(self) -> dict | None:
        return self._fetchone_dict()

    # . fetch (all, many)
    def fetch(self, size: int = 0) -> tuple[dict]:
        return self._fetch_dict(size)

    # Special methods -------------------------------------------------------------------------
    def __enter__(self) -> DictCursor:
        return self

    def __iter__(self) -> DictCursor:
        return self

    def __next__(self) -> dict:
        row = self._fetchone_dict()
        if row is None:
            raise StopIteration
        return row


@cython.cclass
class DfCursor(Cursor):
    # . fetchone
    def fetchone(self) -> DataFrame | None:
        return self._fetchone_df()

    # . fetch (all, many)
    def fetch(self, size: int = 0) -> DataFrame | None:
        return self._fetch_df(size)

    # Special methods -------------------------------------------------------------------------
    def __enter__(self) -> DfCursor:
        return self

    def __iter__(self) -> DfCursor:
        return self

    def __next__(self) -> DataFrame:
        row = self._fetchone_df()
        if row is None:
            raise StopIteration
        return row


# . unbuffered
@cython.cclass
class SSCursor(Cursor):
    def __init__(self, conn: BaseConnection) -> None:
        self._init_setup(conn, True)


@cython.cclass
class SSDictCursor(DictCursor):
    def __init__(self, conn: BaseConnection) -> None:
        self._init_setup(conn, True)


@cython.cclass
class SSDfCursor(DfCursor):
    def __init__(self, conn: BaseConnection) -> None:
        self._init_setup(conn, True)


# Connection ----------------------------------------------------------------------------------
@cython.cclass
class CursorManager:
    "The Context Manager for Cursor."

    _conn: BaseConnection
    _cur_type: type[Cursor]
    _cur: Cursor

    def __init__(self, conn: BaseConnection, cursor: type[Cursor]) -> None:
        """The Context Manager for Cursor.

        :param conn `<'BaseConnection'>`: The Connection that acquires the cursor.
        :param cursor `<'type[Cursor]'>`: The Cursor type (class) to use.
        """
        self._conn = conn
        self._cur_type = cursor
        self._cur = None

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _close(self) -> cython.bint:
        """(cfunc) Close the cursor and cleanup the manager `<'bool'>`.
        This method raises no error."""
        if self._cur is not None:
            try:
                self._cur.close()
            except:  # noqa
                pass
            self._cur = None
        self._conn = None
        self._cur_type = None

    def __enter__(self) -> Cursor:
        try:
            self._conn.connect()
            self._cur = self._cur_type(self._conn)
            return self._cur
        except BaseException as err:
            self._close()
            err.add_note(
                "-> <'%s'> Failed to acquire cursor: %s"
                % (self.__class__.__name__, err)
            )
            raise err

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()

    def __del__(self):
        self._close()


@cython.cclass
class TransactionManager(CursorManager):
    """The Context Manager for Cursor in transaction mode.

    By acquiring cursor through this manager, the following will happen:
    - 1. Use the connection to `START` a transaction.
    - 2. Return the `TransactionManager` that wraps the cursor.
    - 3a. If catches ANY exceptions during the transaction, execute `ROLLBACK`
    - 3b. If the transaction executed successfully, execute `COMMIT`.
    """

    def __init__(self, conn: BaseConnection, cursor: type[Cursor]) -> None:
        """The Context Manager for Cursor in transaction mode.

        :param conn `<'BaseConnection'>`: The Connection that starts the transaction.
        :param cursor `<'type[Cursor]'>`: The Cursor type (class) to use for the transaction.
        """
        self._conn = conn
        self._cur_type = cursor
        self._cur = None

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _close_connection(self) -> cython.bint:
        if self._conn is not None and self._conn.get_transaction_status():
            self._conn.close()
        return True

    def __enter__(self) -> Cursor:
        try:
            self._conn.connect()
            self._cur = self._cur_type(self._conn)
            self._conn.begin()
            return self._cur
        except BaseException as err:
            self._close()
            err.add_note(
                "-> <'%s'> Failed to start transaction: %s"
                % (self.__class__.__name__, err)
            )
            raise err

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Encounter error
        if exc_val is not None:
            self._close_connection()
            self._close()
            exc_val.add_note(
                "-> <'%s'> Failed to commit transaction: %s"
                % (self.__class__.__name__, exc_val)
            )
            raise exc_val

        # Try commit the transaction
        try:
            self._conn.commit()
        except BaseException as err:
            self._close_connection()
            self._close()
            err.add_note(
                "-> <'%s'> Failed to commit transaction: %s"
                % (self.__class__.__name__, err)
            )
            raise err
        self._close()


@cython.cclass
class BaseConnection:
    # Basic
    _host: str
    _port: object  # uint
    _user: bytes
    _password: bytes
    _database: bytes
    # Charset
    _charset: str
    _collation: str
    _charset_id: cython.uint
    _encoding: bytes
    _encoding_c: cython.pchar
    # Timeouts
    _connect_timeout: object  # uint
    _read_timeout: object  # uint | None
    _write_timeout: object  # uint | None
    _wait_timeout: object  # uint | None
    # Client
    _bind_address: str
    _unix_socket: str
    _autocommit_mode: cython.int
    _local_infile: cython.bint
    _max_allowed_packet: cython.uint
    _sql_mode: str
    _init_command: str
    _cursor: type[Cursor]
    _client_flag: cython.uint
    _connect_attrs: bytes
    # SSL
    _ssl_ctx: object  # ssl.SSLContext
    # Auth
    _auth_plugin: AuthPlugin
    _server_public_key: bytes
    # Decode
    _use_decimal: cython.bint
    _decode_json: cython.bint
    # Internal
    # . server
    _server_protocol_version: cython.int
    _server_info: str
    _server_version: tuple[int]
    _server_version_major: cython.int
    _server_vendor: str
    _server_thred_id: cython.longlong
    _server_salt: bytes
    _server_status: cython.int
    _server_capabilities: cython.longlong
    _server_auth_plugin_name: str
    # . client
    _last_used_time: cython.double
    _secure: cython.bint
    _host_info: str
    _close_reason: str
    # . query
    _result: MysqlResult
    _next_seq_id: cython.uint
    # . transport
    _reader: BufferedReader
    _writer: socket.socket

    # Init ------------------------------------------------------------------------------------
    def __init__(
        self,
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
    ):
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

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _init_charset(self, charset: Charset) -> cython.bint:
        """(cfunc) Initialize the charactor set `<'bool'>`."""
        self._charset_id = charset._id
        self._charset = charset._name
        self._collation = charset._collation
        self._encoding = charset._encoding
        self._encoding_c = charset._encoding_c
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _init_client_flag(self, client_flag: cython.uint) -> cython.bint:
        """(cfunc) Initialize the 'client_flag' `<'bool'>`"""
        if self._local_infile:
            client_flag |= _CLIENT.LOCAL_FILES
        if self._ssl_ctx is not None:
            client_flag |= _CLIENT.SSL
        client_flag |= _CLIENT.CAPABILITIES
        if self._database is not None:
            client_flag |= _CLIENT.CONNECT_WITH_DB
        self._client_flag = client_flag
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _init_connect_attrs(self, program_name: str | None) -> cython.bint:
        """(cfunc) Initialize the 'connect_attrs' `<'bool'>`."""
        if program_name is None:
            attrs: bytes = DEFAULT_CONNECT_ATTRS + gen_connect_attrs(  # type: ignore
                ["_pid", str(getpid)]
            )
        else:
            attrs: bytes = DEFAULT_CONNECT_ATTRS + gen_connect_attrs(  # type: ignore
                ["_pid", str(getpid), "program_name", program_name]
            )
        self._connect_attrs = gen_length_encoded_integer(bytes_len(attrs)) + attrs  # type: ignore
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _init_internal(self) -> cython.bint:
        """(cfunc) Initialize internal attributes `<'bool'>`."""
        # . server
        self._server_protocol_version = -1
        self._server_info = None
        self._server_version = None
        self._server_version_major = -1
        self._server_vendor = None
        self._server_thred_id = -1
        self._server_salt = None
        self._server_status = -1
        self._server_capabilities = -1
        self._server_auth_plugin_name = None
        # . client
        self._set_use_time()
        self._secure = False
        self._host_info = None
        self._close_reason = None
        # . query
        self._result = None
        self._next_seq_id = 0
        # . transport
        self._writer = None
        self._reader = None
        return True

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
        return decode_bytes(self._user, self._encoding_c)

    @property
    def password(self) -> str:
        """The 'password' for user authentication `<'str'>`."""
        return decode_bytes(self._password, "latin1")

    @property
    def database(self) -> str | None:
        """The 'database' to be used by the client `<'str/None'>`."""
        if self._database is None:
            return None
        return decode_bytes(self._database, self._encoding_c)

    @property
    def charset(self) -> str:
        """The 'charactor set' to be used by the client `<'str'>`."""
        return self._charset

    @property
    def collation(self) -> str:
        """The 'collation' to be used by the client `<'str'>`."""
        return self._collation

    @property
    def encoding(self) -> str:
        """The 'encoding' to be used by the client `<'str'>`."""
        return decode_bytes(self._encoding, "ascii")

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
        if self._server_status == -1:
            return None if self._autocommit_mode == -1 else bool(self._autocommit_mode)
        else:
            return bool(self._server_status & _SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT)

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
    def host_info(self) -> str:
        """The information about the server `<'str'>`."""
        return "Not connected" if self._host_info is None else self._host_info

    @property
    def ssl(self) -> object | None:
        """The 'ssl.SSLContext' to be used for secure connection `<'SSLContext/None'>`."""
        return self._ssl_ctx

    @property
    def auth_plugin(self) -> AuthPlugin | None:
        """The 'auth_plugin' to be used for authentication `<'AuthPlugin/None'>`."""
        return self._auth_plugin

    # . server
    @property
    def thread_id(self) -> int | None:
        """The thread id from the server `<'int/None'>`."""
        if self._server_thred_id != -1:
            return self._server_thred_id
        else:
            return None

    @property
    def transaction_status(self) -> bool | None:
        if self._server_status != -1:
            return self.get_transaction_status()
        else:
            return None

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
        return self.get_server_version()

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
        return self.get_server_vendor()

    @property
    def server_status(self) -> int | None:
        """The server status `<'int/None'>`."""
        if self._server_status != -1:
            return self._server_status
        else:
            return None

    @property
    def server_capabilites(self) -> int | None:
        """The server capabilities `<'int/None'>`."""
        if self._server_capabilities != -1:
            return self._server_capabilities
        else:
            return None

    @property
    def server_auth_plugin_name(self) -> str | None:
        """The server authentication plugin name `<'str/None'>`."""
        return self._server_auth_plugin_name

    # . query
    @property
    def affected_rows(self) -> int:
        """The number of affected rows by the last query `<'int'>`."""
        return self.get_affected_rows()

    @property
    def insert_id(self) -> int:
        """The id of the last inserted row `<'int'>`."""
        return self.get_insert_id()

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

    # Cursor ----------------------------------------------------------------------------------
    @cython.ccall
    def cursor(self, cursor: type[Cursor] = None) -> CursorManager:
        self._verify_connected()
        return CursorManager(
            self,
            self._cursor if cursor is None else validate_cursor(cursor),  # type: ignore
        )

    @cython.ccall
    def transaction(self, cursor: type[Cursor] = None) -> TransactionManager:
        self._verify_connected()
        return TransactionManager(
            self,
            self._cursor if cursor is None else validate_cursor(cursor),  # type: ignore
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_use_time(self) -> cython.bint:
        self._last_used_time = unix_time()
        return True

    # Query -----------------------------------------------------------------------------------
    @cython.ccall
    def query(self, sql: str, unbuffered: cython.bint = False) -> cython.ulonglong:
        self._execute_command(_COMMAND.COM_QUERY, self.encode_sql(sql))
        return self._read_query_result(unbuffered)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def begin(self) -> cython.bint:
        """Begin transaction `<'bool'>`."""
        self._execute_command(_COMMAND.COM_QUERY, b"BEGIN")
        self._read_ok_packet()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def commit(self) -> cython.bint:
        """Commit changes to stable storage.

        See `Connection.commit() <https://www.python.org/dev/peps/pep-0249/#commit>`_
        in the specification.
        """
        self._execute_command(_COMMAND.COM_QUERY, b"COMMIT")
        self._read_ok_packet()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def rollback(self) -> cython.bint:
        """Roll back the current transaction.

        See `Connection.rollback() <https://www.python.org/dev/peps/pep-0249/#rollback>`_
        in the specification.
        """
        self._execute_command(_COMMAND.COM_QUERY, b"ROLLBACK")
        self._read_ok_packet()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_charset(
        self,
        charset: str | None,
        collation: str | None = None,
    ) -> cython.bint:
        """Set charset and collation[Optional] `<'bool'>`.

        Send "SET NAMES charset [COLLATE collation]" query.
        Update Connection.encoding based on charset.
        """
        ch: Charset = validate_charset(charset, collation)  # type: ignore
        if ch._name != self._charset or ch._collation != self._collation:
            self._init_charset(ch)
            sql = "SET NAMES %s COLLATE %s" % (self._charset, self._collation)
            self._execute_command(_COMMAND.COM_QUERY, self.encode_sql(sql))
            self._read_packet()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_read_timeout(self, value: int | None) -> cython.bint:
        name = "net_read_timeout"
        # Validate timeout
        value = validate_arg_uint(value, name, 1, UINT_MAX)  # type: ignore
        if value is None:
            # Global timeout
            if self._read_timeout is None:
                value = self._get_timeout(name, False)
            # Setting timeout
            else:
                value = self._read_timeout

        # Set timeout
        return self._set_timeout(name, value)

    @cython.ccall
    def get_read_timeout(self) -> cython.uint:
        return self._get_timeout("net_read_timeout", True)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_write_timeout(self, value: int | None) -> cython.bint:
        name = "net_write_timeout"
        # Validate timeout
        value = validate_arg_uint(value, name, 1, UINT_MAX)  # type: ignore
        if value is None:
            # Global timeout
            if self._write_timeout is None:
                value = self._get_timeout(name, False)
            # Setting timeout
            else:
                value = self._write_timeout

        # Set timeout
        return self._set_timeout(name, value)

    @cython.ccall
    def get_write_timeout(self) -> cython.uint:
        return self._get_timeout("net_write_timeout", True)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_wait_timeout(self, value: int | None) -> cython.bint:
        name = "wait_timeout"
        # Validate timeout
        value = validate_arg_uint(value, name, 1, UINT_MAX)  # type: ignore
        if value is None:
            # Global timeout
            if self._wait_timeout is None:
                value = self._get_timeout(name, False)
            # Setting timeout
            else:
                value = self._wait_timeout

        # Set timeout
        return self._set_timeout(name, value)

    @cython.ccall
    def get_wait_timeout(self) -> cython.uint:
        return self._get_timeout("wait_timeout", True)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_timeout(self, name: str, value: object) -> cython.bint:
        with self.cursor() as cur:
            cur.execute("SET SESSION %s = %s" % (name, value))
        return True

    @cython.cfunc
    @cython.inline(True)
    def _get_timeout(self, name: str, session: cython.bint) -> cython.uint:
        with self.cursor() as cur:
            cur.execute(
                "SHOW VARIABLES LIKE '%s'" % name
                if session
                else "SHOW GLOBAL VARIABLES LIKE '%s'" % name
            )
            try:
                return int(cur.fetchone()[1])
            except Exception as err:
                raise errors.DatabaseError(
                    "Failed to get %s '%s' from server: %s"
                    % ("SESSION" if session else "GLOBAL", name, err)
                ) from err

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_autocommit(self) -> cython.bint:
        if self._server_status == -1:
            raise errors.ConnectionClosedError(0, "Connection not connected.")
        return self._server_status & _SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_autocommit(self, auto: cython.bint) -> cython.bint:
        if auto != self.get_autocommit():
            self._execute_command(
                _COMMAND.COM_QUERY,
                b"SET AUTOCOMMIT = 1" if auto else b"SET AUTOCOMMIT = 0",
            )
            self._read_ok_packet()
            self._autocommit_mode = auto
        return True

    @cython.ccall
    def show_warnings(self) -> tuple:
        """Send the "SHOW WARNINGS" SQL command."""
        self._execute_command(_COMMAND.COM_QUERY, b"SHOW WARNINGS")
        result = MysqlResult(self)
        result.read()
        return result.rows

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def select_database(self, db: str) -> cython.bint:
        """Set current db.

        :param db: The name of the db.
        """
        database: bytes = self.encode_sql(db)
        self._execute_command(_COMMAND.COM_INIT_DB, database)
        self._read_ok_packet()
        self._database = database
        return True

    @cython.ccall
    def get_affected_rows(self) -> cython.ulonglong:
        return 0 if self._result is None else self._result.affected_rows

    @cython.ccall
    def get_insert_id(self) -> cython.ulonglong:
        return 0 if self._result is None else self._result.insert_id

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_transaction_status(self) -> cython.bint:
        """Whether the connection is in transaction `<'bool'>`."""
        if self._server_status == -1:
            raise errors.ConnectionClosedError(0, "Connection not connected.")
        return self._server_status & _SERVER_STATUS.SERVER_STATUS_IN_TRANS

    @cython.ccall
    def get_server_version(self) -> tuple[int]:
        if self._server_version is None:
            # Not connected
            if self._server_info is None:
                return None
            # Parser version
            m = SERVER_VERSION_RE.match(self._server_info)
            if m is None:
                raise errors.DatabaseError(
                    "Failed to parse server version from: %s." % self._server_info
                )
            cmp: tuple = m.group(1, 2, 3)
            self._server_version = list_to_tuple(
                [0 if i is None else int(i) for i in cmp]
            )
        return self._server_version

    @cython.ccall
    def get_server_vendor(self) -> str:
        if self._server_vendor is None:
            # Not connected
            if self._server_info is None:
                return None
            # Parser vendor
            if "mariadb" in self._server_info.lower():
                self._server_vendor = "mariadb"
            else:
                self._server_vendor = "mysql"
        return self._server_vendor

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_use_decimal(self, value: cython.bint) -> cython.bint:
        if self._use_decimal != value:
            self._use_decimal = value
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_decode_json(self, value: cython.bint) -> cython.bint:
        if self._decode_json != value:
            self._decode_json = value
        return True

    @cython.ccall
    def escape_args(self, args: object, itemize: cython.bint = True) -> object:
        """Escape object into string `<'str/tuple'>`."""
        return encode_item(args) if itemize else escape_item(args)

    @cython.ccall
    def encode_sql(self, sql: str) -> bytes:
        return encode_str(sql, self._encoding_c)  # type: ignore

    # Connect / Close -------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def connect(self) -> cython.bint:
        """Establish connection with database server `<'bool'>`."""
        if self.closed():
            self._connect()
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _connect(self) -> cython.bint:
        """Establish connection with database server `<'bool'>`."""
        # Connect
        sock = None
        try:
            # . create socket
            if self._unix_socket is not None:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self._connect_timeout)
                sock.connect(self._unix_socket)
                self._host_info = "Localhost via UNIX socket"
                self._secure = True
            else:
                if self._bind_address is not None:
                    source_address = (self._bind_address, 0)
                else:
                    source_address = None
                while True:
                    try:
                        sock = socket.create_connection(
                            (self._host, self._port),
                            self._connect_timeout,
                            source_address=source_address,
                        )
                        break
                    except OSError as err:
                        if err.errno == errno.EINTR:
                            continue
                        raise
                self._host_info = "socket %s:%s" % (self._host, self._port)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(None)

            # . set socket
            self._writer = sock
            self._reader = sock.makefile("rb")

            # . handshake
            self._next_seq_id = 0
            self._setup_server_information()
            self._request_authentication()

            # Send "SET NAMES" query on init for:
            # - Ensure charaset (and collation) is set to the server.
            #   - collation_id in handshake packet may be ignored.
            # - If collation is not specified, we don't know what is server's
            #   default collation for the charset. For example, default collation
            #   of utf8mb4 is:
            #   - MySQL 5.7, MariaDB 10.x: utf8mb4_general_ci
            #   - MySQL 8.0: utf8mb4_0900_ai_ci
            #
            # Reference:
            # - https://github.com/PyMySQL/PyMySQL/issues/1092
            # - https://github.com/wagtail/wagtail/issues/9477
            # - https://zenn.dev/methane/articles/2023-mysql-collation (Japanese)
            sql = "SET NAMES %s COLLATE %s" % (self._charset, self._collation)
            self._execute_command(_COMMAND.COM_QUERY, self.encode_sql(sql))  # type: ignore
            self._read_packet()

            # . timeouts
            if self._read_timeout is not None:
                self._set_timeout("net_read_timeout", self._read_timeout)
            if self._write_timeout is not None:
                self._set_timeout("net_write_timeout", self._write_timeout)
            if self._wait_timeout is not None:
                self._set_timeout("wait_timeout", self._wait_timeout)

            # . sql mode
            if self._sql_mode is not None:
                self.query("SET sql_mode=%s" % self._sql_mode, False)

            # . init command
            if self._init_command is not None:
                self.query(self._init_command, False)
                self.commit()

            # . autocommit
            if (
                self._autocommit_mode != -1
                and self._autocommit_mode != self.get_autocommit()
            ):
                self._execute_command(
                    _COMMAND.COM_QUERY,
                    (
                        b"SET AUTOCOMMIT = 1"
                        if self._autocommit_mode
                        else b"SET AUTOCOMMIT = 0"
                    ),
                )
                self._read_ok_packet()

        # Failed
        except BaseException as err:
            if sock is not None:
                try:
                    sock.close()
                except:  # noqa
                    pass
            self._writer = None
            self._reader = None
            if isinstance(err, (OSError, IOError)):
                raise errors.OpenConnectionError(
                    CR.CR_CONN_HOST_ERROR,
                    "Can't connect to server on '%s' (%s)" % (self._host, err),
                ) from err
            # If err is neither DatabaseError or IOError, It's a bug.
            # But raising AssertionError hides original error.
            # So just reraise it.
            raise err

        # Success
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def close(self) -> cython.bint:
        """Send the quit message and close the socket.

        See `Connection.close() <https://www.python.org/dev/peps/pep-0249/#Connection.close>`_
        in the specification.

        No error will be raised.
        """
        # Already closed
        if self.closed():
            return True
        # Close connection
        try:
            self._write_bytes(pack_IB(1, _COMMAND.COM_QUIT))  # type: ignore
        finally:
            self.force_close()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def force_close(self) -> cython.bint:
        if not self.closed():
            try:
                self._writer.close()
            except:  # noqa
                pass
        self._writer = None
        self._reader = None
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _close_with_reason(self, reason: str) -> cython.bint:
        self._close_reason = "Connection closed: %s" % reason
        self.force_close()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def closed(self) -> cython.bint:
        """The connection status, whether is closed `<'bool'>`."""
        return self._writer is None

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def kill(self, thread_id: cython.int) -> cython.bint:
        try:
            self._execute_command(_COMMAND.COM_PROCESS_KILL, pack_int32(thread_id))
            self._read_ok_packet()
        except errors.OperationalUnknownCommandError:
            # if COM_PROCESS_KILL [0x0C] gets 'unknown command' error,
            # try with 'kill {thread_id}' sql.
            with self.cursor() as cur:
                cur.execute("KILL %d" % thread_id)
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def ping(self, reconnect: cython.bint = True) -> cython.bint:
        """Check if the server is alive.

        :param reconnect: If the connection is closed, reconnect.
        :type reconnect: boolean

        :raise Error: If the connection is closed and reconnect=False.
        """
        if self.closed():
            if not reconnect:
                raise errors.ConnectionClosedError(0, "Connection already closed.")
            self._connect()
            reconnect = False
        try:
            self._execute_command(_COMMAND.COM_PING, b"")
            self._read_ok_packet()
            return True
        except Exception:
            if not reconnect:
                raise
            self._connect()
            return self.ping(False)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _setup_server_information(self) -> cython.bint:
        # Read from packet
        pkt: MysqlPacket = self._read_packet()
        data: cython.pchar = pkt._data_c
        length: cython.Py_ssize_t = pkt._size

        # . protocol version
        self._server_protocol_version = unpack_uint8(data, 0)

        # . server version
        loc: cython.Py_ssize_t = find_null_term(data, 1)  # type: ignore
        self._server_info = decode_bytes(data[1:loc], "latin1")
        self._server_version_major = int(str_split(self._server_info, ".", 1)[0])
        i: cython.Py_ssize_t = loc + 1

        # . server_thred_id
        self._server_thred_id = unpack_uint32(data, i)
        i += 4

        # . salt
        self._server_salt = data[i : i + 8]
        i += 9  # 8 + 1(filler)

        # . server capabilities
        self._server_capabilities = unpack_uint16(data, i)
        i += 2

        # . server_status & adj server_capabilities
        salt_len: cython.Py_ssize_t
        if length >= i + 6:
            i += 1
            self._server_status = unpack_uint16(data, i)
            i += 2
            self._server_capabilities |= unpack_uint16(data, i) << 16
            i += 2
            salt_len = max(12, unpack_uint8(data, i) - 9)
            i += 1
        else:
            salt_len = 0

        # . reserved
        i += 10

        # . adj sale
        if length >= i + salt_len:
            # salt_len includes auth_plugin_data_part_1 and filler
            self._server_salt += data[i : i + salt_len]
            i += salt_len
        i += 1

        # . auth_plugin_name
        if self._server_capabilities & _CLIENT.PLUGIN_AUTH and length > i:
            # Due to Bug#59453 the auth-plugin-name is missing the terminating
            # NUL-char in versions prior to 5.5.10 and 5.6.2.
            # ref: https://dev.mysql.com/doc/internals/en/connection-phase-packets.html#packet-Protocol::Handshake
            # didn't use version checks as mariadb is corrected and reports
            # earlier than those two.
            loc = find_null_term(data, i)  # type: ignore
            if loc < 0:  # pragma: no cover - very specific upstream bug
                # not found \0 and last field so take it all
                auth_plugin_name: bytes = data[i:length]
            else:
                auth_plugin_name: bytes = data[i:loc]
            self._server_auth_plugin_name = decode_bytes_utf8(auth_plugin_name)

        # Finished
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _request_authentication(self) -> cython.bint:
        # https://dev.mysql.com/doc/internals/en/connection-phase-packets.html#packet-Protocol::HandshakeResponse
        # . mysql version 5+
        if self._server_version_major >= 5:
            self._client_flag |= _CLIENT.MULTI_RESULTS

        # . validate username
        if self._user is None:
            raise errors.InvalidConnectionArgsError(
                "<'%s'>\nUsername 'user' is not specified." % self.__class__.__name__
            )

        # . init data
        data: bytes = pack_IIB23s(  # type: ignore
            self._client_flag, MAX_PACKET_LENGTH, self._charset_id
        )

        # . ssl connection
        if self._ssl_ctx is not None and self._server_capabilities & _CLIENT.SSL:
            self._write_packet(data)
            self._writer = self._ssl_ctx.wrap_socket(
                self._writer, server_hostname=self._host
            )
            self._reader = self._writer.makefile("rb")
            self._secure = True

        # . collect
        data += self._user + b"\0"

        # . auth
        plugin_name: bytes
        authres: bytes
        if self._server_auth_plugin_name == "":
            plugin_name = b""
            authres = _auth.scramble_native_password(self._password, self._server_salt)
        elif self._server_auth_plugin_name == "mysql_native_password":
            plugin_name = b"mysql_native_password"
            authres = _auth.scramble_native_password(self._password, self._server_salt)
        elif self._server_auth_plugin_name == "caching_sha2_password":
            plugin_name = b"caching_sha2_password"
            if self._password:
                authres = _auth.scramble_caching_sha2(self._password, self._server_salt)
            else:
                authres = b""
        elif self._server_auth_plugin_name == "sha256_password":
            plugin_name = b"sha256_password"
            if self._ssl_ctx is not None and self._server_capabilities & _CLIENT.SSL:
                authres = self._password + b"\0"
            elif self._password:
                authres = b"\1"  # request public key
            else:
                authres = b"\0"  # empty password
        else:
            plugin_name = None
            authres = b""
        if self._server_capabilities & _CLIENT.PLUGIN_AUTH_LENENC_CLIENT_DATA:
            data += gen_length_encoded_integer(bytes_len(authres)) + authres  # type: ignore
        elif self._server_capabilities & _CLIENT.SECURE_CONNECTION:
            data += pack_uint8(bytes_len(authres)) + authres
        else:
            data += authres + b"\0"

        # . database
        if (
            self._database is not None
            and self._server_capabilities & _CLIENT.CONNECT_WITH_DB
        ):
            data += self._database + b"\0"

        # . auth plugin name
        if self._server_capabilities & _CLIENT.PLUGIN_AUTH:
            data += (b"" if plugin_name is None else plugin_name) + b"\0"

        # . connect attrs
        if self._server_capabilities & _CLIENT.CONNECT_ATTRS:
            data += self._connect_attrs

        # . write packet
        self._write_packet(data)
        auth_pkt = self._read_packet()

        # if authentication method isn't accepted the first byte
        # will have the octet 254
        if auth_pkt.read_auth_switch_request():
            auth_pkt = self._process_authentication(auth_pkt)
        elif auth_pkt.is_extra_auth_data():
            # https://dev.mysql.com/doc/internals/en/successful-authentication.html
            if self._server_auth_plugin_name == "caching_sha2_password":
                auth_pkt = self._process_auth_caching_sha2(auth_pkt)
            elif self._server_auth_plugin_name == "sha256_password":
                auth_pkt = self._process_auth_sha256(auth_pkt)
            else:
                raise errors.AuthenticationError(
                    "Received extra packet for auth method: %s.",
                    self._server_auth_plugin_name,
                )

        # Success
        return True

    @cython.cfunc
    @cython.inline(True)
    def _process_authentication(self, auth_pkt: MysqlPacket) -> MysqlPacket:
        """(cfunc) Process authentication response `<'MysqlPacket'>`."""
        # Validate plugin name & capabilities
        plugin_name: bytes = auth_pkt._plugin_name
        if plugin_name is None or not self._server_capabilities & _CLIENT.PLUGIN_AUTH:
            raise errors.AuthenticationError("received unknown auth switch request.")
        # Custom auth plugin handler
        plugin_handler = None
        if self._auth_plugin is not None:
            if (plugin_class := self._auth_plugin.get(plugin_name)) is not None:
                # . instantiate handler
                try:
                    plugin_handler = plugin_class(self)
                except Exception as err:
                    raise errors.AuthenticationError(
                        CR.CR_AUTH_PLUGIN_CANNOT_LOAD,
                        "Authentication plugin '%s' not loaded: "
                        "%r cannot be constructed with connection object."
                        % (plugin_name.decode("ascii"), plugin_handler),
                    ) from err
                # . try with custom handler
                try:
                    return plugin_handler.authenticate(auth_pkt)
                except AttributeError as err:
                    # . leave dialog to the section below
                    if plugin_name != b"dialog":
                        raise errors.AuthenticationError(
                            CR.CR_AUTH_PLUGIN_CANNOT_LOAD,
                            "Authentication plugin '%s' not loaded: "
                            "%r missing 'authenticate()' method."
                            % (plugin_name.decode("ascii"), plugin_handler),
                        ) from err
        # Process auth
        if plugin_name == b"caching_sha2_password":
            return self._process_auth_caching_sha2(auth_pkt)
        elif plugin_name == b"sha256_password":
            return self._process_auth_sha256(auth_pkt)
        elif plugin_name == b"mysql_native_password":
            data = _auth.scramble_native_password(self._password, auth_pkt._salt)
        elif plugin_name == b"client_ed25519":
            data = _auth.ed25519_password(self._password, auth_pkt._salt)
        elif plugin_name == b"mysql_clear_password":
            # https://dev.mysql.com/doc/internals/en/clear-text-authentication.html
            data = self._password + b"\0"
        elif plugin_name == b"dialog":
            pkt: MysqlPacket = auth_pkt
            while True:
                flag: cython.uint = pkt._read_uint8()
                prompt: bytes = pkt.read_remains()
                if prompt == b"Password: ":
                    self._write_packet(self._password + b"\0")
                elif plugin_handler is not None:
                    resp: bytes = b"no response - TypeError within plugin.prompt method"
                    try:
                        resp = plugin_handler.prompt((flag & 0x06) == 0x02, prompt)
                        self._write_packet(resp + b"\0")
                    except AttributeError as err:
                        raise errors.AuthenticationError(
                            CR.CR_AUTH_PLUGIN_CANNOT_LOAD,
                            "Authentication plugin '%s' not loaded: "
                            "%r missing 'prompt()' method."
                            % (plugin_name.decode("ascii"), plugin_handler),
                        ) from err
                    except TypeError as err:
                        raise errors.AuthenticationError(
                            # fmt: off
                            CR.CR_AUTH_PLUGIN_ERR,
                            "Authentication plugin '%s' didn't respond with string:"
                            "%r returns '%r' to prompt: %r"
                            % ( plugin_name.decode("ascii"), 
                                plugin_handler, resp, prompt ),
                            # fmt: on
                        ) from err
                else:
                    raise errors.AuthenticationError(
                        CR.CR_AUTH_PLUGIN_CANNOT_LOAD,
                        "Authentication plugin '%s' not configured."
                        % plugin_name.decode("ascii"),
                    )
                pkt = self._read_packet()
                pkt.check_error()
                if pkt.is_ok_packet() or (flag & 0x01) == 0x01:
                    break
            return pkt
        else:
            raise errors.AuthenticationError(
                CR.CR_AUTH_PLUGIN_CANNOT_LOAD,
                "Authentication plugin '%s' not configured"
                % plugin_name.decode("ascii"),
            )
        # Auth: 'mysql_native_password', 'client_ed25519' & 'mysql_clear_password'.
        self._write_packet(data)
        pkt = self._read_packet()
        pkt.check_error()
        return pkt

    @cython.cfunc
    @cython.inline(True)
    def _process_auth_caching_sha2(self, pkt: MysqlPacket) -> MysqlPacket:
        """(cfunc) Process 'caching_sha2_password' authentication
        response `<'MysqlPacket'>`."""
        # No password fast path
        if self._password is None:
            return self._process_auth_send_data(b"")

        # Try from fast auth
        if pkt.is_auth_switch_request():
            self._server_salt = pkt._salt
            scrambled = _auth.scramble_caching_sha2(self._password, self._server_salt)
            pkt = self._process_auth_send_data(scrambled)
            # else: fast auth is tried in initial handshake

        # Process auth response
        if not pkt.is_extra_auth_data():
            raise errors.AuthenticationError(
                "Auth error for 'caching sha2': "
                "Unknown packet for fast auth:\n%s" % pkt.read_all_data()
            )
        pkt.advance(1)
        n: cython.uint = pkt._read_uint8()
        # . fast auth succeeded: 3
        if n == 3:
            pkt = self._read_packet()
            pkt.check_error()
            return pkt
        # . unknown result, magic numbers:
        # 2 - request public key
        # 3 - fast auth succeeded
        # 4 - need full auth
        if n != 4:
            raise errors.AuthenticationError(
                "Auth error for 'caching sha2': "
                "Unknown result for fast auth: %d." % n
            )
        # . full auth: 4
        if self._secure:
            # caching sha2: send plain password via secure connection
            return self._process_auth_send_data(self._password + b"\0")
        if self._server_public_key is None:
            # request public key from server
            pkt = self._process_auth_send_data(b"\x02")
            if not pkt.is_extra_auth_data():
                raise errors.AuthenticationError(
                    "Auth error for 'caching sha2': "
                    "Unknown packet for public key:\n%s" % pkt.read_all_data()
                )
            pkt.advance(1)
            self._server_public_key = pkt.read_remains()
        data = _auth.sha2_rsa_encrypt(
            self._password, self._server_salt, self._server_public_key
        )
        return self._process_auth_send_data(data)

    @cython.cfunc
    @cython.inline(True)
    def _process_auth_sha256(self, pkt: MysqlPacket) -> MysqlPacket:
        """(cfunc) Process 'sha256_password' authentication response `<'MysqlPacket'>`."""
        if self._secure:
            # sha256: send plain password via secure connection
            return self._process_auth_send_data(self._password + b"\0")

        if pkt.is_auth_switch_request():
            self._server_salt = pkt._salt
            if self._server_public_key is None and self._password is None:
                # Request server public key
                pkt = self._process_auth_send_data(b"\1")

        if pkt.is_extra_auth_data():
            pkt.advance(1)
            self._server_public_key = pkt.read_remains()

        if self._password is not None:
            if self._server_public_key is None:
                raise errors.AuthenticationError(
                    "Auth error for 'sha256': Couldn't receive server's public key"
                )
            data = _auth.sha2_rsa_encrypt(
                self._password, self._server_salt, self._server_public_key
            )
        else:
            data = b""
        return self._process_auth_send_data(data)

    @cython.cfunc
    @cython.inline(True)
    def _process_auth_send_data(self, data: bytes) -> MysqlPacket:
        """(cfunc) Process authentication - send data `<'MysqlPacket'>`."""
        self._write_packet(data)
        pkt: MysqlPacket = self._read_packet()
        pkt.check_error()
        return pkt

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_connected(self) -> cython.bint:
        "(cfunc) Verify if the connection is connected."
        if self.closed():
            if self._close_reason is None:
                raise errors.ConnectionClosedError(0, "Connection not connected.")
            else:
                raise errors.ConnectionClosedError(0, self._close_reason)
        return True

    # Write -----------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _execute_command(self, command: cython.uint, sql: bytes) -> cython.bint:
        """
        :raise InterfaceError: If the connection is closed.
        :raise ValueError: If no username was specified.
        """
        # Validate connection
        self._verify_connected()

        # If the last query was unbuffered, make sure it finishes
        # before sending new commands
        if self._result is not None:
            if self._result.unbuffered_active:
                warnings.warn("Previous unbuffered result was left incomplete.")
                self._result._drain_result_packet_unbuffered()
            while self._result.has_next:
                self._read_query_result(False)  # next result
            self._result = None

        # Tiny optimization: build first packet manually:
        # using struct.pack('<IB') where 'I' is composed with
        # - payload_length <-> int24 (3-bytes)
        # - sequence_id 0  <-> int8  (1-bytes)
        # and let 'B' falls into the 1st bytes of the payload
        # which is the command.
        sql_size: cython.ulonglong = bytes_len(sql)
        pkt_size: cython.uint = min(
            sql_size + 1, MAX_PACKET_LENGTH
        )  # +1 is for command
        data = pack_IB(pkt_size, command) + sql[0 : pkt_size - 1]  # type: ignore
        self._write_bytes(data)
        self._next_seq_id = 1

        # Write remaining data
        pos: cython.ulonglong = pkt_size - 1
        while pos < sql_size:
            pkt_size = min(sql_size - pos, MAX_PACKET_LENGTH)
            self._write_packet(sql[pos : pos + pkt_size])
            pos += pkt_size

        # Set use time & exit
        self._set_use_time()
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _write_packet(self, payload: bytes) -> cython.bint:
        """Writes an entire 'packet' in its entirety to the network
        adding its length and sequence number.
        """
        # Internal note: when you build packet manually and calls _write_bytes()
        # directly, you should set self._next_seq_id properly.
        data = pack_I24B(bytes_len(payload), self._next_seq_id) + payload  # type: ignore
        self._write_bytes(data)
        self._next_seq_id = (self._next_seq_id + 1) % 256
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _write_bytes(self, data: bytes) -> cython.bint:
        try:
            self._writer.sendall(data)
        except OSError as err:
            msg: str = "Server has gone away (%s)" % err
            self._close_with_reason(msg)
            raise errors.ConnectionLostError(CR.CR_SERVER_GONE_ERROR, msg) from err
        except BaseException as err:
            self._close_with_reason(str(err))
            raise err
        return True

    # Read ------------------------------------------------------------------------------------
    @cython.ccall
    def next_result(self, unbuffered: cython.bint = False) -> cython.ulonglong:
        """Go to the next query result, and returns affected rows `<'int'>`."""
        return self._read_query_result(unbuffered)

    @cython.cfunc
    @cython.inline(True)
    def _read_ok_packet(self) -> MysqlPacket:
        pkt = self._read_packet()
        if not pkt.read_ok_packet():
            raise errors.CommandOutOfSyncError(
                CR.CR_COMMANDS_OUT_OF_SYNC, "Command Out of Sync."
            )
        self._server_status = pkt._server_status
        return pkt

    @cython.cfunc
    @cython.inline(True)
    def _read_query_result(self, unbuffered: cython.bint) -> cython.ulonglong:
        self._result = None
        if unbuffered:
            try:
                result = MysqlResult(self)
                result.init_unbuffered_query()
            except:  # noqa
                result.unbuffered_active = False
                result._conn = None
                raise
        else:
            result = MysqlResult(self)
            result.read()
        self._result = result
        if result.server_status != -1:
            self._server_status = result.server_status
        return result.affected_rows

    @cython.cfunc
    @cython.inline(True)
    def _read_packet(self) -> MysqlPacket:
        """Read the entire 'packet' in its entirety
        from the network and returns `<'MysqlPacket'>`.

        :raise OperationalError: If the connection to the server is lost.
        :raise InternalError: If the packet sequence number is wrong.
        """
        buffer: bytes = self._read_packet_buffer()
        pkt = MysqlPacket(buffer, self._encoding)
        if pkt.is_error_packet():
            if self._result is not None and self._result.unbuffered_active:
                self._result.unbuffered_active = False
            pkt.raise_for_error()
        return pkt

    @cython.cfunc
    @cython.inline(True)
    def _read_field_descriptor_packet(self) -> FieldDescriptorPacket:
        """Read the entire 'packet' in its entirety
        from the network and returns `<'FieldDescriptorPacket'>`.

        :raise OperationalError: If the connection to the server is lost.
        :raise InternalError: If the packet sequence number is wrong.
        """
        buffer: bytes = self._read_packet_buffer()
        pkt = FieldDescriptorPacket(buffer, self._encoding)
        if pkt.is_error_packet():
            if self._result is not None and self._result.unbuffered_active:
                self._result.unbuffered_active = False
            pkt.raise_for_error()
        return pkt

    @cython.cfunc
    @cython.inline(True)
    def _read_packet_buffer(self) -> bytes:
        """Read the entire 'packet' in its entirety from the network
        and return the data buffer `<'bytes'>`.

        :raise OperationalError: If the connection to the server is lost.
        :raise InternalError: If the packet sequence number is wrong.
        """
        # Read buffer data
        buffer: bytearray = bytearray()
        while True:
            data: bytes = self._read_bytes(4)
            packet_header: cython.pchar = data
            btrl: cython.uint = unpack_uint16(packet_header, 0)
            btrh: cython.uint = unpack_uint8(packet_header, 2)
            packet_number: cython.uint = unpack_uint8(packet_header, 3)
            bytes_to_read: cython.uint = btrl + (btrh << 16)
            if packet_number != self._next_seq_id:
                if packet_number == 0:
                    # MariaDB sends error packet with seqno==0 when shutdown
                    msg: str = "Lost connection to server during query"
                    self._close_with_reason(msg)
                    raise errors.ConnectionLostError(CR.CR_SERVER_LOST, msg)
                else:
                    msg: str = "Packet sequence number wrong - got %d expected %d" % (
                        packet_number,
                        self._next_seq_id,
                    )
                    self._close_with_reason(msg)
                    raise errors.InternalError(0, msg)
            self._next_seq_id = (self._next_seq_id + 1) % 256
            recv_data: bytes = self._read_bytes(bytes_to_read)
            buffer += recv_data
            # https://dev.mysql.com/doc/internals/en/sending-more-than-16mbyte.html
            if bytes_to_read < MAX_PACKET_LENGTH:
                break

        # Return buffer
        return bytes(buffer)

    @cython.cfunc
    @cython.inline(True)
    def _read_bytes(self, size: cython.uint) -> bytes:
        """Read 'size' bytes from the network and return the bytes read `<'bytes'>`.

        :raise OperationalError: If the connection to the server is lost.
        """
        while True:
            try:
                data: bytes = self._reader.read(size)
                break
            except OSError as err:
                if err.errno == errno.EINTR:
                    continue
                msg: str = "Lost connection to server during query (%s)." % err
                self._close_with_reason(msg)
                raise errors.ConnectionLostError(CR.CR_SERVER_LOST, msg) from err
            except BaseException as err:
                # Don't convert unknown exception to MySQLError.
                self._close_with_reason(str(err))
                raise err

        if bytes_len(data) < size:
            msg: str = "Lost connection to server during query (reading from server)."
            self._close_with_reason(msg)
            raise errors.ConnectionLostError(CR.CR_SERVER_LOST, msg)
        return data

    # Special methods -------------------------------------------------------------------------
    def __enter__(self) -> BaseConnection:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.force_close()


@cython.cclass
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
    ):
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
        self._init_charset(validate_charset(charset, collation)) # type: ignore
        # . basic
        self._host = validate_arg_str(host, "host", "localhost")  # type: ignore
        self._port = validate_arg_uint(port, "port", 1, 65_535)  # type: ignore
        self._user = validate_arg_bytes(user, "user", self._encoding_c, DEFAULT_USER)  # type: ignore
        self._password = validate_arg_bytes(password, "password", "latin1", "")  # type: ignore
        self._database = validate_arg_bytes(database, "database", self._encoding_c, None)  # type: ignore
        # . timeouts
        self._connect_timeout = validate_arg_uint(connect_timeout, "connect_timeout", 1, MAX_CONNECT_TIMEOUT) # type: ignore
        self._read_timeout = validate_arg_uint(read_timeout, "read_timeout", 1, UINT_MAX) # type: ignore
        self._write_timeout = validate_arg_uint(write_timeout, "write_timeout", 1, UINT_MAX) # type: ignore
        self._wait_timeout = validate_arg_uint(wait_timeout, "wait_timeout", 1, UINT_MAX) # type: ignore
        # . client
        self._bind_address = validate_arg_str(bind_address, "bind_address", None)  # type: ignore
        self._unix_socket = validate_arg_str(unix_socket, "unix_socket", None)  # type: ignore
        self._autocommit_mode = validate_autocommit(autocommit)  # type: ignore
        self._local_infile = bool(local_infile)
        self._max_allowed_packet = validate_max_allowed_packet(max_allowed_packet)  # type: ignore
        self._sql_mode = validate_sql_mode(sql_mode)  # type: ignore
        self._init_command = validate_arg_str(init_command, "init_command", None)  # type: ignore
        self._cursor = validate_cursor(cursor)  # type: ignore
        self._init_client_flag(validate_arg_uint(client_flag, "client_flag", 0, UINT_MAX))  # type: ignore
        self._init_connect_attrs(validate_arg_str(program_name, "program_name", None)) # type: ignore
        # . ssl
        self._ssl_ctx = validate_ssl(ssl)  # type: ignore
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
        self._server_public_key = validate_arg_bytes( # type: ignore
            server_public_key, "server_public_key", "ascii", None) 
        # . decode
        self._use_decimal = bool(use_decimal)
        self._decode_json = bool(decode_json)
        # . internal
        self._init_internal()
        # fmt: on
