# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False
from __future__ import annotations

# Cython imports
import cython
from cython.cimports.cpython.time import time as unix_time  # type: ignore
from cython.cimports.libc.limits import UINT_MAX, ULLONG_MAX  # type: ignore
from cython.cimports.cpython.list import PyList_GET_SIZE as list_len  # type: ignore
from cython.cimports.cpython.list import PyList_AsTuple as list_to_tuple  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GET_SIZE as tuple_len  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GetSlice as tuple_slice  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GetItem as tuple_getitem  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Split as str_split  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_Size as bytes_len  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_AsString as bytes_to_chars  # type: ignore
from cython.cimports.sqlcycli._ssl import SSL  # type: ignore
from cython.cimports.sqlcycli.charset import Charset  # type: ignore
from cython.cimports.sqlcycli._optionfile import OptionFile  # type: ignore
from cython.cimports.sqlcycli.constants import _CLIENT, _COMMAND, _SERVER_STATUS  # type: ignore
from cython.cimports.sqlcycli.protocol import MysqlPacket, FieldDescriptorPacket  # type: ignore
from cython.cimports.sqlcycli.transcode import escape, decode  # type: ignore
from cython.cimports.sqlcycli import _auth, typeref, utils  # type: ignore

# Python imports
from io import BufferedReader
from typing import Literal, Any
import socket, errno, warnings, re
from os import PathLike, getpid as _getpid
from pandas import DataFrame
from sqlcycli import _auth
from sqlcycli._ssl import SSL
from sqlcycli.charset import Charset
from sqlcycli._optionfile import OptionFile
from sqlcycli.transcode import escape, decode
from sqlcycli.protocol import MysqlPacket, FieldDescriptorPacket
from sqlcycli.constants import _CLIENT, _COMMAND, _SERVER_STATUS, CR, ER
from sqlcycli import typeref, utils, errors

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


# Result --------------------------------------------------------------------------------------
@cython.cclass
class MysqlResult:
    """Represents the result of a query execution."""

    # Connection
    _conn: BaseConnection
    # Packet data
    _affected_rows: cython.ulonglong  # Value of 18446744073709551615 means None
    _insert_id: cython.ulonglong  # Value of 18446744073709551615 means None
    _server_status: cython.int  # Value of -1 means None
    _warning_count: cython.uint
    _has_next: cython.bint
    _message: bytes
    # Field data
    _field_count: cython.ulonglong
    _fields: tuple[FieldDescriptorPacket]
    _rows: tuple[tuple]
    # Unbuffered
    _unbuffered_active: cython.bint

    def __init__(self, conn: BaseConnection) -> None:
        """The result of a query execution.

        :param conn `<'BaseConnection'>`: The connection that executed the query.
        """
        # Connection
        self._conn = conn
        # Packet data
        self._affected_rows = 0
        self._insert_id = 0
        self._server_status = -1
        self._warning_count = 0
        self._has_next = False
        self._message = None
        # Field data
        self._field_count = 0
        self._fields = None
        self._rows = None
        # Unbuffered
        self._unbuffered_active = False

    # Property --------------------------------------------------------------------------------
    @property
    def affected_rows(self) -> int:
        """The affected rows of the query result `<'int'>`.

        For UNBUFFERED cursor, the value is either 0 or 18446744073709551615.
        """
        return self._affected_rows

    @property
    def insert_id(self) -> int:
        """The LAST INSERT ID of the query result `<'int'>`."""
        return self._insert_id

    @property
    def server_status(self) -> int:
        """The server status of the query result `<'int/None'>`."""
        return self._server_status

    @property
    def warning_count(self) -> int:
        """Total warning counts of the query result `<'int'>`."""
        return self._warning_count

    @property
    def message(self) -> bytes | None:
        """The message of the query result `<'str/None'>`."""
        return self._message

    @property
    def field_count(self) -> int:
        """Total number of fields (columns) of the query result `<'int'>`."""
        return self._field_count

    @property
    def fields(self) -> tuple[FieldDescriptorPacket] | None:
        """The field (column) descriptors of the query
        result `<'tuple[FieldDescriptorPacket]/None'>`.

        Each item in the tuple is a FieldDescriptorPacket object,
        which contains column's metadata of the result.
        """
        return self._fields

    # Read ------------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def read(self) -> cython.bint:
        """Read the packet data."""
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
        """Initiate unbuffered query mode.

        ### Used in unbuffered mode."""
        self._unbuffered_active = True
        try:
            pkt = self._conn._read_packet()
            if pkt.read_ok_packet():
                self._read_ok_packet(pkt)
                self._unbuffered_active = False
                self._conn = None
            elif pkt.read_load_local_packet():
                self._read_load_local_packet(pkt)
                self._unbuffered_active = False
                self._conn = None
            else:
                self._read_result_packet_fields(pkt)
                # Apparently, MySQLdb picks this number because it's the maximum
                # value of a 64bit unsigned integer. Since we're emulating MySQLdb,
                # we set it to this instead of None, which would be preferred.
                self._affected_rows = ULLONG_MAX  # 18446744073709551615
        except:  # noqa
            self._conn = None
            raise
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_ok_packet(self, pkt: MysqlPacket) -> cython.bint:
        """(cfunc) Read OKPacket data. Call when
        the packet is known to be OKPacket."""
        self._affected_rows = pkt._affected_rows
        self._insert_id = pkt._insert_id
        self._server_status = pkt._server_status
        self._warning_count = pkt._warning_count
        self._has_next = pkt._has_next
        self._message = pkt._message
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_load_local_packet(self, pkt: MysqlPacket) -> cython.bint:
        """(cfunc) Read LoadLocalPacket data. Call when
        the packet is known to be LoadLocalPacket."""
        if not self._conn._local_infile:
            raise RuntimeError(
                "**WARNING**: Received LOAD_LOCAL packet but option 'local_infile=False'."
            )
        try:
            self._conn._verify_connected()
            try:
                with open(pkt._filename, "rb") as file:
                    # 16KB is efficient enough
                    size: int = min(self._conn._max_allowed_packet, 16 * 1024)
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
        """(cfunc) Read EOFPacket data. Call when
        the packet is known to be EOFPacket."""
        # TODO: Support CLIENT.DEPRECATE_EOF
        # 1) Add DEPRECATE_EOF to CAPABILITIES
        # 2) Mask CAPABILITIES with server_capabilities
        # 3) if server_capabilities & CLIENT.DEPRECATE_EOF:
        #    use OKPacketWrapper instead of EOFPacketWrapper
        self._warning_count = pkt._warning_count
        self._has_next = pkt._has_next
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_result_packet(self, pkt: MysqlPacket) -> cython.bint:
        """(cfunc) Read ResultPacket data. Call in buffered
        mode and the packet is known to be ResultPacket."""
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
        self._affected_rows = affected_rows
        self._rows = list_to_tuple(rows)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_result_packet_fields(self, pkt: MysqlPacket) -> cython.bint:
        """(cfunc) Read all the FieldDescriptor packets of the query."""
        self._field_count = pkt.read_length_encoded_integer()
        self._fields = list_to_tuple(
            [
                self._conn._read_field_descriptor_packet()
                for _ in range(self._field_count)
            ]
        )
        eof_packet = self._conn._read_packet()
        if not eof_packet.is_eof_packet():
            raise AssertionError("Protocol error, expecting EOF")
        return True

    @cython.cfunc
    @cython.inline(True)
    def _read_result_packet_row(self, pkt: MysqlPacket) -> tuple:
        """(cfunc) Read and decode one row of data from the query `<'tuple'>`."""
        # Settings
        conn: BaseConnection = self._conn
        encoding_c: cython.pchar = conn._encoding_c
        use_decimal: cython.bint = conn._use_decimal
        decode_bit: cython.bint = conn._decode_bit
        decode_json: cython.bint = conn._decode_json
        # Read data
        row: list = []
        field: FieldDescriptorPacket
        for field in self._fields:
            try:
                value: bytes = pkt.read_length_encoded_string()
            except IndexError:  # MysqlPacketCursorError
                # No more columns in this row
                break
            if value is not None:
                data = decode(
                    value,
                    field._type_code,
                    encoding_c,
                    field._is_binary,
                    use_decimal,
                    decode_bit,
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
        """(cfunc) Read and decode one row of data
        from unbuffered query `<'tuple/None'>`

        ### Used in unbuffered mode."""
        # Check if in an active query
        if not self._unbuffered_active:
            return None
        # EOF
        pkt = self._conn._read_packet()
        if pkt.read_eof_packet():
            self._read_eof_packet(pkt)
            self._unbuffered_active = False
            self._rows = None
            self._conn = None
            return None
        # Read row
        row = self._read_result_packet_row(pkt)
        self._affected_rows = 1
        self._rows = (row,)  # rows should tuple of row for compatibility.
        return row

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _drain_result_packet_unbuffered(self) -> cython.bint:
        """(cfunc) Drain all the remaining unbuffered data.

        ### Used in unbuffered mode."""
        # After much reading on the protocol, it appears that there is,
        # in fact, no way to stop from sending all the data after
        # executing a query, so we just spin, and wait for an EOF packet.
        while self._unbuffered_active:
            try:
                pkt = self._conn._read_packet()
            # If the query timed out we can simply ignore this error
            except errors.OperationalTimeoutError:
                self._unbuffered_active = False
                self._conn = None
                return True
            # Release connection before raising the error
            except:  # noqa:
                self._unbuffered_active = False
                self._conn = None
                raise
            # Exist when receieved EOFPacket
            if pkt.read_eof_packet():
                self._read_eof_packet(pkt)
                self._unbuffered_active = False
                self._conn = None  # release reference to kill cyclic reference.
        return True

    def __del__(self):
        if self._unbuffered_active:
            self._drain_result_packet_unbuffered()


# Cursor --------------------------------------------------------------------------------------
# . buffered
@cython.cclass
class Cursor:
    """Represents the cursor (BUFFERED)
    to interact with the database.

    Fetches data in <'tuple'> or <'tuple[tuple]'>.
    """

    _unbuffered: cython.bint  # Determines whether is SSCursor
    _conn: BaseConnection
    _encoding_c: cython.pchar
    _executed_sql: bytes
    _arraysize: cython.ulonglong
    _result: MysqlResult
    _field_count: cython.ulonglong
    _fields: tuple[FieldDescriptorPacket]
    _rows: tuple[tuple]
    _columns: tuple[str]
    _affected_rows: cython.ulonglong
    _row_idx: cython.ulonglong
    _row_size: cython.ulonglong
    _insert_id: cython.ulonglong
    _warning_count: cython.uint

    def __init__(self, conn: BaseConnection) -> None:
        """The cursor (BUFFERED) to interact with the database.

        :param conn `<'BaseConnection'>`: The connection of the cursor.
        """
        self._setup(conn, False)

    # Setup -----------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _setup(
        self,
        conn: BaseConnection,
        unbuffered: cython.bint,
    ) -> cython.bint:
        """(cfunc) Setup the cursor.

        :param conn `<'BaseConnection'>`: The connection of the cursor.
        :param unbuffered `<'bool'>`: Whether to setup as unbuffered cursor (SSCurosr).
        """
        self._unbuffered = unbuffered  # Determines whether is SSCursor
        self._conn = conn
        self._encoding_c = conn._encoding_c
        self._executed_sql = None
        self._arraysize = 1
        self._columns = None
        self._clear_result()
        return True

    # Property --------------------------------------------------------------------------------
    @property
    def executed_sql(self) -> str | None:
        """The last 'sql' executed by the cursor `<'str/None'>`.

        Returns `None` for operations that do not return rows
        or if the cursor has not had an operation invoked via
        the 'cur.execute*()' method yet.
        """
        if self._executed_sql is None:
            return None
        return utils.decode_bytes(self._executed_sql, self._encoding_c)

    @property
    def field_count(self) -> int:
        """Total number of fields (columns) of the query result `<'int'>`."""
        return self._field_count

    @property
    def fields(self) -> tuple[FieldDescriptorPacket] | None:
        """The field (column) descriptors of the query
        result `<'tuple[FieldDescriptorPacket]'>`.

        Each item in the tuple is a FieldDescriptorPacket object,
        which contains column's metadata of the result.

        Returns `None` for operations that do not return rows
        or if the cursor has not had an operation invoked via
        the 'cur.execute*()' method yet.
        """
        return self._fields

    @property
    def insert_id(self) -> int:
        """The LAST INSERT ID of the query result `<'int'>`."""
        return self._insert_id

    @property
    def affected_rows(self) -> int:
        """The number of rows that the last 'cur.execute*()'
        produced (for DQL statements like SELECT) or affected
        (for DML statements like UPDATE or INSERT) `<'int'>`.

        For UNBUFFERED cursor, the value is either 0 or 18446744073709551615.
        """
        return self._affected_rows

    @property
    def warning_count(self) -> int:
        """Total warnings from the query `<'int'>`."""
        return self._warning_count

    @property
    def lastrowid(self) -> int:
        """The LAST INSERT ID of the query result `<'int'>`.

        ### Compliance with PEP-0249, alias for 'insert_id'.
        """
        return self._insert_id

    @property
    def rowcount(self) -> int:
        """The number of rows that the last 'cur.execute*()'
        produced (for DQL statements like SELECT) or affected
        (for DML statements like UPDATE or INSERT) `<'int'>`.

        For UNBUFFERED cursor, the value is either 0 or 18446744073709551615.

        ### Compliance with PEP-0249, alias for 'affected_rows'.
        """
        return self._affected_rows

    @property
    def rownumber(self) -> int:
        """The 0-based row number (index) of the
        cursor for the query result `<'int'>`.

        ### Compliance with PEP-0249.
        """
        return self._row_idx

    @property
    def description(self) -> tuple[tuple] | None:
        """Returns a tuple of of 7-item tuples. Each of these
        tuple contains information describing one result column:
        - name
        - type_code
        - display_size
        - internal_size
        - precision
        - scale
        - null_ok

        Returns `None` for operations that do not return rows
        or if the cursor has not had an operation invoked via
        the 'cur.execute*()' method yet.

        ### Compliance with PEP-0249.
        """
        if self._fields is None:
            return None
        field: FieldDescriptorPacket
        return list_to_tuple([field.description() for field in self._fields])

    @property
    def arraysize(self) -> int:
        """The default number of rows to fetch at a time
        with 'cur.fetchmany()'. Defaults to `1` `<'int'>`.

        For this value to take effect, the argument
        'size' for 'cur.fetchmany()' must be `0`.

        ### Compliance with PEP-0249.
        """
        return self._arraysize

    @arraysize.setter
    def arraysize(self, value: cython.longlong) -> None:
        self._arraysize = 1 if value < 1 else value

    # Write -----------------------------------------------------------------------------------
    @cython.ccall
    def execute(
        self,
        sql: str,
        args: Any = None,
        many: cython.bint = False,
        itemize: cython.bint = True,
    ) -> cython.ulonglong:
        """Prepare and execute a query, returns the affected/selected rows `<'int'>`.

        :param sql `<'str'>`: The query SQL to execute.
        :param args `<'Any'>`: Arguments to bound to the SQL. Defaults to `None`. Supports:
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
            * Note: For single 'NULL' value, use (None,) or [None].

        :param many `<'bool'>`: Whether to execute multi-row 'args'. Defaults to `False`.
            - When 'many=True', the argument 'itemize' is ignored. This function
              behaves the SAME as the 'executemany()' method. For more information
              and related examples, see 'cur.executemany()'.

        :param itemize `<'bool'>`: Whether to escape each items of the 'args' individual. Defaults to `True`.
            - When 'itemize=True', the 'args' type determines how to escape.
                * 1. Sequence or Mapping (e.g. `list`, `tuple`, `dict`, etc)
                  escapes to `<'tuple[str]'>`. The 'sql' should have '%s'
                  placeholders equal to the tuple length.
                * 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to
                  `<'tuple[str]'>`. The 'sql' should have '%s' placeholders
                  equal to the tuple length.
                * 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes
                  to `<'list[tuple[str]]'>`. This function behaves the SAME
                  as the 'executemany()' method, and the 'sql' should have
                  '%s' placeholders equal to the tuple (element of list) length.
                * 4. Single object (such as `int`, `float`, `str`, etc)
                  escapes to one literal string `<'str'>`. The 'sql' should
                  have one '%s' placeholder.
            - When 'itemize=False', regardless of the 'args' type (besides 'None'),
              all escapes to one single literal string `<'str'>`. The 'sql' should
              have one '%s' placeholder.

        ## Example (itemize=True):
        >>> cur.execute(
                "INSERT INTO table (name, age, height) VALUES (%s, %s, %s)",
                ["John", 25, 170]  # defalut: itemize=True & many=False,
            )
            # escaped as:
            ("'John'", '25', '170')
            # executed as:
            "INSERT INTO table (name, age, height) VALUES ('John', 25, 170);"

        ## Example (nested & itemize=True):
        >>> cur.execute(
                "SELECT * FROM table WHERE name=%s AND age IN %s",
                ["John", (25, 26)]  # defalut: itemize=True & many=False,
            )
            # escaped as:
            ("'John'", "('25','26')")
            # executed as:
            "SELECT * FROM table WHERE name='John' AND age IN (25,26);"

        ## Example (itemize=False):
        >>> cur.execute(
                "INSERT INTO table (name, age, height) VALUES %s",
                ["John", 25, 170], itemize=False  # defalut: many=False
            )
            # escaped as:
            "('John',25,170)"
            # executed as:
            "INSERT INTO table (name, age, height) VALUES ('John',25,170);"

        ## Example (nested & itemize=False):
        >>> cur.execute(
                "INSERT INTO table (name, age, height) VALUES %s",
                [["John", 25, 170], ["Doe", 26, 180]], itemize=False  # defalut: many=False
            )
            # escaped as:
            "('John',25,170),('Doe',26,180)"
            # executed as:
            "INSERT INTO table (name, age, height) VALUES ('John',25,170),('Doe',26,180);"

        ## Example (single object):
        >>> cur.execute("INSERT INTO table (name) VALUES (%s)", "John")
            # escaped as:
            "'John'"
            # executed as:
            "INSERT INTO table (name) VALUES ('John');"
        """
        # Query without args
        if args is None:
            return self._query_str(sql)
        args = escape(args, many, itemize)

        # Single row query
        if not many and not itemize:
            # When 'many=False' & 'itemize=False', the escaped
            # 'args' can only be a single literal <'str'>.
            return self._query_str(self._format(sql, args))
        if type(args) is not list:
            # If the escaped 'args' is not a list, it must be
            # either a <'str'> or <'tuple[str]'>, which should
            # also bind to the 'sql' directly.
            return self._query_str(self._format(sql, args))

        # Multi-rows query
        # The escaped 'args' now on can only be a <'list'>.
        # If the list is empty, execute the query directly.
        if list_len(args) == 0:
            return self._query_str(sql)

        # Bulk INSERT/REPLACE `NOT` matched
        rows: cython.ulonglong = 0
        m = utils.INSERT_VALUES_RE.match(sql)
        if m is None:
            # . execute row by row
            for arg in args:
                rows += self._query_str(self._format(sql, arg))
            self._affected_rows = rows
            return rows

        # Bulk INSERT/REPLACE match
        self._verify_connected()
        conn: BaseConnection = self._conn
        # . split query
        plhs: str
        regex_gps: tuple = m.groups()
        pfix, plhs, sfix = regex_gps
        # . query prefix: INSERT INTO ... VALUES
        prefix: bytes = conn.encode_sql(self._format(pfix, ()))
        # . query placeholders: (%s, %s, ...)
        plhs = plhs.rstrip()
        # . query suffix: AS ... ON DUPLICATE ...
        suffix: bytes = b"" if sfix is None else conn.encode_sql(sfix)
        # . prepare statement
        args_iter = iter(args)
        vals: bytes = conn.encode_sql(self._format(plhs, next(args_iter)))
        stmt: list[bytes] = [prefix, vals]
        fix_len: cython.uint = bytes_len(prefix) + bytes_len(suffix)
        val_len: cython.uint = bytes_len(vals)
        sql_len: cython.uint = fix_len + val_len
        for arg in args_iter:
            vals = conn.encode_sql(self._format(plhs, arg))
            val_len = bytes_len(vals)
            sql_len += 1 + val_len
            if sql_len <= utils.MAX_STATEMENT_LENGTH:
                stmt.append(b",")
                stmt.append(vals)
            else:
                # - execute within limit
                stmt.append(suffix)
                rows += self._query_bytes(b"".join(stmt))
                # - reset stmt & sql_len
                stmt = [prefix, vals]
                sql_len = fix_len + val_len
        # . execute
        stmt.append(suffix)
        rows += self._query_bytes(b"".join(stmt))
        self._affected_rows = rows
        return rows

    @cython.ccall
    def executemany(self, sql: str, args: Any = None) -> cython.ulonglong:
        """Prepare and execute multi-row 'args' against a query,
        returns the affected/selected rows `<'int'>`.

        :param sql `<'str'>`: The query SQL to execute.
        :param args `<'Any'>`: Sequences or mappings to bound to the SQL. Defaults to `None`. Supports:
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

        ## Escape of 'args'
        * 1. Sequence and Mapping (e.g. `list`, `tuple`, `dict`, etc) escapes
          to `<'list[str/tuple[str]]'>`. Each element represents one row of the
          'args', and the 'sql' should have '%s' placeholders equal to the item
          count in each row.
        * 2. `pd.Series` and 1-dimensional `np.ndarray` escapes to `<'list[str]'>`.
          The the 'sql' should have one '%s' placeholder.
        * 3. `pd.DataFrame` and 2-dimensional `np.ndarray` escapes to `<'list[tuple[str]]'>`.
          The 'sql' should have '%s' placeholders equal to the tuple (element of list) length.
        * 4. Single object (such as `int`, `float`, `str`, etc) escapes to one
          literal string `<'str'>`. This function behaves the SAME as the
          'execute()' method, and the 'sql' should have one '%s' placeholder.

        ## Example (sequence & mapping [flat]):
        >>> cur.executemany(
                "INSERT INTO table (name) VALUES (%s)",
                ["John", "Doe"],
            )
            # escaped as:
            ["'John'", "'Doe'"]
            # executed as:
            "INSERT INTO table (name) VALUES ('John'),('Doe');"

        ## Example (sequence & mapping [nested]):
        >>> cur.executemany(
                "INSERT INTO table (name, age) VALUES (%s, %s)",
                [["John", 25], ["Doe", 26]],
            )
            # escaped as:
            [("'John'", "25"), ("'Doe'", "26")]
            # executed as:
            "INSERT INTO table (name, age) VALUES ('John', 25),('Doe', 26);"

        ## Example (pd.Series):
        >>> cur.executemany(
                "INSERT INTO table (age) VALUES (%s)",
                pd.Series([25, 26]),
            )
            # escaped as:
            ["25", "26"]
            # executed as:
            "INSERT INTO table (age) VALUES (25),(26);"

        ## Example (1D ndarray):
        >>> cur.executemany(
                "INSERT INTO table (age) VALUES (%s)",
                np.array([25, 26]),
            )
            # escaped as:
            ["25", "26"]
            # executed as:
            "INSERT INTO table (age) VALUES (25),(26);"

        ## Example (pd.DataFrame):
        >>> cur.executemany(
                "INSERT INTO table (name, age) VALUES (%s, %s)",
                pd.DataFrame({"name": ["John", "Doe"], "age": [25, 26]}),
            )
            # escaped as:
            [("'John'", "25"), ("'Doe'", "26")]
            # executed as:
            "INSERT INTO table (name, age) VALUES ('John', 25),('Doe', 26);"

        ## Example (2D ndarray):
        >>> cur.executemany(
                "INSERT INTO table (age, height) VALUES (%s, %s)",
                np.array([[25, 170], [26, 180]]),
            )
            # escaped as:
            [("25", "170"), ("26", "180")]
            # executed as:
            "INSERT INTO table (age, height) VALUES (25, 170),(26, 180);"
        """
        return self.execute(sql, args, True, True)

    @cython.ccall
    def callproc(self, procname: str, args: tuple | list) -> object:
        """Execute stored procedure 'procname' with 'args'
        and returns the original arguments.

        :param procname `<'str'>`: Name of procedure to execute on server.
        :param args `<'list/tuple'>`: Sequence of parameters to use with procedure.
        :returns `<'list/tuple'>`: The original arguments.

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
        query using 'cur.execute*()' to get any OUT or INOUT values.

        Compatibility warning: The act of calling a stored procedure
        itself creates an empty result set. This appears after any
        result sets generated by the procedure. This is non-standard
        behavior with respect to the DB-API. Be sure to use 'cur.nextset()'
        to advance through all result sets; otherwise you may get
        disconnected.
        """
        # Validate & Escape 'args'
        if args is None:
            _args: tuple = ()
        else:
            items = escape(args, False, True)
            if type(items) is not tuple:
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
            self.nextset()

        # Call procedures
        # fmt: off
        sql: str = "CALL %s(%s)" % (
            procname,
            ",".join([self._format("@_%s_%d", (procname, i)) for i in range(count)]),
        )
        # fmt: on
        self._query_str(sql)
        return args

    @cython.ccall
    def mogrify(
        self,
        sql: str,
        args: Any = None,
        many: cython.bint = False,
        itemize: cython.bint = True,
    ) -> str:
        """Bound the 'args' to the 'sql' and returns the `exact*` string that
        will be sent to the database by calling the execute*() method `<'str'>`.

        :param sql `<'str'>`: The query SQL to mogrify.
        :param args `<'Any'>`: Arguments to bound to the SQL. Defaults to `None`.
        :param many `<'bool'>`: Whether to execute multi-row 'args'. Defaults to `False`.
        :param itemize `<'bool'>`: Whether to escape each items of the 'args' individual. Defaults to `True`.

        ## Explanation
        - When 'many=False' & 'itemize=True' (default), this method behaves
          similar to 'PyMySQL.Cursor.mogrify()'.
        - For multi-row 'args', ONLY the 'first' row [item] will be bound
          to the sql and returned (cleaner illustration `not [excat*]`).
        - For more information about the arguments, please refer to
          'cur.execute()' method.
        """
        # Query without args
        if args is None:
            return sql
        args = escape(args, many, itemize)

        # Single row query
        if not many and not itemize:
            # When 'many=False' & 'itemize=False', the escaped
            # 'args' can only be a single literal <'str'>.
            return self._format(sql, args)
        if type(args) is not list:
            # If the escaped 'args' is not a list, it must be
            # either a <'str'> or <'tuple[str]'>, which should
            # also bind to the 'sql' directly.
            return self._format(sql, args)

        # Multi-row query
        # The escaped 'args' now on can only be a <'list[str/tuple]'>.
        # Only the 'first' row [item] of the args will be bound to the sql.
        if len(args) == 0:
            return sql
        else:
            return self._format(sql, args[0])

    @cython.cfunc
    @cython.inline(True)
    def _query_str(self, sql: str) -> cython.ulonglong:
        """(cfunc) Execute a <'str'> query `<'int'>`."""
        return self._query_bytes(utils.encode_str(sql, self._encoding_c))

    @cython.cfunc
    @cython.inline(True)
    def _query_bytes(self, sql: bytes) -> cython.ulonglong:
        """(cfunc) Execute an encoded <'bytes'> query `<'int'>`."""
        while self.nextset():
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
    def _format(self, sql: str, args: str | tuple) -> str:
        """(cfunc) Format the query with the arguments `<'str'>`.

        :param sql `<'str'>`: The query to format.
        :param args `<'str/tuple'>`: Arguments to bound to the SQL.
        :raises `<'InvalidSQLArgsErorr'>`: If any error occurs.
        """
        try:
            return sql % args
        except Exception as err:
            raise errors.InvalidSQLArgsErorr(
                "\nFailed to format SQL:\n%s\n"
                "With %s arguments:\n%r\n"
                "Error: %s" % (sql, type(args), args, err)
            ) from err

    # Read ------------------------------------------------------------------------------------
    # . fetchone
    def fetchone(self) -> tuple | None:
        """Fetch the next row of the query result set `<'tuple/None'>`.

        :returns `<'tuple/None'>`: If no more rows, returns `None`.
        """
        return self._fetchone_tuple()

    @cython.cfunc
    @cython.inline(True)
    def _fetchone_tuple(self) -> tuple:
        """(cfunc) Fetch the next row of the query result set `<'tuple/None'>`.

        :returns `<'tuple/None'>`: If no more rows, returns `None`.

        ### Used by Cursor & SSCursor.
        """
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
            row = self._next_row_unbuffered()
            if row is None:
                self._warning_count = self._result._warning_count
                return None  # exit: no rows
            return row  # exit: one row

    @cython.cfunc
    @cython.inline(True)
    def _fetchone_dict(self) -> dict:
        """(cfunc) Fetch the next row of the query result set `<'dict/None'>`.

        :returns `<'dict/None'>`: If no more rows, returns `None`.

        ### Used by DictCursor & SSDictCursor.
        """
        # Fetch & validate
        row = self._fetchone_tuple()
        if row is None:
            return None  # exit: no more rows
        cols = self.columns()
        if cols is None:
            return None  # eixt: no columns
        # Generate
        return self._convert_row_to_dict(row, cols, self._field_count)

    @cython.cfunc
    @cython.inline(True)
    def _fetchone_df(self) -> object:
        """(cfunc) Fetch the next row of the query result set `<'DataFrame/None'>`.

        :returns `<'DataFrame/None'>`: If no more rows, returns `None`.

        ### Used by DfCursor & SSDfCursor.
        """
        # Fetch & validate
        row = self._fetchone_tuple()
        if row is None:
            return None  # exit: no more rows
        cols = self.columns()
        if cols is None:
            return None  # eixt: no columns
        # Generate
        return typeref.DATAFRAME([row], columns=cols)

    # . fetchmany
    def fetchmany(self, size: int = 1) -> tuple[tuple]:
        """Fetch the next set of rows of the query result `<'tuple[tuple]'>`.

        :param size `<'int'>`: Number of rows to be fetched. Defaults to `1`.
            - When 'size=0', defaults to 'cur.arraysize'.

        :returns `<'tuple[tuple]'>`: If no more rows, returns an empty `tuple`.
        """
        return self._fetchmany_tuple(size)

    @cython.cfunc
    @cython.inline(True)
    def _fetchmany_tuple(self, size: cython.ulonglong) -> tuple[tuple]:
        """(cfunc) Fetch the next set of rows of the query result `<'tuple[tuple]'>`.

        :param size `<'int'>`: Number of rows to be fetched. Defaults to `1`.
            - When 'size=0', defaults to 'cur.arraysize'.

        :returns `<'tuple[tuple]'>`: If no more rows, returns an empty `tuple`.

        ### Used by Cursor & SSCursor.
        """
        self._verify_executed()
        if size == 0:
            size = self._arraysize
        # Buffered
        if not self._unbuffered:
            # No more rows
            if not self._has_more_rows():
                return ()  # exit: no rows
            # Fetch multi-rows
            end: cython.ulonglong = min(self._row_idx + size, self._row_size)
            idx: cython.ulonglong = self._row_idx
            self._row_idx = end
            return tuple_slice(self._rows, idx, end)  # exit: multi-rows

        # Unbuffered
        else:
            # Fetch multi-rows
            rows: list = []
            for _ in range(size):
                row = self._next_row_unbuffered()
                if row is None:
                    self._warning_count = self._result._warning_count
                    break
                else:
                    rows.append(row)
            return list_to_tuple(rows)  # exit: multi-rows

    @cython.cfunc
    @cython.inline(True)
    def _fetchmany_dict(self, size: cython.ulonglong) -> tuple[dict]:
        """(cfunc) Fetch the next set of rows of the query result `<'tuple[dict]'>`.

        :param size `<'int'>`: Number of rows to be fetched. Defaults to `1`.
            - When 'size=0', defaults to 'cur.arraysize'.

        :returns `<'tuple[dict]'>`: If no more rows, returns an empty `tuple`.

        ### Used by DictCursor & SSDictCursor.
        """
        # Fetch & validate
        rows = self._fetchmany_tuple(size)
        if not rows:
            return ()  # exit: no more rows
        cols = self.columns()
        if cols is None:
            return ()  # eixt: no columns
        # Generate
        field_count: cython.ulonglong = self._field_count
        return list_to_tuple(
            [self._convert_row_to_dict(row, cols, field_count) for row in rows]
        )

    @cython.cfunc
    @cython.inline(True)
    def _fetchmany_df(self, size: cython.ulonglong) -> object:
        """(cfunc) Fetch the next set of rows of the query result `<'DataFrame'>`.

        :param size `<'int'>`: Number of rows to be fetched. Defaults to `1`.
            - When 'size=0', defaults to 'cur.arraysize'.

        :returns `<'DataFrame'>`: If no more rows, returns an empty `DataFrame`.

        ### Used by DfCursor & SSDfCursor.
        """
        # Fetch & validate
        rows = self._fetchmany_tuple(size)
        cols = self.columns()
        if not rows:  # exit: no more rows
            if cols is None:
                return typeref.DATAFRAME()
            return typeref.DATAFRAME(columns=cols)
        if cols is None:
            return typeref.DATAFRAME()  # eixt: no columns
        # Generate
        return typeref.DATAFRAME(rows, columns=cols)

    # . fetchall
    def fetchall(self) -> tuple[tuple]:
        """Fetch all (remaining) rows of the query result `<'tuple[tuple]'>`.

        :returns `<'tuple[tuple]'>`: If no more rows, returns an empty `tuple`.
        """
        return self._fetchall_tuple()

    @cython.cfunc
    @cython.inline(True)
    def _fetchall_tuple(self) -> tuple[tuple]:
        """(cfunc) Fetch all (remaining) rows of the query result `<'tuple[tuple]'>`.

        :returns `<'tuple[tuple]'>`: If no more rows, returns an empty `tuple`.

        ### Used by Cursor & SSCursor.
        """
        self._verify_executed()
        # Buffered
        if not self._unbuffered:
            # No more rows
            if not self._has_more_rows():
                return ()  # exit: no rows
            # Fetch all rows
            row_size: cython.ulonglong = self._row_size
            if self._row_idx == 0:
                self._row_idx = row_size
                return self._rows  # exit: all rows
            end: cython.ulonglong = row_size
            idx: cython.ulonglong = self._row_idx
            self._row_idx = end
            return tuple_slice(self._rows, idx, end)  # exit: remain rows

        # Unbuffered
        else:
            # Fetch all rows
            rows: list = []
            while True:
                row = self._next_row_unbuffered()
                if row is None:
                    self._warning_count = self._result._warning_count
                    break
                else:
                    rows.append(row)
            return list_to_tuple(rows)  # exit: remain rows

    @cython.cfunc
    @cython.inline(True)
    def _fetchall_dict(self) -> tuple[dict]:
        """(cfunc) Fetch all (remaining) rows of the query result `<'tuple[dict]'>`.

        :returns `<'tuple[dict]'>`: If no more rows, returns an empty `tuple`.

        ### Used by DictCursor & SSDictCursor.
        """
        # Fetch & validate
        rows = self._fetchall_tuple()
        if not rows:
            return ()  # exit: no more rows
        cols = self.columns()
        if cols is None:
            return ()  # eixt: no columns
        # Generate
        field_count: cython.ulonglong = self._field_count
        return list_to_tuple(
            [self._convert_row_to_dict(row, cols, field_count) for row in rows]
        )

    @cython.cfunc
    @cython.inline(True)
    def _fetchall_df(self) -> object:
        """(cfunc) Fetch all (remaining) rows of the query result `<'DataFrame'>`.

        :returns `<'DataFrame'>`: If no more rows, returns an empty `DataFrame`.

        ### Used by DfCursor & SSDfCursor.
        """
        # Fetch & validate
        rows = self._fetchall_tuple()
        cols = self.columns()
        if not rows:  # exit: no more rows
            if cols is None:
                return typeref.DATAFRAME()
            return typeref.DATAFRAME(columns=cols)
        if cols is None:
            return typeref.DATAFRAME()  # eixt: no columns
        # Generate
        return typeref.DATAFRAME(rows, columns=cols)

    # . fetch: converter
    @cython.cfunc
    @cython.inline(True)
    def _convert_row_to_dict(
        self,
        row: tuple,
        cols: tuple,
        field_count: cython.ulonglong,
    ) -> dict:
        """(cfunc) Convert one row (tuple) to a dictionary `<'dict'>`."""
        return {cols[i]: row[i] for i in range(field_count)}

    # . rest
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def scroll(self, value: cython.longlong, mode: str = "relative") -> cython.bint:
        """Scroll the cursor to a new position of the query result.

        :param value `<'int'>`: The value for the cursor position.
        :param mode `<'str'>`: The mode of the cursor movement. Defaults to `'relative'`.
            - 'relative': The 'value' is taken as an offset to the current position.
            - 'absolute': The 'value' is taken as the absolute target position.

        :raises `<'InvalidCursorIndexError'>`: If the new position if out of range.
        """
        # Validate
        self._verify_executed()

        # Buffered
        if not self._unbuffered:
            row_size: cython.ulonglong = self._get_row_size()
            if row_size == 0:
                self._row_idx = 0
                return False

            idx: cython.ulonglong
            if mode == "relative":
                if value < 0 and -value > self._row_idx:
                    raise errors.InvalidCursorIndexError(
                        "Cursor index cannot be negative: %d." % (self._row_idx + value)
                    )
                idx = self._row_idx + value
            elif mode == "absolute":
                if value < 0:
                    raise errors.InvalidCursorIndexError(
                        "Cursor index cannot be negative: %d." % value
                    )
                idx = value
            else:
                raise errors.InvalidCursorArgsError("Inavlid scroll mode '%s'." % mode)

            if idx >= row_size:
                raise errors.InvalidCursorIndexError(
                    "Cursor index '%d' out of range: 0~%d." % (idx, row_size)
                )
            self._row_idx = idx
            return True

        # Unbuffered
        if mode == "relative":
            if value < 0:
                raise errors.InvalidCursorIndexError(
                    "Backwards scrolling not supported by <'%s'>."
                    % self.__class__.__name__
                )
        elif mode == "absolute":
            if value < self._row_idx:
                raise errors.InvalidCursorIndexError(
                    "Backwards scrolling not supported by <'%s'>."
                    % self.__class__.__name__
                )
            value -= self._row_idx
        else:
            raise errors.InvalidCursorArgsError("Inavlid scroll mode '%s'." % mode)
        for _ in range(value):
            if self._next_row_unbuffered() is None:
                break
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def nextset(self) -> cython.bint:
        """Go to the next set of result, discarding any remaining
        rows from the current set. Returns boolean to indicate if
        next set exists `<'bool'>`."""
        self._verify_connected()
        conn: BaseConnection = self._conn
        curr_result: MysqlResult = self._result
        if (
            curr_result is None
            or curr_result is not conn._result
            or not curr_result._has_next
        ):
            # . reset columns from previous result
            if self._columns is not None:
                self._columns = None
            return False
        self._clear_result()
        conn.next_result(self._unbuffered)
        self._read_result()
        return True

    @cython.cfunc
    @cython.inline(True)
    def _next_row_unbuffered(self) -> tuple:
        """(cfunc) Get the next unbuffered row `<'tuple/None'>`"""
        row = self._result._read_result_packet_row_unbuffered()
        if row is not None:
            self._row_idx += 1
        return row

    @cython.ccall
    def columns(self) -> tuple[str]:
        """Get the 'column' names for each fields of the
        query result `<'tuple/None'>`.

        Returns `None` for operations that do not return rows
        or if the cursor has not had an operation invoked via
        the 'cur.execute*()' method yet.
        """
        if self._columns is None:
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
            self._columns = list_to_tuple(cols)
        return self._columns

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _read_result(self) -> cython.bint:
        """(cfunc) Read query result."""
        result: MysqlResult = self._conn._result
        self._result = result
        self._field_count = result._field_count
        self._fields = result._fields
        self._rows = result._rows
        self._affected_rows = result._affected_rows
        self._insert_id = result._insert_id
        self._warning_count = result._warning_count
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _clear_result(self) -> cython.bint:
        """(cfunc) Clear previous result."""
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
        """(cfunc) Get the total number of rows of the result `<'int'>`."""
        if self._row_size == 0:
            if self._rows is None:
                return 0
            self._row_size = tuple_len(self._rows)
        return self._row_size

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _has_more_rows(self) -> cython.bint:
        """(cfunc) Whether has more rows in the result `<'bool'>`."""
        row_size: cython.ulonglong = self._get_row_size()
        if self._row_idx >= row_size or row_size == 0:
            return False
        else:
            return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_executed(self) -> cython.bint:
        """(cfunc) Verify if the cursor has already executed a query.

        :raises `<'CursorNotExecutedError'>`: If cursor has not had an
        operation invoked via the 'cur.execute*()' method yet.
        """
        if self._executed_sql is None:
            raise errors.CursorNotExecutedError(
                0, "Please execute a query with the cursor first."
            )
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_connected(self) -> cython.bint:
        """(cfunc) Verify if the cursor is connected (attacted to connection).

        :raises `<'CursorClosedError'>`: If the cursor has already been closed.
        """
        if self.closed():
            raise errors.CursorClosedError(0, "Cursor is closed.")
        return True

    # Compliance ------------------------------------------------------------------------------
    def setinputsizes(self, *args):
        """Does nothing, Compliance with PEP-0249."""

    def setoutputsizes(self, *args):
        """Does nothing, Compliance with PEP-0249."""

    # Close -----------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def close(self) -> cython.bint:
        """Close the cursor, remaining data (if any)
        will be exhausted automatically."""
        if self.closed():
            return True
        try:
            if (
                self._unbuffered
                and self._result is not None
                and self._result is self._conn._result
            ):
                self._result._drain_result_packet_unbuffered()
            while self.nextset():
                pass
        finally:
            self._conn = None
            self._clear_result()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def closed(self) -> cython.bint:
        """Whether the cursor is closed `<'bool'>`."""
        return self._conn is None

    # Special methods -------------------------------------------------------------------------
    def __enter__(self) -> Cursor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self) -> Cursor:
        return self

    def __next__(self) -> tuple:
        row = self._fetchone_tuple()
        if row is None:
            raise StopIteration
        return row

    def __del__(self):
        self.close()


@cython.cclass
class DictCursor(Cursor):
    """Represents the cursor (BUFFERED)
    to interact with the database.

    Fetches data in <'dict'> or <'tuple[dict]'>.
    """

    # . fetchone
    def fetchone(self) -> dict | None:
        """Fetch the next row of the query result set `<'dict/None'>`.

        :returns `<'dict/None'>`: If no more rows, returns `None`.
        """
        return self._fetchone_dict()

    # . fetchmanny
    def fetchmany(self, size: int = 1) -> tuple[dict]:
        """Fetch the next set of rows of the query result `<'tuple[dict]'>`.

        :param size `<'int'>`: Number of rows to be fetched. Defaults to `1`.
            - When 'size=0', defaults to 'cur.arraysize'.

        :returns `<'tuple[dict]'>`: If no more rows, returns an empty `tuple`.
        """
        return self._fetchmany_dict(size)

    # . fetchall
    def fetchall(self) -> tuple[dict]:
        """Fetch all (remaining) rows of the query result `<'tuple[dict]'>`.

        :returns `<'tuple[dict]'>`: If no more rows, returns an empty `tuple`.
        """
        return self._fetchall_dict()

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
    """Represents the cursor (BUFFERED)
    to interact with the database.

    Fetches data in <'DataFrame'>.
    """

    # . fetchone
    def fetchone(self) -> DataFrame | None:
        """Fetch the next row of the query result set `<'DataFrame/None'>`.

        :returns `<'DataFrame/None'>`: If no more rows, returns `None`.
        """
        return self._fetchone_df()

    # . fetchmanny
    def fetchmany(self, size: int = 1) -> DataFrame:
        """Fetch the next set of rows of the query result `<'DataFrame'>`.

        :param size `<'int'>`: Number of rows to be fetched. Defaults to `1`.
            - When 'size=0', defaults to 'cur.arraysize'.

        :returns `<'DataFrame'>`: If no more rows, returns an empty `DataFrame`.
        """
        return self._fetchmany_df(size)

    # . fetchall
    def fetchall(self) -> DataFrame:
        """Fetch all (remaining) rows of the query result `<'DataFrame'>`.

        :returns `<'DataFrame'>`: If no more rows, returns an empty `DataFrame`.
        """
        return self._fetchall_df()

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
    """Represents the cursor (UNBUFFERED)
    to interact with the database.

    Fetches data in <'tuple'> or <'tuple[tuple]'>.
    """

    def __init__(self, conn: BaseConnection) -> None:
        """The cursor (UNBUFFERED) to interact with the database.

        :param conn `<'BaseConnection'>`: The connection of the cursor.
        """
        self._setup(conn, True)


@cython.cclass
class SSDictCursor(DictCursor):
    """Represents the cursor (UNBUFFERED)
    to interact with the database.

    Fetches data in <'dict'> or <'tuple[dict]'>.
    """

    def __init__(self, conn: BaseConnection) -> None:
        """The cursor (UNBUFFERED) to interact with the database.

        :param conn `<'BaseConnection'>`: The connection of the cursor.
        """
        self._setup(conn, True)


@cython.cclass
class SSDfCursor(DfCursor):
    """Represents the cursor (UNBUFFERED)
    to interact with the database.

    Fetches data in <'DataFrame'>.
    """

    def __init__(self, conn: BaseConnection) -> None:
        """The cursor (UNBUFFERED) to interact with the database.

        :param conn `<'BaseConnection'>`: The connection of the cursor.
        """
        self._setup(conn, True)


# Connection ----------------------------------------------------------------------------------
@cython.cclass
class CursorManager:
    """The Context Manager for the Cursor."""

    _conn: BaseConnection
    _cur_type: type[Cursor]
    _cur: Cursor
    _closed: cython.bint

    def __init__(self, conn: BaseConnection, cursor: type[Cursor]) -> None:
        """The Context Manager for the Cursor.

        :param conn `<'BaseConnection'>`: The Connection for the cursor.
        :param cursor `<'type[Cursor]'>`: The Cursor type (class) to use.
        """
        self._conn = conn
        self._cur_type = cursor
        self._cur = None
        self._closed = False

    @cython.cfunc
    @cython.inline(True)
    def _acquire(self) -> Cursor:
        """(cfunc) Acquire the connection cursor `<'Cursor'>`."""
        try:
            return self._cur_type(self._conn)
        except:  # noqa
            self._close()
            raise

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _close(self) -> cython.bint:
        """(cfunc) Close the cursor and cleanup the context manager.

        ### This method does not raise any error.
        """
        if not self._closed:
            if self._cur is not None:
                try:
                    self._cur.close()
                except Exception:
                    pass
                self._cur = None
            self._cur_type = None
            self._conn = None
            self._closed = True
        return True

    def __enter__(self) -> Cursor:
        self._cur = self._acquire()
        return self._cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()

    def __del__(self):
        self._close()


@cython.cclass
class TransactionManager(CursorManager):
    """The Context Manager for the Cursor in `TRANSACTION` mode.

    By acquiring cursor through this context manager, the following happens:
    - 1. Use the connection to `BEGIN` a transaction.
    - 2. Returns the cursor of the connection.
    - 3a. If catches ANY exceptions during the transaction, close the connection.
    - 3b. If the transaction executed successfully, execute `COMMIT`.
    """

    def __init__(self, conn: BaseConnection, cursor: type[Cursor]) -> None:
        """The Context Manager for the Cursor in `TRANSACTION` mode.

        By acquiring cursor through this context manager, the following happens:
        - 1. Use the connection to `BEGIN` a transaction.
        - 2. Returns the cursor of the connection.
        - 3a. If catches ANY exceptions during the transaction, close the connection.
        - 3b. If the transaction executed successfully, execute `COMMIT`.

        :param conn `<'BaseConnection'>`: The Connection to start the transaction.
        :param cursor `<'type[Cursor]'>`: The Cursor type (class) to use.
        """
        self._conn = conn
        self._cur_type = cursor
        self._cur = None
        self._closed = False

    def __enter__(self) -> Cursor:
        self._cur = self._acquire()
        try:
            self._conn.begin()
        except:  # noqa
            self._close()
            raise
        return self._cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Encounter error
        if exc_val is not None:
            self._conn.close()
            self._close()
            raise exc_val

        # Try commit the transaction
        try:
            # exit: commit successfully
            self._conn.commit()
            self._close()
        except:  # noqa
            # fail to commit
            self._conn.close()
            self._close()
            raise


@cython.cclass
class BaseConnection:
    """Represents the socket connection to the server.

    This class serves as the base connection class. It does not perform
    argument validations during initialization. Such validations are
    delegated to the subclass `<'Connection'>`.

    ### Please do `NOT` create an instance of this class directly.
    """

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
    _charset_changed: cython.bint
    # Timeouts
    _connect_timeout: object  # uint
    _read_timeout: object  # uint | None
    _read_timeout_changed: cython.bint
    _write_timeout: object  # uint | None
    _write_timeout_changed: cython.bint
    _wait_timeout: object  # uint | None
    _wait_timeout_changed: cython.bint
    _interactive_timeout: object  # uint | None
    _interactive_timeout_changed: cython.bint
    _lock_wait_timeout: object  # uint | None
    _lock_wait_timeout_changed: cython.bint
    _execution_timeout: object  # uint | None
    _execution_timeout_changed: cython.bint
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
    _auth_plugin: _auth.AuthPlugin
    _server_public_key: bytes
    # Decode
    _use_decimal: cython.bint
    _decode_bit: cython.bint
    _decode_json: cython.bint
    # Internal
    # . server
    _server_protocol_version: cython.int
    _server_info: str
    _server_version: tuple[int]
    _server_version_major: cython.int
    _server_vendor: str
    _server_thread_id: cython.longlong
    _server_salt: bytes
    _server_status: cython.int
    _server_capabilities: cython.longlong
    _server_auth_plugin_name: str
    # . client
    _last_used_time: cython.double
    _secure: cython.bint
    _close_reason: str
    # . query
    _result: MysqlResult
    _next_seq_id: cython.uint
    # . transport
    _reader: BufferedReader
    _writer: socket.socket

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
        cursor: type[Cursor],
        client_flag: cython.uint,
        program_name: str | None,
        ssl_ctx: object | None,
        auth_plugin: _auth.AuthPlugin | None,
        server_public_key: bytes | None,
        use_decimal: cython.bint,
        decode_bit: cython.bint,
        decode_json: cython.bint,
    ):
        """The socket connection to the server.

        This class serves as the base connection class. It does not perform
        argument validations during initialization. Such validations are
        delegated to the subclass `<'Connection'>`.

        ### Please do `NOT` create an instance of this class directly.

        :param host `<'str'>`: The host of the server.
        :param port `<'int'>`: The port of the server.
        :param user `<'bytes/None'>`: The username to login as.
        :param password `<'bytes'>`: The password for login authentication.
        :param database `<'bytes/None'>`: The default database to use by the connection.
        :param charset `<'Charset'>`: The charset for the connection.
        :param connect_timeout `<'int'>`: Set timeout (in seconds) for establishing the connection.
        :param read_timeout `<'int/None>`: Set SESSION 'net_read_timeout' (in seconds). `None` mean to use GLOBAL settings.
        :param write_timeout `<'int/None>`: Set SESSION 'net_write_timeout' (in seconds). `None` mean to use GLOBAL settings.
        :param wait_timeout `<'int/None>`: Set SESSION 'wait_timeout' (in seconds). `None` mean to use GLOBAL settings.
        :param interactive_timeout `<'int/None>`: Set SESSION 'interactive_timeout' (in seconds). `None` mean to use GLOBAL settings.
        :param lock_wait_timeout `<'int/None>`: Set SESSION 'innodb_lock_wait_timeout' (in seconds). `None` mean to use GLOBAL settings.
        :param execution_timeout `<'int/None>`: Set SESSION 'max_execution_time' (in milliseconds). `None` mean to use GLOBAL settings.
        :param bind_address `<'str/None'>`: The interface from which to connect to the host. Accept both hostname or IP address.
        :param unix_socket `<'str/None'>`: The unix socket for establishing connection rather than TCP/IP.
        :param autocommit_mode `<'int'>`: The autocommit mode for the connection. -1: Default, 0: OFF, 1: ON.
        :param local_infile `<'bool'>`: Enable/Disable LOAD DATA LOCAL command.
        :param max_allowed_packet `<'int'>`: The max size of packet sent to server in bytes.
        :param sql_mode `<'str/None'>`: The default SQL_MODE for the connection.
        :param init_command `<'str/None'>`: The initial SQL statement to run when connection is established.
        :param cursor `<'type[Cursor]'>`: The default cursor type (class) to use.
        :param client_flag `<'int'>`: Custom flags to sent to server, see 'constants.CLIENT'.
        :param program_name `<'str/None'>`: The program name for the connection.
        :param ssl_ctx `<ssl.SSLContext/None>`: The SSL context for the connection.
        :param auth_plugin `<'AuthPlugin/None'>`: The authentication plugin handlers.
        :param server_public_key `<'bytes/None'>`: The public key for the server authentication.
        :param use_decimal `<'bool'>`: If `True` use <'Decimal'> to represent DECIMAL column data, else use <'float'>.
        :param decode_bit `<'bool'>`: If `True` decode BIT column data to <'int'>, else keep as original bytes.
        :param decode_json `<'bool'>`: If `True` deserialize JSON column data, else keep as original json string.
        """
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

    # Setup -----------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _setup_charset(self, charset: Charset) -> cython.bint:
        """(cfunc) Setup charset & collation."""
        self._charset_id = charset._id
        self._charset = charset._name
        self._collation = charset._collation
        self._encoding = charset._encoding
        self._encoding_c = charset._encoding_c
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _setup_client_flag(self, client_flag: cython.uint) -> cython.bint:
        """(cfunc) Setup the 'client_flag'"""
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
    def _setup_connect_attrs(self, program_name: str | None) -> cython.bint:
        """(cfunc) Setup the 'connect_attrs'."""
        if program_name is None:
            attrs: bytes = utils.DEFAULT_CONNECT_ATTRS + utils.gen_connect_attrs(
                [str(_getpid)]
            )
        else:
            attrs: bytes = utils.DEFAULT_CONNECT_ATTRS + utils.gen_connect_attrs(
                [str(_getpid), "program_name", program_name]
            )
        self._connect_attrs = utils.gen_length_encoded_integer(bytes_len(attrs)) + attrs
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _setup_internal(self) -> cython.bint:
        """(cfunc) Setup internal attributes."""
        # . charset
        self._charset_changed = False
        # . timeouts
        self._read_timeout_changed = False
        self._write_timeout_changed = False
        self._wait_timeout_changed = False
        self._interactive_timeout_changed = False
        self._lock_wait_timeout_changed = False
        self._execution_timeout_changed = False
        # . server
        self._server_protocol_version = -1
        self._server_info = None
        self._server_version = None
        self._server_version_major = -1
        self._server_vendor = None
        self._server_thread_id = -1
        self._server_salt = None
        self._server_status = -1
        self._server_capabilities = -1
        self._server_auth_plugin_name = None
        # . client
        self._set_use_time()
        self._secure = False
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
        return utils.decode_bytes(self._user, self._encoding_c)

    @property
    def password(self) -> str:
        """The password for login authentication. `<'str'>`."""
        return utils.decode_bytes_latin1(self._password)

    @property
    def database(self) -> str | None:
        """The default database to use by the connection. `<'str/None'>`."""
        if self._database is None:
            return None
        return utils.decode_bytes(self._database, self._encoding_c)

    @property
    def charset(self) -> str:
        """The 'CHARSET' of the connection `<'str'>`."""
        return self._charset

    @property
    def collation(self) -> str:
        """The 'COLLATION' of the connection `<'str'>`."""
        return self._collation

    @property
    def encoding(self) -> str:
        """The 'encoding' of the connection `<'str'>`."""
        return utils.decode_bytes_ascii(self._encoding)

    @property
    def connect_timeout(self) -> int:
        """Timeout in seconds for establishing the connection `<'int'>`."""
        return self._connect_timeout

    @property
    def bind_address(self) -> str | None:
        """The interface from which to connect to the host `<'str/None'>`."""
        return self._bind_address

    @property
    def unix_socket(self) -> str | None:
        """The unix socket (rather than TCP/IP) for establishing the connection `<'str/None'>`."""
        return self._unix_socket

    @property
    def autocommit(self) -> bool | None:
        """The 'autocommit' mode of the connection `<'bool/None'>`.

        - `True` if the connection is operating in autocommit (non-transactional) mode.
        - `False` if the connection is operating in manual commit (transactional) mode.
        - `None` means connection is not connected and use server default.
        """
        if self._server_status == -1:
            return None if self._autocommit_mode == -1 else bool(self._autocommit_mode)
        else:
            return bool(self._server_status & _SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT)

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
        """The default SQL_MODE for the connection `<'str/None'>`."""
        return self._sql_mode

    @property
    def init_command(self) -> str | None:
        """The initial SQL statement to run when connection is established `<'str/None'>`."""
        return self._init_command

    @property
    def client_flag(self) -> int:
        """The current client flag of the connection `<'int'>`."""
        return self._client_flag

    @property
    def ssl(self) -> object | None:
        """The 'ssl.SSLContext' for the connection `<'SSLContext/None'>`."""
        return self._ssl_ctx

    @property
    def auth_plugin(self) -> _auth.AuthPlugin | None:
        """The authentication plugin handlers `<'AuthPlugin/None'>`."""
        return self._auth_plugin

    # . server
    @property
    def thread_id(self) -> int | None:
        """The 'thread id' of connection `<'int/None'>`."""
        if self._server_thread_id != -1:
            return self._server_thread_id
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
        """The server information (name & version) `<'str/None'>`."""
        return self._server_info

    @property
    def server_version(self) -> tuple[int] | None:
        """The server version `<'tuple[int]/None'>`.
        >>> (8, 0, 23)  # example
        """
        return self.get_server_version()

    @property
    def server_version_major(self) -> int | None:
        """The server major version `<'int/None'>`.
        >>> 8  # example
        """
        if self._server_version_major != -1:
            return self._server_version_major
        else:
            return None

    @property
    def server_vendor(self) -> Literal["mysql", "mariadb"] | None:
        """The name of the server vendor (database type) `<'str/None'>`."""
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
        """The authentication plugin name of the server `<'str/None'>`."""
        return self._server_auth_plugin_name

    # . query
    @property
    def affected_rows(self) -> int:
        """The number of affected rows by the last query `<'int'>`."""
        return self.get_affected_rows()

    @property
    def insert_id(self) -> int:
        """The 'LAST_INSERT_ID' of the the last query `<'int'>`."""
        return self.get_insert_id()

    @property
    def transaction_status(self) -> bool | None:
        """Whether the connection is in a TRANSACTION `<'bool/None'>`."""
        if self._server_status != -1:
            return self.get_transaction_status()
        else:
            return None

    # . decode
    @property
    def use_decimal(self) -> bool:
        """Whether to use <'DECIMAL'> to represent
        DECIMAL column data `<'bool'>`.

        If `False`, use <'float'> instead.
        """
        return self._use_decimal

    @property
    def decode_bit(self) -> bool:
        """Whether to decode BIT column data to integer `<'bool'>`.

        If `False`, keep as the original bytes.
        """
        return self._decode_bit

    @property
    def decode_json(self) -> bool:
        """Whether to deserialize JSON column data `<'bool'>`.

        If `False`, keep as the original JSON string.
        """
        return self._decode_json

    # Cursor ----------------------------------------------------------------------------------
    @cython.ccall
    def cursor(
        self,
        cursor: type[Cursor | tuple | dict | DataFrame] | None = None,
    ) -> Cursor:
        """Acquire a new cursor `<'Cursor'>`.

        :param cursor `<'type[Cursor]/None'>`: The cursor type (class) to use. Defaults to `None` (use connection default).
            Also accepts: 'tuple' => 'Cursor' / 'dict' => 'DictCursor' / 'DataFrame' => 'DfCursor'.

        ## Example (context manager):
        >>> with conn.cursor() as cur:
                cur.execute("SELECT * FROM table")

        ## Example (direct - NOT recommended):
        >>> cur = conn.cursor()
            cur.execute("SELECT * FROM table")
            cur.close()  # close manually
        """
        self._verify_connected()
        cur = self._validate_cursor(cursor)
        return cur(self)

    @cython.ccall
    def transaction(
        self,
        cursor: type[Cursor | tuple | dict | DataFrame] | None = None,
    ) -> TransactionManager:
        """Acquire a new cursor in `TRANSACTION` mode
        through context manager `<'TransactionManager'>`.

        :param cursor `<'type[Cursor]/None'>`: The cursor type (class) to use. Defaults to `None` (use connection default).
            Also accepts: 'tuple' => 'Cursor' / 'dict' => 'DictCursor' / 'DataFrame' => 'DfCursor'.

        ## Explanation
        By acquiring cursor through this method, the following happens:
        - 1. Use the connection to `BEGIN` a transaction.
        - 2. Returns a cursor of the connection.
        - 3a. If catches ANY exceptions during the transaction, close the connection.
        - 3b. If the transaction executed successfully, execute `COMMIT` in the end.

        ## Example
        >>> with conn.transaction() as cur:
                cur.execute("INSERT INTO tb (id, name) VALUES (1, 'name')")
            # Equivalent to:
            BEGIN;
            INSERT INTO tb (id, name) VALUES (1, 'test');
            COMMIT;
        """
        self._verify_connected()
        cur = self._validate_cursor(cursor)
        return TransactionManager(self, cur)

    @cython.cfunc
    @cython.inline(True)
    def _validate_cursor(self, cursor: object) -> type:
        """(internal) Validate if the 'cursor' argument is valid `<'type'>`."""
        # Default cursor
        if cursor is None:
            return Cursor if self._cursor is None else self._cursor

        # Validate cursor
        if type(cursor) is type:
            if issubclass(cursor, Cursor):
                return cursor
            if cursor is tuple:
                return Cursor
            if cursor is dict:
                return DictCursor
            if cursor is DataFrame:
                return DfCursor
        raise errors.InvalidConnectionArgsError(
            "Invalid 'cursor' argument: %r.\n"
            "Expects type (subclass) of %r." % (cursor, Cursor)
        )

    # Query -----------------------------------------------------------------------------------
    @cython.ccall
    def query(self, sql: str, unbuffered: cython.bint = False) -> cython.ulonglong:
        """Execute a SQL query `<'int'>`

        :param sql `<'str'>`: The SQL query to execute.
        :param unbuffered `<'bool'>`: Query in unbuffered mode. Defaults to `False`.
        :returns `<'int'>`: The number of affected rows.
        """
        self._execute_command(_COMMAND.COM_QUERY, self.encode_sql(sql))
        return self._read_query_result(unbuffered)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def begin(self) -> cython.bint:
        """BEGIN a transaction."""
        self._execute_command(_COMMAND.COM_QUERY, b"BEGIN")
        self._read_ok_packet()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def start(self) -> cython.bint:
        """START TRANSACTION, alias to 'Connection.begin()'."""
        self._execute_command(_COMMAND.COM_QUERY, b"BEGIN")
        self._read_ok_packet()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def commit(self) -> cython.bint:
        """COMMIT any pending transaction to the database."""
        self._execute_command(_COMMAND.COM_QUERY, b"COMMIT")
        self._read_ok_packet()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def rollback(self) -> cython.bint:
        """ROLLBACK the current transaction."""
        self._execute_command(_COMMAND.COM_QUERY, b"ROLLBACK")
        self._read_ok_packet()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def create_savepoint(self, identifier: str) -> cython.bint:
        """Create a named transaction SAVEPOINT within the current transaction.
        If the current transaction has a savepoint with the same name, the old
        savepoint is deleted and a new one is set.

        :param identifier `<'str'>`: The name of the savepoint.
        """
        return self.query("SAVEPOINT %s;" % identifier)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def rollback_savepoint(self, identifier: str) -> cython.bint:
        """ROLLBACK a transaction to the named SAVEPOINT without terminating the
        transaction. Modifications that the current transaction made to rows after
        the savepoint was set are undone in the rollback, but InnoDB does not release
        the row locks that were stored in memory after the savepoint. (For a new
        inserted row, the lock information is carried by the transaction ID stored
        in the row; the lock is not separately stored in memory. In this case, the
        row lock is released in the undo.) Savepoints that were set at a later time
        than the named savepoint are deleted.

        :param identifier `<'str'>`: The name of the savepoint.
        """
        return self.query("ROLLBACK TO SAVEPOINT %s;" % identifier)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def release_savepoint(self, identifier: str) -> cython.bint:
        """RELEASE the named SAVEPOINT from the set of savepoints of the
        current transaction. No commit or rollback occurs. It raises an
        error if the savepoint does not exist.

        :param identifier `<'str'>`: The name of the savepoint.
        """
        return self.query("RELEASE SAVEPOINT %s;" % identifier)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def kill(self, thread_id: cython.int) -> cython.bint:
        """Execute KILL command.

        :param thread_id `<'int'>`: The thread ID to be killed.
        """
        try:
            self._execute_command(
                _COMMAND.COM_PROCESS_KILL, utils.pack_int32(thread_id)
            )
            self._read_ok_packet()
        except errors.OperationalUnknownCommandError:
            # if COM_PROCESS_KILL [0x0C] raises 'unknown command'
            # error, try execute 'KILL {thread_id}' query instead.
            self._execute_command(_COMMAND.COM_QUERY, b"KILL %d" % thread_id)
            self._read_ok_packet()
        return True

    @cython.ccall
    def show_warnings(self) -> tuple[tuple]:
        """Execute 'SHOW WARNINGS' sql and returns
        the warnings `<'tuple[tuple]'>`."""
        self.query("SHOW WARNINGS")
        return self._result._rows

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def select_database(self, db: str) -> cython.bint:
        """Select database for the connection (SESSION).

        :param db `<'str'>`: The name of the database to select.
        """
        database: bytes = self.encode_sql(db)
        self._execute_command(_COMMAND.COM_INIT_DB, database)
        self._read_ok_packet()
        self._database = database
        return True

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

    @cython.ccall
    def encode_sql(self, sql: str) -> bytes:
        """Encode the 'sql' with connection's encoding `<'bytes'>`.

        :param sql `<'str'>`: The sql to be encoded.
        """
        return utils.encode_str(sql, self._encoding_c)

    # . client
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_charset(
        self,
        charset: str,
        collation: str | None = None,
    ) -> cython.bint:
        """Set CHARSET and COLLATION of the connection.

        :param charset `<'str'>`: The charset.
        :param collation `<'str/None'>`: The collation. Defaults to `None`.
        """
        ch: Charset = utils.validate_charset(charset, collation, utils.DEFUALT_CHARSET)
        if ch._name != self._charset or ch._collation != self._collation:
            self._setup_charset(ch)
            sql = "SET NAMES %s COLLATE %s" % (self._charset, self._collation)
            self._execute_command(_COMMAND.COM_QUERY, self.encode_sql(sql))
            self._read_ok_packet()
            self._charset_changed = True
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_autocommit(self) -> cython.bint:
        """Get the 'autocommit' mode of the connection `<'bool'>`.

        - `True` if the connection is operating in autocommit (non-transactional) mode.
        - `False` if the connection is operating in manual commit (transactional) mode.

        :raises `<'ConnectionClosedError'>`: If connection is not connected.
        """
        if self._server_status == -1:
            raise errors.ConnectionClosedError(0, "Connection not connected.")
        return self._server_status & _SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_autocommit(self, value: cython.bint) -> cython.bint:
        """Set the 'autocommit' mode of the connection (SESSION).

        :param value `<'bool'>`: Enable/Disable autocommit.
            - `True` to operate in autocommit (non-transactional) mode.
            - `False` to operate in manual commit (transactional) mode.
        """
        self._autocommit_mode = int(value)
        if not self.closed() and value != self.get_autocommit():
            self._execute_command(
                _COMMAND.COM_QUERY,
                b"SET AUTOCOMMIT = 1" if value else b"SET AUTOCOMMIT = 0",
            )
            self._read_ok_packet()
        return True

    # . timeouts
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_read_timeout(self, value: int | None) -> cython.bint:
        """Set connection (SESSION) 'net_read_timeout'.

        :param value `<'int/None'>`: The timeout (in seconds) for reading data from the network during a query. Defaults to `None`.
            - When 'value=None', if connection 'read_timeout' is specified, set
              the timeout to the connection default value, else set to the server
              GLOBAL default value.

        ## Explanation:
        - The 'net_read_timeout' defines the total seconds for reading data from the
          network during a query.
        - When the MySQL client or server is waiting to receive data over the network
          (for example, when sending large result sets back to the client), this setting
          specifies how long the server should wait before it gives up and times out.
        - If a query is taking longer than expected to transfer data over the network
          (perhaps due to slow network conditions or a large result set), this value
          dictates how long MySQL will wait before throwing a timeout error.
        """
        name = "net_read_timeout"
        # Validate timeout
        value = utils.validate_arg_uint(value, name, 1, UINT_MAX)

        # Reset timeout
        if value is None:
            if self._read_timeout is None:
                value = self._get_timeout(name, False)
            else:
                value = self._read_timeout
            self._set_timeout(name, value)
            self._read_timeout_changed = False

        # Set timeout
        else:
            self._set_timeout(name, value)
            self._read_timeout_changed = True
        return True

    @cython.ccall
    def get_read_timeout(self) -> cython.uint:
        """Get connection (SESSION) 'net_read_timeout' `<'int'>`.

        ## Explanation:
        - The 'net_read_timeout' defines the total seconds for reading data from the
          network during a query.
        - When the MySQL client or server is waiting to receive data over the network
          (for example, when sending large result sets back to the client), this setting
          specifies how long the server should wait before it gives up and times out.
        - If a query is taking longer than expected to transfer data over the network
          (perhaps due to slow network conditions or a large result set), this value
          dictates how long MySQL will wait before throwing a timeout error.
        """
        return self._get_timeout("net_read_timeout", True)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_write_timeout(self, value: int | None) -> cython.bint:
        """Set connection (SESSION) 'net_write_timeout'.

        :param value `<'int/None'>`: The timeout (in seconds) for writing data to the network during a query. Defaults to `None`.
            - When 'value=None', if connection 'write_timeout' is specified, set
              the timeout to the connection default value, else set to the server
              GLOBAL default value.

        ## Explanation
        - The 'net_write_timeout' defines the total seconds for writing data to
          the network during a query.
        - Similar to 'net_read_timeout', but for writing operations, it determines
          how long the server will attempt to send data before it stops and raises
          an error.
        """
        name = "net_write_timeout"
        # Validate timeout
        value = utils.validate_arg_uint(value, name, 1, UINT_MAX)

        # Reset timeout
        if value is None:
            if self._write_timeout is None:
                value = self._get_timeout(name, False)
            else:
                value = self._write_timeout
            self._set_timeout(name, value)
            self._write_timeout_changed = False

        # Set timeout
        else:
            self._set_timeout(name, value)
            self._write_timeout_changed = True
        return True

    @cython.ccall
    def get_write_timeout(self) -> cython.uint:
        """Get connection (SESSION) 'net_write_timeout' `<'int'>`.

        ## Explanation
        - The 'net_write_timeout' defines the total seconds for writing data to
          the network during a query.
        - Similar to 'net_read_timeout', but for writing operations, it determines
          how long the server will attempt to send data before it stops and raises
          an error.
        """
        return self._get_timeout("net_write_timeout", True)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_wait_timeout(self, value: int | None) -> cython.bint:
        """Set connection (SESSION) 'wait_timeout'.

        :param value `<'int/None'>`: The timeout (in seconds) for waiting a non-interactive client activity before closing the connection. Defaults to `None`.
            - When 'value=None', if connection 'wait_timeout' is specified, set
              the timeout to the connection default value, else set to the server
              GLOBAL default value.

        ## Explanation
        - The 'wait_time' defines the amount of time (in seconds) the MySQL server will wait
          for a new request from a non-interactive client before closing the connection.
        - For long-running connections that do not perform any operations, this is the time
          after which the server will disconnect the client if no query activity happens.
        """
        name = "wait_timeout"
        # Validate timeout
        value = utils.validate_arg_uint(value, name, 1, UINT_MAX)

        # Reset timeout
        if value is None:
            if self._wait_timeout is None:
                value = self._get_timeout(name, False)
            else:
                value = self._wait_timeout
            self._set_timeout(name, value)
            self._wait_timeout_changed = False

        # Set timeout
        else:
            self._set_timeout(name, value)
            self._wait_timeout_changed = True
        return True

    @cython.ccall
    def get_wait_timeout(self) -> cython.uint:
        """Get connection (SESSION) 'wait_timeout' `<'int'>`.

        ## Explanation
        - The 'wait_time' defines the amount of time (in seconds) the MySQL server will wait
          for a new request from a non-interactive client before closing the connection.
        - For long-running connections that do not perform any operations, this is the time
          after which the server will disconnect the client if no query activity happens.
        """
        return self._get_timeout("wait_timeout", True)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_interactive_timeout(self, value: int | None) -> cython.bint:
        """Set connection (SESSION) 'interactive_timeout'.

        :param value `<'int/None'>`: The timeout (in seconds) for waiting an interactive client activity before closing the connection. Defaults to `None`.
            - When 'value=None', if connection 'interactive_timeout' is specified, set
              the timeout to the connection default value, else set to the server
              GLOBAL default value.

        ## Explanation
        - The 'interactive_timeout' defines the amount of time (in seconds) the MySQL server
          will wait for a new request from an interactive client before closing the connection.
        - For long-running connections that do a long pause between queries, this is the time
          after which the server will disconnect the client if no query activity happens.
        """
        name = "interactive_timeout"
        # Validate timeout
        value = utils.validate_arg_uint(value, name, 1, UINT_MAX)

        # Reset timeout
        if value is None:
            if self._interactive_timeout is None:
                value = self._get_timeout(name, False)
            else:
                value = self._interactive_timeout
            self._set_timeout(name, value)
            self._interactive_timeout_changed = False

        # Set timeout
        else:
            self._set_timeout(name, value)
            self._interactive_timeout_changed = True
        return True

    @cython.ccall
    def get_interactive_timeout(self) -> cython.uint:
        """Get connection (SESSION) 'interactive_timeout' `<'int'>`.

        ## Explanation
        - The 'wait_time' defines the amount of time (in seconds) the MySQL server will wait
          for a new request from a non-interactive client before closing the connection.
        - For long-running connections that do not perform any operations, this is the time
          after which the server will disconnect the client if no query activity happens.
        """
        return self._get_timeout("interactive_timeout", True)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_lock_wait_timeout(self, value: int | None) -> cython.bint:
        """Set connection (SESSION) 'innodb_lock_wait_timeout'.

        :param value `<'int/None'>`: The timeout (in seconds) for waiting for a row lock in the InnoDB storage engine. Defaults to `None`.
            - When 'value=None', if connection 'lock_wait_timeout' is specified, set
              the timeout to the connection default value, else set to the server
              GLOBAL default value.

        ## Explanation
        - The 'innodb_lock_wait_timeout' defines how long a transaction will wait
          for an InnoDB row-level lock to be acquired.
        - If a transaction tries to acquire a lock on a row or set of rows and another
          transaction already holds the lock, it will wait for this period before
          returning an error.
        - This timeout does not apply to waits for table locks, since table lock
          does not happen inside InnoDB.
        """
        name = "innodb_lock_wait_timeout"
        # Validate timeout
        value = utils.validate_arg_uint(value, name, 1, UINT_MAX)

        # Reset timeout
        if value is None:
            if self._lock_wait_timeout is None:
                value = self._get_timeout(name, False)
            else:
                value = self._lock_wait_timeout
            self._set_timeout(name, value)
            self._lock_wait_timeout_changed = False

        # Set timeout
        else:
            self._set_timeout(name, value)
            self._lock_wait_timeout_changed = True
        return True

    @cython.ccall
    def get_lock_wait_timeout(self) -> cython.uint:
        """Get connection (SESSION) 'innodb_lock_wait_timeout' `<'int'>`.

        ## Explanation
        - The 'innodb_lock_wait_timeout' defines how long a transaction will wait
          for an InnoDB row-level lock to be acquired.
        - If a transaction tries to acquire a lock on a row or set of rows and another
          transaction already holds the lock, it will wait for this period before
          returning an error.
        - This timeout does not apply to waits for table locks, since table lock
          does not happen inside InnoDB.
        """
        return self._get_timeout("innodb_lock_wait_timeout", True)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_execution_timeout(self, value: int | None) -> cython.bint:
        """Set connection (SESSION) 'max_execution_time'.

        :param value `<'int/None'>`: The timeout (in milliseconds) for executing read-only SELECT statements. Defaults to `None`.
            - When 'value=None', if connection 'execution_timeout' is specified, set
              the timeout to the connection default value, else set to the server
              GLOBAL default value.

        ## Explanation
        - The 'max_execution_time' defines the max execution time (in milliseconds) for read-only
          SELECT statements. Statements that are not read only are those that invoke a stored
          function that modifies data as a side effect.
        - The 'max_execution_time' is ignored for SELECT statements in stored programs.
        - Value of '0' mean to disable the timeout.
        """
        name = "max_execution_time"
        # Validate timeout
        value = utils.validate_arg_uint(value, name, 0, UINT_MAX)

        # Reset timeout
        if value is None:
            if self._execution_timeout is None:
                value = self._get_timeout(name, False)
            else:
                value = self._execution_timeout
            self._set_timeout(name, value)
            self._execution_timeout_changed = False

        # Set timeout
        else:
            self._set_timeout(name, value)
            self._execution_timeout_changed = True
        return True

    @cython.ccall
    def get_execution_timeout(self) -> cython.uint:
        """Get connection (SESSION) 'max_execution_time' `<'int'>`.

        ## Explanation
        - The 'max_execution_time' defines the max execution time (in milliseconds) for read-only
          SELECT statements. Statements that are not read only are those that invoke a stored
          function that modifies data as a side effect.
        - The 'max_execution_time' is ignored for SELECT statements in stored programs.
        - Value of '0' mean the timeout is disabled.
        """
        return self._get_timeout("max_execution_time", True)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_timeout(self, name: str, value: object) -> cython.bint:
        """(cfunc) Set connection (SESSION) timeout.

        :param name `<'str'>`: The name of the timeout setting.
        :param value `<'int'>`: The timeout value in seconds.
        """
        try:
            value = int(value)
        except Exception as err:
            raise errors.ConnectionValueError(
                "Invalid '%s' value: %s." % (name, value)
            ) from err
        sql = "SET SESSION %s = %d" % (name, value)
        self._execute_command(_COMMAND.COM_QUERY, self.encode_sql(sql))
        self._read_ok_packet()
        return True

    @cython.cfunc
    @cython.inline(True)
    def _get_timeout(self, name: str, session: cython.bint) -> cython.uint:
        """(cfunc) Get SESSION/GLOBAL timeout `<'int'>`.

        :param name `<'str'>`: The name of the timeout setting.
        :param session `<'bool'>: If True get SESSION timeout, else get GLOBAL timeout.
        :returns `<'int'>`: The timeout value in seconds.
        """
        if session:
            sql = "SHOW VARIABLES LIKE '%s'" % name
        else:
            sql = "SHOW GLOBAL VARIABLES LIKE '%s'" % name
        self.query(sql, False)
        try:
            return int(self._result._rows[0][1])
        except Exception as err:
            raise errors.DatabaseError(
                "Failed to get %s '%s' from server: %s"
                % ("SESSION" if session else "GLOBAL", name, err)
            ) from err

    # . server
    @cython.ccall
    def get_server_version(self) -> tuple[int]:
        """Get the server version `<'tuple[int]/None'>`.
        >>> (8, 0, 23)  # example
        """
        if self._server_version is None:
            # Not connected
            if self._server_info is None:
                return None
            # Parser version
            m = utils.SERVER_VERSION_RE.match(self._server_info)
            if m is None:
                raise errors.DatabaseError(
                    "Failed to parse server version from '%s'." % self._server_info
                )
            cmp: tuple = m.group(1, 2, 3)
            self._server_version = list_to_tuple(
                [0 if i is None else int(i) for i in cmp]
            )
        return self._server_version

    @cython.ccall
    def get_server_vendor(self) -> str:
        """Get the name of the server vendor (database type) `<'str/None'>`."""
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

    # . status
    @cython.ccall
    def get_affected_rows(self) -> cython.ulonglong:
        """Get the number of affected rows by the last query `<'int'>`."""
        return 0 if self._result is None else self._result._affected_rows

    @cython.ccall
    def get_insert_id(self) -> cython.ulonglong:
        """Get the 'LAST_INSERT_ID' of the the last query `<'int'>`."""
        return 0 if self._result is None else self._result._insert_id

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def get_transaction_status(self) -> cython.bint:
        """Get whether the connection is in a TRANSACTION `<'bool'>`."""
        if self._server_status == -1:
            raise errors.ConnectionClosedError(0, "Connection not connected.")
        return self._server_status & _SERVER_STATUS.SERVER_STATUS_IN_TRANS

    # . decode
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_use_decimal(self, value: cython.bint) -> cython.bint:
        """Set whether to use `<'DECIMAL'> to represent DECIMAL column data.

        :param value `<'bool'>`: Use <'DECIMAL>' if True, else use <'float'>.
        """
        self._use_decimal = value
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_decode_bit(self, value: cython.bint) -> cython.bint:
        """Set whether to decode BIT column data to integer.

        :param value `<'bool'>`: True to decode BIT column, else keep as original bytes.
        """
        self._decode_bit = value
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set_decode_json(self, value: cython.bint) -> cython.bint:
        """Set whether to deserialize JSON column data.

        :param value `<'bool'>`: True to deserialize JSON column, else keep as original JSON string.
        """
        self._decode_json = value
        return True

    # Connect / Close -------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def connect(self) -> cython.bint:
        """Establish connection with server.

        If the connection is already established,
        no action will be taken.
        """
        if self.closed():
            self._connect()
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _connect(self) -> cython.bint:
        """(cfunc) Establish connection with server.

        ### This method will establish connection REGARDLESS of the connection state.
        """
        # Connect
        sock = None
        try:
            # . create socket
            if self._unix_socket is not None:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self._connect_timeout)
                sock.connect(self._unix_socket)
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
            self._execute_command(
                _COMMAND.COM_QUERY,
                self.encode_sql(
                    "SET NAMES %s COLLATE %s" % (self._charset, self._collation)
                ),
            )
            self._read_ok_packet()

            # . timeouts
            if self._read_timeout is not None:
                self._set_timeout("net_read_timeout", self._read_timeout)
            if self._write_timeout is not None:
                self._set_timeout("net_write_timeout", self._write_timeout)
            if self._wait_timeout is not None:
                self._set_timeout("wait_timeout", self._wait_timeout)
            if self._interactive_timeout is not None:
                self._set_timeout("interactive_timeout", self._interactive_timeout)
            if self._lock_wait_timeout is not None:
                self._set_timeout("innodb_lock_wait_timeout", self._lock_wait_timeout)
            if self._execution_timeout is not None:
                self._set_timeout("max_execution_time", self._execution_timeout)

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
                auto: cython.bint = self._autocommit_mode == 1
                self._execute_command(
                    _COMMAND.COM_QUERY,
                    b"SET AUTOCOMMIT = 1" if auto else b"SET AUTOCOMMIT = 0",
                )
                self._read_ok_packet()

        # Failed
        except BaseException as err:
            if sock is not None:
                try:
                    sock.close()
                except Exception:  # noqa
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
        """Send QUIT command and close the connection
        socket. The connection will be unusable after
        this method is called.

        ### This method does not raise any error.
        """
        # Close connection
        if not self.closed():
            try:
                self._write_bytes(utils.pack_IB(1, _COMMAND.COM_QUIT))
            except Exception:
                pass
            finally:
                self.force_close()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def force_close(self) -> cython.bint:
        """Close the connection socket directly without
        sending a QUIT command. The connection will be
        unusable after this method is called.

        ### This method does not raise any error.
        """
        if not self.closed():
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer, self._reader = None, None
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _close_with_reason(self, reason: str) -> cython.bint:
        """(cfunc) Force close the connection socket while
        setting the closed reason. The connection will be
        unusable after this method is called.

        ### This method does not raise any error.
        """
        self._close_reason = "Connection closed: %s" % reason
        self.force_close()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def closed(self) -> cython.bint:
        """Represents the connection state: whether is closed `<'bool'>`."""
        return self._writer is None and self._reader is None

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def ping(self, reconnect: cython.bint = True) -> cython.bint:
        """Check if the connection is alive.

        :param reconnect `<'bool'>`: Whether to reconnect if disconnected. Defaults to `True`.
            - If 'reconnect=False' and the connection is disconnected,
              raise 'ConnectionClosedError' directly.
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
        """(cfunc) [Handshake] Setup server information."""
        # Read from packet
        pkt: MysqlPacket = self._read_packet()
        data: cython.pchar = pkt._data_c
        length: cython.Py_ssize_t = pkt._size

        # . protocol version
        self._server_protocol_version = utils.unpack_uint8(data, 0)

        # . server version
        loc: cython.Py_ssize_t = utils.find_null_term(data, 1)
        self._server_info = utils.decode_bytes_latin1(data[1:loc])
        self._server_version_major = int(str_split(self._server_info, ".", 1)[0])
        i: cython.Py_ssize_t = loc + 1

        # . server_thred_id
        self._server_thread_id = utils.unpack_uint32(data, i)
        i += 4

        # . salt
        self._server_salt = data[i : i + 8]
        i += 9  # 8 + 1(filler)

        # . server capabilities
        self._server_capabilities = utils.unpack_uint16(data, i)
        i += 2

        # . server_status & adj server_capabilities
        salt_len: cython.Py_ssize_t
        if length >= i + 6:
            i += 1
            self._server_status = utils.unpack_uint16(data, i)
            i += 2
            self._server_capabilities |= utils.unpack_uint16(data, i) << 16
            i += 2
            salt_len = max(12, utils.unpack_uint8(data, i) - 9)
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
            loc = utils.find_null_term(data, i)
            if loc < 0:  # pragma: no cover - very specific upstream bug
                # not found \0 and last field so take it all
                auth_plugin_name: bytes = data[i:length]
            else:
                auth_plugin_name: bytes = data[i:loc]
            self._server_auth_plugin_name = utils.decode_bytes_utf8(auth_plugin_name)

        # Finished
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _request_authentication(self) -> cython.bint:
        """(cfunc) [Handshake] Request authentication."""
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
        header: bytes = utils.pack_IIB23s(
            self._client_flag, utils.MAX_PACKET_LENGTH, self._charset_id
        )
        data: list[bytes] = [header]

        # . ssl connection
        if self._ssl_ctx is not None and self._server_capabilities & _CLIENT.SSL:
            self._write_packet(header)
            self._writer = self._ssl_ctx.wrap_socket(
                self._writer, server_hostname=self._host
            )
            self._reader = self._writer.makefile("rb")
            self._secure = True

        # . collect
        data.append(self._user)
        data.append(b"\0")

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
            data.append(utils.gen_length_encoded_integer(bytes_len(authres)))
            data.append(authres)
        elif self._server_capabilities & _CLIENT.SECURE_CONNECTION:
            data.append(utils.pack_uint8(bytes_len(authres)))
            data.append(authres)
        else:
            data.append(authres)
            data.append(b"\0")

        # . database
        if (
            self._database is not None
            and self._server_capabilities & _CLIENT.CONNECT_WITH_DB
        ):
            data.append(self._database)
            data.append(b"\0")

        # . auth plugin name
        if self._server_capabilities & _CLIENT.PLUGIN_AUTH:
            data.append(b"" if plugin_name is None else plugin_name)
            data.append(b"\0")

        # . connect attrs
        if self._server_capabilities & _CLIENT.CONNECT_ATTRS:
            data.append(self._connect_attrs)

        # . write packet
        self._write_packet(b"".join(data))
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
        """(cfunc) [Handshake] Process authentication response `<'MysqlPacket'>`."""
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
                        % (utils.decode_bytes_ascii(plugin_name), plugin_handler),
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
                            % (utils.decode_bytes_ascii(plugin_name), plugin_handler),
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
                            % (utils.decode_bytes_ascii(plugin_name), plugin_handler),
                        ) from err
                    except TypeError as err:
                        raise errors.AuthenticationError(
                            # fmt: off
                            CR.CR_AUTH_PLUGIN_ERR,
                            "Authentication plugin '%s' didn't respond with string:"
                            "%r returns '%r' to prompt: %r"
                            % ( utils.decode_bytes_ascii(plugin_name), 
                                plugin_handler, resp, prompt ),
                            # fmt: on
                        ) from err
                else:
                    raise errors.AuthenticationError(
                        CR.CR_AUTH_PLUGIN_CANNOT_LOAD,
                        "Authentication plugin '%s' not configured."
                        % utils.decode_bytes_ascii(plugin_name),
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
                % utils.decode_bytes_ascii(plugin_name),
            )
        # Auth: 'mysql_native_password', 'client_ed25519' & 'mysql_clear_password'.
        self._write_packet(data)
        pkt = self._read_packet()
        pkt.check_error()
        return pkt

    @cython.cfunc
    @cython.inline(True)
    def _process_auth_caching_sha2(self, pkt: MysqlPacket) -> MysqlPacket:
        """(cfunc) [Handshake] Process 'caching_sha2_password'
        authentication response `<'MysqlPacket'>`."""
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
        """(cfunc) [Handshake] Process 'sha256_password'
        authentication response `<'MysqlPacket'>`."""
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
        """(cfunc) [Handshake] Process authentication: send data `<'MysqlPacket'>`."""
        self._write_packet(data)
        pkt: MysqlPacket = self._read_packet()
        pkt.check_error()
        return pkt

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _verify_connected(self) -> cython.bint:
        """(cfunc) Verify if the connection is connected.

        :raises `<'ConnectionClosedError'>`: If the connection has already been closed.
        """
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
        """(cfunc) Execute SQL command.

        :param command `<'int'>: The command code, see 'constants.COMMAND'.
        :param sql `<'bytes'>`: The query SQL encoded with connection's encoding.
        """
        # Validate connection
        self._verify_connected()

        # If the last query was unbuffered, make sure it finishes
        # before sending new commands
        if self._result is not None:
            if self._result._unbuffered_active:
                warnings.warn("Previous unbuffered result was left incomplete.")
                self._result._drain_result_packet_unbuffered()
            while self._result._has_next:
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
            sql_size + 1, utils.MAX_PACKET_LENGTH
        )  # +1 is for command
        data = utils.pack_IB(pkt_size, command) + sql[0 : pkt_size - 1]
        self._write_bytes(data)
        self._next_seq_id = 1

        # Write remaining data
        pos: cython.ulonglong = pkt_size - 1
        while pos < sql_size:
            pkt_size = min(sql_size - pos, utils.MAX_PACKET_LENGTH)
            self._write_packet(sql[pos : pos + pkt_size])
            pos += pkt_size

        # Set use time & exit
        self._set_use_time()
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _write_packet(self, payload: bytes) -> cython.bint:
        """(cfunc) Writes the 'payload' as packet in its entirety
        to the network (adds length, sequence number and update
        'self._next_seq_id').
        """
        # Internal note: when you build packet manually and calls
        # '_write_bytes()' directly instead of '_write_packet()',
        # you should set 'self._next_seq_id' properly.
        data = utils.pack_I24B(bytes_len(payload), self._next_seq_id) + payload
        self._write_bytes(data)
        self._next_seq_id = (self._next_seq_id + 1) % 256
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _write_bytes(self, data: bytes) -> cython.bint:
        """(cfunc) Writes the 'data' directly to the network."""
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

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_use_time(self) -> cython.bint:
        """(cfunc) Set (recorde) the 'last_used_time' of the connection."""
        self._last_used_time = unix_time()
        return True

    # Read ------------------------------------------------------------------------------------
    @cython.ccall
    def next_result(self, unbuffered: cython.bint = False) -> cython.ulonglong:
        """Go to the next query result, and returns
        the affected/selected rows `<'int'>`.

        :param unbuffered `<'bool'>`: Query in unbuffered mode. Defaults to `False`.
        """
        return self._read_query_result(unbuffered)

    @cython.cfunc
    @cython.inline(True)
    def _read_ok_packet(self) -> MysqlPacket:
        """(cfunc) Read the next packet as OKPacket `<'MysqlPacket'>`

        :raises `<'CommandOutOfSyncError'>`: If next packet is `NOT` OKPacket.
        """
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
        """(cfunc) Read the next query result, and returns
        the affected/selected rows `<'int'>`.

        :param unbuffered `<'bool'>`: Query in unbuffered mode. Defaults to `False`.
        """
        # Read result
        self._result = None
        if unbuffered:
            try:
                result = MysqlResult(self)
                result.init_unbuffered_query()
            except:  # noqa
                result._unbuffered_active = False
                result._conn = None
                raise
        else:
            result = MysqlResult(self)
            result.read()
        self._result = result
        # Update status
        if result._server_status != -1:
            self._server_status = result._server_status
        return result._affected_rows

    @cython.cfunc
    @cython.inline(True)
    def _read_packet(self) -> MysqlPacket:
        """(cfunc) Read the next packet in its entirety
        from the network and returns `<'MysqlPacket'>`.

        :raises `<'OperationalError'>`: If the connection to the server is lost.
        :raises `<'InternalError'>`: If the packet sequence number is wrong.

        ### For FieldDescriptorPacket, use '_read_field_descriptor_packet()' instead.
        """
        buffer: bytes = self._read_packet_buffer()
        pkt = MysqlPacket(buffer, self._encoding)
        if pkt.is_error_packet():
            if self._result is not None and self._result._unbuffered_active:
                self._result._unbuffered_active = False
            pkt.raise_error()
        return pkt

    @cython.cfunc
    @cython.inline(True)
    def _read_field_descriptor_packet(self) -> FieldDescriptorPacket:
        """(cfunc) Read the next packet as `<'FieldDescriptorPacket'>`.

        :raises `<'OperationalError'>`: If the connection to the server is lost.
        :raises `<'InternalError'>`: If the packet sequence number is wrong.
        """
        buffer: bytes = self._read_packet_buffer()
        pkt = FieldDescriptorPacket(buffer, self._encoding)
        if pkt.is_error_packet():
            if self._result is not None and self._result._unbuffered_active:
                self._result._unbuffered_active = False
            pkt.raise_error()
        return pkt

    @cython.cfunc
    @cython.inline(True)
    def _read_packet_buffer(self) -> bytes:
        """(cfunc) Read the next packet in its entirety from
        the network and returns the data buffer `<'bytes'>`.

        :raises `<'OperationalError'>`: If the connection to the server is lost.
        :raises `<'InternalError'>`: If the packet sequence number is wrong.
        """
        # Read buffer data
        buffer: list[bytes] = []
        while True:
            data: bytes = self._read_bytes(4)
            packet_header: cython.pchar = bytes_to_chars(data)
            btrl: cython.uint = utils.unpack_uint16(packet_header, 0)
            btrh: cython.uint = utils.unpack_uint8(packet_header, 2)
            packet_number: cython.uint = utils.unpack_uint8(packet_header, 3)
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
            buffer.append(recv_data)
            # https://dev.mysql.com/doc/internals/en/sending-more-than-16mbyte.html
            if bytes_to_read < utils.MAX_PACKET_LENGTH:
                break

        # Return buffer
        return b"".join(buffer)

    @cython.cfunc
    @cython.inline(True)
    def _read_bytes(self, size: cython.uint) -> bytes:
        """(cfunc) Read data from the network based on the
        given 'size', and returns the data in `<'bytes'>`.

        :param size `<'int'>`: The number of bytes to read.
        :raises `<'OperationalError'>`: If the connection to the server is lost.
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
    """Represents the socket connection to the server.

    This class inherits from the `<'BaseConnection'>` class, and
    will perform argument validations during initialization.
    """

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
        cursor: type[Cursor | tuple | dict | DataFrame] | None = Cursor,
        client_flag: int = 0,
        program_name: str | None = None,
        option_file: str | bytes | PathLike | OptionFile | None = None,
        ssl: SSL | object | None = None,
        auth_plugin: dict[str | bytes, type] | _auth.AuthPlugin | None = None,
        server_public_key: bytes | None = None,
        use_decimal: bool = False,
        decode_bit: bool = False,
        decode_json: bool = False,
    ):
        """The socket connection to the server.

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
        # . internal
        self._setup_internal()
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
        self._setup_charset(utils.validate_charset(charset, collation, utils.DEFUALT_CHARSET))
        encoding: cython.pchar = self._encoding_c
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
        self._local_infile = bool(local_infile)
        self._max_allowed_packet = utils.validate_max_allowed_packet(
            max_allowed_packet, utils.DEFALUT_MAX_ALLOWED_PACKET, utils.MAXIMUM_MAX_ALLOWED_PACKET)
        self._sql_mode = utils.validate_sql_mode(sql_mode)
        self._init_command = utils.validate_arg_str(init_command, "init_command", None)
        self._cursor = None
        self._cursor = self._validate_cursor(cursor)
        self._setup_client_flag(utils.validate_arg_uint(client_flag, "client_flag", 0, UINT_MAX))
        self._setup_connect_attrs(utils.validate_arg_str(program_name, "program_name", None))
        # . ssl
        self._ssl_ctx = utils.validate_ssl(ssl)
        # fmt: on
        # . auth
        if auth_plugin is not None:
            if isinstance(auth_plugin, _auth.AuthPlugin):
                self._auth_plugin = auth_plugin
            elif isinstance(auth_plugin, dict):
                self._auth_plugin = _auth.AuthPlugin(auth_plugin)
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
        self._use_decimal = bool(use_decimal)
        self._decode_bit = bool(decode_bit)
        self._decode_json = bool(decode_json)
