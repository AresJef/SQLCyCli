# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.bytes import PyBytes_GET_SIZE as bytes_len  # type: ignore
from cython.cimports.sqlcycli import errors  # type: ignore
from cython.cimports.sqlcycli.transcode import decode_bytes  # type: ignore
from cython.cimports.sqlcycli.constants import _FIELD_TYPE, _SERVER_STATUS  # type: ignore

# Python imports
from sqlcycli import errors
from sqlcycli.constants import _FIELD_TYPE, _SERVER_STATUS

__all__ = ["MysqlPacket", "FieldDescriptorPacket"]

# Constants -----------------------------------------------------------------------------------
NULL_COLUMN: cython.uint = 251
UNSIGNED_CHAR_COLUMN: cython.uint = 251
UNSIGNED_SHORT_COLUMN: cython.uint = 252
UNSIGNED_INT24_COLUMN: cython.uint = 253
UNSIGNED_INT64_COLUMN: cython.uint = 254


# MySQL Packet --------------------------------------------------------------------------------
@cython.cclass
class MysqlPacket:
    """Representation of a MySQL response packet. Reads in the packet
    from the network socket, removes packet header and provides an interface
    for reading/parsing the packet results."""

    # Raw Data
    _data: bytes
    _data_c: cython.pchar
    _encoding: cython.pchar
    _size: cython.ulonglong
    _pos: cython.ulonglong
    # Packet Data
    _affected_rows: cython.ulonglong
    _insert_id: cython.ulonglong
    _server_status: cython.int  # Value of -1 means None
    _warning_count: cython.uint
    _has_next: cython.bint
    _message: bytes
    _filename: bytes
    _plugin_name: bytes
    _salt: bytes

    def __init__(self, data: bytes, encoding: cython.bytes) -> None:
        """MySQL response packet.

        :param data `<'bytes'>`: The raw data of the packet.
        :param encoding `<'bytes'>`: The encoding of the packet data.
        """
        # Raw Data
        self._data = data
        self._data_c = data
        self._encoding = encoding
        self._size = bytes_len(data)
        self._pos = 0
        # Packet Data
        self._affected_rows = 0
        self._insert_id = 0
        self._server_status = -1
        self._warning_count = 0
        self._has_next = False
        self._message = None
        self._filename = None
        self._plugin_name = None
        self._salt = None

    # Property --------------------------------------------------------------------------------
    @property
    def affected_rows(self) -> int:
        return self._affected_rows

    @property
    def insert_id(self) -> int:
        return self._insert_id

    @property
    def server_status(self) -> int | None:
        return None if self._server_status == -1 else self._server_status

    @property
    def warning_count(self) -> int:
        return self._warning_count

    @property
    def has_next(self) -> bool:
        return self._has_next

    @property
    def message(self) -> bytes | None:
        return self._message

    @property
    def filename(self) -> bytes | None:
        return self._filename

    @property
    def plugin_name(self) -> bytes | None:
        return self._plugin_name

    @property
    def salt(self) -> bytes | None:
        return self._salt

    # Read Data -------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def read_all_data(self) -> bytes:
        """(cfunc) Return all the data of the MySQL packet `<'bytes'>`."""
        return self._data

    @cython.cfunc
    @cython.inline(True)
    def read(self, size: cython.ulonglong) -> bytes:
        """(cfunc) Read the data of the given 'size' from the current
        position and then advance MySQL packet cursor `<'bytes'>`."""
        pos: cython.ulonglong = self._pos
        end: cython.ulonglong = pos + size
        if end > self._size:
            raise errors.MysqlPacketCursorError(
                "<'%s'>\nRequested data size overflow:\n"
                "Expected Size=%d | Current Position: %d | Data Length: %d"
                % (self.__class__.__name__, size, self._pos, self._size)
            )
        self._pos = end
        return self._data_c[pos:end]

    @cython.cfunc
    @cython.inline(True)
    def read_remains(self) -> bytes:
        """(cfunc) Read the data remains in the MySQL packet `<'bytes'>`."""
        # No more data remains
        pos: cython.ulonglong = self._pos
        if pos >= self._size:
            return b""
        # Read remaining data
        self._pos = self._size  # eof
        return self._data_c[pos : self._size]

    @cython.cfunc
    @cython.inline(True)
    def read_length_encoded_integer(self) -> cython.ulonglong:
        """(cfunc) Read a 'Length Coded Integer' from the MySQL packet buffer `<'int'>`.

        Length coded numbers can be anywhere from 1 to 9
        bytes depending on the value of the first byte.
        """
        code: cython.uint = self._read_uint8()
        if code < UNSIGNED_CHAR_COLUMN:
            return code
        if code == UNSIGNED_SHORT_COLUMN:
            return self._read_uint16()
        if code == NULL_COLUMN:
            return 0
        if code == UNSIGNED_INT24_COLUMN:
            return self._read_uint24()
        if code == UNSIGNED_INT64_COLUMN:
            return self._read_uint64()
        return 0

    @cython.cfunc
    @cython.inline(True)
    def read_length_encoded_string(self) -> bytes:
        """(cfunc) Read a 'Length Coded String' from the MySQL packet buffer `<'bytes'>`.

        A 'Length Coded String' consists first of a length coded
        (unsigned, positive) integer represented in 1-9 bytes followed by
        that many bytes of binary data.  (For example "cat" would be "3cat".)
        """
        length: cython.uint = self._read_uint8()
        if length < UNSIGNED_CHAR_COLUMN:
            return self.read(length)
        if length == UNSIGNED_SHORT_COLUMN:
            return self.read(self._read_uint16())
        if length == NULL_COLUMN:
            return None
        if length == UNSIGNED_INT24_COLUMN:
            return self.read(self._read_uint24())
        if length == UNSIGNED_INT64_COLUMN:
            return self.read(self._read_uint64())
        return None

    @cython.cfunc
    @cython.inline(True)
    def _read_uint8(self) -> cython.uint:
        """(cfunc) Read 8-bit binary and advance the cursor `<'int'>`."""
        pos: cython.ulonglong = self._pos
        self._pos = pos + 1
        return unpack_uint8(self._data_c, pos)  # type: ignore

    @cython.cfunc
    @cython.inline(True)
    def _read_uint16(self) -> cython.uint:
        """(cfunc) Read 16-bit binary and advance the cursor `<'int'>`."""
        pos: cython.ulonglong = self._pos
        self._pos = pos + 2
        return unpack_uint16(self._data_c, pos)  # type: ignore

    @cython.cfunc
    @cython.inline(True)
    def _read_uint24(self) -> cython.uint:
        """(cfunc) Read 24-bit binary and advance the cursor `<'int'>`."""
        pos: cython.ulonglong = self._pos
        self._pos = pos + 3
        return unpack_uint24(self._data_c, pos)  # type: ignore

    @cython.cfunc
    @cython.inline(True)
    def _read_uint32(self) -> cython.ulonglong:
        """(cfunc) Read 32-bit binary and advance the cursor `<'int'>`."""
        pos: cython.ulonglong = self._pos
        self._pos = pos + 4
        return unpack_uint32(self._data_c, pos)  # type: ignore

    @cython.cfunc
    @cython.inline(True)
    def _read_uint64(self) -> cython.ulonglong:
        """(cfunc) Read 64-bit binary and advance the cursor `<'int'>`."""
        pos: cython.ulonglong = self._pos
        self._pos = pos + 8
        return unpack_uint64(self._data_c, pos)  # type: ignore

    # Read Packet -----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def is_ok_packet(self) -> cython.bint:
        """(cfunc) Check if is OKPacket `<'bool'>`."""
        # https://dev.mysql.com/doc/internals/en/packet-OK_Packet.html
        return self._size >= 7 and self._data_c[0] == 0  # type: ignore

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def read_ok_packet(self) -> cython.bint:
        """Read OKPacket data `<'bool'>`.

        Returns `True` if is OKPacket and parsed successfully,
        else `False` if the packet is not OKPacket.
        """
        # Not OKPacket
        if not self.is_ok_packet():
            return False
        # Parse Data
        self._pos += 1  # skip 1 (0)
        self._affected_rows = self.read_length_encoded_integer()
        self._insert_id = self.read_length_encoded_integer()
        self._server_status = self._read_uint16()
        self._warning_count = self._read_uint16()
        self._message = self.read_remains()
        self._has_next = self._server_status & _SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def is_load_local_packet(self) -> cython.bint:
        """(cfunc) Check if is LoadLocalPacket `<'bool'>`."""
        return unpack_uint8(self._data_c, 0) == 0xFB  # type: ignore

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def read_load_local_packet(self) -> cython.bint:
        """Read LoadLocalPacket data `<'bool'>`.

        Returns `True` if is LoadLocalPacket and parsed successfully,
        else `False` if the packet is not LoadLocalPacket.
        """
        # Not LoadLocalPacket
        if not self.is_load_local_packet():
            return False
        # Prase Data
        self._pos += 1  # skip 1 (0)
        self._filename = self.read_remains()
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def is_eof_packet(self) -> cython.bint:
        # http://dev.mysql.com/doc/internals/en/generic-response-packets.html#packet-EOF_Packet
        # Caution: \xFE may be LengthEncodedInteger.
        # If \xFE is LengthEncodedInteger header, 8bytes followed.
        return self._size < 9 and unpack_uint8(self._data_c, 0) == 0xFE  # type: ignore

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def read_eof_packet(self) -> cython.bint:
        """Read EOFPacket data `<'bool'>.

        Returns `True` if is EOFPacket and parsed successfully,
        else `False` if the packet is not EOFPacket.
        """
        # Not EOFPacket
        if not self.is_eof_packet():
            return False
        # Parse Data
        self._pos += 1  # skip 1 (0)
        self._warning_count = self._read_uint16()
        self._server_status = self._read_uint16()
        self._has_next = self._server_status & _SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def is_auth_switch_request(self) -> cython.bint:
        """(cfunc) Check if is 'auth switch request' packet `<'bool'>`."""
        # http://dev.mysql.com/doc/internals/en/connection-phase-packets.html#packet-Protocol::AuthSwitchRequest
        return unpack_uint8(self._data_c, 0) == 0xFE  # type: ignore

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def read_auth_switch_request(self) -> cython.bint:
        """Read 'auth switch request' packet data `<'bool'>.

        Returns `True` if is 'auth switch request' packet
        and parsed successfully, else `False`.
        """
        # Not auth switch request
        if not self.is_auth_switch_request():
            return False
        # Parse
        self._pos += 1  # skip 1 (0)
        # . plugin name
        loc: cython.Py_ssize_t = find_null_term(self._data_c, self._pos)  # type: ignore
        if loc < 0:
            return True
        self._plugin_name = self._data_c[self._pos : loc]
        self._pos = loc + 1
        # . salt
        loc = find_null_term(self._data_c, self._pos)  # type: ignore
        if loc < 0:
            return True
        self._salt = self._data_c[self._pos : loc]
        self._pos = loc + 1
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def is_extra_auth_data(self) -> cython.bint:
        """(cfunc) Check if is the packet has extra auth data `<'bool'>`."""
        # https://dev.mysql.com/doc/internals/en/successful-authentication.html
        return self._data_c[0] == 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def is_resultset_packet(self) -> cython.bint:
        """(cfunc) Check if is 'ResultPacket' `<'bool'>`."""
        return 1 <= unpack_uint8(self._data_c, 0) <= 250  # type: ignore

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def is_error_packet(self) -> cython.bint:
        """(cfunc) Check if is 'ErrorPacket' `<'bool'>`."""
        return unpack_uint8(self._data_c, 0) == 0xFF  # type: ignore

    # Curosr ----------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def advance(self, length: cython.ulonglong) -> cython.bint:
        """(cfunc) Advance the MySQL packet cursor by the given 'length' `<'bool'>`."""
        pos: cython.ulonglong = self._pos + length
        if pos > self._size:
            raise errors.MysqlPacketCursorError(
                "'<%s'>\nCan't advance packet cursor position by: %d\n"
                "Current Position=%d | Data Length=%d"
                % (self.__class__.__name__, pos, self._pos, self._size)
            )
        self._pos = pos
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def rewind(self, position: cython.ulonglong) -> cython.bint:
        """(cfunc) Set the MySQL packet cursor to 'position' `<'bool'>`."""
        if position > self._size:
            raise errors.MysqlPacketCursorError(
                "'<%s'>\nCan't set packet cursor position to: %s\n"
                "Current Position=%d | Data Length=%d"
                % (self.__class__.__name__, position, self._pos, self._size)
            )
        self._pos = position
        return True

    # Error -----------------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def check_error(self) -> cython.bint:
        """(cfunc) Check & raise packet error."""
        if self.is_error_packet():
            self.raise_for_error()
        return True

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def raise_for_error(self) -> cython.bint:
        """Raise packet error."""
        errors.raise_mysql_exception(self._data_c, self._size)


@cython.cclass
class FieldDescriptorPacket(MysqlPacket):
    """A MysqlPacket that represents a specific column's metadata in the result.

    Parsing is done automatically and the results are exported via c-attributes
    on the class such as: db, table_name, name, length, type_code.
    """

    # Packet Data
    _catalog: bytes
    _db: str
    _table: str
    _table_org: str
    _column: str
    _column_org: str
    _charsetnr: cython.uint
    _length: cython.ulonglong
    _type_code: cython.uint
    _flags: cython.uint
    _scale: cython.uint
    _is_binary: cython.bint

    def __init__(self, data: bytes, encoding: bytes) -> None:
        super().__init__(data, encoding)
        # Parse
        # fmt: off
        self._catalog = self.read_length_encoded_string()
        self._db = decode_bytes(self.read_length_encoded_string(), self._encoding)
        self._table = decode_bytes(self.read_length_encoded_string(), self._encoding)
        self._table_org = decode_bytes(self.read_length_encoded_string(), self._encoding)
        self._column = decode_bytes(self.read_length_encoded_string(), self._encoding)
        self._column_org = decode_bytes(self.read_length_encoded_string(), self._encoding)
        # fmt: on
        self._pos += 1  # skip 1 (non-null)
        self._charsetnr = self._read_uint16()
        self._length = self._read_uint32()
        self._type_code = self._read_uint8()
        self._flags = self._read_uint16()
        self._scale = self._read_uint8()
        self._pos += 2  # skip 2 (0x00)
        self._is_binary = self._charsetnr == 63

    # Property --------------------------------------------------------------------------------
    @property
    def catalog(self) -> bytes:
        return self._catalog

    @property
    def db(self) -> str:
        return self._db

    @property
    def table(self) -> str:
        return self._table

    @property
    def table_org(self) -> str:
        return self._table_org

    @property
    def column(self) -> str:
        return self._column

    @property
    def column_org(self) -> str:
        return self._column_org

    @property
    def charsetnr(self) -> int:
        return self._charsetnr

    @property
    def length(self) -> int:
        return self._length

    @property
    def type_code(self) -> int:
        return self._type_code

    @property
    def flags(self) -> int:
        return self._flags

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def is_binary(self) -> bool:
        return self._is_binary

    # Read Packet -----------------------------------------------------------------------------
    @cython.ccall
    def description(self) -> tuple[str, int, int, int, int, int, bool]:
        """Provides a 7-item tuple compatible with the Python PEP249 DB Spec.
        >>> (name, type_code, display_length, internal_size, precision, scale, null_ok)
        """
        length: cython.uint = self._get_column_length()
        return (
            self._column,
            self._type_code,
            self._length,  # TODO: display_length; should this be self.length?
            length,  # 'internal_size'
            length,  # 'precision'  # TODO: why!?!?
            self._scale,
            self._flags % 2 == 0,
        )

    @cython.cfunc
    @cython.inline(True)
    def _get_column_length(self) -> cython.ulonglong:
        """(cfunc) Return the column length `<'int'>`."""
        if self._type_code == _FIELD_TYPE.VAR_STRING:
            # PyMySQL.charset MBLENGTH
            # {8: 1, 33: 3, 88: 2, 91: 2}
            if self._charsetnr == 8:
                return self._length
            if self._charsetnr == 33:
                return self._length // 3
            if self._charsetnr == 88:
                return self._length // 2
            if self._charsetnr == 91:
                return self._length // 2
        return self._length

    def __repr__(self) -> str:
        reprs = {
            "catalog": self._catalog,
            "db": self._db,
            "table_name": self._table,
            "org_table": self._table_org,
            "name": self._column,
            "org_name": self._column_org,
            "charsetnr": self._charsetnr,
            "length": self._length,
            "type_code": self._type_code,
            "flags": self._flags,
            "scale": self._scale,
            "is_binary": self._is_binary,
        }
        return "<%s(\n  %s)>" % (
            self.__class__.__name__,
            ",\n  ".join("%s=%r" % (k, v) for k, v in reprs.items()),
        )
