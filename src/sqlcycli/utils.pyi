import cython
from re import Pattern
from sqlcycli.charset import Charset

# Constants
DEFAULT_USER: str
DEFUALT_CHARSET: str
MAX_CONNECT_TIMEOUT: int
DEFALUT_MAX_ALLOWED_PACKET: int
MAXIMUM_MAX_ALLOWED_PACKET: int
DEFAULT_CONNECT_ATTRS: bytes
MAX_PACKET_LENGTH: int
MAX_STATEMENT_LENGTH: int
SERVER_VERSION_RE: Pattern
RE_INSERT_VALUES: Pattern
"""regex pattern: python-constants"""
INSERT_VALUES_RE: Pattern
"""regex pattern: c-constants"""
NULL_COLUMN: cython.uchar
UNSIGNED_CHAR_COLUMN: cython.uchar
UNSIGNED_SHORT_COLUMN: cython.uchar
UNSIGNED_INT24_COLUMN: cython.uchar
UNSIGNED_INT64_COLUMN: cython.uchar

# Utils: Connection
def gen_connect_attrs(attrs: list[str]) -> bytes: ...
def validate_arg_str(arg: str | None, arg_name: str, default: str) -> str | None: ...
def validate_arg_uint(
    arg: int | None,
    arg_name: str,
    min_val: int,
    max_val: int,
) -> int | None: ...
def validate_arg_bytes(
    arg: bytes | str | None,
    arg_name: str,
    encoding: cython.pchar,
    default: str,
) -> bytes | None: ...
def validate_charset(
    charset: str | None,
    collation: str | None,
    default_charset: str,
) -> Charset: ...
def validate_autocommit(auto: bool | None) -> int: ...
def validate_max_allowed_packet(
    max_allowed_packet: int | str | None,
    default: int,
    max_val: int,
) -> int: ...
def validate_sql_mode(sql_mode: str | None) -> str | None: ...
def validate_ssl(ssl: object | None) -> object | None: ...

# Utils: string
def encode_str(obj: str, encoding: cython.pchar) -> bytes:
    """(cfunc) Encode string to bytes using the 'encoding' with
    'surrogateescape' error handling `<'bytes'>`."""

def decode_bytes(data: bytes, encoding: cython.pchar) -> str:
    """(cfunc) Decode bytes to string using the 'encoding' with
    "surrogateescape" error handling `<'str'>`."""

def decode_bytes_utf8(data: bytes) -> str:
    """(cfunc) Decode bytes to string using 'utf-8' encoding with
    'surrogateescape' error handling `<'str'>`."""

def decode_bytes_ascii(data: bytes) -> str:
    """(cfunc) Decode bytes to string using 'ascii' encoding with
    'surrogateescape' error handling `<'str'>`."""

def decode_bytes_latin1(data: bytes) -> str:
    """(cfunc) Decode bytes to string using 'latin1' encoding with
    'surrogateescape' error handling `<'str'>`."""

def find_null_term(data: cython.pchar, pos: int) -> int:
    """(cfunc) Find the next NULL-terminated starting from 'pos' in the data `<'int'>`."""

# Utils: Pack custom
def pack_I24B(i: int, j: int) -> bytes:
    """(cfunc) Pack 'I[24bit]B' in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<I", i)[:3] + struct.pack("<B", b)
    """

def pack_IB(i: int, j: int) -> bytes:
    """(cfunc) Pack 'IB' in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<IB", i, j)
    """

def pack_IIB23s(i: int, j: int, k: int) -> bytes:
    """(cfunc) Pack 'IIB23s' in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<IIB23s", i, j, k, b"")
    """

def gen_length_encoded_integer(i: int) -> bytes:
    """(cfunc) Generate 'Length Coded Integer' for MySQL `<'bytes'>.

    For more information, please refer to:

    https://dev.mysql.com/doc/internals/en/integer.html#packet-Protocol::LengthEncodedInteger
    """

# Utils: Pack unsigned integers
def pack_uint8(value: int) -> bytes:
    """(cfunc) Pack `UNSIGNED` 8-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<B", value)
    """

def pack_uint16(value: int) -> bytes:
    """(cfunc) Pack `UNSIGNED` 16-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<H", value)
    """

def pack_uint24(value: int) -> bytes:
    """(cfunc) Pack `UNSIGNED` 24-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<I", value)[:3]
    """

def pack_uint32(value: int) -> bytes:
    """(cfunc) Pack `UNSIGNED` 32-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<I", value)
    """

def pack_uint64(value: int) -> bytes:
    """(cfunc) Pack `UNSIGNED` 64-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<Q", value)
    """

# Utils: Unpack unsigned integer
def unpack_uint8(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `UNSIGNED` 8-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

def unpack_uint16(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `UNSIGNED` 16-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

def unpack_uint24(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `UNSIGNED` 24-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

def unpack_uint32(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `UNSIGNED` 32-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

def unpack_uint64(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `UNSIGNED` 64-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

# Utils: Pack signed integer
def pack_int8(value: int) -> bytes:
    """(cfunc) Pack `SIGNED` 8-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<b", value)
    """

def pack_int16(value: int) -> bytes:
    """(cfunc) Pack `SIGNED` 16-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<h", value)
    """

def pack_int24(value: int) -> bytes:
    """(cfunc) Pack `SIGNED` 24-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<i", value)[:3]
    """

def pack_int32(value: int) -> bytes:
    """(cfunc) Pack `SIGNED` 32-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<i", value)
    """

def pack_int64(value: int) -> bytes:
    """(cfunc) Pack `SIGNED` 64-bit integer in little-endian order to `<'bytes'>`.

    Equivalent to:
    >>> struct.pack("<q", value)
    """

# Utils: Unpack signed integer
def unpack_int8(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `SIGNED` 8-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

def unpack_int16(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `SIGNED` 16-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

def unpack_int24(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `SIGNED` 24-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

def unpack_int32(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `SIGNED` 32-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """

def unpack_int64(data: cython.pchar, pos: int) -> int:
    """(cfunc) Unpack (read) `SIGNED` 64-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
