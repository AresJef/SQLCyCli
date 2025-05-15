# cython: language_level=3

# Cython imports
import cython
from cython.cimports.cpython.bytes import PyBytes_Size as bytes_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_ReadChar as str_read  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Substring as str_substr  # type: ignore
from cython.cimports.sqlcycli.transcode import escape  # type: ignore
from cython.cimports.sqlcycli._ssl import is_ssl, is_ssl_ctx  # type: ignore
from cython.cimports.sqlcycli.charset import Charset, _charsets  # type: ignore

# Python imports
import re
from sqlcycli.charset import Charset
from sqlcycli.transcode import escape
from sqlcycli._ssl import is_ssl, is_ssl_ctx
from sqlcycli import errors

# Constants -----------------------------------------------------------------------------------
try:
    import getpass

    DEFAULT_USER: str = getpass.getuser()
    del getpass
except (ImportError, KeyError):
    #: KeyError occurs when there's no entry
    #: in OS database for a current user.
    DEFAULT_USER: str = None
DEFUALT_CHARSET: str = "utf8mb4"
MAX_CONNECT_TIMEOUT: cython.int = 31_536_000  # 1 year
DEFALUT_MAX_ALLOWED_PACKET: cython.int = 16_777_216  # 16MB
MAXIMUM_MAX_ALLOWED_PACKET: cython.int = 1_073_741_824  # 1GB
MAX_PACKET_LENGTH: cython.uint = 2**24 - 1
#: Max statement size which :meth:`executemany` generates.
#: Max size of allowed statement is max_allowed_packet - packet_header_size.
#: Default value of max_allowed_packet is 1048576.
MAX_STATEMENT_LENGTH: cython.uint = 1024000
#: Regular expression for :meth:`Cursor.executemany`.
#: executemany only supports simple bulk insert.
#: You can use it to load large dataset.
RE_INSERT_VALUES: re.Pattern = re.compile(
    r"\s*((?:INSERT|REPLACE)\b.+\bVALUES?\s*)"  # prefix: INSERT INTO ... VALUES
    + r"(\(\s*(?:%s|%\(.+\)s)\s*(?:,\s*(?:%s|%\(.+\)s)\s*)*\))"  # placeholders: (%s, %s, ...)
    + r"(\s*(?:AS\b\s.+)?\s*(?:ON DUPLICATE\b.+)?);?\s*\Z",  # suffix: AS ... ON DUPLICATE ...
    re.IGNORECASE | re.DOTALL,
)
INSERT_VALUES_RE: re.Pattern = RE_INSERT_VALUES
#: Regular expression for server version.
SERVER_VERSION_RE: re.Pattern = re.compile(r".*?(\d+)\.(\d+)\.(\d+).*?")

# The following values are for the first byte
# value of MySQL length encoded integer.
NULL_COLUMN: cython.uchar = 251
UNSIGNED_CHAR_COLUMN: cython.uchar = 251
UNSIGNED_SHORT_COLUMN: cython.uchar = 252
UNSIGNED_INT24_COLUMN: cython.uchar = 253
UNSIGNED_INT64_COLUMN: cython.uchar = 254


# Utils: Connection ---------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_connect_attrs(attrs: list[str]) -> bytes:
    """(cfunc) Generate connect attributes for Connection `<'bytes'>`."""
    arr: list = []
    for i in attrs:
        attr = encode_str(i, "utf-8")  # type: ignore
        arr.append(gen_length_encoded_integer(bytes_len(attr)))  # type: ignore
        arr.append(attr)
    return b"".join(arr)


DEFAULT_CONNECT_ATTRS: bytes = gen_connect_attrs(
    ["_client_name", "sqlcycli", "_client_version", "0.0.0", "_pid"]
)


@cython.cfunc
@cython.inline(True)
def validate_arg_str(arg: object, arg_name: str, default: str) -> str:
    """(cfunc) Validate if a string argument is valid `<'str'>`."""
    if arg is None:
        return default
    if isinstance(arg, str):
        return arg if str_len(arg) > 0 else default
    raise errors.InvalidConnectionArgsError(
        "Invalid '%s' argument: %r.\n"
        "Expects <'str'> instead of %s." % (arg_name, arg, type(arg))
    )


@cython.cfunc
@cython.inline(True)
def validate_arg_uint(
    arg: object,
    arg_name: str,
    min_val: cython.uint,
    max_val: cython.uint,
) -> object:
    """(cfunc) Validate if an unsigned integer argument is valid `<'int'>`."""
    if arg is None:
        return None
    try:
        val: cython.longlong = int(arg)
    except Exception as err:
        raise errors.InvalidConnectionArgsError(
            "Invalid '%s' argument: %r.\n"
            "Expects <'int'> instead of %s." % (arg_name, arg, type(arg))
        ) from err
    if not min_val <= val <= max_val:
        raise errors.InvalidConnectionArgsError(
            "Invalid '%s' argument: %r.\n"
            "Expects an integer between %d and %d." % (arg_name, val, min_val, max_val)
        )
    return val


@cython.cfunc
@cython.inline(True)
def validate_arg_bytes(
    arg: object,
    arg_name: str,
    encoding: cython.pchar,
    default: str,
) -> bytes:
    """(cfunc) Validate if a bytes argument is valid `<'bytes'>`."""
    if arg is None:
        if default is not None:
            return encode_str(default, encoding)  # type: ignore
        return None
    if isinstance(arg, str):
        if str_len(arg) > 0:
            return encode_str(arg, encoding)  # type: ignore
        if default is not None:
            return encode_str(default, encoding)  # type: ignore
        return None
    if isinstance(arg, bytes):
        if bytes_len(arg) > 0:
            return arg
        if default is not None:
            return encode_str(default, encoding)  # type: ignore
        return None
    raise errors.InvalidConnectionArgsError(
        "Invalid '%s' argument: %r.\n"
        "Expects <'str/bytes'> instead of %s." % (arg_name, arg, type(arg))
    )


@cython.cfunc
@cython.inline(True)
def validate_charset(
    charset: object,
    collation: object,
    default_charset: str,
) -> Charset:
    """(cfunc) Validate if 'charset' & 'collation' arguments are valid `<'Charset'>`."""
    ch: str = validate_arg_str(charset, "charset", default_charset)
    cl: str = validate_arg_str(collation, "collation", None)
    if cl is None:
        return _charsets.by_name(ch)
    else:
        return _charsets.by_name_n_collation(ch, cl)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def validate_autocommit(auto: object) -> cython.int:
    """(cfunc) Validate if the 'autocommit' argument is valid `<'int'>`.

    Returns -1 if use server default, else 0/1
    as disable/enable autocommit mode.
    """
    if auto is None:
        return -1
    else:
        return 1 if bool(auto) else 0


@cython.cfunc
@cython.inline(True)
def validate_max_allowed_packet(
    max_allowed_packet: object,
    default: cython.int,
    max_val: cython.int,
) -> cython.int:
    """(cfunc) Validate if the 'max_allowed_packet' argument is valid `<'int'>`."""
    # Argument is None
    if max_allowed_packet is None:
        return default
    # Argument is integer
    if isinstance(max_allowed_packet, int):
        try:
            value: cython.longlong = max_allowed_packet
        except Exception as err:
            raise errors.InvalidConnectionArgsError(
                "Invalid 'max_allowed_packet' argument: %r.\n"
                "Expects <'str/int'> instead of %s."
                % (max_allowed_packet, type(max_allowed_packet))
            ) from err
    # Arugment is string
    elif isinstance(max_allowed_packet, str):
        size: cython.Py_ssize_t = str_len(max_allowed_packet)
        try:
            if size < 1:
                raise ValueError("not enough charactors.")
            ch: cython.Py_UCS4 = str_read(max_allowed_packet, size - 1)
            # . skip M/K/G[B] suffix
            if ch in ("B", "b"):
                size -= 1
                if size < 1:
                    raise ValueError("not enough charactors.")
                ch = str_read(max_allowed_packet, size - 1)
            # . K/KiB suffix
            if ch in ("K", "k"):
                size -= 1
                mult: cython.int = 1_024
            # . M/MiB suffix
            elif ch in ("M", "m"):
                size -= 1
                mult: cython.int = 1_048_576
            # . G/GiB suffix
            elif ch in ("G", "g"):
                size -= 1
                mult: cython.int = 1_073_741_824
            # . no unit suffix
            else:
                mult: cython.int = 1
            # . parse integer
            if size < 1:
                raise ValueError("not enough charactors.")
            value: cython.longlong = int(str_substr(max_allowed_packet, 0, size))
        except Exception as err:
            raise errors.InvalidConnectionArgsError(
                "Invalid 'max_allowed_packet' argument: %r %s.\nError: %s"
                % (max_allowed_packet, type(max_allowed_packet), err)
            ) from err
        value *= mult
    # Invalid
    else:
        raise errors.InvalidConnectionArgsError(
            "Invalid 'max_allowed_packet' argument: %r.\n"
            "Expects <'str/int'> instead of %s."
            % (max_allowed_packet, type(max_allowed_packet))
        )
    # Validate value
    if not 0 < value <= max_val:
        raise errors.InvalidConnectionArgsError(
            "Invalid 'max_allowed_packet' argument: %r %s.\n"
            "Value in bytes must between 1 and %s."
            % (max_allowed_packet, type(max_allowed_packet), max_val)
        )
    return value


@cython.cfunc
@cython.inline(True)
def validate_sql_mode(sql_mode: object) -> str:
    """(cfunc) Validate if the 'sql_mode' argument is valid `<'str'>`."""
    if sql_mode is None:
        return None
    if isinstance(sql_mode, str):
        if str_len(sql_mode) > 0:
            return escape(sql_mode, False, False)
        return None
    raise errors.InvalidConnectionArgsError(
        "Invalid 'sql_mode' argument: %r.\n"
        "Expects <'str'> instead of %s." % (sql_mode, type(sql_mode))
    )


@cython.cfunc
@cython.inline(True)
def validate_ssl(ssl: object) -> object:
    """(cfunc) Validate if the 'ssl' argument is valid `<'ssl.SSLContext'>`."""
    if ssl is None:
        return None
    if is_ssl(ssl):
        return ssl.context if ssl else None
    if is_ssl_ctx(ssl):
        return ssl
    raise errors.InvalidConnectionArgsError(
        "Invalid 'ssl' argument: %r.\n"
        "Expects <'SSL/SSLContext'> instead of %s." % (ssl, type(ssl))
    )


########## The REST utility functions are in the utils.pxd file ##########
########## The following functions are for testing purpose only ##########
def _test_utils() -> None:
    _test_encode_decode_utf8()
    _test_encode_decode_ascii()
    _test_encode_decode_latin1()
    _test_find_null_term()
    _test_custom_pack()
    _test_pack_unpack_int8()
    _test_pack_unpack_uint8()
    _test_pack_unpack_int16()
    _test_pack_unpack_uint16()
    _test_pack_unpack_int24()
    _test_pack_unpack_uint24()
    _test_pack_unpack_int32()
    _test_pack_unpack_uint32()
    _test_pack_unpack_int64()
    _test_pack_unpack_uint64()
    _test_validate_max_allowed_packet()


def _test_encode_decode_utf8() -> None:
    val = "中国\n한국어\nにほんご\nEspañol"
    # encode
    n = val.encode("utf-8")
    x = encode_str(val, "utf-8")  # type: ignore
    assert n == x, f"{n} | {x}"
    # decode
    i = n.decode("utf-8")
    j = decode_bytes(n, "utf-8")  # type: ignore
    k = decode_bytes_utf8(n)  # type: ignore
    assert i == j == k == val, f"{i} | {j} | {k} | {val}"
    print("Pass Encode/Decode UTF-8".ljust(80))


def _test_encode_decode_ascii() -> None:
    val = "hello\nworld"
    # encode
    n = val.encode("ascii")
    x = encode_str(val, "ascii")  # type: ignore
    assert n == x, f"{n} | {x}"
    # decode
    i = n.decode("ascii")
    j = decode_bytes(n, "ascii")  # type: ignore
    k = decode_bytes_ascii(n)  # type: ignore
    assert i == j == k == val, f"{i} | {j} | {k} | {val}"
    print("Pass Encode/Decode ASCII".ljust(80))


def _test_encode_decode_latin1() -> None:
    val = "hello\nworld"
    # encode
    n = val.encode("latin1")
    x = encode_str(val, "latin1")  # type: ignore
    assert n == x, f"{n} | {x}"
    # decode
    i = n.decode("latin1")
    j = decode_bytes(n, "latin1")  # type: ignore
    k = decode_bytes_latin1(n)  # type: ignore
    assert i == j == k == val, f"{i} | {j} | {k} | {val}"
    print("Pass Encode/Decode Latin1".ljust(80))


def _test_find_null_term() -> None:
    chs: cython.pchar = b"hello\0world"
    loc = find_null_term(chs, 0)  # type: ignore
    assert loc == 5, f"{chs}: first null term at 5 instead of {loc}"
    loc = find_null_term(chs, 6)  # type: ignore
    assert loc == 11, f"{chs}: second null term at 11 instead of {loc}"
    print("Pass Find Null Term".ljust(80))


def _test_custom_pack() -> None:
    import struct

    # Test pack_I24B
    i24 = 16777215
    b = 250
    v = pack_I24B(i24, b)  # type: ignore
    n1 = struct.pack("<I", i24)[0:3] + struct.pack("<B", b)
    n2 = struct.pack("<I", i24)[0:3] + bytes([b])
    assert v == n1 == n2, f"{v} | {n1} | {n2}"
    print("Pass Pack I24B".ljust(80))

    # Test pack_IB
    i = 16777215
    v = pack_IB(i, b)  # type: ignore
    n = struct.pack("<IB", i, b)
    assert v == n, f"{v} | {n}"
    print("Pass Pack IB".ljust(80))

    # Test pack_IIB23s
    v = pack_IIB23s(i, i, b)  # type: ignore
    n = struct.pack("<IIB23s", i, i, b, b"")
    assert v == n, f"{v} | {n}"
    print("Pass Pack IIB23s".ljust(80))

    # Test gen_length_encoded_integer
    num = 250
    v = gen_length_encoded_integer(num)  # type: ignore
    n = pack_uint8(num)  # type: ignore
    assert v == n, f"{v} | {n} - num: {num}"

    for num in (251, 65_535):
        v = gen_length_encoded_integer(num)  # type: ignore
        n = pack_uint8(UNSIGNED_SHORT_COLUMN) + pack_uint16(num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"

    for num in (65_536, 16_777_215):
        v = gen_length_encoded_integer(num)  # type: ignore
        n = pack_uint8(UNSIGNED_INT24_COLUMN) + pack_uint24(num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"

    for num in (16_777_216, 4_294_967_295):
        v = gen_length_encoded_integer(num)  # type: ignore
        n = pack_uint8(UNSIGNED_INT64_COLUMN) + pack_uint64(num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"
    print("Pass Gen Length Encoded Integer".ljust(80))

    del struct


def _test_pack_unpack_int8() -> None:
    import struct

    for val in range(-128, 128):
        s = struct.pack("<b", val)
        b = pack_int8(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_int8(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack int8".ljust(80))

    del struct


def _test_pack_unpack_uint8() -> None:
    import struct

    for val in range(0, 256):
        s = struct.pack("<B", val)
        b = pack_uint8(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_uint8(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack uint8".ljust(80))

    del struct


def _test_pack_unpack_int16() -> None:
    import struct

    for val in range(-32768, 32768):
        s = struct.pack("<h", val)
        b = pack_int16(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_int16(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack int16".ljust(80))

    del struct


def _test_pack_unpack_uint16() -> None:
    import struct

    for val in range(0, 65536):
        s = struct.pack("<H", val)
        b = pack_uint16(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_uint16(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack uint16".ljust(80))

    del struct


def _test_pack_unpack_int24() -> None:
    import struct

    for val in (-8388608, -8388607, 0, 8388606, 8388607):
        s = struct.pack("<i", val)[:3]
        b = pack_int24(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_int24(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack int24".ljust(80))

    del struct


def _test_pack_unpack_uint24() -> None:
    import struct

    for val in (0, 1, 16777213, 16777214, 16777215):
        s = struct.pack("<I", val)[:3]
        b = pack_uint24(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_uint24(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack uint24".ljust(80))

    del struct


def _test_pack_unpack_int32() -> None:
    import struct

    for val in (-2147483648, -2147483647, 0, 2147483646, 2147483647):
        s = struct.pack("<i", val)
        b = pack_int32(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_int32(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack int32".ljust(80))

    del struct


def _test_pack_unpack_uint32() -> None:
    import struct

    for val in (0, 1, 4294967293, 4294967294, 4294967295):
        s = struct.pack("<I", val)
        b = pack_uint32(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_uint32(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack uint32".ljust(80))

    del struct


def _test_pack_unpack_int64() -> None:
    import struct

    for val in (
        -9223372036854775808,
        -9223372036854775807,
        0,
        9223372036854775806,
        9223372036854775807,
    ):
        s = struct.pack("<q", val)
        b = pack_int64(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_int64(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack int64".ljust(80))

    del struct


def _test_pack_unpack_uint64() -> None:
    import struct

    for val in (
        0,
        1,
        18446744073709551613,
        18446744073709551614,
        18446744073709551615,
    ):
        s = struct.pack("<Q", val)
        b = pack_uint64(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        i = unpack_uint64(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack/Unpack uint64".ljust(80))

    del struct


def _test_validate_max_allowed_packet() -> None:
    default = 16_777_216
    max_val = 1_073_741_824

    # Test None
    v = validate_max_allowed_packet(None, default, max_val)
    assert v == default, f"{v} | {default}"

    # Test integer
    v = validate_max_allowed_packet(1, default, max_val)
    assert v == 1, f"{v} | 1"
    try:
        v = validate_max_allowed_packet(0, default, max_val)
    except ValueError:
        pass
    try:
        v = validate_max_allowed_packet(max_val + 1, default, max_val)
    except ValueError:
        pass

    # Test string
    for val in ("1", "1b", "1B"):  # 1 byte
        v = validate_max_allowed_packet(val, default, max_val)
        assert v == 1, f"{v} | 1"
    for val in ("1k", "1K", "1kb", "1KB"):
        v = validate_max_allowed_packet(val, default, max_val)
        assert v == 1_024, f"{v} | 1_024"
    for val in ("1m", "1M", "1mb", "1MB"):
        v = validate_max_allowed_packet(val, default, max_val)
        assert v == 1_048_576, f"{v} | 1_048_576"
    for val in ("1g", "1G", "1gb", "1GB"):
        v = validate_max_allowed_packet(val, default, max_val)
        assert v == 1_073_741_824, f"{v} | 1_073_741_824"
    # fmt: off
    vals = ["0" + sfix for sfix in ("b", "B")]
    vals += [str(max_val + 1) + sfix for sfix in ("b", "B")]
    vals += ["0" + sfix for sfix in ("k", "K", "kb", "KB")]
    vals += [str(int(max_val / 1024) + 1) + sfix for sfix in ("k", "K", "kb", "KB")]
    vals += ["0" + sfix for sfix in ("m", "M", "mb", "MB")]
    vals += [str(int(max_val / 1_048_576) + 1) + sfix for sfix in ("m", "M", "mb", "MB")]
    vals += ["0" + sfix for sfix in ("g", "G", "gb", "GB")]
    vals += [str(int(max_val / 1_073_741_824) + 1) + sfix for sfix in ("g", "G", "gb", "GB")]
    # fmt: on
    for val in vals:
        try:
            v = validate_max_allowed_packet(val, default, max_val)
        except ValueError:
            pass
        else:
            raise AssertionError("Max allowed packet validation failed.")
    print("Pass Validate Max Allowed Packet".ljust(80))
