# cython: language_level=3

# Cython imports
import cython
from cython.cimports.cpython.bytes import PyBytes_GET_SIZE as bytes_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_ReadChar as read_char  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Substring as str_substr  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_AsUTF8String as encode_str_utf8  # type: ignore
from cython.cimports.sqlcycli.transcode import escape  # type: ignore
from cython.cimports.sqlcycli._ssl import is_ssl, is_ssl_ctx  # type: ignore
from cython.cimports.sqlcycli.charset import Charset, _charsets  # type: ignore

# Python imports
from sqlcycli.charset import Charset
from sqlcycli.transcode import escape
from sqlcycli._ssl import is_ssl, is_ssl_ctx
from sqlcycli import errors


# Utils: Connection ---------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def gen_connect_attrs(attrs: list[str]) -> bytes:
    """(cfunc) Generate connect attributes for Connection `<'bytes'>`."""
    arr: list = []
    for i in attrs:
        attr = encode_str_utf8(i)
        arr.append(gen_length_encoded_integer(bytes_len(attr)))  # type: ignore
        arr.append(attr)
    return b"".join(arr)


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
    """(cfunc) Validate if an unsigned integer argument is valid `<'unsigned int'>`."""
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
        length: cython.Py_ssize_t = str_len(max_allowed_packet)
        ch: cython.Py_UCS4 = read_char(max_allowed_packet, length - 1)
        # . skip M/K/G[B] suffix
        if ch in ("B", "b"):
            length -= 1
            ch = read_char(max_allowed_packet, length - 1)
        # . K/KiB suffix
        if ch in ("K", "k"):
            length -= 1
            mult: cython.int = 1_024
        # . M/MiB suffix
        elif ch in ("M", "m"):
            length -= 1
            mult: cython.int = 1_048_576
        # . G/GiB suffix
        elif ch in ("G", "g"):
            length -= 1
            mult: cython.int = 1_073_741_824
        # . no unit suffix
        else:
            mult: cython.int = 1
        # . parse integer
        try:
            if length < 1:
                raise ValueError("Not enough charactors.")
            value: cython.longlong = int(str_substr(max_allowed_packet, 0, length))
        except Exception as err:
            raise errors.InvalidConnectionArgsError(
                "Invalid 'max_allowed_packet' argument: %r %s."
                % (max_allowed_packet, type(max_allowed_packet))
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
            "Invalid 'max_allowed_packet' argument: %r.\n"
            "Value in bits must between 1 and %s." % (max_allowed_packet, max_val)
        )
    return value


@cython.cfunc
@cython.inline(True)
def validate_sql_mode(sql_mode: object, encoding: cython.pchar) -> str:
    """(cfunc) Validate if the 'sql_mode' argument is valid `<'str'>`."""
    if sql_mode is None:
        return None
    if isinstance(sql_mode, str):
        if str_len(sql_mode) > 0:
            return escape(sql_mode, encoding, False, False)
        return None
    raise errors.InvalidConnectionArgsError(
        "Invalid 'sql_mode' argument: %r.\n"
        "Expects <'str'> instead of %s." % (sql_mode, type(sql_mode))
    )


@cython.cfunc
@cython.inline(True)
def validate_cursor(cursor: object, cursor_class: object) -> object:
    """(cfunc) Validate if the 'cursor' argument is valid `<'object'>`."""
    if cursor is None:
        return cursor_class
    try:
        if not issubclass(cursor, cursor_class):
            raise TypeError("Invalid cursor class.")
    except Exception as err:
        raise errors.InvalidConnectionArgsError(
            "Invalid 'cursor' argument: %r.\n"
            "Expects type [class] of %r." % (cursor, cursor_class)
        ) from err
    return cursor


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


### The REST util functions are in the utils.pxd file ###
### The following functions are for testing purposes only ###
def test_all() -> None:
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


def _test_pack_unpack_uint8() -> None:
    print("Test Pack/Unpack uint8")
    for val in range(0, 256):
        b = pack_uint8(val)  # type: ignore
        i = unpack_uint8(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_int8() -> None:
    print("Test Pack/Unpack int8")
    for val in range(-128, 128):
        b = pack_int8(val)  # type: ignore
        i = unpack_int8(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_uint16() -> None:
    print("Test Pack/Unpack uint16")
    for val in range(0, 65536):
        b = pack_uint16(val)  # type: ignore
        i = unpack_uint16(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_int16() -> None:
    print("Test Pack/Unpack int16")
    for val in range(-32768, 32768):
        b = pack_int16(val)  # type: ignore
        i = unpack_int16(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_int24() -> None:
    print("Test Pack/Unpack int24")
    for val in (-8388608, -8388607, 0, 8388606, 8388607):
        b = pack_int24(val)  # type: ignore
        i = unpack_int24(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_uint24() -> None:
    print("Test Pack/Unpack uint24")
    for val in (0, 1, 16777213, 16777214, 16777215):
        b = pack_uint24(val)  # type: ignore
        i = unpack_uint24(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_int32() -> None:
    print("Test Pack/Unpack int32")
    for val in (-2147483648, -2147483647, 0, 2147483646, 2147483647):
        b = pack_int32(val)  # type: ignore
        i = unpack_int32(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_uint32() -> None:
    print("Test Pack/Unpack uint32")
    for val in (0, 1, 4294967293, 4294967294, 4294967295):
        b = pack_uint32(val)  # type: ignore
        i = unpack_uint32(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_int64() -> None:
    print("Test Pack/Unpack int64")
    for val in (
        -9223372036854775808,
        -9223372036854775807,
        0,
        9223372036854775806,
        9223372036854775807,
    ):
        b = pack_int64(val)  # type: ignore
        i = unpack_int64(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_pack_unpack_uint64() -> None:
    print("Test Pack/Unpack uint64")
    for val in (
        0,
        1,
        18446744073709551613,
        18446744073709551614,
        18446744073709551615,
    ):
        b = pack_uint64(val)  # type: ignore
        i = unpack_uint64(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
