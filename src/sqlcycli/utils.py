# cython: language_level=3

# Cython imports
import cython


### All utils functions are in the utils.pxd file ###
### The following functions are for testing purposes only ###
def test_all(fast: cython.bint = True) -> None:
    _test_uint8(fast)
    _test_int8(fast)
    _test_uint16(fast)
    _test_int16(fast)
    _test_uint24(fast)
    _test_int24(fast)
    _test_uint32(fast)
    _test_int32(fast)
    _test_uint64(fast)
    _test_int64(fast)


def _test_uint8(fast: cython.bint) -> None:
    print("Test Pack/Unpack uint8")
    for val in range(0, 256):
        b = pack_uint8(val)  # type: ignore
        i = unpack_uint8(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_int8(fast: cython.bint) -> None:
    print("Test Pack/Unpack int8")
    for val in range(-128, 128):
        b = pack_int8(val)  # type: ignore
        i = unpack_int8(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_uint16(fast: cython.bint) -> None:
    print("Test Pack/Unpack uint16")
    for val in range(0, 65536):
        b = pack_uint16(val)  # type: ignore
        i = unpack_uint16(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_int16(fast: cython.bint) -> None:
    print("Test Pack/Unpack int16")
    for val in range(-32768, 32768):
        b = pack_int16(val)  # type: ignore
        i = unpack_int16(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")


def _test_uint24(fast: cython.bint) -> None:
    print("Test Pack/Unpack uint24")
    if fast:
        for val in (0, 1, 16777214, 16777215):
            b = pack_uint24(val)  # type: ignore
            i = unpack_uint24(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")
    else:
        for val in range(0, 16777216):
            b = pack_uint24(val)  # type: ignore
            i = unpack_uint24(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")


def _test_int24(fast: cython.bint) -> None:
    print("Test Pack/Unpack int24")
    if fast:
        for val in (-8388608, -8388607, 0, 8388606, 8388607):
            b = pack_int24(val)  # type: ignore
            i = unpack_int24(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")
    else:
        for val in range(-8388608, 8388608):
            b = pack_int24(val)  # type: ignore
            i = unpack_int24(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")


def _test_uint32(fast: cython.bint) -> None:
    print("Test Pack/Unpack uint32")
    if fast:
        for val in (0, 1, 4294967294, 4294967295):
            b = pack_uint32(val)  # type: ignore
            i = unpack_uint32(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")
    else:
        for val in range(0, 4294967296):
            b = pack_uint32(val)  # type: ignore
            i = unpack_uint32(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")


def _test_int32(fast: cython.bint) -> None:
    print("Test Pack/Unpack int32")
    if fast:
        for val in (-2147483648, -2147483647, 0, 2147483646, 2147483647):
            b = pack_int32(val)  # type: ignore
            i = unpack_int32(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")
    else:
        for val in range(-2147483648, 2147483648):
            b = pack_int32(val)  # type: ignore
            i = unpack_int32(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")


def _test_uint64(fast: cython.bint) -> None:
    print("Test Pack/Unpack uint64")
    if fast:
        for val in (0, 1, 18446744073709551614, 18446744073709551615):
            b = pack_uint64(val)  # type: ignore
            i = unpack_uint64(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")
    else:
        for val in range(0, 18446744073709551616):
            b = pack_uint64(val)  # type: ignore
            i = unpack_uint64(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")


def _test_int64(fast: cython.bint) -> None:
    print("Test Pack/Unpack int64")
    if fast:
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
    else:
        for val in range(-9223372036854775808, 9223372036854775808):
            b = pack_int64(val)  # type: ignore
            i = unpack_int64(b, 0)  # type: ignore
            assert val == i, f"val {val} vs i {i}"
            print("- pass: %s" % val, end="\r")
