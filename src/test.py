import numpy as np, pandas as pd, orjson
import datetime, time, unittest, decimal, re, os
from sqlcycli import errors
from sqlcycli.constants import ER, CLIENT
from sqlcycli.connection import (
    Connection,
    Cursor,
    DictCursor,
    DfCursor,
    SSCursor,
    SSDictCursor,
    SSDfCursor,
)


class TestCase(unittest.TestCase):
    name: str = "Case"
    host: str = "localhost"
    user: str = "root"
    password: str = "Password_123456"
    unix_socket: str = None
    db: str = "test"
    tb: str = "test_table"

    @property
    def table(self) -> str:
        if self.db is not None and self.tb is not None:
            return f"{self.db}.{self.tb}"
        return None

    def test_all(self) -> None:
        pass

    # utils
    def get_conn(self, **kwargs) -> Connection:
        conn = Connection(
            host=self.host,
            user=self.user,
            password=self.password,
            unix_socket=self.unix_socket,
            local_infile=True,
            **kwargs,
        )
        conn.connect()
        return conn

    def setup(self, table: str = None, **kwargs) -> Connection:
        conn = self.get_conn(**kwargs)
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {self.db};")
            cur.execute(f"DROP TABLE IF EXISTS {self.db}.{tb}")
        return conn

    def drop(self, conn: Connection, table: str = None) -> None:
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute(f"drop table if exists {self.db}.{tb}")

    def delete(self, conn: Connection, table: str = None) -> None:
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute(f"delete from {self.db}.{tb}")

    def log_start(self, msg: str) -> None:
        msg = "START TEST '%s': %s" % (self.name, msg)
        print(msg.ljust(60), end="\r")
        self._start_time = time.perf_counter()

    def log_ended(self, msg: str, skip: bool = False) -> None:
        self._ended_time = time.perf_counter()
        msg = "%s TEST '%s': %s" % ("SKIP" if skip else "PASS", self.name, msg)
        if self._start_time is not None:
            msg += " (%.6fs)" % (self._ended_time - self._start_time)
        print(msg.ljust(60))


class TestCharset(TestCase):
    name: str = "Charset"

    def test_all(self) -> None:
        self.validate_charsets()
        self.test_utf8()

    def validate_charsets(self) -> None:
        from sqlcycli import charset
        from pymysql import charset as pycharset

        test = "VALIDATE CHARSETS"
        self.log_start(test)

        chs = charset.all_charsets()
        for ch in chs:
            # by_id
            pyCh = pycharset.charset_by_id(ch.id)
            self.assertEqual(
                (ch.name, ch.collation, ch.encoding),
                (pyCh.name, pyCh.collation, pyCh.encoding.encode("ascii")),
            )
            # by_name
            if ch.is_default:
                pyCh = pycharset.charset_by_name(ch.name)
                self.assertEqual(
                    (ch.name, ch.collation, ch.encoding),
                    (pyCh.name, pyCh.collation, pyCh.encoding.encode("ascii")),
                )
        self.log_ended(test)

    def test_utf8(self) -> None:
        from sqlcycli import charset

        test = "UTF8"
        self.log_start(test)

        # utf8mb3
        ch = charset.by_name("utf8mb3")
        self.assertEqual(
            (ch.name, ch.collation, ch.encoding),
            ("utf8mb3", "utf8mb3_general_ci", b"utf8"),
        )

        # utf8mb4
        # MySQL 8.0 changed the default collation for utf8mb4.
        # But we use old default for compatibility.
        ch = charset.by_name("utf8mb4")
        self.assertEqual(
            (ch.name, ch.collation, ch.encoding),
            ("utf8mb4", "utf8mb4_general_ci", b"utf8"),
        )

        # utf8 is alias of utf8mb4 since MySQL 8.0, and PyMySQL v1.1.
        utf8 = charset.by_name("utf8")
        self.assertEqual(ch, utf8)

        self.log_ended(test)


class TestTranscode(TestCase):
    name: str = "Transcode"
    data: dict[str, object] = {
        "bool_0": False,
        "bool_1": True,
        "int_p": 10,
        "int_0": 0,
        "int_n": -10,
        "float_p": 1.1,
        "float_0": 0.0,
        "float_n": -1.1,
        "str": "中国\n\r\032\"'s",
        "none": None,
        "dt_s": datetime.datetime(1970, 1, 2, 3, 4, 5),
        "dt_us": datetime.datetime(1970, 1, 2, 3, 4, 5, 6),
        "dt_tz": datetime.datetime(1970, 1, 2, 3, 4, 5, 6, tzinfo=datetime.UTC),
        "date": datetime.date(1970, 1, 1),
        "time_s": datetime.time(1, 2, 3),
        "time_us": datetime.time(1, 2, 3, 4),
        "td": -datetime.timedelta(1, 1, 1),
        "stime": time.struct_time((1970, 1, 1, 0, 0, 0, 0, 1, 0)),
        "bytes": b"apple\n\r\032\"'s",
        "dec_p": decimal.Decimal("1.1"),
        "dec_0": decimal.Decimal("0.0"),
        "dec_n": decimal.Decimal("-1.1"),
        "dict": {"a": 1, "b": 2, "c": 3},
        "list": [1, 2, 3],
        "tuple": ("1", "2", "3"),
        "set": {1, 2, 3},
        "fset": frozenset({"1", "2", "3"}),
    }

    def test_all(self) -> None:
        self.test_escape_encode_bool()
        self.test_escape_encode_int()
        self.test_escape_encode_float()
        self.test_escape_encode_str()
        self.test_escape_encode_none()
        self.test_escape_encode_datetime()
        self.test_escape_encode_date()
        self.test_escape_encode_time()
        self.test_escape_encode_timedelta()
        self.test_escape_encode_bytes()
        self.test_escape_encode_decimal()
        self.test_escape_encode_dict()
        self.test_escape_encode_sequence()
        self.test_escape_encode_ndarray_series()
        self.test_escape_encode_dataframe()
        self.test_decode()

    def test_escape_encode_bool(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE BOOL"
        self.log_start(test)

        self.assertEqual(escape(True, True, True), "1")
        self.assertEqual(escape(False, True, True), "0")
        self.assertEqual(escape(True, False, True), "1")
        self.assertEqual(escape(False, False, True), "0")
        self.assertEqual(escape(True, False, False), "1")
        self.assertEqual(escape(False, False, False), "0")
        self.log_ended(test)

    def test_escape_encode_int(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE INT"
        self.log_start(test)

        value = 1
        test_values = [value] + [
            d(value) for d in (np.int8, np.int16, np.int32, np.int64)
        ]
        for v in test_values:
            self.assertEqual(escape(v, True, True), str(value))
            self.assertEqual(escape(v, False, True), str(value))
            self.assertEqual(escape(v, False, False), str(value))

        value = -1
        test_values = [value] + [
            d(value) for d in (np.int8, np.int16, np.int32, np.int64)
        ]
        for v in test_values:
            self.assertEqual(escape(v, True, True), str(value))
            self.assertEqual(escape(v, False, True), str(value))
            self.assertEqual(escape(v, False, False), str(value))

        value = 0
        test_values = [value] + [
            d(value) for d in (np.uint8, np.uint16, np.uint32, np.uint64)
        ]
        for v in test_values:
            self.assertEqual(escape(v, True, True), str(value))
            self.assertEqual(escape(v, False, True), str(value))
            self.assertEqual(escape(v, False, False), str(value))

        self.log_ended(test)

    def test_escape_encode_float(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE FLOAT"
        self.log_start(test)

        value = 1.0
        test_values = [value] + [d(value) for d in (np.float16, np.float32, np.float64)]
        for v in test_values:
            self.assertEqual(escape(v, True, True), str(value))
            self.assertEqual(escape(v, False, True), str(value))
            self.assertEqual(escape(v, False, False), str(value))

        value = -1.0
        test_values = [value] + [d(value) for d in (np.float16, np.float32, np.float64)]
        for v in test_values:
            self.assertEqual(escape(v, True, True), str(value))
            self.assertEqual(escape(v, False, True), str(value))
            self.assertEqual(escape(v, False, False), str(value))

        self.log_ended(test)

    def test_escape_encode_str(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE STR"
        self.log_start(test)

        self.assertEqual(escape("foo\nbar", True, True), "'foo\\nbar'")
        self.assertEqual(escape("foo\nbar", False, True), "'foo\\nbar'")
        self.assertEqual(escape("foo\nbar", False, False), "'foo\\nbar'")

        self.log_ended(test)

    def test_escape_encode_none(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE NONE"
        self.log_start(test)

        self.assertEqual(escape(None, True, True), "NULL")
        self.assertEqual(escape(None, False, True), "NULL")
        self.assertEqual(escape(None, False, False), "NULL")

        self.log_ended(test)

    def test_escape_encode_datetime(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE DATETIME"
        self.log_start(test)

        value = datetime.datetime(2021, 1, 1, 12, 0, 0)
        self.assertEqual(escape(value, True, True), "'2021-01-01 12:00:00'")
        self.assertEqual(escape(value, False, True), "'2021-01-01 12:00:00'")
        self.assertEqual(escape(value, False, False), "'2021-01-01 12:00:00'")

        value = np.datetime64(value)
        self.assertEqual(escape(value, True, True), "'2021-01-01 12:00:00'")
        self.assertEqual(escape(value, False, True), "'2021-01-01 12:00:00'")
        self.assertEqual(escape(value, False, False), "'2021-01-01 12:00:00'")

        value = datetime.datetime(2021, 1, 1, 12, 0, 0, 1)
        self.assertEqual(escape(value, True, True), "'2021-01-01 12:00:00.000001'")
        self.assertEqual(escape(value, False, True), "'2021-01-01 12:00:00.000001'")
        self.assertEqual(escape(value, False, False), "'2021-01-01 12:00:00.000001'")

        value = np.datetime64(value)
        self.assertEqual(escape(value, True, True), "'2021-01-01 12:00:00.000001'")
        self.assertEqual(escape(value, False, True), "'2021-01-01 12:00:00.000001'")
        self.assertEqual(escape(value, False, False), "'2021-01-01 12:00:00.000001'")

        value = time.struct_time((2021, 1, 1, 12, 0, 0, 0, 1, 0))
        self.assertEqual(escape(value, True, True), "'2021-01-01 12:00:00'")
        self.assertEqual(escape(value, False, True), "'2021-01-01 12:00:00'")
        self.assertEqual(escape(value, False, False), "'2021-01-01 12:00:00'")

        self.log_ended(test)

    def test_escape_encode_date(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE DATE"
        self.log_start(test)

        value = datetime.date(2021, 1, 1)
        self.assertEqual(escape(value, True, True), "'2021-01-01'")
        self.assertEqual(escape(value, False, True), "'2021-01-01'")
        self.assertEqual(escape(value, False, False), "'2021-01-01'")

        self.log_ended(test)

    def test_escape_encode_time(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE TIME"
        self.log_start(test)

        value = datetime.time(12, 0, 0)
        self.assertEqual(escape(value, True, True), "'12:00:00'")
        self.assertEqual(escape(value, False, True), "'12:00:00'")
        self.assertEqual(escape(value, False, False), "'12:00:00'")

        value = datetime.time(12, 0, 0, 1)
        self.assertEqual(escape(value, True, True), "'12:00:00.000001'")
        self.assertEqual(escape(value, False, True), "'12:00:00.000001'")
        self.assertEqual(escape(value, False, False), "'12:00:00.000001'")

        self.log_ended(test)

    def test_escape_encode_timedelta(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE TIMEDELTA"
        self.log_start(test)

        value = datetime.timedelta(days=1, hours=12, minutes=30, seconds=30)
        self.assertEqual(escape(value, True, True), "'36:30:30'")
        self.assertEqual(escape(value, False, True), "'36:30:30'")
        self.assertEqual(escape(value, False, False), "'36:30:30'")

        value = datetime.timedelta(
            days=1, hours=12, minutes=30, seconds=30, microseconds=1
        )
        self.assertEqual(escape(value, True, True), "'36:30:30.000001'")
        self.assertEqual(escape(value, False, True), "'36:30:30.000001'")
        self.assertEqual(escape(value, False, False), "'36:30:30.000001'")

        value = np.timedelta64(value)
        self.assertEqual(escape(value, True, True), "'36:30:30.000001'")
        self.assertEqual(escape(value, False, True), "'36:30:30.000001'")
        self.assertEqual(escape(value, False, False), "'36:30:30.000001'")

        value = -datetime.timedelta(days=1, hours=12, minutes=30, seconds=30)
        self.assertEqual(escape(value, True, True), "'-36:30:30'")
        self.assertEqual(escape(value, False, True), "'-36:30:30'")
        self.assertEqual(escape(value, False, False), "'-36:30:30'")

        value = np.timedelta64(value)
        self.assertEqual(escape(value, True, True), "'-36:30:30'")
        self.assertEqual(escape(value, False, True), "'-36:30:30'")
        self.assertEqual(escape(value, False, False), "'-36:30:30'")

        value = -datetime.timedelta(
            days=1, hours=12, minutes=30, seconds=30, microseconds=1
        )
        self.assertEqual(escape(value, True, True), "'-36:30:30.000001'")
        self.assertEqual(escape(value, False, True), "'-36:30:30.000001'")
        self.assertEqual(escape(value, False, False), "'-36:30:30.000001'")

        value = np.timedelta64(value)
        self.assertEqual(escape(value, True, True), "'-36:30:30.000001'")
        self.assertEqual(escape(value, False, True), "'-36:30:30.000001'")
        self.assertEqual(escape(value, False, False), "'-36:30:30.000001'")

        self.log_ended(test)

    def test_escape_encode_bytes(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE BYTES"
        self.log_start(test)

        value = b"foo\nbar"
        self.assertEqual(escape(value, True, True), "_binary'foo\\nbar'")
        self.assertEqual(escape(value, False, True), "_binary'foo\\nbar'")
        self.assertEqual(escape(value, False, False), "_binary'foo\\nbar'")

        value = bytearray(value)
        self.assertEqual(escape(value, True, True), "_binary'foo\\nbar'")
        self.assertEqual(escape(value, False, True), "_binary'foo\\nbar'")
        self.assertEqual(escape(value, False, False), "_binary'foo\\nbar'")

        value = memoryview(value)
        self.assertEqual(escape(value, True, True), "_binary'foo\\nbar'")
        self.assertEqual(escape(value, False, True), "_binary'foo\\nbar'")
        self.assertEqual(escape(value, False, False), "_binary'foo\\nbar'")

        self.log_ended(test)

    def test_escape_encode_decimal(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE DECIMAL"
        self.log_start(test)

        value = decimal.Decimal("1.2345")
        self.assertEqual(escape(value, True, True), "1.2345")
        self.assertEqual(escape(value, False, True), "1.2345")
        self.assertEqual(escape(value, False, False), "1.2345")

        self.log_ended(test)

    def test_escape_encode_dict(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE DICT"
        self.log_start(test)

        val_item = escape(self.data, False, True)
        self.assertEqual(val_item, escape(self.data, True, True))
        val_lstr = escape(self.data, False, False)
        self.assertEqual(val_lstr, "(%s)" % ",".join(val_item))

        self.log_ended(test)

    def test_escape_encode_sequence(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE SEQUENCE"
        self.log_start(test)

        # List & Tuple
        data = {
            k: v
            for k, v in self.data.items()
            if k not in ("dict", "list", "tuple", "set", "fset")
        }
        for dtype in (list, tuple):
            seq = dtype(data.values())
            val_item = escape(seq, False, True)
            self.assertTrue(type(val_item) is tuple)
            val_many = escape(seq, True, True)
            self.assertTrue(type(val_many) is list)
            self.assertEqual(val_item, tuple(val_many))

            seq = dtype(self.data.values())
            val_item = escape(seq, False, True)
            self.assertTrue(type(val_item) is tuple)
            self.assertEqual({type(i) for i in val_item}, {str})

            val_many = escape(seq, True, True)
            self.assertTrue(type(val_many) is list)
            self.assertTrue({type(i) for i in val_many}, {str, tuple})

        data = (True, 1, 0, -1, 1.1, -1.1)
        for dtype in (set, frozenset):
            seq = dtype(data)
            val_item = escape(seq, False, True)
            self.assertEqual(val_item, escape(seq, True, True))
            val_astr = escape(seq, False, False)
            self.assertEqual(val_astr, "(%s)" % ",".join(val_item))

        self.log_ended(test)

    def test_escape_encode_ndarray_series(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE NDARRAY/SERIES"
        self.log_start(test)

        # . float
        value = list(range(-2, 3))
        for dtype in (np.float16, np.float32, np.float64):
            # ndarray
            data = np.array(value, dtype=dtype)
            comp = tuple(escape(i, False, False) for i in data)
            self.assertEqual(escape(data, True, True), comp)
            self.assertEqual(escape(data, False, True), comp)
            self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
            # Series
            data = pd.Series(data)
            self.assertEqual(escape(data, True, True), comp)
            self.assertEqual(escape(data, False, True), comp)
            self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        # . integer
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            data = np.array(value, dtype=dtype)
            comp = tuple(escape(i, False, False) for i in data)
            self.assertEqual(escape(data, True, True), comp)
            self.assertEqual(escape(data, False, True), comp)
            self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
            # Series
            data = pd.Series(data)
            self.assertEqual(escape(data, True, True), comp)
            self.assertEqual(escape(data, False, True), comp)
            self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        # . unsigned integer
        value = list(range(0, 5))
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            data = np.array(value, dtype=dtype)
            comp = tuple(escape(i, False, False) for i in data)
            self.assertEqual(escape(data, True, True), comp)
            self.assertEqual(escape(data, False, True), comp)
            self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
            # Series
            data = pd.Series(data)
            self.assertEqual(escape(data, True, True), comp)
            self.assertEqual(escape(data, False, True), comp)
            self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        # . bool
        data = np.array([True, False], dtype=np.bool_)
        comp = tuple(escape(i, False, False) for i in data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
        data = pd.Series(data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        # . datetime64
        data = np.array([1, 2], dtype="datetime64[s]")
        comp = tuple(escape(i, False, False) for i in data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
        data = pd.Series(data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        # . timedelta64
        data = np.array([1, 2], dtype="timedelta64[s]")
        comp = tuple(escape(i, False, False) for i in data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
        data = pd.Series(data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        # . bytes
        data = np.array([b"foo", b"bar"], dtype="S")
        comp = tuple(escape(i, False, False) for i in data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
        data = pd.Series(data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        # . unicode
        data = np.array(["foo", "bar"], dtype="U")
        comp = tuple(escape(i, False, False) for i in data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
        data = pd.Series(data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        # . object
        data = np.array([1, "foo", 1.1, True], dtype="O")
        comp = tuple(escape(i, False, False) for i in data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))
        data = pd.Series(data)
        self.assertEqual(escape(data, True, True), comp)
        self.assertEqual(escape(data, False, True), comp)
        self.assertEqual(escape(data, False, False), "(%s)" % ",".join(comp))

        self.log_ended(test)

    def test_escape_encode_dataframe(self) -> None:
        from sqlcycli.transcode import escape

        test = "ESCAPE DATAFRAME"
        self.log_start(test)

        data = pd.DataFrame([self.data] * 2)
        val_item = escape(data, False, True)
        self.assertEqual(val_item, escape(data, True, True))
        val_astr = escape(data, False, False)
        self.assertEqual(
            val_astr, "%s" % ",".join("(%s)" % ",".join(i) for i in val_item)
        )

        self.log_ended(test)

    def test_decode(self) -> None:
        from sqlcycli.transcode import decode

        test = "DECODE"
        self.log_start(test)

        # fmt: off
        # . TINYINT
        self.assertEqual(-1, decode(b"-1", 1, b"utf8", False, False, False))
        # . SMALLINT
        self.assertEqual(2345, decode(b"2345", 2, b"utf8", False, False, False))
        # . MEDIUMINT
        self.assertEqual(-456789, decode(b"-456789", 9, b"utf8", False, False, False))
        # . INT
        self.assertEqual(1234567890, decode(b"1234567890", 3, b"utf8", False, False, False))
        # . BIGINT
        self.assertEqual(-9223372036854775807,
            decode(b"-9223372036854775807", 8, b"utf8", False, False, False))
        self.assertEqual(18446744073709551615,
            decode(b"18446744073709551615", 8, b"utf8", False, False, False))
        # . YEAR
        self.assertEqual(1970, decode(b"1970", 13, b"utf8", False, False, False))
        # . FLOAT
        self.assertEqual(5.7, decode(b"5.7", 4, b"utf8", False, False, False))
        # . DOUBLE
        self.assertEqual(-6.7, decode(b"-6.7", 5, b"utf8", False, False, False))
        # . DECIMAL
        self.assertEqual(-7.7, decode(b"-7.7", 0, b"utf8", False, False, False))
        self.assertEqual(decimal.Decimal("-8.7"), 
            decode(b"-8.7", 246, b"utf8", False, True, False))
        # . DATETIME
        self.assertEqual(datetime.datetime(2014, 5, 15, 7, 45, 57),
            decode(b"2014-05-15 07:45:57", 12, b"utf8", False, False, False))
        self.assertEqual(datetime.datetime(2014, 5, 15, 7, 45, 57, 1000),
            decode(b"2014-05-15 07:45:57.001", 12, b"utf8", False, False, False))
        # . TIMESTAMP
        self.assertEqual(datetime.datetime(2014, 5, 15, 7, 45, 57),
            decode(b"2014-05-15 07:45:57", 7, b"utf8", False, False, True))
        self.assertEqual(datetime.datetime(2014, 5, 15, 7, 45, 57, 1000),
            decode(b"2014-05-15 07:45:57.001", 7, b"utf8", False, False, True))
        # . DATE
        self.assertEqual(datetime.date(1988, 2, 2),
            decode(b"1988-02-02", 10, b"utf8", False, False, False))
        self.assertEqual(datetime.date(1988, 2, 2),
            decode(b"1988-02-02", 14, b"utf8", False, False, True))
        # . TIME
        self.assertEqual(datetime.timedelta(days=6, seconds=79206),
            decode(b'166:00:06', 11, b"utf8", False, False, False))
        self.assertEqual(-datetime.timedelta(days=6, seconds=79206),
            decode(b'-166:00:06', 11, b"utf8", False, False, False))
        self.assertEqual(datetime.timedelta(seconds=59520, microseconds=1000),
            decode(b'16:32:00.001', 11, b"utf8", False, False, False))
        self.assertEqual(-datetime.timedelta(seconds=59520, microseconds=1000),
            decode(b'-16:32:00.001', 11, b"utf8", False, False, False))
        # . CHAR
        self.assertEqual("char", decode(b"char", 254, b"utf8", False, False, False))
        # . VARCHAR
        self.assertEqual("varchar", decode(b"varchar", 253, b"utf8", False, False, False))
        self.assertEqual("varchar", decode(b"varchar", 15, b"utf8", False, False, False))
        # . TINYTEXT
        self.assertEqual("tinytext", decode(b"tinytext", 249, b"utf8", False, False, False))
        # . TEXT
        self.assertEqual("text", decode(b"text", 252, b"utf8", False, False, False))
        # . MEDIUMTEXT
        self.assertEqual("mediumtext", decode(b"mediumtext", 250, b"utf8", False, False, False))
        # . LONGTEXT
        self.assertEqual("longtext", decode(b"longtext", 251, b"utf8", False, False, False))
        # . BINARY
        self.assertEqual(b"binary", decode(b"binary", 254, b"utf8", True, False, False))
        # . VARBINARY
        self.assertEqual(b"varbinary", decode(b"varbinary", 253, b"utf8", True, False, False))
        self.assertEqual(b"varbinary", decode(b"varbinary", 15, b"utf8", True, False, False))
        # . TINYBLOB
        self.assertEqual(b"tinyblob", decode(b"tinyblob", 249, b"utf8", True, False, False))
        # . BLOB
        self.assertEqual(b"blob", decode(b"blob", 252, b"utf8", True, False, False))
        # . MEDIUMBLOB
        self.assertEqual(b"mediumblob", decode(b"mediumblob", 250, b"utf8", True, False, False))
        # . LONGBLOB
        self.assertEqual(b"longblob", decode(b"longblob", 251, b"utf8", True, False, False))
        # . BIT
        self.assertEqual(b"\x01", decode(b"\x01", 16, b"utf8", True, False, False))
        # . ENUM
        self.assertEqual("red", decode(b"red", 247, b"utf8", False, False, False))
        # . SET
        self.assertEqual({"red", "green"}, decode(b"red,green", 248, b"utf8", False, False, False))
        # . JSON
        self.assertEqual({"key1": "value1", "key2": 2}, decode(b'{"key1": "value1", "key2": 2}', 245, b"utf8", False, False, True))
        self.assertEqual('{"key1": "value1", "key2": 2}', decode(b'{"key1": "value1", "key2": 2}', 245, b"utf8", False, False, False))
        self.assertEqual({"key": "中国"}, decode(b'{"key": "\xe4\xb8\xad\xe5\x9b\xbd"}', 245, b"utf8", False, False, True))
        self.assertEqual('{"key": "中国"}', decode(b'{"key": "\xe4\xb8\xad\xe5\x9b\xbd"}', 245, b"utf8", False, False, False))
        self.assertEqual({"key": "Español"}, decode(b'{"key": "Espa\xc3\xb1ol"}', 245, b"utf8", False, False, True))
        self.assertEqual('{"key": "Español"}', decode(b'{"key": "Espa\xc3\xb1ol"}', 245, b"utf8", False, False, False))
        # fmt: on

        self.log_ended(test)


class TestProtocol(TestCase):
    name: str = "Protocol"

    def test_all(self) -> None:
        self.test_FieldDescriptorPacket()
        self.test_OKPacket()
        self.test_EOFPacket()

    def test_FieldDescriptorPacket(self) -> None:
        from sqlcycli.protocol import FieldDescriptorPacket
        from pymysql.protocol import FieldDescriptorPacket as PyFieldDescriptorPacket

        test = "FIELD DESCRIPTOR PACKET"
        self.log_start(test)

        mysql_des_pkt_raw_bytes = [
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x01b\x01b\x0c?\x00\x01\x00\x00\x00\x10 \x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02ti\x02ti\x0c?\x00\x04\x00\x00\x00\x01\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02si\x02si\x0c?\x00\x06\x00\x00\x00\x02\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02mi\x02mi\x0c?\x00\t\x00\x00\x00\t\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x01i\x01i\x0c?\x00\x0b\x00\x00\x00\x03\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x01l\x01l\x0c?\x00\x14\x00\x00\x00\x08\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02ul\x02ul\x0c?\x00\x14\x00\x00\x00\x08 \x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x01y\x01y\x0c?\x00\x04\x00\x00\x00\r`\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x01f\x01f\x0c?\x00\x16\x00\x00\x00\x05\x00\x00\x1f\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02dc\x02dc\x0c?\x00\x0c\x00\x00\x00\xf6\x00\x00\x02\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x01d\x01d\x0c?\x00\n\x00\x00\x00\n\x80\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02dt\x02dt\x0c?\x00\x13\x00\x00\x00\x0c\x80\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02ts\x02ts\x0c?\x00\x13\x00\x00\x00\x07\x80\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02td\x02td\x0c?\x00\n\x00\x00\x00\x0b\x80\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x01t\x01t\x0c?\x00\n\x00\x00\x00\x0b\x80\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02ch\x02ch\x0c\xff\x00\x80\x00\x00\x00\xfe\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02vc\x02vc\x0c\xff\x00\x80\x00\x00\x00\xfd\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02tt\x02tt\x0c\xff\x00\xfc\x03\x00\x00\xfc\x10\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02tx\x02tx\x0c\xff\x00\xfc\xff\x03\x00\xfc\x10\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02mt\x02mt\x0c\xff\x00\xfc\xff\xff\x03\xfc\x10\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02lt\x02lt\x0c\xff\x00\xff\xff\xff\xff\xfc\x10\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02bi\x02bi\x0c?\x00 \x00\x00\x00\xfe\x80\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02vb\x02vb\x0c?\x00 \x00\x00\x00\xfd\x80\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02tb\x02tb\x0c?\x00\xff\x00\x00\x00\xfc\x90\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02bb\x02bb\x0c?\x00\xff\xff\x00\x00\xfc\x90\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02mb\x02mb\x0c?\x00\xff\xff\xff\x00\xfc\x90\x00\x00\x00\x00",
            b"\x03def\x04test\x0etest_datatypes\x0etest_datatypes\x02lb\x02lb\x0c?\x00\xff\xff\xff\xff\xfc\x90\x00\x00\x00\x00",
            b"\x03def\x00\x00\x00\x0clocation_wkt\x00\x0c\xff\x00\x00\x00\x00\x10\xfb\x00\x00\x1f\x00\x00",
            b"\x03def\x00\x00\x00\x0clocation_wkb\x00\x0c?\x00\xff\xff\xff\xff\xfb\x80\x00\x1f\x00\x00",
            b"\x03def\x00\x00\x00\x10location_geojson\x00\x0c\xff\x00\xfc\xff\xff\xff\xf5\x80\x00\x1f\x00\x00",
            b"\x03def\x00\x00\x00\x06length\x00\x0c?\x00\x17\x00\x00\x00\x05\x80\x00\x1f\x00\x00",
            b"\x03def\x00\x00\x00\x04area\x00\x0c?\x00\x17\x00\x00\x00\x05\x80\x00\x1f\x00\x00",
            b"\x03def\x00\x00\x00\x08centroid\x00\x0c\xff\x00\x00\x00\x00\x10\xfb\x00\x00\x1f\x00\x00",
            b"\x03def\x04test\rtest_geometry\rtest_geometry\x08location\x08location\x0c?\x00\xff\xff\xff\xff\xff\x91\x10\x00\x00\x00",
            b"\x03def\x04test\x08test_set\x08test_set\x02id\x02id\x0c?\x00\x0b\x00\x00\x00\x03\x03B\x00\x00\x00",
            b"\x03def\x04test\x08test_set\x08test_set\x04name\x04name\x0c\xff\x00\x90\x01\x00\x00\xfd\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x08test_set\x08test_set\x05color\x05color\x0c\xff\x008\x00\x00\x00\xfe\x00\x08\x00\x00\x00",
            b"\x03def\x04test\ttest_enum\ttest_enum\x02id\x02id\x0c?\x00\x0b\x00\x00\x00\x03\x03B\x00\x00\x00",
            b"\x03def\x04test\ttest_enum\ttest_enum\x04name\x04name\x0c\xff\x00\x90\x01\x00\x00\xfd\x00\x00\x00\x00\x00",
            b"\x03def\x04test\ttest_enum\ttest_enum\x05color\x05color\x0c\xff\x00\x14\x00\x00\x00\xfe\x00\x01\x00\x00\x00",
            b"\x03def\x04test\ttest_json\ttest_json\x02id\x02id\x0c?\x00\x0b\x00\x00\x00\x03\x03B\x00\x00\x00",
            b"\x03def\x04test\ttest_json\ttest_json\x04name\x04name\x0c\xff\x00\x90\x01\x00\x00\xfd\x00\x00\x00\x00\x00",
            b"\x03def\x04test\ttest_json\ttest_json\x04data\x04data\x0c?\x00\xff\xff\xff\xff\xf5\x90\x00\x00\x00\x00",
            b"\x03def\x04test\ttest_null\ttest_null\x02id\x02id\x0c?\x00\x0b\x00\x00\x00\x03\x03B\x00\x00\x00",
            b"\x03def\x04test\ttest_null\ttest_null\x04name\x04name\x0c\xff\x00\x90\x01\x00\x00\xfd\x00\x00\x00\x00\x00",
            b"\x03def\x04test\ttest_null\ttest_null\x04data\x04data\x0c?\x00\x04\x00\x00\x00\x01\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x13test_encoded_uint16\x13test_encoded_uint16\x01c\x01c\x0c\xff\x00\xfc\xff\x00\x00\xfd\x00\x00\x00\x00\x00",
            b"\x03def\x04test\x13test_encoded_uint24\x13test_encoded_uint24\x01c\x01c\x0c\xff\x00\xff\xff\xff\xff\xfc\x10\x00\x00\x00\x00",
            b"\x03def\x04test\x13test_encoded_uint64\x13test_encoded_uint64\x01c\x01c\x0c\xff\x00\xff\xff\xff\xff\xfc\x10\x00\x00\x00\x00",
        ]
        for data in mysql_des_pkt_raw_bytes:
            pkt1 = FieldDescriptorPacket(data, b"utf8")
            pkt2 = PyFieldDescriptorPacket(data, "utf8")
            self.assertEqual(
                (
                    pkt1.catalog,
                    pkt1.db,
                    pkt1.table,
                    pkt1.table_org,
                    pkt1.column,
                    pkt1.column_org,
                    pkt1.charsetnr,
                    pkt1.length,
                    pkt1.type_code,
                    pkt1.flags,
                    pkt1.scale,
                ),
                (
                    pkt2.catalog,
                    pkt2.db.decode("utf8"),
                    pkt2.table_name,
                    pkt2.org_table,
                    pkt2.name,
                    pkt2.org_name,
                    pkt2.charsetnr,
                    pkt2.length,
                    pkt2.type_code,
                    pkt2.flags,
                    pkt2.scale,
                ),
            )

        self.log_ended(test)

    def test_OKPacket(self) -> None:
        from sqlcycli.protocol import MysqlPacket
        from pymysql.protocol import MysqlPacket as PyMysqlPacket, OKPacketWrapper

        test = "OK PACKET"
        self.log_start(test)

        mysql_ok_pkt_raw_bytes = [
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x00\x00\x00\x01\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x00\x01\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x01\x01\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x01\x01\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x01\x01\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x01\x01\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x00\x01\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x00\x01\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x00\x00\x00\x00\x00\x00",
            b"\x00\x01\x00\x01\x00\x00\x00",
        ]

        def read_ok_packet(data: bytes) -> MysqlPacket:
            pkt = MysqlPacket(data, b"utf8")
            pkt.read_ok_packet()
            return pkt

        for data in mysql_ok_pkt_raw_bytes:
            pkt1 = read_ok_packet(data)
            pkt2 = OKPacketWrapper(PyMysqlPacket(data, "utf8"))
            self.assertEqual(
                (
                    pkt1.affected_rows,
                    pkt1.insert_id,
                    pkt1.server_status,
                    pkt1.warning_count,
                    pkt1.message,
                    pkt1.has_next,
                ),
                (
                    pkt2.affected_rows,
                    pkt2.insert_id,
                    pkt2.server_status,
                    pkt2.warning_count,
                    pkt2.message,
                    pkt2.has_next,
                ),
            )

        self.log_ended(test)

    def test_EOFPacket(self) -> None:
        from sqlcycli.protocol import MysqlPacket
        from pymysql.protocol import MysqlPacket as PyMysqlPacket, EOFPacketWrapper

        test = "EOF PACKET"
        self.log_start(test)

        mysql_eof_pkt_raw_bytes = [
            b"\xfe\x00\x00!\x00",
            b"\xfe\x00\x00!\x00",
            b"\xfe\x00\x00!\x00",
            b"\xfe\x00\x00!\x00",
            b"\xfe\x00\x00!\x00",
            b"\xfe\x00\x00!\x00",
            b"\xfe\x00\x00!\x00",
            b"\xfe\x00\x00!\x00",
            b"\xfe\x00\x00!\x00",
        ]

        def read_eof_packet(data: bytes) -> MysqlPacket:
            pkt = MysqlPacket(data, b"utf8")
            pkt.read_eof_packet()
            return pkt

        for data in mysql_eof_pkt_raw_bytes:
            pkt1 = read_eof_packet(data)
            pkt2 = EOFPacketWrapper(PyMysqlPacket(data, "utf8"))
            self.assertEqual(
                (pkt1.warning_count, pkt1.server_status, pkt1.has_next),
                (pkt2.warning_count, pkt2.server_status, pkt2.has_next),
            )

        self.log_ended(test)


class TestConnection(TestCase):
    name: str = "Connection"

    def test_all(self) -> None:
        self.test_properties()
        self.test_set_charset()
        self.test_set_timeout()
        self.test_largedata()
        self.test_autocommit()
        self.test_select_db()
        self.test_connection_gone_away()
        self.test_sql_mode()
        self.test_init_command()
        self.test_close()
        self.test_connection_exception()
        self.test_transaction_exception()
        self.test_warnings()
        self.test_connect_function()

    def test_properties(self) -> None:
        test = "PROPERTIES"
        self.log_start(test)

        with self.get_conn() as conn:
            self.assertEqual(conn.host, "localhost")
            self.assertEqual(conn.port, 3306)
            self.assertEqual(conn.user, "root")
            self.assertEqual(type(conn.password), str)
            self.assertEqual(conn.database, None)
            self.assertEqual(conn.charset, "utf8mb4")
            self.assertEqual(conn.collation, "utf8mb4_general_ci")
            self.assertEqual(conn.encoding, "utf8")
            self.assertEqual(type(conn.connect_timeout), int)
            self.assertEqual(conn.bind_address, None)
            self.assertEqual(conn.unix_socket, None)
            self.assertEqual(conn.autocommit, False)
            self.assertEqual(conn.local_infile, True)
            self.assertEqual(conn.max_allowed_packet, 16777216)
            self.assertEqual(conn.sql_mode, None)
            self.assertEqual(conn.init_command, None)
            self.assertEqual(type(conn.client_flag), int)
            self.assertEqual(conn.ssl, None)
            self.assertEqual(conn.auth_plugin, None)
            self.assertEqual(conn.closed(), False)
            self.assertEqual(conn.use_decimal, False)
            self.assertEqual(conn.decode_json, False)
            self.assertEqual(type(conn.thread_id), int)
            self.assertEqual(type(conn.protocol_version), int)
            self.assertEqual(type(conn.server_info), str)
            self.assertEqual(type(conn.server_version), tuple)
            self.assertEqual(type(conn.server_version_major), int)
            self.assertEqual(type(conn.server_vendor), str)
            self.assertEqual(type(conn.server_status), int)
            self.assertEqual(type(conn.server_capabilites), int)
            self.assertEqual(conn.affected_rows, 0)
            self.assertEqual(conn.insert_id, 0)
        self.log_ended(test)

    def test_set_charset(self):
        test = "SET CHARSET"
        self.log_start(test)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                conn.set_charset("latin1")
                cur.execute("SELECT @@character_set_connection")
                self.assertEqual(cur.fetchone(), ("latin1",))
                self.assertEqual(conn.encoding, "cp1252")

                conn.set_charset("utf8mb4", "utf8mb4_general_ci")
                cur.execute("SELECT @@character_set_connection, @@collation_connection")
                self.assertEqual(cur.fetchone(), ("utf8mb4", "utf8mb4_general_ci"))
                self.assertEqual(conn.encoding, "utf8")
        self.log_ended(test)

    def test_set_timeout(self):
        test = "SET TIMEOUT"
        self.log_start(test)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW VARIABLES LIKE 'wait_timeout'")
                g_wait = int(cur.fetchone()[1])
                cur.execute("SHOW VARIABLES LIKE 'net_read_timeout'")
                g_read = int(cur.fetchone()[1])
                cur.execute("SHOW VARIABLES LIKE 'net_write_timeout'")
                g_write = int(cur.fetchone()[1])

            conn.set_wait_timeout(180)
            self.assertEqual(conn.get_wait_timeout(), 180)
            conn.set_wait_timeout(None)
            self.assertEqual(conn.get_wait_timeout(), g_wait)

            conn.set_read_timeout(180)
            self.assertEqual(conn.get_read_timeout(), 180)
            conn.set_read_timeout(None)
            self.assertEqual(conn.get_read_timeout(), g_read)

            conn.set_write_timeout(180)
            self.assertEqual(conn.get_write_timeout(), 180)
            conn.set_write_timeout(None)
            self.assertEqual(conn.get_write_timeout(), g_write)

        with self.get_conn(
            read_timeout=120,
            write_timeout=120,
            wait_timeout=120,
        ) as conn:
            self.assertEqual(conn.get_wait_timeout(), 120)
            conn.set_wait_timeout(g_wait)
            self.assertEqual(conn.get_wait_timeout(), g_wait)
            conn.set_wait_timeout(None)
            self.assertEqual(conn.get_wait_timeout(), 120)

            self.assertEqual(conn.get_read_timeout(), 120)
            conn.set_read_timeout(g_read)
            self.assertEqual(conn.get_read_timeout(), g_read)
            conn.set_read_timeout(None)
            self.assertEqual(conn.get_read_timeout(), 120)

            self.assertEqual(conn.get_write_timeout(), 120)
            conn.set_write_timeout(g_write)
            self.assertEqual(conn.get_write_timeout(), g_write)
            conn.set_write_timeout(None)
            self.assertEqual(conn.get_write_timeout(), 120)

        self.log_ended(test)

    def test_largedata(self):
        """Large query and response (>=16MB)"""
        test = "LARGE DATA"
        self.log_start(test)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT @@max_allowed_packet")
                if cur.fetchone()[0] < 16 * 1024 * 1024 + 10:
                    self.log_ended(
                        f"{test}: Set max_allowed_packet to bigger than 17MB", True
                    )
                    return None
                t = "a" * (16 * 1024 * 1024)
                cur.execute("SELECT '%s'" % t)
                row = cur.fetchone()[0]
                assert row == t
        self.log_ended(test)

    def test_autocommit(self):
        test = "AUTOCOMMIT"
        self.log_start(test)

        with self.get_conn() as conn:
            conn.set_autocommit(False)
            self.assertFalse(conn.autocommit)

            with conn.cursor() as cur:
                cur.execute("SET AUTOCOMMIT=1")
                self.assertTrue(conn.autocommit)

            conn.set_autocommit(True)
            self.assertTrue(conn.autocommit)

            with conn.cursor() as cur:
                cur.execute("SELECT @@AUTOCOMMIT")
                self.assertEqual(cur.fetchone()[0], 1)
        self.log_ended(test)

    def test_select_db(self):
        test = "SELECT DB"
        self.log_start(test)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                conn.select_database("mysql")
                cur.execute("SELECT database()")
                self.assertEqual(cur.fetchone()[0], "mysql")

                conn.select_database("test")
                cur.execute("SELECT database()")
                self.assertEqual(cur.fetchone()[0], "test")
        self.log_ended(test)

    def test_connection_gone_away(self):
        """
        http://dev.mysql.com/doc/refman/5.0/en/gone-away.html
        http://dev.mysql.com/doc/refman/5.0/en/error-messages-client.html#error_cr_server_gone_error
        """
        test = "CONNECTION GONE AWAY"
        self.log_start(test)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SET wait_timeout=1")
                time.sleep(1.2)
                with self.assertRaises(errors.OperationalError) as cm:
                    cur.execute("SELECT 1+1")
                    # error occurs while reading, not writing because of socket buffer.
                    # self.assertEqual(cm.exception.args[0], 2006)
                    self.assertIn(cm.exception.args[0], (2006, 2013))
        self.log_ended(test)

    def test_sql_mode(self):
        test = "SQL MODE"
        self.log_start(test)

        with self.get_conn(sql_mode="STRICT_TRANS_TABLES") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT @@sql_mode")
                mode = cur.fetchone()[0]
                self.assertIsInstance(mode, str)
                self.assertEqual("STRICT_TRANS_TABLES", mode)

        with self.get_conn(
            sql_mode="ONLY_FULL_GROUP_BY,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION"
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT @@sql_mode")
                mode = cur.fetchone()[0]
                self.assertIsInstance(mode, str)
                self.assertFalse("STRICT_TRANS_TABLES" in mode)
        self.log_ended(test)

    def test_init_command(self):
        test = "INIT COMMAND"
        self.log_start(test)

        with self.get_conn(
            init_command='SELECT "bar"; SELECT "baz"',
            client_flag=CLIENT.MULTI_STATEMENTS,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute('select "foobar";')
                self.assertEqual(("foobar",), cur.fetchone())
        with self.assertRaises(errors.ConnectionClosedError):
            conn.ping(reconnect=False)
        self.log_ended(test)

    def test_close(self):
        test = "CLOSE"
        self.log_start(test)

        with self.get_conn() as conn:
            self.assertFalse(conn.closed())
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                self.assertFalse(cur.closed())
            self.assertTrue(cur.closed())
        self.assertTrue(conn.closed())
        self.log_ended(test)

    def test_connection_exception(self):
        test = "CONNECTION EXCEPTION"
        self.log_start(test)

        with self.get_conn() as conn:
            with self.assertRaises(RuntimeError) as cm:
                self.assertFalse(conn.closed())
                raise RuntimeError("Test")
            self.assertEqual(type(cm.exception), RuntimeError)
            self.assertEqual(str(cm.exception), "Test")
        self.assertTrue(conn.closed())
        self.log_ended(test)

    def test_transaction_exception(self):
        test = "TRANSACTION EXCEPTION"
        self.log_start(test)

        with self.get_conn() as conn:
            try:
                with conn.transaction() as cur:
                    cur.execute("SELECT 1")
                    raise RuntimeError("Test")
            except RuntimeError:
                pass
            self.assertTrue(cur.closed())
            self.assertTrue(conn.closed())
        self.log_ended(test)

    def test_warnings(self):
        test = "WARNINGS"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cur.execute(f"CREATE TABLE {self.table} (a INT UNIQUE)")
                cur.execute(f"INSERT INTO {self.table} (a) VALUES (1)")
                cur.execute(
                    f"INSERT INTO {self.table} (a) VALUES (1) ON DUPLICATE KEY UPDATE a=VALUES(a)"
                )
                w = conn.show_warnings()
                self.assertIsNotNone(w)
                self.assertEqual(w[0][1], ER.WARN_DEPRECATED_SYNTAX)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_connect_function(self):
        from sqlcycli._connect import connect

        test = "CONNECT FUNCTION"
        self.log_start(test)

        with connect(
            host=self.host,
            user=self.user,
            password=self.password,
            unix_socket=self.unix_socket,
            local_infile=True,
        ) as conn:
            self.assertEqual(conn.host, self.host)
            self.assertEqual(conn.user, self.user)
            self.assertEqual(conn.password, self.password)
            self.assertEqual(conn.unix_socket, self.unix_socket)
            self.assertEqual(conn.local_infile, True)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                self.assertEqual(cur.fetchone(), (1,))

        self.log_ended(test)


class TestAuthentication(TestCase):
    name: str = "Authentication"

    def test_all(self) -> None:
        self.test_password_algo()
        self.test_plugin()

    def test_password_algo(self) -> None:
        test = "PASSWORD ALGORITHM"
        self.log_start(test)

        password = b"mypassword_123"
        salt = b"\x1aOZeFXX{XY\x18\x0c CW (u\x17F"

        from sqlcycli._auth import scramble_native_password
        from pymysql._auth import (
            scramble_native_password as py_scramble_native_password,
        )

        r1 = scramble_native_password(password, salt)
        r2 = py_scramble_native_password(password, salt)
        self.assertEqual(r1, r2)

        from sqlcycli._auth import scramble_caching_sha2
        from pymysql._auth import scramble_caching_sha2 as py_scramble_caching_sha2

        r1 = scramble_caching_sha2(password, salt)
        r2 = py_scramble_caching_sha2(password, salt)
        self.assertEqual(r1, r2)

        from sqlcycli._auth import ed25519_password
        from pymysql._auth import ed25519_password as py_ed25519_password

        r1 = ed25519_password(password, salt)
        r2 = py_ed25519_password(password, salt)
        self.assertEqual(r1, r2)

        self.log_ended(test)

    def test_plugin(self) -> None:
        test = "PLUGIN"
        self.log_start(test)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select plugin from mysql.user where concat(user, '@', host)=current_user()"
                )
                for row in cur:
                    self.assertIn(
                        conn.server_auth_plugin_name, (row[0], "mysql_native_password")
                    )
        self.log_ended(test)


class TestConversion(TestCase):
    name: str = "Conversion"

    def test_all(self) -> None:
        self.test_bool()
        self.test_integer()
        self.test_float()
        self.test_string()
        self.test_null()
        self.test_datetime()
        self.test_date()
        self.test_time()
        self.test_timedelta()
        self.test_binary()
        self.test_dict()
        self.test_sequence()
        self.test_ndarray_series_float()
        self.test_ndarray_series_integer()
        self.test_ndarray_series_bool()
        self.test_ndarray_series_datetime()
        self.test_ndarray_series_timedelta()
        self.test_ndarray_series_bytes()
        self.test_ndarray_series_unicode()
        self.test_ndarray_series_object()
        self.test_dataframe()
        self.test_json()
        self.test_bulk_insert()

    def test_bool(self) -> None:
        test = "BOOL TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(f"create table {self.table} (a bit, b tinyint)")
                # . insert values
                cur.execute(
                    f"insert into {self.table} (a, b) values (%s, %s)", (True, False)
                )
                # . validate
                cur.execute(f"SELECT a, b FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                self.delete(conn)

                ##################################################################
                # numpy bool
                cur.execute(
                    f"insert into {self.table} (a, b) values (%s, %s)",
                    (np.bool_(True), np.bool_(False)),
                )
                # . validate
                cur.execute(f"SELECT a, b FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_integer(self) -> None:
        test = "INTEGER TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"""
                    create table {self.table} (
                        a tinyint, b tinyint, c tinyint unsigned,
                        d smallint, e smallint, f smallint unsigned,
                        g mediumint, h mediumint, i mediumint unsigned,
                        j int, k int, l int unsigned,
                        m bigint, n bigint, o bigint unsigned
                    )"""
                )
                # . insert values
                # fmt: off
                test_value = (
                    -128, 127, 255, 
                    -32768, 32767, 65535, 
                    -8388608, 8388607, 16777215, 
                    -2147483648, 2147483647, 4294967295, 
                    -9223372036854775808, 9223372036854775807, 18446744073709551615,
                )
                # fmt: on
                cur.execute(
                    f"""
                    insert into {self.table} (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) 
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    test_value,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g,h,i,j,k,l,m,n,o FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                # numpy integer
                test_value_np = (
                    np.int8(-128),
                    np.int8(127),
                    np.uint8(255),
                    np.int16(-32768),
                    np.int16(32767),
                    np.uint16(65535),
                    np.int32(-8388608),
                    np.int32(8388607),
                    np.uint32(16777215),
                    np.int64(-2147483648),
                    np.int64(2147483647),
                    np.uint64(4294967295),
                    np.int64(-9223372036854775808),
                    np.int64(9223372036854775807),
                    np.uint64(18446744073709551615),
                )
                cur.execute(
                    f"""
                    insert into {self.table} (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) 
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    test_value_np,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g,h,i,j,k,l,m,n,o FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_float(self) -> None:
        test = "FLOAT TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"""
                    create table {self.table} (
                        a float, b double, c decimal(10, 2)
                    )"""
                )
                # . insert values
                test_value = (5.7, 6.8, decimal.Decimal("7.9"))
                cur.execute(
                    f"insert into {self.table} (a,b,c) values (%s,%s,%s)", test_value
                )
                # . use_decimal = False
                conn.set_use_decimal(False)
                # . validate
                cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual((5.7, 6.8, 7.9), row)

                ##################################################################
                # . use_decimal = True
                conn.set_use_decimal(True)
                # . validate
                cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                # numpy float
                test_value_np = (
                    np.float16(5.7),
                    np.float32(6.8),
                    np.float64(7.9),
                )
                cur.execute(
                    f"insert into {self.table} (a,b,c) values (%s,%s,%s)", test_value_np
                )
                # . validate
                cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_string(self) -> None:
        test = "STRING TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"""
                    create table {self.table} (
                        a char(32), b varchar(32),
                        c tinytext, d text, e mediumtext, f longtext
                    )"""
                )
                # . insert values
                test_value = (
                    "char 中文 한국어 にほんご Español",
                    "varchar 中文 한국어 にほんご Español",
                    "tinytext 中文 한국어 にほんご Español",
                    "text 中文 한국어 にほんご Español",
                    "mediumtext 中国 한국어 にほんご Español",
                    "longtext 中文 한국어 にほんご Español",
                )
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f) values (%s,%s,%s,%s,%s,%s)",
                    test_value,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                # numpy string
                test_value_np = (
                    np.str_("char 中文 한국어 にほんご Español"),
                    np.str_("varchar 中文 한국어 にほんご Español"),
                    np.str_("tinytext 中文 한국어 にほんご Español"),
                    np.str_("text 中文 한국어 にほんご Español"),
                    np.str_("mediumtext 中国 한국어 にほんご Español"),
                    np.str_("longtext 中文 한국어 にほんご Español"),
                )
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f) values (%s,%s,%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_null(self) -> None:
        test = "NULL TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(f"create table {self.table} (a char(32), b varchar(32))")
                # . insert values
                test_value = (None, None)
                cur.execute(
                    f"insert into {self.table} (a,b) values (%s,%s)", test_value
                )
                # . validate
                cur.execute(f"SELECT a,b FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                # . direct select
                cur.execute("select null,''")
                row = cur.fetchone()
                self.assertEqual((None, ""), row)
                # . validate
                cur.execute("select '',null")
                row = cur.fetchone()
                self.assertEqual(("", None), row)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_datetime(self) -> None:
        test = "DATETIME TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} (a datetime, b datetime(6), c timestamp, d timestamp(6), e datetime)"
                )
                # . insert values
                test_value = (
                    datetime.datetime(2014, 5, 15, 7, 45, 57),
                    datetime.datetime(2014, 5, 15, 7, 45, 57, 51000),
                    datetime.datetime(2014, 5, 15, 7, 45, 57),
                    datetime.datetime(2014, 5, 15, 7, 45, 57, 51000),
                    time.localtime(),
                )
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                    test_value,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value[1:4], row[1:4])
                self.assertEqual(datetime.datetime(*test_value[4][0:6]), row[4])
                self.delete(conn)

                ##################################################################
                # numpy datetime
                test_value_np = (
                    np.datetime64("2014-05-15T07:45:57"),
                    np.datetime64("2014-05-15T07:45:57.051000"),
                    np.datetime64("2014-05-15T07:45:57"),
                    np.datetime64("2014-05-15T07:45:57.051000"),
                    np.datetime64(datetime.datetime(*test_value[4][0:6])),
                )
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value[1:4], row[1:4])
                self.assertEqual(datetime.datetime(*test_value[4][0:6]), row[4])
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_date(self) -> None:
        test = "DATE TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(f"create table {self.table} (a date, b date)")
                # . insert values
                test_value = (datetime.date(1988, 2, 2), datetime.date(1988, 2, 2))
                cur.execute(
                    f"insert into {self.table} (a,b) values (%s,%s)", test_value
                )
                # . validate
                cur.execute(f"SELECT a,b FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_time(self) -> None:
        test = "TIME TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"""
                    create table {self.table} (
                        a time, b time, c time, d time, 
                        e time(6), f time(6), g time(6)
                    )"""
                )
                # . insert values
                test_value = (
                    datetime.time(0),
                    datetime.time(1),
                    datetime.time(0, 2),
                    datetime.time(1, 2),
                    datetime.time(0, 0, 3),
                    datetime.time(0, 2, 3),
                    datetime.time(1, 2, 3),
                )
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) values %s",
                    test_value,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = cur.fetchone()

                def time_to_timedelta(t: datetime.time) -> datetime.timedelta:
                    return datetime.timedelta(
                        0, t.hour * 3600 + t.minute * 60 + t.second, t.microsecond
                    )

                self.assertEqual(tuple(time_to_timedelta(i) for i in test_value), row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_timedelta(self) -> None:
        test = "TIMEDELTA TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} "
                    "(a time,b time,c time(6),d time,e time,f time(6),g time)"
                )
                # . insert values
                test_values = (
                    datetime.timedelta(0, 45000),
                    datetime.timedelta(0, 83579),
                    datetime.timedelta(0, 83579, 51000),
                    -datetime.timedelta(0, 45000),
                    -datetime.timedelta(0, 83579),
                    -datetime.timedelta(0, 83579, 51000),
                    -datetime.timedelta(0, 1800),
                )
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) "
                    "values (%s,%s,%s,%s,%s,%s,%s)",
                    test_values,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_values, row)
                self.delete(conn)

                ##################################################################
                # numpy timedelta
                test_values_np = [np.timedelta64(i) for i in test_values]
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) "
                    "values (%s,%s,%s,%s,%s,%s,%s)",
                    test_values_np,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_values, row)
                self.delete(conn)

                ##################################################################
                # . direct select
                cur.execute(
                    "select time('12:30'), time('23:12:59'), time('23:12:59.05100'),"
                    " time('-12:30'), time('-23:12:59'), time('-23:12:59.05100'), time('-00:30')"
                )
                # . validate
                row = cur.fetchone()
                self.assertEqual(test_values, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_binary(self) -> None:
        test = "BINARY TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"""
                    create table {self.table} (
                        a binary(255), 
                        b binary(255), 
                        c binary(255), 
                        d binary(255), 
                        e varbinary(255),
                        f tinyblob, g blob, h mediumblob, i longblob
                    )"""
                )
                # . insert values
                b = bytearray(range(255))
                a = bytes(b)
                c = memoryview(b)
                d = np.bytes_(b)
                test_value = tuple(
                    [a, b, c, d]
                    + [
                        i.encode("utf8")
                        for i in [
                            "varchar 中文 한국어 にほんご Español",
                            "tinytext 中文 한국어 にほんご Español",
                            "text 中文 한국어 にほんご Español" * 100,
                            "mediumtext 中国 한국어 にほんご Español" * 1000,
                            "longtext 中文 한국어 にほんご Español" * 10000,
                        ]
                    ]
                )
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g,h,i) "
                    "values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    test_value,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g,h,i FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_dict(self) -> None:
        test = "DICT ESCAPING"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} (a integer, b integer, c integer)"
                )
                # . insert values
                cur.execute(
                    f"insert into {self.table} (a,b,c) values (%s, %s, %s)",
                    {"a": 1, "b": 2, "c": 3},
                )
                # . validate
                cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual((1, 2, 3), row)
                self.delete(conn)

                ##################################################################
                cur.execute(
                    f"insert into {self.table} (a,b,c) values %s",
                    {"a": 1, "b": 2, "c": 3},
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual((1, 2, 3), row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_sequence(self) -> None:
        test = "SEQUENCE TYPE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(f"create table {self.table} (i integer, l integer)")
                # . insert values
                for seq_type in (list, tuple, set, frozenset, pd.Series, np.array):
                    cur.execute(
                        f"insert into {self.table} (i, l) values (2, 4), (6, 8), (10, 12)"
                    )
                    seq = seq_type([2, 6])
                    cur.execute(
                        "select l from %s where i in (%s) order by i"
                        % (self.table, "%s, %s"),
                        seq,
                    )
                    row = cur.fetchall()
                    self.assertEqual(((4,), (8,)), row)

                    # ------------------------------------------------------
                    cur.execute(
                        "select l from %s where i in %s order by i"
                        % (self.table, "%s"),
                        seq,
                        itemize=False,
                    )
                    row = cur.fetchall()
                    self.assertEqual(((4,), (8,)), row)
                    self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_ndarray_series_float(self) -> None:
        test = "NDARRAY/SERIES FLOAT"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} "
                    "(a double, b double, c double, d double, e double)"
                )
                # . validate float
                test_value = tuple(float(i) for i in range(-2, 3))
                for dtype in (np.float16, np.float32, np.float64):
                    test_value_np = np.array(test_value, dtype=dtype)
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                        test_value_np,
                    )
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                    # ------------------------------------------------------
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_np,
                        itemize=False,
                    )
                    # . validate
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                    # ------------------------------------------------------
                    test_value_pd = pd.Series(test_value_np, dtype=dtype)
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_pd,
                        itemize=False,
                    )
                    # . validate
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_ndarray_series_integer(self) -> None:
        test = "NDARRAY/SERIES INTEGER"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} "
                    "(a bigint, b bigint, c bigint, d bigint, e bigint)"
                )
                # . validate int
                test_value = tuple(range(-2, 3))
                for dtype in (np.int8, np.int16, np.int32, np.int64):
                    test_value_np = np.array(test_value, dtype=dtype)
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                        test_value_np,
                    )
                    # . validate
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                    # ------------------------------------------------------
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_np,
                        itemize=False,
                    )
                    # . validate
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                    # ------------------------------------------------------
                    test_value_pd = pd.Series(test_value_np, dtype=dtype)
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_pd,
                        itemize=False,
                    )
                    # . validate
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                ##################################################################
                # . validate uint
                test_value = tuple(range(5))
                for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                    test_value_np = np.array(test_value, dtype=dtype)
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                        test_value_np,
                    )
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                    # ------------------------------------------------------
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_np,
                        itemize=False,
                    )
                    # . validate
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                    # ------------------------------------------------------
                    test_value_pd = pd.Series(test_value_np, dtype=dtype)
                    cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_pd,
                        itemize=False,
                    )
                    # . validate
                    cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = cur.fetchone()
                    self.assertEqual(test_value, row)
                    self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_ndarray_series_bool(self) -> None:
        test = "NDARRAY/SERIES BOOL"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(f"create table {self.table} (a bit, b tinyint)")
                # . insert values
                test_value_np = np.array([True, False], dtype=np.bool_)
                cur.execute(
                    f"insert into {self.table} (a,b) values (%s,%s)", test_value_np
                )
                # . validate
                cur.execute(f"SELECT a,b FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                self.delete(conn)

                ##################################################################
                cur.execute(
                    f"insert into {self.table} (a,b) values %s",
                    test_value_np,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype=np.bool_)
                cur.execute(
                    f"insert into {self.table} (a,b) values %s",
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_ndarray_series_datetime(self) -> None:
        test = "NDARRAY/SERIES DATETIME"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} "
                    "(a datetime, b datetime(6), c timestamp, d timestamp(6))"
                )
                # . insert values
                test_value = (
                    datetime.datetime(2014, 5, 15, 7, 45, 57),
                    datetime.datetime(2014, 5, 15, 7, 45, 57, 51000),
                    datetime.datetime(2014, 5, 15, 7, 45, 57),
                    datetime.datetime(2014, 5, 15, 7, 45, 57, 51000),
                )
                test_value_np = np.array(test_value, dtype="datetime64[us]")
                cur.execute(
                    f"insert into {self.table} (a,b,c,d) values (%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                cur.execute(
                    f"insert into {self.table} (a,b,c,d) values %s",
                    test_value_np,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype="datetime64[us]")
                cur.execute(
                    f"insert into {self.table} (a,b,c,d) values %s",
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_ndarray_series_timedelta(self) -> None:
        test = "NDARRAY/SERIES TIMEDELTA"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} "
                    "(a time, b time, c time(6), d time, e time, f time(6), g time)"
                )
                # . insert values
                test_values = (
                    datetime.timedelta(0, 45000),
                    datetime.timedelta(0, 83579),
                    datetime.timedelta(0, 83579, 51000),
                    -datetime.timedelta(0, 45000),
                    -datetime.timedelta(0, 83579),
                    -datetime.timedelta(0, 83579, 51000),
                    -datetime.timedelta(0, 1800),
                )
                test_values_np = np.array(test_values, dtype="timedelta64[us]")
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) values (%s,%s,%s,%s,%s,%s,%s)",
                    test_values_np,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_values, row)
                self.delete(conn)

                ##################################################################
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) values %s",
                    test_values_np,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_values, row)
                self.delete(conn)

                ##################################################################
                test_values_pd = pd.Series(test_values_np, dtype="timedelta64[us]")
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) values %s",
                    test_values_pd,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_values, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_ndarray_series_bytes(self) -> None:
        test = "NDARRAY/SERIES BYTES"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} "
                    "(a varbinary(32), b varbinary(32), c varbinary(32), d varbinary(32), e varbinary(32))"
                )
                # . insert values
                test_value = tuple(str(i).encode("utf8") for i in range(5))
                test_value_np = np.array(test_value, dtype="S")
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values %s",
                    test_value_np,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype="S")
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values %s",
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_ndarray_series_unicode(self) -> None:
        test = "NDARRAY/SERIES UNICODE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # . create test table
                cur.execute(
                    f"create table {self.table} "
                    "(a varchar(32), b varchar(32), c varchar(32), d varchar(32), e varchar(32))"
                )
                # . insert values
                test_value = tuple(str(i) for i in range(5))
                test_value_np = np.array(test_value, dtype="U")
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values %s",
                    test_value_np,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype="U")
                cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values %s",
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_ndarray_series_object(self) -> None:
        test = "NDARRAY/SERIES OBJECT"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cols: dict = {
                    "ti": "tinyint",
                    "si": "smallint",
                    "mi": "mediumint",
                    "i": "int",
                    "l": "bigint",
                    "ul": "bigint unsigned",
                    "y": "year",
                    "f": "real",
                    "dc": "decimal(10, 2)",
                    "d": "date",
                    "dt": "datetime",
                    "td": "time(6)",
                    "ch": "char(32)",
                    "vc": "varchar(32)",
                    "tt": "tinytext",
                    "tx": "text",
                    "mt": "mediumtext",
                    "lt": "longtext",
                    "bi": "binary(32)",
                    "vb": "varbinary(32)",
                    "tb": "tinyblob",
                    "bb": "blob",
                    "mb": "mediumblob",
                    "lb": "longblob",
                }
                cur.execute(
                    "create table %s (%s)"
                    % (self.table, ", ".join([f"{k} {v}" for k, v in cols.items()]))
                )
                # . insert values
                test_value = (
                    -1,  # 1 TINY - TINYINT
                    2345,  # 2 SHORT - SMALLINT
                    -3456,  # 9 INT24 - MEDIUMINT
                    456789,  # 3 LONG - INT
                    -9223372036854775807,  # 8 LONGLONG - BIGINT
                    18446744073709551615,  # 8 LONGLONG - BIGINT UNSIGNED
                    1970,  # 13 YEAR - YEAR
                    5.7,  # 5 DOUBLE - DOUBLE
                    6.8,  # 5 NEWDECIMAL - DECIMAL
                    datetime.date(1988, 2, 2),  # 10
                    datetime.datetime(2014, 5, 15, 7, 45, 57),  # 12
                    -datetime.timedelta(5, 6, microseconds=7),  # 11
                    "char 中国 Español",  # 254
                    "varchar 中国 Español",  # 253
                    "tinytext 中国 Español",  # 252
                    "text 中国 Español",  # 252
                    "mediumtext 中国 Español",  # 252
                    "longtext 中国 Español",  # 252
                    bytearray(range(32)),  # 254
                    "varbinary 中国 Español".encode(conn.encoding),  # 253
                    "tinyblob 中国 Español".encode(conn.encoding),  # 252
                    "blob 中国 Español".encode(conn.encoding),  # 252
                    "mediumblob 中国 Español".encode(conn.encoding),  # 252
                    "longblob 中国 Español".encode(conn.encoding),  # 252
                )
                test_value_np = np.array(test_value, dtype="O")
                cur.execute(
                    "insert into %s (%s) values (%s)"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_value_np,
                )
                # . validate
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                cur.execute(
                    "insert into %s (%s) values %s"
                    % (self.table, ",".join(cols.keys()), "%s"),
                    test_value_np,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype="O")
                cur.execute(
                    "insert into %s (%s) values %s"
                    % (self.table, ",".join(cols.keys()), "%s"),
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_dataframe(self) -> None:
        test = "DATAFRAME"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cols: dict = {
                    "a": "real",
                    "b": "bigint",
                    "c": "tinyint",
                    "d": "datetime(6)",
                    "e": "time(6)",
                    "f": "varbinary(32)",
                    "g": "varchar(32)",
                }
                cur.execute(
                    "create table %s (%s)"
                    % (self.table, ", ".join([f"{k} {v}" for k, v in cols.items()]))
                )
                # . insert values
                size = 100
                test_df = pd.DataFrame(
                    # fmt: off
                    {
                        "float64": np.array([i for i in range(size)], dtype=np.float64),
                        "int64": np.array([i for i in range(size)], dtype=np.int64),
                        "bool": np.array([i for i in range(size)], dtype=np.bool_),
                        "dt64": np.array([i for i in range(size)], dtype="datetime64[s]"),
                        "td64": np.array([i for i in range(size)], dtype="timedelta64[s]"),
                        "bytes": np.array([i for i in range(size)], dtype="S"),
                        "unicode": np.array([i for i in range(size)], dtype="U"),
                    }
                    # fmt: on
                )
                test_value = tuple(test_df.iloc[0].tolist())
                cur.execute(
                    "insert into %s (%s) values (%s)"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_df,
                )
                # . validate
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                cur.execute(
                    "insert into %s (%s) values %s"
                    % (self.table, ",".join(cols.keys()), "%s"),
                    test_df,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_json(self) -> None:
        test = "JSON TYPE"
        self.log_start(test)

        with self.setup() as conn:
            if conn.server_version < (5, 7, 0):
                self.log_ended(test, True)
                return None
            with conn.cursor() as cur:
                ##################################################################
                cur.execute(
                    f"create table {self.table} "
                    "(id int primary key not null, json JSON not null)"
                )
                # . insert values
                test_value = '{"hello": "こんにちは"}'
                cur.execute(
                    f"INSERT INTO {self.table} (id, `json`) values (42, %s)",
                    test_value,
                )
                # . decode_json = False
                conn.set_decode_json(False)
                # . validate
                cur.execute(f"SELECT `json` from {self.table} WHERE `id`=42")
                res = cur.fetchone()[0]
                self.assertEqual(orjson.loads(test_value), orjson.loads(res))

                ##################################################################
                # . decode_json = True
                conn.set_decode_json(True)
                # . validate
                cur.execute(f"SELECT `json` from {self.table} WHERE `id`=42")
                res = cur.fetchone()[0]
                self.assertEqual(orjson.loads(test_value), res)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_bulk_insert(self) -> None:
        test = "BULK INSERT"
        self.log_start(test)

        with self.setup() as conn:
            table_encoded = self.table.encode(conn.encoding)
            with conn.cursor() as cur:
                ##################################################################
                cols: dict = {
                    "a": "real",
                    "b": "bigint",
                    "c": "tinyint",
                    "d": "datetime(6)",
                    "e": "time(6)",
                    "f": "varbinary(32)",
                    "g": "varchar(32)",
                }
                cur.execute(
                    "create table %s (%s)"
                    % (self.table, ", ".join([f"{k} {v}" for k, v in cols.items()]))
                )
                # . insert values
                test_value = (
                    1.1,
                    2,
                    True,
                    datetime.datetime(2014, 5, 15, 7, 45, 57),
                    datetime.timedelta(1, 2, 3),
                    b"binary",
                    "varchar",
                )
                test_value_bulk = [test_value, test_value, test_value]
                cur.executemany(
                    "insert into %s (%s) values (%s)"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_value_bulk,
                )
                # . validate
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                # . single format (itemize = False)
                cur.execute(
                    "insert into %s (%s) values %s"
                    % (self.table, ",".join(cols.keys()), "%s"),
                    test_value_bulk,
                    itemize=False,
                )
                # . validate
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                # . as v
                cur.executemany(
                    "insert into %s (%s) values (%s) as v"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_value_bulk,
                )

                sql = cur.executed_sql
                # . validate
                self.assertEqual(
                    sql,
                    b"insert into %s (a,b,c,d,e,f,g) values "
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar'),"
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar'),"
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar') "
                    b"as v" % table_encoded,
                )
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                # . on duplicate
                cur.executemany(
                    "insert into %s (%s) values (%s) on duplicate key update a = values(a)"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_value_bulk,
                )
                sql = cur.executed_sql
                # . validate
                self.assertEqual(
                    sql,
                    b"insert into %s (a,b,c,d,e,f,g) values "
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar'),"
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar'),"
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar') "
                    b"on duplicate key update a = values(a)" % table_encoded,
                )
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                # . as v on duplicate
                cur.executemany(
                    "insert into %s (%s) values (%s) as v on duplicate key update a = v.a"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_value_bulk,
                )
                sql = cur.executed_sql
                # . validate
                self.assertEqual(
                    sql,
                    b"insert into %s (a,b,c,d,e,f,g) values "
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar'),"
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar'),"
                    b"(1.1,2,1,'2014-05-15 07:45:57','24:00:02.000003',_binary'binary','varchar') "
                    b"as v on duplicate key update a = v.a" % table_encoded,
                )
                cur.execute(f"SELECT * FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(test_value, row)
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)


class TestCursor(TestCase):
    name: str = "Cursor"
    bob = {"name": "bob", "age": 21, "DOB": datetime.datetime(1990, 2, 6, 23, 4, 56)}
    jim = {"name": "jim", "age": 56, "DOB": datetime.datetime(1955, 5, 9, 13, 12, 45)}
    fred = {"name": "fred", "age": 100, "DOB": datetime.datetime(1911, 9, 12, 1, 1, 1)}

    def test_all(self) -> None:
        self.test_properties()
        self.test_mogrify()
        self.test_acquire_directly()
        self.test_fetch_no_result()
        self.test_fetch_single_tuple()
        self.test_fetch_aggregates()
        self.test_cursor_iter()
        self.test_cleanup_rows_buffered()
        self.test_cleanup_rows_unbuffered()
        self.test_execute_args()
        self.test_executemany()
        self.test_execution_time_limit()
        self.test_warnings()
        self.test_SSCursor()
        self.test_DictCursor(False)
        self.test_DictCursor(True)
        self.test_DfCursor(False)
        self.test_DfCursor(True)
        self.test_next_set()
        self.test_skip_next_set()
        self.test_next_set_error()
        self.test_ok_and_next()
        self.test_multi_statement_warnings()
        self.test_previous_cursor_not_closed()
        self.test_commit_during_multi_result()
        self.test_transaction()
        self.test_scroll()
        self.test_procedure()

    def test_properties(self) -> None:
        from sqlcycli.protocol import FieldDescriptorPacket

        test = "PROPERTIES"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cur.execute(
                    f"create table {self.table} (id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, value INT)"
                )
                cur.execute(f"insert into {self.table} (value) values (1),(2)")
                self.assertTrue(cur.executed_sql.startswith(b"insert into"))
                self.assertEqual(cur.field_count, 0)
                self.assertIsNone(cur.fields)
                self.assertEqual(cur.insert_id, 1)
                self.assertEqual(cur.affected_rows, 2)
                self.assertEqual(cur.rownumber, 0)
                self.assertEqual(cur.warning_count, 0)
                self.assertEqual(cur.rowcount, 2)
                self.assertIsNone(cur.description)

                cur.execute(f"SELECT value FROM {self.table} where value in (1)")
                self.assertTrue(cur.executed_sql.startswith(b"SELECT value FROM"))
                self.assertEqual(cur.field_count, 1)
                self.assertEqual(type(cur.fields[0]), FieldDescriptorPacket)
                self.assertEqual(cur.insert_id, 0)
                self.assertEqual(cur.affected_rows, 1)
                self.assertEqual(cur.rownumber, 0)
                self.assertEqual(cur.warning_count, 0)
                self.assertEqual(cur.rowcount, 1)
                self.assertEqual(len(cur.description[0]), 7)
                cur.fetchall()
                self.assertEqual(cur.rownumber, 1)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_mogrify(self) -> None:
        test = "MOGRIFY"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                # No args
                sql = "SELECT * FROM foo"
                self.assertEqual(cur.mogrify(sql), sql)
                # one arg
                sql = "SELECT * FROM foo WHERE bar = %s"
                self.assertEqual(
                    cur.mogrify(sql, 42), "SELECT * FROM foo WHERE bar = 42"
                )
                # one row
                sql = "INSERT INTO foo (bar) VALUES (%s, %s)"
                self.assertEqual(
                    cur.mogrify(sql, (42, 43)), "INSERT INTO foo (bar) VALUES (42, 43)"
                )
                # multi-rows
                sql = "INSERT INTO foo (bar) VALUES (%s, %s)"
                self.assertEqual(
                    cur.mogrify(sql, [(42, 43), (44, 45)], many=True),
                    "INSERT INTO foo (bar) VALUES (42, 43)",
                )
                sql = "SELECT * FROM foo WHERE bar IN %s AND foo = %s"
                self.assertEqual(
                    cur.mogrify(sql, ([(1, 2, 3), 42], [(4, 5, 6), 42]), many=True),
                    "SELECT * FROM foo WHERE bar IN (1,2,3) AND foo = 42",
                )

                sql = "SELECT * FROM foo"
                self.assertEqual(cur.mogrify(sql, (), many=True), sql)
                with self.assertRaises(errors.InvalidSQLArgsErorr):
                    sql = "SELECT * FROM foo WHERE bar IN %s AND foo = %s"
                    print(cur.mogrify(sql, (), many=True))

        self.log_ended(test)

    def test_acquire_directly(self) -> None:
        test = "ACQUIRE DIRECTLY"
        self.log_start(test)

        with self.setup() as conn:
            ##################################################################
            try:
                cur = conn.cursor()
                cur.execute(f"create table {self.table} (id integer primary key)")
                cur.execute(f"insert into {self.table} (id) values (1),(2)")
                cur.execute(f"SELECT id FROM {self.table} where id in (1)")
                self.assertEqual(((1,),), cur.fetchall())
            finally:
                cur.close()
            ##################################################################
            self.drop(conn)
        self.log_ended(test)

    def test_fetch_no_result(self) -> None:
        test = "FETCH NO RESULT"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cur.execute(f"create table {self.table} (a varchar(32))")
                cur.execute(f"insert into {self.table} (a) values (%s)", "mysql")
                self.assertEqual(None, cur.fetchone())
                self.assertEqual((), cur.fetchall())
                self.assertEqual((), cur.fetchmany(2))

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_fetch_single_tuple(self) -> None:
        test = "FETCH SINGLE TUPLE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cur.execute(f"create table {self.table} (id integer primary key)")
                cur.execute(f"insert into {self.table} (id) values (1),(2)")
                cur.execute(f"SELECT id FROM {self.table} where id in (1)")
                self.assertEqual(((1,),), cur.fetchall())

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_fetch_aggregates(self) -> None:
        test = "FETCH AGGREGATES"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cur.execute(f"create table {self.table} (id integer primary key)")
                cur.executemany(
                    f"insert into {self.table} (id) values (%s)", tuple(range(10))
                )
                cur.execute(f"SELECT sum(id) FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(sum(range(10)), row[0])

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_cursor_iter(self) -> None:
        test = "CURSOR ITER"
        self.log_start(test)

        with self.setupForCursor() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cur.execute(f"select * from {self.table}")
                self.assertEqual(cur.__iter__(), cur)
                self.assertEqual(cur.__next__(), ("row1",))
                self.assertEqual(cur.__next__(), ("row2",))
                self.assertEqual(cur.__next__(), ("row3",))
                self.assertEqual(cur.__next__(), ("row4",))
                self.assertEqual(cur.__next__(), ("row5",))
                with self.assertRaises(StopIteration):
                    cur.__next__()

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_cleanup_rows_buffered(self) -> None:
        test = "CLEANUP ROWS BUFFERED"
        self.log_start(test)

        with self.setupForCursor() as conn:
            ##################################################################
            with conn.cursor() as cur:
                cur.execute(f"select * from {self.table} as t1, {self.table} as t2")
                for counter, _ in enumerate(cur):
                    if counter > 10:
                        break

            ##################################################################
            with conn.cursor() as cur2:
                cur2.execute("select 1")
                self.assertEqual(cur2.fetchone(), (1,))
                self.assertIsNone(cur2.fetchone())

            ##################################################################
            self.drop(conn)
        self.log_ended(test)

    def test_cleanup_rows_unbuffered(self) -> None:
        test = "CLEANUP ROWS UNBUFFERED"
        self.log_start(test)

        with self.setupForCursor() as conn:
            ##################################################################
            with conn.cursor(SSCursor) as cur:
                cur.execute(f"select * from {self.table} as t1, {self.table} as t2")
                for counter, _ in enumerate(cur):
                    if counter > 10:
                        break

            ##################################################################
            with conn.cursor(SSCursor) as cur2:
                cur2.execute("select 1")
                self.assertEqual(cur2.fetchone(), (1,))
                self.assertIsNone(cur2.fetchone())

            ##################################################################
            self.drop(conn)
        self.log_ended(test)

    def test_execute_args(self) -> None:
        test = "EXECUTE ARGUMENTS"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                # . create table
                cur.execute(f"create table {self.table} (i int, j int)")

                # . insert one element
                # . argument 1 should automatically be escaped into
                # . str '1' and formatted into the sql as one row values.
                cur.execute(f"insert into {self.table} (i) values (%s)", 1)
                self.assertEqual(cur.executed_sql.endswith(b"values (1)"), True)
                cur.execute(f"select i from {self.table}")
                self.assertEqual(cur.fetchone(), (1,))
                self.delete(conn)

                # . insert one row
                # . argument (1, 2) should automacially be escaped into
                # . tuple of str ('1', '2') and formatted into the sql as
                # . one row values.
                cur.execute(f"insert into {self.table} (i, j) values (%s, %s)", (1, 2))
                self.assertEqual(cur.executed_sql.endswith(b"values (1, 2)"), True)
                cur.execute(f"select * from {self.table}")
                self.assertEqual(cur.fetchone(), (1, 2))
                self.delete(conn)

                # . insert many
                # . argument range(10) should be escaped into
                # . tuple of str ('0', '1', ..., '9') but instead of being
                # . formatted into the sql as one row, it should be formatted
                # . as multiple rows values.
                cur.executemany(
                    f"insert into {self.table} (i) values (%s)",
                    list(range(10)),
                )
                self.assertEqual(
                    cur.executed_sql.endswith(
                        b"values (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)"
                    ),
                    True,
                )
                cur.execute(f"select i from {self.table}")
                self.assertEqual(cur.fetchall(), tuple((i,) for i in range(10)))
                self.delete(conn)

                # . insert many
                # . many should automatically set itemize to True.
                cur.execute(
                    f"insert into {self.table} (i) values (%s)",
                    list(range(10)),
                    many=True,
                    itemize=False,
                )
                self.assertEqual(
                    cur.executed_sql.endswith(
                        b"values (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)"
                    ),
                    True,
                )
                cur.execute(f"select i from {self.table}")
                self.assertEqual(cur.fetchall(), tuple((i,) for i in range(10)))
                self.delete(conn)

                # . itemize=False
                # . argument [(1, 2), (3, 4), (5, 6)] should be escaped into
                # . as a single string '(1, 2), (3, 4), (5, 6)' and formatted
                # . into the sql directly.
                cur.execute(
                    f"insert into {self.table} (i, j) values %s",
                    [(1, 2), (3, 4), (5, 6)],
                    itemize=False,
                )
                self.assertEqual(
                    cur.executed_sql.endswith(b"values (1,2),(3,4),(5,6)"), True
                )
                cur.execute(f"select * from {self.table}")
                self.assertEqual(cur.fetchall(), ((1, 2), (3, 4), (5, 6)))
                self.delete(conn)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_executemany(self) -> None:
        test = "EXECUTE MANY"
        self.log_start(test)

        with self.setupForCursor() as conn:
            # Due to complie, INSERT_VALUES_RE can't not be imported directly.
            # The following regex is the exact same one used by the library.
            INSERT_VALUES_RE: re.Pattern = re.compile(
                r"\s*((?:INSERT|REPLACE)\b.+\bVALUES?\s*)"  # prefix: INSERT INTO ... VALUES
                + r"(\(\s*(?:%s|%\(.+\)s)\s*(?:,\s*(?:%s|%\(.+\)s)\s*)*\))"  # values: (%s, %s, ...)
                + r"(\s*(?:AS\s.*?)?\s*(?:ON DUPLICATE.*)?);?\s*\Z",  # suffix: AS ... ON DUPLICATE ...
                re.IGNORECASE | re.DOTALL,
            )

            ##################################################################
            m = INSERT_VALUES_RE.match("INSERT INTO TEST (ID, NAME) VALUES (%s, %s)")
            self.assertIsNotNone(m, "error parse %s")
            self.assertEqual(
                m.group(3), "", "group 3 not blank, bug in RE_INSERT_VALUES?"
            )

            ##################################################################
            m = INSERT_VALUES_RE.match(
                "INSERT INTO TEST (ID, NAME) VALUES (%(id)s, %(name)s)"
            )
            self.assertIsNotNone(m, "error parse %(name)s")
            self.assertEqual(
                m.group(3), "", "group 3 not blank, bug in RE_INSERT_VALUES?"
            )

            ##################################################################
            m = INSERT_VALUES_RE.match(
                "INSERT INTO TEST (ID, NAME) VALUES (%(id_name)s, %(name)s)"
            )
            self.assertIsNotNone(m, "error parse %(id_name)s")
            self.assertEqual(
                m.group(3), "", "group 3 not blank, bug in RE_INSERT_VALUES?"
            )

            ##################################################################
            m = INSERT_VALUES_RE.match(
                "INSERT INTO TEST (ID, NAME) VALUES (%(id_name)s, %(name)s) AS v"
            )
            self.assertIsNotNone(m, "error parse %(id_name)s")
            self.assertEqual(
                m.group(3),
                " AS v",
                "group 3 not AS v, bug in RE_INSERT_VALUES?",
            )

            ##################################################################
            m = INSERT_VALUES_RE.match(
                "INSERT INTO TEST (ID, NAME) VALUES (%(id_name)s, %(name)s) ON duplicate update"
            )
            self.assertIsNotNone(m, "error parse %(id_name)s")
            self.assertEqual(
                m.group(3),
                " ON duplicate update",
                "group 3 not ON duplicate update, bug in RE_INSERT_VALUES?",
            )

            ##################################################################
            m = INSERT_VALUES_RE.match(
                "INSERT INTO TEST (ID, NAME) VALUES (%(id_name)s, %(name)s) AS v ON duplicate update"
            )
            self.assertIsNotNone(m, "error parse %(id_name)s")
            self.assertEqual(
                m.group(3),
                " AS v ON duplicate update",
                "group 3 not AS v ON duplicate update, bug in RE_INSERT_VALUES?",
            )

            # https://github.com/PyMySQL/PyMySQL/pull/597
            m = INSERT_VALUES_RE.match("INSERT INTO bloup(foo, bar)VALUES(%s, %s)")
            assert m is not None

            # cursor._executed must bee "insert into test (data)
            #  values (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)"
            # list args
            with conn.cursor() as cur:
                cur.executemany(
                    f"insert into {self.table} (data) values (%s)", list(range(10))
                )
                self.assertTrue(
                    cur.executed_sql.endswith(b",(7),(8),(9)"),
                    "execute many with %s not in one query",
                )
            self.drop(conn)

            # %% in column set
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.db}.percent_test")
                cur.execute(
                    f"""\
                    CREATE TABLE {self.db}.percent_test (
                        `A%` INTEGER,
                        `B%` INTEGER)"""
                )
                sql = (
                    f"INSERT INTO {self.db}.percent_test (`A%%`, `B%%`) VALUES (%s, %s)"
                )
                self.assertIsNotNone(INSERT_VALUES_RE.match(sql))
                cur.executemany(sql, [(3, 4), (5, 6)])
                self.assertTrue(
                    cur.executed_sql.endswith(b"(3, 4),(5, 6)"),
                    "executemany with %% not in one query",
                )
                cur.execute(f"DROP TABLE {self.db}.percent_test")
        self.log_ended(test)

    def test_execution_time_limit(self) -> None:
        test = "EXECUTION TIME LIMIT"
        self.log_start(test)

        with self.setupForCursor() as conn:
            with conn.cursor() as cur:
                ##################################################################
                # MySQL MAX_EXECUTION_TIME takes ms
                # MariaDB max_statement_time takes seconds as int/float, introduced in 10.1

                # this will sleep 0.01 seconds per row
                if conn.server_vendor == "mysql":
                    sql = f"SELECT /*+ MAX_EXECUTION_TIME(2000) */ data, sleep(0.01) FROM {self.table}"
                else:
                    sql = f"SET STATEMENT max_statement_time=2 FOR SELECT data, sleep(0.01) FROM {self.table}"
                cur.execute(sql)
                rows = cur.fetchall()
                self.assertEqual(
                    rows,
                    (
                        ("row1", 0),
                        ("row2", 0),
                        ("row3", 0),
                        ("row4", 0),
                        ("row5", 0),
                    ),
                )

                if conn.server_vendor == "mysql":
                    sql = f"SELECT /*+ MAX_EXECUTION_TIME(2000) */ data, sleep(0.01) FROM {self.table}"
                else:
                    sql = f"SET STATEMENT max_statement_time=2 FOR SELECT data, sleep(0.01) FROM {self.table}"
                cur.execute(sql)
                row = cur.fetchone()
                self.assertEqual(row, ("row1", 0))

                # this discards the previous unfinished query
                cur.execute("SELECT 1")
                self.assertEqual(cur.fetchone(), (1,))

                if conn.server_vendor == "mysql":
                    sql = f"SELECT /*+ MAX_EXECUTION_TIME(1) */ data, sleep(1) FROM {self.table}"
                else:
                    sql = f"SET STATEMENT max_statement_time=0.001 FOR SELECT data, sleep(1) FROM {self.table}"
                with self.assertRaises(errors.OperationalError) as cm:
                    # in a buffered cursor this should reliably raise an
                    # OperationalError
                    cur.execute(sql)
                    if conn.server_vendor == "mysql":
                        # this constant was only introduced in MySQL 5.7, not sure
                        # what was returned before, may have been ER_QUERY_INTERRUPTED
                        self.assertEqual(cm.exception.args[0], ER.QUERY_TIMEOUT)
                    else:
                        self.assertEqual(cm.exception.args[0], ER.STATEMENT_TIMEOUT)

                # connection should still be fine at this point
                cur.execute("SELECT 1")
                self.assertEqual(cur.fetchone(), (1,))

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_warnings(self) -> None:
        test = "WARNINGS"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                ##################################################################
                cur.execute(f"DROP TABLE IF EXISTS {self.db}.no_exists_table")
                self.assertEqual(cur.warning_count, 1)

                ##################################################################
                cur.execute("SHOW WARNINGS")
                row = cur.fetchone()
                self.assertEqual(row[1], ER.BAD_TABLE_ERROR)
                self.assertIn("no_exists_table", row[2])

                ##################################################################
                cur.execute("SELECT 1")
                self.assertEqual(cur.warning_count, 0)

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_SSCursor(self) -> None:
        test = "CURSOR UNBUFFERED"
        self.log_start(test)

        with self.setup(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            with conn.cursor(SSCursor) as cur:
                # . create table
                cur.execute(
                    f"CREATE TABLE {self.table} "
                    "(region VARCHAR(64), zone VARCHAR(64), name VARCHAR(64))"
                )

                # . test INSERT
                data = [
                    ("America", "", "America/Jamaica"),
                    ("America", "", "America/Los_Angeles"),
                    ("America", "", "America/Lima"),
                    ("America", "", "America/New_York"),
                    ("America", "", "America/Menominee"),
                    ("America", "", "America/Havana"),
                    ("America", "", "America/El_Salvador"),
                    ("America", "", "America/Costa_Rica"),
                    ("America", "", "America/Denver"),
                    ("America", "", "America/Detroit"),
                ]
                conn.begin()
                for row in data:
                    cur.execute(f"INSERT INTO {self.table} VALUES (%s, %s, %s)", row)
                    self.assertEqual(
                        conn.affected_rows, 1, "affected_rows does not match"
                    )
                conn.commit()

                # Test fetchone()
                index = 0
                cur.execute(f"SELECT * FROM {self.table}")
                while True:
                    row = cur.fetchone()
                    if row is None:
                        break
                    index += 1

                    # Test cursor.affected_rows
                    affected_rows = 18446744073709551615
                    self.assertEqual(
                        cur.affected_rows,
                        affected_rows,
                        "cursor.affected_rows != %s" % (str(affected_rows)),
                    )

                    # Test cursor.row_number
                    self.assertEqual(
                        cur.rownumber, index, "cursor.row_number != %s" % (str(index))
                    )

                    # Test row came out the same as it went in
                    self.assertEqual(
                        (row in data), True, "Row not found in source data"
                    )

                # Test fetchall
                cur.execute(f"SELECT * FROM {self.table}")
                self.assertEqual(
                    len(cur.fetchall()),
                    len(data),
                    "fetchall failed. Number of rows does not match",
                )

                # Test fetchmany
                cur.execute(f"SELECT * FROM {self.table}")
                self.assertEqual(
                    len(cur.fetchmany(2)),
                    2,
                    "fetchmany(2) failed. Number of rows does not match",
                )
                cur.fetchall()

                # Test update, affected_rows()
                cur.execute(f"UPDATE {self.table} SET zone = %s", "Foo")
                conn.commit()
                self.assertEqual(
                    cur.affected_rows,
                    len(data),
                    "Update failed cursor.affected_rows != %s" % (str(len(data))),
                )

                # Test executemany
                cur.executemany(f"INSERT INTO {self.table} VALUES (%s, %s, %s)", data)
                self.assertEqual(
                    cur.affected_rows,
                    len(data),
                    "execute many failed. cursor.affected_rows != %s"
                    % (str(len(data))),
                )

                # Test multiple datasets
                cur.execute("SELECT 1; SELECT 2; SELECT 3")
                self.assertListEqual(list(cur), [(1,)])
                self.assertTrue(cur.nextset())
                self.assertListEqual(list(cur), [(2,)])
                self.assertTrue(cur.nextset())
                self.assertListEqual(list(cur), [(3,)])
                self.assertFalse(cur.nextset())

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_DictCursor(self, unbuffered: bool = False) -> None:
        if unbuffered:
            test = "DICT CURSOR UNBUFFERED"
        else:
            test = "DICT CURSOR"
        self.log_start(test)

        with self.setupForDictCursor() as conn:
            with conn.cursor(SSDictCursor if unbuffered else DictCursor) as cur:
                bob, jim, fred = self.bob.copy(), self.jim.copy(), self.fred.copy()
                # try an update which should return no rows
                cur.execute(f"update {self.table} set age=20 where name='bob'")
                bob["age"] = 20
                # pull back the single row dict for bob and check
                cur.execute(f"SELECT * from {self.table} where name='bob'")
                row = cur.fetchone()
                self.assertEqual(bob, row, "fetchone via DictCursor failed")
                if unbuffered:
                    cur.fetchall()

                # same again, but via fetchall => tuple(row)
                cur.execute(f"SELECT * from {self.table} where name='bob'")
                row = cur.fetchall()
                self.assertEqual(
                    (bob,),
                    row,
                    "fetch a 1 row result via fetchall failed via DictCursor",
                )

                # same test again but iterate over the
                cur.execute(f"SELECT * from {self.table} where name='bob'")
                for row in cur:
                    self.assertEqual(
                        bob,
                        row,
                        "fetch a 1 row result via iteration failed via DictCursor",
                    )

                # get all 3 row via fetchall
                cur.execute(f"SELECT * from {self.table}")
                rows = cur.fetchall()
                self.assertEqual(
                    (bob, jim, fred), rows, "fetchall failed via DictCursor"
                )

                # same test again but do a list comprehension
                cur.execute(f"SELECT * from {self.table}")
                rows = list(cur)
                self.assertEqual(
                    [bob, jim, fred], rows, "DictCursor should be iterable"
                )

                # get all 2 row via fetchmany() and iterate the last one
                cur.execute(f"SELECT * from {self.table}")
                rows = cur.fetchmany(2)
                self.assertEqual((bob, jim), rows, "fetchmany failed via DictCursor")
                for row in cur:
                    self.assertEqual(
                        fred,
                        row,
                        "fetch a 1 row result via iteration failed via DictCursor",
                    )
                if unbuffered:
                    cur.fetchall()

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_DfCursor(self, unbuffered: bool = False) -> None:
        if unbuffered:
            test = "DATAFRAME CURSOR UNBUFFERED"
        else:
            test = "DATAFRAME CURSOR"
        self.log_start(test)

        with self.setupForDictCursor() as conn:
            with conn.cursor(SSDfCursor if unbuffered else DfCursor) as cur:
                cur: DfCursor
                df = pd.DataFrame([self.bob, self.jim, self.fred])
                # try an update which should return no rows
                cur.execute(f"update {self.table} set age=20 where name='bob'")
                df.loc[df["name"] == "bob", "age"] = 20
                # pull back the single row dict for bob and check
                cur.execute(f"SELECT * from {self.table} where name='bob'")
                row = cur.fetchone()
                assert row.equals(df.iloc[0:1])
                if unbuffered:
                    cur.fetchall()

                # same again, but via fetchall => tuple(row)
                cur.execute(f"SELECT * from {self.table} where name='bob'")
                row = cur.fetchall()
                assert row.equals(df.iloc[0:1])

                # same test again but iterate over the
                cur.execute(f"SELECT * from {self.table} where name='bob'")
                for i, row in enumerate(cur):
                    assert row.equals(df.iloc[i : i + 1])

                # get all 3 row via fetchall
                cur.execute(f"SELECT * from {self.table}")
                rows = cur.fetchall()
                assert rows.equals(df)

                # get all 2 row via fetchmany(2) and iterate the last one
                cur.execute(f"SELECT * from {self.table}")
                rows = cur.fetchmany(2)
                assert rows.equals(df.iloc[0:2])
                for row in cur:
                    assert row.equals(df.iloc[2:3].reset_index(drop=True))
                if unbuffered:
                    cur.fetchall()

                ##################################################################
                self.drop(conn)
        self.log_ended(test)

    def test_next_set(self) -> None:
        test = "NEXT SET"
        self.log_start(test)

        with self.setup(
            init_command='SELECT "bar"; SELECT "baz"',
            client_flag=CLIENT.MULTI_STATEMENTS,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1; SELECT 2;")
                self.assertEqual([(1,)], list(cur))
                res = cur.nextset()
                self.assertEqual(res, True)
                self.assertEqual([(2,)], list(cur))
                self.assertEqual(cur.nextset(), False)
        self.log_ended(test)

    def test_skip_next_set(self) -> None:
        test = "SKIP NEXT SET"
        self.log_start(test)

        with self.setup(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1; SELECT 2;")
                self.assertEqual([(1,)], list(cur))

                cur.execute("SELECT 42")
                self.assertEqual([(42,)], list(cur))
        self.log_ended(test)

    def test_next_set_error(self) -> None:
        test = "NEXT SET ERROR"
        self.log_start(test)

        with self.setup(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            with conn.cursor() as cur:
                for i in range(3):
                    cur.execute("SELECT %s; xyzzy;", (i,))
                    self.assertEqual([(i,)], list(cur))
                    with self.assertRaises(errors.ProgrammingError):
                        cur.nextset()
                    self.assertEqual((), cur.fetchall())
        self.log_ended(test)

    def test_ok_and_next(self):
        test = "OK AND NEXT"
        self.log_start(test)

        with self.setup(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1; commit; SELECT 2;")
                self.assertEqual([(1,)], list(cur))
                self.assertTrue(cur.nextset())
                self.assertTrue(cur.nextset())
                self.assertEqual([(2,)], list(cur))
                self.assertFalse(cur.nextset())
        self.log_ended(test)

    def test_multi_statement_warnings(self):
        test = "MULTI STATEMENT WARNINGS"
        self.log_start(test)

        conn = self.setup(
            init_command='SELECT "bar"; SELECT "baz"',
            client_flag=CLIENT.MULTI_STATEMENTS,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DROP TABLE IF EXISTS {self.db}.a; DROP TABLE IF EXISTS {self.db}.b;"
                )
            self.log_ended(test)
        except TypeError:
            self.fail()
        finally:
            conn.close()

    def test_previous_cursor_not_closed(self):
        test = "PREVIOUS CURSOR NOT CLOSED"
        self.log_start(test)

        with self.get_conn(
            init_command='SELECT "bar"; SELECT "baz"',
            client_flag=CLIENT.MULTI_STATEMENTS,
        ) as conn:
            with conn.cursor() as cur1:
                cur1.execute("SELECT 1; SELECT 2")
                with conn.cursor() as cur2:
                    cur2.execute("SELECT 3")
                    self.assertEqual(cur2.fetchone()[0], 3)
        self.log_ended(test)

    def test_commit_during_multi_result(self):
        test = "COMMIT DURING MULTI RESULT"
        self.log_start(test)

        with self.get_conn(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1; SELECT 2")
                conn.commit()
                cur.execute("SELECT 3")
                self.assertEqual(cur.fetchone()[0], 3)
        self.log_ended(test)

    def test_transaction(self):
        test = "TRANSACTION"
        self.log_start(test)

        with self.setup() as conn:
            with conn.transaction() as cur:
                cur.execute(
                    f"create table {self.table} (name char(20), age int, DOB datetime)"
                )
                cur.executemany(
                    f"insert into {self.table} values (%s, %s, %s)",
                    [
                        ("bob", 21, "1990-02-06 23:04:56"),
                        ("jim", 56, "1955-05-09 13:12:45"),
                        ("fred", 100, "1911-09-12 01:01:01"),
                    ],
                )

            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {self.table}")
                rows = cur.fetchall()
                self.assertEqual(len(rows), 3)
                self.assertEqual(
                    rows[0], ("bob", 21, datetime.datetime(1990, 2, 6, 23, 4, 56))
                )
                self.assertEqual(
                    rows[1], ("jim", 56, datetime.datetime(1955, 5, 9, 13, 12, 45))
                )
                self.assertEqual(
                    rows[2], ("fred", 100, datetime.datetime(1911, 9, 12, 1, 1, 1))
                )

            self.drop(conn)

            with self.assertRaises(RuntimeError):
                with conn.transaction() as cur:
                    raise RuntimeError("test")
            self.assertTrue(cur.closed())
            self.assertTrue(conn.closed())

        self.log_ended(test)

    def test_scroll(self):
        test = "SCROLL"
        self.log_start(test)

        with self.setupForCursor() as conn:
            # Buffered #########################################################
            with conn.cursor() as cur:
                cur.execute(f"select * from {self.table}")
                with self.assertRaises(errors.ProgrammingError):
                    cur.scroll(1, "foo")

            with conn.cursor() as cur:
                cur.execute(f"select * from {self.table}")
                cur.scroll(1, "relative")
                self.assertEqual(cur.fetchone(), ("row2",))
                cur.scroll(2, "relative")
                self.assertEqual(cur.fetchone(), ("row5",))

            with conn.cursor() as cur:
                cur.execute(f"select * from {self.table}")
                cur.scroll(2, mode="absolute")
                self.assertEqual(cur.fetchone(), ("row3",))
                cur.scroll(4, mode="absolute")
                self.assertEqual(cur.fetchone(), ("row5",))

            with conn.cursor() as cur:
                cur.execute(f"select * from {self.table}")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    cur.scroll(5, "relative")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    cur.scroll(-1, "relative")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    cur.scroll(-5, "absolute")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    cur.scroll(-1, "absolute")

            # Unbuffered #######################################################
            with conn.cursor(SSCursor) as cur:
                cur.execute(f"select * from {self.table}")
                with self.assertRaises(errors.ProgrammingError):
                    cur.scroll(1, "foo")

            with conn.cursor(SSCursor) as cur:
                cur.execute(f"select * from {self.table}")
                cur.scroll(1, "relative")
                self.assertEqual(cur.fetchone(), ("row2",))
                cur.scroll(2, "relative")
                self.assertEqual(cur.fetchone(), ("row5",))

            with conn.cursor(SSCursor) as cur:
                cur.execute(f"select * from {self.table}")
                cur.scroll(2, mode="absolute")
                self.assertEqual(cur.fetchone(), ("row3",))
                cur.scroll(4, mode="absolute")
                self.assertEqual(cur.fetchone(), ("row5",))

            with conn.cursor(SSCursor) as cur:
                cur.execute(f"select * from {self.table}")
                cur.scroll(1, "relative")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    cur.scroll(-1, "relative")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    cur.scroll(0, "absolute")
                cur.scroll(10)
                self.assertEqual(cur.fetchone(), None)
                self.assertEqual(cur.rownumber, 5)

            # ##################################################################
            self.drop(conn)
        self.log_ended(test)

    def test_procedure(self):
        test = "PROCEDURE"
        self.log_start(test)

        with self.setup() as conn:
            conn.select_database(self.db)
            # Buffered #########################################################
            with conn.cursor() as cur:
                cur.execute("DROP PROCEDURE IF EXISTS myinc;")
                cur.execute(
                    """
                    CREATE PROCEDURE myinc(p1 INT, p2 INT)
                    BEGIN
                        SELECT p1 + p2 + 1;
                    END"""
                )
                conn.commit()

            with conn.cursor() as cur:
                cur.callproc("myinc", (1, 2))
                res = cur.fetchone()
                self.assertEqual(res, (4,))

            with self.assertRaises(errors.ProgrammingError):
                cur.callproc("myinc", [1, 2])

            with conn.cursor() as cur:
                cur.execute("DROP PROCEDURE IF EXISTS myinc;")

            # Unbuffered #######################################################
            with conn.cursor(SSCursor) as cur:
                cur.execute("DROP PROCEDURE IF EXISTS myinc;")
                cur.execute(
                    """
                    CREATE PROCEDURE myinc(p1 INT, p2 INT)
                    BEGIN
                        SELECT p1 + p2 + 1;
                    END"""
                )
                conn.commit()

            with conn.cursor(SSCursor) as cur:
                cur.callproc("myinc", (1, 2))
                res = cur.fetchone()
                self.assertEqual(res, (4,))

            with self.assertRaises(errors.ProgrammingError):
                cur.callproc("myinc", [1, 2])

            with conn.cursor(SSCursor) as cur:
                cur.execute("DROP PROCEDURE IF EXISTS myinc;")

        self.log_ended(test)

    # utils
    def setupForCursor(self, table: str = None, **kwargs) -> Connection:
        conn = self.setup(table, **kwargs)
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute(f"create table {self.db}.{tb} (data varchar(10))")
            cur.executemany(
                f"insert into {self.db}.{tb} values (%s)",
                ["row%d" % i for i in range(1, 6)],
            )
        conn.commit()
        return conn

    def setupForDictCursor(self, table: str = None, **kwargs) -> Connection:
        conn = self.setup(table, **kwargs)
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute(
                f"create table {self.db}.{tb} (name char(20), age int, DOB datetime)"
            )
            cur.executemany(
                f"insert into {self.db}.{tb} values (%s, %s, %s)",
                [
                    ("bob", 21, "1990-02-06 23:04:56"),
                    ("jim", 56, "1955-05-09 13:12:45"),
                    ("fred", 100, "1911-09-12 01:01:01"),
                ],
            )
        conn.commit()
        return conn


class TestLoadLocal(TestCase):
    name: str = "Load Local"

    def test_all(self) -> None:
        self.test_no_file()
        self.test_load_file(False)
        self.test_load_file(True)
        self.test_load_warnings()

    def test_no_file(self):
        test = "NO FILE"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                with self.assertRaises(errors.OperationalError):
                    cur.execute(
                        "LOAD DATA LOCAL INFILE 'no_data.txt' INTO TABLE "
                        "test_load_local fields terminated by ','"
                    )
            self.drop(conn)
        self.log_ended(test)

    def test_load_file(self, unbuffered: bool = False):
        test = "LOAD FILE"
        self.log_start(test)

        if unbuffered:
            test += " UNBUFFERED"
        with self.setup() as conn:
            with conn.cursor(SSCursor if unbuffered else SSCursor) as cur:
                filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "test_data",
                    "load_local_data.txt",
                )
                cur.execute(
                    f"LOAD DATA LOCAL INFILE '{filename}' INTO TABLE {self.table}"
                    " FIELDS TERMINATED BY ','"
                )
                cur.execute(f"SELECT COUNT(*) FROM {self.table}")
                self.assertEqual(22749, cur.fetchone()[0])

            self.drop(conn)
        self.log_ended(test)

    def test_load_warnings(self):
        test = "LOAD WARNINGS"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "test_data",
                    "load_local_warn_data.txt",
                )
                cur.execute(
                    f"LOAD DATA LOCAL INFILE '{filename}' INTO TABLE "
                    f"{self.table} FIELDS TERMINATED BY ','"
                )
                self.assertEqual(1, cur.warning_count)

                cur.execute("SHOW WARNINGS")
                row = cur.fetchone()

                self.assertEqual(ER.TRUNCATED_WRONG_VALUE_FOR_FIELD, row[1])
                self.assertIn(
                    "incorrect integer value",
                    row[2].lower(),
                )

            self.drop(conn)
        self.log_ended(test)

    # . utils
    def setup(self, table: str = None, **kwargs) -> Connection:
        conn = super().setup(table, **kwargs)
        tb = self.tb if table is None else table
        with conn.cursor() as cur:
            cur.execute("SET GLOBAL local_infile=ON")
            cur.execute(f"CREATE TABLE {self.db}.{tb} (a INTEGER, b INTEGER)")
        conn.commit()
        return conn


class TestOptionFile(TestCase):
    name: str = "Option File"

    def test_all(self) -> None:
        self.test_option_file()

    def test_option_file(self) -> None:
        from sqlcycli._optionfile import OptionFile

        test = "PARSE OPTIONS"
        self.log_start(test)

        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data", "my.cnf"
        )
        opt = OptionFile(filename, "client")
        self.assertEqual("client", opt.opt_group)
        self.assertEqual("localhost", opt.host)
        self.assertEqual(3306, opt.port)
        self.assertEqual("myuser", opt.user)
        self.assertEqual("mypassword", opt.password)
        self.assertEqual("mydatabase", opt.database)
        self.assertEqual("utf8mb4", opt.charset)
        self.assertEqual("127.0.0.1", opt.bind_address)
        self.assertEqual("/var/run/mysqld/mysqld.sock", opt.unix_socket)
        self.assertEqual("16M", opt.max_allowed_packet)
        self.log_ended(test)


class TestErrors(TestCase):
    name: str = "Errors"

    def test_all(self) -> None:
        self.test_raise_mysql_exception()

    def test_raise_mysql_exception(self) -> None:
        from sqlcycli.errors import raise_mysql_exception

        test = "RAISE MYSQL EXCEPTION"
        self.log_start(test)

        data = b"\xff\x15\x04#28000Access denied"
        with self.assertRaises(errors.OperationalError) as cm:
            raise_mysql_exception(data, len(data))
            assert type(cm.exception) == errors.OperationalError
            assert cm.exception.args[0] == ER.ACCESS_DENIED_ERROR
            assert cm.exception.args[1] == "Access denied"

        data = b"\xff\x10\x04Too many connections"
        with self.assertRaises(errors.OperationalError) as cm:
            raise_mysql_exception(data, len(data))
            assert type(cm.exception) == errors.OperationalError
            assert cm.exception.args[0] == ER.TOO_MANY_USER_CONNECTIONS
            assert cm.exception.args[1] == "Too many connections"

        self.log_ended(test)


class TestOldIssues(TestCase):
    name: str = "Old Issues"

    def test_all(self) -> None:
        self.test_issue_3()
        self.test_issue_4()
        self.test_issue_5()
        self.test_issue_6()
        self.test_issue_8()
        self.test_issue_13()
        self.test_issue_15()
        self.test_issue_16()

    def test_issue_3(self) -> None:
        """undefined methods datetime_or_None, date_or_None"""
        test = "ISSUE 3"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"create table {self.table} (d date, t time, dt datetime, ts timestamp)"
                )
                cur.execute(
                    f"insert into {self.table} (d, t, dt, ts) values (%s,%s,%s,%s)",
                    (None, None, None, None),
                )
                cur.execute(f"select d from {self.table}")
                self.assertEqual(None, cur.fetchone()[0])
                cur.execute(f"select t from {self.table}")
                self.assertEqual(None, cur.fetchone()[0])
                cur.execute(f"select dt from {self.table}")
                self.assertEqual(None, cur.fetchone()[0])
                cur.execute(f"select ts from {self.table}")
                self.assertIn(
                    type(cur.fetchone()[0]),
                    (type(None), datetime.datetime),
                    "expected Python type None or datetime from SQL timestamp",
                )

            self.drop(conn)
        self.log_ended(test)

    def test_issue_4(self):
        """can't retrieve TIMESTAMP fields"""
        test = "ISSUE 4"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(f"create table {self.table} (ts timestamp)")
                cur.execute(f"insert into {self.table} (ts) values (now())")
                cur.execute(f"select ts from {self.table}")
                self.assertTrue(isinstance(cur.fetchone()[0], datetime.datetime))

            self.drop(conn)
        self.log_ended(test)

    def test_issue_5(self):
        """query on information_schema.tables fails"""
        test = "ISSUE 5"
        self.log_start(test)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select * from information_schema.tables")
        self.log_ended(test)

    def test_issue_6(self):
        """exception: TypeError: ord() expected a character, but string of length 0 found"""
        # ToDo: this test requires access to db 'mysql'.
        test = "ISSUE 6"
        self.log_start(test)

        with self.get_conn(database="mysql") as conn:
            with conn.cursor() as cur:
                cur.execute("select * from user")
        self.log_ended(test)

    def test_issue_8(self):
        """Primary Key and Index error when selecting data"""
        test = "ISSUE 8"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE {self.table} (`station` int NOT NULL DEFAULT '0', `dh`
                    datetime NOT NULL DEFAULT '2015-01-01 00:00:00', `echeance` int NOT NULL
                    DEFAULT '0', `me` double DEFAULT NULL, `mo` double DEFAULT NULL, PRIMARY
                    KEY (`station`,`dh`,`echeance`)) ENGINE=MyISAM DEFAULT CHARSET=latin1;
                    """
                )
                self.assertEqual(0, cur.execute(f"SELECT * FROM {self.table}"))
                cur.execute(
                    f"ALTER TABLE {self.table} ADD INDEX `idx_station` (`station`)"
                )
                self.assertEqual(0, cur.execute(f"SELECT * FROM {self.table}"))

            self.drop(conn)
        self.log_ended(test)

    def test_issue_13(self):
        """can't handle large result fields"""
        test = "ISSUE 13"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(f"create table {self.table} (t text)")
                # ticket says 18k
                size = 18 * 1024
                cur.execute(f"insert into {self.table} (t) values (%s)", ("x" * size,))
                cur.execute(f"select t from {self.table}")
                # use assertTrue so that obscenely huge error messages don't print
                r = cur.fetchone()[0]
                self.assertTrue("x" * size == r)

            self.drop(conn)
        self.log_ended(test)

    def test_issue_15(self):
        """query should be expanded before perform character encoding"""
        test = "ISSUE 15"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(f"create table {self.table} (t varchar(32))")
                cur.execute(
                    f"insert into {self.table} (t) values (%s)", ("\xe4\xf6\xfc",)
                )
                cur.execute(f"select t from {self.table}")
                self.assertEqual("\xe4\xf6\xfc", cur.fetchone()[0])

            self.drop(conn)
        self.log_ended(test)

    def test_issue_16(self):
        """Patch for string and tuple escaping"""
        test = "ISSUE 16"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"create table {self.table} (name varchar(32) primary key, email varchar(32))"
                )
                cur.execute(
                    f"insert into {self.table} (name, email) values ('pete', 'floydophone')"
                )
                cur.execute(f"select email from {self.table} where name=%s", ("pete",))
                self.assertEqual("floydophone", cur.fetchone()[0])

            self.drop(conn)
        self.log_ended(test)


class TestNewIssues(TestCase):
    name: str = "New Issues"

    def test_all(self) -> None:
        self.test_issue_33()
        self.test_issue_34()
        self.test_issue_36()
        self.test_issue_37()
        self.test_issue_38()
        self.test_issue_54()

    def test_issue_33(self):
        test = "ISSUE 33"
        self.log_start(test)

        with self.get_conn() as conn:
            table = f"{self.db}.hei\xdfe"
            with conn.cursor() as cur:
                cur.execute(f"drop table if exists {table}")
                cur.execute(f"create table {table} (name varchar(32))")
                cur.execute(f"insert into {table} (name) values ('Pi\xdfata')")
                cur.execute(f"select name from {table}")
                self.assertEqual("Pi\xdfata", cur.fetchone()[0])
                cur.execute(f"drop table {table}")
        self.log_ended(test)

    def test_issue_34(self):
        test = "ISSUE 34"
        self.log_start(test)

        try:
            self.get_conn(port=1237)
            self.fail()
        except errors.OperationalError as err:
            self.assertEqual(2003, err.args[0])
            self.log_ended(test)
        except Exception:
            self.fail()

    def test_issue_36(self):
        test = "ISSUE 36"
        self.log_start(test)

        conn1 = self.get_conn()
        conn2 = self.get_conn()
        try:
            with conn1.cursor() as cur:
                kill_id = None
                cur.execute("show processlist")
                for row in cur:
                    if row[7] == "show processlist":
                        kill_id = row[0]
                        break
                self.assertEqual(kill_id, conn1.thread_id)
                # now nuke the connection
                conn2.kill(kill_id)
                # make sure this connection has broken
                with self.assertRaises(errors.OperationalError) as cm:
                    cur.execute("show tables")
                    self.assertEqual(2013, cm.exception.args[0])

            # check the process list from the other connection
            time.sleep(0.1)
            with conn2.cursor() as cur:
                cur.execute("show processlist")
                ids = [row[0] for row in cur.fetchall()]
                self.assertFalse(kill_id in ids)

            self.log_ended(test)

        finally:
            conn1.close()
            conn2.close()

    def test_issue_37(self):
        test = "ISSUE 37"
        self.log_start(test)

        with self.get_conn() as conn:
            with conn.cursor() as cur:
                self.assertEqual(1, cur.execute("SELECT @foo"))
                self.assertEqual((None,), cur.fetchone())
                self.assertEqual(0, cur.execute("SET @foo = 'bar'"))
                cur.execute("set @foo = 'bar'")
        self.log_ended(test)

    def test_issue_38(self):
        test = "ISSUE 38"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                datum = (
                    "a" * 1024 * 1023
                )  # reduced size for most default mysql installs
                cur.execute(f"create table {self.table} (id integer, data mediumblob)")
                cur.execute(f"insert into {self.table} values (1, %s)", (datum,))

            self.drop(conn)
        self.log_ended(test)

    def test_issue_54(self):
        test = "ISSUE 54"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(f"create table {self.table} (id integer primary key)")
                cur.execute(f"insert into {self.table} (id) values (7)")
                cur.execute(
                    "select * from %s where %s"
                    % (
                        self.table,
                        " and ".join("%d=%d" % (i, i) for i in range(0, 100000)),
                    )
                )
                self.assertEqual(7, cur.fetchone()[0])

            self.drop(conn)
        self.log_ended(test)


class TestGitHubIssues(TestCase):
    name: str = "GitHub Issues"

    def test_all(self) -> None:
        self.test_issue_66()
        self.test_issue_79()
        self.test_issue_95()
        self.test_issue_114()
        self.test_issue_175()
        self.test_issue_363()
        self.test_issue_364()

    def test_issue_66(self):
        """'Connection' object has no attribute 'insert_id'"""
        test = "ISSUE 66"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"create table {self.table} (id integer primary key auto_increment, x integer)"
                )
                cur.execute(f"insert into {self.table} (x) values (1)")
                cur.execute(f"insert into {self.table} (x) values (1)")
                self.assertEqual(2, conn.insert_id)

            self.drop(conn)
        self.log_ended(test)

    def test_issue_79(self):
        """Duplicate field overwrites the previous one in the result of DictCursor"""
        test = "ISSUE 79"
        self.log_start(test)

        with self.get_conn() as conn:
            tb1 = f"{self.db}.a"
            tb2 = f"{self.db}.b"
            with conn.cursor(DictCursor) as cur:
                cur.execute(f"drop table if exists {tb1}")
                cur.execute(f"CREATE TABLE {tb1} (id int, value int)")
                cur.execute(f"insert into {tb1} values (%s, %s)", (1, 11))

                cur.execute(f"drop table if exists {tb2}")
                cur.execute(f"CREATE TABLE {tb2} (id int, value int)")
                cur.execute(f"insert into {tb2} values (%s, %s)", (1, 22))

                cur.execute(f"SELECT * FROM {tb1} inner join {tb2} on a.id = b.id")
                row = cur.fetchone()
                self.assertEqual(row["id"], 1)
                self.assertEqual(row["value"], 11)
                self.assertEqual(row["b.id"], 1)
                self.assertEqual(row["b.value"], 22)

                cur.execute(f"drop table {tb1}")
                cur.execute(f"drop table {tb2}")
        self.log_ended(test)

    def test_issue_95(self):
        """Leftover trailing OK packet for "CALL my_sp" queries"""
        test = "ISSUE 95"
        self.log_start(test)

        with self.get_conn() as conn:
            proc: str = f"{self.db}.foo"
            with conn.cursor() as cur:
                cur.execute(f"DROP PROCEDURE IF EXISTS {proc}")
                cur.execute(
                    f"""
                    CREATE PROCEDURE {proc} ()
                    BEGIN
                        SELECT 1;
                    END
                    """
                )
                cur.execute(f"CALL {proc}()")
                cur.execute("SELECT 1")
                self.assertEqual(cur.fetchone()[0], 1)
                cur.execute(f"DROP PROCEDURE IF EXISTS {proc}")
        self.log_ended(test)

    def test_issue_114(self):
        """autocommit is not set after reconnecting with ping()"""
        test = "ISSUE 114"
        self.log_start(test)

        with self.get_conn() as conn:
            conn.set_autocommit(False)
            with conn.cursor() as cur:
                cur.execute("select @@autocommit;")
                self.assertFalse(cur.fetchone()[0])
            conn.close()
            conn.ping()
            with conn.cursor() as cur:
                cur.execute("select @@autocommit;")
                self.assertFalse(cur.fetchone()[0])

        # Ensure set_autocommit() is still working
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select @@autocommit;")
                self.assertFalse(cur.fetchone()[0])
            conn.close()
            conn.ping()
            conn.set_autocommit(True)
            with conn.cursor() as cur:
                cur.execute("select @@autocommit;")
                self.assertTrue(cur.fetchone()[0])
        self.log_ended(test)

    def test_issue_175(self):
        """The number of fields returned by server is read in wrong way"""
        test = "ISSUE 175"
        self.log_start(test)

        with self.get_conn() as conn:
            tb: str = f"{self.db}.test_field_count"
            with conn.cursor() as cur:
                cur.execute(f"drop table if exists {tb}")
                for length in (100, 200, 300):
                    columns = ", ".join(f"c{i} integer" for i in range(length))
                    sql = f"create table {tb} ({columns})"
                    try:
                        cur.execute(sql)
                        cur.execute(f"select * from {tb}")
                        self.assertEqual(length, cur.field_count)
                    finally:
                        cur.execute(f"drop table if exists {tb}")
        self.log_ended(test)

    def test_issue_363(self):
        """Test binary / geometry types."""
        test = "ISSUE 363"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE TABLE {self.table} ( "
                    "id INTEGER PRIMARY KEY, geom LINESTRING NOT NULL /*!80003 SRID 0 */, "
                    "SPATIAL KEY geom (geom)) "
                    "ENGINE=MyISAM",
                )
                cur.execute(
                    f"INSERT INTO {self.table} (id, geom) VALUES"
                    "(1998, ST_GeomFromText('LINESTRING(1.1 1.1,2.2 2.2)'))"
                )

                # select WKT
                cur.execute(f"SELECT ST_AsText(geom) FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(row, ("LINESTRING(1.1 1.1,2.2 2.2)",))

                # select WKB
                cur.execute(f"SELECT ST_AsBinary(geom) FROM {self.table}")
                row = cur.fetchone()
                self.assertEqual(
                    row,
                    (
                        b"\x01\x02\x00\x00\x00\x02\x00\x00\x00"
                        b"\x9a\x99\x99\x99\x99\x99\xf1?"
                        b"\x9a\x99\x99\x99\x99\x99\xf1?"
                        b"\x9a\x99\x99\x99\x99\x99\x01@"
                        b"\x9a\x99\x99\x99\x99\x99\x01@",
                    ),
                )

                # select internal binary
                cur.execute(f"SELECT geom FROM {self.table}")
                row = cur.fetchone()
                # don't assert the exact internal binary value, as it could
                # vary across implementations
                self.assertTrue(isinstance(row[0], bytes))

            self.drop(conn)
        self.log_ended(test)

    def test_issue_364(self):
        """Test mixed unicode/binary arguments in executemany."""
        test = "ISSUE 364"
        self.log_start(test)

        with self.setup() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"create table {self.table} (value_1 binary(3), value_2 varchar(3)) "
                    "engine=InnoDB default charset=utf8mb4",
                )
                sql = f"insert into {self.table} (value_1, value_2) values (%s, %s)"
                usql = f"insert into {self.table} (value_1, value_2) values (%s, %s)"
                values = (b"\x00\xff\x00", "\xe4\xf6\xfc")

                # test single insert and select
                cur.execute(sql, values)
                cur.execute(f"select * from {self.table}")
                self.assertEqual(cur.fetchone(), values)

                # test single insert unicode query
                cur.execute(usql, values)

                # test multi insert and select
                cur.executemany(sql, args=(values, values, values))
                cur.execute(f"select * from {self.table}")
                for row in cur:
                    self.assertEqual(row, values)

                # test multi insert with unicode query
                cur.executemany(usql, args=(values, values, values))

            self.drop(conn)
        self.log_ended(test)


if __name__ == "__main__":
    for test in [
        TestCharset,
        TestTranscode,
        TestProtocol,
        TestConnection,
        TestAuthentication,
        TestConversion,
        TestCursor,
        TestLoadLocal,
        TestOptionFile,
        TestErrors,
        TestOldIssues,
        TestNewIssues,
        TestGitHubIssues,
    ]:
        tester: TestCase = test()
        tester.test_all()
