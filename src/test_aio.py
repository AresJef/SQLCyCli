import numpy as np, pandas as pd, orjson
import datetime, time, unittest, decimal, re, os
from sqlcycli import errors
from sqlcycli.constants import ER, CLIENT
from sqlcycli.aio.connection import (
    Connection,
    Cursor,
    DictCursor,
    DfCursor,
    SSCursor,
    SSDictCursor,
    SSDfCursor,
)
import asyncio


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

    async def test_all(self) -> None:
        pass

    # utils
    async def get_conn(self, **kwargs) -> Connection:
        conn = Connection(
            host=self.host,
            user=self.user,
            password=self.password,
            unix_socket=self.unix_socket,
            local_infile=True,
            **kwargs,
        )
        await conn.connect()
        return conn

    async def setup(self, table: str = None, **kwargs) -> Connection:
        conn = await self.get_conn(**kwargs)
        tb = self.tb if table is None else table
        async with conn.cursor() as cur:
            await cur.execute(f"CREATE DATABASE IF NOT EXISTS {self.db};")
            await cur.execute(f"DROP TABLE IF EXISTS {self.db}.{tb}")
        return conn

    async def drop(self, conn: Connection, table: str = None) -> None:
        tb = self.tb if table is None else table
        async with conn.cursor() as cur:
            await cur.execute(f"drop table if exists {self.db}.{tb}")

    async def delete(self, conn: Connection, table: str = None) -> None:
        tb = self.tb if table is None else table
        async with conn.cursor() as cur:
            await cur.execute(f"delete from {self.db}.{tb}")

    def log(self, msg: str, skip: bool = False) -> None:
        if skip:
            print(f"SKIP TEST '{self.name}': {msg}")
        else:
            print(f"PASS TEST '{self.name}': {msg}")


class TestConnection(TestCase):
    name: str = "Connection"

    async def test_all(self) -> None:
        await self.test_properties()
        await self.test_set_charset()
        await self.test_set_timeout()
        await self.test_largedata()
        await self.test_autocommit()
        await self.test_select_db()
        await self.test_connection_gone_away()
        await self.test_init_command()
        await self.test_close()
        await self.test_connection_exception()
        await self.test_transaction_exception()

    async def test_properties(self) -> None:
        test = "PROPERTIES"

        async with await self.get_conn() as conn:
            self.assertEqual(conn.host, "localhost")
            self.assertEqual(conn.port, 3306)
            self.assertEqual(conn.user, "root")
            self.assertEqual(type(conn.password), str)
            self.assertEqual(conn.database, None)
            self.assertEqual(conn.charset, "utf8mb4")
            self.assertEqual(conn.collation, "utf8mb4_general_ci")
            self.assertEqual(conn.encoding, "utf8")
            self.assertEqual(conn.connect_timeout, 10)
            self.assertEqual(conn.bind_address, None)
            self.assertEqual(conn.unix_socket, None)
            self.assertEqual(conn.autocommit, False)
            self.assertEqual(conn.local_infile, True)
            self.assertEqual(conn.max_allowed_packet, 16777216)
            self.assertEqual(conn.sql_mode, None)
            self.assertEqual(conn.init_command, None)
            self.assertEqual(type(conn.client_flag), int)
            self.assertEqual(conn.host_info, "socket localhost:3306")
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
        self.log(test)

    async def test_set_charset(self):
        test = "SET CHARACTER SET"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                await conn.set_charset("latin1")
                await cur.execute("SELECT @@character_set_connection")
                self.assertEqual(await cur.fetchone(), ("latin1",))
                self.assertEqual(conn.encoding, "cp1252")

                await conn.set_charset("utf8mb4", "utf8mb4_general_ci")
                await cur.execute(
                    "SELECT @@character_set_connection, @@collation_connection"
                )
                self.assertEqual(
                    await cur.fetchone(), ("utf8mb4", "utf8mb4_general_ci")
                )
                self.assertEqual(conn.encoding, "utf8")
        self.log(test)

    async def test_set_timeout(self):
        test = "SET TIMEOUT"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW VARIABLES LIKE 'wait_timeout'")
                g_wait = int((await cur.fetchone())[1])
                await cur.execute("SHOW VARIABLES LIKE 'net_read_timeout'")
                g_read = int((await cur.fetchone())[1])
                await cur.execute("SHOW VARIABLES LIKE 'net_write_timeout'")
                g_write = int((await cur.fetchone())[1])

            await conn.set_wait_timeout(180)
            self.assertEqual(await conn.get_wait_timeout(), 180)
            await conn.set_wait_timeout(None)
            self.assertEqual(await conn.get_wait_timeout(), g_wait)

            await conn.set_read_timeout(180)
            self.assertEqual(await conn.get_read_timeout(), 180)
            await conn.set_read_timeout(None)
            self.assertEqual(await conn.get_read_timeout(), g_read)

            await conn.set_write_timeout(180)
            self.assertEqual(await conn.get_write_timeout(), 180)
            await conn.set_write_timeout(None)
            self.assertEqual(await conn.get_write_timeout(), g_write)

        async with await self.get_conn(
            read_timeout=120,
            write_timeout=120,
            wait_timeout=120,
        ) as conn:
            self.assertEqual(await conn.get_wait_timeout(), 120)
            await conn.set_wait_timeout(g_wait)
            self.assertEqual(await conn.get_wait_timeout(), g_wait)
            await conn.set_wait_timeout(None)
            self.assertEqual(await conn.get_wait_timeout(), 120)

            self.assertEqual(await conn.get_read_timeout(), 120)
            await conn.set_read_timeout(g_read)
            self.assertEqual(await conn.get_read_timeout(), g_read)
            await conn.set_read_timeout(None)
            self.assertEqual(await conn.get_read_timeout(), 120)

            self.assertEqual(await conn.get_write_timeout(), 120)
            await conn.set_write_timeout(g_write)
            self.assertEqual(await conn.get_write_timeout(), g_write)
            await conn.set_write_timeout(None)
            self.assertEqual(await conn.get_write_timeout(), 120)

        self.log(test)

    async def test_largedata(self):
        """Large query and response (>=16MB)"""
        test = "LARGE DATA"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT @@max_allowed_packet")
                if (await cur.fetchone())[0] < 16 * 1024 * 1024 + 10:
                    self.log(
                        f"{test}: Set max_allowed_packet to bigger than 17MB", True
                    )
                    return None
                t = "a" * (16 * 1024 * 1024)
                await cur.execute("SELECT '%s'" % t)
                row = (await cur.fetchone())[0]
                assert row == t
        self.log(test)

    async def test_autocommit(self):
        test = "AUTOCOMMIT"
        async with await self.get_conn() as conn:
            await conn.set_autocommit(False)
            self.assertFalse(conn.autocommit)

            async with conn.cursor() as cur:
                await cur.execute("SET AUTOCOMMIT=1")
                self.assertTrue(conn.autocommit)

            await conn.set_autocommit(True)
            self.assertTrue(conn.autocommit)

            async with conn.cursor() as cur:
                await cur.execute("SELECT @@AUTOCOMMIT")
                self.assertEqual((await cur.fetchone())[0], 1)
        self.log(test)

    async def test_select_db(self):
        test = "SELECT DB"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                await conn.select_database("mysql")
                await cur.execute("SELECT database()")
                self.assertEqual((await cur.fetchone())[0], "mysql")

                await conn.select_database("test")
                await cur.execute("SELECT database()")
                self.assertEqual((await cur.fetchone())[0], "test")
        self.log(test)

    async def test_connection_gone_away(self):
        """
        http://dev.mysql.com/doc/refman/5.0/en/gone-away.html
        http://dev.mysql.com/doc/refman/5.0/en/error-messages-client.html#error_cr_server_gone_error
        """
        test = "CONNECTION GONE AWAY"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET wait_timeout=1")
                await asyncio.sleep(1.2)
                with self.assertRaises(errors.OperationalError) as cm:
                    await cur.execute("SELECT 1+1")
                    # error occurs while reading, not writing because of socket buffer.
                    # self.assertEqual(cm.exception.args[0], 2006)
                    self.assertIn(cm.exception.args[0], (2006, 2013))
        self.log(test)

    async def test_init_command(self):
        test = "INIT COMMAND"
        async with await self.get_conn(
            init_command='SELECT "bar"; SELECT "baz"',
            client_flag=CLIENT.MULTI_STATEMENTS,
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute('select "foobar";')
                self.assertEqual(("foobar",), await cur.fetchone())
        with self.assertRaises(errors.ConnectionClosedError):
            await conn.ping(reconnect=False)
        self.log(test)

    async def test_close(self):
        test = "CLOSE"
        async with await self.get_conn() as conn:
            self.assertFalse(conn.closed())
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                self.assertFalse(cur.closed())
            self.assertTrue(cur.closed())
        self.assertTrue(conn.closed())
        self.log(test)

    async def test_connection_exception(self):
        test = "CONNECTION EXCEPTION"
        async with await self.get_conn() as conn:
            with self.assertRaises(RuntimeError) as cm:
                self.assertFalse(conn.closed())
                raise RuntimeError("Test")
            self.assertEqual(type(cm.exception), RuntimeError)
            self.assertEqual(str(cm.exception), "Test")
        self.assertTrue(conn.closed())
        self.log(test)

    async def test_transaction_exception(self):
        test = "TRANSACTION EXCEPTION"
        async with await self.get_conn() as conn:
            try:
                async with conn.transaction() as cur:
                    await cur.execute("SELECT 1")
                    raise RuntimeError("Test")
            except RuntimeError:
                pass
            self.assertTrue(cur.closed())
            self.assertTrue(conn.closed())
        self.log(test)


class TestAuthentication(TestCase):
    name: str = "Authentication"

    async def test_all(self) -> None:
        await self.test_plugin()

    async def test_plugin(self) -> None:
        test = "PLUGIN"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "select plugin from mysql.user where concat(user, '@', host)=current_user()"
                )
                async for row in cur:
                    self.assertIn(
                        conn.server_auth_plugin_name, (row[0], "mysql_native_password")
                    )
        self.log(test)


class TestConversion(TestCase):
    name: str = "Conversion"

    async def test_all(self) -> None:
        await self.test_bool()
        await self.test_integer()
        await self.test_float()
        await self.test_string()
        await self.test_null()
        await self.test_datetime()
        await self.test_date()
        await self.test_time()
        await self.test_timedelta()
        await self.test_binary()
        await self.test_dict()
        await self.test_sequence()
        await self.test_ndarray_series_float()
        await self.test_ndarray_series_integer()
        await self.test_ndarray_series_bool()
        await self.test_ndarray_series_datetime()
        await self.test_ndarray_series_timedelta()
        await self.test_ndarray_series_bytes()
        await self.test_ndarray_series_unicode()
        await self.test_ndarray_series_object()
        await self.test_dataframe()
        await self.test_json()
        await self.test_bulk_insert()

    async def test_bool(self) -> None:
        test = "BOOL TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(f"create table {self.table} (a bit, b tinyint)")
                # . insert values
                await cur.execute(
                    f"insert into {self.table} (a, b) values (%s, %s)", (True, False)
                )
                # . validate
                await cur.execute(f"SELECT a, b FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                await self.delete(conn)

                ##################################################################
                # numpy bool
                await cur.execute(
                    f"insert into {self.table} (a, b) values (%s, %s)",
                    (np.bool_(True), np.bool_(False)),
                )
                # . validate
                await cur.execute(f"SELECT a, b FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_integer(self) -> None:
        test = "INTEGER TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
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
                await cur.execute(
                    f"""
                    insert into {self.table} (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) 
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    test_value,
                )
                # . validate
                await cur.execute(
                    f"SELECT a,b,c,d,e,f,g,h,i,j,k,l,m,n,o FROM {self.table}"
                )
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

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
                await cur.execute(
                    f"""
                    insert into {self.table} (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) 
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    test_value_np,
                )
                # . validate
                await cur.execute(
                    f"SELECT a,b,c,d,e,f,g,h,i,j,k,l,m,n,o FROM {self.table}"
                )
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_float(self) -> None:
        test = "FLOAT TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
                    f"""
                    create table {self.table} (
                        a float, b double, c decimal(10, 2)
                    )"""
                )
                # . insert values
                test_value = (5.7, 6.8, decimal.Decimal("7.9"))
                await cur.execute(
                    f"insert into {self.table} (a,b,c) values (%s,%s,%s)", test_value
                )
                # . use_decimal = False
                conn.set_use_decimal(False)
                # . validate
                await cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual((5.7, 6.8, 7.9), row)

                ##################################################################
                # . use_decimal = True
                conn.set_use_decimal(True)
                # . validate
                await cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                # numpy float
                test_value_np = (
                    np.float16(5.7),
                    np.float32(6.8),
                    np.float64(7.9),
                )
                await cur.execute(
                    f"insert into {self.table} (a,b,c) values (%s,%s,%s)", test_value_np
                )
                # . validate
                await cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_string(self) -> None:
        test = "STRING TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
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
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f) values (%s,%s,%s,%s,%s,%s)",
                    test_value,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

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
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f) values (%s,%s,%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_null(self) -> None:
        test = "NULL TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
                    f"create table {self.table} (a char(32), b varchar(32))"
                )
                # . insert values
                test_value = (None, None)
                await cur.execute(
                    f"insert into {self.table} (a,b) values (%s,%s)", test_value
                )
                # . validate
                await cur.execute(f"SELECT a,b FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                # . direct select
                await cur.execute("select null,''")
                row = await cur.fetchone()
                self.assertEqual((None, ""), row)
                # . validate
                await cur.execute("select '',null")
                row = await cur.fetchone()
                self.assertEqual(("", None), row)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_datetime(self) -> None:
        test = "DATETIME TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
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
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                    test_value,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value[1:4], row[1:4])
                self.assertEqual(datetime.datetime(*test_value[4][0:6]), row[4])
                await self.delete(conn)

                ##################################################################
                # numpy datetime
                test_value_np = (
                    np.datetime64("2014-05-15T07:45:57"),
                    np.datetime64("2014-05-15T07:45:57.051000"),
                    np.datetime64("2014-05-15T07:45:57"),
                    np.datetime64("2014-05-15T07:45:57.051000"),
                    np.datetime64(datetime.datetime(*test_value[4][0:6])),
                )
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value[1:4], row[1:4])
                self.assertEqual(datetime.datetime(*test_value[4][0:6]), row[4])
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_date(self) -> None:
        test = "DATE TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(f"create table {self.table} (a date, b date)")
                # . insert values
                test_value = (datetime.date(1988, 2, 2), datetime.date(1988, 2, 2))
                await cur.execute(
                    f"insert into {self.table} (a,b) values (%s,%s)", test_value
                )
                # . validate
                await cur.execute(f"SELECT a,b FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_time(self) -> None:
        test = "TIME TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
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
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) values %s",
                    test_value,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = await cur.fetchone()

                def time_to_timedelta(t: datetime.time) -> datetime.timedelta:
                    return datetime.timedelta(
                        0, t.hour * 3600 + t.minute * 60 + t.second, t.microsecond
                    )

                self.assertEqual(tuple(time_to_timedelta(i) for i in test_value), row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_timedelta(self) -> None:
        test = "TIMEDELTA TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
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
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) "
                    "values (%s,%s,%s,%s,%s,%s,%s)",
                    test_values,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_values, row)
                await self.delete(conn)

                ##################################################################
                # numpy timedelta
                test_values_np = [np.timedelta64(i) for i in test_values]
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) "
                    "values (%s,%s,%s,%s,%s,%s,%s)",
                    test_values_np,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_values, row)
                await self.delete(conn)

                ##################################################################
                # . direct select
                await cur.execute(
                    "select time('12:30'), time('23:12:59'), time('23:12:59.05100'),"
                    " time('-12:30'), time('-23:12:59'), time('-23:12:59.05100'), time('-00:30')"
                )
                # . validate
                row = await cur.fetchone()
                self.assertEqual(test_values, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_binary(self) -> None:
        test = "BINARY TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
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
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g,h,i) "
                    "values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    test_value,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f,g,h,i FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_dict(self) -> None:
        test = "DICT ESCAPING"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
                    f"create table {self.table} (a integer, b integer, c integer)"
                )
                # . insert values
                await cur.execute(
                    f"insert into {self.table} (a,b,c) values (%s, %s, %s)",
                    {"a": 1, "b": 2, "c": 3},
                )
                # . validate
                await cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual((1, 2, 3), row)
                await self.delete(conn)

                ##################################################################
                await cur.execute(
                    f"insert into {self.table} (a,b,c) values %s",
                    {"a": 1, "b": 2, "c": 3},
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual((1, 2, 3), row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_sequence(self) -> None:
        test = "SEQUENCE TYPE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(f"create table {self.table} (i integer, l integer)")
                # . insert values
                for seq_type in (list, tuple, set, frozenset, pd.Series, np.array):
                    await cur.execute(
                        f"insert into {self.table} (i, l) values (2, 4), (6, 8), (10, 12)"
                    )
                    seq = seq_type([2, 6])
                    await cur.execute(
                        "select l from %s where i in (%s) order by i"
                        % (self.table, "%s, %s"),
                        seq,
                    )
                    row = await cur.fetch()
                    self.assertEqual(((4,), (8,)), row)

                    # ------------------------------------------------------
                    await cur.execute(
                        "select l from %s where i in %s order by i"
                        % (self.table, "%s"),
                        seq,
                        itemize=False,
                    )
                    row = await cur.fetch()
                    self.assertEqual(((4,), (8,)), row)
                    await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_ndarray_series_float(self) -> None:
        test = "NDARRAY/SERIES FLOAT"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
                    f"create table {self.table} "
                    "(a double, b double, c double, d double, e double)"
                )
                # . validate float
                test_value = tuple(float(i) for i in range(-2, 3))
                for dtype in (np.float16, np.float32, np.float64):
                    test_value_np = np.array(test_value, dtype=dtype)
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                        test_value_np,
                    )
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                    # ------------------------------------------------------
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_np,
                        itemize=False,
                    )
                    # . validate
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                    # ------------------------------------------------------
                    test_value_pd = pd.Series(test_value_np, dtype=dtype)
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_pd,
                        itemize=False,
                    )
                    # . validate
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_ndarray_series_integer(self) -> None:
        test = "NDARRAY/SERIES INTEGER"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
                    f"create table {self.table} "
                    "(a bigint, b bigint, c bigint, d bigint, e bigint)"
                )
                # . validate int
                test_value = tuple(range(-2, 3))
                for dtype in (np.int8, np.int16, np.int32, np.int64):
                    test_value_np = np.array(test_value, dtype=dtype)
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                        test_value_np,
                    )
                    # . validate
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                    # ------------------------------------------------------
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_np,
                        itemize=False,
                    )
                    # . validate
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                    # ------------------------------------------------------
                    test_value_pd = pd.Series(test_value_np, dtype=dtype)
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_pd,
                        itemize=False,
                    )
                    # . validate
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                ##################################################################
                # . validate uint
                test_value = tuple(range(5))
                for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                    test_value_np = np.array(test_value, dtype=dtype)
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                        test_value_np,
                    )
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                    # ------------------------------------------------------
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_np,
                        itemize=False,
                    )
                    # . validate
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                    # ------------------------------------------------------
                    test_value_pd = pd.Series(test_value_np, dtype=dtype)
                    await cur.execute(
                        f"insert into {self.table} (a,b,c,d,e) values %s",
                        test_value_pd,
                        itemize=False,
                    )
                    # . validate
                    await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                    row = await cur.fetchone()
                    self.assertEqual(test_value, row)
                    await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_ndarray_series_bool(self) -> None:
        test = "NDARRAY/SERIES BOOL"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(f"create table {self.table} (a bit, b tinyint)")
                # . insert values
                test_value_np = np.array([True, False], dtype=np.bool_)
                await cur.execute(
                    f"insert into {self.table} (a,b) values (%s,%s)", test_value_np
                )
                # . validate
                await cur.execute(f"SELECT a,b FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                await self.delete(conn)

                ##################################################################
                await cur.execute(
                    f"insert into {self.table} (a,b) values %s",
                    test_value_np,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                await self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype=np.bool_)
                await cur.execute(
                    f"insert into {self.table} (a,b) values %s",
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual((b"\x01", 0), row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_ndarray_series_datetime(self) -> None:
        test = "NDARRAY/SERIES DATETIME"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
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
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d) values (%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d) values %s",
                    test_value_np,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype="datetime64[us]")
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d) values %s",
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_ndarray_series_timedelta(self) -> None:
        test = "NDARRAY/SERIES TIMEDELTA"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
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
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) values (%s,%s,%s,%s,%s,%s,%s)",
                    test_values_np,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_values, row)
                await self.delete(conn)

                ##################################################################
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) values %s",
                    test_values_np,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_values, row)
                await self.delete(conn)

                ##################################################################
                test_values_pd = pd.Series(test_values_np, dtype="timedelta64[us]")
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e,f,g) values %s",
                    test_values_pd,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e,f,g FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_values, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_ndarray_series_bytes(self) -> None:
        test = "NDARRAY/SERIES BYTES"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
                    f"create table {self.table} "
                    "(a varbinary(32), b varbinary(32), c varbinary(32), d varbinary(32), e varbinary(32))"
                )
                # . insert values
                test_value = tuple(str(i).encode("utf8") for i in range(5))
                test_value_np = np.array(test_value, dtype="S")
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values %s",
                    test_value_np,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype="S")
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values %s",
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_ndarray_series_unicode(self) -> None:
        test = "NDARRAY/SERIES UNICODE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # . create test table
                await cur.execute(
                    f"create table {self.table} "
                    "(a varchar(32), b varchar(32), c varchar(32), d varchar(32), e varchar(32))"
                )
                # . insert values
                test_value = tuple(str(i) for i in range(5))
                test_value_np = np.array(test_value, dtype="U")
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values (%s,%s,%s,%s,%s)",
                    test_value_np,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values %s",
                    test_value_np,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype="U")
                await cur.execute(
                    f"insert into {self.table} (a,b,c,d,e) values %s",
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT a,b,c,d,e FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_ndarray_series_object(self) -> None:
        test = "NDARRAY/SERIES OBJECT"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
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
                await cur.execute(
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
                await cur.execute(
                    "insert into %s (%s) values (%s)"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_value_np,
                )
                # . validate
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await cur.execute(
                    "insert into %s (%s) values %s"
                    % (self.table, ",".join(cols.keys()), "%s"),
                    test_value_np,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                test_value_pd = pd.Series(test_value_np, dtype="O")
                await cur.execute(
                    "insert into %s (%s) values %s"
                    % (self.table, ",".join(cols.keys()), "%s"),
                    test_value_pd,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_dataframe(self) -> None:
        test = "DATAFRAME"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
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
                await cur.execute(
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
                await cur.execute(
                    "insert into %s (%s) values (%s)"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_df,
                )
                # . validate
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await cur.execute(
                    "insert into %s (%s) values %s"
                    % (self.table, ",".join(cols.keys()), "%s"),
                    test_df,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_json(self) -> None:
        test = "JSON TYPE"
        async with await self.setup() as conn:
            if conn.server_version < (5, 7, 0):
                self.log(test, True)
                return None
            async with conn.cursor() as cur:
                ##################################################################
                await cur.execute(
                    f"create table {self.table} "
                    "(id int primary key not null, json JSON not null)"
                )
                # . insert values
                test_value = '{"hello": "こんにちは"}'
                await cur.execute(
                    f"INSERT INTO {self.table} " "(id, `json`) values (42, %s)",
                    test_value,
                )
                # . decode_json = False
                conn.set_decode_json(False)
                # . validate
                await cur.execute(f"SELECT `json` from {self.table} WHERE `id`=42")
                res = (await cur.fetchone())[0]
                self.assertEqual(orjson.loads(test_value), orjson.loads(res))

                ##################################################################
                # . decode_json = True
                conn.set_decode_json(True)
                # . validate
                await cur.execute(f"SELECT `json` from {self.table} WHERE `id`=42")
                res = (await cur.fetchone())[0]
                self.assertEqual(orjson.loads(test_value), res)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_bulk_insert(self) -> None:
        test = "BULK INSERT"
        async with await self.setup() as conn:
            table_encoded = self.table.encode(conn.encoding)
            async with conn.cursor() as cur:
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
                await cur.execute(
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
                await cur.execute(
                    "insert into %s (%s) values (%s)"
                    % (self.table, ",".join(cols.keys()), ",".join(["%s"] * len(cols))),
                    test_value_bulk,
                )
                # . validate
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                # . single format (itemize = False)
                await cur.execute(
                    "insert into %s (%s) values %s"
                    % (self.table, ",".join(cols.keys()), "%s"),
                    test_value_bulk,
                    itemize=False,
                )
                # . validate
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                # . as v
                await cur.execute(
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
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                # . on duplicate
                await cur.execute(
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
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                # . as v on duplicate
                await cur.execute(
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
                await cur.execute(f"SELECT * FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(test_value, row)
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)


class TestCursor(TestCase):
    name: str = "Cursor"
    bob = {"name": "bob", "age": 21, "DOB": datetime.datetime(1990, 2, 6, 23, 4, 56)}
    jim = {"name": "jim", "age": 56, "DOB": datetime.datetime(1955, 5, 9, 13, 12, 45)}
    fred = {"name": "fred", "age": 100, "DOB": datetime.datetime(1911, 9, 12, 1, 1, 1)}

    async def test_all(self) -> None:
        await self.test_fetch_no_result()
        await self.test_fetch_single_tuple()
        await self.test_fetch_aggregates()
        await self.test_cursor_iter()
        await self.test_cleanup_rows_buffered()
        await self.test_cleanup_rows_unbuffered()
        await self.test_execute_args()
        await self.test_executemany()
        await self.test_execution_time_limit()
        await self.test_warnings()
        await self.test_SSCursor()
        await self.test_DictCursor(False)
        await self.test_DictCursor(True)
        await self.test_DfCursor(False)
        await self.test_DfCursor(True)
        await self.test_next_set()
        await self.test_skip_next_set()
        await self.test_next_set_error()
        await self.test_ok_and_next()
        await self.test_multi_statement_warnings()
        await self.test_previous_cursor_not_closed()
        await self.test_commit_during_multi_result()
        await self.test_transaction()
        await self.test_scroll()
        await self.test_procedure()
        await self.test_execute_cancel()

    async def test_fetch_no_result(self) -> None:
        test = "FETCH NO RESULT"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                await cur.execute(f"create table {self.table} (a varchar(32))")
                await cur.execute(f"insert into {self.table} (a) values (%s)", "mysql")
                self.assertEqual(None, await cur.fetchone())
                self.assertEqual((), await cur.fetch(0))
                self.assertEqual((), await cur.fetch(2))

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_fetch_single_tuple(self) -> None:
        test = "FETCH SINGLE TUPLE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                await cur.execute(f"create table {self.table} (id integer primary key)")
                await cur.execute(f"insert into {self.table} (id) values (1),(2)")
                await cur.execute(f"SELECT id FROM {self.table} where id in (1)")
                self.assertEqual(((1,),), await cur.fetch())

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_fetch_aggregates(self) -> None:
        test = "FETCH AGGREGATES"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                await cur.execute(f"create table {self.table} (id integer primary key)")
                await cur.execute(
                    f"insert into {self.table} (id) values (%s)",
                    range(10),
                    force_many=True,
                )
                await cur.execute(f"SELECT sum(id) FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(sum(range(10)), row[0])

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_cursor_iter(self) -> None:
        test = "CURSOR ITER"
        async with await self.setupForCursor() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                await cur.execute(f"select * from {self.table}")
                self.assertEqual(cur.__aiter__(), cur)
                self.assertEqual(await cur.__anext__(), ("row1",))
                self.assertEqual(await cur.__anext__(), ("row2",))
                self.assertEqual(await cur.__anext__(), ("row3",))
                self.assertEqual(await cur.__anext__(), ("row4",))
                self.assertEqual(await cur.__anext__(), ("row5",))
                with self.assertRaises(StopAsyncIteration):
                    await cur.__anext__()

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_cleanup_rows_buffered(self) -> None:
        test = "CLEANUP ROWS BUFFERED"
        async with await self.setupForCursor() as conn:
            ##################################################################
            async with conn.cursor() as cur:
                await cur.execute(
                    f"select * from {self.table} as t1, {self.table} as t2"
                )
                counter = 0
                async for _ in cur:
                    counter += 1
                    if counter > 10:
                        break

            ##################################################################
            async with conn.cursor() as cur2:
                await cur2.execute("select 1")
                self.assertEqual(await cur2.fetchone(), (1,))
                self.assertIsNone(await cur2.fetchone())

            ##################################################################
            await self.drop(conn)
        self.log(test)

    async def test_cleanup_rows_unbuffered(self) -> None:
        test = "CLEANUP ROWS UNBUFFERED"
        async with await self.setupForCursor() as conn:
            ##################################################################
            async with conn.cursor(SSCursor) as cur:
                await cur.execute(
                    f"select * from {self.table} as t1, {self.table} as t2"
                )
                counter = 0
                async for _ in cur:
                    counter += 1
                    if counter > 10:
                        break

            ##################################################################
            async with conn.cursor(SSCursor) as cur2:
                await cur2.execute("select 1")
                self.assertEqual(await cur2.fetchone(), (1,))
                self.assertIsNone(await cur2.fetchone())

            ##################################################################
            await self.drop(conn)
        self.log(test)

    async def test_execute_args(self) -> None:
        test = "EXECUTE ARGUMENTS"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                # . create table
                await cur.execute(f"create table {self.table} (i int, j int)")

                # . insert one element
                # . argument 1 should automatically be escaped into
                # . str '1' and formatted into the sql as one row values.
                await cur.execute(f"insert into {self.table} (i) values (%s)", 1)
                self.assertEqual(cur.executed_sql.endswith(b"values (1)"), True)
                await cur.execute(f"select i from {self.table}")
                self.assertEqual(await cur.fetchone(), (1,))
                await self.delete(conn)

                # . insert one row
                # . argument (1, 2) should automacially be escaped into
                # . tuple of str ('1', '2') and formatted into the sql as
                # . one row values.
                await cur.execute(
                    f"insert into {self.table} (i, j) values (%s, %s)", (1, 2)
                )
                self.assertEqual(cur.executed_sql.endswith(b"values (1, 2)"), True)
                await cur.execute(f"select * from {self.table}")
                self.assertEqual(await cur.fetchone(), (1, 2))
                await self.delete(conn)

                # . insert force_many=True
                # . argument range(10) should be escaped into
                # . tuple of str ('0', '1', ..., '9') but instead of being
                # . formatted into the sql as one row, it should be formatted
                # . as multiple rows values.
                await cur.execute(
                    f"insert into {self.table} (i) values (%s)",
                    range(10),
                    force_many=True,
                )
                self.assertEqual(
                    cur.executed_sql.endswith(
                        b"values (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)"
                    ),
                    True,
                )
                await cur.execute(f"select i from {self.table}")
                self.assertEqual(await cur.fetch(), tuple((i,) for i in range(10)))
                await self.delete(conn)

                # . insert force_many=True & itemize=False
                # . force_many should automatically set itemize to True.
                await cur.execute(
                    f"insert into {self.table} (i) values (%s)",
                    range(10),
                    force_many=True,
                    itemize=False,
                )
                self.assertEqual(
                    cur.executed_sql.endswith(
                        b"values (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)"
                    ),
                    True,
                )
                await cur.execute(f"select i from {self.table}")
                self.assertEqual(await cur.fetch(), tuple((i,) for i in range(10)))
                await self.delete(conn)

                # . itemize=False
                # . argument [(1, 2), (3, 4), (5, 6)] should be escaped into
                # . as a single string '(1, 2), (3, 4), (5, 6)' and formatted
                # . into the sql directly.
                await cur.execute(
                    f"insert into {self.table} (i, j) values %s",
                    [(1, 2), (3, 4), (5, 6)],
                    itemize=False,
                )
                self.assertEqual(
                    cur.executed_sql.endswith(b"values (1,2),(3,4),(5,6)"), True
                )
                await cur.execute(f"select * from {self.table}")
                self.assertEqual(await cur.fetch(), ((1, 2), (3, 4), (5, 6)))
                await self.delete(conn)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_executemany(self) -> None:
        test = "EXECUTE MANY"
        async with await self.setupForCursor() as conn:
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
            async with conn.cursor() as cur:
                await cur.execute(
                    f"insert into {self.table} (data) values (%s)",
                    range(10),
                    force_many=True,
                )
                self.assertTrue(
                    cur.executed_sql.endswith(b",(7),(8),(9)"),
                    "execute many with %s not in one query",
                )
            await self.drop(conn)

            # %% in column set
            async with conn.cursor() as cur:
                await cur.execute(f"DROP TABLE IF EXISTS {self.db}.percent_test")
                await cur.execute(
                    f"""\
                    CREATE TABLE {self.db}.percent_test (
                        `A%` INTEGER,
                        `B%` INTEGER)"""
                )
                sql = (
                    f"INSERT INTO {self.db}.percent_test (`A%%`, `B%%`) VALUES (%s, %s)"
                )
                self.assertIsNotNone(INSERT_VALUES_RE.match(sql))
                await cur.execute(sql, [(3, 4), (5, 6)])
                self.assertTrue(
                    cur.executed_sql.endswith(b"(3, 4),(5, 6)"),
                    "executemany with %% not in one query",
                )
                await cur.execute(f"DROP TABLE {self.db}.percent_test")
        self.log(test)

    async def test_execution_time_limit(self) -> None:
        test = "EXECUTION TIME LIMIT"
        async with await self.setupForCursor() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                # MySQL MAX_EXECUTION_TIME takes ms
                # MariaDB max_statement_time takes seconds as int/float, introduced in 10.1

                # this will sleep 0.01 seconds per row
                if conn.server_vendor == "mysql":
                    sql = f"SELECT /*+ MAX_EXECUTION_TIME(2000) */ data, sleep(0.01) FROM {self.table}"
                else:
                    sql = f"SET STATEMENT max_statement_time=2 FOR SELECT data, sleep(0.01) FROM {self.table}"
                await cur.execute(sql)
                rows = await cur.fetch()
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
                await cur.execute(sql)
                row = await cur.fetchone()
                self.assertEqual(row, ("row1", 0))

                # this discards the previous unfinished query
                await cur.execute("SELECT 1")
                self.assertEqual(await cur.fetchone(), (1,))

                if conn.server_vendor == "mysql":
                    sql = f"SELECT /*+ MAX_EXECUTION_TIME(1) */ data, sleep(1) FROM {self.table}"
                else:
                    sql = f"SET STATEMENT max_statement_time=0.001 FOR SELECT data, sleep(1) FROM {self.table}"
                with self.assertRaises(errors.OperationalError) as cm:
                    # in a buffered cursor this should reliably raise an
                    # OperationalError
                    await cur.execute(sql)
                    if conn.server_vendor == "mysql":
                        # this constant was only introduced in MySQL 5.7, not sure
                        # what was returned before, may have been ER_QUERY_INTERRUPTED
                        self.assertEqual(cm.exception.args[0], ER.QUERY_TIMEOUT)
                    else:
                        self.assertEqual(cm.exception.args[0], ER.STATEMENT_TIMEOUT)

                # connection should still be fine at this point
                await cur.execute("SELECT 1")
                self.assertEqual(await cur.fetchone(), (1,))

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_warnings(self) -> None:
        test = "WARNINGS"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                ##################################################################
                await cur.execute(f"DROP TABLE IF EXISTS {self.db}.no_exists_table")
                self.assertEqual(cur.warning_count, 1)

                ##################################################################
                await cur.execute("SHOW WARNINGS")
                row = await cur.fetchone()
                self.assertEqual(row[1], ER.BAD_TABLE_ERROR)
                self.assertIn("no_exists_table", row[2])

                ##################################################################
                await cur.execute("SELECT 1")
                self.assertEqual(cur.warning_count, 0)

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_SSCursor(self) -> None:
        test = "CURSOR UNBUFFERED"
        async with await self.setup(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            async with conn.cursor(SSCursor) as cur:
                # . create table
                await cur.execute(
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
                await conn.begin()
                for row in data:
                    await cur.execute(
                        f"INSERT INTO {self.table} VALUES (%s, %s, %s)", row
                    )
                    self.assertEqual(
                        conn.affected_rows, 1, "affected_rows does not match"
                    )
                await conn.commit()

                # Test fetchone()
                index = 0
                await cur.execute(f"SELECT * FROM {self.table}")
                while True:
                    row = await cur.fetchone()
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
                        cur.row_number, index, "cursor.row_number != %s" % (str(index))
                    )

                    # Test row came out the same as it went in
                    self.assertEqual(
                        (row in data), True, "Row not found in source data"
                    )

                # Test fetch() all
                await cur.execute(f"SELECT * FROM {self.table}")
                self.assertEqual(
                    len(await cur.fetch()),
                    len(data),
                    "fetch() all failed. Number of rows does not match",
                )

                # Test fetch(2) many
                await cur.execute(f"SELECT * FROM {self.table}")
                self.assertEqual(
                    len(await cur.fetch(2)),
                    2,
                    "fetch(2) many failed. Number of rows does not match",
                )
                await cur.fetch()

                # Test update, affected_rows()
                await cur.execute(f"UPDATE {self.table} SET zone = %s", "Foo")
                await conn.commit()
                self.assertEqual(
                    cur.affected_rows,
                    len(data),
                    "Update failed cursor.affected_rows != %s" % (str(len(data))),
                )

                # Test executemany
                await cur.execute(f"INSERT INTO {self.table} VALUES (%s, %s, %s)", data)
                self.assertEqual(
                    cur.affected_rows,
                    len(data),
                    "execute many failed. cursor.affected_rows != %s"
                    % (str(len(data))),
                )

                # Test multiple datasets
                await cur.execute("SELECT 1; SELECT 2; SELECT 3")
                self.assertListEqual(list(await cur), [(1,)])
                await cur.next_set()
                self.assertListEqual(list(await cur), [(2,)])
                await cur.next_set()
                self.assertListEqual(list(await cur), [(3,)])
                await cur.next_set()

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_DictCursor(self, unbuffered: bool = False) -> None:
        if unbuffered:
            test = "DICT CURSOR UNBUFFERED"
        else:
            test = "DICT CURSOR"
        async with await self.setupForDictCursor() as conn:
            async with conn.cursor(SSDictCursor if unbuffered else DictCursor) as cur:
                bob, jim, fred = self.bob.copy(), self.jim.copy(), self.fred.copy()
                # try an update which should return no rows
                await cur.execute(f"update {self.table} set age=20 where name='bob'")
                bob["age"] = 20
                # pull back the single row dict for bob and check
                await cur.execute(f"SELECT * from {self.table} where name='bob'")
                row = await cur.fetchone()
                self.assertEqual(bob, row, "fetchone via DictCursor failed")
                if unbuffered:
                    await cur.fetch()

                # same again, but via fetchall => tuple(row)
                await cur.execute(f"SELECT * from {self.table} where name='bob'")
                row = await cur.fetch()
                self.assertEqual(
                    (bob,),
                    row,
                    "fetch a 1 row result via fetchall failed via DictCursor",
                )

                # same test again but iterate over the
                await cur.execute(f"SELECT * from {self.table} where name='bob'")
                async for row in cur:
                    self.assertEqual(
                        bob,
                        row,
                        "fetch a 1 row result via iteration failed via DictCursor",
                    )

                # get all 3 row via fetchall
                await cur.execute(f"SELECT * from {self.table}")
                rows = await cur.fetch()
                self.assertEqual(
                    (bob, jim, fred), rows, "fetchall failed via DictCursor"
                )

                # same test again but do a list comprehension
                await cur.execute(f"SELECT * from {self.table}")
                rows = list(await cur)
                self.assertEqual(
                    [bob, jim, fred], rows, "DictCursor should be iterable"
                )

                # get all 2 row via fetch(2) and iterate the last one
                await cur.execute(f"SELECT * from {self.table}")
                rows = await cur.fetch(2)
                self.assertEqual((bob, jim), rows, "fetchmany failed via DictCursor")
                async for row in cur:
                    self.assertEqual(
                        fred,
                        row,
                        "fetch a 1 row result via iteration failed via DictCursor",
                    )
                if unbuffered:
                    await cur.fetch()

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_DfCursor(self, unbuffered: bool = False) -> None:
        if unbuffered:
            test = "DATAFRAME CURSOR UNBUFFERED"
        else:
            test = "DATAFRAME CURSOR"
        async with await self.setupForDictCursor() as conn:
            async with conn.cursor(SSDfCursor if unbuffered else DfCursor) as cur:
                cur: DfCursor
                df = pd.DataFrame([self.bob, self.jim, self.fred])
                # try an update which should return no rows
                await cur.execute(f"update {self.table} set age=20 where name='bob'")
                df.loc[df["name"] == "bob", "age"] = 20
                # pull back the single row dict for bob and check
                await cur.execute(f"SELECT * from {self.table} where name='bob'")
                row = await cur.fetchone()
                assert row.equals(df.iloc[0:1])
                if unbuffered:
                    await cur.fetch()

                # same again, but via fetchall => tuple(row)
                await cur.execute(f"SELECT * from {self.table} where name='bob'")
                row = await cur.fetch()
                assert row.equals(df.iloc[0:1])

                # same test again but iterate over the
                await cur.execute(f"SELECT * from {self.table} where name='bob'")
                i = 0
                async for row in cur:
                    assert row.equals(df.iloc[i : i + 1])
                    i += 1

                # get all 3 row via fetchall
                await cur.execute(f"SELECT * from {self.table}")
                rows = await cur.fetch()
                assert rows.equals(df)

                # get all 2 row via fetch(2) and iterate the last one
                await cur.execute(f"SELECT * from {self.table}")
                rows = await cur.fetch(2)
                assert rows.equals(df.iloc[0:2])
                async for row in cur:
                    assert row.equals(df.iloc[2:3].reset_index(drop=True))
                if unbuffered:
                    await cur.fetch()

                ##################################################################
                await self.drop(conn)
        self.log(test)

    async def test_next_set(self) -> None:
        test = "NEXT SET"
        async with await self.setup(
            init_command='SELECT "bar"; SELECT "baz"',
            client_flag=CLIENT.MULTI_STATEMENTS,
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1; SELECT 2;")
                self.assertEqual([(1,)], list(await cur))
                res = await cur.next_set()
                self.assertEqual(res, True)
                self.assertEqual([(2,)], list(await cur))
                self.assertEqual(await cur.next_set(), False)
        self.log(test)

    async def test_skip_next_set(self) -> None:
        test = "SKIP NEXT SET"
        async with await self.setup(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1; SELECT 2;")
                self.assertEqual([(1,)], list(await cur))

                await cur.execute("SELECT 42")
                self.assertEqual([(42,)], list(await cur))
        self.log(test)

    async def test_next_set_error(self) -> None:
        test = "NEXT SET ERROR"
        async with await self.setup(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            async with conn.cursor() as cur:
                for i in range(3):
                    await cur.execute("SELECT %s; xyzzy;", (i,))
                    self.assertEqual([(i,)], list(await cur))
                    with self.assertRaises(errors.ProgrammingError):
                        await cur.next_set()
                    self.assertEqual((), await cur.fetch())
        self.log(test)

    async def test_ok_and_next(self):
        test = "OK AND NEXT"
        async with await self.setup(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1; commit; SELECT 2;")
                self.assertEqual([(1,)], list(await cur))
                self.assertTrue(await cur.next_set())
                self.assertTrue(await cur.next_set())
                self.assertEqual([(2,)], list(await cur))
                self.assertFalse(await cur.next_set())
        self.log(test)

    async def test_multi_statement_warnings(self):
        test = "MULTI STATEMENT WARNINGS"
        conn = await self.setup(
            init_command='SELECT "bar"; SELECT "baz"',
            client_flag=CLIENT.MULTI_STATEMENTS,
        )
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DROP TABLE IF EXISTS {self.db}.a; DROP TABLE IF EXISTS {self.db}.b;"
                )
            self.log(test)
        except TypeError:
            self.fail()
        finally:
            await conn.close()

    async def test_previous_cursor_not_closed(self):
        test = "PREVIOUS CURSOR NOT CLOSED"
        async with await self.get_conn(
            init_command='SELECT "bar"; SELECT "baz"',
            client_flag=CLIENT.MULTI_STATEMENTS,
        ) as conn:
            async with conn.cursor() as cur1:
                await cur1.execute("SELECT 1; SELECT 2")
                async with conn.cursor() as cur2:
                    await cur2.execute("SELECT 3")
                    self.assertEqual((await cur2.fetchone())[0], 3)
        self.log(test)

    async def test_commit_during_multi_result(self):
        test = "COMMIT DURING MULTI RESULT"
        async with await self.get_conn(client_flag=CLIENT.MULTI_STATEMENTS) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1; SELECT 2")
                await conn.commit()
                await cur.execute("SELECT 3")
                self.assertEqual((await cur.fetchone())[0], 3)
        self.log(test)

    async def test_transaction(self):
        test = "TRANSACTION"
        async with await self.setup() as conn:
            async with conn.transaction() as cur:
                await cur.execute(
                    f"create table {self.table} (name char(20), age int, DOB datetime)"
                )
                await cur.execute(
                    f"insert into {self.table} values (%s, %s, %s)",
                    [
                        ("bob", 21, "1990-02-06 23:04:56"),
                        ("jim", 56, "1955-05-09 13:12:45"),
                        ("fred", 100, "1911-09-12 01:01:01"),
                    ],
                )

            async with conn.cursor() as cur:
                await cur.execute(f"SELECT * FROM {self.table}")
                rows = await cur.fetch()
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

            await self.drop(conn)

            with self.assertRaises(RuntimeError):
                async with conn.transaction() as cur:
                    raise RuntimeError("test")
            self.assertTrue(cur.closed())
            self.assertTrue(conn.closed())

        self.log(test)

    async def test_scroll(self):
        test = "SCROLL"
        async with await self.setupForCursor() as conn:
            # Buffered #########################################################
            async with conn.cursor() as cur:
                await cur.execute(f"select * from {self.table}")
                with self.assertRaises(errors.ProgrammingError):
                    await cur.scroll(1, "foo")

            async with conn.cursor() as cur:
                await cur.execute(f"select * from {self.table}")
                await cur.scroll(1, "relative")
                self.assertEqual(await cur.fetchone(), ("row2",))
                await cur.scroll(2, "relative")
                self.assertEqual(await cur.fetchone(), ("row5",))

            async with conn.cursor() as cur:
                await cur.execute(f"select * from {self.table}")
                await cur.scroll(2, mode="absolute")
                self.assertEqual(await cur.fetchone(), ("row3",))
                await cur.scroll(4, mode="absolute")
                self.assertEqual(await cur.fetchone(), ("row5",))

            async with conn.cursor() as cur:
                await cur.execute(f"select * from {self.table}")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    await cur.scroll(5, "relative")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    await cur.scroll(-1, "relative")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    await cur.scroll(-5, "absolute")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    await cur.scroll(-1, "absolute")

            # Unbuffered #######################################################
            async with conn.cursor(SSCursor) as cur:
                await cur.execute(f"select * from {self.table}")
                with self.assertRaises(errors.ProgrammingError):
                    await cur.scroll(1, "foo")

            async with conn.cursor(SSCursor) as cur:
                await cur.execute(f"select * from {self.table}")
                await cur.scroll(1, "relative")
                self.assertEqual(await cur.fetchone(), ("row2",))
                await cur.scroll(2, "relative")
                self.assertEqual(await cur.fetchone(), ("row5",))

            async with conn.cursor(SSCursor) as cur:
                await cur.execute(f"select * from {self.table}")
                await cur.scroll(2, mode="absolute")
                self.assertEqual(await cur.fetchone(), ("row3",))
                await cur.scroll(4, mode="absolute")
                self.assertEqual(await cur.fetchone(), ("row5",))

            async with conn.cursor(SSCursor) as cur:
                await cur.execute(f"select * from {self.table}")
                await cur.scroll(1, "relative")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    await cur.scroll(-1, "relative")
                with self.assertRaises(errors.InvalidCursorIndexError):
                    await cur.scroll(0, "absolute")
                await cur.scroll(10)
                self.assertEqual(await cur.fetchone(), None)
                self.assertEqual(cur.row_number, 5)

            # ##################################################################
            await self.drop(conn)
        self.log(test)

    async def test_procedure(self):
        test = "PROCEDURE"
        async with await self.setup() as conn:
            await conn.select_database(self.db)
            # Buffered #########################################################
            async with conn.cursor() as cur:
                await cur.execute("DROP PROCEDURE IF EXISTS myinc;")
                await cur.execute(
                    """
                    CREATE PROCEDURE myinc(p1 INT, p2 INT)
                    BEGIN
                        SELECT p1 + p2 + 1;
                    END"""
                )
                await conn.commit()

            async with conn.cursor() as cur:
                await cur.callproc("myinc", (1, 2))
                res = await cur.fetchone()
                self.assertEqual(res, (4,))

            with self.assertRaises(errors.ProgrammingError):
                await cur.callproc("myinc", [1, 2])

            async with conn.cursor() as cur:
                await cur.execute("DROP PROCEDURE IF EXISTS myinc;")

            # Unbuffered #######################################################
            async with conn.cursor(SSCursor) as cur:
                await cur.execute("DROP PROCEDURE IF EXISTS myinc;")
                await cur.execute(
                    """
                    CREATE PROCEDURE myinc(p1 INT, p2 INT)
                    BEGIN
                        SELECT p1 + p2 + 1;
                    END"""
                )
                await conn.commit()

            async with conn.cursor(SSCursor) as cur:
                await cur.callproc("myinc", (1, 2))
                res = await cur.fetchone()
                self.assertEqual(res, (4,))

            with self.assertRaises(errors.ProgrammingError):
                await cur.callproc("myinc", [1, 2])

            async with conn.cursor(SSCursor) as cur:
                await cur.execute("DROP PROCEDURE IF EXISTS myinc;")

        self.log(test)

    async def test_execute_cancel(self):
        test = "EXECUTE CANCEL"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                # Cancel a cursor in the middle of execution, before it could
                # read even the first packet (SLEEP assures the timings)
                task = asyncio.ensure_future(
                    cur.execute("SELECT 1 as id, SLEEP(0.1) as xxx")
                )
                await asyncio.sleep(0.05)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(cur.closed())

            with self.assertRaises(errors.InterfaceError) as cm:
                async with conn.cursor() as cur:
                    pass
                self.assertEqual(
                    cm.exception.args[1],
                    "Connection closed: cancelled during execution.",
                )
        self.log(test)

    # utils
    async def setupForCursor(self, table: str = None, **kwargs) -> Connection:
        conn = await self.setup(table, **kwargs)
        tb = self.tb if table is None else table
        async with conn.cursor() as cur:
            await cur.execute(f"create table {self.db}.{tb} (data varchar(10))")
            await cur.execute(
                f"insert into {self.db}.{tb} values (%s)",
                ["row%d" % i for i in range(1, 6)],
                force_many=True,
            )
        await conn.commit()
        return conn

    async def setupForDictCursor(self, table: str = None, **kwargs) -> Connection:
        conn = await self.setup(table, **kwargs)
        tb = self.tb if table is None else table
        async with conn.cursor() as cur:
            await cur.execute(
                f"create table {self.db}.{tb} (name char(20), age int, DOB datetime)"
            )
            await cur.execute(
                f"insert into {self.db}.{tb} values (%s, %s, %s)",
                [
                    ("bob", 21, "1990-02-06 23:04:56"),
                    ("jim", 56, "1955-05-09 13:12:45"),
                    ("fred", 100, "1911-09-12 01:01:01"),
                ],
            )
        await conn.commit()
        return conn


class TestLoadLocal(TestCase):
    name: str = "Load Local"

    async def test_all(self) -> None:
        await self.test_no_file()
        await self.test_load_file(False)
        await self.test_load_file(True)
        await self.test_load_warnings()

    async def test_no_file(self):
        test = "NO FILE"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                with self.assertRaises(errors.OperationalError):
                    await cur.execute(
                        "LOAD DATA LOCAL INFILE 'no_data.txt' INTO TABLE "
                        "test_load_local fields terminated by ','"
                    )
            await self.drop(conn)
        self.log(test)

    async def test_load_file(self, unbuffered: bool = False):
        test = "LOAD FILE"
        if unbuffered:
            test += " UNBUFFERED"
        async with await self.setup() as conn:
            async with conn.cursor(SSCursor if unbuffered else SSCursor) as cur:
                filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "test_data",
                    "load_local_data.txt",
                )
                await cur.execute(
                    f"LOAD DATA LOCAL INFILE '{filename}' INTO TABLE {self.table}"
                    " FIELDS TERMINATED BY ','"
                )
                await cur.execute(f"SELECT COUNT(*) FROM {self.table}")
                self.assertEqual(22749, (await cur.fetchone())[0])

            await self.drop(conn)
        self.log(test)

    async def test_load_warnings(self):
        test = "LOAD WARNINGS"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "test_data",
                    "load_local_warn_data.txt",
                )
                await cur.execute(
                    f"LOAD DATA LOCAL INFILE '{filename}' INTO TABLE "
                    f"{self.table} FIELDS TERMINATED BY ','"
                )
                self.assertEqual(1, cur.warning_count)

                await cur.execute("SHOW WARNINGS")
                row = await cur.fetchone()

                self.assertEqual(ER.TRUNCATED_WRONG_VALUE_FOR_FIELD, row[1])
                self.assertIn(
                    "incorrect integer value",
                    row[2].lower(),
                )

            await self.drop(conn)
        self.log(test)

    # . utils
    async def setup(self, table: str = None, **kwargs) -> Connection:
        conn = await super().setup(table, **kwargs)
        tb = self.tb if table is None else table
        async with conn.cursor() as cur:
            await cur.execute("SET GLOBAL local_infile=ON")
            await cur.execute(f"CREATE TABLE {self.db}.{tb} (a INTEGER, b INTEGER)")
        await conn.commit()
        return conn


class TestOldIssues(TestCase):
    name: str = "Old Issues"

    async def test_all(self) -> None:
        await self.test_issue_3()
        await self.test_issue_4()
        await self.test_issue_5()
        await self.test_issue_6()
        await self.test_issue_8()
        await self.test_issue_13()
        await self.test_issue_15()
        await self.test_issue_16()

    async def test_issue_3(self) -> None:
        """undefined methods datetime_or_None, date_or_None"""
        test = "ISSUE 3"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"create table {self.table} (d date, t time, dt datetime, ts timestamp)"
                )
                await cur.execute(
                    f"insert into {self.table} (d, t, dt, ts) values (%s,%s,%s,%s)",
                    (None, None, None, None),
                )
                await cur.execute(f"select d from {self.table}")
                self.assertEqual(None, (await cur.fetchone())[0])
                await cur.execute(f"select t from {self.table}")
                self.assertEqual(None, (await cur.fetchone())[0])
                await cur.execute(f"select dt from {self.table}")
                self.assertEqual(None, (await cur.fetchone())[0])
                await cur.execute(f"select ts from {self.table}")
                self.assertIn(
                    type((await cur.fetchone())[0]),
                    (type(None), datetime.datetime),
                    "expected Python type None or datetime from SQL timestamp",
                )

            await self.drop(conn)
        self.log(test)

    async def test_issue_4(self):
        """can't retrieve TIMESTAMP fields"""
        test = "ISSUE 4"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"create table {self.table} (ts timestamp)")
                await cur.execute(f"insert into {self.table} (ts) values (now())")
                await cur.execute(f"select ts from {self.table}")
                self.assertTrue(
                    isinstance((await cur.fetchone())[0], datetime.datetime)
                )

            await self.drop(conn)
        self.log(test)

    async def test_issue_5(self):
        """query on information_schema.tables fails"""
        test = "ISSUE 5"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("select * from information_schema.tables")
        self.log(test)

    async def test_issue_6(self):
        """exception: TypeError: ord() expected a character, but string of length 0 found"""
        # ToDo: this test requires access to db 'mysql'.
        test = "ISSUE 6"
        async with await self.get_conn(database="mysql") as conn:
            async with conn.cursor() as cur:
                await cur.execute("select * from user")
        self.log(test)

    async def test_issue_8(self):
        """Primary Key and Index error when selecting data"""
        test = "ISSUE 8"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    CREATE TABLE {self.table} (`station` int NOT NULL DEFAULT '0', `dh`
                    datetime NOT NULL DEFAULT '2015-01-01 00:00:00', `echeance` int NOT NULL
                    DEFAULT '0', `me` double DEFAULT NULL, `mo` double DEFAULT NULL, PRIMARY
                    KEY (`station`,`dh`,`echeance`)) ENGINE=MyISAM DEFAULT CHARSET=latin1;
                    """
                )
                self.assertEqual(0, await cur.execute(f"SELECT * FROM {self.table}"))
                await cur.execute(
                    f"ALTER TABLE {self.table} ADD INDEX `idx_station` (`station`)"
                )
                self.assertEqual(0, await cur.execute(f"SELECT * FROM {self.table}"))

            await self.drop(conn)
        self.log(test)

    async def test_issue_13(self):
        """can't handle large result fields"""
        test = "ISSUE 13"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"create table {self.table} (t text)")
                # ticket says 18k
                size = 18 * 1024
                await cur.execute(
                    f"insert into {self.table} (t) values (%s)", ("x" * size,)
                )
                await cur.execute(f"select t from {self.table}")
                # use assertTrue so that obscenely huge error messages don't print
                r = (await cur.fetchone())[0]
                self.assertTrue("x" * size == r)

            await self.drop(conn)
        self.log(test)

    async def test_issue_15(self):
        """query should be expanded before perform character encoding"""
        test = "ISSUE 15"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"create table {self.table} (t varchar(32))")
                await cur.execute(
                    f"insert into {self.table} (t) values (%s)", ("\xe4\xf6\xfc",)
                )
                await cur.execute(f"select t from {self.table}")
                self.assertEqual("\xe4\xf6\xfc", (await cur.fetchone())[0])

            await self.drop(conn)
        self.log(test)

    async def test_issue_16(self):
        """Patch for string and tuple escaping"""
        test = "ISSUE 16"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"create table {self.table} (name varchar(32) primary key, email varchar(32))"
                )
                await cur.execute(
                    f"insert into {self.table} (name, email) values ('pete', 'floydophone')"
                )
                await cur.execute(
                    f"select email from {self.table} where name=%s", ("pete",)
                )
                self.assertEqual("floydophone", (await cur.fetchone())[0])

            await self.drop(conn)
        self.log(test)


class TestNewIssues(TestCase):
    name: str = "New Issues"

    async def test_all(self) -> None:
        await self.test_issue_33()
        await self.test_issue_34()
        await self.test_issue_36()
        await self.test_issue_37()
        await self.test_issue_38()
        await self.test_issue_54()

    async def test_issue_33(self):
        test = "ISSUE 33"
        async with await self.get_conn() as conn:
            table = f"{self.db}.hei\xdfe"
            async with conn.cursor() as cur:
                await cur.execute(f"drop table if exists {table}")
                await cur.execute(f"create table {table} (name varchar(32))")
                await cur.execute(f"insert into {table} (name) values ('Pi\xdfata')")
                await cur.execute(f"select name from {table}")
                self.assertEqual("Pi\xdfata", (await cur.fetchone())[0])
                await cur.execute(f"drop table {table}")
        self.log(test)

    async def test_issue_34(self):
        test = "ISSUE 34"
        try:
            await self.get_conn(port=1237)
            self.fail()
        except errors.OperationalError as err:
            self.assertEqual(2003, err.args[0])
            self.log(test)
        except Exception:
            self.fail()

    async def test_issue_36(self):
        test = "ISSUE 36"
        conn1 = await self.get_conn()
        conn2 = await self.get_conn()
        try:
            async with conn1.cursor() as cur:
                kill_id = None
                await cur.execute("show processlist")
                async for row in cur:
                    if row[7] == "show processlist":
                        kill_id = row[0]
                        break
                self.assertEqual(kill_id, conn1.thread_id)
                # now nuke the connection
                await conn2.kill(kill_id)
                # make sure this connection has broken
                with self.assertRaises(errors.OperationalError) as cm:
                    await cur.execute("show tables")
                    self.assertEqual(2013, cm.exception.args[0])

            # check the process list from the other connection
            await asyncio.sleep(0.1)
            async with conn2.cursor() as cur:
                await cur.execute("show processlist")
                ids = [row[0] for row in await cur.fetch()]
                self.assertFalse(kill_id in ids)

            self.log(test)

        finally:
            await conn1.close()
            await conn2.close()

    async def test_issue_37(self):
        test = "ISSUE 37"
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                self.assertEqual(1, await cur.execute("SELECT @foo"))
                self.assertEqual((None,), await cur.fetchone())
                self.assertEqual(0, await cur.execute("SET @foo = 'bar'"))
                await cur.execute("set @foo = 'bar'")
        self.log(test)

    async def test_issue_38(self):
        test = "ISSUE 38"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                datum = (
                    "a" * 1024 * 1023
                )  # reduced size for most default mysql installs
                await cur.execute(
                    f"create table {self.table} (id integer, data mediumblob)"
                )
                await cur.execute(f"insert into {self.table} values (1, %s)", (datum,))

            await self.drop(conn)
        self.log(test)

    async def test_issue_54(self):
        test = "ISSUE 54"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"create table {self.table} (id integer primary key)")
                await cur.execute(f"insert into {self.table} (id) values (7)")
                await cur.execute(
                    "select * from %s where %s"
                    % (
                        self.table,
                        " and ".join("%d=%d" % (i, i) for i in range(0, 100000)),
                    )
                )
                self.assertEqual(7, (await cur.fetchone())[0])

            await self.drop(conn)
        self.log(test)


class TestGitHubIssues(TestCase):
    name: str = "GitHub Issues"

    async def test_all(self) -> None:
        await self.test_issue_66()
        await self.test_issue_79()
        await self.test_issue_95()
        await self.test_issue_114()
        await self.test_issue_175()
        await self.test_issue_363()
        await self.test_issue_364()

    async def test_issue_66(self):
        """'Connection' object has no attribute 'insert_id'"""
        test = "ISSUE 66"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"create table {self.table} (id integer primary key auto_increment, x integer)"
                )
                await cur.execute(f"insert into {self.table} (x) values (1)")
                await cur.execute(f"insert into {self.table} (x) values (1)")
                self.assertEqual(2, conn.insert_id)

            await self.drop(conn)
        self.log(test)

    async def test_issue_79(self):
        """Duplicate field overwrites the previous one in the result of DictCursor"""
        test = "ISSUE 79"
        async with await self.get_conn() as conn:
            tb1 = f"{self.db}.a"
            tb2 = f"{self.db}.b"
            async with conn.cursor(DictCursor) as cur:
                await cur.execute(f"drop table if exists {tb1}")
                await cur.execute(f"CREATE TABLE {tb1} (id int, value int)")
                await cur.execute(f"insert into {tb1} values (%s, %s)", (1, 11))

                await cur.execute(f"drop table if exists {tb2}")
                await cur.execute(f"CREATE TABLE {tb2} (id int, value int)")
                await cur.execute(f"insert into {tb2} values (%s, %s)", (1, 22))

                await cur.execute(
                    f"SELECT * FROM {tb1} inner join {tb2} on a.id = b.id"
                )
                row = await cur.fetchone()
                self.assertEqual(row["id"], 1)
                self.assertEqual(row["value"], 11)
                self.assertEqual(row["b.id"], 1)
                self.assertEqual(row["b.value"], 22)

                await cur.execute(f"drop table {tb1}")
                await cur.execute(f"drop table {tb2}")
        self.log(test)

    async def test_issue_95(self):
        """Leftover trailing OK packet for "CALL my_sp" queries"""
        test = "ISSUE 95"
        async with await self.get_conn() as conn:
            proc: str = f"{self.db}.foo"
            async with conn.cursor() as cur:
                await cur.execute(f"DROP PROCEDURE IF EXISTS {proc}")
                await cur.execute(
                    f"""
                    CREATE PROCEDURE {proc} ()
                    BEGIN
                        SELECT 1;
                    END
                    """
                )
                await cur.execute(f"CALL {proc}()")
                await cur.execute("SELECT 1")
                self.assertEqual((await cur.fetchone())[0], 1)
                await cur.execute(f"DROP PROCEDURE IF EXISTS {proc}")
        self.log(test)

    async def test_issue_114(self):
        """autocommit is not set after reconnecting with ping()"""
        test = "ISSUE 114"
        async with await self.get_conn() as conn:
            await conn.set_autocommit(False)
            async with conn.cursor() as cur:
                await cur.execute("select @@autocommit;")
                self.assertFalse((await cur.fetchone())[0])
            await conn.close()
            await conn.ping()
            async with conn.cursor() as cur:
                await cur.execute("select @@autocommit;")
                self.assertFalse((await cur.fetchone())[0])

        # Ensure set_autocommit() is still working
        async with await self.get_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("select @@autocommit;")
                self.assertFalse((await cur.fetchone())[0])
            await conn.close()
            await conn.ping()
            await conn.set_autocommit(True)
            async with conn.cursor() as cur:
                await cur.execute("select @@autocommit;")
                self.assertTrue((await cur.fetchone())[0])
        self.log(test)

    async def test_issue_175(self):
        """The number of fields returned by server is read in wrong way"""
        test = "ISSUE 175"
        async with await self.get_conn() as conn:
            tb: str = f"{self.db}.test_field_count"
            async with conn.cursor() as cur:
                await cur.execute(f"drop table if exists {tb}")
                for length in (100, 200, 300):
                    columns = ", ".join(f"c{i} integer" for i in range(length))
                    sql = f"create table {tb} ({columns})"
                    try:
                        await cur.execute(sql)
                        await cur.execute(f"select * from {tb}")
                        self.assertEqual(length, cur.field_count)
                    finally:
                        await cur.execute(f"drop table if exists {tb}")
        self.log(test)

    async def test_issue_363(self):
        """Test binary / geometry types."""
        test = "ISSUE 363"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"CREATE TABLE {self.table} ( "
                    "id INTEGER PRIMARY KEY, geom LINESTRING NOT NULL /*!80003 SRID 0 */, "
                    "SPATIAL KEY geom (geom)) "
                    "ENGINE=MyISAM",
                )
                await cur.execute(
                    f"INSERT INTO {self.table} (id, geom) VALUES"
                    "(1998, ST_GeomFromText('LINESTRING(1.1 1.1,2.2 2.2)'))"
                )

                # select WKT
                await cur.execute(f"SELECT ST_AsText(geom) FROM {self.table}")
                row = await cur.fetchone()
                self.assertEqual(row, ("LINESTRING(1.1 1.1,2.2 2.2)",))

                # select WKB
                await cur.execute(f"SELECT ST_AsBinary(geom) FROM {self.table}")
                row = await cur.fetchone()
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
                await cur.execute(f"SELECT geom FROM {self.table}")
                row = await cur.fetchone()
                # don't assert the exact internal binary value, as it could
                # vary across implementations
                self.assertTrue(isinstance(row[0], bytes))

            await self.drop(conn)
        self.log(test)

    async def test_issue_364(self):
        """Test mixed unicode/binary arguments in executemany."""
        test = "ISSUE 364"
        async with await self.setup() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"create table {self.table} (value_1 binary(3), value_2 varchar(3)) "
                    "engine=InnoDB default charset=utf8mb4",
                )
                sql = f"insert into {self.table} (value_1, value_2) values (%s, %s)"
                usql = f"insert into {self.table} (value_1, value_2) values (%s, %s)"
                values = (b"\x00\xff\x00", "\xe4\xf6\xfc")

                # test single insert and select
                await cur.execute(sql, values)
                await cur.execute(f"select * from {self.table}")
                self.assertEqual(await cur.fetchone(), values)

                # test single insert unicode query
                await cur.execute(usql, values)

                # test multi insert and select
                await cur.execute(sql, args=(values, values, values))
                await cur.execute(f"select * from {self.table}")
                async for row in cur:
                    self.assertEqual(row, values)

                # test multi insert with unicode query
                await cur.execute(usql, args=(values, values, values))

            await self.drop(conn)
        self.log(test)


if __name__ == "__main__":
    for test in [
        TestConnection,
        TestAuthentication,
        TestConversion,
        TestCursor,
        TestLoadLocal,
        TestOldIssues,
        TestNewIssues,
        TestGitHubIssues,
    ]:
        tester: TestCase = test()
        asyncio.run(tester.test_all())
