import asyncio, time, warnings
import pandas as pd
import aiomysql, asyncmy, MySQLdb, pymysql, sqlcycli

warnings.filterwarnings("ignore", category=Warning)


# Benchmark
class Benchmark:
    # . server
    host: str = "localhost"
    user: str = "root"
    password: str = "Password_123456"
    # . query
    db: str = "test"
    tb: str = "benchmark"
    data: tuple = (
        1.23,
        "2021-01-01",
        "2020-07-16 22:49:54",
        2.34,
        "SQLCyCli",
        1,
    )
    sql_create_db: str = f"CREATE DATABASE IF NOT EXISTS `{db}`"
    sql_drop_tb = f"DROP TABLE IF EXISTS {db}.`{tb}`"
    sql_create_tb = f"""
    CREATE TABLE IF NOT EXISTS {db}.`{tb}` (
        `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT,
        `decimal` decimal(10,2) DEFAULT NULL,
        `date` date DEFAULT NULL,
        `datetime` datetime DEFAULT NULL,
        `float` float DEFAULT NULL,
        `string` varchar(200) DEFAULT NULL,
        `tinyint` tinyint DEFAULT NULL,
        PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """
    sql_truncate = f"TRUNCATE TABLE {db}.`{tb}`"
    sql_insert = f"""
    INSERT INTO {db}.`{tb}` (`decimal`,`date`,`datetime`,`float`,`string`,`tinyint`) 
    VALUES (%s,%s,%s,%s,%s,%s)
    """
    sql_select_per_row = f"SELECT * from {db}.`{tb}` WHERE `id` = %s"
    sql_select_all = f"SELECT * from {db}.`{tb}`"
    sql_update_per_row = f"UPDATE {db}.`{tb}` SET `string` = %s WHERE `id` = %s"
    sql_update_all = f"UPDATE {db}.`{tb}` SET `string` = %s"
    sql_delete_per_row = f"DELETE from {db}.`{tb}` WHERE `id` = %s"
    sql_delete_all = f"DELETE from {db}.`{tb}`"

    def __init__(self) -> None:
        # . stats
        self._start: float = None
        self._ended: float = None
        self._stats: dict = {}

    # Stats
    @property
    def stats(self) -> dict:
        return self._stats

    # Utils
    def is_conn_closed(self, conn) -> bool:
        try:
            return conn.closed()
        except Exception:
            try:
                return conn.closed
            except Exception:
                try:
                    return conn._closed
                except Exception:
                    try:
                        return not conn.connected
                    except Exception:
                        return False

    def log_start(self) -> None:
        self._start = time.perf_counter()

    def log_ended(self) -> None:
        self._ended = time.perf_counter()

    def get_duration(self) -> float:
        return self._ended - self._start


# Sync --------------------------------------------------------------------------------------------------------------------------
class Benchmark_Sync(Benchmark):
    def run(self, name: str, conn_cls: type, rows: int = 5_0000) -> None:
        print(f"Benchmarking: {name} (Sync)...".ljust(60), end="\r")

        self._stats["name"] = name
        self._stats["type"] = "sync"
        self._stats["rows"] = rows
        data = [self.data for _ in range(rows)]
        with conn_cls(host=self.host, user=self.user, password=self.password) as conn:
            # Connect ----------------------------------------------------------
            if self.is_conn_closed(conn):
                if hasattr(conn, "_connect"):
                    conn._connect()
                elif hasattr(conn, "connect"):
                    conn.connect()
            # Setup ------------------------------------------------------------
            with conn.cursor() as cur:
                cur.execute(self.sql_create_db)
                cur.execute(self.sql_drop_tb)
                cur.execute(self.sql_create_tb)
            # Insert -----------------------------------------------------------
            # . insert per row
            self.log_start()
            with conn.cursor() as cur:
                for row in data:
                    cur.execute(self.sql_insert, row)
            conn.commit()
            self.log_ended()
            self._stats["insert-per-row"] = self.get_duration()
            # . truncate
            with conn.cursor() as cur:
                cur.execute(self.sql_truncate)
            # . insert bulk
            self.log_start()
            with conn.cursor() as cur:
                cur.executemany(self.sql_insert, data)
            conn.commit()
            self.log_ended()
            self._stats["insert-bulk"] = self.get_duration()

            # Select -----------------------------------------------------------
            # . select per row
            self.log_start()
            with conn.cursor() as cur:
                for i in range(rows):
                    cur.execute(self.sql_select_per_row, (i + 1,))
                    res = cur.fetchall()
                    assert len(res) == 1
            self.log_ended()
            self._stats["select-per-row"] = self.get_duration()
            # . select all
            self.log_start()
            with conn.cursor() as cur:
                cur.execute(self.sql_select_all)
                res = cur.fetchall()
                assert len(res) == rows
            self.log_ended()
            self._stats["select-all"] = self.get_duration()

            # Update -----------------------------------------------------------
            # . update per row
            self.log_start()
            with conn.cursor() as cur:
                for i in range(rows):
                    cur.execute(self.sql_update_per_row, ("UPDATED", i + 1))
            conn.commit()
            self.log_ended()
            self._stats["update-per-row"] = self.get_duration()
            # . update all
            self.log_start()
            with conn.cursor() as cur:
                cur.execute(self.sql_update_all, ("SQLCyCli",))
            conn.commit()
            self.log_ended()
            self._stats["update-all"] = self.get_duration()

            # Delete -----------------------------------------------------------
            # . delete per row
            self.log_start()
            with conn.cursor() as cur:
                for i in range(rows):
                    cur.execute(self.sql_delete_per_row, (i + 1,))
            conn.commit()
            self.log_ended()
            self._stats["delete-per-row"] = self.get_duration()
            # . re-insert
            with conn.cursor() as cur:
                cur.executemany(self.sql_insert, data)
            conn.commit()
            # . delete all
            self.log_start()
            with conn.cursor() as cur:
                cur.execute(self.sql_delete_all)
            conn.commit()
            self.log_ended()
            self._stats["delete-all"] = self.get_duration()

            # Cleanup ----------------------------------------------------------
            with conn.cursor() as cur:
                cur.execute(self.sql_drop_tb)

        print(f"Finish Benchmark: {name} (Sync)".ljust(60))


# Async -------------------------------------------------------------------------------------------------------------------------
class Benchmark_Async(Benchmark):
    async def run(self, name: str, conn_cls: type, rows: int = 5_0000) -> None:
        print(f"Benchmarking: {name} (Async)...".ljust(60), end="\r")

        self._stats["name"] = name
        self._stats["type"] = "async"
        self._stats["rows"] = rows
        data = [self.data for _ in range(rows)]
        async with conn_cls(
            host=self.host, user=self.user, password=self.password
        ) as conn:
            # Connect ----------------------------------------------------------
            if self.is_conn_closed(conn):
                if hasattr(conn, "_connect"):
                    await conn._connect()
                elif hasattr(conn, "connect"):
                    await conn.connect()
            # Setup ------------------------------------------------------------
            async with conn.cursor() as cur:
                await cur.execute(self.sql_create_db)
                await cur.execute(self.sql_drop_tb)
                await cur.execute(self.sql_create_tb)
            # Insert -----------------------------------------------------------
            # . insert per row
            self.log_start()
            async with conn.cursor() as cur:
                for row in data:
                    await cur.execute(self.sql_insert, row)
            await conn.commit()
            self.log_ended()
            self._stats["insert-per-row"] = self.get_duration()
            # . truncate
            async with conn.cursor() as cur:
                await cur.execute(self.sql_truncate)
            # . insert bulk
            self.log_start()
            async with conn.cursor() as cur:
                await cur.executemany(self.sql_insert, data)
            await conn.commit()
            self.log_ended()
            self._stats["insert-bulk"] = self.get_duration()

            # Select -----------------------------------------------------------
            # . select per row
            self.log_start()
            async with conn.cursor() as cur:
                for i in range(rows):
                    await cur.execute(self.sql_select_per_row, (i + 1,))
                    res = await cur.fetchall()
                    assert len(res) == 1
            self.log_ended()
            self._stats["select-per-row"] = self.get_duration()
            # . select all
            self.log_start()
            async with conn.cursor() as cur:
                await cur.execute(self.sql_select_all)
                res = await cur.fetchall()
                assert len(res) == rows
            self.log_ended()
            self._stats["select-all"] = self.get_duration()

            # Update -----------------------------------------------------------
            # . update per row
            self.log_start()
            async with conn.cursor() as cur:
                for i in range(rows):
                    await cur.execute(self.sql_update_per_row, ("UPDATED", i + 1))
            await conn.commit()
            self.log_ended()
            self._stats["update-per-row"] = self.get_duration()
            # . update all
            self.log_start()
            async with conn.cursor() as cur:
                await cur.execute(self.sql_update_all, ("SQLCyCli",))
            await conn.commit()
            self.log_ended()
            self._stats["update-all"] = self.get_duration()

            # Delete -----------------------------------------------------------
            # . delete per row
            self.log_start()
            async with conn.cursor() as cur:
                for i in range(rows):
                    await cur.execute(self.sql_delete_per_row, (i + 1,))
            await conn.commit()
            self.log_ended()
            self._stats["delete-per-row"] = self.get_duration()
            # . re-insert
            async with conn.cursor() as cur:
                await cur.executemany(self.sql_insert, data)
            await conn.commit()
            # . delete all
            self.log_start()
            async with conn.cursor() as cur:
                await cur.execute(self.sql_delete_all)
            await conn.commit()
            self.log_ended()
            self._stats["delete-all"] = self.get_duration()

            # Cleanup ----------------------------------------------------------
            async with conn.cursor() as cur:
                await cur.execute(self.sql_drop_tb)

        print(f"Finish Benchmark: {name} (Async)".ljust(60))


# Main
if __name__ == "__main__":
    rows = 50_000
    stats = []
    for cls, args in (
        (Benchmark_Sync, {"name": "mysqlclient", "conn_cls": MySQLdb.Connection}),
        (Benchmark_Sync, {"name": "SQLCyCli", "conn_cls": sqlcycli.Connection}),
        (Benchmark_Sync, {"name": "PyMySQL", "conn_cls": pymysql.Connection}),
        (Benchmark_Async, {"name": "SQLCyCli", "conn_cls": sqlcycli.aio.Connection}),
        (Benchmark_Async, {"name": "aiomysql", "conn_cls": aiomysql.Connection}),
        (Benchmark_Async, {"name": "asyncmy", "conn_cls": asyncmy.Connection}),
    ):
        benchmark = cls()
        args |= {"rows": rows}
        if asyncio.iscoroutinefunction(benchmark.run):
            asyncio.run(benchmark.run(**args))
        else:
            benchmark.run(**args)
        stats.append(benchmark.stats)

    print(pd.DataFrame(stats))
