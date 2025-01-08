# Fast MySQL driver build in Cython (Sync and Async).

Created to be used in a project, this package is published to github for ease of management and installation across different modules.

## Installation

Install from `PyPi`

```bash
pip install sqlcycli
```

Install from `github`

```bash
pip install git+https://github.com/AresJef/SQLCyCli.git
```

For Linux systems, if you encounter the following error when installing the SQLCyCli dependency [mysqlclient](https://github.com/PyMySQL/mysqlclient):

```
Exception: Can not find valid pkg-config name.
Specify MYSQLCLIENT_CFLAGS and MYSQLCLIENT_LDFLAGS env vars manually
```

Try the following to fix dependency issue (source: [Stack Overflow](https://stackoverflow.com/questions/76585758/mysqlclient-cannot-install-via-pip-cannot-find-pkg-config-name-in-ubuntu)):

```bash
sudo apt-get install pkg-config python3-dev default-libmysqlclient-dev build-essential
```

## Requirements

- Python 3.10 or higher.
- MySQL 5.5 or higher.

## Features

- Written in [Cython](https://cython.org/) for optimal performance (especially for SELECT/INSERT query).
- All classes and methods are well documented and type annotated.
- Supports both `Sync` and `Async` connection to the server.
- API Compatiable with [PyMySQL](https://github.com/PyMySQL/PyMySQL) and [aiomysql](https://github.com/aio-libs/aiomysql).
- Support conversion (escape) for most of the native python types, and objects from libaray [numpy](https://github.com/numpy/numpy) and [pandas](https://github.com/pandas-dev/pandas). Does `NOT` support custom conversion (escape).

## Benchmark

The following result comes from [benchmark](./src/benchmark.py):

- Device: MacbookPro M1Pro(2E8P) 32GB
- Python: 3.12.4
- MySQL: 8.3.0
- mysqlclient: 2.2.4
- PyMySQL: 1.1.1
- aiomysql: 0.2.0
- asyncmy: 0.2.9

```
# Unit: second | Lower is better
name        type    rows    insert-per-row  insert-bulk select-per-row  select-all
mysqlclient sync    50000   1.729575        0.435661    1.719481        0.117943
SQLCyCli    sync    50000   2.165910        0.275736    2.215093        0.056679
PyMySQL     sync    50000   2.553401        0.404618    4.212548        0.325706
SQLCyCli    async   50000   3.347850        0.282364    4.153874        0.135656
aiomysql    async   50000   3.478428        0.394711    5.101733        0.321200
asyncmy     async   50000   3.665675        0.397671    5.483239        0.313418
```

```
# Unit: second | Lower is better
name        type    rows    update-per-row  update-all  delete-per-row  delete-all
mysqlclient sync    50000   1.735787        0.345561    1.531275        0.105109
SQLCyCli    sync    50000   2.241458        0.343359    2.078324        0.104441
PyMySQL     sync    50000   2.516349        0.344614    2.264735        0.104326
SQLCyCli    async   50000   3.465996        0.343864    3.269337        0.103967
aiomysql    async   50000   3.534125        0.344573    3.345815        0.104281
asyncmy     async   50000   3.695764        0.352104    3.460674        0.104523
```

## Usage

### Use `connect()` to create one connection (`Sync` or `Async`) to the server.

```python
import asyncio
import sqlcycli

HOST = "localhost"
PORT = 3306
USER = "root"
PSWD = "password"

# Synchronous Connection
def test_sync_connection() -> None:
    with sqlcycli.connect(HOST, PORT, USER, PSWD) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            res = cur.fetchone()
            assert res == (1,)
    # Connection closed
    assert conn.closed()

# Asynchronous Connection
async def test_async_connection() -> None:
    async with sqlcycli.connect(HOST, PORT, USER, PSWD) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            res = await cur.fetchone()
            assert res == (1,)
    # Connection closed
    assert conn.closed()

if __name__ == "__main__":
    test_sync_connection()
    asyncio.run(test_async_connection())
```

### Use `create_pool()` to create a Pool for managing and maintaining `Async` connections to the server.

```python
import asyncio
import sqlcycli

HOST = "localhost"
PORT = 3306
USER = "root"
PSWD = "password"

# Pool (Context Manager: Connected)
async def test_pool_context_connected() -> None:
    async with sqlcycli.create_pool(HOST, PORT, USER, PSWD, min_size=1) as pool:
        # Pool is connected: 1 free connection (min_size=1)
        assert not pool.closed() and pool.free == 1
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                res = await cur.fetchone()
                assert res == (1,)
    # Pool closed
    assert pool.closed() and pool.total == 0

# Pool (Context Manager: Disconnected)
async def test_pool_context_disconnected() -> None:
    with sqlcycli.create_pool(HOST, PORT, USER, PSWD, min_size=1) as pool:
        # Pool is not connected: 0 free connection (min_size=1)
        assert pool.closed() and pool.free == 0
        # Connect automatically
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                res = await cur.fetchone()
                assert res == (1,)
        # 1 free connection
        assert pool.free == 1
    # Pool closed
    assert pool.closed() and pool.total == 0

# Pool (Create Directly: Connected)
async def test_pool_direct_connected() -> None:
    pool = await sqlcycli.create_pool(HOST, PORT, USER, PSWD, min_size=1)
    # Pool is connected: 1 free connection (min_size=1)
    assert not pool.closed() and pool.free == 1
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            res = await cur.fetchone()
            assert res == (1,)
    # Close pool manually
    await pool.close()
    assert pool.closed() and pool.total == 0

if __name__ == "__main__":
    asyncio.run(test_pool_context_connected())
    asyncio.run(test_pool_context_disconnected())
    asyncio.run(test_pool_direct_connected())
```

### Use `sqlfunc` module to escape MySQL function values.

```python
import datetime
from sqlcycli import Connection, sqlfunc

HOST = "localhost"
PORT = 3306
USER = "root"
PSWD = "Password_123456"

conn = Connection(host=HOST, port=PORT, user=USER, password=PSWD)
conn.connect()
with conn.cursor() as cur:
    cur.execute("SELECT %s", sqlfunc.TO_DAYS(datetime.date(2007, 10, 7)))
    res = cur.fetchone()
    print(cur.executed_sql)
    # "SELECT TO_DAYS('2007-10-07')"
    print(res)
    # (733321,)
conn.close()
```

## Acknowledgements

SQLCyCli is build on top of the following open-source repositories:

- [aiomysql](https://github.com/aio-libs/aiomysql)
- [PyMySQL](https://github.com/PyMySQL/PyMySQL)

SQLCyCli is based on the following open-source repositories:

- [numpy](https://github.com/numpy/numpy)
- [orjson](https://github.com/ijl/orjson)
- [pandas](https://github.com/pandas-dev/pandas)
- [mysqlclient](https://github.com/PyMySQL/mysqlclient)
