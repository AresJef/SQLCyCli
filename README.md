## Fast MySQL driver build in Cython (Sync and Async).

Created to be used in a project, this package is published to github for ease of management and installation across different modules.

### Installation

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

### Requirements

- Python 3.10 or higher.
- MySQL 5.5 or higher.

### Features

- Written in [Cython](https://cython.org/) for optimal performance (especially for SELECT/INSERT query).
- All classes and methods are well documented and static typed.
- Supports both `Sync` and `Async` connection to the server.
- API Compatiable with [PyMySQL](https://github.com/PyMySQL/PyMySQL) and [aiomysql](https://github.com/aio-libs/aiomysql).
- Support conversion (escape) for most of the native python types, and objects from libaray [numpy](https://github.com/numpy/numpy) and [pandas](https://github.com/pandas-dev/pandas). Does `NOT` support custom conversion (escape).

### Benchmark

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
mysqlclient sync    50000   1.738263        0.430990    1.726277        0.117284
SQLCyCli    sync    50000   2.239658        0.293476    2.341073        0.062193
PyMySQL     sync    50000   2.662151        0.414243    4.245126        0.327544
SQLCyCli    async   50000   3.421401        0.276260    4.336240        0.139700
aiomysql    async   50000   3.579845        0.396187    5.151401        0.322439
asyncmy     async   50000   3.793308        0.409774    5.477049        0.318582
```

```
# Unit: second | Lower is better
name        type    rows    update-per-row  update-all  delete-per-row  delete-all
mysqlclient sync    50000   1.768599        0.356437    1.571645        0.108843
SQLCyCli    sync    50000   2.282858        0.368871    2.105368        0.113834
PyMySQL     sync    50000   2.560405        0.346608    2.307289        0.105239
SQLCyCli    async   50000   3.475850        0.346580    3.335821        0.105670
aiomysql    async   50000   3.574864        0.347389    3.261974        0.105169
asyncmy     async   50000   3.762817        0.344875    3.575200        0.106150
```

### Usage

#### Use `connect()` to create one connection (`Sync` or `Async`) to the server.

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

#### Use `create_pool()` to create a Pool for managing and maintaining `Async` connections to the server.

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

### Acknowledgements

SQLCyCli is build on top of the following open-source repositories:

- [aiomysql](https://github.com/aio-libs/aiomysql)
- [PyMySQL](https://github.com/PyMySQL/PyMySQL)

SQLCyCli is based on the following open-source repositories:

- [numpy](https://github.com/numpy/numpy)
- [orjson](https://github.com/ijl/orjson)
- [pandas](https://github.com/pandas-dev/pandas)
- [mysqlclient](https://github.com/PyMySQL/mysqlclient)
