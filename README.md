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
- PyMySQL: 1.1.1
- aiomysql: 0.2.0
- asyncmy: 0.2.9

```
# Unit: second | Lower is better
name        type    rows    insert-per-row  insert-bulk select-per-row  select-all
SQLCyCli    sync    50000   2.428453        0.367404    2.526141        0.078057
PyMySQL     sync    50000   2.821481        0.480322    4.784844        0.335978
SQLCyCli    async   50000   3.757844        0.340909    5.017284        0.157078
aiomysql    async   50000   3.845818        0.419444    5.764339        0.333526
asyncmy     async   50000   4.015180        0.484794    6.144809        0.337285
```

```
# Unit: second | Lower is better
name        type    rows    update-per-row  update-all  delete-per-row  delete-all
SQLCyCli    sync    50000   2.597837        0.327441    2.251010        0.131872
PyMySQL     sync    50000   3.044907        0.368951    2.789961        0.158141
SQLCyCli    async   50000   4.226546        0.369085    3.994125        0.139679
aiomysql    async   50000   3.792293        0.356109    3.589203        0.134762
asyncmy     async   50000   4.160017        0.362896    3.928555        0.145456
```

## Usage

### Use `connect()` to create a connection (`Sync` or `Async`) with the server.

```python
import asyncio
import sqlcycli

HOST = "localhost"
PORT = 3306
USER = "root"
PSWD = "password"

# Connection (Sync & Async)
async def test_connection() -> None:
    # Sync Connection - - - - - - - - - - - - - - - - - -
    with sqlcycli.connect(HOST, PORT, USER, PSWD) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)

    # Connection closed
    assert conn.closed()

    # Async Connection - - - - - - - - - - - - - - - - -
    async with sqlcycli.connect(HOST, PORT, USER, PSWD) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            assert await cur.fetchone() == (1,)

    # Connection closed
    assert conn.closed()

if __name__ == "__main__":
    asyncio.run(test_connection())
```

### Use `create_pool()` to create a Pool for managing and maintaining connections (`Sync` or `Async`) with the server.

```python
import asyncio
import sqlcycli

HOST = "localhost"
PORT = 3306
USER = "root"
PSWD = "password"

# Pool (Context: Connected)
async def test_pool_context_connected() -> None:
    async with sqlcycli.create_pool(HOST, PORT, USER, PSWD, min_size=1) as pool:
        # Pool is connected: 1 free connection (min_size=1)
        assert not pool.closed() and pool.free == 1

        # Sync Connection - - - - - - - - - - - - - - - - - -
        with pool.acquire() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                assert cur.fetchone() == (1,)

        # Async Connection - - - - - - - - - - - - - - - - -
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                assert await cur.fetchone() == (1,)

    # Pool closed
    assert pool.closed() and pool.total == 0

# Pool (Context: Disconnected)
async def test_pool_context_disconnected() -> None:
    with sqlcycli.create_pool(HOST, PORT, USER, PSWD, min_size=1) as pool:
        # Pool is not connected: 0 free connection (min_size=1)
        assert pool.closed() and pool.free == 0

        # Sync Connection - - - - - - - - - - - - - - - - - -
        with pool.acquire() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                assert cur.fetchone() == (1,)

        # Async Connection - - - - - - - - - - - - - - - - -
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                assert await cur.fetchone() == (1,)
        # 1 free async connection
        assert pool.free == 1

    # Pool closed
    assert pool.closed() and pool.total == 0

if __name__ == "__main__":
    asyncio.run(test_pool_context_connected())
    asyncio.run(test_pool_context_disconnected())
```

### Use the `Pool` class to create a Pool instance. `Must close manually`.

```python
import asyncio
import sqlcycli

HOST = "localhost"
PORT = 3306
USER = "root"
PSWD = "password"

# Pool (Instance: Connected)
async def test_pool_instance_connected() -> None:
    pool = await sqlcycli.create_pool(HOST, PORT, USER, PSWD, min_size=1)
    # Pool is connected: 1 free connection (min_size=1)
    assert not pool.closed() and pool.free == 1

    # Sync Connection - - - - - - - - - - - - - - - - - -
    with pool.acquire() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)

    # Async Connection - - - - - - - - - - - - - - - - -
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            assert await cur.fetchone() == (1,)

    # Close pool manually
    await pool.close()
    assert pool.closed() and pool.total == 0


# Pool (Instance: Disconnected)
async def test_pool_instance_disconnected() -> None:
    pool = sqlcycli.Pool(HOST, PORT, USER, PSWD, min_size=1)
    # Pool is not connected: 0 free connection (min_size=1)
    assert pool.closed() and pool.free == 0

    # Sync Connection - - - - - - - - - - - - - - - - - -
    with pool.acquire() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)

    # Async Connection - - - - - - - - - - - - - - - - -
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            assert await cur.fetchone() == (1,)
    # 1 free async connection
    assert pool.free == 1

    # Close pool manually
    await pool.close()
    assert pool.closed() and pool.total == 0

if __name__ == "__main__":
    asyncio.run(test_pool_instance_connected())
    asyncio.run(test_pool_instance_disconnected())
```

### Use the `sqlfunc` module to escape values for MySQL functions.

```python
import asyncio
import datetime
import sqlcycli
from sqlcycli import sqlfunc

HOST = "localhost"
PORT = 3306
USER = "root"
PSWD = "Password_123456"

# SQLFunction
def test_sqlfunction() -> None:
    with sqlcycli.connect(HOST, PORT, USER, PSWD) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT %s", sqlfunc.TO_DAYS(datetime.date(2007, 10, 7)))
            # SQLFunction 'TO_DAYS()' escaped as: TO_DAYS('2007-10-07')
            assert cur.executed_sql == "SELECT TO_DAYS('2007-10-07')"
            assert cur.fetchone() == (733321,)

    # Connection closed
    assert conn.closed()

if __name__ == "__main__":
    test_sqlfunction()
```

## Acknowledgements

SQLCyCli is build on top of the following open-source repositories:

- [aiomysql](https://github.com/aio-libs/aiomysql)
- [PyMySQL](https://github.com/PyMySQL/PyMySQL)

SQLCyCli is based on the following open-source repositories:

- [numpy](https://github.com/numpy/numpy)
- [orjson](https://github.com/ijl/orjson)
- [pandas](https://github.com/pandas-dev/pandas)
