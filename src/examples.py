import asyncio
import datetime
import sqlcycli
from sqlcycli import sqlfunc


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


# SQLFunction
def test_sqlfunction() -> None:
    with sqlcycli.connect(HOST, PORT, USER, PSWD) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT %s", sqlfunc.TO_DAYS(datetime.date(2007, 10, 7)))
            # SQLFunction 'TO_DAYS()' escaped to: TO_DAYS('2007-10-07')
            assert cur.executed_sql == "SELECT TO_DAYS('2007-10-07')"
            assert cur.fetchone() == (733321,)

    # Connection closed
    assert conn.closed()


if __name__ == "__main__":
    HOST = "localhost"
    PORT = 3306
    USER = "root"
    PSWD = "Password_123456"

    asyncio.run(test_connection())
    asyncio.run(test_pool_context_connected())
    asyncio.run(test_pool_context_disconnected())
    asyncio.run(test_pool_instance_connected())
    asyncio.run(test_pool_instance_disconnected())
    test_sqlfunction()
