import asyncio
import sqlcycli


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
    HOST = "localhost"
    PORT = 3306
    USER = "root"
    PSWD = "Password_123456"

    test_sync_connection()
    asyncio.run(test_async_connection())
    asyncio.run(test_pool_context_connected())
    asyncio.run(test_pool_context_disconnected())
    asyncio.run(test_pool_direct_connected())
