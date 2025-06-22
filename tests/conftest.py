import asyncio

import pytest
from dotenv import load_dotenv

load_dotenv(".env.pytest", override=True)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
