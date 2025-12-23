# This file is used to define fixtures that are shared across multiple test files.
# See:
#  - https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files
#  - https://faker.readthedocs.io/en/master/pytest-fixtures.html

import os

import pytest
from faker import Faker

from __fixtures__.faker import init_faker_instance


@pytest.fixture(scope="session", autouse=True)
def faker() -> Faker:
    """
    Create a new instance of `Faker` configured with the domain specific providers for each test session.
    """
    faker = Faker()
    init_faker_instance(faker)
    return faker


@pytest.fixture(scope="session")
def ipinfo_io_auth() -> str:
    """
    Retrieve the IPinfo.io API key from the environment.

    If the `IT_TEST_IPINFO_IO_AUTH` environment variable is not set, the tests that require it will be skipped.
    """
    auth = os.getenv("IT_TEST_IPINFO_IO_AUTH")
    if auth is None:
        pytest.skip("IT_TEST_IPINFO_IO_AUTH env variable is not set")

    return auth
