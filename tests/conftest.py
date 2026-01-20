# This file is used to define fixtures that are shared across multiple test files.
# See:
#  - https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files
#  - https://faker.readthedocs.io/en/master/pytest-fixtures.html

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
