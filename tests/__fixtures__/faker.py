import random
import string
from typing import Optional

from faker import Faker
from faker.providers import BaseProvider, geo, internet
from faker_airtravel import AirTravelProvider

from iisa.typing import (
    BASE58_ALPHABET,
    DeploymentId,
    EthAddressStr,
    IndexerId,
    IpfsHashStr,
    QueryIdStr,
    SubgraphId,
)


class CustomProvider(BaseProvider):
    """
    Domain-specific providers for generating random data.
    """

    def query_id(self, iata_code: Optional[str] = None) -> QueryIdStr:
        """
        Generate a random query ID in the format of "[a-f0-9]{16}-{IATA Code}"

        :param iata_code: Optional IATA code to use as the suffix.
        :return: A random query ID.
        """
        prefix = "".join(random.choices(string.hexdigits.lower(), k=16))
        suffix = iata_code or self.generator.airport_iata()
        return QueryIdStr(f"{prefix}-{suffix}")

    def eth_address(self) -> EthAddressStr:
        """
        Generate a random Ethereum address.

        :return: A random Ethereum address.
        """
        prefix = "0x"
        characters = string.hexdigits.lower()
        address = prefix + "".join(random.choices(characters, k=40))
        return EthAddressStr(address)

    def ipfs_hash(self) -> IpfsHashStr:
        """
        Generate a random IPFS hash (CIDv0).

        :return: A random IPFS hash.
        """
        prefix = "Qm"
        characters = string.ascii_letters + string.digits
        cid = prefix + "".join(random.choices(characters, k=44))
        return IpfsHashStr(cid)

    def indexer_id(self) -> IndexerId:
        """
        Generate a random Indexer ID.

        :return: A random Indexer ID.
        """
        return IndexerId(self.eth_address())

    def deployment_id(self) -> DeploymentId:
        """
        Generate a random subgraph deployment ID.

        :return: A random subgraph deployment ID.
        """
        return DeploymentId(self.ipfs_hash())

    def subgraph_id(self) -> SubgraphId:
        """
        Generate a random subgraph ID.

        :return: A random subgraph ID.
        """
        subgraph = "".join(random.choices(BASE58_ALPHABET, k=46))
        return SubgraphId(subgraph)


def init_faker_instance(faker: Faker):
    """
    Initialize a `Faker` instance with custom providers.
    """
    faker.add_provider(geo)
    faker.add_provider(internet)
    faker.add_provider(AirTravelProvider)
    faker.add_provider(CustomProvider)
