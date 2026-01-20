"""
IISA-specific type hints.
"""

from typing import NewType

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

QueryIdStr = NewType("QueryIdStr", str)
HttpUrlStr = NewType("HttpUrlStr", str)
SubgraphId = NewType("SubgraphId", str)
IpfsHashStr = NewType("IpfsHashStr", str)
DeploymentId = IpfsHashStr
EthAddressStr = NewType("EthAddressStr", str)
IndexerId = EthAddressStr
