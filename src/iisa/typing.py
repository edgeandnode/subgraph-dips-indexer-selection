"""
IISA-specific type hints.
"""

from typing import NewType

QueryIdStr = NewType("QueryIdStr", str)
IpfsHashStr = NewType("IpfsHashStr", str)
DeploymentId = IpfsHashStr
EthAddressStr = NewType("EthAddressStr", str)
IndexerId = EthAddressStr
