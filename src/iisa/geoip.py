"""
GeoIP utilities for resolving IP addresses and getting their geolocation information.
"""

import gzip
import json
import logging
import socket
from struct import unpack
from typing import Dict, NewType, Optional, TypedDict
from urllib.parse import urlparse

import requests
from requests.exceptions import ConnectionError as ReqConnectionError, HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .typing import HttpUrlStr

__all__ = [
    "IpInfoLocation",
    "GeoipResolver",
]

logger = logging.getLogger(__name__)

_UrlHostStr = NewType("_UrlHostStr", str)
_IpAddressStr = NewType("_IpAddressStr", str)

IpInfoLocation = TypedDict(
    "IpInfoLocation",
    {
        "ip_addr": Optional[str],
        "org": Optional[str],
        "country": Optional[str],
        "latitude": Optional[float],
        "longitude": Optional[float],
    },
)


def _resolve_host_ipaddr(host: _UrlHostStr) -> Optional[_IpAddressStr]:
    """
    Use DNS resolution to get the IP address of a hostname.

    :param host: Hostname to resolve to an IP address.

    :returns: IP address if host is resolved, otherwise None.
    """
    try:
        (hostname, alias, ip_addrs) = socket.gethostbyname_ex(host)
    except socket.gaierror:
        return None

    # Sort IP addresses to get a deterministic result
    ip_addrs = sorted(ip_addrs)

    if len(ip_addrs) == 0:
        return None

    return _IpAddressStr(ip_addrs[0])


def _is_private_ipaddr(ip_addr: _IpAddressStr) -> bool:
    """
    Check if an IP address is a private address (RFC 1918 & RFC 3330).

    The following private IP address ranges are checked:

    | IP Network   | Subnet Mask  | RFC                                     |
    |--------------|--------------|-----------------------------------------|
    | 127.0.0.0    | 255.0.0.0    | https://www.rfc-editor.org/rfc/rfc3330  |
    | 192.168.0.0  | 255.255.0.0  | https://www.rfc-editor.org/rfc/rfc1918  |
    | 172.16.0.0   | 255.240.0.0  | https://www.rfc-editor.org/rfc/rfc1918  |
    | 10.0.0.0     | 255.0.0.0    | https://www.rfc-editor.org/rfc/rfc1918  |

    Source: https://stackoverflow.com/a/8339939/1099999

    :param ip_addr: IP address to check.
    :return: True if the IP address is private, otherwise False.
    """
    (ip,) = unpack("!I", socket.inet_pton(socket.AF_INET, ip_addr))  # type: ignore

    private_networks = (
        # 127.0.0.0,   255.0.0.0   https://www.rfc-editor.org/rfc/rfc3330
        (0x7F000000, 0xFF000000),
        # 192.168.0.0, 255.255.0.0 https://www.rfc-editor.org/rfc/rfc1918
        (0xC0A80000, 0xFFFF0000),
        # 172.16.0.0,  255.240.0.0 https://www.rfc-editor.org/rfc/rfc1918
        (0xAC100000, 0xFFF00000),
        # 10.0.0.0,    255.0.0.0   https://www.rfc-editor.org/rfc/rfc1918
        (0x0A000000, 0xFF000000),
    )
    return any((ip & mask) == network for network, mask in private_networks)


def _get_url_host(url: HttpUrlStr) -> Optional[_UrlHostStr]:
    """
    Get the hostname from a URL.

    :param url: URL to extract the hostname from.

    :returns: Hostname if it can be extracted, otherwise None.
    """
    try:
        parsed_url = urlparse(url)

        # If the URL does not have a hostname, return None
        if not parsed_url.hostname or parsed_url.hostname == "":
            return None

        return _UrlHostStr(parsed_url.hostname)
    except ValueError:
        return None


_ExceptionsToRetry = (ConnectionError, ReqConnectionError, HTTPError, socket.timeout)


@retry(
    retry=retry_if_exception_type(_ExceptionsToRetry),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, max=60),
)
def _get_ipaddr_location_info(ip_addr: _IpAddressStr, *, auth: str) -> IpInfoLocation:
    """
    Fetch location and organizational details for a given IP address using an external API (ipinfo.io).

    This function makes an HTTP request to the ipinfo.io API to retrieve geographical and
    organizational information associated with the provided IP address. It includes a retry
    mechanism to handle potential network issues or API failures.

    :param ip_addr: The IP address to query.
    :param auth: The API token for ipinfo.io.
    :returns: A dictionary containing the following keys:
        - 'org': Organization associated with the IP
        - 'loc': Geographical coordinates
        - 'ip': The queried IP address
    """
    try:
        response = requests.get(
            f"https://ipinfo.io/{ip_addr}/json?token={auth}", timeout=5
        )
        response.raise_for_status()  # Raise a HTTPError in case of bad response.

        # Try to decode the content manually
        try:
            data = response.json()

        except requests.exceptions.JSONDecodeError:
            # If JSON decoding fails, try to decompress manually
            decompressed_content = gzip.decompress(response.content)
            data = json.loads(decompressed_content)

        ip_addr = data.get("ip", None)
        org = data.get("org", None)
        country = data.get("country", None)

        # Extract the latitude and longitude from the 'loc' field
        loc = data.get("loc", None)
        if loc is not None:
            # Try to convert the latitude and longitude to floats
            try:
                (latitude, longitude) = loc.split(",")
                latitude = float(latitude)
                longitude = float(longitude)
            except ValueError:
                latitude = longitude = None
        else:
            latitude = longitude = None

        return {
            "ip_addr": ip_addr,
            "org": org,
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
        }

    # If there's been a connection error then we can raise the issue to the retry decerator and retry the connection
    except _ExceptionsToRetry as e:
        logger.debug(f"Failed to retrieve IP details: {e}")
        raise  # Raise to trigger retry decorator

    except Exception as e:
        logger.error(f"Unexpected error when retrieving IP details: {e}")
        return {
            "ip_addr": ip_addr,
            "country": None,
            "org": None,
            "latitude": None,
            "longitude": None,
        }


class GeoipResolver:
    """
    A simple cache-based resolver for IP addresses using the ipinfo.io API.
    """

    def __init__(self, auth: str) -> None:
        self._ipinfo_io_auth = auth

        self._host_ipaddr_cache: Dict[_UrlHostStr, _IpAddressStr] = {}
        self._ipinfo_cache: Dict[_IpAddressStr, IpInfoLocation] = {}

    def _host_ipaddr_cache_entries(self) -> int:
        """
        Get the number of entries in the host-to-IP address cache.

        This is a helper method for testing purposes.

        :return: The number of entries in the cache.
        """
        return len(self._host_ipaddr_cache)

    def _ipinfo_cache_entries(self) -> int:
        """
        Get the number of entries in the IP address geolocation cache.

        This is a helper method for testing purposes.

        :return: The number of entries in the cache.
        """
        return len(self._ipinfo_cache)

    def resolve_url_host_info(self, url: HttpUrlStr) -> IpInfoLocation:
        """
        Resolve the geolocation information for a given URL.

        :param url: The URL to resolve.
        :return: A dictionary containing the geolocation information.
        """
        # Extract the host from the URL
        url_host = _get_url_host(url)

        # Bail out if the URL host cannot be extracted
        if url_host is None:
            return {
                "ip_addr": None,
                "org": None,
                "country": None,
                "latitude": None,
                "longitude": None,
            }

        # If the IP address for the host is not in the cache,
        # resolve it and cache the result
        ipaddr = self._host_ipaddr_cache.get(url_host, _resolve_host_ipaddr(url_host))
        if ipaddr is not None:
            self._host_ipaddr_cache[url_host] = ipaddr
        else:
            # Bail out if the IP address is unknown
            return {
                "ip_addr": None,
                "org": None,
                "country": None,
                "latitude": None,
                "longitude": None,
            }

        # If the IP address is private, return None
        if _is_private_ipaddr(ipaddr):
            return {
                "ip_addr": ipaddr,
                "org": None,
                "country": None,
                "latitude": None,
                "longitude": None,
            }

        # If the IP address geolocation info is not in the cache,
        # resolve it and cache the result
        if ipaddr not in self._ipinfo_cache:
            self._ipinfo_cache[ipaddr] = _get_ipaddr_location_info(
                ipaddr, auth=self._ipinfo_io_auth
            )

        return self._ipinfo_cache[ipaddr]
