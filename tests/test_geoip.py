"""
Test suite covering the geoip module.
"""

import pytest

from iisa.geoip import (
    GeoipResolver,
    _UrlHostStr,
    _get_ipaddr_location_info,
    _get_url_host,
    _resolve_host_ipaddr,
)
from iisa.typing import HttpUrlStr


class TestResolveUrlHostIpaddr:
    def test_get_host_from_url(self):
        ## Given
        url = HttpUrlStr("https://thegraph.com")

        ## When
        result = _get_url_host(url)

        ## Then
        assert result == "thegraph.com"

    def test_get_host_from_ipaddr_url(self):
        ## Given
        host = HttpUrlStr("https://192.168.0.1:8080/index.html")

        ## When
        result = _get_url_host(host)

        ## Then
        assert result == "192.168.0.1"

    def test_get_hot_from_url_with_no_host(self):
        ## Given
        host = HttpUrlStr("https://")

        ## When
        result = _get_url_host(host)

        ## Then
        assert result is None

    def test_resolve_ipaddr_from_host_str(self):
        ## Given
        # DNS domain that resolves to localhost (127.0.0.1)
        host = _UrlHostStr("localtest.me")

        ## When
        result = _resolve_host_ipaddr(host)

        ## Then
        assert result == "127.0.0.1"

    def test_resolve_ipaddr_from_host_str_with_no_resolution(self):
        ## Given
        # Invalid hostname
        host = _UrlHostStr("invalid-hostname.local")

        ## When
        result = _resolve_host_ipaddr(host)

        ## Then
        assert result is None


@pytest.mark.skip(reason="requires a new IPinfo.io API key")
class TestGetIpaddrLocationInfo:
    def test_get_geolocation_info_for_ipaddr(self, ipinfo_io_auth):
        ## Given
        # Use one of the Google APIs US-east URLs to ensure it's a US-based IP address
        url = HttpUrlStr("https://storage.us-east1.rep.googleapis.com")

        # Resolve the hostname to an IP address
        host = _get_url_host(url)
        ipaddr = _resolve_host_ipaddr(host)

        ## When
        result = _get_ipaddr_location_info(ipaddr, auth=ipinfo_io_auth)

        ## Then
        assert result["ip_addr"] == ipaddr
        assert result["org"].endswith("Google LLC")
        assert result["country"] == "US"
        assert result["latitude"] is not None
        assert result["longitude"] is not None


@pytest.mark.skip(reason="requires a new IPinfo.io API key")
class TestGeoipResolver:
    def test_resolve_url_host_info(self, ipinfo_io_auth):
        ## Given
        url = HttpUrlStr("https://dns.google")

        geoip = GeoipResolver(ipinfo_io_auth)

        ## When
        result = geoip.resolve_url_host_info(url)

        ## Then
        # Assert the IP address and geolocation information is returned
        assert result["ip_addr"] == "8.8.4.4"
        assert result["org"] == "AS15169 Google LLC"
        assert result["country"] == "US"
        assert result["latitude"] is not None
        assert result["longitude"] is not None

        # Assert caches are no longer empty
        assert geoip._host_ipaddr_cache_entries() == 1
        assert geoip._ipinfo_cache_entries() == 1

    def test_resolve_url_host_info_with_no_host_url(self, ipinfo_io_auth):
        ## Given
        url = HttpUrlStr("https://")

        geoip = GeoipResolver(ipinfo_io_auth)

        ## When
        result = geoip.resolve_url_host_info(url)

        ## Then
        # Assert no information is returned
        assert result["ip_addr"] is None
        assert result["org"] is None
        assert result["country"] is None
        assert result["latitude"] is None
        assert result["longitude"] is None

        # Assert cache is empty
        assert geoip._host_ipaddr_cache_entries() == 0
        assert geoip._ipinfo_cache_entries() == 0

    def test_resolve_url_host_info_with_non_resolvable_host(self, ipinfo_io_auth):
        ## Given
        url = HttpUrlStr("https://invalid-hostname.local")

        geoip = GeoipResolver(ipinfo_io_auth)

        ## When
        result = geoip.resolve_url_host_info(url)

        ## Then
        # Assert no information is returned
        assert result["ip_addr"] is None
        assert result["org"] is None
        assert result["country"] is None
        assert result["latitude"] is None
        assert result["longitude"] is None

        # Assert cache is empty
        assert geoip._host_ipaddr_cache_entries() == 0
        assert geoip._ipinfo_cache_entries() == 0

    def test_resolve_url_host_info_with_private_ipaddr(self, ipinfo_io_auth):
        ## Given
        # Resolve the hostname to the localhost IP address
        url = HttpUrlStr("https://localtest.me")

        geoip = GeoipResolver(ipinfo_io_auth)

        ## When
        result = geoip.resolve_url_host_info(url)

        ## Then
        # Assert the localhost IP address is returned, but no geolocation information
        assert result["ip_addr"] == "127.0.0.1"
        assert result["org"] is None
        assert result["country"] is None
        assert result["latitude"] is None
        assert result["longitude"] is None

        # Assert cache is not empty
        assert geoip._host_ipaddr_cache_entries() == 1
        assert geoip._ipinfo_cache_entries() == 0
