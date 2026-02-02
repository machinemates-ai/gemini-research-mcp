"""
Unit tests for content.py - SSRF protection and content extraction.

These tests run without network access using mocks where needed.
"""

import pytest

from gemini_research_mcp.content import (
    FetchResult,
    is_private_ip,
    validate_url,
)


class TestSSRFProtection:
    """Tests for SSRF protection functions."""

    def test_localhost_blocked(self):
        """localhost should be blocked."""
        assert is_private_ip("localhost") is True
        assert is_private_ip("LOCALHOST") is True
        assert is_private_ip("127.0.0.1") is True

    def test_private_ipv4_blocked(self):
        """Private IPv4 ranges should be blocked."""
        # 10.x.x.x
        assert is_private_ip("10.0.0.1") is True
        assert is_private_ip("10.255.255.255") is True

        # 172.16.x.x - 172.31.x.x
        assert is_private_ip("172.16.0.1") is True
        assert is_private_ip("172.31.255.255") is True

        # 192.168.x.x
        assert is_private_ip("192.168.0.1") is True
        assert is_private_ip("192.168.255.255") is True

    def test_cloud_metadata_blocked(self):
        """Cloud metadata endpoints should be blocked."""
        # AWS/GCP/Azure metadata
        assert is_private_ip("169.254.169.254") is True

        # GCP metadata
        assert is_private_ip("metadata.google.internal") is True
        assert is_private_ip("metadata.goog") is True

    def test_ipv6_loopback_blocked(self):
        """IPv6 loopback should be blocked."""
        assert is_private_ip("::1") is True
        assert is_private_ip("[::1]") is True

    def test_ipv6_private_ranges_blocked(self):
        """IPv6 private ranges should be blocked."""
        # Unique local (fd00::/8)
        assert is_private_ip("fd12:3456:789a::1") is True

        # Link-local (fe80::/10)
        assert is_private_ip("fe80::1") is True

    def test_public_ips_allowed(self):
        """Public IPs should be allowed."""
        # Google DNS
        assert is_private_ip("8.8.8.8") is False

        # Cloudflare DNS
        assert is_private_ip("1.1.1.1") is False


class TestURLValidation:
    """Tests for URL validation."""

    def test_valid_https_url(self):
        """Valid HTTPS URL should pass."""
        is_valid, error = validate_url("https://example.com/page")
        assert is_valid is True
        assert error == ""

    def test_valid_http_url(self):
        """Valid HTTP URL should pass."""
        is_valid, error = validate_url("http://example.com/page")
        assert is_valid is True
        assert error == ""

    def test_missing_scheme_rejected(self):
        """URL without scheme should be rejected."""
        is_valid, error = validate_url("example.com/page")
        assert is_valid is False
        assert "scheme" in error.lower() or "host" in error.lower()

    def test_file_scheme_rejected(self):
        """file:// URLs should be rejected."""
        is_valid, error = validate_url("file:///etc/passwd")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_ftp_scheme_rejected(self):
        """FTP URLs should be rejected."""
        is_valid, error = validate_url("ftp://ftp.example.com/file")
        assert is_valid is False
        assert "scheme" in error.lower()

    def test_localhost_url_rejected(self):
        """localhost URLs should be rejected."""
        is_valid, error = validate_url("http://localhost:8080/api")
        assert is_valid is False
        assert "SSRF" in error or "private" in error.lower()

    def test_private_ip_url_rejected(self):
        """Private IP URLs should be rejected."""
        is_valid, error = validate_url("http://192.168.1.1/admin")
        assert is_valid is False
        assert "SSRF" in error or "private" in error.lower()

    def test_metadata_url_rejected(self):
        """Cloud metadata URLs should be rejected."""
        is_valid, error = validate_url("http://169.254.169.254/latest/meta-data/")
        assert is_valid is False
        assert "SSRF" in error or "private" in error.lower()


class TestFetchResult:
    """Tests for FetchResult dataclass."""

    def test_successful_result(self):
        """Successful fetch should have content."""
        result = FetchResult(
            url="https://example.com",
            title="Example Page",
            content="# Hello World\n\nThis is content.",
            word_count=4,
        )
        assert result.url == "https://example.com"
        assert result.title == "Example Page"
        assert result.content == "# Hello World\n\nThis is content."
        assert result.word_count == 4
        assert result.error is None

    def test_error_result(self):
        """Error result should have error field set."""
        result = FetchResult(
            url="https://blocked.internal",
            title=None,
            content="",
            word_count=0,
            error="SSRF blocked: private IP",
        )
        assert result.url == "https://blocked.internal"
        assert result.error == "SSRF blocked: private IP"
        assert result.content == ""
        assert result.word_count == 0


class TestFetchWebpageAsync:
    """Async tests for fetch_webpage function."""

    @pytest.mark.asyncio
    async def test_ssrf_blocked_returns_error(self):
        """Fetching private IP should return error without making request."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("http://192.168.1.1/admin")

        assert result.error is not None
        assert "SSRF" in result.error or "private" in result.error.lower()
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_localhost_blocked(self):
        """Fetching localhost should return error."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("http://localhost:3000/api")

        assert result.error is not None
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_metadata_endpoint_blocked(self):
        """Fetching cloud metadata should return error."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("http://169.254.169.254/latest/meta-data/")

        assert result.error is not None
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_invalid_scheme_rejected(self):
        """Non-http(s) schemes should be rejected."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("file:///etc/passwd")

        assert result.error is not None
        assert "scheme" in result.error.lower()
