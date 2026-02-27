"""
Unit tests for content.py - SSRF protection and content extraction.

These tests run without network access using mocks where needed.
"""

import pytest

from gemini_research_mcp.content import (
    FetchResult,
    _slice_content,
    check_robots_txt,
    is_private_ip,
    validate_proxy_url,
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

    def test_proxy_url_validation(self):
        """Proxy URLs should use SSRF-safe validation."""
        ok, err = validate_proxy_url("https://example.com:3128")
        assert ok is True
        assert err == ""

        ok, err = validate_proxy_url("http://127.0.0.1:3128")
        assert ok is False
        assert "proxy_url" in err.lower()


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
        assert result.is_truncated is False
        assert result.total_content_length == 0


class TestChunking:
    """Tests for chunk slicing helper."""

    def test_slice_content_with_max_length(self):
        """Should return truncated chunk and mark truncated."""
        chunk, is_truncated, total_len = _slice_content("abcdefghijklmnopqrstuvwxyz", 5, 7)

        assert chunk == "fghijkl"
        assert is_truncated is True
        assert total_len == 26

    def test_slice_content_to_end(self):
        """Should not mark truncated when reaching end."""
        chunk, is_truncated, total_len = _slice_content("abcdefghijklmnopqrstuvwxyz", 20, 6)

        assert chunk == "uvwxyz"
        assert is_truncated is False
        assert total_len == 26

    def test_slice_content_out_of_bounds(self):
        """Out-of-bounds start index should return empty chunk."""
        chunk, is_truncated, total_len = _slice_content("abc", 100, 10)

        assert chunk == ""
        assert is_truncated is False
        assert total_len == 3


class TestRobotsCache:
    """Tests for robots cache behavior."""

    @pytest.mark.asyncio
    async def test_check_robots_cached_allow_none(self):
        """Cached None parser should allow fetch."""
        import gemini_research_mcp.content as content

        content._ROBOTS_CACHE.clear()
        content._ROBOTS_CACHE["https://example.com"] = None

        allowed = await check_robots_txt("https://example.com/docs")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_robots_cached_disallow(self):
        """Cached parser should be consulted for can_fetch."""
        import gemini_research_mcp.content as content

        class FakeParser:
            def can_fetch(self, url: str, user_agent: str) -> bool:
                return False

        content._ROBOTS_CACHE.clear()
        content._ROBOTS_CACHE["https://example.com"] = FakeParser()

        allowed = await check_robots_txt("https://example.com/private")
        assert allowed is False


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

    @pytest.mark.asyncio
    async def test_invalid_start_index_rejected(self):
        """Negative start_index should be rejected before network calls."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("https://example.com", start_index=-1)
        assert result.error == "start_index must be >= 0"

    @pytest.mark.asyncio
    async def test_invalid_max_length_rejected(self):
        """Non-positive max_length should be rejected."""
        from gemini_research_mcp.content import fetch_webpage

        result = await fetch_webpage("https://example.com", max_length=0)
        assert result.error == "max_length must be > 0 when provided"

    @pytest.mark.asyncio
    async def test_robots_block_returns_error(self, monkeypatch: pytest.MonkeyPatch):
        """Disallowed robots.txt should short-circuit fetch."""
        import gemini_research_mcp.content as content

        async def fake_robots(*args: object, **kwargs: object) -> bool:
            return False

        monkeypatch.setattr(content, "validate_url", lambda _: (True, ""))
        monkeypatch.setattr(content, "check_robots_txt", fake_robots)

        result = await content.fetch_webpage("https://example.com")
        assert result.error == "Blocked by robots.txt"

    @pytest.mark.asyncio
    async def test_proxy_url_passed_to_httpx(self, monkeypatch: pytest.MonkeyPatch):
        """Proxy URL should be forwarded to httpx AsyncClient."""
        import gemini_research_mcp.content as content

        captured_proxy: dict[str, str | None] = {"value": None}

        class MockResponse:
            status_code = 200
            headers: dict[str, str] = {}
            text = "<html><head><title>T</title></head><body><p>Hello world.</p></body></html>"

            def raise_for_status(self) -> None:
                return None

        class MockClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured_proxy["value"] = kwargs.get("proxy")  # type: ignore[assignment]

            async def __aenter__(self) -> "MockClient":
                return self

            async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

            async def get(self, url: str) -> MockResponse:
                return MockResponse()

        async def always_true(*args: object, **kwargs: object) -> bool:
            return True

        monkeypatch.setattr(content, "validate_url", lambda _: (True, ""))
        monkeypatch.setattr(content, "check_robots_txt", always_true)
        monkeypatch.setattr(content.httpx, "AsyncClient", MockClient)

        proxy = "http://proxy.example.com:3128"
        result = await content.fetch_webpage("https://example.com", proxy_url=proxy)

        assert result.error is None
        assert captured_proxy["value"] == proxy
