"""The version is declared in three places and must agree in all of them."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from flagquantum_mcp_server import __version__

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match is not None, "pyproject.toml has no static version"
    return match.group(1)


def _server_json() -> dict[str, object]:
    return json.loads((PACKAGE_ROOT / "server.json").read_text(encoding="utf-8"))


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version()


def test_version_matches_the_registry_manifest() -> None:
    assert __version__ == _server_json()["version"]


def test_registry_manifest_package_versions_agree() -> None:
    manifest = _server_json()
    entry = manifest["packages"][0]

    assert entry["version"] == manifest["version"]
    assert entry["identifier"] == "flagquantum-mcp-server"
    assert entry["registryType"] == "pypi"
    assert entry["transport"]["type"] == "stdio"


def test_registry_manifest_names_the_repository() -> None:
    manifest = _server_json()

    assert manifest["name"] == "io.github.FlagQuantum/flagquantum-mcp-server"
    assert manifest["repository"]["url"] == "https://github.com/FlagQuantum/mcp-servers"
    assert manifest["repository"]["subfolder"] == "flagquantum-mcp-server"


def test_the_dependency_is_a_version_range_not_a_url() -> None:
    """A URL dependency makes every job that installs another ref unresolvable.

    Only the dependency table is checked: the ``[project.urls]`` section is
    supposed to contain URLs.
    """
    text = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)

    assert block is not None, "no static dependencies table"
    declared = block.group(1)

    assert "flagquantum>=0.2,<0.3" in declared
    assert "git+" not in declared
    assert "http" not in declared


def test_readme_carries_the_registry_ownership_marker() -> None:
    """The MCP Registry reads this line to prove we control the PyPI package.

    PyPI has no "verified publisher" field, so the registry verifies ownership
    by looking for ``mcp-name: <server-name>`` in the published README. Remove
    it and registry publishing fails at the ownership check, not at validation
    — which is why it is pinned here rather than left to a comment.
    """
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    name = _server_json()["name"]

    assert f"mcp-name: {name}" in readme


def test_registry_description_fits_the_registry_limit() -> None:
    """The registry rejects a description longer than 100 characters.

    This is a registry constraint, not a packaging one; a longer description
    passes every local check and fails only when publishing.
    """
    description = str(_server_json()["description"])

    assert 0 < len(description) <= 100, len(description)


def test_registry_manifest_declares_only_supported_transports() -> None:
    """The registry understands stdio and streamable-http, not sse."""
    for entry in _server_json()["packages"]:
        assert entry["transport"]["type"] in {"stdio", "streamable-http"}
