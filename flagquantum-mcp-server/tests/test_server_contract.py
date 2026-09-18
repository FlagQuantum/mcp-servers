"""The MCP surface: registration, schemas, resources, prompts, error envelope.

These are the tests that would catch a tool silently disappearing from the
server, or a schema changing shape in a way a client depends on.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ValidationError

from flagquantum_mcp_server.server import mcp

pytestmark = pytest.mark.unit

EXPECTED_TOOLS = {
    "analyze_circuit_tool",
    "serialize_circuit_tool",
    "deserialize_circuit_tool",
    "optimize_circuit_tool",
    "route_circuit_tool",
    "compare_topologies_tool",
    "emit_openqasm_tool",
    "emit_qcis_tool",
    "plan_execution_tool",
}

EXPECTED_RESOURCES = {
    "flagquantum://version",
    "flagquantum://gate-set",
    "flagquantum://ir-schema",
}

EXPECTED_PROMPTS = {
    "build_and_analyze_circuit",
    "compile_for_topology",
    "export_circuit",
}


def test_server_is_named_and_versioned() -> None:
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "FlagQuantum"


async def test_every_expected_tool_is_registered() -> None:
    tools = await mcp.list_tools()

    assert {tool.name for tool in tools} == EXPECTED_TOOLS


async def test_tools_are_declared_read_only_and_closed_world() -> None:
    for tool in await mcp.list_tools():
        assert tool.annotations is not None, tool.name
        assert tool.annotations.readOnlyHint is True, tool.name
        assert tool.annotations.openWorldHint is False, tool.name


async def test_circuit_tools_take_a_circuit_and_a_format() -> None:
    """Eight tools accept any input format; deserialize is the exception.

    ``deserialize_circuit_tool`` takes IR text only, because deciding *what* a
    payload is would be exactly the ambiguity that tool exists to remove.
    """
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    for name, tool in tools.items():
        properties = (tool.parameters or {}).get("properties", {})
        if name == "deserialize_circuit_tool":
            assert set(properties) == {"ir_json", "indent"}
            continue
        assert "circuit" in properties, name
        assert "circuit_format" in properties, name


async def test_circuit_format_is_a_closed_enum_not_a_free_string() -> None:
    """The schema must publish the enum, not just document it in prose.

    A model reads the JSON schema, not this repository. When ``circuit_format``
    was a plain ``str``, nothing in the schema said "openqasm" or "IR" was
    wrong, so a wrong guess cost a round trip through the error envelope. The
    enum makes the wrong guess impossible to express.
    """
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    for name, tool in tools.items():
        properties = (tool.parameters or {}).get("properties", {})
        if "circuit_format" not in properties:
            assert name == "deserialize_circuit_tool", name
            continue
        schema = properties["circuit_format"]
        assert schema.get("enum") == ["ir", "qir"], (name, schema)
        assert schema["type"] == "string", name
        assert schema["default"] in {"ir", "qir"}, name


async def test_the_circuit_argument_shows_both_formats_by_example() -> None:
    """Every tool that takes a circuit must say what one looks like.

    The failure this guards against is a model that knows Qiskit and reaches
    for OpenQASM. The parameter description has to rule that out where the
    model actually reads it.
    """
    for tool in await mcp.list_tools():
        properties = (tool.parameters or {}).get("properties", {})
        if "circuit" not in properties:
            continue
        description = properties["circuit"]["description"]
        assert 'circuit_format="qir"' in description, tool.name
        assert '"name": "h"' in description, tool.name
        assert 'circuit_format="ir"' in description, tool.name
        assert "OpenQASM" in description, tool.name


async def test_an_invalid_format_is_rejected_before_the_tool_body_runs() -> None:
    """Schema violations are a protocol error, not our envelope.

    That is the intended split: the schema catches what it can describe, and
    the envelope catches everything else. Both reach the client as an error it
    can read; only the layer differs.
    """
    with pytest.raises(ValidationError, match="Input should be 'ir' or 'qir'"):
        await mcp.call_tool("analyze_circuit_tool", {"circuit": "[]", "circuit_format": "openqasm"})


async def test_every_resource_and_prompt_is_registered() -> None:
    resources = await mcp.list_resources()
    prompts = await mcp.list_prompts()

    assert {str(resource.uri) for resource in resources} == EXPECTED_RESOURCES
    assert {prompt.name for prompt in prompts} == EXPECTED_PROMPTS


async def test_a_successful_call_returns_the_payload_directly(bell_qir: str) -> None:
    result = await mcp.call_tool(
        "analyze_circuit_tool", {"circuit": bell_qir, "circuit_format": "qir"}
    )

    assert result.structured_content is not None
    assert result.structured_content["status"] == "success"
    assert result.structured_content["analysis"]["depth"] == 2


async def test_a_failure_returns_the_error_envelope_not_an_exception() -> None:
    result = await mcp.call_tool(
        "analyze_circuit_tool", {"circuit": "{not json", "circuit_format": "ir"}
    )

    assert result.structured_content is not None
    assert result.structured_content["status"] == "error"
    assert result.structured_content["error"]["code"] == "INVALID_INPUT"


OVERSIZED = "[" + "1," * 400_000 + "1]"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"circuit": "[]", "circuit_format": "ir"}, "INVALID_INPUT"),
        ({"circuit": OVERSIZED, "circuit_format": "qir"}, "LIMIT_EXCEEDED"),
        (
            {"circuit": '{"kind": "nope"}', "circuit_format": "ir"},
            "INVALID_INPUT",
        ),
        (
            {"circuit": '[{"name": "nope", "index": [0]}]', "circuit_format": "qir"},
            "INVALID_INPUT",
        ),
    ],
)
async def test_the_error_envelope_is_the_same_shape_for_every_code(
    arguments: dict[str, str], expected: str
) -> None:
    result = await mcp.call_tool("analyze_circuit_tool", arguments)

    assert result.structured_content is not None
    assert result.structured_content["status"] == "error"
    error = result.structured_content["error"]
    assert error["code"] == expected
    assert isinstance(error["message"], str)
    assert error["message"]


async def test_resources_return_parseable_json() -> None:
    for uri in EXPECTED_RESOURCES:
        result = await mcp.read_resource(uri)
        payload = json.loads(result.contents[0].content)
        assert isinstance(payload, dict)


async def test_gate_set_resource_matches_the_sdk() -> None:
    result = await mcp.read_resource("flagquantum://gate-set")
    payload = json.loads(result.contents[0].content)

    assert payload["n_gates"] == 45
    assert payload["gates"] == sorted(payload["gates"])


async def test_version_resource_reports_the_installed_sdk() -> None:
    import flagquantum as fq

    result = await mcp.read_resource("flagquantum://version")
    payload = json.loads(result.contents[0].content)

    assert payload["flagquantum_version"] == fq.__version__
    assert payload["ir_version"] == fq.IR_VERSION


async def test_ir_schema_resource_shows_a_working_example() -> None:
    result = await mcp.read_resource("flagquantum://ir-schema")
    payload = json.loads(result.contents[0].content)

    assert payload["example"]["kind"] == "flagquantum.circuit_ir"
    assert payload["ir_version"] == payload["example"]["version"]
    assert len(payload["content_hash"]) == 64
    assert "opcode" in json.dumps(payload["example"])


async def test_prompts_render_with_their_arguments() -> None:
    rendered = await mcp.render_prompt(
        "build_and_analyze_circuit", {"circuit_description": "a Bell state"}
    )

    assert "a Bell state" in str(rendered)


async def test_export_prompt_mentions_the_qcis_limitation() -> None:
    rendered = await mcp.render_prompt("export_circuit", {"circuit": "[]", "target_format": "qcis"})

    assert "matrix gate" in str(rendered)
