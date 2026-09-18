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
    "describe_gate_set_tool",
    "inspect_parameters_tool",
    "bind_parameters_tool",
    "describe_layers_tool",
    "describe_topology_tool",
    "draw_circuit_tool",
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
    """Any tool that takes a circuit also takes its format.

    Three tools take no circuit, each for its own reason:
    ``deserialize_circuit_tool`` takes IR text alone, because deciding *what* a
    payload is would be the ambiguity that tool exists to remove;
    ``describe_topology_tool`` describes a connectivity rather than a circuit;
    ``describe_gate_set_tool`` describes gates.
    """
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    circuitless = {
        "deserialize_circuit_tool",
        "describe_topology_tool",
        "describe_gate_set_tool",
    }

    for name, tool in tools.items():
        properties = (tool.parameters or {}).get("properties", {})
        if name == "deserialize_circuit_tool":
            assert set(properties) == {"ir_json", "indent"}
            continue
        if name in circuitless:
            assert "circuit" not in properties, name
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
            assert name in {
                "deserialize_circuit_tool",
                "describe_topology_tool",
                "describe_gate_set_tool",
            }, name
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

    # 35 opcodes, reachable under 45 names once aliases are counted.
    assert payload["n_gates"] == 35
    assert payload["n_names"] == 45
    opcodes = [record["opcode"] for record in payload["gates"]]
    assert opcodes == sorted(opcodes)
    assert {"h", "cx", "rz", "ccx", "swap"} <= set(opcodes)
    assert all("arity" in record and "parameters" in record for record in payload["gates"])


async def test_version_resource_reports_the_installed_sdk() -> None:
    import flagquantum as fq

    result = await mcp.read_resource("flagquantum://version")
    payload = json.loads(result.contents[0].content)

    assert payload["flagquantum_version"] == fq.__version__
    assert payload["ir_version"] == fq.IR_VERSION


async def test_the_ir_schema_resource_shows_a_working_example() -> None:
    result = await mcp.read_resource("flagquantum://ir-schema")
    payload = json.loads(result.contents[0].content)

    assert payload["example"]["kind"] == "flagquantum.circuit_ir"
    assert payload["ir_version"] == payload["example"]["version"]
    assert len(payload["content_hash"]) == 64
    assert "opcode" in json.dumps(payload["example"])


async def test_the_ir_schema_resource_shows_how_to_write_a_symbol() -> None:
    """The marker was undocumented, so agents guessed and guessed wrong.

    A hand-written string is the natural first attempt at a symbol, and the
    resource said nothing either way. An agent session had to read the SDK
    source to find the spelling, which no client of this server should need.
    """
    result = await mcp.read_resource("flagquantum://ir-schema")
    payload = json.loads(result.contents[0].content)

    assert "$parameter" in json.dumps(payload["parameter_example"])
    assert "$parameter" in " ".join(payload["notes"])
    assert "bare string" in " ".join(payload["notes"])


# --- what a model can actually read ---
#
# FastMCP publishes only the docstring summary; ``Args:`` becomes the parameter
# descriptions and ``Returns:`` is dropped entirely. So a fact written under
# ``Returns:`` is invisible to every client — which is where the meaning of
# wire_usage, the concurrency reading of a layer, and the QCIS matrix-gate
# refusal all used to live. These assertions read the published description, not
# the docstring, so moving a fact back into ``Returns:`` fails the build.


@pytest.mark.parametrize(
    ("tool", "phrase"),
    [
        ("analyze_circuit_tool", "wire_usage has one entry per wire"),
        ("analyze_circuit_tool", "multi_qubit_gates counts"),
        ("describe_layers_tool", "run concurrently"),
        ("describe_topology_tool", "distance is the number of edges"),
        ("emit_qcis_tool", "matrix gate"),
        ("plan_execution_tool", "Nothing is executed"),
        ("bind_parameters_tool", "no longer parameterized"),
        ("emit_openqasm_tool", "measures every wire"),
    ],
)
async def test_tool_descriptions_publish_what_a_caller_cannot_infer(tool: str, phrase: str) -> None:
    description = (await mcp.get_tool(tool)).description or ""

    assert phrase in description, f"{tool} does not publish {phrase!r}"


async def test_no_tool_hides_its_only_documentation_in_a_returns_section() -> None:
    """A tool whose summary is one bare line is a tool documented nowhere."""
    tools = await mcp.list_tools()

    thin = [tool.name for tool in tools if len((tool.description or "").strip()) < 60]

    assert thin == [], f"these tools publish no explanation: {thin}"


async def test_the_documented_symbol_actually_parameterizes_a_circuit() -> None:
    """Pins the example: written as the resource shows it, a symbol is a symbol."""
    from flagquantum_mcp_server.parameters import inspect_parameters

    result = await mcp.read_resource("flagquantum://ir-schema")
    payload = json.loads(result.contents[0].content)
    example = json.dumps(payload["parameter_example"])

    assert inspect_parameters(example, "ir")["parameter_names"] == ["theta"]


async def test_prompts_render_with_their_arguments() -> None:
    rendered = await mcp.render_prompt(
        "build_and_analyze_circuit", {"circuit_description": "a Bell state"}
    )

    assert "a Bell state" in str(rendered)


async def test_export_prompt_mentions_the_qcis_limitation() -> None:
    rendered = await mcp.render_prompt("export_circuit", {"circuit": "[]", "target_format": "qcis"})

    assert "matrix gate" in str(rendered)


async def test_the_ir_schema_resource_says_what_the_hash_covers() -> None:
    """`content_hash` hashes the payload, metadata included.

    An agent session noticed the same circuit reporting two different hashes
    and explained it correctly: the second payload had `metadata` stripped. The
    hash is the SDK's own, and omitting an optional field is legal, so the fix
    is to say what the hash covers rather than to change it. A caller that
    treats it as a gate-sequence identity will be surprised.
    """
    result = await mcp.read_resource("flagquantum://ir-schema")
    payload = json.loads(result.contents[0].content)

    notes = " ".join(payload["notes"])
    assert "'metadata'" in notes
    assert "identifies the payload" in notes


def test_an_ir_payload_without_metadata_is_legal_but_hashes_differently(
    bell_qir: str,
) -> None:
    """Pins the behaviour the resource now documents.

    If FlagQuantum ever starts normalizing metadata on load, this test fails and
    the note above becomes wrong — which is the point.
    """
    from flagquantum_mcp_server.circuits import serialize

    canonical = json.loads(serialize(bell_qir, "qir")["ir_json"])
    assert "metadata" in canonical

    trimmed = {key: value for key, value in canonical.items() if key != "metadata"}
    from flagquantum_mcp_server.analysis import analyze

    with_metadata = analyze(json.dumps(canonical), "ir")["circuit"]["content_hash"]
    without_metadata = analyze(json.dumps(trimmed), "ir")["circuit"]["content_hash"]

    assert with_metadata != without_metadata


# --- a tool that says what another tool will do is making a claim ---
#
# ``inspect_parameters_tool``'s note tells a caller what to do next, so it is a
# claim about the other fourteen tools, and nothing held it to one. It read "A
# parameterized circuit cannot be planned or exported as-is". Export does
# refuse. Planning does not: a plan comes from the payload's shape and dtype and
# never reads a parameter value. An agent session believed the note, bound eight
# invented numbers before planning, and then reported a resource table computed
# from values it had made up. The note was right about half of what it said,
# which is the hardest kind of wrong to notice — every clause is plausible, and
# a caller has no way to check one without doing the work the note just told it
# to skip.
#
# So the tools are exercised here and the note is read. The parametrized cases
# pin which tools refuse; the last test pins that the note says so.


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        ("analyze_circuit_tool", {}, "success"),
        ("describe_layers_tool", {}, "success"),
        ("draw_circuit_tool", {}, "success"),
        ("optimize_circuit_tool", {}, "success"),
        ("plan_execution_tool", {}, "success"),
        ("route_circuit_tool", {"topology": "line"}, "success"),
        ("serialize_circuit_tool", {}, "success"),
        ("emit_openqasm_tool", {}, "error"),
        ("emit_qcis_tool", {}, "error"),
    ],
)
async def test_only_the_emitters_refuse_an_unbound_circuit(
    angled_qir: str, tool: str, arguments: dict[str, object], expected: str
) -> None:
    result = await mcp.call_tool(
        tool, {"circuit": angled_qir, "circuit_format": "qir", **arguments}
    )
    payload = json.loads(result.content[0].text)

    assert payload["status"] == expected, f"{tool} on an unbound circuit returned {payload}"


async def test_the_refusal_names_the_fix(angled_qir: str) -> None:
    """A refusal that does not name the way out is a refusal a caller has to guess at."""
    result = await mcp.call_tool(
        "emit_openqasm_tool", {"circuit": angled_qir, "circuit_format": "qir"}
    )
    payload = json.loads(result.content[0].text)

    assert payload["error"]["code"] == "UNSUPPORTED_FORMAT"
    assert "bind_parameters" in payload["error"]["message"]


async def test_the_parameter_claims_agree_with_what_the_tools_do(angled_qir: str) -> None:
    """Two published places tell a caller to bind, and both overstated it.

    A text assertion on purpose: the behaviour is pinned above, and no
    behavioural test can catch prose that contradicts it. "planned" is the
    discriminator — it appears only in the claim that planning is blocked, and
    never in the corrected wording, which names planning among the steps that
    accept an unbound circuit.
    """
    from flagquantum_mcp_server.parameters import inspect_parameters

    note = inspect_parameters(angled_qir, "qir")["note"].lower()
    published = ((await mcp.get_tool("bind_parameters_tool")).description or "").lower()

    for text, where in ((note, "the note"), (published, "bind_parameters_tool's summary")):
        assert "export" in text, f"{where} has to name export as the blocked step"
        assert "planned" not in text, f"{where} claims an unbound circuit cannot be planned"


# --- the shape of an observable, which used to exist only inside a rejection ---


async def test_the_documented_observable_example_is_what_the_sdk_stores(bell_qir: str) -> None:
    """The entry shape was reachable only by being wrong first.

    README showed ``"observables": []`` and this resource showed the same empty
    list, so the form of an entry lived only in the message a caller got after
    getting it wrong. An agent session building a VQE ansatz reported that it
    left the field empty because it could not tell how to fill it — which drops
    the Hamiltonian out of the one circuit that exists to carry it.
    """
    from flagquantum_mcp_server.circuits import serialize

    result = await mcp.read_resource("flagquantum://ir-schema")
    example = json.loads(result.contents[0].content)["observable_example"]

    envelope = json.loads(serialize(bell_qir, "qir")["ir_json"])
    envelope.update(example["sent"])
    stored = json.loads(serialize(json.dumps(envelope), "ir")["ir_json"])

    assert {key: stored[key] for key in example["sent"]} == example["canonical"]


async def test_a_hamiltonian_weight_can_be_a_symbol(bell_qir: str) -> None:
    """The resource states this; here is the claim being held to it."""
    from flagquantum_mcp_server.circuits import serialize

    envelope = json.loads(serialize(bell_qir, "qir")["ir_json"])
    envelope["observables"] = [{"name": "zz", "wires": [0, 1], "coefficient": {"$parameter": "w"}}]
    stored = json.loads(serialize(json.dumps(envelope), "ir")["ir_json"])

    assert stored["observables"][0]["coefficient"] == {"$parameter": "w"}


async def test_the_resource_documents_both_node_lists() -> None:
    result = await mcp.read_resource("flagquantum://ir-schema")
    notes = " ".join(json.loads(result.contents[0].content)["notes"])

    assert "'coefficient'" in notes
    assert "'shots'" in notes
    assert "lowercased" in notes
