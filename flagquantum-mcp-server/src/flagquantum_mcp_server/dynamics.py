"""Thin JSON adapter over FlagQuantum's public Lindblad evolution API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from flagquantum_mcp_server._bridge import load_module, load_torch
from flagquantum_mcp_server.errors import ToolError

Precision = Literal["complex64", "complex128"]


def _arguments(
    hamiltonian: Sequence[Mapping[str, Any]],
    initial_state: Any,
    n_wires: int,
    times: Sequence[float],
    collapse_operators: Sequence[Mapping[str, Any]],
    observables: Sequence[Mapping[str, Any]] | None,
    *,
    device: str,
    precision: Precision,
    return_density_matrices: bool,
) -> dict[str, Any]:
    torch = load_torch()
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    return {
        "hamiltonian": hamiltonian,
        "initial_state": initial_state,
        "n_qubits": n_wires,
        "times": times,
        "collapse_operators": collapse_operators,
        "observables": observables,
        "device": device,
        "dtype": dtype,
        "return_density_matrices": return_density_matrices,
    }


def _invoke(name: str, arguments: dict[str, Any]) -> Any:
    simulation = load_module("flagquantum.simulation")
    function = getattr(simulation, name)
    try:
        return function(**arguments)
    except ValueError as exc:
        code = getattr(exc, "code", "invalid_evolution_request")
        field = getattr(exc, "field", None)
        message = f"{field}: {exc}" if field else str(exc)
        error = ToolError(message)
        error.code = code
        raise error from exc


def plan_evolution(
    hamiltonian: Sequence[Mapping[str, Any]],
    initial_state: Any,
    n_wires: int,
    times: Sequence[float],
    collapse_operators: Sequence[Mapping[str, Any]],
    observables: Sequence[Mapping[str, Any]] | None = None,
    *,
    device: str = "cpu",
    precision: Precision = "complex128",
    return_density_matrices: bool = False,
) -> dict[str, Any]:
    """Validate and plan a local continuous-time evolution."""

    arguments = _arguments(
        hamiltonian,
        initial_state,
        n_wires,
        times,
        collapse_operators,
        observables,
        device=device,
        precision=precision,
        return_density_matrices=return_density_matrices,
    )
    plan = _invoke("plan_density_matrix_evolution", arguments)
    return {
        "status": "success",
        "plan": plan.to_dict(),
        "execution": {
            "execution_path": "local_density_matrix",
            "platform_provider": "pytorch_cpu",
            "deterministic": True,
            "network_access": False,
        },
    }


def evolve_dynamics(
    hamiltonian: Sequence[Mapping[str, Any]],
    initial_state: Any,
    n_wires: int,
    times: Sequence[float],
    collapse_operators: Sequence[Mapping[str, Any]],
    observables: Sequence[Mapping[str, Any]] | None = None,
    *,
    device: str = "cpu",
    precision: Precision = "complex128",
    return_density_matrices: bool = False,
) -> dict[str, Any]:
    """Execute a local continuous-time evolution and serialize its trajectory."""

    arguments = _arguments(
        hamiltonian,
        initial_state,
        n_wires,
        times,
        collapse_operators,
        observables,
        device=device,
        precision=precision,
        return_density_matrices=return_density_matrices,
    )
    result = _invoke("evolve_density_matrix", arguments)
    payload = cast("dict[str, Any]", result.to_dict())
    payload.update(
        {
            "status": "success",
            "execution": {
                "execution_path": "local_density_matrix",
                "platform_provider": "pytorch_cpu",
                "deterministic": True,
                "network_access": False,
            },
        }
    )
    return payload


__all__ = ("Precision", "evolve_dynamics", "plan_evolution")
