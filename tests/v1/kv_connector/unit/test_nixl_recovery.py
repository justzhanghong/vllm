# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_RECOVERY_PATH = (
    Path(__file__).parents[4]
    / "vllm/distributed/kv_transfer/kv_connector/v1/nixl/recovery.py"
)
_SPEC = importlib.util.spec_from_file_location("nixl_recovery", _RECOVERY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RECOVERY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RECOVERY)

SharedRecoveryBarrier = _RECOVERY.SharedRecoveryBarrier
retain_remote_engine_state = _RECOVERY.retain_remote_engine_state


def test_shared_recovery_barrier_is_generation_safe(tmp_path: Path) -> None:
    first = SharedRecoveryBarrier(tmp_path, "pp0-tp0", 2)
    second = SharedRecoveryBarrier(tmp_path, "pp0-tp1", 2)

    first_incident = first.declare("prefill-0")
    first_id = first.incident_id("prefill-0")
    assert first_id is not None
    assert second.active_engines() == {"prefill-0"}

    assert first.acknowledge_quiescent("prefill-0", first_id) == (1, False)
    assert second.acknowledge_quiescent("prefill-0", first_id) == (2, True)
    assert first.mark_cleaned("prefill-0", first_id) == (1, False)
    assert second.mark_cleaned("prefill-0", first_id) == (2, True)
    assert not first.any_active()

    second_incident = first.declare("prefill-0")
    second_id = first.incident_id("prefill-0")
    assert second_id is not None and second_id != first_id
    assert second_incident != first_incident

    # A delayed participant from generation 1 must not complete generation 2.
    assert second.mark_cleaned("prefill-0", first_id) == (0, True)
    assert first.is_active("prefill-0")


def test_retain_remote_engine_state_preserves_oscar_metadata() -> None:
    worker = SimpleNamespace(
        _remote_agents={"prefill-0": {0: "agent-0"}},
        dst_xfer_side_handles={"prefill-0": {0: 10}},
        kv_caches_base_addr={"prefill-0": [100]},
        dst_num_blocks={"prefill-0": 32},
        _physical_blocks_per_logical={"prefill-0": 1},
        _remote_oscar_mla_agent_meta={"prefill-0": object()},
    )
    oscar_metadata = worker._remote_oscar_mla_agent_meta["prefill-0"]

    retained = retain_remote_engine_state(
        worker,
        "prefill-0",
        extra_remote_agents={1: "agent-1"},
    )

    assert worker._remote_agents["prefill-0"] == {
        0: "agent-0",
        1: "agent-1",
    }
    assert worker._remote_oscar_mla_agent_meta["prefill-0"] is oscar_metadata
    assert retained == {
        "remote_agents": 2,
        "dlist_handles": 1,
        "base_addresses": True,
        "block_metadata": True,
        "physical_ratio": True,
        "agent_conflicts": 0,
    }
