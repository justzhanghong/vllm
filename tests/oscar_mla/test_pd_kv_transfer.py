# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from types import SimpleNamespace

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.oscar_mla import (
    OscarMLAAgentMetadata,
    OscarMLARequestMetadata,
    build_oscar_mla_descriptor_pairs,
    project_oscar_mla_request_prefix,
    validate_oscar_mla_request,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)


def _agent(
    *,
    artifact_hash: str = "artifact",
    num_blocks: int = 32,
    max_num_seqs: int = 4,
):
    return OscarMLAAgentMetadata(
        protocol_version=1,
        layer_names=("layer.0", "layer.1"),
        auxiliary_layer_names=("indexer.0", "indexer.1"),
        auxiliary_page_bytes=(2112, 2112),
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
        block_size=16,
        latent_rank=512,
        rope_head_size=64,
        group_size=128,
        prefix_tokens=64,
        recent_tokens=256,
        speculative_tokens=0,
        hp_dtype="bfloat16",
        artifact_manifest_sha256=artifact_hash,
        artifact_rotations_sha256="rotations",
    )


def _request(
    *,
    generation: int = 3,
    hp_row: int = 2,
    block_ids=tuple(range(10, 32)),
):
    return OscarMLARequestMetadata(
        generation=generation,
        cache_version=9,
        logical_length=352,
        hp_row=hp_row,
        block_ids=block_ids,
        history_pages=block_ids[4:6],
        partial_history_slots=0,
        history_tokens=32,
    )


def test_descriptor_pairs_use_pool_specific_strides_and_local_remap():
    remote = _request(hp_row=2)
    local = _request(generation=11, hp_row=1, block_ids=tuple(range(22)))

    local_ids, remote_ids = build_oscar_mla_descriptor_pairs(
        _agent(), local, _agent(), remote
    )

    assert remote_ids[:6] == (14, 15, 46, 47, 78, 79)
    assert remote_ids[6:28] == tuple(range(106, 128))
    assert remote_ids[28:30] == (130, 134)
    assert remote_ids[30:36] == (150, 151, 182, 183, 214, 215)
    assert local_ids[:6] == (4, 5, 36, 37, 68, 69)
    assert local_ids[6:28] == tuple(range(96, 118))
    assert local_ids[28:30] == (129, 133)
    assert remote_ids[-44:-22] == tuple(range(282, 304))
    assert remote_ids[-22:] == tuple(range(314, 336))
    assert local_ids[-44:-22] == tuple(range(272, 294))
    assert local_ids[-22:] == tuple(range(304, 326))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("artifact_manifest_sha256", "other", "artifact"),
        ("speculative_tokens", 5, "geometry"),
    ],
)
def test_descriptor_pairs_fail_closed_on_incompatible_agents(field, value, match):
    local = _agent()
    remote_values = dict(local.__dict__)
    remote_values[field] = value
    remote = OscarMLAAgentMetadata(**remote_values)

    with pytest.raises(ValueError, match=match):
        build_oscar_mla_descriptor_pairs(local, _request(), remote, _request())


def test_descriptor_pairs_allow_different_cache_capacities():
    local_agent = _agent(num_blocks=24, max_num_seqs=3)
    remote_agent = _agent(num_blocks=40, max_num_seqs=6)
    local_request = _request(hp_row=1, block_ids=tuple(range(22)))
    remote_request = _request(hp_row=4, block_ids=tuple(range(10, 32)))

    local_ids, remote_ids = build_oscar_mla_descriptor_pairs(
        local_agent,
        local_request,
        remote_agent,
        remote_request,
    )

    assert local_agent.geometry_fingerprint == remote_agent.geometry_fingerprint
    assert len(local_ids) == len(remote_ids)
    assert local_ids != remote_ids


def test_request_validation_rejects_stale_generation_and_bad_partition():
    agent = _agent()
    request = _request()
    validate_oscar_mla_request(agent, request, expected_generation=3)

    with pytest.raises(ValueError, match="generation"):
        validate_oscar_mla_request(agent, request, expected_generation=4)

    bad = OscarMLARequestMetadata(
        generation=3,
        cache_version=9,
        logical_length=352,
        hp_row=2,
        block_ids=tuple(range(10, 32)),
        history_pages=(14,),
        partial_history_slots=0,
        history_tokens=32,
    )
    with pytest.raises(ValueError, match="history page"):
        validate_oscar_mla_request(agent, bad)


def test_transfer_bytes_are_native_compressed_payload():
    agent = _agent()
    request = _request()

    # Per layer: 2 history pages x (INT2 data + FP32 scale/zero + BF16 RoPE)
    # plus one full BF16 prefix row and one full BF16 recent row.
    expected_per_layer = (
        2 * 16 * (128 + 2 * 4 * 4)
        + 22 * 16 * 64 * 2
        + 64 * 1024
        + 256 * 1024
    )
    expected_auxiliary = 22 * (2112 + 2112)
    assert agent.transfer_byte_breakdown(request) == {
        "history_data": 2 * 2 * 16 * 128,
        "history_scale": 2 * 2 * 16 * 4 * 4,
        "history_zero": 2 * 2 * 16 * 4 * 4,
        "rope": 2 * 22 * 16 * 64 * 2,
        "prefix": 2 * 64 * 512 * 2,
        "recent": 2 * 256 * 512 * 2,
        "auxiliary": expected_auxiliary,
    }
    assert agent.transfer_bytes(request) == 2 * expected_per_layer + expected_auxiliary
    assert agent.transfer_bytes(request) < agent.bf16_reference_bytes(request)


def test_agent_metadata_msgpack_round_trip_preserves_oscar_contract():
    metadata = NixlAgentMetadata(
        engine_id="producer",
        agent_metadata=b"agent",
        kv_caches_base_addr=list(range(14)),
        device_id=0,
        num_blocks=10,
        block_lens=[2048, 256],
        kv_cache_layout="NHD",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        oscar_mla=_agent(),
    )

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata), type=NixlAgentMetadata
    )
    assert decoded == metadata


def test_request_wire_round_trip_is_strict():
    request = _request()
    assert OscarMLARequestMetadata.from_wire(request.to_wire()) == request

    invalid = request.to_wire()
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        OscarMLARequestMetadata.from_wire(invalid)


def test_project_request_prefix_excludes_replayed_prompt_tail():
    agent = _agent()
    request = OscarMLARequestMetadata(
        generation=3,
        cache_version=9,
        logical_length=321,
        hp_row=2,
        block_ids=tuple(range(21)),
        history_pages=(4,),
        partial_history_slots=1,
        history_tokens=1,
    )

    projected = project_oscar_mla_request_prefix(agent, request, 320)

    assert projected.logical_length == 320
    assert projected.block_ids == tuple(range(20))
    assert projected.history_tokens == 0
    assert projected.history_pages == ()
    assert projected.partial_history_slots == 0
    validate_oscar_mla_request(agent, projected)


def test_wire_ownership_drops_scheduler_only_mtp_tail_block():
    agent = _agent()
    request = OscarMLARequestMetadata(
        generation=3,
        cache_version=9,
        logical_length=352,
        hp_row=2,
        block_ids=tuple(range(10, 33)),
        history_pages=(14, 15),
        partial_history_slots=0,
        history_tokens=32,
    )

    trimmed = request.trim_speculative_tail_blocks(agent.block_size)

    assert trimmed.block_ids == tuple(range(10, 32))
    validate_oscar_mla_request(agent, trimmed)


def test_project_request_prefix_accepts_scheduler_only_mtp_tail_block():
    agent = _agent()
    request = OscarMLARequestMetadata(
        generation=3,
        cache_version=9,
        logical_length=352,
        hp_row=2,
        block_ids=tuple(range(10, 33)),
        history_pages=(14, 15),
        partial_history_slots=0,
        history_tokens=32,
    )

    projected = project_oscar_mla_request_prefix(agent, request, 351)

    assert projected.logical_length == 351
    assert projected.block_ids == tuple(range(10, 32))
    validate_oscar_mla_request(agent, projected)


def test_descriptor_pairs_allow_projected_remote_prompt_tail_replay():
    agent = _agent()
    remote = OscarMLARequestMetadata(
        generation=3,
        cache_version=9,
        logical_length=321,
        hp_row=2,
        block_ids=tuple(range(21)),
        history_pages=(4,),
        partial_history_slots=1,
        history_tokens=1,
    )
    local = project_oscar_mla_request_prefix(agent, remote, 320)
    remote_prefix = project_oscar_mla_request_prefix(agent, remote, 320)

    local_ids, remote_ids = build_oscar_mla_descriptor_pairs(
        agent, local, agent, remote_prefix
    )

    auxiliary_base = len(agent.layer_names) * agent.descriptors_per_layer
    assert auxiliary_base + 20 not in remote_ids
    assert len(local_ids) == len(remote_ids)


def test_worker_descriptor_registration_matches_protocol_index_space():
    agent = _agent()
    bases = [index * 10_000_000 for index in range(14)]

    descriptors = NixlConnectorWorker._build_oscar_mla_blocks_data(
        bases, agent, 7
    )

    assert len(descriptors) == (
        2 * agent.descriptors_per_layer + 2 * agent.num_blocks
    )
    assert descriptors[0] == (bases[0], 2048, 7)
    assert descriptors[32] == (bases[1], 256, 7)
    assert descriptors[64] == (bases[2], 256, 7)
    assert descriptors[96] == (bases[3], 2048, 7)
    assert descriptors[128] == (bases[4], 65536, 7)
    assert descriptors[132] == (bases[5], 262144, 7)
    assert descriptors[136] == (bases[6], 2048, 7)
    assert descriptors[272] == (bases[12], 2112, 7)
    assert descriptors[304] == (bases[13], 2112, 7)


def test_worker_ignores_stale_and_duplicate_generation_ack():
    class FakeNotifications:
        def __init__(self):
            self.payloads = [b"OSCAR:req-0:6:8"]

        def get_new_notifs(self):
            payloads, self.payloads = self.payloads, []
            return {"remote": payloads}

    worker = object.__new__(NixlConnectorWorker)
    worker.transfer_topo = SimpleNamespace(tp_ratio=lambda _: 1)
    worker.nixl_wrapper = FakeNotifications()
    worker.world_size = 8
    worker._reqs_to_send = {"req-0": float("inf")}
    worker._reqs_to_process = {"req-0"}
    worker._oscar_mla_send_generations = {"req-0": 7}
    worker.consumer_notification_counts_by_req = defaultdict(int)

    assert worker._get_new_notifs() == set()
    assert "req-0" in worker._reqs_to_send

    worker.nixl_wrapper.payloads = [
        b"OSCAR:req-0:7:8",
        b"OSCAR:req-0:7:8",
    ]
    assert worker._get_new_notifs() == {"req-0"}
    assert "req-0" not in worker._reqs_to_send
    assert "req-0" not in worker._oscar_mla_send_generations
