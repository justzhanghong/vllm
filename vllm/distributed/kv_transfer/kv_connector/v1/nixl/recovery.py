# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for cluster-safe recovery of failed NIXL remote engines.

A NIXL agent is process-local, but a PPxTP decode engine has one process per
worker. Removing an active remote agent can tear down the producer UCX endpoint
and task 95321 showed that this remains unsafe even after all decode workers
drain. ``SharedRecoveryBarrier`` therefore fences new READs while every worker
quiesces; hot recovery retains all native state and revalidates metadata.
"""

import contextlib
import errno
import hashlib
import json
import os
import uuid
from pathlib import Path


def retain_remote_engine_state(
    worker, engine_id, extra_remote_agents=None, logger=None
):
    """Retain and expose all state needed to reuse a live remote engine.

    This is the only helper used by hot transfer-failure recovery. It performs
    no NIXL, dlist, address, block-metadata, or topology teardown. Partial agents
    registered by a failed multi-rank handshake are merged into the published
    map so the subsequent metadata revalidation reuses them.
    """
    remote_agents = worker._remote_agents.setdefault(engine_id, {})
    conflicts = 0
    for rank, agent_name in (extra_remote_agents or {}).items():
        previous = remote_agents.get(rank)
        if previous is None:
            remote_agents[rank] = agent_name
        elif previous != agent_name:
            # Never remove either native object on a hot path. The previously
            # published rank remains authoritative for the process lifetime.
            conflicts += 1
            if logger is not None:
                logger.warning(
                    "NIXL retained-agent rank conflict for engine %s rank %s: "
                    "published=%s partial=%s",
                    engine_id,
                    rank,
                    previous,
                    agent_name,
                )
    return {
        "remote_agents": len(set(remote_agents.values())),
        "dlist_handles": len(
            set(worker.dst_xfer_side_handles.get(engine_id, {}).values())
        ),
        "base_addresses": engine_id in worker.kv_caches_base_addr,
        "block_metadata": engine_id in worker.dst_num_blocks,
        "physical_ratio": engine_id in worker._physical_blocks_per_logical,
        "agent_conflicts": conflicts,
    }


def cleanup_remote_engine_state(
    worker, engine_id, extra_remote_agents=None, logger=None
):
    """Destructively remove cached state for shutdown or stale-engine TTL only.

    Never call this helper from hot transfer-failure recovery: removing a live
    remote agent can crash the producer's UCX progress thread. Native cleanup is
    best-effort; Python-side state is removed for an explicit terminal lifecycle.
    """
    remote_agents = worker._remote_agents.pop(engine_id, {})
    if extra_remote_agents:
        remote_agents = dict(remote_agents)
        remote_agents.update(extra_remote_agents)
    dst_handles = worker.dst_xfer_side_handles.pop(engine_id, {})
    had_base_addresses = engine_id in worker.kv_caches_base_addr
    had_block_metadata = engine_id in worker.dst_num_blocks
    had_physical_ratio = engine_id in worker._physical_blocks_per_logical
    for dlist_handle in set(dst_handles.values()):
        try:
            worker.nixl_wrapper.release_dlist_handle(dlist_handle)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "NIXL reconnect cleanup: release dlist failed for engine %s: %s",
                    engine_id,
                    exc,
                )
    for agent_name in set(remote_agents.values()):
        try:
            if logger is not None:
                logger.info(
                    "NIXL_NATIVE_REMOTE_AGENT_REMOVE "
                    "reason=stale_or_terminal_cleanup engine=%s agent=%s",
                    engine_id,
                    agent_name,
                )
            worker.nixl_wrapper.remove_remote_agent(agent_name)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "NIXL reconnect cleanup: remove invalid agent %s "
                    "for engine %s failed: %s",
                    agent_name,
                    engine_id,
                    exc,
                )
    worker.kv_caches_base_addr.pop(engine_id, None)
    worker.dst_num_blocks.pop(engine_id, None)
    worker._physical_blocks_per_logical.pop(engine_id, None)
    topology_unregistered = False
    if worker.transfer_topo is not None:
        try:
            worker.transfer_topo.unregister_remote_engine(engine_id)
            topology_unregistered = True
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "NIXL reconnect cleanup: unregister topology for engine "
                    "%s failed: %s",
                    engine_id,
                    exc,
                )
    return {
        "remote_agents": len(set(remote_agents.values())),
        "dlist_handles": len(set(dst_handles.values())),
        "base_addresses": had_base_addresses,
        "block_metadata": had_block_metadata,
        "physical_ratio": had_physical_ratio,
        "topology_unregistered": topology_unregistered,
    }


def _read_matches(path, text):
    try:
        return Path(path).read_text(encoding="utf-8") == text
    except OSError:
        return False


def _exclusive_write(path, text):
    """Best-effort create-once fallback for filesystems without hard links."""
    path = Path(path)
    created = False
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        return True
    except FileExistsError:
        return _read_matches(path, text)
    except OSError:
        # Do not leave a permanently partial create-once marker. A later
        # EngineCore poll retries publication while recovery stays active.
        if created:
            with contextlib.suppress(OSError):
                path.unlink()
        return _read_matches(path, text)


def _publish_once(path, text, attempts=4):
    """Publish an immutable coordination marker without replacing its target.

    All writers for one marker publish identical content. ``os.replace`` is a
    poor fit on NFS: concurrent replacement of a still-open inode can turn it
    into a ``.nfs*`` file and fail with ``EBUSY``. A same-directory temporary
    plus ``link(2)`` gives atomic create-once semantics instead. A competing
    winner is accepted only after its payload is validated. Coordination I/O
    errors are contained here; callers keep the recovery generation active and
    retry on a later poll rather than terminating EngineCore.
    """
    path = Path(path)
    if _read_matches(path, text):
        return True

    retry_errnos = {
        errno.EBUSY,
        errno.EEXIST,
        errno.EINTR,
        errno.ENOENT,
        getattr(errno, "ESTALE", 116),
    }
    unsupported_errnos = {
        errno.EPERM,
        errno.EXDEV,
        getattr(errno, "EOPNOTSUPP", 95),
        getattr(errno, "ENOTSUP", 95),
    }
    for _ in range(max(1, int(attempts))):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with open(str(tmp), "x", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(str(tmp), str(path))
                return True
            except FileExistsError:
                if _read_matches(path, text):
                    return True
            except OSError as exc:
                if _read_matches(path, text):
                    return True
                if exc.errno in unsupported_errnos:
                    return _exclusive_write(path, text)
                if exc.errno not in retry_errnos:
                    return False
        except FileExistsError:
            # UUID collisions are implausible, but treating one as a retry keeps
            # this helper total and prevents filesystem errors escaping recovery.
            pass
        except OSError:
            if _read_matches(path, text):
                return True
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()
    return _read_matches(path, text)


class SharedRecoveryBarrier:
    """Task-scoped, generation-safe barrier for NIXL recovery commits.

    Each remote engine has immutable, monotonically numbered incident
    directories. The highest generation is current and becomes inactive by
    writing its own ``complete`` tombstone; directories are never renamed or
    removed. This avoids a mutable pointer that a delayed writer could regress.
    This matters because a late worker may still hold a path from the preceding
    poll. Its acknowledgement contains the incident id and therefore cannot be
    counted in (or complete) a newer generation.

    Every task gets a fresh root. All D PPxTP workers first acknowledge that
    they have no local transfer, handshake, or ready request. Only after all
    participants are quiescent may each process publish retained state and arm
    metadata revalidation. The on-disk ``cleaned`` name is legacy-compatible.
    """

    def __init__(self, root, participant_id, expected_participants):
        if not root:
            raise ValueError("VLLM_NIXL_RECOVERY_COORD_DIR must be set")
        if expected_participants <= 0:
            raise ValueError("expected_participants must be positive")
        self.root = Path(root)
        self.participant_id = str(participant_id)
        self.expected_participants = int(expected_participants)
        # A local pending declaration prevents a transient metadata publication
        # failure from being mistaken for a generation completed by a sibling.
        self._pending_declarations = {}
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _engine_key(engine_id):
        return hashlib.sha256(str(engine_id).encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _incident_identity(engine_id, generation):
        value = f"{engine_id}\0{generation:d}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]

    def _engine_dir(self, engine_id):
        return self.root / self._engine_key(engine_id)

    def _load_record(self, incident_path):
        try:
            data = json.loads(
                (incident_path / "incident.json").read_text(encoding="utf-8")
            )
            generation = int(data["generation"])
            incident_id = str(data["incident_id"])
            engine_id = str(data["engine_id"])
            if generation <= 0 or not incident_id or not engine_id:
                return None
            return {
                "generation": generation,
                "incident_id": incident_id,
                "engine_id": engine_id,
            }
        except (OSError, KeyError, ValueError, TypeError):
            return None

    def _load_current(self, engine_id):
        incidents = self._engine_dir(engine_id) / "incidents"
        records = []
        try:
            paths = list(incidents.iterdir())
        except OSError:
            return None
        for path in paths:
            if not path.is_dir():
                continue
            record = self._load_record(path)
            if record is not None and record["engine_id"] == str(engine_id):
                records.append(record)
        return max(records, key=lambda item: item["generation"]) if records else None

    def _incident_dir(self, engine_id, current):
        return (
            self._engine_dir(engine_id)
            / "incidents"
            / f"{current['generation']:08d}-{current['incident_id']}"
        )

    def _complete(self, engine_id, current):
        path = self._incident_dir(engine_id, current) / "complete"
        try:
            return path.read_text(encoding="utf-8").strip() == current["incident_id"]
        except OSError:
            return False

    def _current_active(self, engine_id):
        current = self._load_current(engine_id)
        if current is None or current["engine_id"] != str(engine_id):
            return None
        return None if self._complete(engine_id, current) else current

    def _count_acknowledgements(self, directory, incident_id):
        count = 0
        try:
            paths = list(directory.iterdir())
        except OSError:
            return 0
        for path in paths:
            # _atomic_write temporaries begin with '.', and only final files
            # whose payload matches this generation are participants.
            if path.name.startswith(".") or not path.is_file():
                continue
            try:
                if path.read_text(encoding="utf-8").strip() == incident_id:
                    count += 1
            except OSError:
                continue
        return count

    def declare(self, engine_id):
        """Join the active generation, or deterministically create its successor."""
        engine_id = str(engine_id)
        engine_dir = self._engine_dir(engine_id)
        engine_dir.mkdir(parents=True, exist_ok=True)
        (engine_dir / "incidents").mkdir(exist_ok=True)
        current = self._load_current(engine_id)
        if current is not None and not self._complete(engine_id, current):
            return self._incident_dir(engine_id, current)

        generation = (current["generation"] + 1) if current is not None else 1
        # All simultaneous declarers derive the same directory/id. A delayed
        # writer can only touch that immutable generation and cannot regress a
        # newer generation selected by _load_current().
        record = {
            "engine_id": engine_id,
            "generation": generation,
            "incident_id": self._incident_identity(engine_id, generation),
        }
        incident = self._incident_dir(engine_id, record)
        incident.mkdir(parents=True, exist_ok=True)
        (incident / "quiesced").mkdir(exist_ok=True)
        (incident / "cleaned").mkdir(exist_ok=True)
        self._pending_declarations[engine_id] = record
        published = _publish_once(
            incident / "incident.json",
            json.dumps(record, sort_keys=True) + "\n",
        )
        if published:
            self._pending_declarations.pop(engine_id, None)
        return incident

    def incident_id(self, engine_id):
        engine_id = str(engine_id)
        current = self._current_active(engine_id)
        if current is not None:
            self._pending_declarations.pop(engine_id, None)
            return current["incident_id"]
        pending = self._pending_declarations.get(engine_id)
        if pending is None:
            return None
        # Retry a transient publication failure instead of allowing the worker
        # to discard its local failed-engine state as if a sibling had completed
        # the generation.
        incident = self._incident_dir(engine_id, pending)
        try:
            incident.mkdir(parents=True, exist_ok=True)
            (incident / "quiesced").mkdir(exist_ok=True)
            (incident / "cleaned").mkdir(exist_ok=True)
        except OSError:
            return pending["incident_id"]
        published = _publish_once(
            incident / "incident.json",
            json.dumps(pending, sort_keys=True) + "\n",
        )
        if published:
            self._pending_declarations.pop(engine_id, None)
        return pending["incident_id"]

    def active_engines(self):
        engines = set()
        if not self.root.exists():
            return engines
        for engine_dir in self.root.iterdir():
            if not engine_dir.is_dir():
                continue
            # Engine IDs are stored in incident metadata; directory names are
            # hashes and intentionally cannot be reversed.
            records = []
            try:
                paths = list((engine_dir / "incidents").iterdir())
            except OSError:
                continue
            for path in paths:
                if path.is_dir():
                    record = self._load_record(path)
                    if record is not None:
                        records.append(record)
            if not records:
                continue
            current = max(records, key=lambda item: item["generation"])
            engine_id = current["engine_id"]
            if not self._complete(engine_id, current):
                engines.add(engine_id)
        return engines

    def any_active(self):
        return bool(self.active_engines())

    def is_active(self, engine_id):
        return self._current_active(engine_id) is not None

    def acknowledge_quiescent(self, engine_id, incident_id=None):
        current = self._current_active(engine_id)
        if current is None:
            return 0, False
        expected_id = incident_id or current["incident_id"]
        if current["incident_id"] != expected_id:
            return 0, False
        incident = self._incident_dir(engine_id, current)
        quiesced = incident / "quiesced"
        _publish_once(quiesced / self.participant_id, expected_id + "\n")
        count = self._count_acknowledgements(quiesced, expected_id)
        all_path = incident / "all_quiescent"
        if count >= self.expected_participants:
            _publish_once(all_path, expected_id + "\n")
        try:
            ready = all_path.read_text(encoding="utf-8").strip() == expected_id
        except OSError:
            ready = False
        return count, ready

    def mark_cleaned(self, engine_id, incident_id=None):
        current = self._load_current(engine_id)
        if current is None:
            return 0, False
        expected_id = incident_id or current["incident_id"]
        if current["incident_id"] != expected_id:
            # A newer generation superseded this caller; its old work is done.
            return 0, True
        incident = self._incident_dir(engine_id, current)
        if self._complete(engine_id, current):
            return self.expected_participants, True
        cleaned = incident / "cleaned"
        _publish_once(cleaned / self.participant_id, expected_id + "\n")
        count = self._count_acknowledgements(cleaned, expected_id)
        complete = count >= self.expected_participants
        if complete:
            # Never rename/remove the incident: a late process may hold this
            # path. The generation-specific tombstone is an idempotent commit.
            complete = _publish_once(incident / "complete", expected_id + "\n")
        return count, complete
