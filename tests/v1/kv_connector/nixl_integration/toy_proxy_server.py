# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import itertools
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PREFILL_BLOCKING_STATES = {"quarantined", "restarting", "warming", "unavailable"}


class PrefillRouteUnavailable(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 503,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefiller_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "root_client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}",
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
                "state": "healthy",
                "state_reason": "startup",
                "state_updated_at": time.time(),
                "stuck_count": 0,
                "health_fail_count": 0,
                "last_metrics": {},
                "long_prefill_semaphore": asyncio.Semaphore(_long_prefill_limit()),
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decoder_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))
    app.state.prefill_monitor_task = asyncio.create_task(_prefill_monitor_loop(app))

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    app.state.prefill_monitor_task.cancel()
    try:
        await app.state.prefill_monitor_task
    except asyncio.CancelledError:
        pass

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()
        await client_info["root_client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def _long_prefill_limit() -> int:
    return max(1, int(os.environ.get("PREFILL_PROXY_LONG_PREFILL_LIMIT", "1")))


def _prefill_route_policy() -> str:
    return os.environ.get(
        "PREFILL_PROXY_LONG_PREFILL_ROUTE_POLICY", "least_available"
    ).strip().lower()


def _decode_first_chunk_timeout_s() -> float:
    raw = os.environ.get("PREFILL_PROXY_DECODE_FIRST_CHUNK_TIMEOUT_S", "900")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 900.0


def _prefill_health_interval_s() -> float:
    raw = os.environ.get("PREFILL_PROXY_HEALTH_INTERVAL_SEC", "10")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 10.0


def _prefill_health_timeout_s() -> float:
    raw = os.environ.get("PREFILL_PROXY_HEALTH_TIMEOUT_SEC", "15")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 15.0


def _prefill_health_fail_threshold() -> int:
    raw = os.environ.get("PREFILL_PROXY_HEALTH_FAIL_THRESHOLD", "3")
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _prefill_stuck_fail_threshold() -> int:
    raw = os.environ.get("PREFILL_PROXY_STUCK_FAIL_THRESHOLD", "2")
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _context_length_guard_enabled() -> bool:
    return os.environ.get(
        "PREFILL_PROXY_CONTEXT_LENGTH_GUARD", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _max_model_len() -> int:
    raw = os.environ.get("PREFILL_PROXY_MAX_MODEL_LEN", "202752")
    try:
        return max(1, int(raw))
    except ValueError:
        return 202752


def _context_length_safety_margin() -> int:
    raw = os.environ.get("PREFILL_PROXY_CONTEXT_LENGTH_SAFETY_MARGIN", "1024")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1024


def _coerce_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _requested_output_tokens(req_data: dict) -> int | None:
    fields = _requested_output_token_fields(req_data)
    if not fields:
        return None
    return max(fields.values())


def _requested_output_token_fields(req_data: dict) -> dict[str, int]:
    fields: dict[str, int] = {}
    for key in ("max_tokens", "max_completion_tokens"):
        value = _coerce_int(req_data.get(key))
        if value is not None:
            fields[key] = value
    return fields


def _system_content_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def _normalize_anthropic_system_messages(
    req_data: dict,
    request_id: str,
    api: str,
) -> dict:
    messages = req_data.get("messages")
    if not isinstance(messages, list):
        return req_data

    normalized_messages = []
    moved_system_parts: list[str] = []
    moved = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            moved += 1
            text = _system_content_to_text(message.get("content"))
            if text:
                moved_system_parts.append(text)
            continue
        normalized_messages.append(message)

    if moved == 0:
        return req_data

    normalized = req_data.copy()
    normalized["messages"] = normalized_messages
    existing_system = _system_content_to_text(normalized.get("system"))
    system_parts = [part for part in [existing_system, *moved_system_parts] if part]
    if system_parts:
        normalized["system"] = "\n\n".join(system_parts)
    else:
        normalized.pop("system", None)
    print(
        "PROXY_ANTHROPIC_SYSTEM_NORMALIZED "
        f"request_id={request_id} api={api} moved_system_messages={moved}",
        flush=True,
    )
    return normalized


def _context_length_error_response(
    *,
    request_id: str,
    api: str,
    input_tokens: int,
    max_tokens: int,
    max_model_len: int,
    safety_margin: int = 0,
    allowed_output: int | None = None,
) -> JSONResponse:
    total_tokens = input_tokens + max_tokens
    if allowed_output is not None and allowed_output <= 0:
        message = (
            f"This model's maximum context length is {max_model_len} tokens. "
            f"After reserving {safety_margin} safety tokens, your prompt "
            f"contains {input_tokens} input tokens and leaves no room for "
            "generation. Please reduce the length of the input prompt."
        )
    else:
        message = (
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {max_tokens} output tokens and your prompt "
            f"contains {input_tokens} input tokens, for a total of {total_tokens} "
            "tokens. Please reduce the length of the input prompt or the number "
            "of requested output tokens."
        )
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "input_tokens",
            },
            "details": {
                "request_id": request_id,
                "api": api,
                "input_tokens": input_tokens,
                "max_tokens": max_tokens,
                "total_tokens": total_tokens,
                "max_model_len": max_model_len,
                "safety_margin": safety_margin,
                "allowed_output": allowed_output,
                "guard": "proxy_pre_prefill",
            },
        },
    )


def _context_precheck_error_response(
    *,
    request_id: str,
    api: str,
    status_code: int,
    message: str,
    details: dict | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "service_unavailable",
                "code": "context_length_precheck_failed",
            },
            "details": {
                "request_id": request_id,
                "api": api,
                **(details or {}),
            },
        },
    )


def _extract_decode_error_message(body: object, raw_text: str) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
        for key in ("message", "detail"):
            message = body.get(key)
            if message:
                return str(message)
    text = raw_text.strip()
    return text[:1024]


def _extract_decode_error_input_tokens(body: object, raw_text: str) -> int | None:
    message = _extract_decode_error_message(body, raw_text)
    candidates = [message, raw_text]
    patterns = [
        r"parameter=input_tokens,\s*value=(\d+)",
        r"contains at least (\d+) input tokens",
        r"contains (\d+) input tokens",
    ]
    for text in candidates:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
    return None


async def _count_messages_tokens_with_decode(
    request: Request,
    req_data: dict,
    request_id: str,
) -> tuple[int | None, JSONResponse | None]:
    decode_client_info = get_next_client(request.app, "decode")
    payload = req_data.copy()
    payload.pop("stream", None)
    payload.pop("stream_options", None)
    headers = {
        "Authorization": request.headers.get(
            "authorization", f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        ),
        "X-Request-Id": request_id,
    }
    response = await decode_client_info["client"].post(
        "/messages/count_tokens", json=payload, headers=headers
    )
    if response.status_code >= 400:
        text = response.text
        try:
            body = response.json()
        except Exception:
            body = {"raw": text[:4096]}
        print(
            "PROXY_CONTEXT_LENGTH_COUNT_ERROR "
            f"request_id={request_id} status_code={response.status_code} "
            f"decode={decode_client_info['host']}:{decode_client_info['port']} "
            f"body={text[:4096]!r}",
            flush=True,
        )
        input_tokens_from_error = _extract_decode_error_input_tokens(body, text)
        if input_tokens_from_error is not None:
            print(
                "PROXY_CONTEXT_LENGTH_COUNT_INPUT_TOKENS_FROM_ERROR "
                f"request_id={request_id} input_tokens={input_tokens_from_error} "
                f"decode={decode_client_info['host']}:{decode_client_info['port']}",
                flush=True,
            )
            return input_tokens_from_error, None
        upstream_message = _extract_decode_error_message(body, text)
        message = "context length precheck failed before prefill"
        if upstream_message:
            message = f"{message}: {upstream_message}"
        return None, _context_precheck_error_response(
            request_id=request_id,
            api="/messages",
            status_code=(
                response.status_code
                if 400 <= response.status_code < 500
                else 503
            ),
            message=message,
            details={
                "upstream_status_code": response.status_code,
                "upstream_message": upstream_message,
                "upstream_body": body,
                "decode": f"{decode_client_info['host']}:{decode_client_info['port']}",
                "guard": "proxy_pre_prefill",
            },
        )
    try:
        body = response.json()
    except Exception as exc:
        return None, _context_precheck_error_response(
            request_id=request_id,
            api="/messages",
            status_code=503,
            message="context length precheck returned invalid JSON before prefill",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
    input_tokens = body.get("input_tokens")
    if not isinstance(input_tokens, int):
        return None, _context_precheck_error_response(
            request_id=request_id,
            api="/messages",
            status_code=503,
            message="context length precheck did not return input_tokens",
            details={"upstream_body": body, "guard": "proxy_pre_prefill"},
        )
    return input_tokens, None


async def _prevalidate_context_length(
    api: str,
    request: Request,
    req_data: dict,
    request_id: str,
) -> JSONResponse | None:
    if not _context_length_guard_enabled() or api != "/messages":
        return None
    output_fields = _requested_output_token_fields(req_data)
    if not output_fields:
        return None
    input_tokens, error_response = await _count_messages_tokens_with_decode(
        request, req_data, request_id
    )
    if error_response is not None:
        return error_response
    assert input_tokens is not None
    max_model_len = _max_model_len()
    safety_margin = _context_length_safety_margin()
    allowed_output = max_model_len - input_tokens - safety_margin
    max_requested_output = max(output_fields.values())
    if allowed_output <= 0:
        print(
            "PROXY_CONTEXT_LENGTH_REJECT "
            f"request_id={request_id} api={api} input_tokens={input_tokens} "
            f"max_requested_output={max_requested_output} "
            f"allowed_output={allowed_output} safety_margin={safety_margin} "
            f"max_model_len={max_model_len} guard=proxy_pre_prefill",
            flush=True,
        )
        return _context_length_error_response(
            request_id=request_id,
            api=api,
            input_tokens=input_tokens,
            max_tokens=max_requested_output,
            max_model_len=max_model_len,
            safety_margin=safety_margin,
            allowed_output=allowed_output,
        )
    clipped_fields = {}
    for field, value in output_fields.items():
        if value > allowed_output:
            req_data[field] = allowed_output
            clipped_fields[field] = value
    if not clipped_fields:
        return None
    print(
        "PROXY_CONTEXT_LENGTH_CLIP "
        f"request_id={request_id} api={api} input_tokens={input_tokens} "
        f"allowed_output={allowed_output} safety_margin={safety_margin} "
        f"max_model_len={max_model_len} clipped_fields="
        f"{json.dumps(clipped_fields, sort_keys=True)} "
        f"guard=proxy_pre_prefill",
        flush=True,
    )
    return None


def _prefill_state_dir() -> Path:
    return Path(
        os.environ.get("PREFILL_PROXY_STATE_DIR", "/tmp/vllm_prefill_proxy_state")
    )


def _prefill_state_file(client_info: dict) -> Path:
    return _prefill_state_dir() / f"prefill_{client_info['id']}.state"


def _prefill_events_file() -> Path:
    return _prefill_state_dir() / "events.jsonl"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _warmup_internal_token() -> str:
    return os.environ.get("PREFILL_PROXY_INTERNAL_TOKEN", "").strip()


def _is_internal_warmup(request: Request) -> bool:
    token = _warmup_internal_token()
    if not token:
        return False
    return request.headers.get("x-prefill-warmup-token", "") == token


def _append_prefill_event(event: dict) -> None:
    try:
        state_dir = _prefill_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        row = {"time_utc": _now_iso(), **event}
        with _prefill_events_file().open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"PREFILL_STATE_EVENT_WRITE_ERROR error={exc}", flush=True)


def _write_prefill_state_file(client_info: dict, state: str, reason: str) -> None:
    try:
        state_dir = _prefill_state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "state": state,
            "reason": reason,
            "updated_at": _now_iso(),
            "prefill_id": client_info["id"],
            "host": client_info["host"],
            "port": client_info["port"],
        }
        _prefill_state_file(client_info).write_text(
            json.dumps(data, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"PREFILL_STATE_WRITE_ERROR id={client_info['id']} error={exc}", flush=True)


def _read_prefill_state_file(client_info: dict) -> tuple[str | None, str | None]:
    path = _prefill_state_file(client_info)
    if not path.exists():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return None, None
        try:
            data = json.loads(text)
            return str(data.get("state", "")).strip().lower(), str(
                data.get("reason", "")
            )
        except json.JSONDecodeError:
            parts = text.split(maxsplit=1)
            state = parts[0].strip().lower()
            reason = parts[1].strip() if len(parts) > 1 else "external_state_file"
            return state, reason
    except Exception as exc:  # noqa: BLE001
        print(f"PREFILL_STATE_READ_ERROR id={client_info['id']} error={exc}", flush=True)
        return None, None


def _set_prefill_state(
    client_info: dict,
    state: str,
    reason: str,
    *,
    persist: bool = False,
) -> None:
    old_state = client_info.get("state", "unknown")
    old_reason = client_info.get("state_reason", "")
    client_info["state"] = state
    client_info["state_reason"] = reason
    client_info["state_updated_at"] = time.time()
    if state == "healthy":
        client_info["stuck_count"] = 0
    if old_state != state or old_reason != reason:
        print(
            "PREFILL_STATE_CHANGE "
            f"prefill_id={client_info['id']} "
            f"prefill={client_info['host']}:{client_info['port']} "
            f"old_state={old_state} state={state} reason={reason}",
            flush=True,
        )
        _append_prefill_event(
            {
                "event_type": "prefill_state_change",
                "prefill_id": client_info["id"],
                "prefill": f"{client_info['host']}:{client_info['port']}",
                "old_state": old_state,
                "state": state,
                "reason": reason,
            }
        )
    if persist:
        _write_prefill_state_file(client_info, state, reason)


def _apply_external_prefill_state(client_info: dict) -> None:
    external_state, external_reason = _read_prefill_state_file(client_info)
    if not external_state:
        return
    if external_state in {"healthy", "ready", "online"}:
        if client_info.get("state") != "healthy":
            _set_prefill_state(
                client_info,
                "healthy",
                external_reason or "external_state_healthy",
            )
        return
    if external_state in PREFILL_BLOCKING_STATES:
        _set_prefill_state(
            client_info,
            external_state,
            external_reason or "external_state_blocking",
        )


def _parse_metric_value(line: str) -> float | None:
    if not line or line.startswith("#"):
        return None
    try:
        return float(line.rsplit(None, 1)[-1])
    except (IndexError, ValueError):
        return None


def _parse_prefill_metrics(text: str) -> dict:
    metrics = {"running": None, "waiting": None, "capacity_waiting": 0.0}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        value = _parse_metric_value(line)
        if value is None:
            continue
        if "num_requests_running" in line:
            metrics["running"] = value
        elif "num_requests_waiting" in line:
            metrics["waiting"] = value
        elif "waiting_by_reason" in line and re.search(r"capacity", line):
            metrics["capacity_waiting"] = max(float(metrics["capacity_waiting"]), value)
    running = metrics["running"]
    capacity_waiting = float(metrics["capacity_waiting"] or 0.0)
    metrics["stuck_signal"] = (
        running is not None and float(running) == 0.0 and capacity_waiting > 0.0
    )
    return metrics


async def _refresh_prefill_health(client_info: dict) -> None:
    _apply_external_prefill_state(client_info)
    timeout = _prefill_health_timeout_s()
    try:
        models_response = await client_info["client"].get(
            "/models", timeout=timeout
        )
        models_response.raise_for_status()
        metrics_response = await client_info["root_client"].get(
            "/metrics", timeout=timeout
        )
        metrics_response.raise_for_status()
        metrics = _parse_prefill_metrics(metrics_response.text)
        client_info["last_metrics"] = metrics
        if metrics.get("stuck_signal"):
            if str(client_info.get("state", "")).lower() in {
                "warming",
                "restarting",
            }:
                client_info["stuck_count"] = 0
                client_info["health_fail_count"] = 0
                return
            client_info["stuck_count"] = int(client_info.get("stuck_count", 0)) + 1
            if client_info["stuck_count"] >= _prefill_stuck_fail_threshold():
                _set_prefill_state(
                    client_info,
                    "quarantined",
                    (
                        "metrics_stuck "
                        f"running={metrics.get('running')} "
                        f"capacity_waiting={metrics.get('capacity_waiting')}"
                    ),
                    persist=True,
                )
            return
        if client_info.get("state") in {"unavailable"}:
            _set_prefill_state(
                client_info,
                "healthy",
                "health_probe_recovered",
                persist=True,
            )
        client_info["stuck_count"] = 0
        client_info["health_fail_count"] = 0
    except Exception as exc:  # noqa: BLE001
        client_info["health_fail_count"] = (
            int(client_info.get("health_fail_count", 0)) + 1
        )
        threshold = _prefill_health_fail_threshold()
        if client_info["health_fail_count"] < threshold and str(
            client_info.get("state", "")
        ).lower() in {"healthy", "warming"}:
            print(
                "PREFILL_HEALTH_PROBE_WARN "
                f"prefill_id={client_info['id']} "
                f"prefill={client_info['host']}:{client_info['port']} "
                f"consecutive={client_info['health_fail_count']}/{threshold} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        _set_prefill_state(
            client_info,
            "unavailable",
            f"health_probe_failed {type(exc).__name__}: {exc}",
            persist=True,
        )


async def _prefill_monitor_loop(app: FastAPI) -> None:
    while True:
        try:
            await asyncio.gather(
                *[
                    _refresh_prefill_health(client_info)
                    for client_info in app.state.prefill_clients
                ],
                return_exceptions=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"PREFILL_MONITOR_ERROR error={type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(_prefill_health_interval_s())


def _prefill_details(app: FastAPI) -> list[dict]:
    details = []
    for client_info in app.state.prefill_clients:
        _apply_external_prefill_state(client_info)
        details.append(
            {
                "id": client_info["id"],
                "host": client_info["host"],
                "port": client_info["port"],
                "state": client_info.get("state", "unknown"),
                "reason": client_info.get("state_reason", ""),
                "stuck_count": client_info.get("stuck_count", 0),
                "health_fail_count": client_info.get("health_fail_count", 0),
                "last_metrics": client_info.get("last_metrics", {}),
            }
        )
    return details


def _is_prefill_routable(client_info: dict, allow_warming: bool) -> bool:
    _apply_external_prefill_state(client_info)
    state = str(client_info.get("state", "healthy")).lower()
    if state == "healthy":
        return True
    if allow_warming and state == "warming":
        return True
    return False


def _openai_error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    request_id: str | None = None,
    details: dict | None = None,
) -> JSONResponse:
    body = {
        "error": {
            "message": message,
            "type": "prefill_path_error",
            "param": None,
            "code": code,
        }
    }
    if request_id:
        body["request_id"] = request_id
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def _prefill_failure_code(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else 0
        if status_code >= 500:
            return "current_prefill_node_unavailable"
    if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
        return "current_prefill_node_unavailable"
    if isinstance(exc, httpx.TransportError):
        return "prefill_path_unavailable"
    return "prefill_path_unavailable"


def _request_text_size(req_data: dict) -> int:
    messages = req_data.get("messages") or []
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("text", "")))
                else:
                    total += len(str(item))
        else:
            total += len(str(content))
    prompt = req_data.get("prompt")
    if isinstance(prompt, str):
        total += len(prompt)
    elif isinstance(prompt, list):
        total += sum(len(str(x)) for x in prompt)
    return total


def _is_long_prefill(req_data: dict) -> bool:
    threshold = int(os.environ.get("PREFILL_PROXY_MIN_DISAGG_CHARS", "2048"))
    return _request_text_size(req_data) >= threshold


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


def _semaphore_available_slots(semaphore: asyncio.Semaphore) -> int:
    # asyncio.Semaphore intentionally keeps this private; for this proxy it is
    # the least invasive way to avoid routing long prefills to a busy node.
    return max(0, int(getattr(semaphore, "_value", 0)))


def get_prefill_client(app, long_prefill: bool, allow_warming: bool = False):
    clients = app.state.prefill_clients
    if not clients:
        raise PrefillRouteUnavailable(
            "no_healthy_prefill_path", "no prefill clients configured"
        )

    start_idx = next(app.state.prefill_iterator)
    ordered_clients = [
        clients[(start_idx + offset) % len(clients)] for offset in range(len(clients))
    ]
    routable_clients = [
        client_info
        for client_info in ordered_clients
        if _is_prefill_routable(client_info, allow_warming)
    ]
    if not routable_clients:
        raise PrefillRouteUnavailable(
            "no_healthy_prefill_path",
            "no healthy Prefill path is available",
            details={"prefills": _prefill_details(app)},
        )

    if not long_prefill or _prefill_route_policy() not in {
        "least_available",
        "least_busy",
        "available",
    }:
        return routable_clients[0]

    if len(routable_clients) <= 1:
        return routable_clients[0]

    return max(
        routable_clients,
        key=lambda client_info: _semaphore_available_slots(
            client_info["long_prefill_semaphore"]
        ),
    )


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    # These args are not supported for P
    min_tokens = req_data.pop("min_tokens", None)
    min_completion_tokens = req_data.pop("min_completion_tokens", None)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    # read/consume the response body to release the connection
    # otherwise, it would http.ReadError
    await response.aread()

    # Add back the min_tokens and min_completion_tokens so D can use them
    req_data["min_tokens"] = min_tokens
    req_data["min_completion_tokens"] = min_completion_tokens

    return response


async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        chunk_iter = response.aiter_bytes().__aiter__()
        first_chunk = True
        timeout_s = _decode_first_chunk_timeout_s()
        while True:
            try:
                if first_chunk and timeout_s > 0:
                    chunk = await asyncio.wait_for(
                        chunk_iter.__anext__(), timeout=timeout_s
                    )
                else:
                    chunk = await chunk_iter.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                print(
                    "PROXY_STREAM_TIMEOUT "
                    f"request_id={request_id} endpoint={endpoint} "
                    f"timeout_s={timeout_s} "
                    f"decode={client_info['host']}:{client_info['port']}",
                    flush=True,
                )
                raise
            first_chunk = False
            yield chunk


def _passthrough_headers(request: Request) -> dict:
    headers = {
        "Authorization": request.headers.get(
            "authorization", f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        )
    }
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type
    request_id = request.headers.get("x-request-id")
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


async def _decode_v1_passthrough(endpoint: str, request: Request):
    decode_client_info = get_next_client(request.app, "decode")
    headers = _passthrough_headers(request)
    method = request.method

    if method in ("GET", "DELETE"):
        response = await decode_client_info["client"].request(
            method, endpoint, headers=headers
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    req_data = await request.json()
    if endpoint == "/messages/count_tokens" and isinstance(req_data, dict):
        request_id = str(uuid.uuid4())
        req_data = _normalize_anthropic_system_messages(
            req_data, request_id, endpoint
        )
    if req_data.get("stream"):

        async def generate_stream():
            async with decode_client_info["client"].stream(
                method, endpoint, json=req_data, headers=headers
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    response = await decode_client_info["client"].request(
        method, endpoint, json=req_data, headers=headers
    )
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


async def _decode_root_passthrough(endpoint: str, request: Request):
    decode_client_info = get_next_client(request.app, "decode")
    headers = _passthrough_headers(request)
    url = f"http://{decode_client_info['host']}:{decode_client_info['port']}{endpoint}"
    body = await request.body()
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.request(
            request.method,
            url,
            headers=headers,
            content=body if body else None,
        )
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


async def _decode_prefixed_v1_passthrough(endpoint: str, request: Request):
    return await _decode_v1_passthrough(endpoint, request)


async def _handle_completions(api: str, request: Request):
    try:
        t0 = time.perf_counter()
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        if api == "/messages":
            req_data = _normalize_anthropic_system_messages(
                req_data, request_id, api
            )
        text_chars = _request_text_size(req_data)
        long_prefill = _is_long_prefill(req_data)
        context_error = await _prevalidate_context_length(
            api, request, req_data, request_id
        )
        if context_error is not None:
            return context_error
        max_tokens = _requested_output_tokens(req_data)

        # Prefer an idle Prefill node for long prompts. Plain round-robin can
        # queue a request behind an already-running long prefill while another
        # Prefill node is idle, and that wait used to be reported as prefill_ms.
        allow_warming = _is_internal_warmup(request)
        prefill_client_info = get_prefill_client(
            request.app, long_prefill, allow_warming=allow_warming
        )

        # Send request to prefill service
        prefill_queue_ms = 0.0
        t_prefill_rpc_start = time.perf_counter()
        try:
            if long_prefill:
                semaphore = prefill_client_info["long_prefill_semaphore"]
                t_prefill_wait_start = time.perf_counter()
                await semaphore.acquire()
                t_prefill_rpc_start = time.perf_counter()
                prefill_queue_ms = (t_prefill_rpc_start - t_prefill_wait_start) * 1000.0
                try:
                    response = await send_request_to_service(
                        prefill_client_info, api, req_data, request_id
                    )
                finally:
                    semaphore.release()
            else:
                response = await send_request_to_service(
                    prefill_client_info, api, req_data, request_id
                )
        except Exception as exc:  # noqa: BLE001
            code = _prefill_failure_code(exc)
            _set_prefill_state(
                prefill_client_info,
                "unavailable",
                f"prefill_request_failed {type(exc).__name__}: {exc}",
                persist=True,
            )
            print(
                "PROXY_PREFILL_ERROR "
                f"request_id={request_id} api={api} "
                f"code={code} error={type(exc).__name__}: {exc} "
                f"prefill_id={prefill_client_info['id']} "
                f"prefill={prefill_client_info['host']}:{prefill_client_info['port']}",
                flush=True,
            )
            return _openai_error_response(
                503,
                code,
                "current Prefill node is unavailable",
                request_id=request_id,
                details={
                    "prefill_id": prefill_client_info["id"],
                    "prefill": f"{prefill_client_info['host']}:{prefill_client_info['port']}",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        t_prefill_done = time.perf_counter()
        prefill_rpc_ms = (t_prefill_done - t_prefill_rpc_start) * 1000.0

        # Extract the needed fields
        response_json = response.json()
        await response.aclose()  # CRITICAL: Release connection back to pool
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, "decode")

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            t_decode_start = time.perf_counter()
            first_chunk = True
            chunk_count = 0
            try:
                async for chunk in stream_service_response(
                    decode_client_info, api, req_data, request_id=request_id
                ):
                    chunk_count += 1
                    if first_chunk:
                        first_chunk = False
                        t_first_chunk = time.perf_counter()
                        print(
                            "PROXY_TIMING "
                            f"request_id={request_id} api={api} "
                            f"max_tokens={max_tokens} "
                            f"text_chars={text_chars} "
                            f"long_prefill={long_prefill} "
                            f"prefill_ms={(t_prefill_done - t0) * 1000:.3f} "
                            f"prefill_queue_ms={prefill_queue_ms:.3f} "
                            f"prefill_rpc_ms={prefill_rpc_ms:.3f} "
                            f"decode_first_ms={(t_first_chunk - t_decode_start) * 1000:.3f} "
                            f"ttft_proxy_ms={(t_first_chunk - t0) * 1000:.3f} "
                            f"prefill_route_policy={_prefill_route_policy()} "
                            f"prefill_id={prefill_client_info['id']} "
                            f"prefill={prefill_client_info['host']}:{prefill_client_info['port']} "
                            f"decode={decode_client_info['host']}:{decode_client_info['port']}",
                            flush=True,
                        )
                    yield chunk
            except Exception as e:
                t_error = time.perf_counter()
                print(
                    "PROXY_STREAM_ERROR "
                    f"request_id={request_id} api={api} "
                    f"elapsed_ms={(t_error - t0) * 1000:.3f} "
                    f"chunks={chunk_count} error={type(e).__name__}: {e}",
                    flush=True,
                )
                raise
            if chunk_count == 0:
                t_empty = time.perf_counter()
                print(
                    "PROXY_STREAM_EMPTY "
                    f"request_id={request_id} api={api} "
                    f"elapsed_ms={(t_empty - t0) * 1000:.3f}",
                    flush=True,
                )
            t_done = time.perf_counter()
            print(
                "PROXY_DONE "
                f"request_id={request_id} api={api} "
                f"total_ms={(t_done - t0) * 1000:.3f} chunks={chunk_count}",
                flush=True,
            )

        return StreamingResponse(generate_stream(), media_type="application/json")

    except PrefillRouteUnavailable as e:
        return _openai_error_response(
            e.status_code,
            e.code,
            str(e),
            details=e.details,
        )
    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/v1/models")
async def handle_models(request: Request):
    return await _decode_v1_passthrough("/models", request)


@app.post("/v1/messages")
async def handle_messages(request: Request):
    return await _handle_completions("/messages", request)


@app.post("/v1/messages/count_tokens")
async def handle_messages_count_tokens(request: Request):
    return await _decode_v1_passthrough("/messages/count_tokens", request)


@app.post("/v1/chat/completions/batch")
async def handle_chat_completions_batch(request: Request):
    return await _decode_prefixed_v1_passthrough("/chat/completions/batch", request)


@app.post("/v1/chat/completions/render")
async def handle_chat_completions_render(request: Request):
    return await _decode_prefixed_v1_passthrough("/chat/completions/render", request)


@app.post("/v1/completions/render")
async def handle_completions_render(request: Request):
    return await _decode_prefixed_v1_passthrough("/completions/render", request)


@app.post("/v1/responses")
async def handle_responses(request: Request):
    return await _decode_prefixed_v1_passthrough("/responses", request)


@app.get("/v1/responses/{response_id}")
async def handle_response_get(response_id: str, request: Request):
    return await _decode_prefixed_v1_passthrough(f"/responses/{response_id}", request)


@app.post("/v1/responses/{response_id}/cancel")
async def handle_response_cancel(response_id: str, request: Request):
    return await _decode_prefixed_v1_passthrough(f"/responses/{response_id}/cancel", request)


@app.get("/health")
async def handle_health(request: Request):
    return await _decode_root_passthrough("/health", request)


@app.get("/metrics")
async def handle_metrics(request: Request):
    return await _decode_root_passthrough("/metrics", request)


@app.get("/version")
async def handle_version(request: Request):
    return await _decode_root_passthrough("/version", request)


@app.get("/ping")
async def handle_ping_get(request: Request):
    return await _decode_root_passthrough("/ping", request)


@app.post("/ping")
async def handle_ping_post(request: Request):
    return await _decode_root_passthrough("/ping", request)


@app.get("/load")
async def handle_load(request: Request):
    return await _decode_root_passthrough("/load", request)


@app.post("/tokenize")
async def handle_tokenize(request: Request):
    return await _decode_root_passthrough("/tokenize", request)


@app.post("/detokenize")
async def handle_detokenize(request: Request):
    return await _decode_root_passthrough("/detokenize", request)


@app.post("/generative_scoring")
async def handle_generative_scoring(request: Request):
    return await _decode_root_passthrough("/generative_scoring", request)


@app.post("/is_scaling_elastic_ep")
async def handle_is_scaling_elastic_ep(request: Request):
    return await _decode_root_passthrough("/is_scaling_elastic_ep", request)


@app.post("/scale_elastic_ep")
async def handle_scale_elastic_ep(request: Request):
    return await _decode_root_passthrough("/scale_elastic_ep", request)


@app.get("/prefill_health")
async def handle_prefill_health(request: Request):
    details = _prefill_details(request.app)
    available = [
        item
        for item in details
        if item.get("state") == "healthy"
        or (
            item.get("state") == "warming"
            and request.headers.get("x-prefill-warmup-token", "")
            == _warmup_internal_token()
        )
    ]
    return {
        "status": "ok" if available else "no_healthy_prefill_path",
        "prefill_instances": len(details),
        "prefill_available": len(available),
        "prefills": details,
        "state_dir": str(_prefill_state_dir()),
    }


@app.post("/admin/prefill/{prefill_id}/state/{state}")
async def handle_admin_prefill_state(prefill_id: int, state: str, request: Request):
    token = _warmup_internal_token()
    if token and request.headers.get("x-prefill-warmup-token", "") != token:
        return _openai_error_response(
            403,
            "forbidden_prefill_admin",
            "prefill admin token is required",
        )
    if not 0 <= prefill_id < len(request.app.state.prefill_clients):
        return _openai_error_response(
            404,
            "unknown_prefill_id",
            f"unknown prefill id {prefill_id}",
        )
    normalized = state.strip().lower()
    if normalized not in {"healthy", "quarantined", "restarting", "warming", "unavailable"}:
        return _openai_error_response(
            400,
            "invalid_prefill_state",
            f"invalid prefill state {state}",
        )
    try:
        body = await request.json()
    except Exception:
        body = {}
    reason = str(body.get("reason", "admin_state_update")) if isinstance(body, dict) else "admin_state_update"
    client_info = request.app.state.prefill_clients[prefill_id]
    _set_prefill_state(client_info, normalized, reason, persist=True)
    return {"status": "ok", "prefill_id": prefill_id, "state": normalized, "reason": reason}


@app.post("/invocations")
async def handle_invocations(request: Request):
    return await _decode_root_passthrough("/invocations", request)


@app.post("/inference/v1/generate")
async def handle_inference_v1_generate(request: Request):
    return await _decode_root_passthrough("/inference/v1/generate", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    details = _prefill_details(app)
    available = [item for item in details if item.get("state") == "healthy"]
    return {
        "status": "ok" if available else "no_healthy_prefill_path",
        "prefill_instances": len(app.state.prefill_clients),
        "prefill_available": len(available),
        "prefills": details,
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
