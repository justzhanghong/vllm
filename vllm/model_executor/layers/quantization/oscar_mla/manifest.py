"""Deterministic, official-suite-independent calibration manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class CalibrationTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


@dataclass(frozen=True)
class CalibrationSource:
    name: str
    category: str
    path: str
    uri: str
    revision: str
    file_sha256: str
    text_fields: tuple[str, ...]
    id_field: str | None = None

    def __post_init__(self) -> None:
        if self.category not in {"general", "math", "code"}:
            raise ValueError(f"unsupported calibration category {self.category!r}")
        if not self.name or not self.uri or not self.revision:
            raise ValueError("source name, URI, and revision must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.file_sha256):
            raise ValueError("source file_sha256 must be a lowercase SHA256")
        if not self.text_fields:
            raise ValueError("source text_fields must not be empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationSource:
        values = dict(data)
        values["text_fields"] = tuple(values["text_fields"])
        return cls(**values)


@dataclass(frozen=True)
class CalibrationManifestResult:
    manifest_sha256: str
    summary_sha256: str
    sample_counts: dict[str, dict[str, int]]
    token_counts: dict[str, dict[str, int]]
    excluded_official_exact: int
    excluded_duplicate: int


def build_calibration_manifest(
    *,
    sources: Sequence[CalibrationSource],
    official_manifest_path: str | Path,
    quotas: Mapping[str, Mapping[str, int]],
    tokenizer: CalibrationTokenizer,
    tokenizer_sha256: str,
    seed: int,
    holdout_fraction: float,
    max_chunk_tokens: int,
    output_path: str | Path,
) -> CalibrationManifestResult:
    """Build a fixed train/holdout manifest and fail if any quota is unmet."""
    _validate_build_inputs(
        sources,
        quotas,
        tokenizer_sha256,
        holdout_fraction,
        max_chunk_tokens,
    )
    official_hashes = _load_official_prompt_hashes(official_manifest_path)
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {
        (split, category): []
        for split, category_quotas in quotas.items()
        for category in category_quotas
    }
    excluded_official = 0
    for source in sources:
        actual_hash = _sha256_file(Path(source.path))
        if actual_hash != source.file_sha256:
            raise ValueError(
                f"source SHA256 mismatch for {source.name}: "
                f"{actual_hash} != {source.file_sha256}"
            )
        for row_index, row in _read_jsonl(Path(source.path)):
            source_id = str(
                row.get(source.id_field, row_index)
                if source.id_field is not None
                else row_index
            )
            text = "\n\n".join(
                str(row[field]).strip()
                for field in source.text_fields
                if row.get(field) not in (None, "")
            ).strip()
            if not text:
                continue
            content_sha256 = _normalized_text_sha256(text)
            if content_sha256 in official_hashes:
                excluded_official += 1
                continue
            split = _deterministic_split(
                source.name,
                source_id,
                seed=seed,
                holdout_fraction=holdout_fraction,
            )
            key = (split, source.category)
            if key not in candidates:
                continue
            candidates[key].append(
                {
                    "source": source,
                    "source_id": source_id,
                    "source_row": row_index,
                    "text": text,
                    "order": _stable_hash(f"{seed}:order:{source.name}:{source_id}"),
                }
            )

    entries: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    excluded_duplicate = 0
    sample_counts = {
        split: {category: 0 for category in category_quotas}
        for split, category_quotas in quotas.items()
    }
    token_counts = {
        split: {category: 0 for category in category_quotas}
        for split, category_quotas in quotas.items()
    }
    for split, category_quotas in quotas.items():
        for category, quota in category_quotas.items():
            records = sorted(
                candidates[(split, category)],
                key=lambda record: record["order"],
            )
            for record in records:
                source = record["source"]
                token_ids = tokenizer.encode(
                    record["text"],
                    add_special_tokens=False,
                )
                for chunk_index, start in enumerate(
                    range(0, len(token_ids), max_chunk_tokens)
                ):
                    remaining = quota - token_counts[split][category]
                    if remaining <= 0:
                        break
                    chunk_length = min(max_chunk_tokens, remaining)
                    chunk_ids = token_ids[start : start + chunk_length]
                    text = tokenizer.decode(
                        chunk_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    if not text:
                        continue
                    actual_ids = tokenizer.encode(text, add_special_tokens=False)
                    if len(actual_ids) > remaining:
                        actual_ids = actual_ids[:remaining]
                        text = tokenizer.decode(
                            actual_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ).strip()
                        actual_ids = tokenizer.encode(text, add_special_tokens=False)
                    content_sha256 = _normalized_text_sha256(text)
                    if content_sha256 in official_hashes:
                        excluded_official += 1
                        continue
                    if content_sha256 in seen_content:
                        excluded_duplicate += 1
                        continue
                    seen_content.add(content_sha256)
                    entry = {
                        "id": (
                            f"{source.name}:{record['source_id']}:"
                            f"chunk_{chunk_index:04d}"
                        ),
                        "split": split,
                        "category": category,
                        "source_name": source.name,
                        "source_uri": source.uri,
                        "source_revision": source.revision,
                        "source_file_sha256": source.file_sha256,
                        "source_id": record["source_id"],
                        "source_row": record["source_row"],
                        "chunk_index": chunk_index,
                        "text_sha256": content_sha256,
                        "tokens": len(actual_ids),
                        "text": text,
                    }
                    entries.append(entry)
                    sample_counts[split][category] += 1
                    token_counts[split][category] += len(actual_ids)
                if token_counts[split][category] >= quota:
                    break
            if token_counts[split][category] < quota:
                raise ValueError(
                    f"calibration quota unmet for {split}/{category}: "
                    f"{token_counts[split][category]} < {quota}"
                )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output)
    manifest_sha256 = _sha256_file(output)
    summary = {
        "format_version": 1,
        "manifest_sha256": manifest_sha256,
        "official_manifest_sha256": _sha256_file(Path(official_manifest_path)),
        "tokenizer_sha256": tokenizer_sha256,
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "max_chunk_tokens": max_chunk_tokens,
        "quotas": {split: dict(values) for split, values in quotas.items()},
        "sample_counts": sample_counts,
        "token_counts": token_counts,
        "excluded_official_exact": excluded_official,
        "excluded_duplicate": excluded_duplicate,
        "sources": [asdict(source) for source in sources],
    }
    summary_path = output.with_suffix(".summary.json")
    summary_temporary = summary_path.with_suffix(".json.tmp")
    summary_temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(summary_temporary, summary_path)
    return CalibrationManifestResult(
        manifest_sha256=manifest_sha256,
        summary_sha256=_sha256_file(summary_path),
        sample_counts=sample_counts,
        token_counts=token_counts,
        excluded_official_exact=excluded_official,
        excluded_duplicate=excluded_duplicate,
    )


def _validate_build_inputs(
    sources: Sequence[CalibrationSource],
    quotas: Mapping[str, Mapping[str, int]],
    tokenizer_sha256: str,
    holdout_fraction: float,
    max_chunk_tokens: int,
) -> None:
    if not sources:
        raise ValueError("calibration sources must not be empty")
    if set(quotas) != {"train", "holdout"}:
        raise ValueError("calibration quotas must contain train and holdout")
    if any(
        category not in {"general", "math", "code"} or tokens <= 0
        for category_quotas in quotas.values()
        for category, tokens in category_quotas.items()
    ):
        raise ValueError(
            "calibration quotas contain an invalid category or token count"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", tokenizer_sha256):
        raise ValueError("tokenizer_sha256 must be a lowercase SHA256")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")


def _load_official_prompt_hashes(path: str | Path) -> set[str]:
    hashes = set()
    for _, row in _read_jsonl(Path(path)):
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            hashes.add(_normalized_text_sha256(prompt))
    if not hashes:
        raise ValueError("official manifest contains no prompts")
    return hashes


def _read_jsonl(path: Path):
    try:
        with path.open(encoding="utf-8") as stream:
            for row_index, line in enumerate(stream):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{row_index + 1} is not an object")
                    yield row_index, row
    except FileNotFoundError as error:
        raise ValueError(f"missing JSONL source: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSONL source {path}: {error}") from error


def _deterministic_split(
    source_name: str,
    source_id: str,
    *,
    seed: int,
    holdout_fraction: float,
) -> str:
    value = int(_stable_hash(f"{seed}:split:{source_name}:{source_id}")[:16], 16)
    return "holdout" if value / 2**64 < holdout_fraction else "train"


def _normalized_text_sha256(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise ValueError(f"missing file: {path}") from error
    return digest.hexdigest()
