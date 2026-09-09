"""Download and select a reproducible MS MARCO passage subset."""

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from scaling_config import ScalingConfigError, identity_hash


SelectionManifest = dict[str, Any]

SOURCE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
ARCHIVE_NAME = "collectionandqueries.tar.gz"
EXPECTED_ARCHIVE_SHA256 = "decac356eb8cc5b9cea2e30b8738dc6f367e4147aaefe8fc7526ddda382fd2fc"
COLLECTION_MEMBER = "collection.tsv"
QUERY_MEMBER = "queries.dev.small.tsv"
QRELS_MEMBER = "qrels.dev.small.tsv"
EXPECTED_COLLECTION_ROWS = 8_841_823
EXPECTED_QUERY_ROWS = 6_980
EXPECTED_QREL_ROWS = 7_437
DOWNLOAD_TIMEOUT_SECONDS = 60

_OUTPUT_FILES = {
    "source_positions": "source-positions.npy",
    "document_ids": "document-ids.jsonl",
    "documents": "documents.jsonl",
    "queries": "queries.jsonl",
    "provider_qrels": "provider-qrels.tsv",
}


def _invalid(code: str, field: str, message: str) -> None:
    raise ScalingConfigError(code, field, message)


def _integer(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _invalid("INPUT_MISMATCH", field, f"must be an integer >= {minimum}")
    return value


def _cached_integer(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _invalid("CACHE_INVALID", field, f"must be an integer >= {minimum}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_line(stream, value: dict[str, str]) -> None:
    json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    stream.write("\n")


def _decode_tsv_line(raw_line: bytes, member_name: str, row: int) -> str:
    try:
        line = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        _invalid("INPUT_MISMATCH", member_name, f"row {row} is not valid UTF-8: {error}")
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def _read_id_text(raw_line: bytes, member_name: str, row: int) -> tuple[str, str]:
    line = _decode_tsv_line(raw_line, member_name, row)
    if "\t" not in line:
        _invalid("INPUT_MISMATCH", member_name, f"row {row} must contain an ID and text")
    row_id, text = line.split("\t", 1)
    if not row_id:
        _invalid("INPUT_MISMATCH", member_name, f"row {row} has an empty ID")
    return row_id, text


def select_positions(total_rows: int, count: int, seed: int) -> np.ndarray:
    """Return unique source rows in the saved random order."""
    total_rows = _integer(total_rows, "total_rows", 1)
    count = _integer(count, "count", 1)
    seed = _integer(seed, "seed", 0)
    if count > total_rows:
        _invalid("INPUT_MISMATCH", "count", "cannot exceed total_rows")
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.permutation(total_rows)[:count].astype(np.int64)


def _download_archive(cache_dir: Path) -> tuple[Path, bool]:
    archive_path = cache_dir / ARCHIVE_NAME
    if archive_path.is_file():
        try:
            digest = _sha256(archive_path)
        except OSError as error:
            _invalid("CACHE_INVALID", str(archive_path), str(error))
        if digest != EXPECTED_ARCHIVE_SHA256:
            _invalid(
                "INPUT_MISMATCH",
                str(archive_path),
                f"archive SHA256 is {digest}, expected {EXPECTED_ARCHIVE_SHA256}",
            )
        return archive_path, False
    if archive_path.exists():
        _invalid("CACHE_INVALID", str(archive_path), "archive path is not a regular file")

    partial_path = cache_dir / f"{ARCHIVE_NAME}.partial"
    partial_path.unlink(missing_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "bitplane-study/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with partial_path.open("wb") as output:
                shutil.copyfileobj(response, output)
    except Exception as error:
        partial_path.unlink(missing_ok=True)
        _invalid("DOWNLOAD_FAILED", SOURCE_URL, str(error))

    digest = _sha256(partial_path)
    if digest != EXPECTED_ARCHIVE_SHA256:
        partial_path.unlink(missing_ok=True)
        _invalid(
            "INPUT_MISMATCH",
            SOURCE_URL,
            f"downloaded archive SHA256 is {digest}, expected {EXPECTED_ARCHIVE_SHA256}",
        )
    return partial_path, True


def _member_stream(archive: tarfile.TarFile, member: tarfile.TarInfo) -> BinaryIO:
    stream = archive.extractfile(member)
    if stream is None:
        _invalid("INPUT_MISMATCH", member.name, "archive member cannot be read")
    return stream


def _read_collection(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    positions: np.ndarray,
) -> tuple[list[str], list[str], int, str]:
    selected_ids = [""] * len(positions)
    selected_texts = [""] * len(positions)
    source_order = np.argsort(positions)
    sorted_positions = positions[source_order]
    selected_offset = 0
    digest = hashlib.sha256()
    row_count = 0

    with _member_stream(archive, member) as stream:
        for source_row, raw_line in enumerate(stream):
            digest.update(raw_line)
            document_id, text = _read_id_text(raw_line, member.name, source_row + 1)
            if (
                selected_offset < len(sorted_positions)
                and source_row == int(sorted_positions[selected_offset])
            ):
                output_row = int(source_order[selected_offset])
                selected_ids[output_row] = document_id
                selected_texts[output_row] = text
                selected_offset += 1
            row_count += 1

    if row_count != EXPECTED_COLLECTION_ROWS:
        _invalid(
            "INPUT_MISMATCH",
            member.name,
            f"has {row_count} rows, expected {EXPECTED_COLLECTION_ROWS}",
        )
    if selected_offset != len(positions):
        _invalid("INPUT_MISMATCH", member.name, "selected source positions were not found")
    if len(set(selected_ids)) != len(selected_ids):
        _invalid("INPUT_MISMATCH", member.name, "selected document IDs must be unique")
    return selected_ids, selected_texts, row_count, digest.hexdigest()


def _read_queries(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    count: int,
    seed: int,
) -> tuple[list[str], list[str], set[str], int, str]:
    query_text_by_id: dict[str, str] = {}
    digest = hashlib.sha256()
    row_count = 0
    with _member_stream(archive, member) as stream:
        for source_row, raw_line in enumerate(stream):
            digest.update(raw_line)
            query_id, text = _read_id_text(raw_line, member.name, source_row + 1)
            if query_id in query_text_by_id:
                _invalid("INPUT_MISMATCH", member.name, f"duplicate query ID {query_id!r}")
            query_text_by_id[query_id] = text
            row_count += 1

    if row_count != EXPECTED_QUERY_ROWS:
        _invalid(
            "INPUT_MISMATCH",
            member.name,
            f"has {row_count} rows, expected {EXPECTED_QUERY_ROWS}",
        )
    sorted_ids = sorted(query_text_by_id)
    positions = select_positions(len(sorted_ids), count, seed)
    selected_ids = [sorted_ids[int(position)] for position in positions]
    selected_texts = [query_text_by_id[query_id] for query_id in selected_ids]
    return selected_ids, selected_texts, set(sorted_ids), row_count, digest.hexdigest()


def _copy_and_validate_qrels(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> tuple[set[str], int, str]:
    qrel_query_ids: set[str] = set()
    digest = hashlib.sha256()
    row_count = 0
    with _member_stream(archive, member) as stream, destination.open("wb") as output:
        for source_row, raw_line in enumerate(stream):
            digest.update(raw_line)
            output.write(raw_line)
            line = _decode_tsv_line(raw_line, member.name, source_row + 1)
            fields = line.split("\t")
            if len(fields) != 4 or not all(fields):
                _invalid("INPUT_MISMATCH", member.name, f"row {source_row + 1} is malformed")
            qrel_query_ids.add(fields[0])
            row_count += 1

    if row_count != EXPECTED_QREL_ROWS:
        _invalid(
            "INPUT_MISMATCH",
            member.name,
            f"has {row_count} rows, expected {EXPECTED_QREL_ROWS}",
        )
    return qrel_query_ids, row_count, digest.hexdigest()


def _write_selected_files(
    folder: Path,
    positions: np.ndarray,
    document_ids: list[str],
    document_texts: list[str],
    query_ids: list[str],
    query_texts: list[str],
) -> None:
    np.save(folder / _OUTPUT_FILES["source_positions"], positions, allow_pickle=False)
    with (folder / _OUTPUT_FILES["document_ids"]).open("w", encoding="utf-8") as output:
        for document_id in document_ids:
            _write_json_line(output, {"document_id": document_id})
    with (folder / _OUTPUT_FILES["documents"]).open("w", encoding="utf-8") as output:
        for document_id, text in zip(document_ids, document_texts, strict=True):
            _write_json_line(output, {"document_id": document_id, "text": text})
    with (folder / _OUTPUT_FILES["queries"]).open("w", encoding="utf-8") as output:
        for query_id, text in zip(query_ids, query_texts, strict=True):
            _write_json_line(output, {"query_id": query_id, "text": text})


def _identity(
    source: dict[str, Any],
    parameters: dict[str, int],
    document_count: int,
    query_ids: list[str],
    development_rows: list[int],
    evaluation_rows: list[int],
    file_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": source,
        "parameters": parameters,
        "document_count": document_count,
        "query_count": len(query_ids),
        "query_ids": query_ids,
        "development_rows": development_rows,
        "evaluation_rows": evaluation_rows,
        "file_sha256": file_hashes,
    }


def _validate_cached_manifest(path: Path) -> SelectionManifest:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _invalid("CACHE_INVALID", str(path), str(error))
    required = {
        "schema_version",
        "selection_hash",
        "identity",
        "source",
        "parameters",
        "document_count",
        "query_count",
        "query_ids",
        "development_rows",
        "evaluation_rows",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        _invalid("CACHE_INVALID", str(path), "manifest fields do not match schema version 1")
    schema_version = _cached_integer(manifest["schema_version"], "schema_version", 1)
    if schema_version != 1:
        _invalid("CACHE_INVALID", "schema_version", "unsupported schema version")
    if not isinstance(manifest["identity"], dict):
        _invalid("CACHE_INVALID", str(path), "identity must be an object")
    if manifest["identity"].get("schema_version") != schema_version:
        _invalid("CACHE_INVALID", "schema_version", "does not match identity")
    document_count = _cached_integer(manifest["document_count"], "document_count", 1)
    query_count = _cached_integer(manifest["query_count"], "query_count", 1)
    query_ids = manifest["query_ids"]
    if not isinstance(query_ids, list) or any(
        not isinstance(query_id, str) or not query_id for query_id in query_ids
    ):
        _invalid("CACHE_INVALID", "query_ids", "must contain nonempty strings")
    if len(query_ids) != query_count:
        _invalid("CACHE_INVALID", "query_count", "does not match query_ids")
    if manifest["identity"].get("document_count") != document_count:
        _invalid("CACHE_INVALID", "document_count", "does not match identity")
    if manifest["identity"].get("query_count") != query_count:
        _invalid("CACHE_INVALID", "query_count", "does not match identity")
    if not isinstance(manifest["selection_hash"], str):
        _invalid("CACHE_INVALID", str(path), "selection_hash must be a string")
    try:
        observed_selection_hash = identity_hash(manifest["identity"])
    except (TypeError, ValueError) as error:
        _invalid("CACHE_INVALID", str(path), f"identity cannot be hashed: {error}")
    if observed_selection_hash != manifest["selection_hash"]:
        _invalid("CACHE_INVALID", str(path), "selection hash does not match identity")
    if path.parent.name != manifest["selection_hash"]:
        _invalid("CACHE_INVALID", str(path), "selection folder does not match selection hash")
    if manifest["source"] != manifest["identity"].get("source"):
        _invalid("CACHE_INVALID", str(path), "source does not match identity")
    if manifest["parameters"] != manifest["identity"].get("parameters"):
        _invalid("CACHE_INVALID", str(path), "parameters do not match identity")

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(_OUTPUT_FILES):
        _invalid("CACHE_INVALID", str(path), "file records do not match schema version 1")
    observed_hashes: dict[str, str] = {}
    for name, filename in _OUTPUT_FILES.items():
        record = files[name]
        expected_path = (path.parent / filename).resolve()
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            _invalid("CACHE_INVALID", name, "file record must contain path and sha256")
        if not isinstance(record["path"], str) or not isinstance(record["sha256"], str):
            _invalid("CACHE_INVALID", name, "path and sha256 must be strings")
        if Path(record["path"]) != expected_path or not expected_path.is_file():
            _invalid("CACHE_INVALID", name, "file is missing or has an unexpected path")
        try:
            observed_hashes[name] = _sha256(expected_path)
        except OSError as error:
            _invalid("CACHE_INVALID", name, str(error))
        if observed_hashes[name] != record["sha256"]:
            _invalid("CACHE_INVALID", name, "file hash does not match manifest")

    rebuilt_identity = _identity(
        manifest["source"],
        manifest["parameters"],
        manifest["document_count"],
        manifest["query_ids"],
        manifest["development_rows"],
        manifest["evaluation_rows"],
        observed_hashes,
    )
    if rebuilt_identity != manifest["identity"]:
        _invalid("CACHE_INVALID", str(path), "manifest values do not match identity")
    try:
        positions = np.load(files["source_positions"]["path"], allow_pickle=False)
    except (OSError, ValueError) as error:
        _invalid("CACHE_INVALID", "source_positions", str(error))
    if positions.dtype != np.int64 or positions.shape != (document_count,):
        _invalid("CACHE_INVALID", "source_positions", "array shape or dtype is invalid")
    return manifest


def prepare_selection(
    cache_dir: Path,
    max_documents: int,
    document_seed: int,
    query_seed: int,
    development_queries: int,
    evaluation_queries: int,
) -> SelectionManifest:
    """Create or validate one immutable document and query selection."""
    max_documents = _integer(max_documents, "max_documents", 1)
    document_seed = _integer(document_seed, "document_seed", 0)
    query_seed = _integer(query_seed, "query_seed", 0)
    development_queries = _integer(development_queries, "development_queries", 1)
    evaluation_queries = _integer(evaluation_queries, "evaluation_queries", 1)
    if max_documents > EXPECTED_COLLECTION_ROWS:
        _invalid("INPUT_MISMATCH", "max_documents", "cannot exceed the provider collection count")
    selected_query_count = development_queries + evaluation_queries
    if selected_query_count > EXPECTED_QUERY_ROWS:
        _invalid("INPUT_MISMATCH", "queries", "requested splits exceed the provider query count")

    cache_dir = Path(cache_dir).resolve()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _invalid("CACHE_INVALID", str(cache_dir), str(error))
    archive_path, downloaded = _download_archive(cache_dir)
    archive_sha256 = _sha256(archive_path)
    parameters = {
        "max_documents": max_documents,
        "document_seed": document_seed,
        "query_seed": query_seed,
        "development_queries": development_queries,
        "evaluation_queries": evaluation_queries,
    }
    source_stub = {
        "url": SOURCE_URL,
        "archive_sha256": archive_sha256,
    }

    if not downloaded:
        for manifest_path in sorted(cache_dir.glob("*/selection.json")):
            try:
                candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            candidate_source = candidate.get("source", {}) if isinstance(candidate, dict) else {}
            source_matches = (
                isinstance(candidate_source, dict)
                and candidate_source.get("url") == SOURCE_URL
                and candidate_source.get("archive_sha256") == archive_sha256
            )
            if source_matches and candidate.get("parameters") == parameters:
                return _validate_cached_manifest(manifest_path)

    positions = select_positions(EXPECTED_COLLECTION_ROWS, max_documents, document_seed)
    partial_archive = archive_path if downloaded else None
    try:
        with tempfile.TemporaryDirectory(prefix=".selection-", dir=cache_dir) as temporary:
            staging = Path(temporary)
            try:
                seen_members: set[str] = set()
                with tarfile.open(archive_path, mode="r|gz") as archive:
                    for member in archive:
                        if member.name not in {COLLECTION_MEMBER, QUERY_MEMBER, QRELS_MEMBER}:
                            continue
                        if member.name in seen_members or not member.isfile():
                            _invalid(
                                "INPUT_MISMATCH",
                                member.name,
                                "archive must contain one regular member",
                            )
                        seen_members.add(member.name)
                        if member.name == COLLECTION_MEMBER:
                            document_ids, document_texts, collection_rows, collection_sha = (
                                _read_collection(archive, member, positions)
                            )
                        elif member.name == QUERY_MEMBER:
                            query_ids, query_texts, provider_query_ids, query_rows, query_sha = (
                                _read_queries(archive, member, selected_query_count, query_seed)
                            )
                        else:
                            qrel_query_ids, qrel_rows, qrel_sha = _copy_and_validate_qrels(
                                archive,
                                member,
                                staging / _OUTPUT_FILES["provider_qrels"],
                            )
                missing_members = {
                    COLLECTION_MEMBER,
                    QUERY_MEMBER,
                    QRELS_MEMBER,
                } - seen_members
                if missing_members:
                    missing_names = sorted(missing_members)
                    _invalid(
                        "INPUT_MISMATCH",
                        missing_names[0] if len(missing_names) == 1 else str(archive_path),
                        f"missing required members {missing_names}",
                    )
                if qrel_query_ids != provider_query_ids:
                    missing = sorted(provider_query_ids - qrel_query_ids)[:5]
                    unknown = sorted(qrel_query_ids - provider_query_ids)[:5]
                    _invalid(
                        "INPUT_MISMATCH",
                        QRELS_MEMBER,
                        f"query membership differs: missing {missing}, unknown {unknown}",
                    )
            except (tarfile.TarError, EOFError, OSError) as error:
                _invalid("INPUT_MISMATCH", str(archive_path), str(error))

            _write_selected_files(
                staging,
                positions,
                document_ids,
                document_texts,
                query_ids,
                query_texts,
            )
            file_hashes = {
                name: _sha256(staging / filename) for name, filename in _OUTPUT_FILES.items()
            }
            source = {
                **source_stub,
                "member_sha256": {
                    COLLECTION_MEMBER: collection_sha,
                    QUERY_MEMBER: query_sha,
                    QRELS_MEMBER: qrel_sha,
                },
                "counts": {
                    "collection_rows": collection_rows,
                    "query_rows": query_rows,
                    "qrel_rows": qrel_rows,
                },
            }
            development_rows = list(range(development_queries))
            evaluation_rows = list(range(development_queries, selected_query_count))
            selection_identity = _identity(
                source,
                parameters,
                max_documents,
                query_ids,
                development_rows,
                evaluation_rows,
                file_hashes,
            )
            selection_hash = identity_hash(selection_identity)
            destination = cache_dir / selection_hash
            files = {
                name: {
                    "path": str((destination / filename).resolve()),
                    "sha256": file_hashes[name],
                }
                for name, filename in _OUTPUT_FILES.items()
            }
            manifest: SelectionManifest = {
                "schema_version": 1,
                "selection_hash": selection_hash,
                "identity": selection_identity,
                "source": source,
                "parameters": parameters,
                "document_count": max_documents,
                "query_count": selected_query_count,
                "query_ids": query_ids,
                "development_rows": development_rows,
                "evaluation_rows": evaluation_rows,
                "files": files,
            }
            manifest_path = staging / "selection.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )

            if downloaded:
                archive_path.replace(cache_dir / ARCHIVE_NAME)
                partial_archive = None
            if destination.exists():
                existing = _validate_cached_manifest(destination / "selection.json")
                if existing != manifest:
                    _invalid("CACHE_INVALID", str(destination), "selection identity collision")
                return existing
            staging.replace(destination)
            return _validate_cached_manifest(destination / "selection.json")
    except ScalingConfigError:
        raise
    except OSError as error:
        _invalid("CACHE_INVALID", str(cache_dir), str(error))
    finally:
        if partial_archive is not None:
            partial_archive.unlink(missing_ok=True)
