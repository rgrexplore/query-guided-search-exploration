import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

import msmarco
from msmarco import prepare_selection, select_positions
from scaling_config import ScalingConfigError


def _member_bytes(lines: list[str]) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def _write_archive(path: Path, members: dict[str, bytes]) -> bytes:
    archive_stream = io.BytesIO()
    with tarfile.open(fileobj=archive_stream, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    archive_bytes = archive_stream.getvalue()
    path.write_bytes(archive_bytes)
    return archive_bytes


def _fixture_members(
    *,
    collection: list[str] | None = None,
    queries: list[str] | None = None,
    qrels: list[str] | None = None,
) -> dict[str, bytes]:
    collection = collection or [
        "d0\tzero",
        "d1\tone\tkeeps this tab",
        "d2\ttwo",
        "d3\tthree",
        "d4\tfour",
        "d5\tfive",
        "d6\tsix",
    ]
    queries = queries or [
        "q9\tnine",
        "q1\tone",
        "q7\tseven",
        "q3\tthree",
        "q5\tfive\tkeeps this tab",
    ]
    qrels = qrels or [
        "q1\tQ0\td1\t1",
        "q3\tQ0\td3\t1",
        "q5\tQ0\td5\t1",
        "q7\tQ0\td0\t1",
        "q9\tQ0\td2\t1",
    ]
    return {
        msmarco.COLLECTION_MEMBER: _member_bytes(collection),
        msmarco.QUERY_MEMBER: _member_bytes(queries),
        msmarco.QRELS_MEMBER: _member_bytes(qrels),
    }


def _install_fixture(
    monkeypatch: pytest.MonkeyPatch,
    cache_dir: Path,
    members: dict[str, bytes],
) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_bytes = _write_archive(cache_dir / msmarco.ARCHIVE_NAME, members)
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_ARCHIVE_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_COLLECTION_ROWS",
        len(members[msmarco.COLLECTION_MEMBER].splitlines()),
    )
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_QUERY_ROWS",
        len(members[msmarco.QUERY_MEMBER].splitlines()),
    )
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_QREL_ROWS",
        len(members[msmarco.QRELS_MEMBER].splitlines()),
    )
    return archive_bytes


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_select_positions_uses_unique_ordered_pcg64_positions():
    positions = select_positions(total_rows=7, count=5, seed=42)

    expected = np.random.Generator(np.random.PCG64(42)).permutation(7)[:5].astype(np.int64)
    assert positions.dtype == np.int64
    assert np.array_equal(positions, expected)
    assert len(set(positions.tolist())) == 5
    assert positions.tolist() != sorted(positions.tolist())


@pytest.mark.parametrize(
    "total_rows,count,seed",
    [(0, 0, 0), (3, 0, 0), (3, 4, 0), (3, 2, -1), (True, 1, 0)],
)
def test_select_positions_rejects_invalid_ranges(total_rows, count, seed):
    with pytest.raises(ScalingConfigError) as error:
        select_positions(total_rows=total_rows, count=count, seed=seed)

    assert error.value.code == "INPUT_MISMATCH"


def test_prepare_selection_preserves_ordered_id_text_mapping_and_nested_pools(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    members = _fixture_members()
    _install_fixture(monkeypatch, cache_dir, members)

    large = prepare_selection(cache_dir, 5, 42, 43, 2, 2)
    small = prepare_selection(cache_dir, 3, 42, 43, 2, 2)

    large_positions = np.load(large["files"]["source_positions"]["path"], allow_pickle=False)
    small_positions = np.load(small["files"]["source_positions"]["path"], allow_pickle=False)
    large_documents = _read_jsonl(large["files"]["documents"]["path"])
    small_documents = _read_jsonl(small["files"]["documents"]["path"])
    large_ids = _read_jsonl(large["files"]["document_ids"]["path"])

    assert np.array_equal(large_positions, np.array([3, 2, 6, 4, 1], dtype=np.int64))
    assert np.array_equal(small_positions, large_positions[:3])
    assert small_documents == large_documents[:3]
    assert [row["document_id"] for row in large_documents] == [
        row["document_id"] for row in large_ids
    ]
    assert large_documents[-1] == {
        "document_id": "d1",
        "text": "one\tkeeps this tab",
    }
    assert len({row["document_id"] for row in large_documents}) == 5


def test_prepare_selection_creates_disjoint_seeded_query_splits_and_preserves_qrels(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    members = _fixture_members()
    _install_fixture(monkeypatch, cache_dir, members)

    manifest = prepare_selection(cache_dir, 4, 7, 43, 2, 2)
    queries = _read_jsonl(manifest["files"]["queries"]["path"])
    sorted_ids = ["q1", "q3", "q5", "q7", "q9"]
    order = np.random.Generator(np.random.PCG64(43)).permutation(len(sorted_ids))[:4]
    expected_ids = [sorted_ids[index] for index in order]

    assert manifest["query_ids"] == expected_ids
    assert [row["query_id"] for row in queries] == expected_ids
    assert manifest["development_rows"] == [0, 1]
    assert manifest["evaluation_rows"] == [2, 3]
    assert set(manifest["development_rows"]).isdisjoint(manifest["evaluation_rows"])
    query_by_id = {row["query_id"]: row["text"] for row in queries}
    assert query_by_id["q5"] == "five\tkeeps this tab"
    assert Path(manifest["files"]["provider_qrels"]["path"]).read_bytes() == members[
        msmarco.QRELS_MEMBER
    ]


def test_manifest_records_source_counts_hashes_and_complete_file_hashes(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    members = _fixture_members()
    archive_bytes = _install_fixture(monkeypatch, cache_dir, members)

    manifest = prepare_selection(cache_dir, 4, 7, 43, 2, 2)

    assert manifest["schema_version"] == 1
    assert manifest["source"] == {
        "url": msmarco.SOURCE_URL,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "member_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in members.items()
        },
        "counts": {"collection_rows": 7, "query_rows": 5, "qrel_rows": 5},
    }
    assert manifest["document_count"] == 4
    assert manifest["query_count"] == 4
    assert len(manifest["selection_hash"]) == 64
    for file_record in manifest["files"].values():
        path = Path(file_record["path"])
        assert path.is_absolute()
        assert path.is_file()
        assert file_record["sha256"] == _sha256(path)
    selection_path = cache_dir / manifest["selection_hash"] / "selection.json"
    assert selection_path.is_file()
    assert json.loads(selection_path.read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize(
    "members,expected_field",
    [
        (
            {
                msmarco.COLLECTION_MEMBER: _member_bytes(["d0\tvalid"]),
                msmarco.QUERY_MEMBER: _member_bytes(["q0\tvalid", "q1\tvalid"]),
            },
            msmarco.QRELS_MEMBER,
        ),
        (_fixture_members(collection=["d0 has no tab"]), msmarco.COLLECTION_MEMBER),
        (_fixture_members(collection=["\tmissing id"]), msmarco.COLLECTION_MEMBER),
        (
            _fixture_members(collection=["same\tfirst", "same\tsecond"]),
            msmarco.COLLECTION_MEMBER,
        ),
    ],
)
def test_prepare_selection_rejects_missing_malformed_or_duplicate_selected_rows(
    tmp_path, monkeypatch, members, expected_field
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    archive_bytes = _write_archive(cache_dir / msmarco.ARCHIVE_NAME, members)
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_ARCHIVE_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_COLLECTION_ROWS",
        len(members.get(msmarco.COLLECTION_MEMBER, b"").splitlines()),
    )
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_QUERY_ROWS",
        len(members.get(msmarco.QUERY_MEMBER, b"").splitlines()),
    )
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_QREL_ROWS",
        len(members.get(msmarco.QRELS_MEMBER, b"").splitlines()),
    )

    with pytest.raises(ScalingConfigError) as error:
        prepare_selection(cache_dir, max(1, msmarco.EXPECTED_COLLECTION_ROWS), 0, 0, 1, 1)

    assert error.value.code == "INPUT_MISMATCH"
    assert error.value.field == expected_field
    assert not list(cache_dir.glob("*/selection.json"))


def test_prepare_selection_rejects_provider_count_and_qrel_membership_changes(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    members = _fixture_members(qrels=["unknown\tQ0\td0\t1"])
    _install_fixture(monkeypatch, cache_dir, members)

    with pytest.raises(ScalingConfigError) as error:
        prepare_selection(cache_dir, 4, 7, 43, 2, 2)

    assert error.value.code == "INPUT_MISMATCH"

    members = _fixture_members()
    archive_bytes = _write_archive(cache_dir / msmarco.ARCHIVE_NAME, members)
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_ARCHIVE_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(msmarco, "EXPECTED_COLLECTION_ROWS", 8)
    monkeypatch.setattr(msmarco, "EXPECTED_QUERY_ROWS", 5)
    monkeypatch.setattr(msmarco, "EXPECTED_QREL_ROWS", 5)

    with pytest.raises(ScalingConfigError) as error:
        prepare_selection(cache_dir, 4, 7, 43, 2, 2)

    assert error.value.code == "INPUT_MISMATCH"


def test_prepare_selection_rejects_a_changed_archive(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_archive(cache_dir / msmarco.ARCHIVE_NAME, _fixture_members())
    monkeypatch.setattr(msmarco, "EXPECTED_ARCHIVE_SHA256", "0" * 64)

    with pytest.raises(ScalingConfigError) as error:
        prepare_selection(cache_dir, 4, 7, 43, 2, 2)

    assert error.value.code == "INPUT_MISMATCH"


def test_download_uses_partial_file_and_publishes_only_valid_archive(tmp_path, monkeypatch):
    members = _fixture_members()
    archive_bytes = _write_archive(tmp_path / "source.tar.gz", members)
    (tmp_path / "source.tar.gz").unlink()
    monkeypatch.setattr(
        msmarco,
        "EXPECTED_ARCHIVE_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(msmarco, "EXPECTED_COLLECTION_ROWS", 7)
    monkeypatch.setattr(msmarco, "EXPECTED_QUERY_ROWS", 5)
    monkeypatch.setattr(msmarco, "EXPECTED_QREL_ROWS", 5)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

    requested_urls = []

    def open_fixture(request, timeout):
        requested_urls.append((request.full_url, timeout))
        return Response(archive_bytes)

    monkeypatch.setattr(msmarco.urllib.request, "urlopen", open_fixture)

    prepare_selection(tmp_path / "cache", 4, 7, 43, 2, 2)

    assert requested_urls == [(msmarco.SOURCE_URL, msmarco.DOWNLOAD_TIMEOUT_SECONDS)]
    assert (tmp_path / "cache" / msmarco.ARCHIVE_NAME).read_bytes() == archive_bytes
    assert not (tmp_path / "cache" / f"{msmarco.ARCHIVE_NAME}.partial").exists()


def test_download_failure_has_named_error_and_leaves_no_archive(tmp_path, monkeypatch):
    def fail_download(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr(msmarco.urllib.request, "urlopen", fail_download)

    with pytest.raises(ScalingConfigError) as error:
        prepare_selection(tmp_path / "cache", 1, 0, 0, 1, 1)

    assert error.value.code == "DOWNLOAD_FAILED"
    assert not (tmp_path / "cache" / msmarco.ARCHIVE_NAME).exists()
    assert not (tmp_path / "cache" / f"{msmarco.ARCHIVE_NAME}.partial").exists()


def test_cached_selection_detects_file_corruption(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    _install_fixture(monkeypatch, cache_dir, _fixture_members())
    manifest = prepare_selection(cache_dir, 4, 7, 43, 2, 2)
    Path(manifest["files"]["documents"]["path"]).write_text("changed\n", encoding="utf-8")

    with pytest.raises(ScalingConfigError) as error:
        prepare_selection(cache_dir, 4, 7, 43, 2, 2)

    assert error.value.code == "CACHE_INVALID"
