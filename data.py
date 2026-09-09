"""Load BEIR documents, test queries, and relevance labels without changing their IDs."""

import csv
import hashlib
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    """The text arrays share row order; qrels still use the original document IDs."""

    corpus_ids: list[str]
    titles: list[str]
    texts: list[str]
    query_ids: list[str]
    queries: list[str]
    qrels: dict[str, dict[str, int]]
    fingerprint: str
    source: str
    full_document_count: int
    full_query_count: int

    @property
    def is_subset(self) -> bool:
        fewer_documents = len(self.corpus_ids) != self.full_document_count
        fewer_queries = len(self.query_ids) != self.full_query_count
        return fewer_documents or fewer_queries


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _dataset_folder(cache_dir: Path, name: str, url: str) -> Path:
    folder = cache_dir / name
    required = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]
    if all((folder / file).is_file() for file in required):
        return folder
    if folder.exists():
        raise ValueError(f"Incomplete dataset at {folder}; remove it and retry.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"{name}.zip"
    if not archive_path.exists():
        print(f"Downloading {name}...", flush=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".download", delete=False) as output:
            temporary = Path(output.name)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "bitplane-study/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    shutil.copyfileobj(response, output)
                output.flush()
                if not zipfile.is_zipfile(temporary):
                    raise ValueError(f"Download from {url} is not a ZIP archive.")
                temporary.replace(archive_path)
            finally:
                temporary.unlink(missing_ok=True)

    digest = _sha256(archive_path)
    checksum_path = archive_path.with_suffix(".zip.sha256")
    if checksum_path.exists() and checksum_path.read_text().strip() != digest:
        raise ValueError(f"Dataset archive checksum changed: {archive_path}")

    # Keep partial extraction out of the final folder, just like the download above.
    with tempfile.TemporaryDirectory(dir=cache_dir) as temporary:
        extraction_root = Path(temporary).resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (extraction_root / member.filename).resolve()
                outside_folder = not target.is_relative_to(extraction_root)
                is_symlink = stat.S_ISLNK(member.external_attr >> 16)
                if outside_folder or is_symlink:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            archive.extractall(extraction_root)
        extracted = extraction_root / name
        if not all((extracted / file).is_file() for file in required):
            raise ValueError(f"The {name} archive is missing its corpus, queries, or test qrels.")
        extracted.replace(folder)
    checksum_path.write_text(digest + "\n")
    return folder


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    ids = [str(row["_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def load_dataset(
    cache_dir: Path,
    dataset_name: str = "fiqa",
    max_documents: int = 0,
    max_queries: int = 0,
) -> Dataset:
    """Zero limits use the full corpus and test split. Positive limits take the first rows."""
    if dataset_name not in {"fiqa", "scifact"}:
        raise ValueError("dataset_name must be 'fiqa' or 'scifact'.")
    if max_documents < 0 or max_queries < 0:
        raise ValueError("Dataset limits must be nonnegative; 0 means all rows.")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    folder = _dataset_folder(Path(cache_dir), dataset_name, url)

    # BEIR keeps documents, queries, and relevance labels in separate files.
    corpus = _read_jsonl(folder / "corpus.jsonl")
    all_queries = _read_jsonl(folder / "queries.jsonl")
    labels: dict[str, dict[str, int]] = {}
    with (folder / "qrels/test.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            query_id = row["query-id"]
            document_id = row["corpus-id"]
            relevance = int(row["score"])
            query_labels = labels.setdefault(query_id, {})
            query_labels[document_id] = relevance

    # Join on IDs, not row positions. File ordering doesn't tell us which label belongs where.
    corpus_ids = {str(row["_id"]) for row in corpus}
    query_ids = {str(row["_id"]) for row in all_queries}
    missing_queries = set(labels) - query_ids
    labeled_documents = set()
    for query_labels in labels.values():
        labeled_documents.update(query_labels)
    missing_documents = labeled_documents - corpus_ids
    if missing_queries or missing_documents:
        raise ValueError(
            f"Qrels refer to missing IDs: queries={sorted(missing_queries)[:5]}, "
            f"documents={sorted(missing_documents)[:5]}"
        )
    test_queries = []
    for row in all_queries:
        query_id = str(row["_id"])
        if query_id in labels:
            test_queries.append(row)
    if not corpus or not test_queries:
        raise ValueError("The dataset needs at least one document and one labeled test query.")
    full_document_count = len(corpus)
    full_query_count = len(test_queries)
    if max_documents:
        corpus = corpus[:max_documents]
    if max_queries:
        test_queries = test_queries[:max_queries]

    # Don't pick documents from gold labels. A small prefix is only a plumbing check.
    selected_document_ids = [str(row["_id"]) for row in corpus]
    selected_query_ids = [str(row["_id"]) for row in test_queries]
    file_hashes = {}
    for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        file_hashes[name] = _sha256(folder / name)
    identity = {
        "name": dataset_name,
        "files": file_hashes,
        "document_ids": selected_document_ids,
        "query_ids": selected_query_ids,
    }
    identity_bytes = json.dumps(identity, sort_keys=True).encode()
    fingerprint = hashlib.sha256(identity_bytes).hexdigest()
    checksum_path = Path(cache_dir) / f"{dataset_name}.zip.sha256"
    source = url
    if checksum_path.exists():
        source += f" (archive sha256: {checksum_path.read_text().strip()})"

    # Keep targets outside a requested subset. Dropping them would inflate recall.
    selected_labels = {}
    for query_id in selected_query_ids:
        selected_labels[query_id] = labels[query_id]

    return Dataset(
        corpus_ids=selected_document_ids,
        titles=[str(row.get("title", "")) for row in corpus],
        texts=[str(row["text"]) for row in corpus],
        query_ids=selected_query_ids,
        queries=[str(row["text"]) for row in test_queries],
        qrels=selected_labels,
        fingerprint=fingerprint,
        source=source,
        full_document_count=full_document_count,
        full_query_count=full_query_count,
    )
