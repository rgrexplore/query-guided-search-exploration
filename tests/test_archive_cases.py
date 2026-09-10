import json
import zipfile

import pytest

from experiments.archive_cases import archive_cases


def test_archive_preserves_raw_bytes_and_keeps_local_cases(tmp_path):
    (tmp_path/'schedule.json').write_text(json.dumps([{'method':'scan'}]))
    case = tmp_path/'cases'/'0000'
    case.mkdir(parents=True)
    (case/'process.json').write_text(json.dumps({'status':'complete'}))
    raw = b'{"query":0,"recall":1.0}\n'
    (case/'queries.jsonl').write_bytes(raw)
    archive_cases(tmp_path)
    with zipfile.ZipFile(tmp_path/'cases.zip') as archive:
        assert archive.read('cases/0000/queries.jsonl') == raw
    assert (case/'queries.jsonl').read_bytes() == raw
    assert (tmp_path/'.gitignore').read_text() == '/cases/\n'


def test_unfinished_run_is_not_archived(tmp_path):
    (tmp_path/'schedule.json').write_text(json.dumps([{'method':'scan'}]))
    with pytest.raises(ValueError, match='unfinished'):
        archive_cases(tmp_path)
    assert not (tmp_path/'cases.zip').exists()
