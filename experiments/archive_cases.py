"""Package completed raw cases for Git without deleting the local working files."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def archive_cases(folder):
    schedule = json.loads((folder/'schedule.json').read_text())
    terminal = {'complete', 'failed', 'timeout', 'memory_stop', 'missing_result'}
    for number in range(len(schedule)):
        status = folder/'cases'/f'{number:04d}'/'process.json'
        if not status.exists() or json.loads(status.read_text())['status'] not in terminal:
            raise ValueError(f'case {number} is unfinished; keep collecting before archiving')
    files = sorted(path for path in (folder/'cases').rglob('*') if path.is_file())
    hashes = {path.relative_to(folder).as_posix(): digest(path) for path in files}
    target = folder/'cases.zip'
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(folder).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(hashes)
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    manifest = dict(cases=len(schedule), files=len(files), archive_sha256=digest(target),
                    source_file_sha256=hashes,
                    restore='Extract cases.zip in this directory, then run the normal analysis command.')
    (folder/'archive.json').write_text(json.dumps(manifest, indent=2)+'\n')
    ignore = folder/'.gitignore'
    existing = ignore.read_text() if ignore.exists() else ''
    if '/cases/' not in existing.splitlines():
        ignore.write_text(existing + ('\n' if existing and not existing.endswith('\n') else '') + '/cases/\n')
    print(f'Archived and verified {len(files)} files from {len(schedule)} cases in {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    archive_cases(args.folder)


if __name__ == '__main__':
    main()
