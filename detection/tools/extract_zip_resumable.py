#!/usr/bin/env python3
"""
Extract selected members of a zip so that an interrupted run can be resumed (used by download_truckscenes.sbatch).

A member is skipped if a file of exactly its uncompressed size already exists, so a re-run only extracts what is
missing or was cut off when the previous job was killed (a cut-off file is always smaller). Unlike `unzip -o`, a job
that hits its time limit loses nothing. Files get the extraction time as mtime (like unzip -DD), so the /scratch
14-day cleanup counts from today.

Several worker processes each take every n-th member (each opens the zip itself), which helps on shared file systems
where creating many small files is the bottleneck.

  python3 extract_zip_resumable.py ZIP DEST [--keep LIDAR RADAR] [--workers 3]

--keep: only members man-truckscenes/<samples|sweeps>/<SENSOR>_*/<file> whose sensor starts with one of these;
omit to extract everything (metadata zips).
"""
import argparse
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor


def wanted(name, keep):
    if name.endswith('/'):
        return False
    parts = name.split('/')
    if name.startswith('/') or '..' in parts:
        return False
    if keep is None:
        return True
    return len(parts) == 4 and any(parts[2].startswith(k + '_') for k in keep)


def work(job):
    zip_path, dest, keep, shard, nshards = job
    extracted = skipped = 0
    made_dirs = set()
    with zipfile.ZipFile(zip_path) as zf:
        for i, info in enumerate(zf.infolist()):
            if i % nshards != shard or not wanted(info.filename, keep):
                continue
            out = os.path.join(dest, info.filename)
            try:
                if os.path.getsize(out) == info.file_size:
                    skipped += 1
                    continue
            except OSError:
                pass
            d = os.path.dirname(out)
            if d not in made_dirs:
                os.makedirs(d, exist_ok=True)
                made_dirs.add(d)
            with zf.open(info) as src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            extracted += 1
    return extracted, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('zip')
    p.add_argument('dest')
    p.add_argument('--keep', nargs='+', default=None)
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()

    start = time.time()
    jobs = [(a.zip, a.dest, a.keep, s, a.workers) for s in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(work, jobs))
    extracted = sum(r[0] for r in results)
    skipped = sum(r[1] for r in results)
    print('%s: extracted %d, already there %d, %.0f s'
          % (os.path.basename(a.zip), extracted, skipped, time.time() - start), flush=True)
    if extracted + skipped == 0:
        print('ERROR: no member matched --keep %s' % a.keep, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
