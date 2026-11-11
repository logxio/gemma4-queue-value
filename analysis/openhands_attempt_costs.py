"""Extract public OpenHands attempt metadata without downloading dialogue text.

Pin the upstream commit and project Parquet's ``trajectory...role`` leaf.  The
compatibility column assistant_calls counts *recorded assistant messages*, not
billed API calls or hidden retries. model_EmptyPatch is an exit-status indicator;
patch_empty separately measures whitespace-empty or null model_patch values.
Only the resulting small tables are cached; text is never cached.

Run: python analysis/openhands_attempt_costs.py --smoke
     python analysis/openhands_attempt_costs.py
Dependencies: duckdb, pyarrow. No model requests or accelerators are used.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/results/openhands_attempt_costs.csv'
SUMMARY = OUT.with_name('openhands_attempt_costs_summary.json')
CACHE = OUT.parent / 'openhands_extract_cache'
DATASET = 'nebius/SWE-rebench-openhands-trajectories'
COMMIT = '35455389ab51bf5e2306bfd436ef72d0f98bf882'
SOURCE_SIZE = 2079503354
SOURCE_SHA256 = '14048dd1fcd22ce094b6e85f8a38f223a9ef1327031aaaad052804870212efa1'
URL = f'https://huggingface.co/datasets/{DATASET}/resolve/{COMMIT}/trajectories.parquet'
HEADER = ['trajectory_id', 'instance_id', 'repo', 'resolved', 'assistant_calls',
          'messages', 'exit_status', 'model_EmptyPatch']
# Reading only this nested leaf excludes message content and tool-call payloads.
COLUMNS = ['trajectory_id', 'instance_id', 'repo', 'resolved', 'exit_status',
           'trajectory.list.element.role']
MAX_CHUNK = 8 * 1024 * 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


class RangeReader(io.RawIOBase):
    """Bounded HTTP ranges; raw bytes stay in memory only.

    A failed request retries only its <=8 MiB block, never a global download.
    Completed row-group projections are durable, hashed resume checkpoints.
    Size and Content-Range are checked before computing each block's SHA-256.
    """
    def __init__(self):
        self.position = 0
        self.blocks = {}
        self.receipts = []
        self.downloaded = 0
        self.retries = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else SOURCE_SIZE) + offset
        return self.position

    def fetch(self, start, stop):
        key = (start, stop)
        if key in self.blocks:
            return self.blocks[key]
        expected = f'bytes {start}-{stop-1}/{SOURCE_SIZE}'
        for attempt in range(6):
            try:
                # Unique range query prevents intermediary caches reusing another range.
                req = urllib.request.Request(f'{URL}?range={start}-{stop-1}',
                    headers={'Range': f'bytes={start}-{stop-1}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise IOError(f'Range mismatch: {response.status}, {response.headers.get("Content-Range")!r}; expected {expected}')
                    data = response.read(stop-start+1)
                if len(data) != stop-start:
                    raise IOError(f'Short range {start}:{stop}: {len(data)}')
                sha = hashlib.sha256(data).hexdigest()
                self.receipts.append({'start': start, 'stop_exclusive': stop,
                                      'bytes': len(data), 'sha256': sha})
                self.downloaded += len(data)
                self.blocks[key] = data
                return data
            except Exception:
                if attempt == 5:
                    raise
                self.retries += 1
                time.sleep(min(2 ** attempt, 16))

    def read(self, size=-1):
        if size < 0:
            size = SOURCE_SIZE-self.position
        stop = min(self.position + size, SOURCE_SIZE)
        chunks = []
        while self.position < stop:
            end = min(self.position+MAX_CHUNK, stop)
            chunks.append(self.fetch(self.position, end))
            self.position = end
        return b''.join(chunks)

    def readinto(self, target):
        data = self.read(len(target))
        target[:len(data)] = data
        return len(data)

    def release(self):
        self.blocks.clear()


def validate_schema(parquet):
    paths = [parquet.metadata.row_group(0).column(j).path_in_schema
             for j in range(parquet.metadata.num_columns)]
    if set(COLUMNS) - set(paths):
        raise ValueError(f'Missing required Parquet leaves: {set(COLUMNS)-set(paths)}; paths={paths}')
    if parquet.metadata.num_rows != 67074:
        raise ValueError(f'Pinned source row count changed: {parquet.metadata.num_rows}')
    schema = parquet.schema_arrow
    for name in ('trajectory_id', 'instance_id', 'repo', 'exit_status'):
        if not pa.types.is_string(schema.field(name).type):
            raise ValueError(f'Unexpected {name} type: {schema.field(name).type}')
    if not pa.types.is_integer(schema.field('resolved').type):
        raise ValueError('resolved must be integer')
    return paths


def extract(con, table):
    con.register('projected', table)
    try:
        return con.execute('''SELECT trajectory_id, instance_id, repo, resolved,
            len(list_filter(trajectory, lambda msg: msg.role = 'assistant'))::BIGINT,
            len(trajectory)::BIGINT, exit_status,
            coalesce(exit_status = 'model_EmptyPatch', false)
            FROM projected''').fetchall()
    finally:
        con.unregister('projected')


def write_rows(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(HEADER)
        writer.writerows(rows)
    temp.replace(path)


def read_checkpoint(path, receipt_path, nrows):
    if not path.exists() or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('source_commit') != COMMIT or receipt.get('sha256') != digest(path):
        raise ValueError(f'Invalid resume checkpoint: {path}')
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError(f'Checkpoint schema mismatch: {path}')
        rows = list(reader)
    if len(rows) != nrows:
        raise ValueError(f'Checkpoint row count mismatch: {path}')
    return rows


def add_patch_flags(smoke=False):
    """Append text-derived flags, retaining no patch text on disk.

    This independent second phase resumes each hashed row-group Boolean table.
    All network reads remain <=8MiB with local retries, even if the sum of the
    patch-column transfers exceeds 100MiB.
    """
    summary = json.loads(SUMMARY.read_text())
    reader = RangeReader()
    parquet = pq.ParquetFile(reader, pre_buffer=False, buffer_size=0)
    validate_schema(parquet)
    flags = {}
    receipts = []
    patch_header = ['trajectory_id', 'patch_empty', 'patch_is_null']
    compressed_bytes = 0
    for i in (range(1) if smoke else range(parquet.num_row_groups)):
        group = parquet.metadata.row_group(i)
        for j in range(group.num_columns):
            column = group.column(j)
            if column.path_in_schema == 'model_patch':
                compressed_bytes += column.total_compressed_size
        path = CACHE / f'patch_flags_{i:04}.csv'
        if smoke:
            path = CACHE / 'patch_smoke_10.csv'
        receipt_path = path.with_suffix('.json')
        if not smoke and path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['source_commit'] != COMMIT or receipt['sha256'] != digest(path):
                raise ValueError(f'Invalid patch checkpoint: {path}')
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != group.num_rows or list(rows[0]) != patch_header:
                raise ValueError(f'Patch checkpoint schema/size mismatch: {path}')
        else:
            # Parquet only transfers ID and patch column chunks. It never reads
            # the trajectory content leaf; patch strings die with this table.
            table = parquet.read_row_group(i, columns=['trajectory_id', 'model_patch'], use_threads=False)
            if smoke:
                table = table.slice(0,10)
            rows = []
            for record in table.to_pylist():
                patch = record['model_patch']
                if patch is not None and not isinstance(patch, str):
                    raise ValueError('Unexpected model_patch type')
                rows.append({'trajectory_id': record['trajectory_id'],
                             'patch_empty': patch is None or patch.strip() == '',
                             'patch_is_null': patch is None})
            temp = path.with_suffix('.tmp')
            with temp.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, patch_header, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            temp.replace(path)
            receipt = {'source_commit': COMMIT, 'row_group': i, 'rows':len(rows),
                       'sha256':digest(path), 'projection':['trajectory_id','model_patch'],
                       'download_ranges':reader.receipts[:], 'network_retries':reader.retries,
                       'cached_content':'trajectory_id and Boolean flags only; no patch text'}
            write_json(receipt_path, receipt)
        for row in rows:
            if row['trajectory_id'] in flags:
                raise ValueError('Duplicate trajectory in patch flags')
            flags[row['trajectory_id']] = [str(row[k]).lower() in ('true','1')
                                           for k in ('patch_empty','patch_is_null')]
        receipts.append({'file':path.name, 'sha256':digest(path), 'rows':len(rows)})
        print(f'patch_group={i+1}/{parquet.num_row_groups} flags={len(flags)} downloaded={reader.downloaded}',flush=True)
        del rows
        reader.release()
        reader.receipts.clear()
    if smoke:
        with path.open(newline='') as handle:
            check=list(csv.DictReader(handle))
        if len(check) != 10 or any(r['patch_empty'] not in ('True','False') for r in check):
            raise ValueError('Patch smoke read-back mismatch')
        print('PATCH SMOKE PASS: 10 records -> text-derived Booleans -> CSV -> read-back; no patch text persisted',flush=True)
        return
    with OUT.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(flags) != len(rows) or set(flags) != {r['trajectory_id'] for r in rows}:
        raise ValueError('Patch flag/attempt ID mismatch')
    extended_header = HEADER + ['patch_empty','patch_is_null']
    temp = OUT.with_suffix('.tmp')
    with temp.open('w',newline='') as handle:
        writer=csv.DictWriter(handle, extended_header, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            row['patch_empty'],row['patch_is_null']=flags[row['trajectory_id']]
            writer.writerow(row)
    temp.replace(OUT)
    summary.update({'patch_empty_definition':"model_patch is null OR Python str.strip(model_patch) == ''; nulls separately identified by patch_is_null.",
                    'patch_is_null_definition':'model_patch is null in upstream Parquet.',
                    'patch_empty_count':sum(v[0] for v in flags.values()),
                    'patch_is_null_count':sum(v[1] for v in flags.values()),
                    'patch_column_compressed_bytes':compressed_bytes,
                    'patch_flag_checkpoints':receipts,
                    'patch_phase_this_run_downloaded_bytes':reader.downloaded,
                    'patch_phase_this_run_network_retries':reader.retries,
                    'csv_sha256':digest(OUT),'csv_bytes':OUT.stat().st_size})
    write_json(SUMMARY,summary)
    print(json.dumps({k:summary[k] for k in ('patch_empty_count','patch_is_null_count','patch_column_compressed_bytes','patch_phase_this_run_downloaded_bytes')},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Project ten records and verify CSV round trip; do not publish full CSV')
    parser.add_argument('--patch-empty-only', action='store_true', help='Append actual patch-empty flags to an existing complete base extraction')
    parser.add_argument('--skip-patch-empty', action='store_true', help='Extract only base columns; default also appends actual patch-empty flags')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.patch_empty_only:
        add_patch_flags(smoke=args.smoke)
        return 0
    start_time = time.monotonic()
    reader = RangeReader()
    con = duckdb.connect(config={'memory_limit': '2GB', 'threads': '2'})
    # No httpfs is needed: HTTP's bounded request/retry behavior is explicit here.
    parquet = pq.ParquetFile(reader, pre_buffer=False, buffer_size=0)
    schema_paths = validate_schema(parquet)
    all_rows = []
    checkpoint_receipts = []
    groups = range(1) if args.smoke else range(parquet.num_row_groups)
    for i in groups:
        path = CACHE / f'row_group_{i:04}.csv'
        receipt_path = path.with_suffix('.json')
        nrows = parquet.metadata.row_group(i).num_rows
        rows = None if args.smoke else read_checkpoint(path, receipt_path, nrows)
        if rows is None:
            table = parquet.read_row_group(i, columns=COLUMNS, use_threads=False)
            if args.smoke:
                table = table.slice(0, 10)
            if table.column_names != ['trajectory_id','instance_id','repo','resolved','exit_status','trajectory']:
                # Arrow preserves source order rather than caller order.
                if set(table.column_names) != {'trajectory_id','instance_id','repo','resolved','exit_status','trajectory'}:
                    raise ValueError(f'Projection mismatch: {table.schema}')
            # This assertion ensures no dialogue or patch text reaches DuckDB/checkpoints.
            nested = table.schema.field('trajectory').type.value_type
            if nested.names != ['role']:
                raise ValueError(f'Unexpected nested fields: {nested}')
            rows = extract(con, table)
            if len(rows) != (10 if args.smoke else nrows):
                raise ValueError('Projection lost records')
            if args.smoke:
                independent = [(sum(m['role'] == 'assistant' for m in r['trajectory']),
                                len(r['trajectory'])) for r in table.to_pylist()]
                if independent != [(r[4],r[5]) for r in rows]:
                    raise ValueError('DuckDB/Python message-count disagreement')
                path, receipt_path = CACHE / 'smoke_10.csv', CACHE / 'smoke_10.json'
            write_rows(path, rows)
            receipt = {'source_commit': COMMIT, 'row_group': i, 'rows':len(rows),
                       'sha256': digest(path), 'projection': COLUMNS,
                       'download_ranges': reader.receipts[:], 'network_retries': reader.retries}
            write_json(receipt_path, receipt)
        checkpoint_receipts.append({'file': path.name, 'sha256':digest(path), 'rows':len(rows)})
        all_rows.extend(rows)
        reader.release()
        reader.receipts.clear()
        print(f'row_group={i+1}/{parquet.num_row_groups} rows={len(rows)} cumulative={len(all_rows)} downloaded={reader.downloaded}', flush=True)
    if args.smoke:
        with path.open(newline='') as handle:
            got = list(csv.DictReader(handle))
        assert len(got) == 10 and all(int(r['assistant_calls']) <= int(r['messages']) for r in got)
        print('SMOKE PASS: 10-record column projection -> DuckDB counts -> CSV -> read-back', flush=True)
        return 0
    # Dict checkpoints and fresh SQL tuples share this canonical representation.
    records = []
    for row in all_rows:
        if not isinstance(row, dict):
            row = dict(zip(HEADER, row))
        records.append([row['trajectory_id'],row['instance_id'],row['repo'],int(row['resolved']),
                        int(row['assistant_calls']),int(row['messages']),row['exit_status'],
                        str(row['model_EmptyPatch']).lower() in ('true','1')])
    records.sort(key=lambda r:(r[1],r[0]))
    if len(records) != 67074 or len({r[0] for r in records}) != len(records):
        raise ValueError('Missing or duplicate trajectory IDs')
    if any(r[3] not in (0,1) or not 0 <= r[4] <= r[5] for r in records):
        raise ValueError('Invalid result or message count')
    write_rows(OUT, records)
    counts = collections.Counter(r[6] for r in records)
    summary = {
        'dataset': DATASET, 'source_url': URL, 'source_commit': COMMIT,
        'source_file_bytes': SOURCE_SIZE, 'upstream_lfs_sha256': SOURCE_SHA256,
        'upstream_hash_verified': False,
        'upstream_hash_note': 'Full 2.08GB file deliberately not downloaded; identity pinned by commit and upstream LFS pointer. Each transferred range validated by length/Content-Range and SHA-256 recorded.',
        'projection': COLUMNS, 'source_schema_leaves': schema_paths,
        'attempts':len(records), 'issues':len({r[1] for r in records}),
        'repositories':len({r[2] for r in records}), 'resolved':sum(r[3] for r in records),
        'assistant_calls_total':sum(r[4] for r in records),
        'messages_total':sum(r[5] for r in records),
        'assistant_calls_definition':'Number of recorded assistant-role messages; not billed API calls and excludes unrecorded retries.',
        'messages_definition':'Number of all recorded trajectory messages across all roles.',
        'model_EmptyPatch_definition':"Boolean indicator exit_status == 'model_EmptyPatch'; not derived from model_patch text and does not claim to identify every empty patch.",
        'exit_status_counts':dict(counts), 'model_EmptyPatch_count':sum(r[7] for r in records),
        'csv_sha256':digest(OUT), 'csv_bytes':OUT.stat().st_size,
        'checkpoints':checkpoint_receipts, 'this_run_downloaded_bytes':reader.downloaded,
        'this_run_network_retries':reader.retries, 'elapsed_seconds':time.monotonic()-start_time,
        'duckdb_version':duckdb.__version__, 'pyarrow_version':pa.__version__,
        'memory_limit':'2GB', 'threads':2,
        'sampling_note':'All released attempts; source collection is not a randomized attempt allocation.',
        'evaluation_use':'Descriptive extraction only; no final-evaluation outcome used to tune a policy.'}
    write_json(SUMMARY,summary)
    print(json.dumps({k: summary[k] for k in ('attempts','issues','repositories','resolved','assistant_calls_total','messages_total','exit_status_counts','this_run_downloaded_bytes','elapsed_seconds')},indent=2))
    if not args.skip_patch_empty:
        add_patch_flags()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, TimeoutError) as exc:
        print(f'Range extraction interrupted; rerun resumes hashed row-group projections: {exc}',file=sys.stderr)
        raise SystemExit(75)
