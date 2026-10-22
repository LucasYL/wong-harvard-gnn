"""Read-only terminal monitor for all stages of a declared transfer run."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'outputs/coadd-transfer-v1/status.json'


def emit(message):
    print(message, flush=True)


def describe(line, stage, limit):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return line
    if not isinstance(row, dict) or 'epoch' not in row:
        return line
    if row.get('event') == 'batch':
        return (f"Epoch {row['epoch']}/{limit} | batch {row['batch']}/{row['batches']} | "
                f"batch loss={row['batch_loss']:.5f} | elapsed {row['elapsed_seconds']:.1f}s")
    if 'validation_smooth_l1' in row:
        metric = f"validation SmoothL1={row['validation_smooth_l1']:.6f} (lower is better)"
    elif 'validation_ap' in row:
        metric = f"validation AP={row['validation_ap']:.4f} (higher is better)"
    else:
        return line
    seconds = row.get('seconds', row.get('epoch_seconds_including_validation', 0))
    return (f"Epoch {row['epoch']}/{limit} | {metric} | {seconds:.1f}s | "
            f"peak process memory={row['peak_rss_mib']:.0f} MiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='Show a snapshot and exit')
    parser.add_argument('--experiment', choices=['coadd', 'synthemol'], default='coadd')
    args = parser.parse_args()
    config_path = 'configs/transfer_v1.json' if args.experiment == 'coadd' else 'configs/synthemol_transfer_v1.json'
    config = json.loads((ROOT / config_path).read_text())
    name = config['name']
    status_path = ROOT / f'outputs/{name}/status.json'
    target = json.loads((ROOT / config['target_config']).read_text())
    stages = [(seed, stage) for seed in config['seeds'] for stage in ('source', 'target')]
    offsets, pending = {}, {}
    last_status, last_heartbeat = None, 0.0
    emit(f'{name} → Wong | all seeds and stages | Ctrl+C stops monitoring only')
    emit(f"Source: up to {config['source_epochs']} epochs. Target: up to {target['max_epochs']} epochs.")
    emit('Existing logs: last 3 lines each; new output: followed automatically. Heartbeat: 30s.')
    while True:
        try:
            status = json.loads(status_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            emit('Waiting for a readable queue status file...')
            if args.once:
                return
            time.sleep(2)
            continue
        marker = json.dumps(status, sort_keys=True)
        if marker != last_status or time.monotonic() - last_heartbeat >= 30:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(status['started_at_utc'])).total_seconds() / 60
            active = f"seed {status.get('current_seed')} / {status.get('current_stage')}"
            emit(f"\n[{datetime.now():%H:%M:%S}] {status['state'].upper()} | "
                 f"stages {len(status['completed_stages'])}/{len(stages)} complete | "
                 f"{active} | elapsed {elapsed:.1f} min")
            last_status, last_heartbeat = marker, time.monotonic()
        for seed, stage in stages:
            path = ROOT / f'logs/{name}-{stage}{seed}.log'
            if not path.exists():
                continue
            first = path not in offsets
            with path.open('rb') as stream:
                stream.seek(offsets.get(path, 0))
                chunk = stream.read()
                offsets[path] = stream.tell()
            lines = (pending.get(path, b'') + chunk).split(b'\n')
            pending[path] = lines.pop()
            if first:
                lines = lines[-3:]
            limit = config['source_epochs'] if stage == 'source' else target['max_epochs']
            for line in lines:
                if line.strip():
                    emit(f"[seed {seed} / {stage}] {describe(line.decode('utf-8', errors='replace'), stage, limit)}")
        if status['state'] in ('complete', 'failed'):
            emit(status.get('error', f'Queue finished. Results: reports/{args.experiment}_transfer_comparison_v1.json'))
            return
        try:
            os.kill(status['pid'], 0)
        except ProcessLookupError:
            emit('WARNING: queue PID is absent; saved RUNNING status may be stale. Inspect logs.')
            return
        except PermissionError:
            pass
        if args.once:
            return
        time.sleep(2)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        emit('\nMonitor stopped. Training was not interrupted.')
