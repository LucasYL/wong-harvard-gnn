"""Read-only terminal monitor; Ctrl-C stops the monitor, not the evaluation."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/harvard-rn4220-evaluation-v1'


def show():
    p = OUT/'status.json'
    if not p.exists():
        print('No Harvard evaluation status yet.'); return False
    s = json.loads(p.read_text()); n = s.get('total', 225829); done = s.get('completed', 0)
    print(f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z} | {s['state']} | {s.get('stage','')} | {done:,}/{n:,} ({done/n:.1%})")
    if 'iteration' in s:
        print(f"  Bootstrap: {s['iteration']}/{s['iterations']}")
    if 'peak_rss_mib' in s:
        print(f"  Peak memory: {s['peak_rss_mib']:.0f} MiB")
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(s['updated_at_utc'])).total_seconds()
    print(f'  Last stage update: {age:.0f}s ago; detailed log: logs/harvard-rn4220-evaluation-v1.log')
    receipts = list((OUT/'chunks').glob('*.json'))
    if receipts and done < n:
        chunks = [json.loads(p.read_text()) for p in receipts]
        elapsed = sum(sum(c['seconds'].values()) for c in chunks)
        rows = sum(c['rows'] for c in chunks)
        print(f'  Saved chunks: {len(chunks)}/221; estimated prediction time remaining: {(n-rows)*elapsed/rows/60:.1f} min (metrics follow)')
    if s['state'] == 'running' and s.get('pid'):
        try:
            os.kill(s['pid'], 0)
        except ProcessLookupError:
            print('  ATTENTION: recorded process is absent; saved chunks are retained.'); return True
    if s.get('error'):
        print('  ERROR:', s['error'])
    return s['state'] in ('complete', 'failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--once', action='store_true'); args = p.parse_args()
    print('Harvard external evaluation: nearest neighbour + nine neural members (three ensembles). No training.\nCtrl-C exits only this monitor.')
    try:
        while True:
            if show() or args.once:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('\nMonitor stopped. Evaluation continues.')
