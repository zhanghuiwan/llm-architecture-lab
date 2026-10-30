#!/usr/bin/env python3
"""Run each independent chapter in a fresh working directory."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

CHAPTERS = (
    '00-preparation', '01-language-modeling', '02-attention', '03-minimal-gpt',
    '04-training', '05-tokenization', '06-modern-block', '07-gqa',
    '08-kv-cache', '09-performance', '10-long-context', '11-moe',
    '12-model-reading', '13-research',
)

def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    results = []
    for chapter in CHAPTERS:
        started = time.perf_counter()
        item = {'chapter': chapter, 'passed': False}
        try:
            script = root / chapter / 'main.py'
            if not script.is_file():
                raise RuntimeError(f'Missing entry: {script}')
            with tempfile.TemporaryDirectory(prefix='llm-chapter-') as cwd:
                cmd = [sys.executable, str(script), '--seed', str(args.seed)]
                if args.quick:
                    cmd.append('--quick')
                process = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                         timeout=180 if args.quick else 600)
            if process.returncode:
                raise RuntimeError(f'exit={process.returncode}: {process.stderr[-3000:]} {process.stdout[-1000:]}')
            report = json.loads(process.stdout)
            if not isinstance(report, dict) or not {'chapter', 'seed', 'checks', 'metrics'} <= report.keys():
                raise ValueError('Missing required report fields')
            checks = report['checks']
            if not isinstance(checks, dict) or not checks or not all(v is True for v in checks.values()):
                raise ValueError(f'Failed or invalid checks: {checks}')
            if report['seed'] != args.seed or not finite(report):
                raise ValueError('Seed mismatch or non-finite numeric result')
            item.update(passed=True, report=report)
            print(f'PASS {chapter}: {len(checks)} checks', file=sys.stderr)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            item['error'] = str(exc)
            print(f'FAIL {chapter}: {exc}', file=sys.stderr)
        item['wall_seconds'] = time.perf_counter() - started
        results.append(item)
    versions = {}
    for dependency in ('torch', 'numpy'):
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    report = {'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'dependencies': versions},
              'quick': args.quick, 'seed': args.seed, 'total': len(results),
              'passed': sum(r['passed'] for r in results), 'results': results}
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    print(output, end='')
    return 0 if report['passed'] == report['total'] else 1

if __name__ == '__main__':
    sys.exit(main())
