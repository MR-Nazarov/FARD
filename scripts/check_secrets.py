#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""Fail if anything credential-shaped is about to enter the repository.

Credentials belong in the environment, never in a config file -- wandb in
particular uploads whatever config it is handed, so a key in a conf becomes a key
in every run's metadata.

Run over the whole tree:
    python scripts/check_secrets.py
Run over what is staged (what the pre-commit hook does):
    python scripts/check_secrets.py --staged

Exit status is 0 when clean, 1 when something matches.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# (name, pattern). Kept deliberately narrow: a rule that cries wolf gets disabled.
PATTERNS = [
    ('wandb/generic 40-hex API key', re.compile(r'\b[0-9a-f]{40}\b')),
    ('api_key assignment', re.compile(r'api_?key\s*[:=]\s*["\']?[A-Za-z0-9_\-]{16,}', re.I)),
    ('GitHub token', re.compile(r'\b(ghp_|gho_|ghs_|github_pat_)[A-Za-z0-9_]{20,}')),
    ('AWS access key id', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('Azure SAS token', re.compile(r'[?&]s(ig|v)=[A-Za-z0-9%/+]{20,}')),
    ('private key block', re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----')),
    ('password assignment', re.compile(r'password\s*[:=]\s*["\'][^"\']{6,}', re.I)),
]

# Suffixes that are data, not source, and would only produce noise.
SKIP_SUFFIXES = {'.pth', '.npy', '.nii', '.gz', '.dcm', '.png', '.jpg', '.pdf',
                 '.svg', '.xlsx', '.zip', '.wandb'}
SKIP_DIRS = {'.git', '__pycache__', 'wandb', 'models', 'results', 'node_modules'}

# A 40-hex string is also what git object ids and checksums look like, so allow
# lines that are clearly one of those.
HEX_FALSE_POSITIVE = re.compile(r'md5|sha|checksum|commit|revision|digest', re.I)


def candidate_files(staged: bool):
    if staged:
        out = subprocess.run(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
                             capture_output=True, text=True, check=True).stdout
        paths = [Path(p) for p in out.split('\n') if p]
    else:
        paths = [p for p in Path('.').rglob('*') if p.is_file()]

    for p in paths:
        if not p.is_file():
            continue
        if set(p.parts) & SKIP_DIRS:
            continue
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield p


def scan(path: Path):
    try:
        text = path.read_text(errors='ignore')
    except OSError:
        return
    for lineno, line in enumerate(text.splitlines(), 1):
        if len(line) > 4000:          # minified or generated
            continue
        for name, pattern in PATTERNS:
            if not pattern.search(line):
                continue
            if name.startswith('wandb/generic') and HEX_FALSE_POSITIVE.search(line):
                continue
            yield lineno, name, line.strip()[:110]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    ap.add_argument('--staged', action='store_true',
                    help='scan only staged changes (pre-commit)')
    args = ap.parse_args()

    findings = [(p, ln, nm, txt)
                for p in candidate_files(args.staged)
                for ln, nm, txt in scan(p)]

    if not findings:
        print('check_secrets: clean')
        return 0

    print('check_secrets: possible credentials found\n', file=sys.stderr)
    for path, lineno, name, text in findings:
        print(f'  {path}:{lineno}  [{name}]\n    {text}', file=sys.stderr)
    print('\nMove the value to an environment variable and reference it by name.\n'
          'wandb authenticates via `wandb login` / WANDB_API_KEY -- it must never\n'
          'appear in a conf file, and never inside the config passed to wandb.init.',
          file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
