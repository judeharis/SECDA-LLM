#!/usr/bin/env python3
"""Check that every change in the llama.cpp fork is bracketed by Jude: markers.

Diffs llama.cpp against the upstream commit in scripts/llama_fork_base and fails if:
  - a file from upstream is deleted, or an added file is a symlink or binary;
  - an added or changed line is outside a start/end marker pair:
        // Jude: Added [- note]   ...   // Jude: Added end
        // Jude: Edited [- note]  ...   // Jude: Edited end
    (# instead of // in CMake and .gitignore);
  - lines removed from upstream are not inside an "Edited" region.

usage: scripts/check_fork_markers.py [--worktree] [--fork DIR] [--base REV]   (default: llama.cpp at HEAD)
"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK = os.path.join(ROOT, 'llama.cpp')
BASE = open(os.path.join(ROOT, 'scripts', 'llama_fork_base')).read().strip()
argv = sys.argv[1:]
if '--fork' in argv:
    FORK = argv[argv.index('--fork') + 1]
if '--base' in argv:
    BASE = argv[argv.index('--base') + 1]
START = re.compile(r'^\s*(//|#)\s*Jude:\s*(Added|Edited)\b(?!\s+end\s*$)(\s*-.*)?\s*$')
END = re.compile(r'^\s*(//|#)\s*Jude:\s*(Added|Edited)\s+end\b(\s*-.*)?\s*$')
worktree = '--worktree' in sys.argv[1:]

def git(*a):
    return subprocess.run(['git', '-C', FORK, *a], capture_output=True, text=True, check=True).stdout

rev = [] if worktree else ['HEAD']
errors = []
for line in git('diff', '--raw', '--no-renames', BASE, *rev).splitlines():
    meta, _, paths = line.partition('\t')
    old_mode, new_mode, _, _, status = meta[1:].split()
    path = paths.split('\t')[-1]
    if status.startswith('D'):
        errors.append(f'{path}: deleted from upstream')
    if new_mode == '120000':
        errors.append(f'{path}: symlink added to the fork (make it in SECDA-LLM/setup.sh instead)')
    if old_mode != new_mode and not status.startswith(('A', 'D')):
        errors.append(f'{path}: mode change {old_mode} -> {new_mode}')
numstat = git('diff', '--numstat', BASE, *rev)
for l in numstat.splitlines():
    a, d, p = l.split('\t', 2)
    if a == '-' and d == '-':
        errors.append(f'{p}: binary change')

diff = git('diff', '-U0', '--no-renames', BASE, *rev)
cur = None
regions = {}

def load_regions(path):
    try:
        text = git('show', f'HEAD:{path}') if not worktree else open(os.path.join(FORK, path), errors='replace').read()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    inside, kind, start, out = False, None, 0, [None]  # 1-based
    for n, l in enumerate(text.split('\n'), 1):
        if END.match(l):
            if not inside:
                errors.append(f'{path}:{n}: Jude: end marker without a start')
            elif END.match(l).group(2) != kind:
                errors.append(f'{path}:{n}: Jude: {END.match(l).group(2)} end closes a {kind} region (line {start})')
            out.append(kind)
            inside, kind = False, None
        elif START.match(l):
            if inside:
                errors.append(f'{path}:{n}: Jude: start marker inside the {kind} region from line {start}')
            inside, kind, start = True, START.match(l).group(2), n
            out.append(kind)
        elif not inside and l.strip() == '':
            out.append('blank')  # whitespace-only lines are not changes
        else:
            out.append(kind if inside else None)
    if inside:
        errors.append(f'{path}:{start}: Jude: {kind} region never closed')
    return out

hunks = []  # (path, na, nn, removed non-blank line count)
for l in diff.splitlines():
    if l.startswith('+++ '):
        cur = l[6:] if l.startswith('+++ b/') else None
        if cur and cur not in regions:
            regions[cur] = load_regions(cur)
        continue
    m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', l)
    if m:
        na = int(m.group(3)); nn = int(m.group(4)) if m.group(4) is not None else 1
        hunks.append([cur, na, nn, 0])
    elif l.startswith('-') and not l.startswith('--- ') and hunks and l[1:].strip():
        hunks[-1][3] += 1

for cur, na, nn, removed in hunks:
    reg = regions.get(cur)
    if not cur or reg is None:
        continue
    for ln in range(na, na + nn):
        if ln >= len(reg) or reg[ln] is None:  # 'blank' lines pass
            errors.append(f'{cur}:{ln}: change outside a Jude: start/end region')
            break
    if removed:
        where = range(na, na + nn) if nn > 0 else [na, na + 1]
        if not any(0 < w < len(reg) and reg[w] == 'Edited' for w in where):
            errors.append(f'{cur}:{na}: upstream lines removed outside a Jude: Edited region')

files = [l.split('\t')[-1] for l in git('diff', '--name-only', BASE, *rev).splitlines()]
print(f'llama.cpp fork vs {BASE}: {len(files)} file(s) changed: ' + ', '.join(files))
if errors:
    print(f'{len(errors)} problem(s):')
    for e in errors:
        print('  ' + e)
    sys.exit(1)
print('every change is inside Jude: marker regions')
