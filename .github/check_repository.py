from pathlib import Path
import subprocess

allowed_roots = {'IsaacLab', 'supermimic_g1', '.github', 'docs'}
reviewed_mimic_files = set(Path('.github/mimic-publication-files.txt').read_text().splitlines())
allowed_files = {'.gitignore', '.gitattributes', '.gitmodules', 'AGENTS.md', 'README.md', 'IsaacLab.upstream.md'}
paths = subprocess.check_output(['git', 'ls-files', '-z']).decode().split(chr(0))
for name in filter(None, paths):
    path = Path(name)
    if name not in allowed_files and name not in reviewed_mimic_files and path.parts[0] not in allowed_roots:
        raise SystemExit(f'Unreviewed publication path: {name}')
    if {'data', 'logs'} & set(path.parts) or path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.pem', '.key'}:
        raise SystemExit(f'Generated or sensitive file tracked: {name}')
    if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
        raise SystemExit(f'Large artifact tracked: {name}')
print('Repository hygiene passed')
