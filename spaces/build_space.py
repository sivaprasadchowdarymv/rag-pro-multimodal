"""Assemble the folder that becomes the Hugging Face Space repository.

    python spaces/build_space.py                  # -> space_build/
    python spaces/build_space.py --out D:\\space   # any folder

A Docker Space needs its own README.md (with the YAML header) and Dockerfile at
its root, so those two are swapped in from spaces/. Tests and dev files are left
out. Works on Windows and Linux; the GitHub Action uses it too.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".github", ".devcontainer", "tests", "data", "space_build", "venv", ".venv",
             "__pycache__", ".pytest_cache", "benchmark"}
SKIP_FILES = {"Dockerfile", "docker-compose.yml", "README.md", ".env", "requirements-dev.txt"}


def build(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for src in ROOT.iterdir():
        if src.name in SKIP_DIRS or src.name in SKIP_FILES or src.resolve() == out.resolve():
            continue
        if src.is_dir():
            shutil.copytree(src, out / src.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pdf"))
        elif src.suffix not in {".pdf", ".zip"}:
            shutil.copy2(src, out / src.name)
    shutil.copy2(ROOT / "spaces" / "Dockerfile", out / "Dockerfile")
    shutil.copy2(ROOT / "spaces" / "README.md", out / "README.md")
    (out / ".gitattributes").write_text("*.png filter=lfs diff=lfs merge=lfs -text\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "space_build"))
    path = build(Path(ap.parse_args().out))
    files = sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
    print(f"Space folder ready: {path} ({len(files)} files)")
