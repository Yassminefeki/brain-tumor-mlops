"""Stages ``download`` and ``extract``: fetch the dataset from Figshare and unzip it.

Usage::

    python -m brain_tumor.data.ingest download   # -> data/raw/
    python -m brain_tumor.data.ingest extract    # -> data/interim/*.mat
    python -m brain_tumor.data.ingest all
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from brain_tumor.config import load_params, project_path
from brain_tumor.utils import get_logger, save_json

log = get_logger("ingest")

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
CHUNK = 1024 * 1024


def list_figshare_files(article_id: int, timeout: int = 30) -> list[dict[str, Any]]:
    """Return the file descriptors (name, size, download_url, md5) of a Figshare article."""
    response = requests.get(FIGSHARE_API.format(article_id=article_id), timeout=timeout)
    response.raise_for_status()
    files = response.json().get("files", [])
    if not files:
        raise RuntimeError(f"Figshare article {article_id} lists no files")
    return files


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str, target: Path, expected_size: int | None = None, retries: int = 10
) -> None:
    """Stream a file to ``target`` through a temporary ``.part`` file.

    Network errors are retried, and each retry resumes from the bytes already in the
    ``.part`` file (HTTP Range request), so a dropped connection does not restart a
    200 MB download from zero.
    """
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        if expected_size and done >= expected_size:
            break
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()
                if done and r.status_code != 206:  # server ignored Range: start over
                    done = 0
                total = expected_size or (int(r.headers.get("content-length", 0)) + done) or None
                with (
                    tmp.open("ab" if done else "wb") as f,
                    tqdm(
                        total=total,
                        initial=done,
                        unit="B",
                        unit_scale=True,
                        desc=target.name,
                        leave=False,
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        f.write(chunk)
                        bar.update(len(chunk))
            break
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = min(5 * attempt, 30)
            log.warning("%s: %s; retry %d/%d in %ds", target.name, exc, attempt, retries, wait)
            time.sleep(wait)
    tmp.replace(target)


def download(cfg: dict[str, Any]) -> Path:
    raw_dir = project_path(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = list_figshare_files(cfg["data"]["figshare_article_id"])
    manifest = []
    for info in files:
        target = raw_dir / info["name"]
        expected_md5 = info.get("computed_md5") or info.get("supplied_md5")
        if target.exists() and target.stat().st_size == info["size"]:
            log.info("Already present: %s", info["name"])
        else:
            log.info("Downloading %s (%.1f MiB)", info["name"], info["size"] / 1024**2)
            download_file(info["download_url"], target, info["size"])
        actual_md5 = md5sum(target)
        if expected_md5 and actual_md5 != expected_md5:
            target.unlink()
            raise RuntimeError(f"Checksum mismatch for {info['name']}: file deleted, re-run")
        manifest.append({"name": info["name"], "size": info["size"], "md5": actual_md5})
    save_json(
        {"article_id": cfg["data"]["figshare_article_id"], "files": manifest},
        raw_dir / "manifest.json",
    )
    log.info("Downloaded %d files to %s", len(manifest), raw_dir)
    return raw_dir


def extract(cfg: dict[str, Any]) -> Path:
    raw_dir = project_path(cfg, "raw_dir")
    interim_dir = project_path(cfg, "interim_dir")
    if interim_dir.exists():
        shutil.rmtree(interim_dir)
    interim_dir.mkdir(parents=True)

    archives = sorted(raw_dir.glob("brainTumorDataPublic_*.zip"))
    if not archives:
        raise FileNotFoundError(f"No brainTumorDataPublic_*.zip in {raw_dir}; run download first")
    for archive in archives:
        log.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                name = Path(member.filename).name
                if member.is_dir() or not name.endswith(".mat"):
                    continue
                with zf.open(member) as src, (interim_dir / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    # Note: cvind.mat (the original paper's CV indices) stays in data/raw; this
    # project builds its own patient-level folds in the split stage.
    n_files = len(list(interim_dir.glob("*.mat")))
    expected = cfg["data"]["expected_images"]
    if n_files != expected:
        raise RuntimeError(f"Expected {expected} .mat files, found {n_files}")
    log.info("Extracted %d .mat files to %s", n_files, interim_dir)
    return interim_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["download", "extract", "all"])
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. paths.raw_dir=...")
    args = parser.parse_intermixed_args(argv)
    cfg = load_params(overrides=args.overrides)
    if args.step in ("download", "all"):
        download(cfg)
    if args.step in ("extract", "all"):
        extract(cfg)


if __name__ == "__main__":
    main()
