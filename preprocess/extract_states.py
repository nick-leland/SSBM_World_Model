"""Batch-extract game state from .slp files into per-frame parquet files.

Wraps slippi-frame-extractor/extract.py.  For each replay two parquet files
are produced — one per player perspective (p1 and p2 are symmetric training
samples).

Usage:
    python preprocess/extract_states.py
    python preprocess/extract_states.py --slp-dir data/raw --out-dir data/states -j 8
"""
import argparse
import importlib.util
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import EXTRACTOR_DIR, RAW_DIR, STATES_DIR

# Load process_replay from slippi-frame-extractor/extract.py by absolute path
# to avoid sys.path ordering issues.
def _load_extractor():
    spec = importlib.util.spec_from_file_location(
        "slippi_extract", EXTRACTOR_DIR / "extract.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.process_replay

process_replay = _load_extractor()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(slp_dir: Path) -> list[Path]:
    """Load .slp paths from manifest.json, or fall back to directory scan."""
    manifest_path = slp_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        paths = [Path(p) for p in data["slp_files"]]
        log.info("Loaded %d files from manifest", len(paths))
        return paths

    log.info("No manifest found, scanning %s for .slp files", slp_dir)
    paths = sorted(slp_dir.rglob("*.slp"))
    log.info("Found %d .slp files via directory scan", len(paths))
    return paths


def already_extracted(slp_path: Path, out_dir: Path) -> bool:
    """Return True if this replay already has parquet output in out_dir."""
    # We can't know the exact output filename without re-running (it includes
    # timestamps + UUIDs from extract.py), but we can check a done-log instead.
    done_log = out_dir / ".done_slps.txt"
    if not done_log.exists():
        return False
    done = set(done_log.read_text().splitlines())
    return slp_path.name in done


def mark_done(slp_path: Path, out_dir: Path) -> None:
    done_log = out_dir / ".done_slps.txt"
    with open(done_log, "a") as f:
        f.write(slp_path.name + "\n")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_init():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _worker_fn(args: tuple) -> tuple[str, bool, str]:
    slp_path, out_dir = args
    try:
        process_replay(str(slp_path), out_dir)
        return (str(slp_path), True, "")
    except Exception as exc:
        return (str(slp_path), False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Slippi game state to parquet files."
    )
    parser.add_argument(
        "--slp-dir", type=Path, default=RAW_DIR,
        help=f"Directory containing .slp files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=STATES_DIR,
        help=f"Output directory for parquet files (default: {STATES_DIR})",
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=None,
        help="Parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip replays already listed in .done_slps.txt (default: True)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slp_paths = load_manifest(args.slp_dir)

    if args.skip_existing:
        before = len(slp_paths)
        slp_paths = [p for p in slp_paths if not already_extracted(p, args.out_dir)]
        log.info("Skipping %d already-extracted replays", before - len(slp_paths))

    if not slp_paths:
        log.info("Nothing to process.")
        return

    n_workers = args.workers or mp.cpu_count()
    total = len(slp_paths)
    log.info("Processing %d replays with %d workers", total, n_workers)

    tasks = [(p, args.out_dir) for p in slp_paths]
    errors_path = args.out_dir / "errors.jsonl"

    succeeded = failed = 0
    t0 = time.time()

    def handle(i: int, path: str, ok: bool, err: str) -> None:
        nonlocal succeeded, failed
        if ok:
            succeeded += 1
            mark_done(Path(path), args.out_dir)
        else:
            failed += 1
            with open(errors_path, "a") as f:
                f.write(json.dumps({"path": path, "error": err}) + "\n")
            log.warning("FAILED %s: %s", path, err)

        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] %.1f files/s  ETA %dm%ds  (%d ok, %d fail)",
                i, total, rate, eta // 60, eta % 60, succeeded, failed,
            )

    if n_workers <= 1:
        for i, task in enumerate(tasks, 1):
            path, ok, err = _worker_fn(task)
            handle(i, path, ok, err)
    else:
        with mp.Pool(n_workers, initializer=_worker_init) as pool:
            for i, (path, ok, err) in enumerate(
                pool.imap_unordered(_worker_fn, tasks, chunksize=4), 1
            ):
                handle(i, path, ok, err)

    elapsed = time.time() - t0
    log.info(
        "Done. %d ok, %d failed out of %d total (%.0fs, %.1f files/s)",
        succeeded, failed, total, elapsed, total / elapsed if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()
