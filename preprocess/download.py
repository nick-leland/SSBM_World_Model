"""Download a subset of Slippi .slp replay files from HuggingFace.

Usage:
    python preprocess/download.py --n-replays 1000
    python preprocess/download.py --n-replays 1000 --output-dir data/raw
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from config import HF_DATASET_ID, RAW_DIR, RANDOM_SEED

log = logging.getLogger(__name__)


def list_slp_files(repo_id: str) -> list[str]:
    """List all .slp files in the HuggingFace dataset repository."""
    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    slp_files = [f for f in all_files if f.lower().endswith(".slp")]
    log.info("Found %d .slp files in %s", len(slp_files), repo_id)
    return slp_files


def download_subset(
    repo_id: str,
    file_paths: list[str],
    out_dir: Path,
) -> list[Path]:
    """Download files from HF and save them to out_dir.

    Returns list of local file paths that were successfully downloaded.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []

    for i, repo_path in enumerate(file_paths, 1):
        filename = Path(repo_path).name
        local_path = out_dir / filename

        if local_path.exists():
            log.info("[%d/%d] Already exists, skipping: %s", i, len(file_paths), filename)
            local_paths.append(local_path)
            continue

        log.info("[%d/%d] Downloading %s", i, len(file_paths), filename)
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                repo_type="dataset",
                local_dir=out_dir,
            )
            # hf_hub_download may nest files in subdirs; move to out_dir root
            cached_path = Path(cached)
            if cached_path.parent != out_dir:
                dest = out_dir / filename
                cached_path.rename(dest)
                local_paths.append(dest)
            else:
                local_paths.append(cached_path)
        except Exception as e:
            log.warning("Failed to download %s: %s", repo_path, e)

    return local_paths


def write_manifest(out_dir: Path, local_paths: list[Path]) -> Path:
    """Write a JSON manifest listing all downloaded .slp file paths."""
    manifest_path = out_dir / "manifest.json"
    manifest = {"slp_files": [str(p) for p in local_paths]}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to %s (%d files)", manifest_path, len(local_paths))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Slippi replay subset from HuggingFace.")
    parser.add_argument(
        "--n-replays", type=int, default=1000,
        help="Number of .slp files to download (default: 1000)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RAW_DIR,
        help=f"Directory for downloaded .slp files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for replay selection (default: 42)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True,
        help="Randomly sample replays instead of taking the first N (default: True)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Listing files in %s ...", HF_DATASET_ID)
    all_files = list_slp_files(HF_DATASET_ID)

    if not all_files:
        log.error("No .slp files found in dataset. Check dataset ID and access permissions.")
        sys.exit(1)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(all_files)

    selected = all_files[: args.n_replays]
    log.info("Downloading %d of %d available replays to %s", len(selected), len(all_files), args.output_dir)

    local_paths = download_subset(HF_DATASET_ID, selected, args.output_dir)
    write_manifest(args.output_dir, local_paths)

    log.info("Done. %d files ready in %s", len(local_paths), args.output_dir)


if __name__ == "__main__":
    main()
