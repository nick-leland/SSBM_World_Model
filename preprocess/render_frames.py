"""Render game frames from .slp replay files via Slippi Playback Dolphin.

For each replay, Dolphin is launched in batch mode with frame dumping enabled.
Dumped frames are renamed by their Slippi game frame number so they align
directly with the parquet files produced by extract_states.py.

Requirements (on training machine):
  - Slippi Playback Dolphin binary  (set via --dolphin-bin or $DOLPHIN_BIN)
  - SSBM NTSC 1.02 ISO              (set via --iso or $MELEE_ISO)
  - xvfb-run  (for headless rendering on Linux, optional if display is available)

Alignment note:
  Dolphin renders one frame for every frame in the .slp file, starting from
  the first frame (typically Slippi frame -123, the pre-game countdown).
  We do a fast scan of each .slp via libmelee to find the first frame number,
  then rename Dolphin's sequentially-numbered dump files to match Slippi frame
  numbers.  build_hdf5.py can then load frames directly by the frame numbers
  in the parquet.

Usage:
    python preprocess/render_frames.py \\
        --dolphin-bin /path/to/dolphin-emu \\
        --iso /path/to/GALE01.iso \\
        --slp-dir data/raw \\
        --out-dir data/frames
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from config import EXTRACTOR_DIR, FRAMES_DIR, RAW_DIR

sys.path.insert(0, str(EXTRACTOR_DIR))

log = logging.getLogger(__name__)

# Dolphin's frame dump writes files named like "frame0000001.png" into
# User/Dump/Frames/ inside the Dolphin user directory.
DUMP_SUBDIR = Path("User") / "Dump" / "Frames"
DUMP_PREFIX = "frame"


# ---------------------------------------------------------------------------
# Fast-scan: get first Slippi frame number from .slp without rendering
# ---------------------------------------------------------------------------

def get_first_slippi_frame(slp_path: Path) -> int | None:
    """Return the frame number of the very first frame in the .slp file.

    Uses libmelee's file-reader mode (no Dolphin required).  Returns None if
    the file cannot be read.
    """
    try:
        from melee import Console
        from melee.enums import Menu  # noqa: F401

        console = Console(is_dolphin=False, path=str(slp_path), allow_old_version=True)
        console.connect()

        first_frame = None
        while True:
            gs = console.step()
            if gs is None:
                break
            first_frame = gs.frame
            break  # only need the very first frame

        return first_frame
    except Exception as exc:
        log.warning("Could not read first frame from %s: %s", slp_path, exc)
        return None


# ---------------------------------------------------------------------------
# Dolphin config helpers
# ---------------------------------------------------------------------------

def write_dolphin_ini(user_dir: Path, iso_path: Path) -> None:
    """Write minimal Dolphin.ini + GFX.ini to enable frame dumping."""
    dolphin_ini_dir = user_dir / "User" / "Config"
    dolphin_ini_dir.mkdir(parents=True, exist_ok=True)

    # Dolphin.ini: enable frame dump
    dolphin_ini = dolphin_ini_dir / "Dolphin.ini"
    dolphin_ini.write_text(
        "[Core]\n"
        "GFXBackend = OGL\n"
        "[Movie]\n"
        "DumpFrames = True\n"
        "DumpFramesSilent = True\n"
    )

    # GFX.ini: dump as PNG instead of AVI, set resolution
    gfx_ini = dolphin_ini_dir / "GFX.ini"
    gfx_ini.write_text(
        "[Settings]\n"
        "DumpFormat = PNG\n"
        "InternalResolutionFrameDumps = True\n"
        "EFBScale = 2\n"         # 2× native ≈ 640×528; will be resized to 224×224 later
    )


# ---------------------------------------------------------------------------
# Core rendering logic for a single replay
# ---------------------------------------------------------------------------

def render_replay(
    slp_path: Path,
    out_dir: Path,
    dolphin_bin: Path,
    iso_path: Path,
    use_xvfb: bool = True,
    timeout: int = 600,
) -> bool:
    """Render all frames of one .slp replay and save them by Slippi frame number.

    Returns True on success.
    """
    replay_id = slp_path.stem
    frame_dir = out_dir / replay_id

    if frame_dir.exists() and any(frame_dir.iterdir()):
        log.info("Already rendered, skipping: %s", replay_id)
        return True

    frame_dir.mkdir(parents=True, exist_ok=True)

    # --- Fast-scan to find the first frame number in the .slp file ----------
    first_frame = get_first_slippi_frame(slp_path)
    if first_frame is None:
        log.warning("Could not determine first frame for %s, skipping", slp_path.name)
        return False

    log.info("Rendering %s (first Slippi frame: %d)", replay_id, first_frame)

    # --- Create a temporary Dolphin user directory ---------------------------
    with tempfile.TemporaryDirectory(prefix="dolphin_") as tmp_user_dir:
        tmp_user = Path(tmp_user_dir)
        write_dolphin_ini(tmp_user, iso_path)
        dump_dir = tmp_user / DUMP_SUBDIR

        # --- Build the Dolphin command ----------------------------------------
        cmd = [
            str(dolphin_bin),
            "--batch",                         # exit when game ends
            "--exec", str(iso_path),           # the Melee ISO
            "--slippi-input", str(slp_path),   # Slippi Playback Dolphin flag
            "--user", str(tmp_user),           # isolated user dir (avoids config conflicts)
        ]

        if use_xvfb:
            # Wrap with xvfb-run for headless rendering on Linux
            if shutil.which("xvfb-run"):
                cmd = ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x960x24"] + cmd
            else:
                log.warning("xvfb-run not found; attempting to render without virtual display")

        env = dict(os.environ)

        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.TimeoutExpired:
            log.warning("Dolphin timed out after %ds for %s", timeout, replay_id)
            return False
        except Exception as exc:
            log.warning("Failed to launch Dolphin for %s: %s", replay_id, exc)
            return False

        if result.returncode != 0:
            log.warning(
                "Dolphin exited with code %d for %s\nstderr: %s",
                result.returncode, replay_id, result.stderr[-500:],
            )

        # --- Collect and rename dumped frames --------------------------------
        dump_files = sorted(dump_dir.glob(f"{DUMP_PREFIX}*.png"))
        if not dump_files:
            log.warning("No frames dumped for %s (dump dir: %s)", replay_id, dump_dir)
            return False

        for dump_index, src in enumerate(dump_files, start=1):
            # Dolphin dump frame 1 = Slippi frame first_frame
            slippi_frame = first_frame + (dump_index - 1)
            dest = frame_dir / f"{slippi_frame:06d}.png"
            shutil.copy2(src, dest)

        log.info(
            "Rendered %d frames for %s (Slippi frames %d–%d)",
            len(dump_files), replay_id, first_frame, first_frame + len(dump_files) - 1,
        )
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_slp_paths(slp_dir: Path) -> list[Path]:
    manifest = slp_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        return [Path(p) for p in data["slp_files"]]
    return sorted(slp_dir.rglob("*.slp"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Slippi replay frames via Slippi Playback Dolphin."
    )
    parser.add_argument(
        "--dolphin-bin", type=Path,
        default=Path(os.environ.get("DOLPHIN_BIN", "dolphin-emu")),
        help="Path to Slippi Playback Dolphin binary (or set $DOLPHIN_BIN)",
    )
    parser.add_argument(
        "--iso", type=Path,
        default=Path(os.environ.get("MELEE_ISO", "")),
        help="Path to SSBM NTSC 1.02 ISO (or set $MELEE_ISO)",
    )
    parser.add_argument(
        "--slp-dir", type=Path, default=RAW_DIR,
        help=f"Directory containing .slp files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=FRAMES_DIR,
        help=f"Output directory for rendered frames (default: {FRAMES_DIR})",
    )
    parser.add_argument(
        "--no-xvfb", action="store_true",
        help="Disable xvfb-run wrapper (use when a display is already available)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Per-replay Dolphin timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.iso or not args.iso.exists():
        log.error(
            "SSBM ISO not found at %s.  Pass --iso or set $MELEE_ISO.", args.iso
        )
        sys.exit(1)

    if not shutil.which(str(args.dolphin_bin)) and not args.dolphin_bin.exists():
        log.error(
            "Dolphin binary not found at %s.  Pass --dolphin-bin or set $DOLPHIN_BIN.",
            args.dolphin_bin,
        )
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slp_paths = load_slp_paths(args.slp_dir)
    log.info("Rendering %d replays", len(slp_paths))

    ok = failed = 0
    for i, slp_path in enumerate(slp_paths, 1):
        success = render_replay(
            slp_path=slp_path,
            out_dir=args.out_dir,
            dolphin_bin=args.dolphin_bin,
            iso_path=args.iso,
            use_xvfb=not args.no_xvfb,
            timeout=args.timeout,
        )
        if success:
            ok += 1
        else:
            failed += 1
        log.info("[%d/%d] ok=%d failed=%d", i, len(slp_paths), ok, failed)

    log.info("Done. %d ok, %d failed", ok, failed)


if __name__ == "__main__":
    main()
