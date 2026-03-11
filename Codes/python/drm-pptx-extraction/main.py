"""Export slide screenshots from DRM-protected PPTX files.

Each input presentation is opened in Microsoft PowerPoint via COM automation,
and every slide is exported as a PNG into `output/<presentation_name>/`.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pythoncom

from export_slides import export_slides_to_png

DEFAULT_INPUT_PATTERN = "input/*.pptx"
DEFAULT_OUTPUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture each slide of DRM PPTX files as PNG images."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="PPTX file paths or glob patterns. Defaults to input/*.pptx",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def resolve_input_files(inputs: list[str]) -> list[Path]:
    """Resolve file paths and glob patterns into a sorted PPTX file list."""
    candidates = inputs or [DEFAULT_INPUT_PATTERN]
    resolved_files: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        candidate_path = Path(candidate)

        if candidate_path.is_file():
            resolved = candidate_path.resolve()
            if resolved.suffix.lower() == ".pptx" and resolved not in seen:
                resolved_files.append(resolved)
                seen.add(resolved)
            continue

        for match in sorted(glob.glob(candidate)):
            match_path = Path(match)
            if not match_path.is_file() or match_path.suffix.lower() != ".pptx":
                continue

            resolved = match_path.resolve()
            if resolved in seen:
                continue

            resolved_files.append(resolved)
            seen.add(resolved)

    return sorted(resolved_files)


def main() -> int:
    """Capture slides from each input PPTX into PNG files."""
    args = parse_args()
    pptx_files = resolve_input_files(args.inputs)

    if not pptx_files:
        print(f"No PPTX files found. Inputs: {args.inputs or [DEFAULT_INPUT_PATTERN]}")
        return 1

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pythoncom.CoInitialize()

    try:
        total_slides = 0

        print(f"Found {len(pptx_files)} file(s) to process")
        print()

        for index, pptx_path in enumerate(pptx_files, start=1):
            deck_output_dir = output_root / pptx_path.stem

            print("=" * 60)
            print(f"[{index}/{len(pptx_files)}] {pptx_path.name}")
            print(f"Output: {deck_output_dir}")
            print("=" * 60)

            try:
                exported_count = export_slides_to_png(pptx_path, deck_output_dir)
                total_slides += exported_count
                print(f"  -> Exported {exported_count} slide(s)")
            except Exception as exc:
                print(f"  [ERROR] Failed: {exc}")

            print()

        print("=" * 60)
        print(f"All done! Processed {len(pptx_files)} file(s).")
        print(f"Exported {total_slides} slide(s) into: {output_root}")
        print("=" * 60)
        return 0
    finally:
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    raise SystemExit(main())
