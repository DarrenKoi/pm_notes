"""DRM PPTX Extraction Pipeline: copy slides via COM, then extract text/images."""

import argparse
import os
import sys

from extract import extract_pptx


def main():
    parser = argparse.ArgumentParser(
        description="Extract text and images from DRM-protected PPTX files"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- copy-and-extract: full pipeline ---
    full_parser = subparsers.add_parser(
        "full",
        help="Copy slides from DRM PPTX to non-DRM PPTX, then extract",
    )
    full_parser.add_argument("drm_pptx", help="Path to DRM-protected PPTX")
    full_parser.add_argument("non_drm_pptx", help="Path to empty non-DRM PPTX (template)")
    full_parser.add_argument(
        "-o", "--output-dir", default="output",
        help="Output directory (default: output/)",
    )

    # --- extract-only: just extract from a non-DRM file ---
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract text/images from an already non-DRM PPTX",
    )
    extract_parser.add_argument("pptx", help="Path to non-DRM PPTX file")
    extract_parser.add_argument(
        "-o", "--output-dir", default="output",
        help="Output directory (default: output/)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.command == "full":
        # Import here because copy_slides requires Windows + pywin32
        from copy_slides import copy_slides

        copied_path = os.path.join(args.output_dir, "copied.pptx")
        print("=" * 60)
        print("Step 1: Copying slides from DRM PPTX to non-DRM PPTX")
        print("=" * 60)
        copy_slides(args.drm_pptx, args.non_drm_pptx, copied_path)

        print()
        print("=" * 60)
        print("Step 2: Extracting text and images")
        print("=" * 60)
        result = extract_pptx(copied_path, args.output_dir)

    elif args.command == "extract":
        print("=" * 60)
        print("Extracting text and images from PPTX")
        print("=" * 60)
        result = extract_pptx(args.pptx, args.output_dir)

    print()
    print("=" * 60)
    print(f"Done! {result['total_slides']} slides processed.")
    total_images = sum(len(s["images"]) for s in result["slides"])
    print(f"  - Text blocks extracted per slide")
    print(f"  - {total_images} images saved to {args.output_dir}/images/")
    print(f"  - JSON result: {args.output_dir}/extraction_result.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
