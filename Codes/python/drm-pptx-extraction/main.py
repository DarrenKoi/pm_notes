"""DRM PPTX Extraction Pipeline.

1. Open DRM + non-DRM PPTX in PowerPoint via COM
2. Copy-paste slides from DRM to non-DRM
3. Extract text/images directly from the open COM object (no saving to disk)
4. Close PowerPoint and flush COM memory

Supports batch processing — loops over multiple DRM files with memory cleanup.
"""

import gc
import os
import glob

import pythoncom
import win32com.client

from copy_slides import open_and_copy_slides
from extract import extract_from_com

# ============================================================
# Hardcoded paths — change these to match your files
# ============================================================
NON_DRM_PPTX = "input/empty.pptx"
OUTPUT_DIR = "output"

# Single file mode: set this to a specific file path
# DRM_PPTX = "input/drm_file.pptx"

# Batch mode: set this to a glob pattern (e.g., all pptx in input/)
DRM_PATTERN = "input/*.pptx"
# ============================================================


def process_one(drm_path: str, non_drm_path: str, output_dir: str):
    """Process a single DRM PPTX: copy slides via COM, extract in-memory, cleanup."""
    ppt_app = None
    presentation = None

    try:
        # Step 1: Copy slides
        ppt_app, presentation = open_and_copy_slides(drm_path, non_drm_path)

        # Step 2: Extract from open COM object
        result = extract_from_com(presentation, output_dir)

        total_images = sum(len(s["images"]) for s in result["slides"])
        print(f"  -> {result['total_slides']} slides, {total_images} images")

        return result

    finally:
        # Cleanup: close presentation and quit PowerPoint
        if presentation is not None:
            try:
                presentation.Close()
            except Exception:
                pass
        if ppt_app is not None:
            try:
                ppt_app.Quit()
            except Exception:
                pass

        # Flush COM references and force garbage collection
        del presentation
        del ppt_app
        gc.collect()


def main():
    pythoncom.CoInitialize()

    try:
        non_drm_path = os.path.abspath(NON_DRM_PPTX)

        # Collect DRM files (exclude the non-DRM template)
        all_pptx = glob.glob(DRM_PATTERN)
        drm_files = [f for f in all_pptx if os.path.abspath(f) != non_drm_path]
        drm_files.sort()

        if not drm_files:
            print(f"No DRM PPTX files found matching: {DRM_PATTERN}")
            return

        print(f"Found {len(drm_files)} file(s) to process")
        print()

        for idx, drm_path in enumerate(drm_files, start=1):
            filename = os.path.basename(drm_path)
            file_output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0])
            os.makedirs(file_output_dir, exist_ok=True)

            print("=" * 60)
            print(f"[{idx}/{len(drm_files)}] {filename}")
            print("=" * 60)

            try:
                process_one(drm_path, NON_DRM_PPTX, file_output_dir)
            except Exception as e:
                print(f"  [ERROR] Failed: {e}")

            print()

        print("=" * 60)
        print(f"All done! Processed {len(drm_files)} file(s).")
        print(f"Results in: {OUTPUT_DIR}/")
        print("=" * 60)

    finally:
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
