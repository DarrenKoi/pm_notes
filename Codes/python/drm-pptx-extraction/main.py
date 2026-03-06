"""DRM PPTX Extraction Pipeline: copy slides via COM, then extract text/images."""

import os

from extract import extract_pptx

# ============================================================
# Hardcoded paths — change these to match your files
# ============================================================
DRM_PPTX = "input/drm_file.pptx"
NON_DRM_PPTX = "input/empty.pptx"
OUTPUT_DIR = "output"
# ============================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Copy slides from DRM to non-DRM via COM
    from copy_slides import copy_slides

    copied_path = os.path.join(OUTPUT_DIR, "copied.pptx")
    print("=" * 60)
    print("Step 1: Copying slides from DRM PPTX to non-DRM PPTX")
    print("=" * 60)
    copy_slides(DRM_PPTX, NON_DRM_PPTX, copied_path)

    # Step 2: Extract text and images
    print()
    print("=" * 60)
    print("Step 2: Extracting text and images")
    print("=" * 60)
    result = extract_pptx(copied_path, OUTPUT_DIR)

    # Summary
    print()
    print("=" * 60)
    print(f"Done! {result['total_slides']} slides processed.")
    total_images = sum(len(s["images"]) for s in result["slides"])
    print(f"  - {total_images} images saved to {OUTPUT_DIR}/images/")
    print(f"  - JSON result: {OUTPUT_DIR}/extraction_result.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
