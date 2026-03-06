"""Test script to verify COM copy-paste and in-memory extraction feasibility.

Usage:
    python test_copy_paste.py
"""

import os
import sys
import time
import traceback

# ============================================================
# Hardcoded paths — change these to match your files
# ============================================================
DRM_PPTX = "input/drm_file.pptx"
NON_DRM_PPTX = "input/empty.pptx"
# ============================================================


def test():
    drm_path = os.path.abspath(DRM_PPTX)
    non_drm_path = os.path.abspath(NON_DRM_PPTX)

    print("=" * 60)
    print("Step 1: Check input files")
    print("=" * 60)
    for label, path in [("DRM PPTX", drm_path), ("Non-DRM PPTX", non_drm_path)]:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  [OK] {label}: {path} ({size_kb:.1f} KB)")
        else:
            print(f"  [FAIL] {label}: {path} NOT FOUND")
            sys.exit(1)

    print()
    print("=" * 60)
    print("Step 2: Open PowerPoint via COM and try copy-paste")
    print("=" * 60)

    import win32com.client
    import pythoncom

    pythoncom.CoInitialize()
    ppt = None

    try:
        ppt = win32com.client.Dispatch("PowerPoint.Application")
        ppt.Visible = True
        print("  [OK] PowerPoint launched")

        # Open DRM file
        print(f"  Opening DRM file (ReadOnly)...")
        try:
            drm_prs = ppt.Presentations.Open(drm_path, ReadOnly=True)
            print(f"  [OK] DRM file opened - {drm_prs.Slides.Count} slides")
        except Exception as e:
            print(f"  [FAIL] Cannot open DRM file: {e}")
            return

        # Open non-DRM file
        print(f"  Opening non-DRM file...")
        try:
            out_prs = ppt.Presentations.Open(non_drm_path)
            print(f"  [OK] Non-DRM file opened - {out_prs.Slides.Count} slides")
        except Exception as e:
            print(f"  [FAIL] Cannot open non-DRM file: {e}")
            drm_prs.Close()
            return

        # Delete existing slides
        while out_prs.Slides.Count > 0:
            out_prs.Slides(1).Delete()

        # Try copying just slide 1
        print()
        print("  Attempting to copy slide 1 from DRM...")
        try:
            drm_prs.Slides(1).Copy()
            time.sleep(0.5)
            print("  [OK] Slide copied to clipboard")
        except Exception as e:
            print(f"  [FAIL] Copy failed: {e}")
            drm_prs.Close()
            out_prs.Close()
            return

        print("  Attempting to paste into non-DRM...")
        try:
            out_prs.Slides.Paste()
            time.sleep(0.5)
            print(f"  [OK] Paste succeeded - now has {out_prs.Slides.Count} slides")
        except Exception as e:
            print(f"  [FAIL] Paste failed: {e}")
            drm_prs.Close()
            out_prs.Close()
            return

        drm_prs.Close()

        # Step 3: Try extracting from the open COM object (no save needed)
        print()
        print("=" * 60)
        print("Step 3: Extract from open COM object (in-memory)")
        print("=" * 60)

        slide = out_prs.Slides(1)
        print(f"  Slide has {slide.Shapes.Count} shapes")

        for shape_idx in range(1, slide.Shapes.Count + 1):
            shape = slide.Shapes(shape_idx)
            shape_type = shape.Type
            print(f"  Shape {shape_idx}: Type={shape_type}, Name={shape.Name}")

            if shape.HasTextFrame:
                try:
                    text = shape.TextFrame.TextRange.Text.strip()
                    if text:
                        print(f"    Text: \"{text[:80]}\"")
                except Exception as e:
                    print(f"    [WARN] Cannot read text: {e}")

            # Type 13 = Picture
            if shape_type == 13:
                print(f"    [OK] Image shape found")
                os.makedirs("output/images", exist_ok=True)
                try:
                    img_path = os.path.abspath("output/images/test_export.png")
                    shape.Export(img_path, 2)  # ppShapeFormatPNG
                    print(f"    [OK] Image exported to: {img_path}")
                except Exception as e:
                    print(f"    [FAIL] Image export failed: {e}")

        print()
        print("  [SUCCESS] In-memory extraction works!")
        print("  You can run main.py for the full pipeline.")

        out_prs.Close()

    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        if ppt is not None:
            ppt.Quit()
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    test()
