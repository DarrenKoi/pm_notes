"""Copy slides from DRM-protected PPTX to non-DRM PPTX via PowerPoint COM automation."""

import os
import shutil
import time

import win32com.client
import pythoncom


def copy_slides(drm_path: str, non_drm_path: str, output_path: str) -> str:
    """Open DRM PPTX and non-DRM PPTX in PowerPoint, copy all slides, save to output_path.

    Args:
        drm_path: Path to DRM-protected PPTX file.
        non_drm_path: Path to empty non-DRM PPTX file (template).
        output_path: Path where the copied (non-DRM) result will be saved.

    Returns:
        Absolute path to the saved output file.
    """
    drm_path = os.path.abspath(drm_path)
    non_drm_path = os.path.abspath(non_drm_path)
    output_path = os.path.abspath(output_path)

    if not os.path.exists(drm_path):
        raise FileNotFoundError(f"DRM file not found: {drm_path}")
    if not os.path.exists(non_drm_path):
        raise FileNotFoundError(f"Non-DRM file not found: {non_drm_path}")

    # Copy the non-DRM template so the original stays untouched
    shutil.copy2(non_drm_path, output_path)

    pythoncom.CoInitialize()
    ppt = None
    try:
        ppt = win32com.client.Dispatch("PowerPoint.Application")
        ppt.Visible = True  # PowerPoint must be visible for copy-paste

        # Open DRM file as read-only
        drm_prs = ppt.Presentations.Open(drm_path, ReadOnly=True, WithWindow=False)
        # Open the output copy (writable)
        out_prs = ppt.Presentations.Open(output_path, WithWindow=False)

        # Delete any existing slides in the output presentation
        while out_prs.Slides.Count > 0:
            out_prs.Slides(1).Delete()

        slide_count = drm_prs.Slides.Count
        print(f"[copy_slides] DRM file has {slide_count} slides")

        for i in range(1, slide_count + 1):
            # Select and copy the slide from DRM presentation
            drm_prs.Slides(i).Copy()
            time.sleep(0.3)  # Brief pause for clipboard

            # Paste into output presentation
            out_prs.Slides.Paste()
            time.sleep(0.3)

            print(f"  Copied slide {i}/{slide_count}")

        out_prs.Save()
        print(f"[copy_slides] Saved to: {output_path}")

        drm_prs.Close()
        out_prs.Close()

    finally:
        if ppt is not None:
            ppt.Quit()
        pythoncom.CoUninitialize()

    return output_path
