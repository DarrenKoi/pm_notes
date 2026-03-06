"""Copy slides from DRM-protected PPTX to non-DRM PPTX via PowerPoint COM automation.

The non-DRM PPTX is opened directly (not copied to disk first) because
it also cannot be saved locally. Extraction happens in-memory via COM.
"""

import os
import time

import win32com.client


def open_and_copy_slides(drm_path: str, non_drm_path: str):
    """Open both PPTX files in PowerPoint, copy slides, and return the COM presentation.

    Does NOT save or close the output presentation — the caller should
    extract content from the returned COM object while it's still open.

    Caller is responsible for pythoncom.CoInitialize/CoUninitialize.

    Args:
        drm_path: Path to DRM-protected PPTX file.
        non_drm_path: Path to empty non-DRM PPTX file.

    Returns:
        Tuple of (ppt_app, out_presentation) — caller must close/quit when done.
    """
    drm_path = os.path.abspath(drm_path)
    non_drm_path = os.path.abspath(non_drm_path)

    if not os.path.exists(drm_path):
        raise FileNotFoundError(f"DRM file not found: {drm_path}")
    if not os.path.exists(non_drm_path):
        raise FileNotFoundError(f"Non-DRM file not found: {non_drm_path}")

    ppt = win32com.client.Dispatch("PowerPoint.Application")
    ppt.Visible = True  # PowerPoint must be visible for copy-paste

    # Open DRM file as read-only
    drm_prs = ppt.Presentations.Open(drm_path, ReadOnly=True)
    # Open non-DRM file (don't copy — open the original directly)
    out_prs = ppt.Presentations.Open(non_drm_path)

    # Delete any existing slides in the non-DRM presentation
    while out_prs.Slides.Count > 0:
        out_prs.Slides(1).Delete()

    slide_count = drm_prs.Slides.Count
    print(f"[copy_slides] DRM file has {slide_count} slides")

    for i in range(1, slide_count + 1):
        drm_prs.Slides(i).Copy()
        time.sleep(0.3)

        out_prs.Slides.Paste()
        time.sleep(0.3)

        print(f"  Copied slide {i}/{slide_count}")

    # Close the DRM file — we're done with it
    drm_prs.Close()
    print(f"[copy_slides] All slides copied. DRM file closed.")

    # Return the app and the open presentation for in-memory extraction
    return ppt, out_prs
