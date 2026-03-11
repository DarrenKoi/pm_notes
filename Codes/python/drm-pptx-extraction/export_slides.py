"""PowerPoint COM helpers for exporting slides as PNG images."""

from __future__ import annotations

import gc
from pathlib import Path

import win32com.client


def export_slides_to_png(pptx_path: str | Path, output_dir: str | Path) -> int:
    """Open a presentation and export each slide as a PNG image."""
    presentation_path = Path(pptx_path).resolve()
    output_path = Path(output_dir).resolve()

    if not presentation_path.exists():
        raise FileNotFoundError(f"PPTX file not found: {presentation_path}")

    _prepare_output_dir(output_path)

    powerpoint = None
    presentation = None

    try:
        powerpoint = win32com.client.DispatchEx("PowerPoint.Application")
        powerpoint.DisplayAlerts = 0
        powerpoint.Visible = True

        presentation = powerpoint.Presentations.Open(
            str(presentation_path),
            ReadOnly=True,
        )

        slide_count = presentation.Slides.Count
        digits = max(3, len(str(slide_count)))
        exported_count = 0

        print(f"[export] {presentation_path.name}: {slide_count} slide(s)")

        for slide_number in range(1, slide_count + 1):
            image_path = output_path / f"slide_{slide_number:0{digits}d}.png"
            try:
                presentation.Slides(slide_number).Export(str(image_path), "PNG")
                exported_count += 1
                print(f"  Saved {image_path.name}")
            except Exception as exc:
                print(f"  [WARN] Failed slide {slide_number}: {exc}")

        return exported_count
    finally:
        if presentation is not None:
            try:
                presentation.Close()
            except Exception:
                pass

        if powerpoint is not None:
            try:
                powerpoint.Quit()
            except Exception:
                pass

        del presentation
        del powerpoint
        gc.collect()


def _prepare_output_dir(output_dir: Path) -> None:
    """Create output dir and clear previous slide captures for reruns."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for existing_file in output_dir.glob("slide_*.png"):
        existing_file.unlink()
