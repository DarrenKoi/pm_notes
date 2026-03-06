"""Extract text and images from an open PowerPoint COM presentation object."""

import json
import os


# PowerPoint shape type constants (msoShapeType)
MSO_PICTURE = 13
MSO_GROUP = 6
MSO_PLACEHOLDER = 14


def extract_from_com(presentation, output_dir: str) -> dict:
    """Extract text and images from an open COM Presentation object.

    This works directly on the in-memory PowerPoint object,
    so no file saving to disk is needed.

    Args:
        presentation: win32com PowerPoint Presentation object (already open).
        output_dir: Directory to write images and result JSON.

    Returns:
        Structured dict with slide-by-slide extraction results.
    """
    output_dir = os.path.abspath(output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    slide_count = presentation.Slides.Count
    result = {
        "source": presentation.Name,
        "total_slides": slide_count,
        "slides": [],
    }

    for slide_idx in range(1, slide_count + 1):
        slide = presentation.Slides(slide_idx)
        slide_data = {
            "slide_number": slide_idx,
            "title": None,
            "texts": [],
            "images": [],
        }

        for shape_idx in range(1, slide.Shapes.Count + 1):
            shape = slide.Shapes(shape_idx)
            _process_shape(shape, slide_idx, slide_data, images_dir)

        # Try to get title from the slide's title placeholder
        if slide_data["title"] is None:
            try:
                title_shape = slide.Shapes.Title
                if title_shape and title_shape.HasTextFrame:
                    title_text = title_shape.TextFrame.TextRange.Text.strip()
                    if title_text:
                        slide_data["title"] = title_text
            except Exception:
                pass

        print(f"[extract] Slide {slide_idx}: title={slide_data['title']!r}, "
              f"{len(slide_data['texts'])} text blocks, {len(slide_data['images'])} images")

        result["slides"].append(slide_data)

    # Save JSON result
    json_path = os.path.join(output_dir, "extraction_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[extract] Result saved to: {json_path}")

    return result


def _process_shape(shape, slide_idx: int, slide_data: dict, images_dir: str):
    """Extract text and images from a single COM shape."""
    # Extract text
    if shape.HasTextFrame:
        try:
            text = shape.TextFrame.TextRange.Text.strip()
            if text:
                # Check if this is the title placeholder
                try:
                    if shape.PlaceholderFormat.Type == 1:  # ppPlaceholderTitle
                        slide_data["title"] = text
                    else:
                        slide_data["texts"].append(text)
                except Exception:
                    slide_data["texts"].append(text)
        except Exception:
            pass

    # Extract image by exporting shape as PNG
    shape_type = shape.Type
    if shape_type == MSO_PICTURE:
        _export_shape_as_image(shape, slide_idx, slide_data, images_dir)

    # Recurse into group shapes
    if shape_type == MSO_GROUP:
        try:
            for child_idx in range(1, shape.GroupItems.Count + 1):
                child = shape.GroupItems(child_idx)
                _process_shape(child, slide_idx, slide_data, images_dir)
        except Exception:
            pass


def _export_shape_as_image(shape, slide_idx: int, slide_data: dict, images_dir: str):
    """Export a shape as a PNG image file."""
    img_num = len(slide_data["images"]) + 1
    img_filename = f"slide_{slide_idx:02d}_img_{img_num:02d}.png"
    img_path = os.path.join(images_dir, img_filename)

    try:
        shape.Export(img_path, 2)  # 2 = ppShapeFormatPNG
        slide_data["images"].append(os.path.join("images", img_filename))
        print(f"  Saved image: {img_filename}")
    except Exception as e:
        print(f"  [WARN] Failed to export image on slide {slide_idx}: {e}")
