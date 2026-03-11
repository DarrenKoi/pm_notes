# drm-pptx-extraction

> DRM protected PPTX files are opened in Microsoft PowerPoint and each slide is exported as a PNG image.

## What It Does

This project no longer copies slides into a separate PPTX or extracts text/shapes.
It now does one thing:

- open each DRM PPTX with PowerPoint COM automation
- capture every slide as a PNG
- save images into a folder named after the PPTX file

PNG is used by default because it avoids JPEG recompression artifacts on text-heavy slides.

## Requirements

- Windows
- Microsoft PowerPoint installed
- Python 3.10+

## Install

```bash
pip install -r requirements.txt
```

## Usage

Export every `*.pptx` file under `input/`:

```bash
python main.py
```

Export one specific file:

```bash
python main.py input/drm_file.pptx
```

Export multiple files or patterns:

```bash
python main.py input/a.pptx input/team-*.pptx
```

Choose a different output root:

```bash
python main.py input/*.pptx -o captured_slides
```

## Output Layout

```text
output/
├── file_a/
│   ├── slide_001.png
│   ├── slide_002.png
│   └── ...
├── file_b/
│   ├── slide_001.png
│   └── ...
└── ...
```

Each folder name matches the PPTX filename without the extension.

## Files

```text
├── main.py
├── export_slides.py
├── requirements.txt
├── input/
└── output/
```

## Notes

- The script opens PowerPoint for each file and closes it after export.
- Existing `slide_*.png` files in the target folder are cleared before a rerun.
- Actual export must be run on Windows because PowerPoint COM automation is required.
