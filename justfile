# format, lint, typecheck
check:
    uv run ruff format
    uv run ruff check --fix
    uv run ty check

# run all detectors on example images
measure IMGS_DIR="img":
    uv run run_images.py {{IMGS_DIR}}

# run the detector in real time using a webcam
cam DETECTOR DEVICE="0":
    uv run webcam.py {{DETECTOR}} {{DEVICE}}
