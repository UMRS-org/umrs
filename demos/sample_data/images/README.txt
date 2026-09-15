Sample Images for UMRS Demos
=============================

This directory should contain test images for the SAM3 segmentation
and HMAE inference demos.  Large binary files are not committed to git.

To set up sample images:

1. Copy or symlink PNG/JPG images into this directory:

     cp ~/my_images/*.png demos/sample_data/images/

2. Supported formats: .png, .jpg, .tiff

3. Recommended test images:
   - SAM3 segmentation: natural or microscopy images (any resolution)
   - HMAE inference: grayscale volume slices (1024x1024 PNG recommended)
   - Defect analysis: SEM images of materials with visible defects

4. Alternatively, use images from the data/ directory in the repo root:

     ln -s ../../data/img/*.png demos/sample_data/images/

Note: The demos will check for the presence of images and print
a helpful error message if none are found.
