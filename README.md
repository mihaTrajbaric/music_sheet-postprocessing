# Music sheet postprocessing
Script for postprocessing scanned music sheet.

# Features
- removing scanned artefacts on the edges
- thresholding
- deskewing

# How to use
## Basic usage
In this mode, MegaMusicProcessing is intended to be used as a sole step in image postprocessing.
To use this mode, set `mode='production'`. For other params, see [Parameters](#Parameters).

## Advance mode
In this mode, MegaMusicProcessing should be run in several steps:
1) Run Script in 'photoshop' mode. This mode will perform masking but pixels, that will pass filtering, will retain its original colours. This will allow a user in step 2) to easily discriminate between dark black notes and greyer pencil marks.
2) Open exported images, use photoshop to remove any remaining pencil marks and other unwanted artefacts.
3) Run the script in 'production' mode on photoshopped images. It will output images in binarized format (all non-white pixels will become  black)

For other params, see [Parameters](#Parameters).

## Parameters
- in_path: Path to directory with scanned images, relative to `MegaMusicProcessing.py` script.
- out_path: Path where a script will output processed images, also relative to `MegaMusicProcessing.py` script.
- out_format: Desired output format for processed images. Usually bmp or jpeg. See [OpenCV docs](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) for more info.
- mode: Must be set to either 'production' or 'photoshop'. 'production' will output binarized image, while 'photoshop' will perform masking with pixels, but leave remaining pixels on their original values.
- edge_size: Size (in px) of edge to be removed and replaced with white pixels. The image will retain its previous dimensions.
- deskew (bool): Deskew image
- max_skew: Max angle (in degrees) to deskew. If angle, determined by algorithm will be greater, Image will be rotated for [max_skew] angle only.
- align_mode: One of 'none', 'x', 'y', or 'full'. Chooses axis on which to preform aligning. `align_mode='full'` will preform aligning on both axis.


## Example function call
```Python
from MegaMusicPreprocessing import process_images
process_images(
        in_path='music_sheet_scanned',
        out_path='music_sheet_production',
        out_format='bmp',
        mode='production',
        edge_size=50,
        deskew=True,
        max_skew=10,
        align_mode='full'
    )
```
