# Output file structure

The HDF output (`*_mesoscaler.h5`) will have the following structure
(the MATLAB output should have the similar structure).

## `/images512` group

Annotated images for humans to validate the state of atlas registration.
Each entry contains a colored 512x512 image, and (in Python) is the
512x512x3 array of 8-bit unsigned integers.

As the registration procedure occurs in the 512x512 px format, all the
images are in 512x512 px no matter the size of the original brain image.

- `/images512/source` is the original, unannotated brain image.
- `/images512/landmarks` overlays the estimated landmark positions (in colors)
  on top of the original brain image.
- `/images512/alignment` overlays the projected AllenCCF-based reference atlas
  on top of the 'landmarks' image. The white dots indicate the landmark positions
  of the reference atlas.

## `/landmarks512` group

The landmarks estimated using the DeepLabCut network being applied
on the source brain image.

- The `names` attribute of the group contains the array of the
  names of the landmarks.
- The `/landmarks512/x`, `/landmarks512/y`, and `/landmarks512/likelihood`
  datasets contain the estimated x- and the y- coordinates, as well as the
  likelihood, of the landmarks.

> [!NOTE] 
> The X axis corresponds to the horizontal axis from left to right.
> The Y axis corresponds to the vertical axis from top to bottom.

## `/affine_ref_to_data512` dataset

This corresponds to the affine transformation matrix used to project
the reference atlas onto the source image. It is in the 'compressed'
form, and has (in Python) 2 rows and 3 columns.

This is intended to be used with the `cv2.warpAffine()` method.

## `/rois` group

Contains spatial ROI masks, in the binary form (0 or 1) as
unsigned 8-bit integer values.

> [!CAUTION]
> These ROI masks are merely the result of projecting the reference atlas
> onto the source brain, so they do not take into account whether there
> exist calcium signals of any sort whithin these ROIs.
> 
> Determining whether there are signals must be performed elsewhere
> (for the time being, at least).

- The `/rois/outline` dataset corresponds to the outline of the 'whole'
  brain (i.e. as far as Xiao et al., 2019 Nat Commun used), containing
  both left and the right hemispheres.
- The `/rois/left` and `/rois/right` groups contain individual ROI masks
  of the left and the right hemispheres.

Each entry (dataset) containing an individual ROI mask has following
information as attributes:

- `name` (string): the name of the ROI (normally the same as the name of the entry).
- `side` (string: `left` or `right`): whether this ROI is in the left or the right hemisphere
- `description` (string): a more detailed description about the ROI definition.
- `AllenID` (int): the Allen ID of the ROI.

> [!NOTE]
> In case when there is no corresponding Allen region, the value `0` is set as `AllenID`.
> For example, `/rois/right/outline` corresponds to the mask of 'whole' right hemisphere,
> and its `AllenID` is set to 0.
