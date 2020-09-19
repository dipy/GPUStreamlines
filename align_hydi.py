import numpy as np
from dipy.align.imaffine import AffineMap
import nibabel as nib
static = nib.load("BR_eddy_corrected_HYDI_MB3_125_avg_LPCA_1.00_gaussian_SRcollaborative_x2.nii")
moving = nib.load("brain_mask_1.25.nii.gz")
identity = np.eye(4)
affine_map = AffineMap(identity,
                       static.shape[:3], static.affine,
                       moving.shape, moving.affine)
resampled = affine_map.transform(moving.get_fdata())
nib.save(nib.Nifti1Image(resampled, static.affine), "aligned_mask.nii.gz")
nib.save(nib.Nifti1Image(static.get_fdata()[..., 0], static.affine),
         "hydi_b0_vol.nii.gz")
