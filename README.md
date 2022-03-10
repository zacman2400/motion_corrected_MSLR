# motion_corrected_MSLR
motion modeling integrated into MSLR recon
TO EDIT
block_operator.py: minimally modified from Frank's code, creates block operator in sigpy, use on cupy L and R arrays to from back into image, use B.H on images to break into cupy L and R arrays,
needs sense maps for original image size
motion_correction jupyter notebook: (will be) step by step run through of recon
motion_correction_MSLR26: (rename), contains all utility functions for complete recon, need to comment this
run_MSLR: (to edit): scripted version of the jupyter notebook
