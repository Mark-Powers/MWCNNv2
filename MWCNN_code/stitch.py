from data import common
import rawpy
import numpy as np
import imageio
from PIL import Image

hr_img = None
lr_img = None
sr_img = None
for i in range(6):
    hr_row = None
    lr_row = None
    sr_row = None
    for j in range(6):
        prefix = "/home/mppowers/output/1019.dng"+str(6*i + j)
        new_hr = np.asarray(Image.open( prefix+"_hr.png"))[:500][:500]
        if hr_row is None:
            hr_row = new_hr
        else:
            hr_row = np.vstack([hr_row, new_hr])
        
        new_lr = np.asarray(Image.open( prefix+"_lr.png"))[:500][:500]
        if lr_row is None:
            lr_row = new_lr
        else:
            lr_row = np.vstack([lr_row, new_lr])
        
        new_sr = np.asarray(Image.open( prefix+"_sr.png"))[:500][:500]
        if sr_row is None:
            sr_row = new_sr
        else:
            sr_row = np.vstack([sr_row, new_sr])
    if hr_img is None:
        hr_img = hr_row
    else:
        hr_img = np.hstack([hr_img, hr_row])

    if sr_img is None:
        sr_img = sr_row
    else:
        sr_img = np.hstack([sr_img, sr_row])

    if lr_img is None:
        lr_img = lr_row
    else:
        lr_img = np.hstack([lr_img, lr_row])

imageio.imwrite("/home/mppowers/output/1019.dng_stitched_hr.png", hr_img)
imageio.imwrite("/home/mppowers/output/1019.dng_stitched_sr.png", sr_img)
imageio.imwrite("/home/mppowers/output/1019.dng_stitched_lr.png", lr_img)
