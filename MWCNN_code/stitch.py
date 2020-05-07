import imageio                                                                                       
import glob
import numpy as np

filelist = open("filelist.txt")
for fn in filelist.readlines():
    orig = imageio.imread("/home/mark/validation/"+fn.strip()+"/ref.png")
    print()
    print(fn)
    print("orig", orig.shape)
    diffH = min(orig.shape[0]-512, 500)
    diffW = min(orig.shape[1]-512, 500)
    print("diffs", diffH, diffW)
    for ty in ["hr", "lr", "sr"]:
        print(ty)
        files = sorted(glob.glob("/home/mark/log_3/"+fn.strip()+"_*_"+ty+".png"))
        d = {}
        for filename in files:
            i = int(filename.split("_")[-2])
            d[i] = filename 
        for key in sorted(d):
            filename = d[key]
            new = imageio.imread(filename)
            if key % 4 == 0:
                row1 = new[:diffH, :diffW]
            elif key % 4 == 1:
                new = new[:, :diffW]
                row1 = np.vstack([row1, new])
            elif key % 4 == 2:
                row2 = new[:diffH, :]  
            else:
                new = new[:, :]  
                row2 = np.vstack([row2, new])
                img = np.hstack([row1, row2])
                print("row1", row1.shape, "row2", row2.shape, "img", img.shape)
                out_fn = "/home/mark/log_3_stitch/"+fn.strip()+"_"+"_stitched_"+ty+".png"
                imageio.imwrite(out_fn, img)
