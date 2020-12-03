import fnmatch
import os
import numpy as np
import pims
from skimage.external.tifffile import imsave


folder = "X:\\Raw Data\\2020\\20200407\\Buffer Exchange\\"

for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*.seq'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
print(Targetfile)

Segment_steps = 450

for i in Targetfile:
    Image = pims.open(folder+i)

    ExpFile = np.str(np.char.rstrip(i, '.seq'))
    npImage = np.empty((Segment_steps, Image._height, Image._width), dtype=np.uint8)
    FirstFrame_index = np.arange(0, Image._image_count, Segment_steps)
    Seg_N = len(FirstFrame_index)
    for j,Findex in enumerate(FirstFrame_index):
        npImage[:,:,:] = Image[Findex:Findex+Segment_steps]
        FileName = ExpFile+"-"+"{:0>3d}".format(j)
        imsave(folder + FileName + '.tif', npImage)
        print(FileName, "\nProgress:", np.round(100*j/Seg_N), "%")

print("Program finished")
