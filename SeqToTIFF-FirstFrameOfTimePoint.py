import fnmatch
import os
import numpy as np
import pims
from skimage.io import imsave


folder = "X:\\Raw Data\\2020\\20201026\\Buffer Exchange\\"

for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*.seq'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
print(Targetfile)

Frame_steps = 450

for i in Targetfile:
    Image = pims.open(folder+i)

    ExpFile = np.str(np.char.rstrip(i, '.seq'))
    FirstFrame_index = np.arange(0,Image._image_count,Frame_steps)
    #FirstFrame_index = np.arange(0, 1000, Frame_steps)
    FisrtFramgImage = Image[FirstFrame_index]
    #imsave(folder + ExpFile + '.tif', Image[FirstFrame_index])
    for frame in FisrtFramgImage:
        imsave(folder + ExpFile + '-FirstFrame.tif', frame,append=True)


print("Program finished")
