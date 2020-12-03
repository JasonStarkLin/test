import fnmatch
import os
import numpy as np
import pims
from skimage.io import imsave


folder = "C:\\Users\\ULTRABBUG\\Downloads\\temp\\"

for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*.seq'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
print(Targetfile)

N_frames = 450
N_seq = 1
for i in range(len(Targetfile)):
    Image = pims.open(folder+Targetfile[i])
    ExpFile = np.str(np.char.rstrip(Targetfile[i], '.seq'))
    npImage = np.empty((N_frames, Image._height, Image._width), dtype=np.uint8)
    npImage[:,:,:] = Image[(N_seq-1)*450:(N_seq-1)*450+N_frames]
    imsave(folder + ExpFile + '.tif', npImage)#,bigtiff=True)
print("Program finished")