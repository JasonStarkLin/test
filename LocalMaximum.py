import fnmatch
import os
import numpy as np
import pims
import scipy.optimize as opt
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import warnings
import time
from skimage.morphology import disk
from skimage.util import pad
from skimage.external.tifffile import imsave

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.morphology import closing, square,opening
from skimage.color import label2rgb
#from imagej_roi import write_imagej_roi
import pandas as pd
import Fun_BeadAssay as BA

def CentOfMass(Image):
    size_y, size_x = np.shape(Image)
    x = np.linspace(1,size_x,size_x) # from 1 in order to avoid position 0.
    y = np.linspace(1,size_y,size_y)
    x, y = np.meshgrid(x, y)
    Int_sum = np.sum(Image)
    Cen_x = (np.sum(x*Image)/Int_sum) -1 # minus one in order to shift back position which cause by in_weight_x
    Cen_y = (np.sum(y*Image)/Int_sum) -1

    return (Cen_x, Cen_y)

def ParticleShape(SpotImage,CM):
    #See reference:
    #Henriques, R. et al., 2010. QuickPALM: 3D real-time photoactivation nanoscopy image processing in ImageJ.
    size_y, size_x = np.shape(SpotImage)
    x = np.linspace(0,size_x-1,size_x) # from 1 in order to avoid position 0.
    y = np.linspace(0,size_y-1,size_y)
    x, y = np.meshgrid(x, y)
    Cx = int(CM[0])
    Cy = int(CM[1])
    s_l = np.sum((CM[0]-x[:,0:Cx+1])*SpotImage[:,0:Cx+1])/np.sum(SpotImage[:,0:Cx+1]) #Calculate the left half x direction center of mass taking center as origin.
    s_r = np.sum((x[:,Cx+1::]-CM[0])*SpotImage[:,Cx+1::])/np.sum(SpotImage[:,Cx+1::])
    s_a = np.sum((CM[1]-y[0:Cy+1,:])*SpotImage[0:Cy+1,:])/np.sum(SpotImage[0:Cy+1,:])
    s_b = np.sum((y[Cy+1::,:]-CM[1])*SpotImage[Cy+1::,:])/np.sum(SpotImage[Cy+1::,:])
    Sx = 1 - (abs(s_l - s_r) / (s_l + s_r))
    Sy = 1 - (abs(s_a - s_b) / (s_a + s_b))
    #print("s_l,s_r,s_a,s_b,Sx,Sy :",s_l,s_r,s_a,s_b,Sx,Sy)
    #imsave('E:\\YS1294-BufferExchange-85mM\\SpotImage-test.tif', SpotImage)


    return(Sx,Sy)


def LocalMaximum(Image,Rs,SNRlimit):
    PixelSize = 90
    FWHM = 800
    SNRLimit = 0.8
    Shapelimit = (0.75,0.75)
    IntMaxThre = 80
    Rs = np.round(FWHM / PixelSize / 2)
    Rs = int(Rs)

    Im_shape = np.shape(Image)
    Backextend = 2
    SignalMask = np.zeros((1 + 2 * Rs + Backextend * 2, 1 + 2 * Rs + Backextend * 2),dtype=np.int8)
    SignalMask[Backextend:-Backextend, Backextend:-Backextend] = np.zeros((1 + 2 * Rs, 1 + 2 * Rs),dtype=np.int8)+1
    BackgroundMask = (-1) * (SignalMask - 1)
    #print(SignalMask)
    SignalIndex = np.where(SignalMask == 1)
    Back_Index = np.where(SignalMask == 0)

    Rmax=Rs

    if Rs + Backextend >= Rmax:
        Rpad = Rs+Backextend
    else:
        Rpad = Rmax

    Mask = disk(Rmax)
    PImage = np.zeros((Im_shape[0],Im_shape[1])) #Processing image.
    PadImage = pad(Image,Rpad,'mean')
    #PadImage = pad(Image, Rpad, 'constant',constant_values=0) # Pad with 0.


    #Dilate and Merge the neighboring local maximum
    for i in range(Rpad,Im_shape[0]+Rpad):
        for j in range(Rpad,Im_shape[1]+Rpad):
            #print("j",j)
            Maxvalue = np.max(PadImage[i-Rmax:i+Rmax+1,j-Rmax:j+Rmax+1]*Mask)
            PImage[i-Rpad,j-Rpad] = Maxvalue

    plt.imshow(Image)
    #Filter Local Maximum
    for i in range(Rpad,Im_shape[0]+Rpad):
        for j in range(Rpad,Im_shape[1]+Rpad):
            #find local maximum
            if PImage[i-Rpad,j-Rpad] == Image[i-Rpad,j-Rpad]:
                LocalImage = PadImage[i-Rs-Backextend:i+Rs+Backextend+1,j-Rs-Backextend:j+Rs+Backextend+1]
                #Calculate SNR.
                Signal = np.mean(LocalImage[SignalIndex[0],SignalIndex[1]])
                Background = np.mean(LocalImage[Back_Index[0],Back_Index[1]])
                SNR = (Signal-Background)/np.std(LocalImage[Back_Index[0],Back_Index[1]])
                #print(SNR)
                if SNR >= SNRLimit and np.max(LocalImage)>IntMaxThre :
                    #Calculate the shape of the bright spot.
                    #Use center of mass to calculate the shape of the spot.
                    CropImage = PadImage[i-Rs:i+Rs+1,j-Rs:j+Rs+1]
                    CM = CentOfMass(CropImage)
                    S = ParticleShape(CropImage,CM)
                    print(S)

                    if S[0] > Shapelimit[0] and S[1] >Shapelimit[1]:
                        plt.scatter(j - Rpad,i - Rpad,marker='o',facecolors='none',edgecolors='r',s=2)
    plt.show()


    print("test")


folder = "E:\\20190717\\T0\\"

for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*.seq'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
print(Targetfile)

window_size = 6
Bead_sheet = pd.DataFrame(columns=["Label","DateTime","X","Y","Speed(Hz)"])
for i in Targetfile:
    Image = pims.open(folder+i)
    Split_FileName = i.rsplit('.seq')
    Sample = Split_FileName[0]
    print("File: ", i)
    print("     Frame Shape", Image._shape)
    print("     Image type: ", Image.pixel_type)
    print("     Frame Counts: ", Image._image_count)

'''
    LocalMaximum(ImageSTD, 3, 0.75)

    for j in range(Image._image_count):
    #for j in range(215,216):
        print("     Frame:",j)
        #Crop_image = Image[j][bead_y-window_size:bead_y+window_size,bead_x-window_size:bead_x+window_size]
        LocalMaximum(Image[j],3,0.75)
        plt.imshow(Image[j],cmap='gray')
        plt.show()
'''
print("Program finished")
