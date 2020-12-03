import fnmatch
import os
import numpy as np
import pims
import scipy.optimize as opt
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import warnings
from skimage.external.tifffile import imsave
import sys
import csv
from read_roi import read_roi_zip
import time


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)+ c*((y-yo)**2)))
    return g.ravel()

def CentOfMass(Image):
    size_y, size_x = np.shape(Image)
    x = np.linspace(1,size_x,size_x) # from 1 in order to avoid position 0.
    y = np.linspace(1,size_y,size_y)
    x, y = np.meshgrid(x, y)
    Int_sum = np.sum(Image)
    Cen_x = (np.sum(x*Image)/Int_sum) -1 # minus one in order to shift back position which cause by in_weight_x
    Cen_y = (np.sum(y*Image)/Int_sum) -1

    return (Cen_x, Cen_y)

def Gaussian2Dfit(Image):
    size_y, size_x = np.shape(Image)
    y, x = np.mgrid[:size_y, :size_x]
    max_p = np.where(Image == np.max(Image))
    #initial_guess = (np.max(Image), max_p[1][0], max_p[0][0], 5, 5, 0, np.min(Crop_image))

    p_init = models.Gaussian2D(amplitude=np.max(Image), x_mean=max_p[1][0], y_mean=max_p[0][0])
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, Image)
    return p, p(x,y)

def SpotCentral(Image,**kwargs):
    '''THis function will fit the image as my define'''
    try:
        #Centroid method
        if kwargs['fun'] == 'Centroid':
            Cent = CentOfMass(Image)
        elif kwargs['fun'] == 'npGaussian':
            #Use numerical python Gaussian fitting
            #Learn from: https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
            try:
                max_p = np.where(Image == np.max(Image))
                initial_guess = (np.max(Image), max_p[1][0], max_p[0][0], 5, 5, 0, np.min(Image))
                height, width = np.shape(Image)
                fit_bound = ((0, 0, 0, 0, 0, 0, 0),
                             (np.inf, width, height, width/2.0, height/2.0, 2 * np.pi, 255))
                x = np.linspace(0, width-1,width)
                y = np.linspace(0, height-1,height)
                x, y = np.meshgrid(x, y)
                popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), Image.flatten(), p0=initial_guess, bounds=fit_bound)
                Cent = (popt[1], popt[2])
            except:
                Cent = (max_p[1][0], max_p[0][0])
        elif kwargs['fun'] == 'astropyGaussian':
            #Use the 2D gaussain fit learn from astropy.
            Gp, Gp_image = Gaussian2Dfit(Image)
            Cent = (Gp._parameters[1],Gp._parameters[2])
        else:
            print("No such Method in my function")
            print("Try following: Centroid, npGaussian, astropyGaussian")
    except:
        Cent = CentOfMass(Image)

    return Cent

def progressBar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r     Progress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def GetBeadsPosition(folder,Targetfile, window_size, Bead_xy, **kwargs):
    for i in range(len(Targetfile)):
        Image_info = pims.open(folder + Targetfile[i])
        if "N_Frames" not in locals():
            N_Frames = Image_info._image_count
        else:
            N_Frames = N_Frames + Image_info._image_count

    Crop_images = np.zeros((N_Frames, 2 * window_size + 1, 2 * window_size + 1), dtype=np.uint8)
    frame_count = 0
    # Run for all the files
    # len(Targetfile)
    #for i in range(0, 1):
    for i in range(0,len(Targetfile)):
        Image = pims.open(folder + Targetfile[i])
        O_Stage = Targetfile[i].rsplit('-')[-2]
        print("     File: ", Targetfile[i])
        print("     Frame Shape", Image._shape)
        print("     Image type: ", Image.pixel_type)
        print("     Frame Counts: ", Image._image_count)
        print("     Operation Stage: ", O_Stage)
        # Run for the frames
        #for j in range(0, 450):
        for j in range(Image._image_count):
            # print("     Frame:",j)
            progressBar(100 * frame_count / N_Frames, 100)
            Crop_images[frame_count, :, :] = Image[j][bead_xy[1] - window_size:bead_xy[1] + window_size + 1,
                                       bead_xy[0] - window_size:bead_xy[0] + window_size + 1]
            #Kick out background
            temp = Crop_images[frame_count, :, :] * (Crop_images[frame_count, :, :] > np.average(Crop_images[frame_count, :, :]) + np.std(Crop_images[frame_count, :, :]))
            Cent = SpotCentral(temp, fun='Centroid')

            if 'Bead_P' in locals():
                Bead_P = np.append(Bead_P, np.array([[frame_count, Cent[0], Cent[1]]]), axis=0)
            else:
                Bead_P = np.array([[frame_count, Cent[0], Cent[1]]])

            # For saving the images.
            Cx = int(np.round(Cent[0]))
            Cy = int(np.round(Cent[1]))
            Crop_images[frame_count, Cy, Cx] = 255
            frame_count = frame_count + 1
        print("")
    print('     Totally frames number : ', frame_count)
    #save the images
    if kwargs["ImageSave"] is True:
        Split_FileName = Targetfile[0].rsplit('-', 2)
        Sample = Split_FileName[0]
        imsave(ExpFolder + Sample + '-' + kwargs['Bead_label'] + '.tif', Crop_images[0:frame_count, :, :], photometric='minisblack')
    return Bead_P


folder = "E:\\20190724-YS1294-S1\\"
ExpFolder = "E:\\20190724-YS1294-S1\\"

#Search files for posiiton calculation.
for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*.seq'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
print(Targetfile)

#Calculate position base on ROI file from imagej.
window_size = 12
Split_FileName = Targetfile[0].rsplit('-', 2)
Sample = Split_FileName[0]
Rois = read_roi_zip(folder+Sample+"-RefROI.zip")

for i in Rois:
    print(i)
    #print(Rois[i]['x'],Rois[i]['y'])
    bead_xy = np.array([np.int(Rois[i]['x'][0]),np.int(Rois[i]['y'][0])])
    Bead_P = GetBeadsPosition(folder,Targetfile,window_size,bead_xy,ImageSave=True,Bead_label=i)
    #Output the positions.
    with open(ExpFolder + Sample +'-'+ i + '-Position.csv', 'w', newline='') as SpeedData:
        SpeedWriter = csv.writer(SpeedData)
        SpeedWriter.writerow(['Frames', 'x-Center', 'y-Center'])
        for row in range(len(Bead_P)):
            # print(len(angle))
            SpeedWriter.writerow(Bead_P[row, :])



print("Program finished")
