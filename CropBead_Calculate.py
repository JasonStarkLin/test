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
import pandas as pd

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
    if Int_sum == 0:
        print("SUM of image intensity is 0. Error")
        return(0,0)
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

def CropBeadsImageToTiff(folder,Targetfile,Rois,window_size, **kwargs):
    for i in range(len(Targetfile)):
        Image_info = pims.open(folder + Targetfile[i])
        if "N_Frames" not in locals():
            N_Frames = Image_info._image_count
        else:
            N_Frames = N_Frames + Image_info._image_count

    N_beads = len(Rois)
    Time_info = pd.DataFrame(columns=['Frame', 'DateTime', 'T-Stage'])
    Crop_images = np.zeros((N_beads,N_Frames, 2 * window_size + 1, 2 * window_size + 1), dtype=np.uint8)
    frame_count = 0
    # Run for all the files
    # Import the transfer function for the beads position calibration.
    if "TranFun" in kwargs:
        TranFun = kwargs["TranFun"]
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
            #if frame_count<41400 :
                #frame_count = frame_count + 1
                #continue
            #if j != 3339 : continue
            progressBar(100 * frame_count / N_Frames, 100)

            #Crop images for each beads
            bead_index = 0
            for k in Rois:
                bead_xy = np.array([np.int(Rois[k]['x'][0]), np.int(Rois[k]['y'][0])])
                if "TranFun" in kwargs:
                    Timepoint = (frame_count//450)-1  # Because the imagej did not record transfer function for the first frame.
                    if Timepoint >=0:
                        Crop_cent_x = bead_xy[0] - TranFun.iloc[Timepoint,2]
                        Crop_cent_y = bead_xy[1] - TranFun.iloc[Timepoint,3]
                    else:
                        Crop_cent_x = bead_xy[0]
                        Crop_cent_y = bead_xy[1]
                else:
                    Crop_cent_x = bead_xy[0]
                    Crop_cent_y = bead_xy[1]

                Crop_images[bead_index,frame_count, :, :] = Image[j][Crop_cent_y - window_size:Crop_cent_y + window_size + 1,
                                       Crop_cent_x - window_size:Crop_cent_x + window_size + 1]
                bead_index = bead_index+1
            #Recorded the time informaiton for each beads.
            s = pd.Series({"Frame":frame_count,"DateTime":Image[j].metadata['time'],"T-Stage":O_Stage})
            Time_info = Time_info.append(s, ignore_index=True)
            frame_count = frame_count + 1
        print("")
    print('     Total frames number : ', frame_count)
    #save the images
    Split_FileName = Targetfile[0].rsplit('-', 2)
    Sample = Split_FileName[0]
    bead_index=0
    for i in Rois:
        imsave(ExpFolder + Sample + '-' + i + '.tif', Crop_images[bead_index,0:frame_count, :, :], photometric='minisblack')
        bead_index = bead_index + 1
    #Save time information
    Time_info.to_csv(ExpFolder + Sample + '-TimeInfo.csv', index=False)

def GetBeadPosition(folder, Sample,Bead_label,**kwargs):
    #Find the number of frames.
    BeadFilePath = folder + Sample + '-' + Bead_label + '.tif'
    ExpBeadFilePath = folder + Sample + '-'+"Fitted"+ Bead_label + '.tif'
    TimeFilePath = folder + Sample + '-TimeInfo.csv'
    TimeFile = pd.read_csv(TimeFilePath)
    Image = pims.open(BeadFilePath)
    N_Images = len(Image)
    #Read in all the images.
    size_y, size_x = Image._im_sz
    Images = np.zeros((N_Images, size_y, size_x), dtype=np.uint8)
    PositionSheet = pd.DataFrame(columns=['Frame', 'DateTime', 'T-Stage', 'x-Center', 'y-Center'])
    for i in range(N_Images):
        temp = Image[i]*(Image[i] > np.average(Image[i]) + np.std(Image[i]))
        Cent = SpotCentral(temp, fun='Centroid')
        s = pd.Series({"Frame": TimeFile.iloc[i,0], "DateTime": TimeFile.iloc[i,1], "T-Stage": TimeFile.iloc[i,2],'x-Center':Cent[0],'y-Center':Cent[1]})
        Images[i,:,:] = Image[i]
        PositionSheet = PositionSheet.append(s, ignore_index=True)

        Cx = int(np.round(Cent[0]))
        Cy = int(np.round(Cent[1]))
        temp = Image[i]
        temp[Cy, Cx] = 255
        #if i == 0:
            #imsave(ExpBeadFilePath, temp,photometric='minisblack',append = False)
        #else:
            #imsave(ExpBeadFilePath, temp, photometric='minisblack', append=True)
        progressBar(100 * i / N_Images, 100)
    print("")
    ExpSheet = folder + Sample + '-' + Bead_label + '-Position.csv'
    PositionSheet.to_csv(ExpSheet)
    imsave(ExpBeadFilePath, Images,photometric='minisblack',append = False)

    # Run for all the files
    # len(Targetfile)
    #for i in range(0, 1):
    # Import the transfer function for the beads position calibration.
    '''
    if "TranFun" in kwargs:
        TranFun = kwargs["TranFun"]
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
            #if frame_count<41400 :
                #frame_count = frame_count + 1
                #continue
            #if j != 3339 : continue
            progressBar(100 * frame_count / N_Frames, 100)

            if "TranFun" in kwargs:
                Timepoint = (frame_count//450)-1  # Because the imagej did not record transfer function for the first frame.
                if Timepoint >=0:
                    Crop_cent_x = bead_xy[0] - TranFun.iloc[Timepoint,2]
                    Crop_cent_y = bead_xy[1] - TranFun.iloc[Timepoint,3]
                else:
                    Crop_cent_x = bead_xy[0]
                    Crop_cent_y = bead_xy[1]
            else:
                Crop_cent_x = bead_xy[0]
                Crop_cent_y = bead_xy[1]

            Crop_images[frame_count, :, :] = Image[j][Crop_cent_y - window_size:Crop_cent_y + window_size + 1,
                                       Crop_cent_x - window_size:Crop_cent_x + window_size + 1]
            #Kick out background
            temp = Crop_images[frame_count, :, :] * (Crop_images[frame_count, :, :] > np.average(Crop_images[frame_count, :, :]) + np.std(Crop_images[frame_count, :, :]))
            Cent = SpotCentral(temp, fun='Centroid')

            if 'Bead_P' in locals():
                Bead_P = np.append(Bead_P, np.array([[frame_count,Image[j].metadata['time'],O_Stage, Cent[0], Cent[1]]]), axis=0)
            else:
                Bead_P = np.array([[frame_count,Image[j].metadata['time'],O_Stage, Cent[0], Cent[1]]])

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
    return Bead_P'''


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
window_size = 6
Split_FileName = Targetfile[0].rsplit('-', 2)
Sample = Split_FileName[0]
Rois = read_roi_zip(folder+Sample+"-ROI.zip") #Genreate from imagej
TranFun = pd.read_csv(folder+Sample+"-TransferFunction.csv") #Generate from imagej. For each timepoint.

print("Crop all the beads image.")
CropBeadsImageToTiff(folder,Targetfile,Rois,window_size,TranFun=TranFun)

#print("Fit the beads position")
#for i in Rois:
    #print(i)
    #GetBeadPosition(folder, Sample, i, ImageSave=True)

print("Program finished")
