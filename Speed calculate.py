import fnmatch
import os
import numpy as np
from scipy.fftpack import fft,ifft
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

folder = "E:\\20190724-YS1294-S1\\"
ExpFolder = "E:\\20190724-YS1294-S1\\"

#Search files for posiiton calculation.
for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*Position.csv'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
print(Targetfile)

Steps_N = 450
Fps = 451
FFT_Amp_threshold = 0.5

#Read position and calculate speed by FFT.
for Sample in Targetfile:
    Split_FileName = Sample.rsplit('-', 1)
    SampleName = Split_FileName[0]
    PositionPath = folder + Sample
    PositionData = np.genfromtxt(PositionPath, delimiter=',')
    PositionData = PositionData[1:, 1:]
    P_cmp = 1j*PositionData[:,1]
    P_cmp = P_cmp + PositionData[:,0] #position comples Z = x +yi
    N_Speed = len(P_cmp)//Steps_N
    Speed_seq = np.zeros((N_Speed,2))
    Speed_seq[:,0]=np.linspace(1,N_Speed,N_Speed)-1
    for i in range(0,N_Speed):
        Temp_Pdata = P_cmp[i*Steps_N:(i+1)*Steps_N]
        ave_FFT_P = np.average(Temp_Pdata)
        Normal_factor = 2/len(Temp_Pdata)
        FFT_P = abs(fft(Temp_Pdata-ave_FFT_P)*Normal_factor) #Calculate FFT and normalize the amplitude.
        freq = np.fft.fftfreq(Temp_Pdata.size,1/Fps) #Generate the frequence sequence for FFT results.
        freq_max_ind = np.where(FFT_P == np.max(FFT_P)) #Find the position of the maximum in FFT results.
        #print('Speed (Hz) :',freq[freq_max_ind[0]])
        if FFT_P[freq_max_ind[0]] > FFT_Amp_threshold:   #Extract Speed and filter by a amplitube threshold
            Speed = freq[freq_max_ind[0]]
        else:
            Speed = 0
        Speed_seq[i,1]=abs(Speed)
        ''' # For debug -> see the spectrum and orbit
        if i >21 and SampleName == "20190712-YS1294-S1-Bead-08":
            fig, axs = plt.subplots(2, 1)
            fig.suptitle(SampleName)
            axs[0].scatter(Temp_Pdata.real,Temp_Pdata.imag,s=2)
            axs[0].set_aspect('equal')
            axs[1].scatter(freq,FFT_P,s=1)
            str_lable = "(%.1f,%.1f)"%(Speed,FFT_P[freq_max_ind[0]])
            axs[1].text(Speed,FFT_P[freq_max_ind[0]],str_lable,fontsize=10)
            axs[1].set_title('Speed : %.1f Hz' %Speed_seq[i,1])
            plt.show()
        '''
    with open(ExpFolder + SampleName + '-Speed.csv', 'w', newline='') as SpeedData:
        SpeedWriter = csv.writer(SpeedData)
        SpeedWriter.writerow(['Time', 'Speed (Hz)'])
        for row in range(len(Speed_seq)):
            SpeedWriter.writerow(Speed_seq[row, :])

    plt.scatter(Speed_seq[:,0],Speed_seq[:,1],s=1)
    plt.title(SampleName)
    plt.xlabel('Time')
    plt.ylabel('Speed(Hz)')
    plt.savefig(ExpFolder + SampleName + '-Speed.png')
    plt.clf()








#Search files for Speeed calculation.