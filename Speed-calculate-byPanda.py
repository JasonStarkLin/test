import fnmatch
import os
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import tz
import csv
import pandas as pd
import Fun_BeadAssay as BA

folder = "K:\\For NAS Temp\\20191009-BE-850085-T1-S01-P01\\"
ExpFolder = "K:\\For NAS Temp\\20191009-BE-850085-T1-S01-P01\\"

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
    PositionData = pd.read_csv(PositionPath,delimiter=',')
    #PositionData = np.genfromtxt(PositionPath, delimiter=',')
    #PositionData = PositionData[1:, 1:]
    P_cmp = 1j*PositionData["y-Center"]
    P_cmp = P_cmp + PositionData["x-Center"] #position comples Z = x +yi
    N_Speed = len(P_cmp)//Steps_N
    SpeedSheet = pd.DataFrame(columns=["Frame","DateTime","T-Stage","Speed(Hz)"])
    #Speed_seq = np.zeros((N_Speed,2))
    #Speed_seq[:,0]=np.linspace(1,N_Speed,N_Speed)-1
    for i in range(N_Speed):
        Temp_Pdata = P_cmp[i*Steps_N:(i+1)*Steps_N]
        ave_FFT_P = np.average(Temp_Pdata)
        Normal_factor = 2/len(Temp_Pdata)
        FFT_P = abs(fft(Temp_Pdata-ave_FFT_P)*Normal_factor) #Calculate FFT and normalize the amplitude.
        freq = np.fft.fftfreq(Temp_Pdata.size,1/Fps) #Generate the frequence sequence for FFT results.
        freq_max_ind = np.where(FFT_P == np.max(FFT_P)) #Find the position of the maximum in FFT results.
        #print('Speed (Hz) :',freq[freq_max_ind[0]])
        if FFT_P[freq_max_ind[0]] > FFT_Amp_threshold:   #Extract Speed and filter by a amplitube threshold
            Speed = freq[freq_max_ind[0]][0]
        else:
            Speed = 0
        Frame = PositionData.iloc[i*Steps_N,0]
        DateTime = PositionData.iloc[i*Steps_N,1]
        Tstage = PositionData.iloc[i*Steps_N,2]

        s = pd.Series({"Frame":Frame,"DateTime":DateTime,"T-Stage":Tstage,"Speed(Hz)":abs(Speed)})
        SpeedSheet = SpeedSheet.append(s, ignore_index=True)

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
    SpeedSheet.to_csv(ExpFolder + SampleName + '-Speed.csv',index=False)
    plt.plot_date(pd.to_datetime(SpeedSheet["DateTime"]), SpeedSheet["Speed(Hz)"],markersize=1)
    ax = plt.gca()
    fig = plt.gcf()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title(SampleName)
    plt.xlabel('Time')
    plt.ylabel('Speed(Hz)')
    plt.savefig(ExpFolder + SampleName + '-Speed.png')
    plt.clf()