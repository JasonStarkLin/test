import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import Fun_BeadAssay as BA


folder = "D:\\NAS-TEMP_BE\\20201119\\P05\\"
#ExpFolder = "D:\\NAS-TEMP_BE\\20200520\\P01\\"
ExpFolder = folder

#Search files for posiiton calculation.
for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*Position.csv'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
#print(Targetfile)

Steps_N = 450
Fps = 451
FFT_Amp_threshold = 0.5

#Read position and calculate speed by FFT.
for Sample in Targetfile:
    print("Analyzing.........:")
    print("     ",Sample)
    Split_FileName = Sample.rsplit('-', 1)
    SampleName = Split_FileName[0]
    PositionPath = folder + Sample
    PositionData = pd.read_csv(PositionPath,delimiter=',')
    SpeedSheet = BA.GetSpeed(PositionData, Steps_N=Steps_N, Fps=Fps)
    SpeedSheet.loc[SpeedSheet["FFT_amp"] < FFT_Amp_threshold, "Speed(Hz)"] = 0 #Extract Speed and filter by a amplitube threshold
    SpeedSheet.to_csv(ExpFolder + SampleName + '-Speed.csv',index=False)

    #Plot basic results
    plt.plot_date(pd.to_datetime(SpeedSheet["DateTime"]), SpeedSheet["Speed(Hz)"],markersize=1)
    ax = plt.gca()
    fig = plt.gcf()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title(SampleName)
    plt.xlabel('Time')
    plt.ylabel('Speed(Hz)')
    plt.savefig(ExpFolder + SampleName + '-Speed.png')
    plt.clf()