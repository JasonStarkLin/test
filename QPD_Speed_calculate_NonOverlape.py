import fnmatch
import os
import numpy as np
from nptdms import TdmsFile
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import Fun_BeadAssay as BA
from matplotlib.gridspec import GridSpec

def Trans_PositionData(FilePath):
    PositionData = pd.DataFrame()
    Tdms_file = TdmsFile(FilePath)
    #Tdms_file.as_dataframe(time_index=True, absolute_time=True)
    X_channel = Tdms_file.object('QPD', 'Dev1/ai5')
    Y_channel = Tdms_file.object('QPD', 'Dev1/ai6')
    X_data = X_channel.data*x_ratio      #unit: nm
    Y_data = Y_channel.data*y_ratio      #unit: nm
    PositionData["x-Center"] = X_data
    PositionData["y-Center"] = Y_data
    PositionData["Frame"] = PositionData.index
    PositionData['DateTime'] = X_channel.time_track(absolute_time=True, accuracy='ns')   #use X channel as reference
    PositionData['DateTime'] = PositionData['DateTime'].dt.tz_localize('Etc/UTC') #The time from time_track method gives the time at another time zone.
    PositionData['DateTime'] = PositionData['DateTime'].dt.tz_convert('Asia/Taipei') #Convert tht time zone to Taipei.
    PositionData['DateTime'] = PositionData['DateTime'].dt.tz_localize(None) #Remove the time zone information to simplify data.
    PositionData = PositionData[['Frame','DateTime','x-Center','y-Center']]
    return PositionData


folder = "X:\\20201113\\sample-2\\"
ExpFolder = "C:\\Users\\ULTRABBUG\\Downloads\\"
SampleName = '20201113-sample2'


SampleRate = 1000
Steps_N = 1000 #1000  #how many samples to analyze per loop.
FFT_Amp_threshold = 25
x_ratio = 1000/0.14  # unit: nm/Volt   # this value is estimated from 20190617
y_ratio = 1000/0.1  # unit: nm/Volt


for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*.tdms'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
Targetfile = np.sort(Targetfile)
N_Files = len(Targetfile)

#print(Targetfile)
'''****Sort the files order***************************'''
seq_list = np.zeros(N_Files,dtype=int)
for i in range(N_Files):
    tem_seq_str = Targetfile[i].split(".tdms")
    temp_seq_num = tem_seq_str[0].split("File")
    seq_num = int(temp_seq_num[1])
    #print(seq_num)
    seq_list[i]=seq_num
Sort_indices = np.argsort(seq_list)
Targetfile = Targetfile[Sort_indices]
print(Targetfile)
'''****************************************************'''
SpeedSheet = pd.DataFrame()
for i,filename in enumerate(Targetfile):
    filepath = folder+filename
    PositionData = Trans_PositionData(filepath)
    #print(i)
    TempSpeedSheet, Temp_FT = BA.GetSpeed(PositionData, Steps_N=Steps_N, Fps=SampleRate,FT_Output=True)
    SpeedSheet = SpeedSheet.append(TempSpeedSheet, ignore_index=True)

    '''********Here collect the FT data to plot the contour.****************'''

    if i == 0:
        FT_x = np.full([len(Temp_FT[0]), 1], TempSpeedSheet.loc[0, "DateTime"], dtype='datetime64[ns]')
        FT_y = np.reshape(Temp_FT[0],(len(Temp_FT[0]),1))
        FT_z = np.reshape(Temp_FT[1],(len(Temp_FT[1]),1))/np.max(Temp_FT[1])    #normalize each amplitude to 0-1
    else:
        temp_x = np.full([len(Temp_FT[0]), 1], TempSpeedSheet.loc[0, "DateTime"], dtype='datetime64[ns]')
        FT_x = np.append(FT_x, temp_x, axis=1)
        temp_y = np.reshape(Temp_FT[0],(len(Temp_FT[0]),1))
        FT_y = np.append(FT_y, temp_y, axis=1)
        temp_z = np.reshape(Temp_FT[1],(len(Temp_FT[1]),1))/np.max(Temp_FT[1])  #normalize each amplitude to 0-1
        FT_z = np.append(FT_z, temp_z, axis=1)

    '''*********************************************************************'''
    BA.progressBar(100 * i /N_Files, 100)
    #print(SpeedSheet)
FT_x = (FT_x - FT_x[0][0]).astype('timedelta64[s]') / np.timedelta64(1, 's')


SpeedSheet.loc[SpeedSheet["FFT_amp"] < FFT_Amp_threshold, "Speed(Hz)"] = 0  # Extract Speed and filter by a amplitube threshold
SpeedSheet.to_csv(ExpFolder + SampleName + '-Speed.csv',index=False)

'''***********************************This part plot and save the results********************************************'''
fig = plt.figure(constrained_layout=True,figsize=[19.20*0.8,10.80*0.8])#
gs = GridSpec(4, 2, figure=fig)
ax1_Speed = fig.add_subplot(gs[0, 0])
ax2_Ex = fig.add_subplot(gs[1,0],sharex = ax1_Speed)
ax3_Ey = fig.add_subplot(gs[2,0],sharex = ax1_Speed)
ax4_Radius = fig.add_subplot(gs[3,0],sharex = ax1_Speed)
ax5_FTamp = fig.add_subplot(gs[2,-1],sharex = ax1_Speed)
ax6_EQuality = fig.add_subplot(gs[3,-1],sharex = ax1_Speed)
ax7_FTmap = fig.add_subplot(gs[0:2,-1],sharex = ax1_Speed)
fig.suptitle(SampleName)

deltaT = (SpeedSheet["DateTime"]-SpeedSheet.loc[0,"DateTime"]).astype('timedelta64[s]')
ax1_Speed.scatter(deltaT, SpeedSheet["Speed(Hz)"],s=2,marker='.')
ax1_Speed.set_ylabel('Speed(Hz)')
ax1_Speed.set_ylim(0,150)
ax1_Speed.set_xlim(left=0,right=np.max(deltaT))


ax2_Ex.scatter(deltaT, SpeedSheet["e-x"],s=2,marker='.')
ax2_Ex.set_ylabel('E_fit_X(nm)')

ax3_Ey.scatter(deltaT, SpeedSheet["e-y"],s=2,marker='.')
ax3_Ey.set_ylabel('E_fit_Y(nm)')

ax4_Radius.scatter(deltaT, SpeedSheet["Radius"],s=2,marker='.')
ax4_Radius.set_ylabel('Radius(nm)')

ax5_FTamp.scatter(deltaT, SpeedSheet["FFT_amp"],s=2,marker='.')
ax5_FTamp.set_ylabel('FFT_amp')

ax6_EQuality.scatter(deltaT, SpeedSheet["e-FitQaulity"],s=2,marker='.')
ax6_EQuality.set_ylabel("e-FitQaulity")
ax6_EQuality.set_yscale('log')

pc=ax7_FTmap.pcolormesh(FT_x ,FT_y ,FT_z,cmap ="gnuplot2")
ax7_FTmap.set_ylim(150,-150)
fig.colorbar(pc)
ax7_FTmap.set_yticks([])
ax7_FTmap.set_ylabel("Frequency(Hz)")

axes_list = fig.axes
for i in range(len(axes_list)-1):
    axes_list[i].grid(which="both", axis='x', linestyle=':')

fig_2, ax_2 = plt.subplots(figsize=(12,6))
x_time = np.arange(0,len(SpeedSheet),1)/60
ax_2.plot(x_time,SpeedSheet["Speed(Hz)"],ls="None",marker='o',ms=2.5,color='k')
#ax_2.plot_date(SpeedSheet["DateTime"], SpeedSheet["Speed(Hz)"],markersize=1)
#ax = plt.gca()
#fig = plt.gcf()
#ax_2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#plt.title(SampleName)
ax_2.set_xlim(0,15.5)
ax_2.set_ylim(-1,120)
ax_2.set_xlabel('Time(minutes)',fontsize = 26)
ax_2.set_ylabel('Speed(Hz)',fontsize = 26)
ax_2.tick_params(axis='both', labelsize= 30,width = 1)
plt.tight_layout()

fig.savefig(ExpFolder + SampleName + '-Analysis.png')
fig_2.savefig(ExpFolder + SampleName + '-Speed.png')
#np.savez(ExpFolder + SampleName + '-FFTdata.npz',FT_x=FT_x,FT_y=FT_y,FT_z=FT_z)
plt.show()
plt.clf()
