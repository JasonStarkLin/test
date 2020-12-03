import fnmatch
import os
import numpy as np
from nptdms import TdmsFile
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import Fun_BeadAssay as BA
import time
from scipy.spatial.transform import Rotation as Rot

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


folder = "X:\\Raw Data\\2020\\20200106\\Buffer exchange\\QPD-P02\\"
ExpFolder = "X:\\Raw Data\\2020\\20200106\\Buffer exchange\\QPD-P02\\"
SampleName = '20200106-BE-137-300530-T1-P02'


SampleRate = 1000
Steps_N = 1000  #how many samples to analyze per loop.
FFT_Amp_threshold = 50
x_ratio =  1000/0.14  # unit: nm/Volt   # this value is estimated from 20190617
y_ratio =  1000/0.1  # unit: nm/Volt

WindowSize = 3 #unit: seconds
smooth_Length = 0.03 #unit:seconds
#ShiftSize = 0.01 #unit: seconds


window_Np = int(SampleRate*WindowSize)
Smooth_Np = int(SampleRate*smooth_Length)

#if Shift_Np <= 0:
    #print("\nShiftSize is too small. The shift point is smaller than 1")
    #exit()


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

'''*****Combine the total data to one sheet************'''
temp_time = time.time()
print("Combing data")
for i,filename in enumerate(Targetfile):
    #if i > 50:
        #break
    filepath = folder+filename
    if i == 0:
        PositionData = Trans_PositionData(filepath)
    else:
        PositionData = PositionData.append(Trans_PositionData(filepath),ignore_index=True)
    BA.progressBar(100 * i /N_Files, 100)
    CostTime = (time.time() - temp_time) / 60
PositionData.to_csv(ExpFolder + SampleName + '-CombinePos.csv',index=False)
print("")
CostTime = (time.time()-temp_time)/60
print("     Combing data cost: ",CostTime," Minutes")
print("")
'''****************************************************'''

'''*****************Calculate the speed****************'''
temp_time = time.time()
print("Calculating the speed")
LoopSize = (len(PositionData)-1)//(window_Np-1)
for i in range(LoopSize):
    if i == 0 :
        Start_ind = 0
    else:
        Start_ind = End_ind-1
    End_ind = Start_ind + window_Np+1 #I should modify it. Because the dataframe count for the last index.
    #print("index(start,End) : ",Start_ind," , ", End_ind)
    Temp_Pdata = PositionData[Start_ind:End_ind].reset_index(drop=True)
    Temp_Pdata = BA.EllipseCorrection(Temp_Pdata)
    TempSpeedSheet = BA.GetSpeed_Ins(Temp_Pdata, Steps_N=window_Np, Fps=SampleRate)
    TempSpeedSheet = BA.SmoothSpeed(TempSpeedSheet,Smooth_Np)

    if i == 0:
        SpeedSheet=TempSpeedSheet[:]
    else:
        SpeedSheet = SpeedSheet.append(TempSpeedSheet,ignore_index=True)
    BA.progressBar(100 * i /LoopSize, 100)
print("")
CostTime = (time.time()-temp_time)/60
print("     Get speed cost: ",CostTime," Minutes")
print("")

'''****************************************************'''
SpeedSheet["Speed(Hz)"] = SpeedSheet["Speed(Hz)"]*(-1)
SpeedSheet.to_csv(ExpFolder + SampleName + '-Speed.csv',index=False)
plt.plot_date(SpeedSheet["DateTime"], SpeedSheet["Speed(Hz)"],markersize=1)
#plt.plot_date(SpeedSheet["DateTime"], SpeedSheet["Speed(Hz)"],fmt='-',linewidth=1)
ax = plt.gca()
fig = plt.gcf()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.title(SampleName)
plt.xlabel('Time')
plt.ylabel('Speed(Hz)')
plt.savefig(ExpFolder + SampleName + '-Speed.png')
plt.show()
plt.clf()

print("program finished")