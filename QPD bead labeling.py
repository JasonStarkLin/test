"""
This program search the ROI.zip in buffer exchange data. And use the minima distance to find the trap position as the QPD bead.
"""
import pandas as pd
import os
from read_roi import read_roi_zip
import time


folder = "X:\\Analyzed Data\\YS1294-BufferExchange\\" #global
ExpListName = "BE exp List.xlsx"
Trap_XY = {'x':923.326,'y':175.799}

def roi_toSheet(roi_path):
    roi = read_roi_zip(roi_path)
    roi_list = [[roi[i]["name"],roi[i]['x'][0],roi[i]['y'][0]] for i in roi ]
    return pd.DataFrame(roi_list,columns=["name","x","y"])

ExpListSheet = pd.read_excel(folder+ExpListName)
ExpListSheet=ExpListSheet.set_index("SlideSample",drop=False)
#exclude those do not have QPD data.
judge = pd.to_datetime(ExpListSheet["Date"])> pd.to_datetime("2019-10-17")
ExpListSheet = ExpListSheet[judge]

bead_list =[]
error_list =[]
Ini_time = time.time()
for i in ExpListSheet.index:
    Slidefolder = folder + ExpListSheet.loc[i, "Strain"] + "\\" + ExpListSheet.loc[i, "Exp"] + "\\" + i + "\\"
    temp_time = time.time()
    #Find bead ROI labeling.
    for root, dirs, files in os.walk(Slidefolder):
        for file in files:
            if file.endswith('-ROI.zip'):  # py為結尾
                #print(os.path.join(root, file))
                temp_ROI_path = os.path.join(root, file)
                #Read ROI.
                temp_ROI_sheet = roi_toSheet(temp_ROI_path)
                #find the trap bead by distance minima
                d = (temp_ROI_sheet['x'] - Trap_XY['x']) ** 2 + (temp_ROI_sheet['y'] - Trap_XY['y']) ** 2   #distance
                temp_judge = d == min(d)
                bead = temp_ROI_sheet[temp_judge]
                if len(bead)>1 : error_list.append(temp_ROI_path) #incase there exist two minima.
                sample = str.split(file,'-ROI.zip')[0]+'-'+bead.iloc[0,0]
                bead_list.append([i,sample])
    print(Slidefolder,"    Cost time: ", (time.time() - temp_time),'s')

if len(error_list)>=1:
    print("QPD bead more than one. Plz check.")
    print(error_list)

bead_sheet = pd.DataFrame(bead_list)
bead_sheet.to_csv("C:\\Users\\ULTRABBUG\\Downloads\\QPD-bead connect.csv")

print("Totally Cost time: ", (time.time() - Ini_time),'s')
print("Program finished")