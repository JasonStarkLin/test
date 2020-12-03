import fnmatch
import os
import shutil
import numpy as np
import pims
import matplotlib.pyplot as plt

from skimage.segmentation import clear_border
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import Fun_BeadAssay as BA
import time

import numpy as np
import pims
from skimage.io import imsave
import sys
import pandas as pd

import matplotlib.pyplot as plt




from matplotlib.patches import Ellipse

def DrawTrace(Data,Speed_Sheet,FT,filepath,title):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(Data[0], Data[1], 'ro', label='test data', zorder=1, markersize=1)
    e_cen = [Speed_Sheet.loc[0,"e-x"],Speed_Sheet.loc[0,"e-y"]]
    width = Speed_Sheet.loc[0,"e-major"]
    height = Speed_Sheet.loc[0,"e-minor"]
    phi = Speed_Sheet.loc[0,"e-angle"]
    ellipse = Ellipse(xy=e_cen, width=2 * width, height=2 * height, angle=np.rad2deg(phi),
                      edgecolor='b', fc='None', lw=2, label='Fit', zorder=2)
    ax[0].add_patch(ellipse)
    ax[0].set_title(title)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xlim((0,12))
    ax[0].set_ylim((0,12))


    ax[1].scatter(FT[0],FT[1],s=1,color='k')
    ax[1].set_xlabel("Speed(Hz)")
    ax[1].set_title("FFT results")
    #plt.tight_layout()
    max_ind = np.where(FT[1]==np.max(FT[1]))
    max_amp = FT[1][max_ind[0]][0]
    max_F = FT[0][max_ind[0]][0]
    s = "["+"{:.0f}".format(max_F)+","+"{:.1f}".format(max_amp)+"]"
    ax[1].text(max_F,max_amp,s,color='c',horizontalalignment="left",verticalalignment="top")

    #plt.draw()
    fig.savefig(filepath)
    plt.close()

def ExFolderConstr(Con_folder):
    if os.path.exists(Con_folder):
        print("Analysis Exporting folder have been existed")
        print("Deleted the Analysis exporting folder and Created a new one")
        shutil.rmtree(Con_folder)
        os.mkdir(Con_folder)

    else:
        print("Analysis Exporting folder is not exist")
        os.mkdir(Con_folder)
        print("Created the analysis folder")

def GetBeadsPosition(folder,Targetfile, window_size, bead_xy, **kwargs):
    '''

    Parameters
    ----------
    folder : string
        The destination for the files position.
    Targetfile : list
        The list includes all the name of the files which intend to calculate the position. The filename older in the list is the iterate sequence.
    window_size : int or float
        The number that indicate how large the extended size to crop the images to calculate the position.
    bead_xy : 1d-array.
        A numpy array included 2D central posiiton of the target. (x,y)
    kwargs

    Returns
    -------
    2d array
        The beads informations. Including:
        frame_count : the position of the slice.
        Time : The time data for that position.
        x: x position in the Image.
        y: y position in the Image.

    '''

    for i in range(len(Targetfile)):
        Image_info = pims.open(folder + Targetfile[i])
        if "N_Frames" not in locals():
            N_Frames = len(Image_info)
        else:
            N_Frames = N_Frames + len(Image_info)

    Crop_images = np.zeros((N_Frames, 2 * window_size + 1, 2 * window_size + 1), dtype=np.uint8)
    frame_count = 0
    # Run for all the files
    # len(Targetfile)
    #for i in range(0, 1):
    # Import the transfer function for the beads position calibration.
    if "TranFun" in kwargs:
        TranFun = kwargs["TranFun"]
    for i in range(0,len(Targetfile)):
        Image = pims.open(folder + Targetfile[i])
        #O_Stage = Targetfile[i].rsplit('-')[-2]
        print("     File: ", Targetfile[i])
        #print("     Frame Shape", Image._shape)
        print("     Image type: ", Image.pixel_type)
        print("     Frame Counts: ", len(Image))
        #print("     Operation Stage: ", O_Stage)
        # Run for the frames
        #for j in range(0, 450):
        for j in range(len(Image)):
            #if frame_count<41400 :
                #frame_count = frame_count + 1
                #continue
            #if j != 3339 : continue
            progressBar(100 * frame_count / N_Frames, 100)

            if "TranFun" in kwargs:
                #Timepoint = (frame_count//450)-1  # Because the imagej did not record transfer function for the first frame.
                Timepoint = frame_count  # Correct for each frame. The Tranfer function is not the same as Imagej. Need to be careful
                if Timepoint >=0:
                    Crop_cent_x = np.int(bead_xy[0] - TranFun.iloc[Timepoint,2])
                    Crop_cent_y = np.int(bead_xy[1] - TranFun.iloc[Timepoint,3])
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
            Cent = BA.SpotCentral(temp, fun='Centroid')

            if 'Bead_P' in locals():
                Bead_P = np.append(Bead_P, np.array([[frame_count,0, Cent[0], Cent[1]]]), axis=0)
            else:
                Bead_P = np.array([[frame_count,0, Cent[0], Cent[1]]])

            # For saving the images.
            Cx = int(np.round(Cent[0]))
            Cy = int(np.round(Cent[1]))
            Crop_images[frame_count, Cy, Cx] = 255
            frame_count = frame_count + 1
        print("")
    print('     Totally frames number : ', frame_count)
    print("")
    #save the images
    if kwargs["ImageSave"] is True:
        Split_FileName = Targetfile[0].rsplit('-', 2)
        Sample = Split_FileName[0]
        imsave(ExpFolder + Sample + '-Fitted' + kwargs['Bead_label'] + '.tif', Crop_images[0:frame_count, :, :], photometric='minisblack')
    return Bead_P

def progressBar(value, endvalue,ProText = "Progress:", bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r      "+ProText+" [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

'''
folder = "X:\\Raw Data\\2020\\20201119\\Speed Distribution\\"
SampleName = '20201119-BE-Ibidi-850585'
#folder = "D:\\NAS-TEMP_BE\\20200926\\Speed distribution\\"
#SampleName = 'test'

FPS = 451
N_frame = 450 # for one analysis section.
window_size = 6 #my setting is 6
'''

folder = "C:\\Users\\ULTRABBUG\\Downloads\\Pos2\\"
SampleName = 'P02'
#folder = "D:\\NAS-TEMP_BE\\20200926\\Speed distribution\\"
#SampleName = 'test'

FPS = 498
N_frame = 500 # for one analysis section.
window_size = 15 #my setting is 6

temp_time = time.time()
#Searching files.
for file in os.listdir(folder):
    if fnmatch.fnmatch(file,'*.tif'):
        #print(file)
        if 'Targetfile' in locals():
            Targetfile=np.append(Targetfile,file)
        else:
            Targetfile=[file]
print(Targetfile)




#Check and Creating the exporting file folder
FigFolder = folder+SampleName+"-OrbitResults\\"
B_Label_Folder = folder+SampleName+"-Bead_Label\\"


ExFolderConstr(FigFolder)
ExFolderConstr(B_Label_Folder)

Bead_sheet = pd.DataFrame(columns=["Label","DateTime","X","Y","Speed(Hz)","FFT_amp","e-x","e-y","e-major",
                                   "e-minor","e-angle","e-FitQaulity","Eccentricity","AspectRatio","Opening angle(dgree)","Radius"])
for num,i in enumerate(Targetfile):

    #if num != 5:
        #continue

    Image = pims.open(folder+i)
    Split_FileName = i.rsplit('.seq')
    Sample = Split_FileName[0]
    print("File: ",num, i)
    print("     Frame Shape", Image.frame_shape)
    print("     Image type: ", Image.pixel_type)
    print("     Frame Counts: ", 500)

    ImageSTD = np.std(Image[:], axis=0)
    ThreImage = ImageSTD > np.mean(ImageSTD) + 7*np.std(ImageSTD)
    CloImage = closing(ThreImage,square(6))
    BI = clear_border(CloImage)
    Label_image = sk_label(BI)
    Image_label_overlay = label2rgb(Label_image, image=ThreImage)


    fig,ax = plt.subplots(figsize=(20.48,3.5))
    ax.imshow(Image[num],cmap="gray")
    for label,region in enumerate(regionprops(Label_image)):
        center = region.centroid
        ax.scatter(center[1],center[0],facecolors='none',edgecolor='g')
        Bead_Name = str("{:0>2d}".format(label))
        ax.text(center[1], center[0], Bead_Name,fontsize=8, ha="right", va="bottom", color='yellow')
    fig.savefig(B_Label_Folder+Sample + ".jpg")
    plt.close(fig)
    #plt.show()

    #continue

    for label,region in enumerate(regionprops(Label_image)):
        center = region.centroid
        #ax.scatter(center[1],center[0],facecolors='none',edgecolor='g')
        ROIName = '-Bead-'+str("{:0>2d}".format(label))
        KeyLabel = Sample+ROIName
        print("    KeyLabel: ", KeyLabel)
        bead_xy = np.array([np.int(center[1]), np.int(center[0])])
        if BA.EdgeTest(bead_xy,Image.frame_shape,Extensize= window_size):
            Bead_P = GetBeadsPosition(folder, [i], window_size, bead_xy, ImageSave=False, Bead_label=ROIName)
            Position_sheet = pd.DataFrame(Bead_P,columns=['Frame','DateTime', 'x-Center', 'y-Center'])
            Position_sheet = Position_sheet.infer_objects()
            xy_data = [Position_sheet["x-Center"].to_numpy(), Position_sheet["y-Center"].to_numpy()]
            Speed_Sheet,FT_Data = BA.GetSpeed(Position_sheet, Steps_N=N_frame, Fps=FPS,FT_Output=True)
            #Plot orbit information
            title = "{:.1f}".format(Speed_Sheet.loc[0,"Speed(Hz)"]) + "Hz"\
                    +", AR:" + "{:.1f}".format(Speed_Sheet.loc[0,"AspectRatio"]) \
                    +", Eccen:"+"{:.1f}".format(Speed_Sheet.loc[0,"Eccentricity"])\
                    +"\n, FT_amp:"+"{:.2f}".format(Speed_Sheet.loc[0,"FFT_amp"]) \
                    +", Ell_Q:"+"{:.1f}".format(Speed_Sheet.loc[0,"e-FitQaulity"])
            filepath = FigFolder + KeyLabel + ".jpg"
            #DrawTrace function need to modify.
            DrawTrace(xy_data,Speed_Sheet,FT_Data,filepath,title)  # If results contain not only one speed. The results would be wrong. Need to upgrade.
            #Try to improve the code for temp_Sheet if possible.
            temp_Sheet = pd.Series({"Label": KeyLabel,
                                    "DateTime":Speed_Sheet.loc[0,"DateTime"],
                                    "X": center[1], "Y": center[0],
                                    "Speed(Hz)": Speed_Sheet.loc[0,"Speed(Hz)"],
                                    "FFT_amp": Speed_Sheet.loc[0,"FFT_amp"],
                                    "e-x":Speed_Sheet.loc[0,"e-x"],
                                    "e-y":Speed_Sheet.loc[0,"e-y"],
                                    "e-major":Speed_Sheet.loc[0,"e-major"],
                                    "e-minor":Speed_Sheet.loc[0,"e-minor"],
                                    "e-angle":Speed_Sheet.loc[0,"e-angle"],
                                    "e-FitQaulity":Speed_Sheet.loc[0,"e-FitQaulity"],
                                    "Eccentricity":Speed_Sheet.loc[0,"Eccentricity"],
                                    "AspectRatio":Speed_Sheet.loc[0,"AspectRatio"],
                                    "Opening angle(dgree)": Speed_Sheet.loc[0,"Opening angle(dgree)"],
                                    "Radius":Speed_Sheet.loc[0,"Radius"]
                                    })
            Bead_sheet = Bead_sheet.append(temp_Sheet, ignore_index=True)


Bead_sheet = Bead_sheet.set_index("Label")
ExpBeadFileName = folder + SampleName +'.csv'
Bead_sheet.to_csv(ExpBeadFileName)
print("Cost Time: ",(time.time()-temp_time)/60," Minutes")