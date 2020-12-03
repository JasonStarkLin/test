from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Ellipse
import Fun_BeadAssay as BA
import ellipses as el

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1,projection='3d')
ax1.set_aspect("equal")

# draw sphere
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax1.plot_surface(x, y, z, color="black",alpha=0.2)

# draw a circle

step = 2*np.pi/40
phi = np.arange(0,2*np.pi+step,step)
theta = np.zeros(len(phi))+np.pi/5
cir_x = np.cos(phi)*np.sin(theta)
cir_y = np.sin(phi)*np.sin(theta)
cir_z = np.cos(theta)




cir_x = np.reshape(cir_x,(len(cir_x),1))
cir_y = np.reshape(cir_y,(len(cir_y),1))
cir_z = np.reshape(cir_z,(len(cir_z),1))

#Rotate the circle
xAngle = 0
yAngle = 120 #90 have a problem (Become a line) #270 a line alignment would have problem. At some angle like 120 the correction angle have proble.
zAngle = 0
vector = np.append(cir_z,cir_y,axis=1)
vector = np.append(vector,cir_x,axis=1)
r = R.from_euler('zyx', [xAngle,yAngle,zAngle], degrees=True)
New_p = r.apply(vector)

ax1.plot(vector[:,2],vector[:,1],vector[:,0])

ax1.plot(New_p[:,2],New_p[:,1],New_p[:,0],color="y")
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')


#plot the projection
ax2=fig.add_subplot(1,3,2)
ax2.scatter(New_p[:,2],New_p[:,1],color="y",s=4)
ax2.set_aspect("equal")


#Fitt the ellipse and calculate the angle of rotation.

lsqe = el.LSqEllipse()
Data = [New_p[:,2], New_p[:,1]]
lsqe.fit(Data)
e_cen, width, height, phi = lsqe.parameters()
ellipse = Ellipse(xy=e_cen, width=2 * width, height=2 * height, angle=np.rad2deg(phi),
                  edgecolor='b', fc='None', lw=2, label='Fit', zorder=2, alpha=0.2)
ax2.add_patch(ellipse)
Ar = BA.GetAspectRatio(width,height)
theda_radian = np.arccos(Ar)
theda_degree = np.rad2deg(theda_radian)
print("open angle: ",theda_radian*180/np.pi)
print("Z rotation:",phi,np.rad2deg(phi))

##########Correct the orbit and plot the results##########
ax3=fig.add_subplot(1,3,3)
ax3.set_aspect("equal")

#Align the long axis angle
if width > height:
    AlignAngle = 90+np.rad2deg(phi)
    print("width > height", AlignAngle)
else:
    AlignAngle = np.rad2deg(phi)
    print("width <= height", AlignAngle)
r = R.from_euler('zyx', [0,0,AlignAngle], degrees=True)
Alin_p = r.apply(New_p)
ax3.scatter(Alin_p[:,2],Alin_p[:,1],color="y",s=4)

#orbit correction
Correct_angle = -1*theda_radian*180/np.pi ## the correction angle have bug.
r = R.from_euler('zyx', [0,Correct_angle,0], degrees=True)
Cor_p = r.apply(Alin_p)
ax3.scatter(Cor_p[:,2],Cor_p[:,1],color="b",s=4)
###########################################################


plt.tight_layout()
plt.show()