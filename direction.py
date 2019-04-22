import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtl
import matplotlib.colors as colors
#[theta,inches,mean,SE]
dB = np.array([[0,2,98.1,.1],[0,9,84.1,.4],[0,19.5,65.7,.3],[0,20,66.6,.1],[0,30,59.9,.1],[0,47,55.4,.1],[10,20,55.4,.2],[-10,20,58.4,.1],
      [10,30,55.2,.2],[-10,30,55.0,.2],[20,9,74.2,.2],[20,20,57.0,.1],[-20,20,55.3,.1],[20,30,55.0,.2],[-20,30,55.7,.3],[30,9,55.9,.1],
      [-30,10,57.8,.1],[30,20,57.6,.2],[30,30,55.2,.1],[-30,30,55.7,.1],[40,9,55.7,.3],[40,10,56.1,.2],[40,20,55.5,.1],[-40,20,55.7,.2],
      [40,30,55.5,.2],[-40,30,55.5,.1],[50,9,55.7,.2]])
BackGround = [54.4,.1]
def convertDatatoDir(db,BG):
    theta = db[0]
    inch = db[1]
    dB = db[2]
    SE = db[3]
    x,y = convert2coord(theta, inch)
    adjustedDB = 10*np.log10(10**(dB/10)-10**(BG/10))
    P = convert2Pressure(adjustedDB)
    PSE = convert2Pressure(SE)
    newData = [x,y,P,PSE]
    return newData

def convert2coord(deg,inch):
    meter = inch*0.0254
    theta = (np.pi*deg)/180
    x = meter*np.cos(theta)
    y = meter*np.sin(theta)
    return x,y

def convert2Pressure(dB):
    return 10**((dB-94)/20)

def pathag(x, y):
    return np.sqrt(x ** 2 + y ** 2)


# calculate theta based on coordinates(meters on all units)
def calc_theta(x, y):
    return np.arctan(y / x)


# calculate the 2nd derivative of envelope frequency
def deriv_P(w, t, dist):
    p0 = 1.225  # ambiant density kg/m^3
    c0 = 343.6  # sound velocity m/s
    return p0*np.sin(w * t + (w / c0) * dist)

# main directional equation
def dir(x, y, t, w1, w2):
    theta = calc_theta(x, y)
    R0 = pathag(x, y)
    DP2 = deriv_P(w2, t, R0)
    cross_section = 1  # this subject to change based on experiment
    P0 = 1  # this is subject to change based on experiment
    ws = w1 - w2;  # angular frequency
    c0 = 343.6  # sound velocity m/s
    p0 = 1.225  # ambient density kg/m^3
    alph = .16  # absorption coefficient: http://resource.npl.co.uk/acoustics/techguides/absorption/
    Ks = (w1 - w2) / c0  # wave number type thing
    P = ((ws ** 2) * (P0 ** 2) * cross_section) / (8 * np.pi * R0 * p0 * (c0 ** 4)) * (
                1 + .5 * p0 * (c0 ** (-2)) * DP2) * ((alph ** 2) + (Ks ** 2) * np.sin(.5 * theta) ** 4) ** (-1 / 2)
    return P
"""
mean = np.zeros(len(dB))
x = np.zeros(len(dB))
y = np.zeros(len(dB))
for i in range(len(dB)):
    new_data = convertDatatoDir(dB[i],BackGround[0])
    x[i] = new_data[0]
    y[i] = new_data[1]
    mean[i] = new_data[2]
plt.tricontourf(x,y,mean,norm=colors.LogNorm(vmin=mean.min(), vmax=mean.max()),cmap='rainbow')
print(mean.min(),mean.max())
plt.show()
#plt.pcolor(x, y, mean, cmap='rainbow')
#plt.show()
"""
w1 = 40000
w2 = 20000
t = np.linspace(0, 10, 10)
x = np.linspace(.01, 1, 10)
y = np.linspace(-.5, .5, 10)
xx, yy = np.meshgrid(x, y)
zz = 0
#average over different times
for i in range(len(t)):
    z = dir(xx, yy, t[i], w1, w2)
    if i == 0:
        print(xx)
        print(yy)
    zz += z
ave_z = zz / len(t)
#display mesh
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
print(ave_z.shape)
plt.pcolor(xx,yy,ave_z, norm=colors.LogNorm(vmin=ave_z.min(), vmax=ave_z.max()),
                   cmap='rainbow')
plt.colorbar(label='Predicted Amplitude(dB)/Source Amplitude(dB)')
plt.show()
print(ave_z.max())
print('done')