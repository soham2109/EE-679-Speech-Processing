# initial package imports
import numpy as np
from scipy.signal import zpk2tf,freqz,sawtooth,square,impulse
from math import pi
from numpy import exp,zeros_like,cos,sin,log10,angle
from numpy import convolve as conv

# to make the plots more TEX-like
import matplotlib
matplotlib.use('PS')
import pylab as plt
plt.switch_backend('PS')

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode']=True
plt.style.use(['bmh'])
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams.update({"axes.facecolor" : "white",
                     "axes.edgecolor":  "black"})

# given data
f1 = 900 #formant frequency
b1 = 200 #bandwidth
fs = 16000 #sampling frequency
ts = 1.0/fs # sampling time

# calculation of poles and zeros from F1 and B1 and Coeffs
r = np.exp(-pi*b1*ts)
theta = 2*pi*f1*ts
poles = [r*exp(1j*theta) , r*exp(-1j*theta)]
zeros = zeros_like(poles)
b,a = zpk2tf(zeros,poles,k=1)

#### Frequency Response calculation ###
w,h = freqz(b,a)
plt.figure()
plt.subplot(1,2,1)
plt.plot(fs * w/(2*pi),20*log10(abs(h)),'b')
s="Frequency Response of Vocal Tract with F1: {} and B1: {}"
plt.suptitle(s.format(f1,b1),fontsize=12)
plt.title(r"Magnitude response",fontsize=12)
plt.ylabel(r"$|H(\Omega|$ (db)",fontsize=10)
plt.xlabel(r"$\Omega$")
plt.subplot(1,2,2)
angles = np.angle(h)
plt.plot(fs * w/(2*pi),angles,'b')
plt.title(r"Angle",fontsize=12)
plt.ylabel(r"Angle (rad)",fontsize=10)
plt.xlabel(r"$\Omega$",fontsize=10)
plt.subplots_adjust(left=0.125, 
                    wspace=0.4)
plt.savefig("../plots/Question1.png",bbox_inches="tight",pad=-1,format="png")

#### Impulse Response calculation ###
# forming the impulse input
pulse = np.zeros((200,1))
pulse[0] = 1

# initializing the impulse response
y = zeros_like(pulse)
time = np.linspace(0,len(pulse)*1.0/fs , 200, endpoint=False)

for n in range(len(pulse)):
    y[n] += b[0] * pulse[n]
    for k in range(1,len(a)):
        if (n-k)>=0:
            y[n] -= a[k] * y[n-k]

plt.figure()
plt.suptitle(r"Excitation Response",fontsize=12)
plt.subplot(1,2,1)
plt.plot(time,pulse,'b')
plt.title("Excitation Signal")
plt.ylabel(r"Amplitude",fontsize=10)
plt.xlabel(r"Time (sec)",fontsize=10)
plt.subplot(1,2,2)
plt.plot(time,y,'b')
plt.title("Impulse Response")
plt.ylabel(r"Amplitude",fontsize=10)
plt.xlabel(r"Time (sec)",fontsize=10)
plt.savefig("../plots/Question1 Impulse Response.png",bbox_inches="tight",pad=-1,format="png")
