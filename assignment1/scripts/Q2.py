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
                     
# initial package imports
import numpy as np
from scipy.signal import zpk2tf,freqz,sawtooth,square,impulse
from math import pi
from numpy import exp,zeros_like,cos,sin,log10,angle
from numpy import convolve as conv
from scipy.io.wavfile import write

# input data
f1 = 900 #formant frequency
b1 = 200 #bandwidth
f_sampling = 16000
f_signal = 140
time = 0.5
t_sampling = 1/f_sampling
num_samples = int(f_sampling*time)

r = np.exp(-pi*b1*ts)
theta = 2*pi*f1*ts
poles = [r*exp(1j*theta) , r*exp(-1j*theta)]
zeros = 0
b,a = zpk2tf(zeros,poles,k=1)

# Excitation signal formation
t = np.linspace(0,time,num_samples)
# sawtooth approximation using square
sig = square(2 * pi * f_signal* t, duty=0.01)+1

plt.figure()
plt.plot(t[:1000],sig[:1000])
plt.xlabel("$Time (sec)$",fontsize=10)
plt.ylabel("$Amplitude$",fontsize=10)
plt.title("Approximated Triangular Pulses")
plt.savefig("../plots/Question2 Triangular Impulses.png",bbox_inches="tight",pad=-1,format="png")

#Calculation Excitation response
y = zeros_like(sig)
# difference equation
for n in range(len(sig)):
    for k in range(len(b)):
            if (n-k)>=0:
                y[n] += b[k] * sig[n-k]
    for k in range(1,len(b)):
        if (n-k)>=0:
            y[n] += b[k] * sig[n-k]
    for k in range(1,len(a)):
        if (n-k)>=0:
            y[n] -= a[k] * y[n-k]

#plotting the excitation response
plt.figure()
plt.title("Excitation Response",fontsize=12)
plt.plot(t[:2514],y[:2514],'b')
plt.ylabel("Amplitude",fontsize=10)
plt.xlabel("Time (sec)",fontsize=10)
plt.savefig("../plots/Question2 Response.png",bbox_inches="tight",pad=-1)

# saving the wav file
write("../wavfiles/Q2output"+"_".join([str(f_signal),str(f1),str(b1)])+".wav",f_sampling,y)
