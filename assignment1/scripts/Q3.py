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

def generate_signal_response(time,sig,b,a):
    """
    Given the signal, its duration, the filter numerator and denominator
    coefficients, this function calculates the excitation signal response
    of the filter, saves the plots in the plots directory and returns the 
    response.
    inputs: time (time duration of the signal)
            sig (the excitation signal to the filter)
            b,a : filter numerator and denominator coefficients
    outputs: filter response y
    """
    y = zeros_like(sig)
    # difference equation
    for n in range(len(sig)):
        for k in range(len(b)):
            if (n-k)>=0:
                y[n] += b[k] * sig[n-k]
        for k in range(1,len(a)):
            if (n-k)>=0:
                y[n] -= a[k] * y[n-k]
    return y
    
def plot_and_save_waveform(t,y,f_signal,f1,b1,f_sampling):
    """
    Plots and saves the output of the filter excited with the signal upto a few pitch periods.
    inputs: t(time-vector of the excitation signal)
            y( output response of the filter)
            f_signal ( excitation signal frequency )
            f1 (formant frequency of the filter)
            b1 (bandwidth of the filter)
            f_sampling (sampling frequency)
    outputs: None
    """
    plt.figure()
    plt.title(r"Excitation Response",fontsize=12)
    plt.plot(t[:2514],y[:2514],'b')
    plt.ylabel(r"Amplitude",fontsize=10)
    plt.xlabel(r"Time (sec)",fontsize=10)
    plt.savefig("../plots/Q3_Signal_Response"+str(f1)+"_"+str(b1)+".png",bbox_inches="tight",pad=-1,format="png")
    write("../wavfiles/output"+"_".join([str(f_signal),str(f1),str(b1)])+".wav",f_sampling,y)    
    
def plot_magnitude_response(b,a,f1,b1):
    """
    Plots the magnitude and phase response of the filter using the numerator and denominator
    coefficients of the filter.
    inputs: b,a (filter numerator and denominator coefficients)
            f1,b1 (formant frequency and bandwidth, used to save the figure only)
    outputs: None (saves the magnitude and frequency response)
    """
    # frequency response calculation
    w,h = freqz(b,a)
    plt.figure()
    s = "Frequency response of vocal tract with F1: {}Hz and B1: {}Hz"
    plt.suptitle(s.format(f1,b1),fontsize=12)
    plt.subplot(1,2,1)
    plt.plot(fs * w/(2*pi),20*log10(abs(h)),'b')
    plt.title(r"Magnitude response",fontsize=12)
    plt.ylabel(r"$|H(\Omega|$",fontsize=10)
    plt.xlabel(r"$\Omega$")
    plt.subplot(1,2,2)
    angles = np.angle(h)
    plt.plot(fs * w/(2*pi),angles,'b')
    plt.title(r"Angle",fontsize=12)
    plt.ylabel(r"Angle (rad)",fontsize=10)
    plt.xlabel(r"$\Omega$",fontsize=10)
    plt.subplots_adjust(left=0.125,
                    wspace=0.4, )
    plt.savefig("../plots/Q3_Freq_resp_"+str(f1)+"_"+str(b1)+".png",bbox_inches="tight",pad=-1,format="png")

def generate_waveform(f1,b1,f_signal,fs=16000):
    """
    Compiles all the support functions to produce the output
    inputs: f1 (first formant frequency of the filter)
            b1 (bandwidth around the first formant frequency)
            f_signal (excitation signal frequency)
            fs (sampling frequency)
    output: None
    """
    time = 0.5 # total time duration
    ts = 1/fs # sampling time
    num_samples = int(f_sampling*time) # total number of signal samples
    r = np.exp(-pi*b1*ts) #radius in z-plane
    theta = 2*pi*f1*ts #angle in z-plane

    poles = [r*exp(1j*theta) , r*exp(-1j*theta)] #poles : 2 for every formant
    zeros = zeros_like(poles) # zeros 
    
    b,a = zpk2tf(zeros,poles,k=1)
    plot_magnitude_response(b,a,f1,b1)
    t = np.linspace(0,time,num_samples)

    # sawtooth approximation using square
    sig = square(2 * pi * f_signal* t, duty=0.01)+1

    # 
    response = generate_signal_response(t,sig,b,a)
    plot_and_save_waveform(t,response,f_signal,f1,b1,fs)

formant_frequencies = [300, 1200, 300]
bandwidths= [100, 200, 100]
signal_frequencies = [120,120,180]

for i,j,k in list(zip(formant_frequencies,bandwidths,signal_frequencies)):
    generate_waveform(i,j,k)
