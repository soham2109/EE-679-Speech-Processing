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

def generate_signal_response(t,sig,b,a):
    """
    Generates the excitation signal response from the signal and filter coefficients using the difference equation
    inputs: t (time-vector of the excitation signal)
            sig (excitation signal)
            b,a (filter numerator and denominator coefficients)
    output: returns the filter excitation response
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

def plot_magnitude_response(b,a,vowel,f0):
    """
    Plots the magnitude and phase response of the filter using the numerator and denominator
    coefficients of the filter.
    inputs: b,a (filter numerator and denominator coefficients)
            vowel (the vowel parameters being used)
            f0 (excitation signal frequency)
    outputs: None (saves the magnitude and frequency response)
    """
    w,h = freqz(b,a)
    plt.figure()
    s = "Vocal tract response for vowel: /'{}'/ with signal freq: {}Hz"
    plt.suptitle(s.format(vowel,f0) ,fontsize=12,weight=2)
    plt.subplot(1,2,1)
    plt.plot(fs * w/(2*pi),20*log10(abs(h)),'b')
    plt.title("Magnitude response",fontsize=12)
    plt.ylabel(r"$|H(\Omega|$",fontsize=10)
    plt.xlabel(r"$\Omega$")
    plt.subplot(1,2,2)
    angles = np.angle(h)
    plt.plot(fs * w/(2*pi),angles,'b')
    plt.title(r"Angle",fontsize=12)
    plt.ylabel(r"Angle (rad)",fontsize=10)
    plt.xlabel(r"$\Omega$",fontsize=10)
    plt.subplots_adjust(left=0.125,
                    wspace=0.4)
    plt.savefig("../plots/Q4_Freq_resp_"+vowel+"_"+str(f0)+".png",bbox_inches="tight",pad=-1,format="png")

def plot_and_save_waveform(t,y,f_signal,f_sampling,vowel):
    """
    Plots and saves the output of the filter excited with the signal upto a few pitch periods.
    inputs: t(time-vector of the excitation signal)
            y( output response of the filter)
            f_signal ( excitation signal frequency )
            f_sampling (sampling frequency)
            vowel (the vowel being coded)
    outputs: None
    """
    plt.figure()
    plt.title("Excitation",fontsize=12)
    plt.plot(t[:2514],y[:2514],'b')
    plt.ylabel("Impulse Response",fontsize=10)
    plt.xlabel("Time (sec)",fontsize=10)
    plt.savefig("../plots/Q4_Signal_Response"+str(f_signal)+"_"+vowel+".png",bbox_inches="tight",pad=-1,format="png")
    write("../wavfiles/output"+"_"+str(f_signal)+"_"+vowel+".wav",f_sampling,y)

def vocal_tract(formant_frequencies):
    """
    Given the formant frequencies calculates the numerator and denominator coefficients
    by convolving between the different formant frequencies
    inputs: formant_frequencies (list of the formant frequencies)
    outputs: numerator and denominator coefficients
    """
    global bw
    r = []
    theta = []
    for i in formant_frequencies:
        r.append(np.exp(-pi*bw*ts)) #radius in z-plane
        theta.append(2*pi*i*ts) #angle in z-plane

    denom_coeffs = []
    num_coeffs = []
    convolved_a = 1
    for radius,angle in zip(r,theta):
        poles = [radius*exp(1j*angle),radius*exp(-1j*angle)]
        zeros = zeros_like(poles)
        b,a = zpk2tf(zeros,poles,k=1)
        num_coeffs.append(b)
        denom_coeffs.append(a)
        convolved_a = conv(convolved_a,a)

    denom_coeffs = zeros_like(convolved_a)
    denom_coeffs[0] = 1
    
    return denom_coeffs,convolved_a

def generate_vowels(formant_frequencies,bandwidth,signal_frequency,vowel,time,f_sampling):
    ts = 1/f_sampling # sampling time
    num_samples = int(f_sampling*time) # total number of signal samples

    b,a = vocal_tract(formant_frequencies)
    plot_magnitude_response(b,a,vowel,signal_frequency)

    t = np.linspace(0,time,num_samples)

    # sawtooth approximation using square
    sig = square(2 * pi * signal_frequency* t, duty=0.01)+1

    response = generate_signal_response(t,sig,b,a)
    plot_and_save_waveform(t,response,signal_frequency,f_sampling,vowel)
    
f0 = [120,220]
f1 = [730,270,300]
f2 = [1090,2290,870]
f3 = [2440,3010,2240]
bw = 100
vow = ["a","i","u"]
duration = 0.5
fs = 16000 #sampling frequency
vowels = {}
for i in range(len(vow)):
    vowels[vow[i]] = {"formants":[f1[i],f2[i],f3[i]]}

for sig_freq in f0:
    for vowel in vowels:
        generate_vowels(vowels[vowel]["formants"],bw,sig_freq,vowel,duration,fs)