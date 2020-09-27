# initial package imports
import numpy as np
from scipy.signal import zpk2tf,freqz,sawtooth,square,impulse
from scipy.fft import fft,fftfreq
from math import pi
from numpy import exp,zeros_like,cos,sin,log10,angle,hamming
from numpy import convolve as conv
#import matplotlib.pyplot as plt


# to view the plots
# comment all upto plt.rcParams.update
# and uncomment the plt import before this commenting started
# and in functions hamming and rectnagular
# uncomment the plt.show (commented) and comment plt.savefig() part
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

def hamming_window(win_length,fs,output_signal,vowel,f0):
    """
    """
    window_size = int(win_length*fs/1000)
    window_signal = output_signal[:window_size] * hamming(window_size)
    dft = fft(window_signal, n=1024)
    freq = fftfreq(dft.shape[-1], 1/fs)

    
    len_ = int(len(dft)/2)
    s = "Hamming Window response of window length: {}ms  for vowel: /'{}'/ with signal freq: {}Hz"
    plt.figure()
    plt.title(s.format(win_length,vowel,f0) ,fontsize=12,weight="bold")
    #plt.plot(freq[:len_],20*log10(abs(dft[:len_])),'b',linewidth=0.7)
    plt.plot(abs(freq),20*log10(abs(dft)),'b',linewidth=0.7)
    plt.ylabel(r"$|H(\Omega|$",fontsize=10)
    plt.xlabel(r"$\Omega$")
    plt.xlim(xmin=0)
    plt.grid("True")
    plt.tight_layout()
    #plt.show()
    plt.savefig("../plots/Hamming_Window_Freq_resp_"+vowel+"_"+str(f0)+"_"+str(win_length)+".png",bbox_inches="tight",pad=-1,format="png")
    
    
    
def rectangular_window(win_length,fs,output_signal,vowel,f0):
    """
    """
    window_size = int(win_length*fs/1000)
    window_signal = output_signal[:window_size]
    dft = fft(window_signal, n=1024)
    freq = fftfreq(dft.shape[-1], 1/fs)
    len_ = int(len(dft)/2)
    s = "Rectangular Window response of window length: {}ms for vowel: /'{}'/ with signal freq: {}Hz"
    plt.figure()
    plt.title(s.format(win_length,vowel,f0) ,fontsize=12,weight="bold")
    #plt.plot(freq[:len_],20*log10(abs(dft[:len_])),'b',linewidth=0.7)
    plt.plot(abs(freq),20*log10(abs(dft)),'b',linewidth=0.7)
    plt.ylabel(r"$|H(\Omega|$",fontsize=10)
    plt.xlabel(r"$\Omega$")
    plt.xlim(xmin=0)
    plt.grid("True")
    plt.tight_layout()
    #plt.show()
    plt.savefig("../plots/Rect_Window_Freq_resp_"+vowel+"_"+str(f0)+"_"+str(win_length)+".png",bbox_inches="tight",pad=-1,format="png")
    
def vocal_tract(formant_frequencies,f_sampling):
    """
    Given the formant frequencies calculates the numerator and denominator coefficients
    by convolving between the different formant frequencies
    inputs: formant_frequencies (list of the formant frequencies)
    outputs: numerator and denominator coefficients
    """
    global bw
    r = []
    theta = []
    ts = 1/f_sampling
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

def generate_vowels(formant_frequencies,bandwidth,signal_frequency,vowel,time,f_sampling,window,win_length):
    ts = 1/f_sampling # sampling time
    num_samples = int(f_sampling*time) # total number of signal samples
    b,a = vocal_tract(formant_frequencies,f_sampling)
    
    t = np.linspace(0,time,num_samples)
    # sawtooth approximation using square
    sig = square(2 * pi * signal_frequency* t, duty=0.01)+1

    response = generate_signal_response(t,sig,b,a)
    #plot_and_save_waveform(t,response,signal_frequency,f_sampling,vowel)
    if window=="hamming":
        hamming_window(win_length,f_sampling,response,vowel,signal_frequency)
    elif window=="rectangular":
        rectangular_window(win_length,f_sampling,response,vowel,signal_frequency)
        
f0 = [120,220]
formants = [300,870,2240]
vowel = "u"
duration = 0.5
fs = 16000
bw = 100

windows = ["hamming","rectangular"]
window_lengths = [5,10,20,40]

for sig_freq in f0:
    for window in windows:
        for win_len in window_lengths:
            generate_vowels(formants,bw,sig_freq,vowel,duration,fs,window,win_len)
