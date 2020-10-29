#!/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile as wav
from matplotlib import patches
from collections import defaultdict

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


def read_file(filename="../wavfiles/aa.wav",verbose=False):
    input_wav = wav.read(filename)
    sound = input_wav[1]
    Fs = input_wav[0]
    if verbose:
        plt.figure()
        plt.title("Original Sound Waveform")
        plt.plot(sound)
        plt.grid(color='0.9', linestyle='-')
        plt.tight_layout()
        plt.xlim(xmin=0)
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.savefig("../plots/OriginalSound.png",bbox_inches="tight")

    return sound,Fs



def pre_emphasize(sound,alpha=0.95,verbose=False):
    alpha_pre_emphasis = alpha
    pre_emphasis = np.zeros_like(sound)
    pre_emphasis[0] = sound[0]
    for i in range(1,len(pre_emphasis)):
        pre_emphasis[i] = sound[i] - alpha_pre_emphasis*sound[i-1]
    
    if verbose:
        plt.figure()
        plt.title("Pre-Emphasized Sound Waveform")
        plt.plot(pre_emphasis)
        plt.grid(color='0.9', linestyle='-')
        plt.tight_layout()
        plt.xlim(xmin=0)
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.savefig("../plots/PreEmphasizedSound.png",bbox_inches="tight")
    
    return pre_emphasis


def hamming_window(sound,duration=30,center=True,verbose=True):
    
    window_duration = int((duration/1000)*Fs)
    if center:
        center = len(sound)//2
        windowed_sound = sound[center-window_duration//2:center+window_duration//2]
        xticks = np.linspace(center-window_duration//2,center+window_duration//2,window_duration)
    else:
        windowed_sound = sound[:window_duration]

    #print(windowed_sound.shape)
    #print(window_duration)

    hamming_output = np.hamming(window_duration)*windowed_sound

    # Magnitude Response
    w,h_window = signal.freqz(hamming_output)
    plt.figure()
    plt.plot((Fs*w/(2*np.pi))/1000,20*np.log10(h_window))
    plt.grid(color='0.9', linestyle='-')
    plt.title("Magnitude Response of Hamming Output")
    plt.xlabel("Frequency (KHz)")
    plt.ylabel(r"Magnitude $|H(\omega)|$ (dB)")
    plt.tight_layout()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig("../plots/MagnitudeResponseHamming_"+str(duration)+"ms.png",bbox_inches="tight")

    if verbose:
        plt.figure()
        if center:
            plt.plot(xticks,hamming_output)
            plt.xlim(xmin=xticks[0])
        else:
            plt.plot(hamming_output)
            plt.xlim(xmin=0)
        plt.grid(color='0.9', linestyle='-')
        plt.title("Hamming Window")
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.savefig("../plots/HammingOut.png",bbox_inches="tight")

    return hamming_output



def LPanalysis(signal, p):	
    #p = LP order

    R = np.correlate(signal,signal, mode = 'full')	#Autocorrelation
    R = R[-(len(signal)):]	#Keep autocorrelation values for positive values of i in summation(x[n]x[n-i])

    #Levinson Algorithm
    E = np.zeros(p+1)	#Vector to store error values
    a = np.zeros((p+1,p+1))
    G = np.zeros(p+1)
    E[0] = R[0]	#Initial Condition    
    for i in range(1, p+1):		# 1 <= i <= p
        if i==1:
            k = R[1]/E[0]
            a[1][1] = k
            E[1] = (1-k**2)*E[0]
            a[1][0] = 1
            G[1] = np.sqrt(E[1])
        else:
        #sum_{j=1}^{i-1} \alpha_j^{i-1}*r[i-j] calculation
            temp = 0
            for j in range(1, i):	# 1 <= j <= i-1
                temp += a[i-1][j] * R[i-j]

            k = (R[i] - temp)/E[i-1]
            a[i][i] = k
            
            for j in range(1, i):	# 1<=j<=i-1
                a[i][j] = a[i-1][j] - k * a[i-1][i-j]

            E[i] = (1 - k**2) * E[i-1]
            
            G[i] = np.sqrt(E[i])
            a[i][0] = 1

    return(E, G, a)

def plot_error_signal_energy(E,p=10):
    x = [i for i in range(p+1)]
    plt.figure()
    plt.plot(x,10*np.log10(E),marker="*")
    plt.xlabel("Number of Poles used in LP Analysis")
    plt.ylabel("Error Signal Energy (dB)")
    plt.title("Error Signal Energy vs Poles")
    plt.grid(color='0.9', linestyle='-')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig("../plots/ErrorSignalEnergy_vs_numPoles.png",bbox_inches="tight")




def zplane(ax, z, p, filename=None):
    """Plot the complex z-plane given zeros and poles.
    """

    # Add unit circle and zero axes    
    unit_circle = patches.Circle((0,0), radius=1, fill=False,
                                 color='black', ls='solid', alpha=0.5)
    ax.add_patch(unit_circle)
    plt.axvline(0, color='0.7')
    plt.axhline(0, color='0.7')
    
    # Plot the poles and set marker properties
    poles = plt.plot(p.real, p.imag, 'x', markersize=9)
    
    # Plot the zeros and set marker properties
    zeros = plt.plot(z.real, z.imag,  'o', markersize=9, 
             color='none',
             markeredgecolor=poles[0].get_color(), # same color as poles
             )

    # Scale axes to fit
    r = 1.5 * np.amax(np.concatenate((abs(z), abs(p), [1])))
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])

    """
    If there are multiple poles or zeros at the same point, put a 
    superscript next to them.
    TODO: can this be made to self-update when zoomed?
    """
    # Finding duplicates by same pixel coordinates (hacky for now):
    poles_xy = ax.transData.transform(np.vstack(poles[0].get_data()).T)
    zeros_xy = ax.transData.transform(np.vstack(zeros[0].get_data()).T)    

    # dict keys should be ints for matching, but coords should be floats for 
    # keeping location of text accurate while zooming

    # TODO make less hacky, reduce duplication of code
    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in poles_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.items():
        if value > 1:
            x, y = ax.transData.inverted().transform(coords[key])
            plt.text(x, y, 
                        r' ${}^{' + str(value) + '}$',
                        fontsize=13,
                        )

    d = defaultdict(int)
    coords = defaultdict(tuple)
    for xy in zeros_xy:
        key = tuple(np.rint(xy).astype('int'))
        d[key] += 1
        coords[key] = xy
    for key, value in d.items():
        if value > 1:
            x, y = ax.transData.inverted().transform(coords[key])
            plt.text(x, y, 
                        r' ${}^{' + str(value) + '}$',
                        fontsize=13,
                        )

def plot_poles_and_zeros(req,a,G):
    for p_ in req:
        poles = [a[p_][0],*(-a[p_][1:p_+1])]
        gain = G[p_]
        z,p,k = signal.tf2zpk(gain,poles)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        zplane(ax, z, p)
        plt.grid(True, color='0.9', linestyle='-', which='both', axis='both')
        plt.title('Poles and zeros for p='+str(p_))
        plt.savefig("../plots/PoleZeroPlot_p_"+str(p_)+".png",bbox_inches="tight")
        plt.clf()



def plot_LPC_Spectrum(a,G,Fs,h_w,p=[1,2,4,6,8,10]):
    n = len(p)
    plt.figure(figsize=(20,10))
    for i in range(n):
        poles = [a[p[i]][0],*(-a[p[i]][1:p[i]+1])]
        w,h = signal.freqz(G[p[i]],poles)
        w_ham,h_ham = signal.freqz(h_w)
        plt.suptitle("")
        plt.subplot(2,3,i+1)
        plt.plot((Fs*w/(2*np.pi)),20*np.log10(abs(h)),label="Estimated Spectrum")
        plt.plot((Fs*w_ham/(2*np.pi)),20*np.log10(abs(h_ham)),"r",linestyle='dashed',alpha=0.5,label="Windowed Spectrum")
        plt.title("LPC Spectrum for p = {}".format(p[i]))
        plt.grid(color='0.9', linestyle='-')
        plt.xlim(xmin=-5)
        plt.legend(loc="best")
        plt.xlabel("Frequency (KHz)")
        plt.ylabel(r"Magnitude $|H(\omega)|$ (dB)")
        plt.autoscale(enable=True, axis='x', tight=True)

    plt.savefig("../plots/LPCSpectrum.png",bbox_inches="tight")



def autocorrelate(gain,poles,segment_signal,duration=30,verbose=True,center=True):
    
    window_duration = int((duration/1000)*Fs)

    inverse_filter = np.zeros_like(segment_signal)
    for i in range(segment_signal.shape[0]):
        inverse_filter[i] = segment_signal[i]
        for j in range(len(poles)):
            if (i-j)>=0:
                inverse_filter[i] -= poles[j]*segment_signal[i-j]
        inverse_filter[i] /= gain

    if verbose:
        plt.figure()
        plt.plot(inverse_filter)
        plt.title("Residual Signal")
        plt.grid(color='0.9', linestyle='-')
        plt.xlabel("Time")
        plt.ylabel("Residual Amplitude")
        plt.tight_layout()
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.savefig("../plots/ResidualPlot.png",bbox_inches="tight")


    
    autocorrelate = np.correlate(inverse_filter,inverse_filter,mode="same")

    signal_autocorrelate = np.correlate(segment_signal,segment_signal,mode="same")

    maxima = np.argmax(autocorrelate)
    second_maxima = np.argmax(autocorrelate[autocorrelate<0.7*np.max(autocorrelate)])
    
    print("Index of maxima",maxima)
    print("Index of second maxima",second_maxima)

    F0 = (Fs/(maxima - second_maxima))
    print("F0 detected from ACF : ",F0,"Hz")
    
    if center:
        xticks = np.linspace(center-window_duration//2,center+window_duration//2,window_duration)
    
    plt.figure()
    plt.subplot(121)
    plt.title("ACF of Residual")
    if center:
        plt.plot(xticks,autocorrelate)
    else:
        plt.plot(autocorrelate)
    plt.grid(color='0.9', linestyle='-')
    plt.xlabel("Time")
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()

    plt.subplot(122)
    plt.title("ACF of Original Sound Segment")
    if center:
        plt.plot(xticks,signal_autocorrelate,"r",alpha=0.7)
    else:
        plt.plot(signal_autocorrelate,"r",alpha=0.7)
    plt.grid(color='0.9', linestyle='-')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    plt.savefig("../plots/ComparisonBetweenAutocorrelation.png",bbox_inches="tight")


    return F0



# Bonus
def reconstruct(gain,poles,F0,Fs,p=10,duration=300):
    period = F0/1000
    total = int((duration/1000)*Fs)
    t = np.linspace(0,duration,total)
    impulse_train = (signal.square(2 * np.pi *period * t,duty=0.08)+1)/2
    plt.figure()
    plt.title("Input Impulse Train")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.plot(t[:1000], impulse_train[:1000])
    plt.grid(color='0.9', linestyle='-')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig("../plots/ReconstructionImpulse(segment).png",bbox_inches="tight")

    output_signal = np.zeros_like(impulse_train)
    for i in range(len(impulse_train)):
        output_signal[i] = gain*impulse_train[i]
        for j in range(len(poles)):
            if (i-j)>=0:
                output_signal[i] += poles[j]*output_signal[i-j]

    plt.figure()
    plt.plot(t,output_signal)
    plt.grid(color='0.9', linestyle='-')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("Reconstructed Signal using p="+str(p))
    plt.savefig("../plots/ReconstructedSignal_p"+str(p)+".png",bbox_inches="tight")

    wav.write("../wavfiles/Reconstructed_Wave_p"+str(p)+".wav",8000,output_signal)
    return output_signal



if __name__ == '__main__':
	filename="../wavfiles/aa.wav"
	sound,Fs = read_file(filename,verbose=True)
	# Question 1 : PRE-EMPHASIS USING ALPHA=0.95
	pre_emphasis = pre_emphasize(sound,verbose=True)
	hamming_output = hamming_window(pre_emphasis)
	E,G,a = LPanalysis(hamming_output,10)
	plot_error_signal_energy(E)
	plot_poles_and_zeros([6,10],a,G)
	plot_LPC_Spectrum(a,G,Fs,hamming_output)
	F0 = autocorrelate(G[10],a[10],hamming_output)
	reconstructed_signal = reconstruct(G[10],a[10],F0,Fs)


