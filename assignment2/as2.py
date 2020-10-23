#EE 679 Speech Processing
#Computing Assignment 2: Linear predictive analysis and synthesis A
#170260009

from __future__ import division
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write
import scipy.io.wavfile as wav

B = 100
Fs = 8000
duration = 50/1000	#50 ms sound suration

def vowel(F0, F1, F2, F3):
	r = np.exp(-B*pi/Fs)    #Pole radius
	theta = [2*pi*F1/Fs, 2*pi*F2/Fs, 2*pi*F3/Fs]
	time = np.linspace(0, duration, duration*Fs)

	a1 = [1, -r*np.exp(-1j*theta[0])-r*np.exp(1j*theta[0]), r*r]
	a2 = [1, -r*np.exp(-1j*theta[1])-r*np.exp(1j*theta[1]), r*r]
	a3 = [1, -r*np.exp(-1j*theta[2])-r*np.exp(1j*theta[2]), r*r]

	x = np.zeros(int(duration*Fs))
	y = np.zeros(int(duration*Fs))

	temp1 = np.zeros(int(duration*Fs))
	temp2 = np.zeros(int(duration*Fs))

	for i in range(0, int(duration*F0)):
		x[i*int(np.floor(Fs/F0))] = 1
  
	temp1[0] = x[0]
	temp1[1] = x[1] - a1[1].real*temp1[0]	# a1[0] not written since a1[0] = 1
	for i in range(2, int(duration*Fs)):
			temp1[i] = x[i] - a1[1].real*temp1[i-1] - a1[2].real*temp1[i-2]
	
	temp2[0] = temp1[0]
	temp2[1] = temp1[1] - a2[1].real*temp2[0]	# a2[0] not written since a2[0] = 1
	for i in range(2, int(duration*Fs)):
			temp2[i] = temp1[i] - a2[1].real*temp2[i-1] - a2[2].real*temp2[i-2]
		
	y[0] = temp2[0]
	y[1] = temp2[1] - a3[1].real*y[0]	# a3[0] not written since a3[0] = 1
	for i in range(2, int(duration*Fs)):
			y[i] = temp2[i] - a3[1].real*y[i-1] - a3[2].real*y[i-2]

	w = np.linspace(0, 2*pi, Fs/2)
	z = np.exp(-1*1j*w)
	H1 = 1/(a1[0] + a1[1].real*z + a1[2]*z*z)
	H2 = 1/(a2[0] + a2[1].real*z + a2[2]*z*z)
	H3 = 1/(a3[0] + a3[1].real*z + a3[2]*z*z)
	H = H1*H2*H3

	return (y, H)

def LPanalysis(signal, p, windowLength):	#p = LP order
	#Windowing
	duration = signal.shape[-1]
	window = signal[(duration-int((windowLength)*Fs))//2:(duration+int((windowLength)*Fs))//2]*np.hamming(windowLength*Fs)
	R = np.correlate(signal,signal, mode = 'full')	#Autocorrelation
	R = R[-(len(signal)):len(R)]	#Keep autocorrelation values for positive values of i in summation(x[n]x[n-i])
    
	#Levinson Algorithm
	E = np.zeros(p+1)	#Vector to store error values
	a = np.zeros((p+1,p+1))
	G = np.zeros(p+1)
	E[0] = R[0]	#Initial Condition
	for i in range(1, p+1):		# 1 <= i <= p
		temp = 0
		for j in range(1, i):	# 1 <= j <= i-1
			temp = temp + a[i-1][j] * R[i-j]
		k = (R[i] - temp)/E[i-1]
		a[i][i] = k
		for j in range(1, i):	# 1<=j<=i-1
			a[i][j] = a[i-1][j] - k * a[i-1][i-j]
		E[i] = (1 - (k*k)) * E[i-1]
		G[i] = np.sqrt(E[i])
		a[i][0] = 1
	return(G, a)

a120, H120 = vowel(120, 730, 1090, 2440)
a220, H220 = vowel(220, 730, 1090, 2440)

windowLength = .03		#30 ms Hamming window
G1, a1 = LPanalysis(a120, 10, windowLength)
G2, a2 = LPanalysis(a220, 10, windowLength)
P = [2,4,6,8,10]

for p in P:
	print("Gain for /a/ of F0 = 120 Hz for LP order ",p," is ", G1[p])
	print("LP coefficients for the same are")
	print(a1[p])

	a1[p][1:len(a1[p])] = -a1[p][1:len(a1[p])]
	w1, h1 = signal.freqz(G1[p], a1[p], None, 1)
	fig = plt.figure()
	w = np.linspace(0, 2*pi, Fs/2)
	plt.plot(w*Fs/(2*pi), 20*np.log10(abs(H120)))
	plt.plot(w1*Fs/(2*pi), 20*np.log10(abs(h1)))

	fig.suptitle('Magnitude Response for order '+str(p)+' for 120 Hz')
	plt.ylabel('Magnitude in dB')
	plt.xlabel('Frequency (Hz)')
	plt.legend(['Original', 'Order '+str(p)])
	plt.show()

	print("Gain for /a/ of F0 = 300 Hz for LP order ",p," is ", G2[p])
	print("LP coefficients for the same are")
	print(a2[p])

	a2[p][1:len(a2[p])] = -a2[p][1:len(a2[p])]
	w2, h2 = signal.freqz(G2[p], a2[p], None, 1)
	fig = plt.figure()
	w = np.linspace(0, 2*pi, Fs/2)
	plt.plot(w*Fs/(2*pi), 20*np.log10(abs(H220)))
	plt.plot(w2*Fs/(2*pi), 20*np.log10(abs(h2)))
	fig.suptitle('Magnitude Response for order '+str(p)+ ' for 300 Hz')
	plt.ylabel('Magnitude in dB')
	plt.xlabel('Frequency (Hz)')
	plt.legend(['Original', 'Order '+str(p)])
	plt.show()
