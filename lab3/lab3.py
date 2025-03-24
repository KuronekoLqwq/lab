import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
N = 8  # Number of cycles
f_clk = 2.4e9  # Clock frequency in Hz
Cs = 15.925e-12  # Sampling capacitor in Farads

# Transfer function for 1,2,3

# For 1
num1 = np.ones(N)
den1 = [1]

w, h = signal.freqz(num1, den1, fs=f_clk)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(w, 20 * np.log10(np.abs(h)+ 1e-10))
plt.title('Magnitude Response of 1')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)

plt.savefig('./images/1.png', dpi=300, bbox_inches='tight')

# For 2
Cr = 0.5e-12
a1 = Cs/(Cs+Cr)
num2 = [1]
den2 = [1, -a1]

w2 , h2 = signal.freqz(num2,den2, fs=f_clk/(2*N))

h_cascade = h*h2

# Cascaded response
# use w2 as the new system frequency
plt.figure(figsize=(10, 8))
plt.plot(w2, 20 * np.log10(np.abs(h_cascade) + 1e-10))
plt.title('Cascaded System Magnitude Response of 2')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.savefig('./images/2.png', dpi=300, bbox_inches='tight')

# 3a
L = 4 # Capacitors in each bank
num3a = np.ones(L)
den3a = [1]

w3a, h3a = signal.freqz(num3a, den3a, fs=f_clk/(2*N))

h3a_cascade = h3a * h_cascade

plt.figure(figsize=(10, 8))
plt.plot(w2, 20 * np.log10(np.abs(h3a_cascade) + 1e-10))
plt.title('Cascaded System Magnitude Response of 3a')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.savefig('./images/3a.png', dpi=300, bbox_inches='tight')

num3b = [1]
den3b = [1, -1]

w3b , h3b = signal.freqz(num3b,den3b, fs=f_clk/(2*N*L))
h3b_cascade = h3a_cascade * h3b

plt.figure(figsize=(10, 8))
plt.plot(w3b, 20 * np.log10(np.abs(h3b_cascade) + 1e-10))
plt.title('Cascaded System Magnitude Response of 3b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.savefig('./images/3b.png', dpi=300, bbox_inches='tight')

# 3c

# Example usage:
# Calculate a_i values for each phase using CH and CRi values
CH = 15.425e-12  # History capacitor in farads
CR = [0.5e-12, 2e-12, 4e-12, 8e-12]  # Different rotating capacitors in farads
L = 4  # Number of phases

# Calculate a_i coefficients
a_values = [CH / (CH + cr) for cr in CR]

# Create the transfer function
w3c, h3c1 = signal.freqz([1],[1, -a_values[0]],fs=f_clk/(2*N))
w3c, h3c2 = signal.freqz([0,1],[1, -a_values[1]],fs=f_clk/(2*N))
w3c, h3c3 = signal.freqz([0,0,1],[1, -a_values[2]],fs=f_clk/(2*N))
w3c, h3c4 = signal.freqz([0,0,0,1],[1, -a_values[3]],fs=f_clk/(2*N))
h3c = (h3c1 + h3c2 + h3c3 + h3c4) * h

plt.figure(figsize=(10, 8))
plt.plot(w3c, 20 * np.log10(np.abs(h3c) + 1e-10))
plt.title('Cascaded System Magnitude Response of 3b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.savefig('./images/3c.png', dpi=300, bbox_inches='tight')