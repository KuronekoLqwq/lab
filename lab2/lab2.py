import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

# Signal initialization
fin = 2e6
fs = 5e6
t = np.arange(0, 0.001, 1/fs)
xn = np.cos(2*np.pi * fin*t)

# Noise generation
signal_power = np.mean(xn**2)
signal_power_db = 10*np.log10(signal_power)
snr = 50
noise_power_db = signal_power_db - snr
noise_power = 10 ** (noise_power_db/10) 

# The variance needed is to power of noise
noise_gaussian = np.random.normal(0,  np.sqrt(noise_power), len(xn))
x_noise =noise_gaussian + xn

X = np.fft.fft(x_noise)
psd = np.abs(X)**2 / (len(xn)**2) 
psd_nominal = 10*np.log10(psd) - 10*np.log10(max(psd))
freq = np.fft.fftfreq(len(xn), 1/fs) 
positive_freq = freq >= 0

# SNR calculation
signal_bin = np.argmax(psd)
signal_power_dft = psd[signal_bin]
mask = np.ones(len(freq), dtype=bool)
mask[signal_bin] = False
mask[len(xn)-signal_bin] = False  # Exclude negative frequency bin too
noise_power_dft = np.sum(psd[mask])
snr_dft = 10 * np.log10(2*signal_power_dft /noise_power_dft) # because noise is calculated 2 sided, so signal power should time 2
print(f"SNR: {snr_dft}")

# For uniform noise with the same SNR
# Variance of uniform distribution is (b-a)²/12
# For zero-mean: a = -b, so variance = b²/3
# So b = sqrt(3 * noise_power)
noise_range = np.sqrt(12 * noise_power) / 2
noise_uniform = np.random.uniform(-noise_range, noise_range, len(xn))
x_noise_uniform = xn + noise_uniform
Xu = np.fft.fft(x_noise_uniform)
psd_uniform = np.abs(Xu)**2 / (len(xn)**2)
psd_uniform_nominal = 10*np.log10(psd_uniform) - 10*np.log10(max(psd_uniform))
freq = np.fft.fftfreq(len(xn), 1/fs) 
positive_freq = freq >= 0

# SNR calculation
signal_bin_uniform = np.argmax(psd_uniform)
signal_power_uniform_dft = psd_uniform[signal_bin_uniform]
mask2 = np.ones(len(freq), dtype=bool)
mask2[signal_bin_uniform] = False
mask2[len(xn)-signal_bin_uniform] = False  # Exclude negative frequency bin too
noise_power_uniform_dft = np.sum(psd_uniform[mask2])
snr_uniform_dft = 10 * np.log10(2*signal_power_uniform_dft /noise_power_uniform_dft) # because noise is calculated 2 sided, so signal power should time 2
print(f"SNR_Uniform: {snr_uniform_dft}")

plt.figure(figsize=(10, 6))
plt.plot(freq[positive_freq]/1e6, psd_nominal[positive_freq])
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('PSD (dB/Hz)')
plt.title('PSD of Noisy Signal')
plt.savefig('1a.png', dpi=300, bbox_inches='tight')

# Now, before DFT, add 3 kinds of windows. 
window1 = np.hanning(len(xn))
window2 = np.hamming(len(xn))
window3 = np.blackman(len(xn))

x_noise_windowed1 = x_noise*window1
x_noise_windowed2 = x_noise*window2
x_noise_windowed3 = x_noise*window3

X_1 = np.fft.fft(x_noise_windowed1)
X_2 = np.fft.fft(x_noise_windowed2)
X_3 = np.fft.fft(x_noise_windowed3)
psd_1 = np.abs(X_1)**2 / (len(xn)**2)
psd_1_nominal = 10*np.log10(psd_1) - 10*np.log10(max(psd_1))
psd_2 = np.abs(X_2)**2 / (len(xn)**2) 
psd_2_nominal = 10*np.log10(psd_2) - 10*np.log10(max(psd_2))
psd_3 = np.abs(X_3)**2 / (len(xn)**2) 
psd_3_nominal = 10*np.log10(psd_3) - 10*np.log10(max(psd_3))

# SNR calculation
signal_bin_uniform = np.argmax(psd_uniform)
signal_power_uniform_dft = psd_uniform[signal_bin_uniform]
mask2 = np.ones(len(freq), dtype=bool)
mask2[signal_bin_uniform] = False
mask2[len(xn)-signal_bin_uniform] = False  # Exclude negative frequency bin too
noise_power_uniform_dft = np.sum(psd_uniform[mask2])
snr_uniform_dft = 10 * np.log10(2*signal_power_uniform_dft /noise_power_uniform_dft) # because noise is calculated 2 sided, so signal power should time 2
print(f"SNR_Uniform: {snr_uniform_dft}")


plt.figure(figsize=(10, 6))
plt.plot(freq[positive_freq]/1e6, psd_1_nominal[positive_freq])
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('PSD (dB/Hz)')
plt.title('PSD of Noisy Signal')
plt.savefig('./images/1b1.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(freq[positive_freq]/1e6, psd_2_nominal[positive_freq])
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('PSD (dB/Hz)')
plt.title('PSD of Noisy Signal')
plt.savefig('./images/1b2.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(freq[positive_freq]/1e6, psd_3_nominal[positive_freq])
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('PSD (dB/Hz)')
plt.title('PSD of Noisy Signal')
plt.savefig('./images/1b3.png', dpi=300, bbox_inches='tight')

# Quantization part
# Here the quantizer is for digital signal 
# Mostly I change the value and save different plot only, so the whole results can't be obtained directly from the script
fin2 = 200e6
fs2 = 510e6
full_scale = 2
bits = 12
t2 = np.arange(0, 100/fin2, 1/fs2) # 30 periods , change it to 100 for 2a2
xn2 = full_scale/2 * np.cos(2 * np.pi *fin2*t2 )

noise_power_dbe = 25
noise_powere = 10 ** (noise_power_dbe/10) 

# The variance needed is to power of noise
noisee_gaussian = np.random.normal(0,  np.sqrt(noise_powere), len(xn2))


LSB = full_scale / (2**bits)
quant1 = np.round(xn2 / LSB) 
quant1_hanning = quant1 * np.hanning(len(quant1))
x_noise_2 =noisee_gaussian + quant1
x_noise_2_hanning = noisee_gaussian + quant1_hanning


X2a = np.fft.fft(x_noise_2_hanning)
psd2 = np.abs(X2a)**2 / (len(xn2)**2) 
psd2_nominal = 10*np.log10(psd2) - 10*np.log10(max(psd2))
freq2 = np.fft.fftfreq(len(xn2), 1/fs2) 
positive_freq2 = freq2 >= 0

plt.figure(figsize=(10, 6))
plt.stem(freq2/1e6, 10*np.log10(psd2))
plt.grid(True)
plt.xlabel('Frequency (MHz)')
plt.ylabel('PSD (dB/Hz)')
plt.title('PSD of Quantized Signal')
plt.savefig('./images/2e2.png', dpi=300, bbox_inches='tight')

# SNR calculation
signal_bin2 = np.argmax(psd2)
signal_power_dft2 = psd2[signal_bin2]
mask2 = np.ones(len(freq2), dtype=bool)
#mask2[signal_bin2] = False
#mask2[len(xn2) - signal_bin2] = False
mask2[signal_bin2-8:signal_bin2+8] = False
mask2[len(xn2)-signal_bin2-8:signal_bin2+8+len(xn2) ] = False  # Exclude negative frequency bin too
noise_power_dft2 = np.sum(psd2[mask2])
snr_dft2 = 10 * np.log10(2*signal_power_dft2 /noise_power_dft2) # because noise is calculated 2 sided, so signal power should time 2
print(f"SNR 2a: {snr_dft2}")