import numpy as np
from scipy.signal.windows import blackman
import matplotlib.pyplot as plt

def compute_and_plot_dft(x, Fs, title, N=64, window=None, save_name=None):
    """
    Compute and plot DFT with optional windowing
    """
    # Apply window if specified
    if window is not None:
        window_vals = window(len(x))
        x_windowed = x * window_vals
        # Normalize
        x_windowed = x_windowed * len(x_windowed) / np.sum(window_vals)
    else:
        x_windowed = x

    # Compute DFT
    X = np.fft.fft(x_windowed, N)
    freq = np.fft.fftfreq(N, 1/Fs)

    # Plot
    plt.figure(figsize=(12, 6))
    
    # Only plot positive frequencies
    positive_freq = freq >= 0
    magnitude = np.abs(X) / len(x)  # Normalize by signal length
    
    plt.plot(freq[positive_freq]/1e6, 20*np.log10(magnitude[positive_freq]))
    plt.grid(True)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(title)
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    return freq, magnitude

# Signal parameters
F = 2e6  # 2MHz
F1 = 200e6  # 200MHz
F2 = 400e6  # 400MHz

# Time vectors and signals
def generate_signal(t, freq):
    return np.cos(2 * np.pi * freq * t)

# 1. First signal (2MHz sampled at 5MHz)
Fs_1 = 5e6
t_1 = np.arange(0, 0.001, 1/Fs_1)
x_t = generate_signal(t_1, F)

# Without window
freq1, mag1 = compute_and_plot_dft(
    x_t, Fs_1, 
    f'DFT of {F/1e6}MHz signal sampled at {Fs_1/1e6}MHz',
    save_name='dft_2MHz_no_window.png'
)

# With Blackman window
freq1_w, mag1_w = compute_and_plot_dft(
    x_t, Fs_1, 
    f'DFT of {F/1e6}MHz signal with Blackman window',
    window=blackman,
    save_name='dft_2MHz_with_window.png'
)

# 2. Second signal (200MHz + 400MHz sampled at 1GHz)
Fs_2 = 1e9
t_2 = np.arange(0, 0.001, 1/Fs_2)
y_t = generate_signal(t_2, F1) + generate_signal(t_2, F2)

# Without window
freq2, mag2 = compute_and_plot_dft(
    y_t, Fs_2,
    f'DFT of {F1/1e6}MHz + {F2/1e6}MHz signal sampled at {Fs_2/1e9}GHz',
    save_name='dft_200_400MHz_1GHz_no_window.png'
)

# With Blackman window
freq2_w, mag2_w = compute_and_plot_dft(
    y_t, Fs_2,
    f'DFT of {F1/1e6}MHz + {F2/1e6}MHz signal with Blackman window',
    window=blackman,
    save_name='dft_200_400MHz_1GHz_with_window.png'
)

# 3. Same signal sampled at 500MHz (aliasing case)
Fs_3 = 500e6
t_3 = np.arange(0, 0.001, 1/Fs_3)
y_t_alias = generate_signal(t_3, F1) + generate_signal(t_3, F2)

# Without window
freq3, mag3 = compute_and_plot_dft(
    y_t_alias, Fs_3,
    f'DFT of {F1/1e6}MHz + {F2/1e6}MHz signal sampled at {Fs_3/1e6}MHz (aliasing)',
    save_name='dft_200_400MHz_500MHz_no_window.png'
)

# With Blackman window
freq3_w, mag3_w = compute_and_plot_dft(
    y_t_alias, Fs_3,
    f'DFT of {F1/1e6}MHz + {F2/1e6}MHz signal with Blackman window (aliasing)',
    window=blackman,
    save_name='dft_200_400MHz_500MHz_with_window.png'
)

# Print peak frequencies found in each case
def find_peaks(freq, mag, threshold=-40):
    peaks = []
    for i in range(1, len(mag)-1):
        if mag[i] > threshold and mag[i] > mag[i-1] and mag[i] > mag[i+1]:
            peaks.append((abs(freq[i]), 20*np.log10(mag[i])))
    return sorted(peaks)

print("\nPeak Frequencies Found:")
print("\n1. 2MHz signal sampled at 5MHz:")
print("Without window:", find_peaks(freq1, mag1))
print("With window:", find_peaks(freq1_w, mag1_w))

print("\n2. 200MHz + 400MHz signal sampled at 1GHz:")
print("Without window:", find_peaks(freq2, mag2))
print("With window:", find_peaks(freq2_w, mag2_w))

print("\n3. 200MHz + 400MHz signal sampled at 500MHz (aliasing):")
print("Without window:", find_peaks(freq3, mag3))
print("With window:", find_peaks(freq3_w, mag3_w))