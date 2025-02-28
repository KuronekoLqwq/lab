import numpy as np
from scipy.signal.windows import blackman
import matplotlib.pyplot as plt

def compute_and_plot_dft(x, Fs, title, N=64, window=None, save_name=None):
    """
    Compute and plot DFT with proper window amplitude correction
    """
    if window is not None:
        window_vals = window(len(x))
        # Apply window
        x_windowed = x * window_vals
        # Correct for window scaling (coherent gain)
        coherent_gain = np.mean(window_vals)
        x_windowed = x_windowed / coherent_gain
    else:
        x_windowed = x

    # Compute DFT
    X = np.fft.fft(x_windowed, N)
    freq = np.fft.fftfreq(N, 1/Fs)

    # Plot
    plt.figure(figsize=(12, 6))
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

# Test with a single frequency
F = 200e6  # 200MHz
Fs = 1e9   # 1GHz
t = np.arange(0, 0.001, 1/Fs)
x = np.cos(2 * np.pi * F * t)

# Without window
freq1, mag1 = compute_and_plot_dft(
    x, Fs, 
    'DFT without Window',
    save_name='dft_no_window_corrected.png'
)

# With corrected Blackman window
freq2, mag2 = compute_and_plot_dft(
    x, Fs, 
    'DFT with Corrected Blackman Window',
    window=blackman,
    save_name='dft_with_window_corrected.png'
)

# Print peak magnitudes
def find_peak_magnitude(mag):
    return 20 * np.log10(np.max(mag))

print("\nPeak Magnitudes:")
print(f"Without window: {find_peak_magnitude(mag1):.2f} dB")
print(f"With window: {find_peak_magnitude(mag2):.2f} dB")

# Also show the window's effect
window = blackman(1000)
plt.figure(figsize=(12, 4))
plt.plot(window)
plt.title('Blackman Window')
plt.grid(True)
plt.savefig('blackman_window.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nWindow correction factor (coherent gain): {np.mean(window):.3f}")