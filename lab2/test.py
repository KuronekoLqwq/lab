import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as fftpack

def plot_psd(x, fs, title, window=None):
    """
    Calculate and plot Power Spectral Density
    
    Parameters:
    x: input signal
    fs: sampling frequency
    title: plot title
    window: optional window function
    
    Returns:
    frequencies, psd values, and figure
    """
    N = len(x)
    if window is not None:
        if window == 'hanning':
            w = np.hanning(N)
        elif window == 'hamming':
            w = np.hamming(N)
        elif window == 'blackman':
            w = np.blackman(N)
        else:
            w = np.ones(N)
            
        # Apply window
        x_windowed = x * w
        # Window scaling factor (for correct power estimation)
        scale = N / np.sum(w**2)
    else:
        x_windowed = x
        scale = 1.0
    
    # Compute FFT
    X = np.fft.fft(x_windowed)
    # Compute PSD
    psd = np.abs(X)**2 / (N**2) * scale
    # Single-sided PSD
    psd_single = 2 * psd[:N//2]
    
    # Calculate frequencies
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs/1e6, 10*np.log10(psd_single))
    plt.grid(True)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.title(title)
    
    return freqs, psd_single, plt.gcf()

def calculate_snr_from_psd(freqs, psd, signal_freq, bw=None):
    """
    Calculate SNR from PSD
    
    Parameters:
    freqs: frequency bins
    psd: power spectral density
    signal_freq: frequency of the signal tone
    bw: bandwidth to consider for the noise calculation
    
    Returns:
    SNR in dB
    """
    # Find the index of the signal frequency
    idx = np.argmin(np.abs(freqs - signal_freq))
    
    # Calculate signal power (tone power)
    signal_power = psd[idx]
    
    # Create a mask to exclude the signal and DC components
    mask = np.ones_like(freqs, dtype=bool)
    mask[0] = False  # Exclude DC
    mask[idx-1:idx+2] = False  # Exclude signal and adjacent bins
    
    # Calculate noise power
    if bw is None:
        noise_power = np.mean(psd[mask])
    else:
        # Calculate noise power within specified bandwidth
        bw_mask = (freqs >= signal_freq - bw/2) & (freqs <= signal_freq + bw/2) & mask
        noise_power = np.mean(psd[bw_mask])
    
    # Calculate SNR
    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    
    return snr_db

def quantize(x, n_bits, full_scale=2.0):
    """
    Quantize a signal to n_bits resolution
    
    Parameters:
    x: input signal
    n_bits: number of quantization bits
    full_scale: full scale range (default: -1.0 to 1.0, range = 2.0)
    
    Returns:
    quantized signal
    """
    # Number of quantization levels
    n_levels = 2**n_bits
    
    # Quantization step size
    delta = full_scale / n_levels
    
    # Scale and round to nearest integer level
    scaled = x * (n_levels / full_scale)
    quantized_levels = np.round(scaled)
    
    # Clip to prevent overflow
    quantized_levels = np.clip(quantized_levels, -n_levels/2, n_levels/2 - 1)
    
    # Scale back to original range
    quantized = quantized_levels * (full_scale / n_levels)
    
    return quantized

# Part 1: Signal to Noise Ratio
def part1_snr():
    print("Part 1: Signal to Noise Ratio")
    
    # a) Generate 2 MHz tone, sample at 5 MHz, add Gaussian noise for 50 dB SNR
    f_tone = 2e6  # 2 MHz
    f_s = 5e6     # 5 MHz
    amplitude = 1.0
    snr_target = 50  # dB
    
    # Time vector (10 periods of the sine wave)
    num_periods = 100
    t_period = 1 / f_tone
    t_total = num_periods * t_period
    num_samples = int(t_total * f_s)
    t = np.arange(num_samples) / f_s
    
    # Generate clean sine wave
    x_clean = amplitude * np.sin(2 * np.pi * f_tone * t)
    
    # Calculate signal power
    signal_power = np.mean(x_clean**2)
    print(f"Signal power: {signal_power:.6f}")
    
    # Calculate noise variance required for 50 dB SNR
    # SNR = 10*log10(signal_power/noise_power)
    # noise_power = signal_power / 10^(SNR/10)
    noise_power_gaussian = signal_power / (10**(snr_target/10))
    print(f"Required Gaussian noise variance for {snr_target} dB SNR: {noise_power_gaussian:.10f}")
    
    # Generate Gaussian noise
    noise_gaussian = np.random.normal(0, np.sqrt(noise_power_gaussian), len(x_clean))
    
    # Add noise to signal
    x_noisy_gaussian = x_clean + noise_gaussian
    
    # Calculate actual SNR
    measured_snr_gaussian = 10 * np.log10(signal_power / np.var(noise_gaussian))
    print(f"Measured SNR (Gaussian noise): {measured_snr_gaussian:.2f} dB")
    
    # For uniform noise with the same SNR
    # Variance of uniform distribution is (b-a)²/12
    # For zero-mean: a = -b, so variance = b²/3
    # So b = sqrt(3 * noise_power)
    noise_range = np.sqrt(12 * noise_power_gaussian) / 2
    print(f"Required uniform noise range for {snr_target} dB SNR: ±{noise_range:.6f}")
    
    # Generate uniform noise
    noise_uniform = np.random.uniform(-noise_range, noise_range, len(x_clean))
    
    # Add uniform noise to signal
    x_noisy_uniform = x_clean + noise_uniform
    
    # Calculate and plot PSD
    print("\nPSD without window:")
    freqs, psd, fig1 = plot_psd(x_noisy_gaussian, f_s, f"PSD of 2 MHz tone with Gaussian noise ({snr_target} dB SNR)")
    
    # Calculate SNR from PSD
    snr_from_psd = calculate_snr_from_psd(freqs, psd, f_tone)
    print(f"SNR calculated from PSD: {snr_from_psd:.2f} dB")
    
    # b) Repeat with window functions
    print("\nPSD with different windows:")
    windows = ['hanning', 'hamming', 'blackman']
    
    for window in windows:
        print(f"\nUsing {window} window:")
        freqs, psd, fig = plot_psd(x_noisy_gaussian, f_s, f"PSD with {window.capitalize()} window", window)
        snr_from_psd = calculate_snr_from_psd(freqs, psd, f_tone)
        print(f"SNR calculated from PSD: {snr_from_psd:.2f} dB")

# Part 2: Quantization
def part2_quantization():
    print("\nPart 2: Quantization")
    
    # a) 6-bit quantizer, 200 MHz tone sampled at 400 MHz
    f_tone = 200e6  # 200 MHz
    f_s = 400e6     # 400 MHz
    amplitude = 1.0  # Full scale = 2V (-1V to +1V)
    
    # Generate 30 periods
    num_periods_30 = 30
    t_period = 1 / f_tone
    t_total_30 = num_periods_30 * t_period
    num_samples_30 = int(t_total_30 * f_s)
    t_30 = np.arange(num_samples_30) / f_s
    
    # Generate 100 periods
    num_periods_100 = 100
    t_total_100 = num_periods_100 * t_period
    num_samples_100 = int(t_total_100 * f_s)
    t_100 = np.arange(num_samples_100) / f_s
    
    # Generate sine waves
    x_clean_30 = amplitude * np.sin(2 * np.pi * f_tone * t_30)
    x_clean_100 = amplitude * np.sin(2 * np.pi * f_tone * t_100)
    
    # Quantize with 6 bits
    n_bits = 6
    x_quant_30 = quantize(x_clean_30, n_bits)
    x_quant_100 = quantize(x_clean_100, n_bits)
    
    # Calculate quantization error
    q_error_30 = x_quant_30 - x_clean_30
    q_error_100 = x_quant_100 - x_clean_100
    
    # Plot PSD of quantized signal (30 periods)
    print("\n6-bit quantization, 30 periods:")
    freqs_30, psd_30, fig = plot_psd(x_quant_30, f_s, f"PSD of 6-bit quantized 200 MHz tone (30 periods)")
    snr_30 = calculate_snr_from_psd(freqs_30, psd_30, f_tone)
    print(f"SNR from PSD (30 periods): {snr_30:.2f} dB")
    
    # Plot PSD of quantized signal (100 periods)
    print("\n6-bit quantization, 100 periods:")
    freqs_100, psd_100, fig = plot_psd(x_quant_100, f_s, f"PSD of 6-bit quantized 200 MHz tone (100 periods)")
    snr_100 = calculate_snr_from_psd(freqs_100, psd_100, f_tone)
    print(f"SNR from PSD (100 periods): {snr_100:.2f} dB")
    
    # b) Find incommensurate sampling frequency
    # Use a sampling frequency that is not a multiple of the signal frequency
    # For example, use a prime number near the Nyquist rate
    f_s_incomm = 401e6  # 401 MHz (just above Nyquist)
    
    # Generate sine wave with incommensurate sampling
    num_samples_incomm = int(t_total_30 * f_s_incomm)
    t_incomm = np.arange(num_samples_incomm) / f_s_incomm
    x_clean_incomm = amplitude * np.sin(2 * np.pi * f_tone * t_incomm)
    
    # Quantize with 6 bits
    x_quant_incomm = quantize(x_clean_incomm, n_bits)
    
    # Plot PSD with incommensurate sampling
    print("\n6-bit quantization, incommensurate sampling:")
    freqs_incomm, psd_incomm, fig = plot_psd(x_quant_incomm, f_s_incomm, f"PSD of 6-bit quantized 200 MHz tone (Incommensurate sampling)")
    snr_incomm = calculate_snr_from_psd(freqs_incomm, psd_incomm, f_tone)
    print(f"SNR from PSD (incommensurate sampling): {snr_incomm:.2f} dB")
    
    # c) 12-bit quantizer
    n_bits_12 = 12
    x_quant_12_30 = quantize(x_clean_30, n_bits_12)
    
    # Plot PSD with 12-bit quantizer
    print("\n12-bit quantization:")
    freqs_12, psd_12, fig = plot_psd(x_quant_12_30, f_s, f"PSD of 12-bit quantized 200 MHz tone")
    snr_12 = calculate_snr_from_psd(freqs_12, psd_12, f_tone)
    print(f"SNR from PSD (12-bit): {snr_12:.2f} dB")
    
    # Verify SNR ~ 6N
    print(f"\nTheoretical SNR for 6-bit: {6*n_bits:.2f} dB")
    print(f"Theoretical SNR for 12-bit: {6*n_bits_12:.2f} dB")
    print(f"Measured SNR for 6-bit: {snr_30:.2f} dB")
    print(f"Measured SNR for 12-bit: {snr_12:.2f} dB")
    
    # d) Use Hanning window with 12-bit quantizer
    print("\n12-bit quantization with Hanning window:")
    freqs_12_hann, psd_12_hann, fig = plot_psd(x_quant_12_30, f_s, f"PSD of 12-bit quantized 200 MHz tone (Hanning window)", 'hanning')
    snr_12_hann = calculate_snr_from_psd(freqs_12_hann, psd_12_hann, f_tone)
    print(f"SNR from PSD (12-bit with Hanning): {snr_12_hann:.2f} dB")
    
    # e) Add noise for 38 dB SNR
    snr_target_e = 38  # dB
    
    # Calculate signal power
    signal_power = np.mean(x_clean_30**2)
    
    # Calculate noise variance for 38 dB SNR
    noise_power_e = signal_power / (10**(snr_target_e/10))
    
    # Generate and add noise
    noise_e = np.random.normal(0, np.sqrt(noise_power_e), len(x_clean_30))
    x_noisy_e = x_clean_30 + noise_e
    
    # Quantize with 6 and 12 bits
    x_quant_6_noisy = quantize(x_noisy_e, n_bits)
    x_quant_12_noisy = quantize(x_noisy_e, n_bits_12)
    
    # Plot PSD with 6-bit quantizer + noise
    print("\n6-bit quantization with 38 dB SNR input:")
    freqs_6_noisy, psd_6_noisy, fig = plot_psd(x_quant_6_noisy, f_s, f"PSD of 6-bit quantized 200 MHz tone with 38 dB SNR input")
    snr_6_noisy = calculate_snr_from_psd(freqs_6_noisy, psd_6_noisy, f_tone)
    print(f"SNR from PSD (6-bit with noise): {snr_6_noisy:.2f} dB")
    
    # Plot PSD with 12-bit quantizer + noise
    print("\n12-bit quantization with 38 dB SNR input:")
    freqs_12_noisy, psd_12_noisy, fig = plot_psd(x_quant_12_noisy, f_s, f"PSD of 12-bit quantized 200 MHz tone with 38 dB SNR input")
    snr_12_noisy = calculate_snr_from_psd(freqs_12_noisy, psd_12_noisy, f_tone)
    print(f"SNR from PSD (12-bit with noise): {snr_12_noisy:.2f} dB")
    
    # With Hanning window
    print("\n6-bit quantization with 38 dB SNR input and Hanning window:")
    freqs_6_noisy_hann, psd_6_noisy_hann, fig = plot_psd(x_quant_6_noisy, f_s, f"PSD of 6-bit quantized 200 MHz tone with 38 dB SNR input (Hanning)", 'hanning')
    snr_6_noisy_hann = calculate_snr_from_psd(freqs_6_noisy_hann, psd_6_noisy_hann, f_tone)
    print(f"SNR from PSD (6-bit with noise and Hanning): {snr_6_noisy_hann:.2f} dB")
    
    print("\n12-bit quantization with 38 dB SNR input and Hanning window:")
    freqs_12_noisy_hann, psd_12_noisy_hann, fig = plot_psd(x_quant_12_noisy, f_s, f"PSD of 12-bit quantized 200 MHz tone with 38 dB SNR input (Hanning)", 'hanning')
    snr_12_noisy_hann = calculate_snr_from_psd(freqs_12_noisy_hann, psd_12_noisy_hann, f_tone)
    print(f"SNR from PSD (12-bit with noise and Hanning): {snr_12_noisy_hann:.2f} dB")

# Run the parts
if __name__ == "__main__":
    part1_snr()
    part2_quantization()