"""
Testbench for fully differential pipeline ADC model
Tests both ideal and non-ideal cases, calculates SNR, ENOB, and provides time domain analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import sys
import time

# Add the parent directory to the path so we can import the model files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_adc_config import *
from mdac_model import mdac_stage

# Create results directory if it doesn't exist
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'mdac'))
os.makedirs(results_dir, exist_ok=True)

# Calculate coherent sampling parameters for minimum spectral leakage
# For coherent sampling: fin/fs = M/N where M and N are integers and relatively prime
# N is the FFT length, M is the number of input signal cycles
def find_coherent_params(fs, N):
    # Find M closest to the desired fin that makes fin/fs = M/N coherent
    desired_fin = 20e6  # Desired input frequency
    desired_M = desired_fin * N / fs
    M = round(desired_M)
    
    # Make sure M and N are coprime (no common factors)
    # This ensures minimum spectral leakage
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    while gcd(M, N) != 1:
        M += 1
    
    actual_fin = M * fs / N
    print(f"Coherent sampling parameters:")
    print(f"N (FFT length) = {N}")
    print(f"M (input cycles) = {M}")
    print(f"Actual fin = {actual_fin/1e6:.6f} MHz")
    print(f"fs = {fs/1e6:.1f} MHz")
    
    return M, actual_fin

# Setup coherent sampling
N = 2**14  # FFT length
M, fin = find_coherent_params(fs, N)

# Calculate test length to capture exactly M cycles
# Each cycle is 1/fin seconds, each timestep is timebase seconds
length = int(M / fin / timebase)
print(f"Test length = {length} points ({M} cycles)")

# Generate input sine wave
timeseries = np.linspace(0, timebase*length, length, endpoint=False)
vinp = fullscale / 2 * np.sin(2*np.pi*fin*timeseries)
vinn = -fullscale / 2 * np.sin(2*np.pi*fin*timeseries)

# Initialize arrays for output
output_codes = np.zeros(len(timeseries), dtype=int)
residue_p = np.zeros(len(timeseries))
residue_n = np.zeros(len(timeseries))
dac_voltages = np.zeros(len(timeseries))
vinp_track = np.zeros(len(timeseries))
vinn_track = np.zeros(len(timeseries))

print("Running MDAC simulation...")
start_time = time.time()

# cap with mismatch initialization
cap_mismatch = np.random.normal(1.0, cap_mismatch_sigma, size=2)
[C1real, C2real] = [C1, C2] * cap_mismatch

# Run the MDAC simulation
for i in range(len(timeseries)-1):
    output_codes[i+1], residue_p[i+1], residue_n[i+1], dac_voltages[i+1], vinp_track[i+1], vinn_track[i+1] = mdac_stage(
        timeseries[i+1], vinp[i+1], vinn[i+1], C1real, C2real, residue_p[i], residue_n[i], 
        dac_voltages[i], fullscale, vinp_track[i], vinn_track[i]
    )
    
    # Print progress every 10% of the simulation
    if (i+1) % (length//10) == 0:
        print(f"Progress: {100*(i+1)/length:.1f}% complete")

print(f"Simulation completed in {time.time() - start_time:.2f} seconds")

# Calculate differential residue (output)
residue_diff = residue_p - residue_n

# Time domain analysis (3 cycles)
cycles_to_plot = 3
points_per_cycle = int(1/fin / timebase)
plot_length = cycles_to_plot * points_per_cycle

plt.figure(figsize=(12, 10))

# Input signals
plt.subplot(3, 1, 1)
plt.plot(timeseries[:plot_length]*1e9, vinp[:plot_length], 'b-', label='Vin+')
plt.plot(timeseries[:plot_length]*1e9, vinn[:plot_length], 'r-', label='Vin-')
plt.title('Differential Input')
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# MDAC output codes
plt.subplot(3, 1, 2)
plt.step(timeseries[:plot_length]*1e9, output_codes[:plot_length], 'g-', where='post')
plt.title('MDAC Output Codes')
plt.xlabel('Time (ns)')
plt.ylabel('Code')
plt.grid(True)

# Residue voltages
plt.subplot(3, 1, 3)
plt.plot(timeseries[:plot_length]*1e9, residue_p[:plot_length], 'b-', label='Residue+')
plt.plot(timeseries[:plot_length]*1e9, residue_n[:plot_length], 'r-', label='Residue-')
plt.plot(timeseries[:plot_length]*1e9, residue_diff[:plot_length], 'g-', label='Differential')
plt.title('Residue Voltages')
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'mdac_time_domain.png'))
print(f"Time domain plot saved to {os.path.join(results_dir, 'mdac_time_domain.png')}")

# FFT Analysis
# Only take samples from the hold phase for FFT analysis
hold_samples = residue_diff[1::2]  # Take every other sample (hold phase)

# Make sure we have enough samples for the FFT
if len(hold_samples) < N:
    print(f"Warning: Not enough samples for {N}-point FFT. Reducing FFT size.")
    N = 2**int(np.log2(len(hold_samples)))

# Take the first N hold samples
hold_samples = hold_samples[:N]

# Apply Hann window to reduce spectral leakage
window = np.hanning(N)
windowed_samples = hold_samples * window

# Compute FFT
fft_result = np.fft.fft(windowed_samples) / N
fft_mag = np.abs(fft_result)

# Compensate for window energy loss
# For Hann window, the correction factor is 2
fft_mag *= 2

# Convert to dB
fft_mag_db = 20 * np.log10(fft_mag + 1e-12)  # Add small constant to avoid log(0)

# Generate frequency axis
freq_axis = np.fft.fftfreq(N, 2*timebase)  # *2 because we only take hold samples

# Plot the FFT up to Nyquist frequency (fs/2)
nyquist_idx = N // 2
plt.figure(figsize=(12, 8))
plt.plot(freq_axis[:nyquist_idx] / 1e6, fft_mag_db[:nyquist_idx])
plt.title('FFT of MDAC Residue (Hold Phase)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.xlim(0, fs / 2e6)
plt.ylim(-120, 0)
plt.savefig(os.path.join(results_dir, 'mdac_fft.png'))
print(f"FFT plot saved to {os.path.join(results_dir, 'mdac_fft.png')}")

# Find signal and noise bins
signal_bin = int(N * fin / fs + 0.5) % N
signal_power = fft_mag[signal_bin]**2

# Calculate noise power (exclude DC and signal bin)
noise_bins = list(range(1, nyquist_idx))
noise_bins.remove(signal_bin)
noise_power = np.sum(fft_mag[noise_bins]**2)

# Calculate SNR
snr = 10 * np.log10(signal_power / noise_power)

# Calculate ENOB (Effective Number of Bits)
enob = (snr - 1.76) / 6.02

print("\nPerformance Analysis:")
print(f"Signal Frequency: {fin/1e6:.2f} MHz")
print(f"Sampling Frequency: {fs/1e6:.2f} MHz")
print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
print(f"Effective Number of Bits (ENOB): {enob:.2f} bits")

# Save results to a text file
with open(os.path.join(results_dir, 'mdac_results.txt'), 'w') as f:
    f.write("MDAC Performance Analysis\n")
    f.write("=======================\n\n")
    f.write(f"Configuration:\n")
    f.write(f"- Sampling Frequency: {fs/1e6:.2f} MHz\n")
    f.write(f"- Input Frequency: {fin/1e6:.2f} MHz\n")
    f.write(f"- Full Scale Voltage: {fullscale:.2f} V\n")
    f.write(f"- Op-Amp Gain: {gain:.1e} V/V\n")
    f.write(f"- Op-Amp Offset: {op_offset*1e3:.2f} mV\n")
    f.write(f"- Capacitor Mismatch Sigma: {cap_mismatch_sigma*100:.3f}%\n")
    f.write(f"- Comparator Offset: {comp_offset*1e3:.2f} mV\n")
    f.write(f"- Loop Bandwidth: {loop_bw/1e9:.2f} GHz\n")
    f.write(f"- Time Constant: {tao*1e9:.2f} ns\n\n")
    f.write(f"Results:\n")
    f.write(f"- Signal-to-Noise Ratio (SNR): {snr:.2f} dB\n")
    f.write(f"- Effective Number of Bits (ENOB): {enob:.2f} bits\n")

print(f"Results saved to {os.path.join(results_dir, 'mdac_results.txt')}")

plt.show()