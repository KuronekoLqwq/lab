
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import pandas as pd

# 1. First order model of a ZOH sampling circuit



def simulate_sampling_circuit(input_signal, t, fs, tau, duty_cycle=0.5):
    """
    Simulate a sample and hold circuit based on the differential equation:
    V_out = V_in - τ × dV_out/dt
    
    Parameters:
    - input_signal: Input signal to be sampled
    - t: Time vector
    - fs: Sampling frequency
    - tau: Time constant of the sampling circuit
    - duty_cycle: Duty cycle of the sampling switch
    
    Returns:
    - sampled_output: Output signal after sample and hold operation
    """
    Ts = 1/fs
    dt = t[1] - t[0]  # Time step
    
    # Initialize output array
    sampled_output = np.zeros_like(input_signal)
    
    # Calculate number of points for one sample period and for sampling phase
    points_per_period = int(Ts / dt)
    points_sample_phase = int(duty_cycle * points_per_period)
    
    # Initial value
    v_out = 0
    
    # For each sampling period
    for i in range(0, len(t), points_per_period):
        # Sampling phase (switch ON) - solve differential equation
        for j in range(min(points_sample_phase, len(t) - i)):
            if i+j >= len(t):
                break
                
            # Current time point and input value
            idx = i+j
            v_in = input_signal[idx]
            
            # Solve the differential equation: V_out = V_in - τ × dV_out/dt
            # Rearranging: dV_out/dt = (V_in - V_out)/τ
            # Using Euler method to integrate
            if j > 0:  # Skip first point as we need a previous value to compute derivative
                dv_out_dt = (v_in - v_out) / tau
                v_out = v_out + dv_out_dt * dt
            else:
                # For very first point in the acquisition phase, use previous held value
                v_out = sampled_output[i-1] if i > 0 else 0
            
            sampled_output[idx] = v_out
        
        # Hold phase (switch OFF) - hold the final value
        hold_value = v_out
        hold_end = min(i + points_per_period, len(t))
        sampled_output[i+points_sample_phase:hold_end] = hold_value
    
    return sampled_output

# 1.a - Plot the output of the sampling circuit
def plot_sampling_circuit_output():
    # Parameters
    fs = 10e9  # Sampling frequency: 10 GHz
    fin = 1e9  # Input signal frequency: 1 GHz
    tau = 10e-12  # Time constant: 10 ps
    num_cycles = 3  # Number of input cycles to simulate
    
    # Generate time vector with fine resolution
    t_end = num_cycles / fin
    num_points = int(t_end * fs * 100)  # 100 points per sample
    t = np.linspace(0, t_end, num_points)
    
    # Generate input sinusoidal signal
    input_signal = np.sin(2 * np.pi * fin * t)
    
    # Simulate sampling circuit
    output_signal = simulate_sampling_circuit(input_signal, t, fs, tau)
    
    # Find sampling points for visualization
    sample_indices = np.arange(0, len(t), int(len(t)/(fs*t_end)))
    sample_times = t[sample_indices]
    sample_values = output_signal[sample_indices]
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t * 1e9, input_signal, label='Input Signal (1 GHz)')
    plt.plot(t * 1e9, output_signal, label='Sampled Output')
    plt.scatter(sample_times * 1e9, sample_values, color='r', marker='o', label='Sampling Points')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title('ZOH Sampling Circuit: 1 GHz Input, 10 GHz Sampling, 10 ps Time Constant')
    plt.grid(True)
    plt.legend()
    plt.savefig('./images/sampling_circuit_output.png')
    plt.close()
    
    return sample_values

# 2. Sampling Error
def compute_time_constant_nrz(verbose=True):
    """Calculate the required time constant for NRZ input"""
    # Parameters
    bit_rate = 10e9  # Data rate: 10 Gb/s
    bit_period = 1 / bit_rate
    sampling_time = bit_period / 2  # Sample in the middle of bit period
    
    # Switch is ON for 50% of the sampling period (duty cycle)
    switch_on_time = 0.5 * bit_period
    
    # ADC parameters
    bits = 7
    full_scale = 1.0  # V
    LSB = full_scale / (2**bits)
    
    # NRZ signal
    amplitude = 0.5  # V
    
    # Calculate the maximum allowed time constant
    # For error < 1 LSB: amplitude * e^(-t/τ) < LSB
    # Therefore: τ < -t/ln(LSB/amplitude)
    max_error_ratio = LSB / amplitude
    max_tau = -switch_on_time / np.log(max_error_ratio)
    
    if verbose:
        print(f"2.a - Time Constant for NRZ Input:")
        print(f"  - 1 LSB = {LSB*1000:.2f} mV")
        print(f"  - Bit period = {bit_period*1e12:.2f} ps")
        print(f"  - Switch ON time = {switch_on_time*1e12:.2f} ps")
        print(f"  - Maximum allowed time constant: τ < {max_tau*1e12:.2f} ps")
        
        # Verify with this time constant
        tau = max_tau * 0.99  # Use 99% of maximum to ensure it's below threshold
        error = amplitude * np.exp(-switch_on_time/tau)
        print(f"  - Error with τ = {tau*1e12:.2f} ps: {error*1000:.2f} mV ({error/LSB:.2f} LSB)")
    
    return max_tau

def compute_time_constant_multitone(verbose=True):
    """Calculate the required time constant for multitone input"""
    # Parameters
    frequencies = np.array([0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9])  # GHz
    fs = 10e9  # Sampling frequency: 10 GHz
    
    # ADC parameters
    bits = 7
    full_scale = 1.0  # V
    LSB = full_scale / (2**bits)
    
    # Find the highest frequency component
    max_freq = np.max(frequencies)
    
    # For sinusoidal signals, the maximum slew rate occurs at zero-crossing
    # and is given by: SR = 2πfA where f is frequency and A is amplitude
    tone_amplitude = 0.5   # Assuming equal amplitudes summing to 0.5V
    max_slew = 2 * np.pi * max_freq * tone_amplitude
    
    # Time available for charging (50% of sampling period)
    sampling_period = 1/fs
    switch_on_time = 0.5 * sampling_period
    
    # For a continuously changing input with slope dV/dt,
    # the tracking error is approximately: error ≈ τ * dV/dt
    # For error < LSB: τ * dV/dt < LSB
    # Therefore: τ < LSB / (dV/dt)
    max_tau = LSB / max_slew
    
    if verbose:
        print(f"\n2.b - Time Constant for Multi-tone Input:")
        print(f"  - Maximum frequency component: {max_freq/1e9:.2f} GHz")
        print(f"  - Maximum slew rate: {max_slew/1e9:.2f} V/ns")
        print(f"  - Maximum allowed time constant: τ < {max_tau*1e12:.2f} ps")
        
        # Compare with the time constant from 2.a
        max_tau_nrz = compute_time_constant_nrz(verbose=False)
        print(f"\nComparison:")
        if max_tau < max_tau_nrz:
            print(f"  - Multi-tone requires a smaller time constant ({max_tau*1e12:.2f} ps < {max_tau_nrz*1e12:.2f} ps)")
            print(f"  - This is because the high-frequency components cause faster signal changes.")
        else:
            print(f"  - Multi-tone allows a larger time constant ({max_tau*1e12:.2f} ps > {max_tau_nrz*1e12:.2f} ps)")
    
    return max_tau

# 3. Sampling Error Estimation
def generate_multitone_signal(t, frequencies):
    """Generate a multi-tone signal with given frequencies"""
    amplitude = 0.5 / len(frequencies)
    signal = np.zeros_like(t)
    for freq in frequencies:
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    return signal

def quantize(signal, bits, full_scale):
    """Quantize a signal with given number of bits and full scale range"""
    levels = 2**bits
    step = full_scale / levels
    
    # Map from -0.5*full_scale to +0.5*full_scale to 0 to levels-1
    scaled = (signal + full_scale/2) / full_scale * levels
    quantized = np.floor(scaled)
    
    # Clip to valid range
    digital_value = np.clip(quantized, 0, levels-1)
    
    analog_value = digital_value * step - full_scale/2 + step/2
    # Map back to voltage
    return analog_value, digital_value

def estimate_sampling_error(verbose=True):
    """Estimate the sampling error with an N-bit quantizer using the ZOH sampling circuit model from part 1"""
    # Parameters
    fs = 10e9  # Sampling frequency: 10 GHz
    Ts = 1/fs
    duty_cycle = 0.5  # 50% duty cycle for the sampling switch
    switch_on_time = duty_cycle * Ts
    
    frequencies = np.array([0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9])
    bits = 7
    full_scale = 1.0  # V
    
    # Use the time constant from 2.a for the sampling circuit
    tau_2a = compute_time_constant_nrz(verbose=False)   
    #
    # tau_2a = 1e-15

    # Generate time vector for simulation with high time resolution
    sim_duration = 10 / frequencies[0]  # 10 cycles of the lowest frequency
    points_per_sample = 100000  # This determines the resolution of the time vector
    num_points = int(sim_duration * fs * points_per_sample)
    t = np.linspace(0, sim_duration, num_points)
    
    # Generate multi-tone input signal
    input_signal = generate_multitone_signal(t, frequencies)
    
    # Calculate the number of points in a sample period and in the switch ON time
    points_per_period = int(points_per_sample)
    points_switch_on = int(duty_cycle * points_per_period)
    
    # Simulate sampling circuit
    real_output = simulate_sampling_circuit(input_signal, t, fs, tau_2a, duty_cycle)
    
    # The actual sampling instant is at the end of the switch ON phase
    # This is when the switch opens and the capacitor holds its value
    sampling_indices = np.arange(points_switch_on , len(t), points_per_period)
    sampling_indices = np.clip(sampling_indices, 0, len(t)-1)  # Ensure indices are valid
    
    # The ideal samples are the input values at these same sampling instants
    ideal_samples = input_signal[sampling_indices]
    
    # The real samples are the output values of the sampling circuit at these instants
    real_samples = real_output[sampling_indices]
    
    # Get the times for both ideal and real samples (they're at the same instants)
    sampling_times = t[sampling_indices]
    
    # Quantize both ideal and real samples
    quantized_ideal, digital_quantized_ideal = quantize(ideal_samples, bits, full_scale)
    quantized_real, digital_quantized_real = quantize(real_samples, bits, full_scale)
    
    # Calculate error between quantized real samples and quantized ideal samples
    errors = quantized_real - quantized_ideal
    
    # Calculate error statistics
    error_mean = np.mean(errors)
    error_variance = np.var(errors)
    
    # Calculate quantization noise variance (theoretical)
    # For uniform quantization, variance = Δ²/12 where Δ is the quantization step
    quantization_step = full_scale / (2**bits)
    quantization_noise_variance = quantization_step**2 / 12
    
    if verbose:
        print(f"\n3.a - Sampling Error Estimation:")
        print(f"  - Number of samples: {len(ideal_samples)}")
        print(f"  - Error variance: {error_variance:.6e} V²")
        print(f"  - Quantization noise variance: {quantization_noise_variance:.6e} V²")
        print(f"  - Ratio of error variance to quantization noise variance: {error_variance/quantization_noise_variance:.4f}")
        
        # Verify the error is less than 1 LSB for most samples
        error_magnitudes = np.abs(errors)
        max_error = np.max(error_magnitudes)
        samples_exceeding_LSB = np.sum(error_magnitudes > quantization_step)
        percent_exceeding_LSB = (samples_exceeding_LSB / len(ideal_samples)) * 100
        
        print(f"  - Maximum error: {max_error:.6e} V ({max_error/quantization_step:.2f} LSB)")
        print(f"  - Samples exceeding 1 LSB: {samples_exceeding_LSB} ({percent_exceeding_LSB:.2f}% of total)")
        
        # Plot signals and errors
        num_samples_to_plot = 20
        plot_points = num_samples_to_plot * points_per_period
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(t[:plot_points] * 1e9, input_signal[:plot_points], 
                 label='Input Signal', alpha=0.7)
        plt.plot(t[:plot_points] * 1e9, real_output[:plot_points], 
                 label='Sampling Circuit Output', alpha=0.7)
        
        # Mark sampling instants
        plt.scatter(sampling_times[:num_samples_to_plot] * 1e9, ideal_samples[:num_samples_to_plot], 
                   color='red', marker='o', label='Ideal Value at Sampling Instant')
        plt.scatter(sampling_times[:num_samples_to_plot] * 1e9, real_samples[:num_samples_to_plot], 
                   color='green', marker='x', label='Real Sampled Value')
        
        # Mark sampling windows
        for i in range(min(num_samples_to_plot, len(sampling_times)-1)):
            start_idx = i * points_per_period
            end_idx = start_idx + points_switch_on
            if end_idx < len(t):
                plt.axvspan(t[start_idx] * 1e9, t[end_idx] * 1e9, alpha=0.2, color='gray')
        
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude (V)')
        plt.title('Sampling Circuit Operation')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(sampling_times[:num_samples_to_plot*5] * 1e9, errors[:num_samples_to_plot*5], 'o-', label='Sampling Error')
        plt.axhline(y=quantization_step, color='r', linestyle='--', label='1 LSB')
        plt.axhline(y=-quantization_step, color='r', linestyle='--')
        plt.xlabel('Time (ns)')
        plt.ylabel('Error (V)')
        plt.title('Sampling Error')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('./images/sampling_error_2a.png')
        plt.close()
    
    return errors, ideal_samples, real_samples, quantized_ideal, quantized_real, quantization_step, digital_quantized_real, digital_quantized_ideal


def fir_error_estimation(M=5, verbose=True):
    """
    Use least squares estimation to construct an M-tap FIR filter
    that estimates the sampling error.
    """
    # Get the error data from the previous simulation
    errors, ideal_samples, real_samples, quantized_ideal, quantized_real, quantization_step, digital_quantized_real, digital_quantized_ideal = estimate_sampling_error(verbose=False)
    
    # Use least squares to estimate FIR filter coefficients
    # We want to estimate the error at time n using M-1 previous ADC outputs
    # The matrix equation is: X*h = e where X is the matrix of past outputs,
    # h is the filter coefficient vector, and e is the error vector
    
    # Create input matrix X of previous ADC outputs
    N = len(quantized_real) - M  # Number of rows in X
    X = np.zeros((N, M))
    
    # Fill in the X matrix properly
    for i in range(M):
        X[:, i] = quantized_real[M-i-1:N+M-i-1]
    
    # Target error vector
    e = errors[M:M+N]
    
    # Solve for filter coefficients using least squares
    h = np.linalg.lstsq(X, e, rcond=None)[0]
    
    # Apply the filter to estimate errors
    estimated_errors = np.zeros_like(errors)
    for i in range(M, len(errors)):
        for j in range(M):
            estimated_errors[i] += h[j] * quantized_real[i-j]
    
    # Calculate error after correction
    corrected_output = quantized_real - estimated_errors
    residual_errors = corrected_output - quantized_ideal

    # Calculate error statistics
    original_error_variance = np.var(errors)
    residual_error_variance = np.var(residual_errors[M:])  # Skip the first M samples
    
    # Calculate quantization noise variance (theoretical)
    quantization_noise_variance = quantization_step**2 / 12
    
    if verbose:
        print(f"\n3.b - FIR Filter Error Estimation (M={M}):")
        print(f"  - Original error variance: {original_error_variance:.6e} V²")
        print(f"  - Residual error variance after correction: {residual_error_variance:.6e} V²")
        print(f"  - Ratio of original error to quantization noise: {original_error_variance/quantization_noise_variance:.4f}")
        print(f"  - Ratio of residual error to quantization noise: {residual_error_variance/quantization_noise_variance:.4f}")
        print(f"  - Improvement factor: {original_error_variance/residual_error_variance:.2f}x")
        
        # Plot results for visual comparison
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(errors[M:M+500], label='Original Error')
        plt.plot(estimated_errors[M:M+500], label='Estimated Error')
        plt.title(f'Error Estimation using {M}-tap FIR Filter')
        plt.ylabel('Error (V)')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(errors[M:M+500], label='Original Error')
        plt.plot(residual_errors[M:M+500], label='Residual Error')
        plt.title('Error Comparison')
        plt.xlabel('Sample Index')
        plt.ylabel('Error (V)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'./images/fir_error_estimation_M{M}.png')
        plt.close()
    
    return residual_error_variance/quantization_noise_variance

def plot_error_ratio_vs_taps():
    """Plot the ratio of error variance to quantization noise variance as M varies"""
    M_values = range(2, 11)
    ratios = []
    
    print("\n3.b - FIR Filter Error Estimation for varying M:")
    
    for M in M_values:
        ratio = fir_error_estimation(M, verbose=False)
        ratios.append(ratio)
        print(f"  - M={M}: Ratio of residual error to quantization noise = {ratio:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(M_values, ratios, 'o-')
    plt.xlabel('Number of FIR Taps (M)')
    plt.ylabel('Error Variance / Quantization Noise Variance')
    plt.title('Effect of FIR Filter Length on Error Reduction')
    plt.grid(True)
    plt.savefig('./images/error_ratio_vs_taps.png')
    plt.close()
    
    return ratios

# 4. Calibration of Errors in a Two-Channel TI-ADC
def simulate_ti_adc(verbose=True):
    """Simulate a 2-way TI-ADC that includes time, offset and bandwidth mismatches"""
    # Parameters
    fs = 10e9  # Sampling frequency: 10 GHz
    bit_rate = 10e9  # From 2.a
    bits = 7
    full_scale = 1.0
    
    # Generate NRZ input as in 2.a
    Ts = 1/fs
    bit_period = 1/bit_rate
    num_bits = 100
    t = np.linspace(0, num_bits * bit_period, num_bits * 1000)  # High time resolution
    
    # Generate random NRZ data
    np.random.seed(42)  # For reproducibility
    bit_data = np.random.randint(0, 2, num_bits) * 0.5  # 0 or 0.5V
    
    # Convert to continuous-time NRZ signal
    input_signal = np.zeros_like(t)
    for i in range(num_bits):
        idx = (t >= i*bit_period) & (t < (i+1)*bit_period)
        input_signal[idx] = bit_data[i]
    
    # Define channel mismatches
    time_mismatch = 0.1 * Ts  # 10% of sampling period
    offset_mismatch = 0.05  # 5% of full-scale
    bandwidth_mismatch = 0.2  # 20% difference in bandwidth
    
    # Nominal time constant for the sampler
    tau_nom = compute_time_constant_nrz(verbose=False) * 0.5  # Use half of the maximum allowed
    
    # Different time constants for the two channels
    tau_ch1 = tau_nom
    tau_ch2 = tau_nom * (1 + bandwidth_mismatch)
    
    # Sample with the two ADCs
    # Channel 1 samples even samples
    t_ch1 = np.arange(0, t[-1], 2*Ts)
    # Channel 2 samples odd samples with timing offset
    t_ch2 = np.arange(Ts, t[-1], 2*Ts) + time_mismatch
    
    # Interpolate input signal to get values at these sampling times
    from scipy.interpolate import interp1d
    input_interp = interp1d(t, input_signal, kind='linear', bounds_error=False, fill_value=0)
    
    # Get samples for each channel
    samples_ch1 = input_interp(t_ch1)
    samples_ch2 = input_interp(t_ch2)
    
    # Apply bandwidth limitations with first-order RC filter
    a1 = np.exp(-Ts/tau_ch1)
    a2 = np.exp(-Ts/tau_ch2)
    
    filtered_ch1 = lfilter([1-a1], [1, -a1], samples_ch1)
    filtered_ch2 = lfilter([1-a2], [1, -a2], samples_ch2)
    
    # Add offset to channel 2
    samples_ch2_offset = filtered_ch2 + offset_mismatch
    
    # Quantize
    quantized_ch1_analog, quantized_ch1_digital = quantize(filtered_ch1, bits, full_scale)
    quantized_ch2_analog, quantized_ch2_digital = quantize(samples_ch2_offset, bits, full_scale)
    
    # Interleave the outputs (simple alternating)
    ti_adc_output_analog = np.zeros(len(quantized_ch1_analog) + len(quantized_ch2_analog))
    ti_adc_output_digital = np.zeros(len(quantized_ch1_digital) + len(quantized_ch2_digital), dtype=int)
    
    ti_adc_output_analog[0::2] = quantized_ch1_analog
    ti_adc_output_analog[1::2] = quantized_ch2_analog
    
    ti_adc_output_digital[0::2] = quantized_ch1_digital
    ti_adc_output_digital[1::2] = quantized_ch2_digital
    
    # Calculate effective sampling times for the interleaved output
    t_interleaved = np.zeros_like(ti_adc_output_analog)
    t_interleaved[0::2] = t_ch1
    t_interleaved[1::2] = t_ch2
    
    # Sort by time for proper sequential output
    sorted_indices = np.argsort(t_interleaved)
    ti_adc_output_analog = ti_adc_output_analog[sorted_indices]
    ti_adc_output_digital = ti_adc_output_digital[sorted_indices]
    t_interleaved = t_interleaved[sorted_indices]
    
    # Calculate SNDR (Signal-to-Noise-and-Distortion Ratio)
    # Generate an ideal NRZ signal sampled at the correct times
    t_ideal = np.arange(0, t[-1], Ts)
    ideal_samples = input_interp(t_ideal)
    quantized_ideal_analog, quantized_ideal_digital = quantize(ideal_samples, bits, full_scale)
    
    # Calculate the error
    error = ti_adc_output_analog[:len(quantized_ideal_analog)] - quantized_ideal_analog
    signal_power = np.var(quantized_ideal_analog)
    noise_power = np.var(error)
    sndr_db = 10 * np.log10(signal_power / noise_power)
    
    if verbose:
        print(f"\n4.a - Two-Channel TI-ADC Simulation:")
        print(f"  - Time mismatch: {time_mismatch/Ts*100:.1f}% of Ts")
        print(f"  - Offset mismatch: {offset_mismatch:.3f} V")
        print(f"  - Bandwidth mismatch: {bandwidth_mismatch*100:.1f}%")
        print(f"  - Time constant channel 1: {tau_ch1*1e12:.2f} ps")
        print(f"  - Time constant channel 2: {tau_ch2*1e12:.2f} ps")
        print(f"  - SNDR: {sndr_db:.2f} dB")
        
        # Plot the results
        plt.figure(figsize=(12, 10))
        
        # Original signal and interleaved output
        plt.subplot(3, 1, 1)
        plt.plot(t[:2000], input_signal[:2000], label='Input Signal')
        plt.plot(t_interleaved[:100], ti_adc_output_analog[:100], 'o-', label='TI-ADC Output')
        plt.title('Input Signal and TI-ADC Output')
        plt.ylabel('Amplitude (V)')
        plt.grid(True)
        plt.legend()
        
        # Channel outputs separately
        plt.subplot(3, 1, 2)
        plt.plot(t_ch1[:50], quantized_ch1_analog[:50], 'o-', label='Channel 1')
        plt.plot(t_ch2[:50], quantized_ch2_analog[:50], 'o-', label='Channel 2')
        plt.title('Individual Channel Outputs')
        plt.ylabel('Amplitude (V)')
        plt.grid(True)
        plt.legend()
        
        # Error
        plt.subplot(3, 1, 3)
        plt.plot(t_ideal[:100], error[:100])
        plt.title('Error (TI-ADC Output - Ideal)')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (V)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('./images/ti_adc_simulation.png')
        plt.close()
        
        # Calculate and plot FFT to see frequency-domain effects
        from scipy.signal import welch
        
        # Use Welch method to estimate power spectral density
        f_ch1, psd_ch1 = welch(quantized_ch1_analog, fs/2, nperseg=min(1024, len(quantized_ch1_analog)))
        f_ch2, psd_ch2 = welch(quantized_ch2_analog, fs/2, nperseg=min(1024, len(quantized_ch2_analog)))
        f_ti, psd_ti = welch(ti_adc_output_analog[:len(quantized_ideal_analog)], fs, nperseg=min(1024, len(ti_adc_output_analog[:len(quantized_ideal_analog)])))
        f_ideal, psd_ideal = welch(quantized_ideal_analog, fs, nperseg=min(1024, len(quantized_ideal_analog)))
        
        plt.figure(figsize=(12, 6))
        plt.semilogy(f_ti/1e9, psd_ti, label='TI-ADC Output')
        plt.semilogy(f_ideal/1e9, psd_ideal, label='Ideal ADC')
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('PSD (V^2/Hz)')
        plt.grid(True, which='both')
        plt.legend()
        plt.savefig('./images/ti_adc_psd.png')
        plt.close()
        
        # Plot digital output values
        plt.figure(figsize=(12, 6))
        plt.step(t_interleaved[:100], ti_adc_output_digital[:100], where='post', label='TI-ADC Digital Output')
        plt.step(t_ideal[:100], quantized_ideal_digital[:100], where='post', label='Ideal ADC Digital Output', linestyle='--')
        plt.title('Digital Domain Outputs')
        plt.xlabel('Time (s)')
        plt.ylabel('Digital Code')
        plt.grid(True)
        plt.legend()
        plt.savefig('./images/ti_adc_digital.png')
        plt.close()
    
    return (quantized_ch1_analog, quantized_ch1_digital, 
            quantized_ch2_analog, quantized_ch2_digital, 
            ti_adc_output_analog, ti_adc_output_digital, 
            t_interleaved, sndr_db)


# Execute the main functions
if __name__ == "__main__":
    print("LAB 4: Sampler Error Modeling and Correction")
    print("="*50)
    
    # 1.a - Plot the output of the sampling circuit
    print("\n1.a - Executing simulation of ZOH sampling circuit")
    sample_values = plot_sampling_circuit_output()
    print(f"  - First 5 sampled values: {sample_values[:5]}")
    
    # 2.a - Calculate the time constant for NRZ input
    tau_nrz = compute_time_constant_nrz()
    
    # 2.b - Calculate the time constant for multi-tone input
    tau_multitone = compute_time_constant_multitone()
    
    # 3.a - Estimate sampling error
    errors, ideal_samples, real_samples, quantized_ideal, quantized_real, quantization_step, digital1, digital2 = estimate_sampling_error()
    
    # 3.b - FIR error estimation and correction
    ratios = plot_error_ratio_vs_taps()
    
    # 4.a - Simulate a 2-way TI-ADC with mismatches
    results = simulate_ti_adc()    
    
    print("\nLAB 4 Execution Completed")
    print("="*50)