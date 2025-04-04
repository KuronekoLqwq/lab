import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter


def simulate_improved_sh_circuit(input_signal, t, fs, tau, duty_cycle=0.5):
    """
    Simulate a sample-and-hold circuit with time constant tau, modeling as described
    in "Bandwidth Mismatch and Its Correction in Time-Interleaved ADCs" (Tsai et al).
    
    Parameters:
    - input_signal: Input signal to be sampled
    - t: Time vector
    - fs: Sampling frequency
    - tau: Time constant of the sampling circuit (R_on * C)
    - duty_cycle: Duty cycle of the sampling switch
    
    Returns:
    - output_signal: Complete S&H output (continuous-time)
    - sampled_values: Values at sampling instants (discrete-time)
    """
    Ts = 1/fs
    dt = t[1] - t[0]  # Time step
    
    # Calculate number of points for one sample period and for ON phase
    points_per_period = int(Ts / dt)
    points_switch_on = int(duty_cycle * points_per_period)
    
    # Initialize output arrays
    output_signal = np.zeros_like(input_signal)
    
    # Timing 
    num_periods = len(t) // points_per_period
    sampling_indices = []
    
    # For each sampling period
    for n in range(num_periods):
        period_start = n * points_per_period
        switch_off = period_start + points_switch_on
        
        if switch_off >= len(t):
            break
            
        # Calculate the impulse response for the sampling window (ON phase)
        # h(t) = (1/tau) * exp(-t/tau) for t >= 0
        t_window = t[period_start:switch_off] - t[period_start]
        h_impulse = (1/tau) * np.exp(-t_window/tau)
        h_impulse = h_impulse / np.sum(h_impulse)  # Normalize
        
        # Calculate the convolution for the sampling window (tracking phase)
        # This implements the first-order low-pass filter response
        input_window = input_signal[period_start:switch_off]
        
        # Get the initial condition (previous held value)
        if period_start > 0:
            initial_value = output_signal[period_start-1]
        else:
            initial_value = 0
            
        # Compute the step response from the initial value to the input
        for i in range(len(input_window)):
            idx = period_start + i
            if idx < len(output_signal):
                # First-order response: v_out(t) = v_in + (v_initial - v_in) * exp(-t/tau)
                t_elapsed = i * dt
                output_signal[idx] = input_window[i] - (input_window[i] - initial_value) * np.exp(-t_elapsed/tau)
        
        # Hold phase (switch OFF)
        hold_value = output_signal[switch_off-1]
        sampling_indices.append(switch_off-1)  # This is when we sample (end of tracking phase)
        
        hold_end = min((n+1) * points_per_period, len(output_signal))
        output_signal[switch_off:hold_end] = hold_value
    
    # Extract the sampled values at the end of each tracking phase
    sampled_indices = np.array(sampling_indices)
    sampled_indices = sampled_indices[sampled_indices < len(output_signal)]
    sampled_values = output_signal[sampled_indices]
    sampling_times = t[sampled_indices]
    
    return output_signal, sampled_values, sampling_times, sampled_indices

def demonstrate_sh_behavior():
    """Demonstrate the behavior of the improved S&H model"""
    # Parameters
    fs = 1e9  # 1 GHz sampling frequency
    input_freqs = [100e6, 400e6]  # 100 MHz and 400 MHz input frequencies
    tau1 = 0.2e-9  # 200 ps time constant (channel 1)
    tau2 = 0.22e-9  # 220 ps time constant (channel 2) - 10% mismatch
    
    # Time vector - simulate for 20 sample periods with high resolution
    num_periods = 20
    points_per_period = 100
    num_points = num_periods * points_per_period
    t = np.linspace(0, num_periods/fs, num_points)
    
    # Generate two-tone input signal
    input_signal = 0.4 * np.sin(2 * np.pi * input_freqs[0] * t) + 0.3 * np.sin(2 * np.pi * input_freqs[1] * t)
    
    # Simulate S&H for two channels with different time constants
    output1, samples1, times1, indices1 = simulate_improved_sh_circuit(input_signal, t, fs, tau1)
    output2, samples2, times2, indices2 = simulate_improved_sh_circuit(input_signal, t, fs, tau2)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Plot the input signal and S&H outputs
    plt.subplot(211)
    plt.plot(t*1e9, input_signal, 'b-', label='Input Signal')
    plt.plot(t*1e9, output1, 'r-', label=f'Channel 1 Output (τ = {tau1*1e9:.1f} ns)')
    plt.plot(t*1e9, output2, 'g-', label=f'Channel 2 Output (τ = {tau2*1e9:.1f} ns)')
    plt.scatter(times1*1e9, samples1, color='r', marker='o', label='Channel 1 Samples')
    plt.scatter(times2*1e9, samples2, color='g', marker='x', label='Channel 2 Samples')
    
    # Add time interleaving
    interleaved_times = np.sort(np.concatenate((times1, times2)))
    interleaved_samples = np.zeros_like(interleaved_times)
    
    # Assign samples to interleaved output (alternating channels)
    for i in range(len(interleaved_times)):
        if i % 2 == 0:  # Even samples from channel 1
            idx = np.where(times1 == interleaved_times[i])[0]
            if len(idx) > 0:
                interleaved_samples[i] = samples1[idx[0]]
        else:  # Odd samples from channel 2
            idx = np.where(times2 == interleaved_times[i])[0]
            if len(idx) > 0:
                interleaved_samples[i] = samples2[idx[0]]
    
    plt.scatter(interleaved_times*1e9, interleaved_samples, color='k', marker='*', 
                label='Interleaved Samples')
    
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title('Sample-and-Hold Circuit Operation with Bandwidth Mismatch')
    plt.grid(True)
    plt.legend()
    
    # Plot the error due to bandwidth mismatch
    plt.subplot(212)
    
    # Create ideal samples (with infinite bandwidth)
    ideal_samples = input_signal[indices1]
    
    # Calculate errors
    error1 = samples1 - ideal_samples
    error2 = samples2 - ideal_samples[:len(samples2)]
    
    plt.stem(times1*1e9, error1, 'r', label='Channel 1 Error', markerfmt='ro')
    plt.stem(times2*1e9, error2, 'g', label='Channel 2 Error', markerfmt='gx')
    
    plt.xlabel('Time (ns)')
    plt.ylabel('Error Amplitude')
    plt.title('Sampling Error Due to Finite Bandwidth')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./images/sh_bandwidth_mismatch.png')
    plt.show()
    
    return output1, output2, samples1, samples2

# Frequency domain analysis
def analyze_frequency_response(tau1, tau2, fs, fmax=None):
    """
    Analyze the frequency response of two S&H circuits with different time constants
    as described in the paper.
    """
    if fmax is None:
        fmax = fs/2  # Nyquist frequency
    
    # Create frequency vector from DC to fmax
    num_points = 1000
    f = np.linspace(0, fmax, num_points)
    omega = 2 * np.pi * f
    
    # Sample period
    Ts = 1/fs
    T = Ts
    
    # Channel 1 frequency response (Equation 8 in the paper)
    G1 = np.abs((1 - np.exp(-(T/(2*tau1)) + 1j*omega*T/2)) / 
                ((1 + 1j*omega*tau1) * (1 - np.exp(-(T/(2*tau1)) + 1j*omega*T))))
    
    # Channel 2 frequency response (Equation 10 in the paper)
    G2 = np.abs((1 - np.exp(-(T/(2*tau2)) + 1j*omega*T/2)) / 
                ((1 + 1j*omega*tau2) * (1 - np.exp(-(T/(2*tau2)) + 1j*omega*T))))
    
    # Plot frequency responses
    plt.figure(figsize=(10, 6))
    plt.plot(f/1e6, G1, 'r-', label=f'Channel 1 (τ = {tau1*1e9:.1f} ns)')
    plt.plot(f/1e6, G2, 'g-', label=f'Channel 2 (τ = {tau2*1e9:.1f} ns)')
    plt.plot(f/1e6, G2/G1, 'b--', label='Gain Ratio (G2/G1)')
    
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response of S&H Circuits with Bandwidth Mismatch')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, fmax/1e6)
    plt.savefig('./images/sh_frequency_response.png')
    plt.show()
    
    return G1, G2

if __name__ == "__main__":
    print("Demonstrating improved S&H model with bandwidth mismatch...")
    output1, output2, samples1, samples2 = demonstrate_sh_behavior()
    
    # Analyze frequency response
    tau1 = 0.2e-9  # 200 ps
    tau2 = 0.22e-9  # 220 ps (10% mismatch)
    fs = 1e9  # 1 GHz
    G1, G2 = analyze_frequency_response(tau1, tau2, fs, fmax=500e6)
    
    print("Analysis complete. Check the generated plots.")