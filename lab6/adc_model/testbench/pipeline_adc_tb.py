"""
Pipeline ADC Continuous Ramp Testing
- Simulates a continuous ramp input with progressing timebase
- Records the ADC output codes over time
- Plots the input signal and resulting digital outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_adc_config import *
from mdac_model import mdac_stage, mdac_1bit_stage
from pipeline_adc_model import pipeline_adc

# Create results directory if it doesn't exist

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results', 'pipeline_adc_ramp'))
os.makedirs(results_dir, exist_ok=True)

def continuous_ramp_test():
    """Test the ADC with a continuous ramp input signal"""
    
    print("Starting Pipeline ADC continuous ramp testing...")
    
    # Test parameters
    num_samples = 100000  # Number of samples to simulate
    ramp_min = -fullscale  # Minimum voltage of ramp
    ramp_max = fullscale   # Maximum voltage of ramp
    
    # ADC configuration
    stages = 7  # 6 2.5-bit stages + 1 1-bit stage
    
    # Time array - continuous progression
    t_values = np.arange(num_samples) * timebase
    
    # Generate ramp input signal (differential)
    ramp_slope = (ramp_max - ramp_min) / num_samples
    vin_diff = ramp_min + ramp_slope * np.arange(num_samples)
    
    # Split into differential inputs
    vin_p = vin_diff 
    vin_n = -vin_diff 
    
    # Initialize arrays to store results
    digital_outputs = np.zeros(num_samples, dtype=int)
    
    # Initialize state variables for the ADC
    prev_residue_p = np.zeros((num_samples,stages))
    prev_residue_n = np.zeros((num_samples,stages))
    prev_dac_voltage = np.zeros((num_samples,stages))
    prev_vin_p_track = np.zeros((num_samples,stages))
    prev_vin_n_track = np.zeros((num_samples,stages))
    prev_delays = np.zeros((num_samples, stages-1, stages-1, 1))
    
    # Run the simulation
    for i in range(num_samples-1):
        # Current time
        t = t_values[i]
        
        # Run the ADC with current input and time
        digital_outputs[i+1], prev_residue_p[i+1], prev_residue_n[i+1], prev_dac_voltage[i+1], prev_vin_p_track[i+1], prev_vin_n_track[i+1], prev_delays[i+1] = pipeline_adc(
            t, vin_p[i], vin_n[i], fullscale, 
            prev_residue_p[i], prev_residue_n[i], prev_dac_voltage[i], 
            prev_vin_p_track[i], prev_vin_n_track[i], prev_delays[i]
        )
             
        # Print progress
        if (i+1) % (num_samples//10) == 0:
            print(f"Progress: {100*(i+1)/num_samples:.1f}% complete")
    
    print("Simulation complete. Generating plots...")
    
    # Get information about the digital outputs
    unique_codes = np.unique(digital_outputs)
    num_codes = len(unique_codes)
    print(f"Number of unique output codes: {num_codes}")
    print(f"Range of output codes: {min(digital_outputs)} to {max(digital_outputs)}")
    
    # Figure 1: Input Ramp Signal
    plt.figure(figsize=(12, 4))
    plt.plot(t_values, vin_diff, 'b-', linewidth=1)
    plt.title('Ramp Input Signal', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Differential Input Voltage (V)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'input_ramp.png'), dpi=300)
    
    # Figure 2: ADC Output Codes
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, digital_outputs, 'r-', linewidth=1)
    plt.title('Pipeline ADC Output Codes (Continuous Ramp Input)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Digital Output Code', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'adc_output_codes.png'), dpi=300)
    
    # Figure 3: Combined Input and Output
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Input ramp (left y-axis)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Differential Input Voltage (V)', color='b', fontsize=12)
    ax1.plot(t_values, vin_diff, 'b-', linewidth=1)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    # Output codes (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Digital Output Code', color='r', fontsize=12)
    ax2.plot(t_values, digital_outputs, 'r-', linewidth=1)
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Pipeline ADC Input and Output', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'input_output_combined.png'), dpi=300)
    
    # Figure 4: Transfer function (Input vs Output)
    plt.figure(figsize=(12, 6))
    plt.plot(vin_diff, digital_outputs, 'b.', markersize=1)
    plt.title('Pipeline ADC Transfer Function', fontsize=14)
    plt.xlabel('Differential Input Voltage (V)', fontsize=12)
    plt.ylabel('Digital Output Code', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'transfer_function.png'), dpi=300)
    
    # Save the data as CSV for further analysis
    data = np.column_stack((t_values, vin_diff, digital_outputs))
    header = "Time,Input_Voltage,Digital_Output"
    np.savetxt(os.path.join(results_dir, 'ramp_test_data.csv'), 
               data, delimiter=',', header=header, comments='')
    print(f"Ramp test data saved to {os.path.join(results_dir, 'ramp_test_data.csv')}")
    
    return {
        'time': t_values,
        'input': vin_diff,
        'output': digital_outputs,
        'unique_codes': unique_codes,
        'num_codes': num_codes
    }

if __name__ == "__main__":
    # Run the ramp test
    results = continuous_ramp_test()
    
    # Also print progression of first 20 samples to check proper time behavior
    print("\nFirst 20 samples:")
    for i in range(min(20, len(results['time']))):
        print(f"t = {results['time'][i]:.3e}s, Vin = {results['input'][i]:.3f}V, Code = {results['output'][i]}")
    
    plt.show()