"""
MDAC Transfer Function Analysis Testbench with Correct Ranges
- Updated with correct ±2V input/output ranges
- No limits on the plot y-axis to show full behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_adc_config import *
from mdac_model import mdac_stage,mdac_1bit_stage

# Create results directory if it doesn't exist
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'mdac_analysis'))
os.makedirs(results_dir, exist_ok=True)


def analyze_mdac():
    """Analyze the MDAC to produce residue and code plots"""
    
    print("Analyzing 2.5-bit MDAC with redundancy...")
    
    # Set up input voltage sweep (differential input)
    num_points = 1001
    vin_diff_range = np.linspace(-fullscale, fullscale, num_points)  # Updated to ±2V range
    
    # Initialize arrays to store results
    output_codes = np.zeros(num_points, dtype=int)
    residue_diffs = np.zeros(num_points)
    dac_voltages = np.zeros(num_points)


    # First track phase (t = 0)
    for i, vin_diff in enumerate(vin_diff_range):
        # Set differential inputs properly
        vin_p = vin_diff
        vin_n = -vin_diff
        
        # Run track phase
        t_track = 0
        code, _, _, dac_v, vin_p_track, vin_n_track = mdac_stage(
            t_track, vin_p, vin_n, vref=fullscale
        )
        
        # Store results
        output_codes[i] = code
        dac_voltages[i] = dac_v
        
        # Run hold phase to get residue
        t_hold = timebase
        _, residue_p, residue_n, _, _, _ = mdac_stage(
            t_hold, vin_p, vin_n, 
            prev_residue_p=0, prev_residue_n=0,
            prev_dac_voltage=dac_v, vref=fullscale,
            prev_vin_p_track=vin_p_track, prev_vin_n_track=vin_n_track
        )
        
        # Store residue
        residue_diffs[i] = residue_p - residue_n
        
        # Print progress
        if (i+1) % (num_points//10) == 0:
            print(f"Progress: {100*(i+1)/num_points:.1f}% complete")
    
    print("Analysis complete. Generating plots...")
    
    # Calculate residue statistics
    max_residue = np.max(residue_diffs)
    min_residue = np.min(residue_diffs)
    print(f"Residue range: Min = {min_residue:.4f}V, Max = {max_residue:.4f}V")
    
    # Figure 1: Output Codes (with full range)
    plt.figure(figsize=(12, 6))
    plt.step(vin_diff_range*2, output_codes, 'b-', where='post', linewidth=2)
    plt.title('2.5-bit MDAC Output Codes', fontsize=14)
    plt.xlabel('Differential Input Voltage (V)', fontsize=12)
    plt.ylabel('Output Code', fontsize=12)
    plt.grid(True)
    plt.xlim(-2, 2)  # Set x-axis to ±2V
    # No y-axis limits as requested
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mdac_output_codes.png'), dpi=300)
    print(f"Output codes plot saved to {os.path.join(results_dir, 'mdac_output_codes.png')}")
    
    # Figure 2: Residue
    plt.figure(figsize=(12, 6))
    plt.plot(vin_diff_range*2, residue_diffs, 'b-', linewidth=2)
    plt.title('2.5-bit MDAC Residue', fontsize=14)
    plt.xlabel('Differential Input Voltage (V)', fontsize=12)
    plt.ylabel('Differential Residue (V)', fontsize=12)
    plt.grid(True)
    plt.xlim(-2, 2)  # Set x-axis to ±2V
    # No y-axis limits as requested
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mdac_residue.png'), dpi=300)
    print(f"Residue plot saved to {os.path.join(results_dir, 'mdac_residue.png')}")
    
    # Save the data as CSV for further analysis
    data = np.column_stack((vin_diff_range, output_codes, residue_diffs, dac_voltages))
    header = "Input_Voltage,Output_Code,Residue,DAC_Voltage"
    np.savetxt(os.path.join(results_dir, 'mdac_transfer_data.csv'), 
               data, delimiter=',', header=header, comments='')
    print(f"Data saved to {os.path.join(results_dir, 'mdac_transfer_data.csv')}")
    
    # Save a text file with analysis summary
    with open(os.path.join(results_dir, 'mdac_analysis.txt'), 'w') as f:
        f.write("MDAC Transfer Function Analysis\n")
        f.write("==============================\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Full Scale Voltage: {fullscale:.2f} V\n")
        f.write(f"- Op-Amp Gain: {gain:.1e} V/V\n")
        f.write(f"- Op-Amp Offset: {op_offset*1e3:.2f} mV\n")
        f.write(f"- Capacitor Values: C1 = {C1*1e15:.2f} fF, C2 = {C2*1e15:.2f} fF\n")
        f.write(f"- Capacitor Mismatch: {cap_mismatch_sigma*100:.2f}%\n")
        f.write(f"- Comparator Offset: {comp_offset*1e3:.2f} mV\n")
        f.write(f"- Time Constant: {tao*1e9:.2f} ns\n\n")
        
        # Calculate some basic statistics
        unique_codes = np.unique(output_codes)
        f.write(f"Results:\n")
        f.write(f"- Number of output codes: {len(unique_codes)}\n")
        f.write(f"- Output codes observed: {', '.join(map(str, unique_codes))}\n")
        f.write(f"- Maximum residue: {max_residue:.4f} V\n")
        f.write(f"- Minimum residue: {min_residue:.4f} V\n")
    
    print(f"Analysis summary saved to {os.path.join(results_dir, 'mdac_analysis.txt')}")
    
    # Return data for further analysis if needed
    return {
        'vin': vin_diff_range,
        'codes': output_codes,
        'residue': residue_diffs,
        'dac': dac_voltages
    }

if __name__ == "__main__":
    # Run the analysis
    results = analyze_mdac()
    plt.show()