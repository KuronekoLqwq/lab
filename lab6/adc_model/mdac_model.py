"""
fully differential pipeline adc model
"""

import numpy as np
from pipeline_adc_config import *

# consider track and hold, without auto-zeroing technique
# fully differential
# consider finite gain, offset, capacitor mismatch, comparator offset, non-linear op-amp gain, finite op-amp bandwidth
def mdac_stage(t, vin_p, vin_n, C1real, C2real, prev_residue_p=None, prev_residue_n=None, prev_dac_voltage=0, vref=1, prev_vin_p_track = 0, prev_vin_n_track=0):
    track_hold_phase = int(np.mod(t / timebase, 2))  # if even time(begin at 0), then track, else hold
        
    # Initialize output_code to a default value to avoid UnboundLocalError
    output_code = 0
    dac_voltage = 0
    
    if track_hold_phase == 0:  # Track phase
        # Apply finite bandwidth effect during tracking
        vin_p_track = vin_p * (1 - np.exp( - timebase/tao))
        vin_n_track = vin_n * (1 - np.exp( - timebase/tao))

        # Flash ADC & DAC ideal conversion
        thresholds_p = np.array([-0.625, -0.375, -0.125, 0.125, 0.375, 0.625]) * vref + comp_offset
        thresholds_n = np.array([0.625, 0.375, 0.125, -0.125, -0.375, -0.625]) * vref + comp_offset

        comparator_outputs = []
        for i in range(len(thresholds_p)):
            p_diff = vin_p_track - thresholds_p[i]
            n_diff = vin_n_track - thresholds_n[i]
            comparison = p_diff > n_diff
            comparator_outputs.append(comparison)

        # The thermometer code is the number of 'True' comparisons
        output_code = np.sum(comparator_outputs)

        # Clip to right range
        output_code = max(0, min(output_code, 6))

        # Consider cap mismatch
        dac_levels = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]) * vref
        dac_voltage = dac_levels[output_code]
        
        # During track phase, residue outputs are floating and retain previous values
        # If no previous values provided, initialize to zero or common-mode voltage
        residue_p = prev_residue_p if prev_residue_p is not None else 0
        residue_n = prev_residue_n if prev_residue_n is not None else 0
        
        return output_code, residue_p, residue_n, dac_voltage, vin_p_track, vin_n_track
        
    else:  # Hold/amplify phase
        # Use the previous DAC voltage for residue calculation
        # This accurately models the held voltage on capacitors
        dac_voltage = prev_dac_voltage

        # suppose there is a mismatch on C1 and C2 with same std dev
        vhold_diff = prev_vin_p_track - prev_vin_n_track
        vhold_nominal = vhold_diff / (2*fullscale)

        # polynomial op-amp finite gain model
        #real_gain = gain * (1 + 0.1*vhold_nominal**2 + 0.2*vhold_nominal**3 + 0.15*vhold_nominal**4 + 0.1*vhold_nominal**5)
        real_gain = gain

        # consider vos, finite gain, cap mismatch
        residue_p = ((C1real + C2real) * prev_vin_p_track - C2real * prev_dac_voltage + (C1real + C2real) / real_gain * op_offset) / (C1real - (C1real + C2real) / real_gain)
        residue_n = ((C1real + C2real) * prev_vin_n_track - C2real * (- prev_dac_voltage) + (C1real + C2real) / real_gain * op_offset) / (C1real - (C1real + C2real) / real_gain)

        # In the hold phase, we don't produce a new output code
        # So we return the previous code (default 0)
        return output_code, residue_p, residue_n, dac_voltage, 0, 0