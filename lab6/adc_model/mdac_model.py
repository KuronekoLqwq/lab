"""
fully differential pipeline adc model
"""

import numpy as np
from pipeline_adc_config import *

# consider track and hold, without auto-zeroing technique
# fully differential
# consider finite gain, offset, capacitor mismatch, comparator offset, non-linear op-amp gain, finite op-amp bandwidth
def mdac_stage(t, vin_p, vin_n, prev_residue_p=None, prev_residue_n=None, prev_dac_voltage=0, vref=1, prev_vin_p_track = 0, prev_vin_n_track=0):
    track_hold_phase = int(np.mod(t / timebase, 2))  # if even time(begin at 0), then track, else hold
        
    # Initialize output_code to a default value to avoid UnboundLocalError
    output_code = 0
    dac_voltage = 0
    dac_levels = np.array([-1, -2/3, -1/3, 0, 1/3, 2/3, 1]) * vref
   
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
        residue_p_aim = ((C1real + C2real) * prev_vin_p_track - C2real * prev_dac_voltage + (C1real + C2real) / real_gain * op_offset) / (C1real - (C1real + C2real) / real_gain)
        residue_n_aim = ((C1real + C2real) * prev_vin_n_track - C2real * (- prev_dac_voltage) + (C1real + C2real) / real_gain * op_offset) / (C1real - (C1real + C2real) / real_gain)

        # consider opamp bw, make an attenuation like track phase
        residue_p = residue_p_aim * (1 - np.exp( - timebase*loop_bw))
        residue_n = residue_n_aim * (1 - np.exp( - timebase*loop_bw))

        # hold the output of the last track phase based on dac voltage
        output_code = np.argmin(np.abs(dac_levels - prev_dac_voltage))

        # In the hold phase, we don't produce a new output code
        # So we return the previous code (default 0)
        return output_code, residue_p, residue_n, dac_voltage, 0, 0
    


"""
1-bit MDAC (Multiplying Digital-to-Analog Converter) stage model
For pipeline ADC implementation

This model implements a 1-bit MDAC stage with:
- Track and hold functionality
- Single comparator per stage
- Fully differential architecture
- Modeling of non-idealities:
  - Finite op-amp gain
  - Op-amp offset
  - Capacitor mismatch
  - Comparator offset
  - Finite bandwidth
  - Non-linear op-amp gain
"""

def mdac_1bit_stage(t, vin_p, vin_n, prev_residue_p=None, prev_residue_n=None, prev_dac_voltage=0, vref=1, prev_vin_p_track=0, prev_vin_n_track=0):
    """
    1-bit MDAC stage model
    
    Parameters:
    - t: Current time
    - vin_p: Positive input voltage
    - vin_n: Negative input voltage
    - prev_residue_p: Previous positive residue (for hold phase)
    - prev_residue_n: Previous negative residue (for hold phase)
    - prev_dac_voltage: Previous DAC voltage (for hold phase)
    - vref: Reference voltage
    - prev_vin_p_track: Previous positive input during track phase
    - prev_vin_n_track: Previous negative input during track phase
    
    Returns:
    - output_code: Digital output code (0 or 1)
    - residue_p: Positive residue voltage
    - residue_n: Negative residue voltage
    - dac_voltage: DAC output voltage
    - vin_p_track: Tracked positive input
    - vin_n_track: Tracked negative input
    """
    track_hold_phase = int(np.mod(t / timebase, 2))  # if even time(begin at 0), then track, else hold
    
    # Initialize output_code to a default value
    output_code = 0
    dac_voltage = 0
    
    if track_hold_phase == 0:  # Track phase
        # Apply finite bandwidth effect during tracking
        vin_p_track = vin_p * (1 - np.exp(-timebase/tao))
        vin_n_track = vin_n * (1 - np.exp(-timebase/tao))
        
        # For 1-bit MDAC, only one comparator with threshold at zero differential
        # Add comparator offset to the threshold
        threshold_p = 0 + comp_offset
        threshold_n = 0 + comp_offset
        
        # Compare differential input to threshold
        p_diff = vin_p_track - threshold_p
        n_diff = vin_n_track - threshold_n
        comparison = p_diff > n_diff
        
        # Convert comparison result to output code (0 or 1)
        output_code = 1 if comparison else 0
        
        # DAC voltage based on 1-bit decision
        # For 1-bit, we use either -vref/2 or +vref/2
        dac_voltage = vref/2 if output_code == 1 else -vref/2
        
        # During track phase, residue outputs are floating and retain previous values
        residue_p = prev_residue_p if prev_residue_p is not None else 0
        residue_n = prev_residue_n if prev_residue_n is not None else 0
        
        return output_code, residue_p, residue_n, dac_voltage, vin_p_track, vin_n_track
    
    else:  # Hold/amplify phase
        # Use the previous DAC voltage for residue calculation
        dac_voltage = prev_dac_voltage
        
        # Calculate differential held voltage for gain modeling
        vhold_diff = prev_vin_p_track - prev_vin_n_track
        vhold_nominal = vhold_diff / (2*fullscale)
        
        # Polynomial op-amp finite gain model
        real_gain = gain * (1 + 0.1*vhold_nominal**2 + 0.2*vhold_nominal**3 + 0.15*vhold_nominal**4 + 0.1*vhold_nominal**5)
        
        # Calculate residue with non-idealities
        # For 1-bit MDAC, the ideal gain is 2
        # residue = 2*(vin - dac_voltage)
        
        # Consider vos, finite gain, cap mismatch
        residue_p_aim = ((C1real + C3real) * prev_vin_p_track - C3real * prev_dac_voltage + (C1real + C3real) / real_gain * op_offset) / (C1real - (C1real + C3real) / real_gain)
        residue_n_aim = ((C1real + C3real) * prev_vin_n_track - C3real * (-prev_dac_voltage) + (C1real + C3real) / real_gain * op_offset) / (C1real - (C1real + C3real) / real_gain)
        
        # Apply finite bandwidth effect on the residue
        residue_p = residue_p_aim * (1 - np.exp(-timebase*loop_bw))
        residue_n = residue_n_aim * (1 - np.exp(-timebase*loop_bw))

        dac_levels = [-1/2, 1/2]
        output_code = np.argmin(np.abs(dac_levels - prev_dac_voltage))

        # In the hold phase, we don't produce a new output code
        return output_code, residue_p, residue_n, dac_voltage, 0, 0