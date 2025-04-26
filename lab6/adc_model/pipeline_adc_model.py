"""
13-bit Pipeline ADC Implementation using 2.5 bits/stage with 6 stages
"""

import numpy as np
from mdac_model import mdac_stage, mdac_1bit_stage
from pipeline_adc_config import *


def cascaded_mdac(t, vinp, vinn, vref, stages, prev_residue_p, prev_residue_n, prev_dac_voltage, prev_vin_p_track, prev_vin_n_track):
    track_hold_phase = int(np.mod(t / timebase, 2))  # if even time(begin at 0), then track, else hold

    # 6 2.5b stages + 1 1bit stages
    code_output_25b = np.zeros((stages-1, 1))
    code_last = np.zeros(1)

    # intermediate variables
    dac_voltage = np.zeros(stages)
    residue_p = np.zeros(stages)
    residue_n = np.zeros(stages)
    vin_p_track = np.zeros(stages)
    vin_n_track = np.zeros(stages)

    # mdac stage connection
    if track_hold_phase == 0: 
        code_output_25b[0], _, _, dac_voltage[0], vin_p_track[0], vin_n_track[0] = mdac_stage(
        t, vinp, vinn, vref = vref
        )
        
        for i in range(1, stages - 1):
            code_output_25b[i], _, _, dac_voltage[i], vin_p_track[i], vin_n_track[i] = mdac_stage(
                t, prev_residue_p[i-1], prev_residue_n[i-1], vref = vref
            )

        code_last, _, _, dac_voltage[stages-1], vin_p_track[stages-1], vin_n_track[stages-1] = mdac_1bit_stage(
            t, prev_residue_p[stages-2], prev_residue_n[stages-2], vref = vref
        )

        return code_output_25b,code_last,_,_,vin_p_track,vin_n_track

    else:
        _, residue_p[0], residue_n[0], _, _, _, = mdac_stage(
            t, vinp, vinn, prev_dac_voltage=prev_dac_voltage[0],vref=vref,prev_vin_p_track=prev_vin_p_track[0],prev_vin_n_track=prev_vin_n_track[0] 
        )

        for i in range(1, stages - 1): 
            _, residue_p[i], residue_n[i], _, _, _, = mdac_stage(
                t, prev_residue_p[i-1], prev_residue_n[i-1], prev_dac_voltage=prev_dac_voltage[i],vref=vref,prev_vin_p_track=prev_vin_p_track[i],prev_vin_n_track=prev_vin_n_track[i] 
            )
        
        _, residue_p[stages-1], residue_n[stages-1], _, _, _, = mdac_1bit_stage(
            t, prev_residue_p[stages-2], prev_residue_n[stages-2], prev_dac_voltage=prev_dac_voltage[stages-1],vref=vref,prev_vin_p_track=prev_vin_p_track[stages-1],prev_vin_n_track=prev_vin_n_track[stages-1] 
        )
        return code_output_25b, code_last, residue_p, residue_n, _,_


# time align delay line
def delay_line(t, code_output_25b, code_last, stages,prev_delays):

    track_hold_phase = int(np.mod(t / timebase, 2))  # if even time(begin at 0), then track, else hold

    delays = np.zeros((stages-1, stages-1,1))

    output_codes = np.zeros((stages-1,1))
    if track_hold_phase ==0 :
        # delay matrix moving
        for j in range(stages-1):
            delays[j][0][:] = code_output_25b[0][:]

        for j in range(stages-1):
            for k in range(1,stages-1):
                delays[j][k][:] = prev_delays[j][k-1][:]

        # output 
        for j in range(stages-1):
            output_codes[j][:] = delays[j][stages-j-2][:]

        return output_codes,delays,code_last
    
    else:
        # directly get output from prev delays
        for j in range(stages-1):
            output_codes[j][:] = prev_delays[j][stages-j-2][:]        
        delays = prev_delays
        return output_codes,delays,code_last


def digital_adding(output_codes,stages,code_last): 
    digital_out_codes = code_last
    for j in range(stages-1):
            digital_out_codes += (4**(stages-2-j))*output_codes[stages-2-j]
    
    return digital_out_codes

def pipeline_adc(t, vinp, vinn, vref, prev_residue_p, prev_residue_n, prev_dac_voltage, prev_vin_p_track, prev_vin_n_track,prev_delays):
    stages = 7

    code_output_25b = np.zeros((stages-1,1))
    code_last = np.zeros(1)

    # intermediate variables
    dac_voltage = np.zeros(stages)
    residue_p = np.zeros(stages)
    residue_n = np.zeros(stages)
    vin_p_track = np.zeros(stages)
    vin_n_track = np.zeros(stages)

    delays = np.zeros((stages-1, stages-1,1))

    output_codes = np.zeros((stages-1,1))
    
    code_output_25b, code_last, residue_p, residue_n, vin_p_track, vin_n_track = cascaded_mdac(
        t,vinp,vinn,vref, stages,prev_residue_p, prev_residue_n, prev_dac_voltage, prev_vin_p_track, prev_vin_n_track
    )

    output_codes, delays, code_last = delay_line(
        t,code_output_25b,code_last,stages,prev_delays
    )

    digital_output_codes = digital_adding(output_codes,stages,code_last)

    return digital_output_codes,residue_p,residue_n,dac_voltage,vin_p_track,vin_n_track,delays