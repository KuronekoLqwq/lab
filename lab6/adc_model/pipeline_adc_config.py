"""
configuration for adc
"""

# ADC spec
fs = 500e6
timebase = 1/(2*fs) # each sampling has 2 events: track&hold, so timebase should be half of 1/fs
fullscale = 1
num_stages = 6

C1 = 200e-15 # feedback cap
C2 = 800e-15 # dac cap
# bit per stage is 2.5, fixed

# test spec
fin = 200e6 # asked input freq

N = 2**14 # points of FFT
# fin = fs / N * 7 # best input for testing
length = int(1/fin / timebase * 10000) # 10000 cycles simulation length

# static errors in mdac: 
gain = 1e4 # in volts
op_offset = 0
cap_mismatch_sigma = 0 # std deviation
comp_offset = 0
loop_bw = 5e9 # in Hz, the loop bw of mdac
tao = 0.25e-10 # in s, the time constant of S&H
nonlinear_coefficients = [1,0.1,0.2,0.15,0.1]