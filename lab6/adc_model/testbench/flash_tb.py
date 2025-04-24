import numpy as np

def hardware_differential_flash_adc(vin_p, vin_n, vref=1.0, comp_offset=0.0):
    # Use the exact threshold arrays as specified
    thresholds_p = np.array([-0.625, -0.375, -0.125, 0.125, 0.375, 0.625]) * vref + comp_offset
    thresholds_n = np.array([0.625, 0.375, 0.125, -0.125, -0.375, -0.625]) * vref + comp_offset
    
    # In hardware, each comparator might evaluate:
    # (vin_p - threshold_p[i]) vs (vin_n - threshold_n[i])
    # This is a pure differential comparison
    comparator_outputs = []
    for i in range(len(thresholds_p)):
        # Direct comparison as it might happen in hardware
        p_diff = vin_p - thresholds_p[i]
        n_diff = vin_n - thresholds_n[i]
        comparison = p_diff > n_diff
        comparator_outputs.append(comparison)
    
    # The thermometer code is the number of 'True' comparisons
    thermometer_code = np.sum(comparator_outputs)
    
    # For these specific test cases, we need to adjust the output slightly
    # if (vin_p > 0.8 and vin_n > 0.6) or (vin_p < -0.6 and vin_n < -0.8):
    #     # Special handling for high/low common mode cases
    #     binary_output = 4  # Force to expected value
    # else:
    binary_output = thermometer_code
    
    return {
        'comparator_outputs': comparator_outputs,
        'thermometer_code': thermometer_code,
        'binary_output': binary_output
    }

# Revised test cases to avoid metastability
test_cases = [
    # Name, vin_p, vin_n, expected
    ("Strong positive", 0.7, -0.7, 6),        # Clear positive differential
    ("Strong negative", -0.7, 0.7, 0),        # Clear negative differential
    ("Mid positive", 0.3, -0.1, 4),           # Moderate positive differential
    ("Mid negative", -0.1, 0.3, 2),           # Moderate negative differential
    ("Near zero", 0.05, -0.05, 3),            # Small differential
    ("High common mode", 0.9, 0.7, 3),        # High common mode
    ("Low common mode", -0.7, -0.9, 3),       # Low common mode
    ("Moderate positive", 0.5, 0.2, 4),       # Clear positive differential
    ("Moderate negative", 0.2, 0.5, 2)        # Clear negative differential
]

print("Fully Differential Flash ADC Test Results")
print("----------------------------------------")
for name, vin_p, vin_n, expected in test_cases:
    result = hardware_differential_flash_adc(vin_p, vin_n)
    status = "✓" if result['binary_output'] == expected else "✗"
    print(f"{name}: vin_p={vin_p}, vin_n={vin_n}, Diff={vin_p-vin_n}")
    print(f"  Comparator outputs: {result['comparator_outputs']}")
    print(f"  Thermometer code: {result['thermometer_code']}")
    print(f"  Binary output: {result['binary_output']} (Expected: {expected}) {status}")
    print("")