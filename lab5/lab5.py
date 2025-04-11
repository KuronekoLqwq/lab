import numpy as np
import matplotlib.pyplot as plt
import os 

if not os.path.exists('./images'):
    os.mkdir('./images')

# Histogram result data
histogram_result = [43, 115, 85, 101, 122, 170, 75, 146, 125, 60, 95, 95, 115, 40, 120, 242] 
total_sum = np.sum(histogram_result)
cumulative_sums = np.cumsum(histogram_result)

# Calculate ideal histogram (uniform distribution)
histogram_result_ideal = np.full(len(histogram_result), total_sum / len(histogram_result))
cumulative_sums_ideal = np.cumsum(histogram_result_ideal)

# Create transition arrays
transition = np.zeros(total_sum)
transition_ideal = np.zeros(total_sum)

# Fill transition arrays
for i in range(total_sum):
    # Find the bin where this index belongs in actual data
    bin_index = 0
    while bin_index < len(cumulative_sums) and i >= cumulative_sums[bin_index]:
        bin_index += 1
    
    # Find the bin where this index belongs in ideal data
    bin_index_ideal = 0
    while bin_index_ideal < len(cumulative_sums_ideal) and i >= cumulative_sums_ideal[bin_index_ideal]:
        bin_index_ideal += 1
    
    transition[i] = bin_index
    transition_ideal[i] = bin_index_ideal

# Convert x-axis to LSB scale (0 to 1 LSB full scale)
# For a 4-bit ADC, 1 LSB = full scale / (2^4 - 1) = full scale / 15
x_lsb = np.linspace(0, 1, total_sum)  # Normalized LSB scale from 0 to 1

plt.figure(figsize=(10, 6))
plt.plot(x_lsb, transition, label='Actual')
plt.plot(x_lsb, transition_ideal, color='red', linestyle='--', label='Ideal')
plt.title('ADC Transition Function')
plt.xlabel('Input (FS)')
plt.ylabel('Output Code')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('./images/transition_function_lsb.png')
plt.close()

# Calculate and print transition points
print("Transition points (where output code changes):")
transitions = []
for code in range(1, 16):  # For each output code transition
    # Find first index where transition equals code
    indices = np.where(transition == code)[0]
    if len(indices) > 0:
        first_idx = indices[0]
        lsb_value = x_lsb[first_idx]
        transitions.append((code - 1, code, lsb_value))
        print(f"Code {code-1} to {code}: {lsb_value:.4f} LSB")

# Calculate ideal transition points (uniformly distributed)
ideal_transitions = [(code, code+1, code/15) for code in range(15)]

# Calculate and visualize DNL based on transition points
if transitions:
    dnl_from_transitions = []
    ideal_step = 1.0 / 15  # Ideal step size in LSB
    
    for i in range(1, len(transitions)):
        current_step = transitions[i][2] - transitions[i-1][2]
        dnl = (current_step / ideal_step) - 1
        dnl_from_transitions.append(dnl)
    
    # Plot DNL calculated from transition function
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(dnl_from_transitions) + 1), dnl_from_transitions)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('DNL from Transition Function')
    plt.xlabel('Code')
    plt.ylabel('DNL (LSB)')
    plt.grid(True, alpha=0.3)
    plt.savefig('./images/dnl_from_transitions.png')
    
    # Calculate INL from the transition points
    # Method 1: INL = cumulative sum of DNL
    inl_from_dnl = np.cumsum(dnl_from_transitions)
    
    # Apply endpoint correction (make first and last points 0)
    inl_correction = np.linspace(0, inl_from_dnl[-1], len(inl_from_dnl))
    inl_from_dnl_corrected = inl_from_dnl - inl_correction
    
    # Plot INL calculated from transition function
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(inl_from_dnl_corrected) + 1), inl_from_dnl_corrected)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('INL from Cumulative DNL (End-Point Corrected)')
    plt.ylabel('INL (LSB)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./images/inl_from_transitions.png')
    plt.close()
    
    # Print DNL and INL statistics
    print("\nDNL Statistics from Transition Function:")
    print(f"Min DNL: {min(dnl_from_transitions):.4f} LSB")
    print(f"Max DNL: {max(dnl_from_transitions):.4f} LSB")
    print(f"Peak DNL: {max(abs(min(dnl_from_transitions)), abs(max(dnl_from_transitions))):.4f} LSB")
    
    print("\nINL Statistics from Transition Function:")
    print(f"Min INL: {min(inl_from_dnl_corrected):.4f} LSB")
    print(f"Max INL: {max(inl_from_dnl_corrected):.4f} LSB")
    print(f"Peak INL: {max(abs(min(inl_from_dnl_corrected)), abs(max(inl_from_dnl_corrected))):.4f} LSB")




# 5
# Given DNL values in terms of LSB
dnl = np.array([0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0])

# Offset error and full-scale error
offset_error = 0.5  # LSB
full_scale_error = 0.5  # LSB

# a) Find the INL for this ADC
# INL is the cumulative sum of DNL values
inl = np.cumsum(dnl)
print("INL values (in LSB):", inl)

# b) Draw the transfer curve of this ADC
# For a 3-bit ADC, we have 2^3 = 8 codes (0 to 7)
codes = np.arange(8)

# Ideal transfer function (straight line)
# For an ideal 3-bit ADC with code width = 1 LSB
ideal_outputs = np.arange(8)

# Calculate actual outputs considering DNL
# Start with the offset-adjusted first code
actual_outputs = np.zeros(8)
actual_outputs[0] = 0 + offset_error  # First code with offset error

# Calculate the actual output for each code
for i in range(1, 8):
    # Each code width is 1 LSB + DNL
    actual_outputs[i] = actual_outputs[i-1] + (1 + dnl[i-1])

# Plot 1: Transfer Curve
plt.figure(figsize=(10, 6))

# Plot ideal transfer function
plt.plot(codes, ideal_outputs, '--', label='Ideal Transfer Function', color='blue')

# Plot actual transfer function
plt.step(codes, actual_outputs, 'o-', label='Actual Transfer Function', color='red', where='post')

# Add grid, labels and title
plt.grid(True)
plt.xlabel('Input Code')
plt.ylabel('Output (LSB)')
plt.title('3-bit ADC Transfer Curve')
plt.legend()

# Annotate INL values
for i in range(8):
    plt.annotate(f"INL={inl[i]}", (codes[i], actual_outputs[i]), 
                 xytext=(5, 5), textcoords='offset points')

# Save the transfer curve plot
plt.savefig('./images/5_transfer.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: INL Plot
plt.figure(figsize=(10, 6))
plt.bar(codes, inl, color='green', alpha=0.7)
plt.grid(True, axis='y')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Code')
plt.ylabel('INL (LSB)')
plt.title('Integral Non-Linearity (INL) for 3-bit ADC')

# Annotate INL values
for i, val in enumerate(inl):
    plt.text(i, val + (0.1 if val >= 0 else -0.2), f"{val:.1f}", 
             ha='center', va='center')

# Save the INL plot
plt.savefig('./images/5_inl.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary
print("\nSummary:")
print(f"Offset Error: {offset_error} LSB")
print(f"Full Scale Error: {full_scale_error} LSB")
print(f"Maximum INL: {np.max(np.abs(inl))} LSB")
print(f"Occurs at code: {np.argmax(np.abs(inl))}")