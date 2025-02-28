import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_and_reconstruct(F, Fs, num_cycles=10):
    # Time vector for original signal (high resolution)
    T = num_cycles / F  # Total time for num_cycles
    t = np.linspace(0, T, num=int(T * F * 100))  # 100 points per cycle
    
    # Original signal
    x_t = np.cos(2 * np.pi * F * t)
    
    # Sampling times
    t_sampled = np.arange(0, T, 1/Fs)
    x_sampled = np.cos(2 * np.pi * F * t_sampled)
    
    # Reconstruction using sinc interpolation
    x_r = np.zeros_like(t)
    for n, t_n in enumerate(t_sampled):
        x_r += x_sampled[n] * np.sinc((t - t_n) * Fs)
    
    # Shifted sampling times (Ts/2 shift)
    t_sampled_shifted = t_sampled + 1/(2*Fs)
    x_sampled_shifted = np.cos(2 * np.pi * F * t_sampled_shifted)
    
    # Reconstruction for shifted samples
    x_r_shifted = np.zeros_like(t)
    for n, t_n in enumerate(t_sampled_shifted):
        x_r_shifted += x_sampled_shifted[n] * np.sinc((t - t_n) * Fs)
    
    # Calculate MSE
    mse = np.mean((x_r - x_t)**2)
    mse_shifted = np.mean((x_r_shifted - x_t)**2)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Regular sampling plot
    ax1.plot(t, x_t, 'b-', label='Original', alpha=0.7)
    ax1.plot(t, x_r, 'r--', label=f'Reconstructed (MSE={mse:.6f})')
    ax1.plot(t_sampled, x_sampled, 'go', label='Samples')
    ax1.set_title(f'Regular Sampling (Fs={Fs/1e6:.1f}MHz, F={F/1e6:.1f}MHz)')
    ax1.grid(True)
    ax1.legend()
    
    # Shifted sampling plot
    ax2.plot(t, x_t, 'b-', label='Original', alpha=0.7)
    ax2.plot(t, x_r_shifted, 'r--', label=f'Reconstructed (MSE={mse_shifted:.6f})')
    ax2.plot(t_sampled_shifted, x_sampled_shifted, 'go', label='Shifted Samples')
    ax2.set_title(f'Shifted Sampling (Ts/2 shift)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig, mse, mse_shifted

# Test different sampling frequencies for F1 = 300MHz
F1 = 300e6  # 300MHz
sampling_frequencies = [500e6, 800e6, 1000e6]  # 500MHz, 800MHz, 1GHz

for Fs in sampling_frequencies:
    fig, mse, mse_shifted = generate_and_reconstruct(F1, Fs)
    plt.savefig(f'reconstruction_Fs_{int(Fs/1e6)}MHz.png', dpi=300, bbox_inches='tight')
    print(f"\nResults for Fs = {Fs/1e6}MHz:")
    print(f"Regular sampling MSE: {mse:.6f}")
    print(f"Shifted sampling MSE: {mse_shifted:.6f}")
    plt.close()

# Now test with the higher frequency F2 = 800MHz to show aliasing
F2 = 800e6  # 800MHz
fig, mse, mse_shifted = generate_and_reconstruct(F2, 500e6)  # Sampling at 500MHz
plt.savefig('reconstruction_aliasing_example.png', dpi=300, bbox_inches='tight')
print("\nAliasing Example (F2 = 800MHz, Fs = 500MHz):")
print(f"Regular sampling MSE: {mse:.6f}")
print(f"Shifted sampling MSE: {mse_shifted:.6f}")
plt.close()