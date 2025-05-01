#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# --------------------- begin vmdpy.py (self-contained) ---------------------
def VMD(signal, alpha, tau, K, DC, init, tol, Niter=500):
    """
    Minimal Variational Mode Decomposition implementation.
    Returns:
      u       : array (K, N) of modes in time domain
      u_hat   : array (K, T) of modes in freq. domain (mirrored signal)
      omega_p : array (K,) center frequencies (one per mode)
    """
    N = len(signal)
    # mirror signal to reduce boundary artifacts
    f_mirror = np.concatenate([signal[::-1], signal, signal[::-1]])
    T = len(f_mirror)
    t = np.arange(1, T+1) / T
    freqs = t - 0.5 - 1/T

    # FFT of mirrored signal
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))

    # initialize modes
    if init == 1:
        u_hat = np.tile(f_hat / K, (K, 1)).astype(complex)
    else:
        u_hat = np.random.randn(K, T) + 1j * np.random.randn(K, T)
    lambda_hat = np.zeros(T, dtype=complex)
    omega_plus = np.zeros(K)

    # iterate until convergence
    for n in range(Niter):
        u_hat_prev = u_hat.copy()
        for k in range(K):
            # residual excluding mode k
            sum_others = f_hat - np.sum(u_hat, axis=0) + u_hat[k]
            denom = 1 + alpha * (freqs - (omega_plus[k] if n>0 else 0))**2
            u_hat[k] = (sum_others + lambda_hat/2) / denom
            # update center frequency (skip DC)
            if not (DC and k == 0):
                num = np.sum(freqs * np.abs(u_hat[k])**2)
                den = np.sum(np.abs(u_hat[k])**2) + 1e-16
                omega_plus[k] = num / den
        lambda_hat += tau * (np.sum(u_hat, axis=0) - f_hat)
        # check convergence
        if np.linalg.norm(u_hat - u_hat_prev)**2 < tol:
            break

    # reconstruct modes (discard mirrors)
    u = np.zeros((K, N))
    for k in range(K):
        u_full = np.fft.ifft(np.fft.ifftshift(u_hat[k]))
        u[k, :] = np.real(u_full[N:2*N])
    return u, u_hat, omega_plus
# ---------------------- end vmdpy.py ----------------------

# 1) SYNTHETIC GPS-STYLE LANDSLIDE DISPLACEMENT
def generate_synthetic_data(N=1000, duration=10.0, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, duration, N)
    # Trend: accelerating creep
    trend = 0.1 * t**2 + 0.5 * t
    # Periodic: seasonal cycles
    periodic = 0.3 * np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.sin(2 * np.pi * 2.5 * t)
    # Random: Gaussian noise + random jumps
    noise = 0.05 * np.random.randn(N)
    jumps = np.zeros(N)
    for _ in range(5):
        idx = np.random.randint(0, N)
        jumps[idx:] += np.random.uniform(-0.2, 0.2)
    random_comp = noise + jumps
    # Composite signal
    signal = trend + periodic + random_comp
    return t, signal

# Generate and save synthetic GPS data
t, signal = generate_synthetic_data()
data = np.column_stack((t, signal))
np.savetxt('synthetic_gps.csv', data, delimiter=',', header='time,displacement', comments='')
print("Synthetic GPS data saved to 'synthetic_gps.csv'")

# 2) VMD PARAMETERS
alpha = 2000  # bandwidth constraint
tau   = 0.0   # noise tolerance
K     = 3     # number of modes
DC    = 0     # no DC mode
init  = 1     # uniform init
tol   = 1e-6

# 3) RUN VMD
u, u_hat, omega_plus = VMD(signal, alpha, tau, K, DC, init, tol)

# 4) SORT & ASSIGN MODES
order = np.argsort(omega_plus)  # low->high frequency
trend_mode    = u[order[0], :]
periodic_mode = u[order[1], :]
random_mode   = u[order[2], :]

# 5) PLOT & SAVE RESULTS
fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
axs[0].plot(t, signal, color='k')
axs[0].set_title('Original Synthetic Signal')
axs[1].plot(t, trend_mode)
axs[1].set_title('VMD Trend Mode')
axs[2].plot(t, periodic_mode)
axs[2].set_title('VMD Periodic Mode')
axs[3].plot(t, random_mode)
axs[3].set_title('VMD Random Mode')
axs[3].set_xlabel('Time (yr)')
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.savefig('vmd_decomposition.png', dpi=300)
print("Decomposition plot saved to 'vmd_decomposition.png'")
