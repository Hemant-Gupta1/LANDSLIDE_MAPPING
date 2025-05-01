#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# --------------------- VMD Implementation (self-contained) ---------------------
def VMD(signal, alpha, tau, K, DC, init, tol, Niter=500):
    N = len(signal)
    # mirror signal
    f_mirror = np.concatenate([signal[::-1], signal, signal[::-1]])
    T = len(f_mirror)
    t_freq = np.arange(1, T+1) / T
    freqs = t_freq - 0.5 - 1/T
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    # initialize modes in frequency domain
    if init == 1:
        u_hat = np.tile(f_hat / K, (K, 1)).astype(complex)
    else:
        u_hat = np.random.randn(K, T) + 1j * np.random.randn(K, T)
    lambda_hat = np.zeros(T, dtype=complex)
    omega_plus = np.zeros(K)
    # iterative update
    for n in range(Niter):
        u_hat_old = u_hat.copy()
        for k in range(K):
            sum_others = f_hat - np.sum(u_hat, axis=0) + u_hat[k]
            denom = 1 + alpha * (freqs - (omega_plus[k] if n>0 else 0))**2
            u_hat[k] = (sum_others + lambda_hat/2) / denom
            if not (DC and k == 0):
                num = np.sum(freqs * np.abs(u_hat[k])**2)
                den = np.sum(np.abs(u_hat[k])**2) + 1e-16
                omega_plus[k] = num / den
        lambda_hat += tau * (np.sum(u_hat, axis=0) - f_hat)
        if np.linalg.norm(u_hat - u_hat_old)**2 < tol:
            break
    # reconstruct modes
    u = np.zeros((K, N))
    for k in range(K):
        u_full = np.fft.ifft(np.fft.ifftshift(u_hat[k]))
        u[k, :] = np.real(u_full[N:2*N])
    return u, omega_plus
# --------------------- End VMD ---------------------

# 1) GENERATE SYNTHETIC IMPACT FACTORS
# Time vector: 10 years sampled daily ~ 3650 points
N = 3650
t = np.linspace(0, 10, N)  # years

# a) Rainfall: seasonal monsoon cycle + noise (mm per year unit)
rainfall_seasonal = 200 + 100 * np.sin(2 * np.pi * t)  # annual cycle
rain_noise = 50 * np.random.randn(N)  # random variability
rainfall = rainfall_seasonal + rain_noise

# b) Reservoir level: responds to rainfall with smoothing and trend (m)
# base level + response to rainfall + slow drift
drip = np.convolve(rainfall, np.ones(30)/30, mode='same')  # 30-day smoothing
reservoir_trend = 0.1 * t  # slow increasing trend
reservoir = 50 + 0.05 * drip + reservoir_trend + 2 * np.random.randn(N)

# Save original synthetic data
data = np.column_stack((rainfall, reservoir))
np.savetxt('impact_factors.csv', data, delimiter=',', header='rainfall,reservoir', comments='')
print("Synthetic impact factors saved to 'impact_factors.csv'")

# 2) DECOMPOSE EACH FACTOR WITH VMD (K=2: periodic + random)
alpha, tau, K, DC, init, tol = 1000, 0.0, 2, 0, 1, 1e-6
u_rain, omega_rain = VMD(rainfall, alpha, tau, K, DC, init, tol)
u_res, omega_res = VMD(reservoir, alpha, tau, K, DC, init, tol)

# Sort and assign: mode 0 = periodic (lower freq), mode 1 = random (higher freq)
order_rain = np.argsort(omega_rain)
rain_periodic = u_rain[order_rain[0], :]
rain_random   = u_rain[order_rain[1], :]

order_res = np.argsort(omega_res)
res_periodic = u_res[order_res[0], :]
res_random   = u_res[order_res[1], :]

# 3) OPTIONAL: plot decomposition
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
axs[0].plot(t, rainfall)
axs[0].set_title('Rainfall: Periodic + Random')
axs[1].plot(t, rain_periodic)
axs[1].set_title('Rainfall Periodic Mode')
axs[2].plot(t, rain_random)
axs[2].set_title('Rainfall Random Mode')
axs[3].plot(t, reservoir)
axs[3].set_title('Reservoir Level (combined)')
axs[3].set_xlabel('Time (years)')
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.savefig('impact_vmd.png', dpi=300)
print("VMD decomposition saved to 'impact_vmd.png'")
