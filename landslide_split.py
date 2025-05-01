# #!/usr/bin/env python3
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import datetime

# # --------------------- VMD Implementation (self-contained) ---------------------
# def VMD(signal, alpha, tau, K, DC, init, tol, Niter=500):
#     N = len(signal)
#     f_mirror = np.concatenate([signal[::-1], signal, signal[::-1]])
#     T = len(f_mirror)
#     t_freq = np.arange(1, T+1) / T
#     freqs = t_freq - 0.5 - 1/T
#     f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
#     if init == 1:
#         u_hat = np.tile(f_hat / K, (K, 1)).astype(complex)
#     else:
#         u_hat = np.random.randn(K, T) + 1j * np.random.randn(K, T)
#     lambda_hat = np.zeros(T, dtype=complex)
#     omega_plus = np.zeros(K)
#     for n in range(Niter):
#         u_prev = u_hat.copy()
#         for k in range(K):
#             residual = f_hat - np.sum(u_hat, axis=0) + u_hat[k]
#             denom = 1 + alpha * (freqs - (omega_plus[k] if n>0 else 0))**2
#             u_hat[k] = (residual + lambda_hat/2) / denom
#             if not (DC and k == 0):
#                 omega_plus[k] = np.sum(freqs * np.abs(u_hat[k])**2) / (np.sum(np.abs(u_hat[k])**2) + 1e-16)
#         lambda_hat += tau * (np.sum(u_hat, axis=0) - f_hat)
#         if np.linalg.norm(u_hat - u_prev)**2 < tol:
#             break
#     u = np.zeros((K, N))
#     for k in range(K):
#         full = np.fft.ifft(np.fft.ifftshift(u_hat[k]))
#         u[k] = np.real(full[N:2*N])
#     return u, omega_plus
# # --------------------- End VMD ---------------------

# # Load data
# df = pd.read_csv('synthetic_gps.csv')  # expects columns time,displacement
# # Convert fractional years to dates starting 2004-01-01
# start = datetime.date(2004,1,1)
# dates = df['time'].apply(lambda yr: start + datetime.timedelta(days=yr*365))
# disp = df['displacement'].values

# # VMD parameters
# alpha, tau, K, DC, init, tol = 2000, 0.0, 3, 0, 1, 1e-6
# modes, omegas = VMD(disp, alpha, tau, K, DC, init, tol)
# order = np.argsort(omegas)
# trend = modes[order[0]]
# periodic = modes[order[1]]
# random_mode = modes[order[2]]

# # Create separate plots
# # 1) Original GPS Signal
# plt.figure(figsize=(8,4))
# plt.plot(dates, disp, '-k')
# plt.title('Original GPS Displacement Signal')
# plt.xlabel('Date')
# plt.ylabel('Displacement (m)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('gps_original.png', dpi=300)
# plt.close()

# # 2) Trend Component
# plt.figure(figsize=(8,4))
# plt.plot(dates, trend, '-r')
# plt.title('Trend Displacement Mode')
# plt.xlabel('Date')
# plt.ylabel('Displacement (m)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('gps_trend.png', dpi=300)
# plt.close()

# # 3) Periodic Component
# plt.figure(figsize=(8,4))
# plt.plot(dates, periodic, '-b')
# plt.title('Periodic Displacement Mode')
# plt.xlabel('Date')
# plt.ylabel('Displacement (m)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('gps_periodic.png', dpi=300)
# plt.close()

# # 4) Random Component
# plt.figure(figsize=(8,4))
# plt.plot(dates, random_mode, '-g')
# plt.title('Random Displacement Mode')
# plt.xlabel('Date')
# plt.ylabel('Displacement (m)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('gps_random.png', dpi=300)
# plt.close()

# print("Generated: gps_original.png, gps_trend.png, gps_periodic.png, gps_random.png")
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD

# 1) build the time vector
N = 110
t = np.arange(N)

# 2) construct the trend (logistic‐style)
#    starts around 100 mm, rises to ~500 mm around t≈50–100
trend = 100 + 400 / (1 + np.exp(-0.1 * (t - 50)))

# 3) add a realistic periodic sway (e.g. seasonal or hydrological)
#    period ≃ 20 samples, amplitude ≃ 25 mm
periodic = 25 * np.sin(2 * np.pi * t / 20)

# 4) overlay high‐frequency “jitter” (GPS noise, small slips…) 
#    zero‐mean Gaussian, σ ≃ 10 mm
noise = 10 * np.random.randn(N)

# 5) assemble the synthetic displacement
signal = trend + periodic + noise

# 6) run VMD to extract 3 modes
alpha = 2000       # bandwidth‐constraint
tau   = 0.0        # noise‐tolerance (0 = strict data‐fidelity)
K     = 3          # we know there are 3 components
DC    = False      # no DC‐mode enforced
init  = 1          # initialize omegas uniformly
tol   = 1e-7

u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

# 7) plot original + the three IMFs
plt.figure(figsize=(8, 6))
plt.subplot(4,1,1)
plt.plot(t, signal, 'r');  plt.title('Synthetic Landslide Displacement')
plt.ylabel('mm')

for k in range(K):
    plt.subplot(4,1,k+2)
    plt.plot(t, u[k], 'C{}'.format(k))
    plt.ylabel(f'IMF {k+1}')
    if k == K-1: 
        plt.xlabel('Sample index')

# plt.tight_layout()
plt.tight_layout()
plt.savefig('vmd_decomposition.png', dpi=300)
plt.close()

