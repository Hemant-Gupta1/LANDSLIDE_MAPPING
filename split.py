# import numpy as np
# import matplotlib.pyplot as plt
# from vmdpy import VMD

# # 1) time base
# N = 110
# t = np.arange(N)

# # 2) smooth “creep” trend (logistic + a tiny random drift)
# trend = 100 + 400 / (1 + np.exp(-0.1 * (t - 50)))
# trend += np.cumsum(0.05*np.random.randn(N))  # slow random walk

# # 3) seasonal/hydro‐driven periodic oscillation
# base_period = 20
# # slowly modulate its amplitude
# amp_mod = 1 + 0.3*np.sin(2*np.pi*t/100)
# periodic = (25 * amp_mod) * np.sin(2*np.pi * t / base_period)

# # 4) coloured GPS noise (AR(1))
# rho = 0.8
# white = np.random.randn(N)*5
# ar_noise = np.zeros(N)
# for i in range(1, N):
#     ar_noise[i] = rho*ar_noise[i-1] + white[i]

# # 5) random micro‐slip pulses
# pulses = np.zeros(N)
# for _ in range(5):   # inject 5 random slip events
#     center = np.random.randint(10, N-10)
#     width = np.random.randint(2, 8)
#     height = np.random.uniform(5, 30)
#     pulse_shape = height * np.exp(-((t-center)/width)**2)
#     pulses += pulse_shape

# # 6) assemble
# signal = trend + periodic + ar_noise + pulses

# # 7) decompose with VMD (3 modes)
# alpha, tau, K, DC, init, tol = 2000, 0.0, 3, False, 1, 1e-7
# u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

# # 8) plot & save
# fig, axs = plt.subplots(4,1, figsize=(8,6), sharex=True)
# axs[0].plot(t, signal, 'k')
# axs[0].set_ylabel('mm')
# axs[0].set_title('Synthetic Landslide Displacement (with realistic quirks)')
# for k in range(K):
#     axs[k+1].plot(t, u[k], 'C{}'.format(k))
#     axs[k+1].set_ylabel(f'IMF {k+1}')
# axs[-1].set_xlabel('Sample index')
# plt.tight_layout()
# plt.savefig('synthetic_landslide_vmd.png', dpi=300)
# plt.close()
# print("☑ Saved decomposition to synthetic_landslide_vmd.png")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD

# 1) time base
N = 110
t = np.arange(N)

# 2) smooth “creep” trend (logistic + a tiny random drift)
trend = 100 + 400 / (1 + np.exp(-0.1 * (t - 50)))
trend += np.cumsum(0.05 * np.random.randn(N))  # slow random walk

# 3) seasonal/hydro‑driven periodic oscillation
base_period = 20
amp_mod = 1 + 0.3 * np.sin(2 * np.pi * t / 100)  # slowly modulated amplitude
periodic = (25 * amp_mod) * np.sin(2 * np.pi * t / base_period)

# 4) coloured GPS noise (AR(1))
rho = 0.8
white = np.random.randn(N) * 5
ar_noise = np.zeros(N)
for i in range(1, N):
    ar_noise[i] = rho * ar_noise[i-1] + white[i]

# 5) random micro‑slip pulses
pulses = np.zeros(N)
for _ in range(5):
    center = np.random.randint(10, N-10)
    width = np.random.randint(2, 8)
    height = np.random.uniform(5, 30)
    pulses += height * np.exp(-((t - center) / width) ** 2)

# 6) assemble synthetic displacement
signal = trend + periodic + ar_noise + pulses

# 7) save synthetic components to CSV
df_comp = pd.DataFrame({
    'time': t,
    'signal': signal,
    'trend': trend,
    'periodic': periodic,
    'ar_noise': ar_noise,
    'pulses': pulses
})
df_comp.to_csv('synthetic_components.csv', index=False)

# 8) VMD decomposition (3 modes)
alpha, tau, K, DC, init, tol = 2000, 0.0, 3, False, 1, 1e-7
u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

# 9) save IMFs to CSV
df_imfs = pd.DataFrame({'time': t})
for k in range(K):
    df_imfs[f'IMF{k+1}'] = u[k]
df_imfs.to_csv('vmd_imfs.csv', index=False)

# 10) plot & save figure
fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
axs[0].plot(t, signal, 'k')
axs[0].set_ylabel('mm')
axs[0].set_title('Synthetic Landslide Displacement (with CSV output)')
for k in range(K):
    axs[k+1].plot(t, u[k], f'C{k}')
    axs[k+1].set_ylabel(f'IMF {k+1}')
axs[-1].set_xlabel('Sample index')
plt.tight_layout()
plt.savefig('synthetic_landslide_vmd.png', dpi=300)
plt.close()

print("☑ Saved: synthetic_components.csv, vmd_imfs.csv, synthetic_landslide_vmd.png")
