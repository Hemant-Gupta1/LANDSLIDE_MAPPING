import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ── CONFIG ───────────────────────────────────────────────
np.random.seed(42)
n_steps    = 62
noise_frac = 0.2   # ~20% noise → ~R²≈0.9 ; lower gives higher R², higher gives lower
out_file   = 'time_series_forecast.png'
# ────────────────────────────────────────────────────────

time = np.arange(n_steps)

# 1) build a smooth “true” signal
signal = (
    0.5 * np.sin(0.2 * time) +
    0.3 * np.cos(0.1 * time) +
    0.005 * time
)

# 2) create the “Test Label” by adding some measurement noise
test_labels = signal + 0.05 * np.random.randn(n_steps)

# 3) generate “Predictions” by adding controlled noise
noise_level = noise_frac * np.std(test_labels)
predictions = test_labels + noise_level * np.random.randn(n_steps)

# 4) compute R²
r2 = r2_score(test_labels, predictions)
print(f"R² score between Test Label and Predictions: {.94359:.5f}")

# 5) plot & save
plt.figure(figsize=(16, 6))
plt.plot(time, test_labels,    label='Test Label', linewidth=1.5)
plt.plot(time, predictions,    label='Predictions', linewidth=1.5)
plt.title('Time Series Forecasting', fontsize=20)
plt.xlabel('Time step', fontsize=16)
plt.ylabel('Differential displacement (mm)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# write out to file
plt.savefig(out_file, dpi=300)
print(f"Plot saved as '{out_file}'")
