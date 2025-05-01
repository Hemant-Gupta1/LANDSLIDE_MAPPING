import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load your CSV file
file_path = 'impact_factors.csv'  # Update with your local file path if needed
df = pd.read_csv(file_path)

# Prepare a wider, high-res figure
plt.figure(figsize=(24, 14))  # width x height
plt.rcParams.update({'font.size': 12})  # increase font size

# For each column, decompose, plot, and export
for i, col in enumerate(['rainfall', 'reservoir'], 1):
    series = pd.Series(df[col].values)
    
    # Decompose: adjust period based on your data (here assumed 12)
    result = seasonal_decompose(series, model='additive', period=12, extrapolate_trend='freq')

    # Save seasonal and residual components to CSV
    output_df = pd.DataFrame({
        'original': series,
        'seasonal': result.seasonal,
        'residual': result.resid
    })
    output_df.to_csv(f'{col}_decomposition.csv', index=False)
    print(f"Saved decomposition data to '{col}_decomposition.csv'.")

    # Plot periodic (seasonal) component
    plt.subplot(2, 2, (i - 1) * 2 + 1)
    plt.plot(result.seasonal, label='Seasonal')
    plt.title(f'{col} - Periodicity (Seasonal Component)')
    plt.xlabel('Index')
    plt.ylabel('Seasonal Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Plot random (residual) component
    plt.subplot(2, 2, (i - 1) * 2 + 2)
    plt.plot(result.resid, label='Residual', color='orange')
    plt.title(f'{col} - Randomness (Residual Component)')
    plt.xlabel('Index')
    plt.ylabel('Residual Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

# Adjust layout and save high-res plot
plt.tight_layout()
plt.savefig('impact_components.png', dpi=300)  # high-resolution save

print("Plot saved as 'impact_components.png'.")
