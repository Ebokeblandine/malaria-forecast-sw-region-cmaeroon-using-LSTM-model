import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("processed_malaria_dataset.csv")

cols = [
    "Confirmed_Malaria_Cases",
    "Cases_lag1",
    "Cases_lag2",
    "Rainfall_mm",
    "Rainfall_lag1",
    "Rainfall_lag2"
]

corr = df[cols].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="YlGnBu",
    fmt=".2f"
)

plt.title(
    "Lag Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    "plots/lag_correlation.png",
    dpi=300
)

plt.show()