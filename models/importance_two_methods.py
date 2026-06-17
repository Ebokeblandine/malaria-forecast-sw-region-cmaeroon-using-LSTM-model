import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(
    "processed_malaria_dataset.csv"
)

features = [
    "Month_Num",
    "Rainfall_mm",
    "Temperature_C",
    "RH_percent",
    "Rainfall_lag1",
    "Rainfall_lag2",
    "Cases_lag1",
    "Cases_lag2"
]

target = "Confirmed_Malaria_Cases"

corr_scores = []

for col in features:
    corr_scores.append(
        abs(df[col].corr(df[target]))
    )

corr_scores = np.array(corr_scores)

perm_scores = np.array([
    142.72,
    2.24,
    0.50,
    4.65,
    0.09,
    0.30,
    21.94,
    4.33
])

corr_scores = MinMaxScaler().fit_transform(
    corr_scores.reshape(-1,1)
).flatten()

perm_scores = MinMaxScaler().fit_transform(
    perm_scores.reshape(-1,1)
).flatten()

x = np.arange(len(features))
width = 0.35

plt.figure(figsize=(12,5))

plt.bar(
    x-width/2,
    corr_scores,
    width,
    label="Correlation"
)

plt.bar(
    x+width/2,
    perm_scores,
    width,
    label="Permutation"
)

plt.xticks(
    x,
    [
        "Month",
        "Rain",
        "Temp",
        "RH",
        "RainL1",
        "RainL2",
        "CaseL1",
        "CaseL2"
    ]
)

plt.ylabel("Normalized Importance")

plt.title(
    "Feature Importance Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/importance_two_methods.png",
    dpi=300
)

plt.show()