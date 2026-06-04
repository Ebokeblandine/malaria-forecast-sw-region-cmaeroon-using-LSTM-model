import matplotlib.pyplot as plt
import numpy as np

features = [
    "Month",
    "Rainfall",
    "Temperature",
    "Humidity",
    "Rain Lag1",
    "Rain Lag2",
    "Cases Lag1",
    "Cases Lag2"
]

scores = np.array([
    142.72,
    2.24,
    0.50,
    4.65,
    0.09,
    0.30,
    21.94,
    4.33
])

plt.figure(figsize=(8,8))

plt.pie(
    scores,
    labels=features,
    autopct="%1.1f%%",
    startangle=90
)

centre_circle = plt.Circle(
    (0,0),
    0.70,
    fc="white"
)

fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title(
    "Feature Importance Distribution"
)

plt.tight_layout()

plt.savefig(
    "plots/importance_donut.png",
    dpi=300
)

plt.show()