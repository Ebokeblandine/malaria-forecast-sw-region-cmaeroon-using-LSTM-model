import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# LOAD DATA


df = pd.read_csv("malaria_SW_NASA_FINAL.csv")

print("Dataset Loaded")
print(df.head())


# MONTH ENCODING


month_map = {
    "January":1,
    "February":2,
    "March":3,
    "April":4,
    "May":5,
    "June":6,
    "July":7,
    "August":8,
    "September":9,
    "October":10,
    "November":11,
    "December":12
}

df["Month_Num"] = df["Month"].map(month_map)


# SORT DATA


df = df.sort_values(
    by=["District","Year","Month_Num"]
)


# CREATE LAG FEATURES


df["Rainfall_lag1"] = (
    df.groupby("District")["Rainfall_mm"]
      .shift(1)
)

df["Rainfall_lag2"] = (
    df.groupby("District")["Rainfall_mm"]
      .shift(2)
)

df["Cases_lag1"] = (
    df.groupby("District")["Confirmed_Malaria_Cases"]
      .shift(1)
)

df["Cases_lag2"] = (
    df.groupby("District")["Confirmed_Malaria_Cases"]
      .shift(2)
)


# REMOVE NaN ROWS


df = df.dropna()

print("\nAfter lag creation:")
print(df.shape)


# NORMALIZATION


features = [
    "Rainfall_mm",
    "Temperature_C",
    "RH_percent",
    "Rainfall_lag1",
    "Rainfall_lag2",
    "Cases_lag1",
    "Cases_lag2"
]

scaler = MinMaxScaler()

df[features] = scaler.fit_transform(df[features])


# SAVE CLEAN DATA


df.to_csv(
    "processed_malaria_dataset.csv",
    index=False
)

print("\nProcessed dataset saved!")


# QUICK EDA PLOT


plt.figure(figsize=(12,5))

plt.plot(
    df["Confirmed_Malaria_Cases"].values
)

plt.title("Malaria Cases Trend")

plt.xlabel("Time")

plt.ylabel("Cases")

plt.grid()

plt.show()

# PLOT 2 — Seasonal pattern by month

raw = pd.read_csv("malaria_SW_NASA_FINAL.csv")
month_map = {"January":1,"February":2,"March":3,"April":4,
             "May":5,"June":6,"July":7,"August":8,
             "September":9,"October":10,"November":11,"December":12}
raw["Month_Num"] = raw["Month"].map(month_map)

colors = {"Buea":"#378ADD","Limbe":"#1D9E75",
          "Muyuka":"#D85A30","Tiko":"#7F77DD","Kumba":"#BA7517"}

plt.figure(figsize=(11,5))
for dist in ["Buea","Limbe","Muyuka","Tiko","Kumba"]:
    sub = raw[raw["District"]==dist]
    seasonal = sub.groupby("Month_Num")["Confirmed_Malaria_Cases"].mean()
    plt.plot(seasonal.index, seasonal.values,
             marker="o", label=dist, color=colors[dist], linewidth=2)

plt.xticks(range(1,13),
           ["Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"])
plt.title("Average Monthly Malaria Seasonality by District")
plt.xlabel("Month")
plt.ylabel("Average Confirmed Cases")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("eda_seasonality.png")
plt.show()

# PLOT 3 — Rainfall vs Cases (Buea)
buea = raw[raw["District"]=="Buea"].copy()
buea["Rainfall_lag1"] = buea["Rainfall_mm"].shift(1)

fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax, col, title in zip(axes,
    ["Rainfall_mm","Rainfall_lag1"],
    ["Rainfall vs Cases (same month)",
     "Rainfall vs Cases (1 month lag)"]):
    data = buea[[col,"Confirmed_Malaria_Cases"]].dropna()
    ax.scatter(data[col], data["Confirmed_Malaria_Cases"],
               alpha=0.5, color="#378ADD", s=30)
    corr = data.corr().iloc[0,1]
    ax.set_title(f"{title}\nr = {corr:.3f}")
    ax.set_xlabel(col)
    ax.set_ylabel("Confirmed Cases")
    ax.grid(True, alpha=0.3)

plt.suptitle("Rainfall Correlation — Buea District", fontsize=12)
plt.tight_layout()
plt.savefig("eda_rainfall_correlation.png")
plt.show()
print("EDA plots saved.")