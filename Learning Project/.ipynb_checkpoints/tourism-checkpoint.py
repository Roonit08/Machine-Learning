import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1) LOAD DATA

file_path = "Nepal_Tourism_Statistics_2024.xlsx"

arrivals_raw = pd.read_excel(file_path, sheet_name="Arrivals 1964-2024", skiprows=2)
earnings_raw = pd.read_excel(file_path, sheet_name="Foreign Exchange Earnings", skiprows=2)

print("Raw Arrivals (head):")
print(arrivals_raw.head())
print("\nRaw Earnings (head):")
print(earnings_raw.head())


# 2) CLEAN ARRIVALS

arrivals = arrivals_raw[["Year", "Total Arrivals"]].copy()
arrivals.columns = ["Year", "Total_Arrivals"]

arrivals["Year"] = pd.to_numeric(arrivals["Year"], errors="coerce")
arrivals["Total_Arrivals"] = pd.to_numeric(arrivals["Total_Arrivals"], errors="coerce")
arrivals = arrivals.dropna(subset=["Year", "Total_Arrivals"])

print("\nClean Arrivals (head):")
print(arrivals.head())

# 3) CLEAN EARNINGS

earnings = earnings_raw[["Fiscal Year", "US$ (Millions)"]].copy()
earnings.columns = ["Fiscal_Year", "USD_Earnings_Millions"]

earnings = earnings.dropna(subset=["Fiscal_Year"])

# FY "2000/01" ends in 2001, so end-year = 2000 + 1
years = []
for fy in earnings["Fiscal_Year"]:
    try:
        start_year = int(str(fy).split("/")[0])
        years.append(start_year + 1)
    except:
        years.append(np.nan)

earnings["Year"] = years
earnings["USD_Earnings_Millions"] = pd.to_numeric(earnings["USD_Earnings_Millions"], errors="coerce")
earnings = earnings.dropna(subset=["Year", "USD_Earnings_Millions"])

print("\nClean Earnings (head):")
print(earnings.head())

print("\nEarnings Year sample (check FY alignment):")
print(earnings[["Fiscal_Year", "Year"]].head(10))


# 4) MERGE DATA (Year)

df = pd.merge(arrivals, earnings[["Year", "USD_Earnings_Millions"]], on="Year", how="inner")
df = df.dropna(subset=["Total_Arrivals", "USD_Earnings_Millions"])
df = df.sort_values("Year").reset_index(drop=True)

print("\nMerged Dataset (head):")
print(df.head(10))
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())

# 5) EDA (CORE)
print("\nSummary statistics:")
print(df.describe())

corr = df["Total_Arrivals"].corr(df["USD_Earnings_Millions"])
print("\nCorrelation (Arrivals vs Earnings):", round(corr, 3))

# -------------------------
# A) Line chart: Total arrivals over time
# -------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["Year"], df["Total_Arrivals"], marker="o", linewidth=2)
plt.axvspan(2019.5, 2021.5, color="red", alpha=0.1, label="COVID period")
plt.title("Total Tourist Arrivals Over Time")
plt.xlabel("Year")
plt.ylabel("Total Arrivals")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# B) Line chart: Tourism earnings over time
# -------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["Year"], df["USD_Earnings_Millions"], marker="o", linewidth=2, color="darkorange")
plt.axvspan(2019.5, 2021.5, color="red", alpha=0.1, label="COVID period")
plt.title("Tourism Foreign Exchange Earnings Over Time (US$ Millions)")
plt.xlabel("Year")
plt.ylabel("US$ Earnings (Millions)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# C) Scatter + regression line
# -------------------------
plt.figure(figsize=(10, 6))
sns.regplot(x="Total_Arrivals", y="USD_Earnings_Millions", data=df, ci=95)
plt.title(f"Arrivals vs Earnings (r = {corr:.3f})")
plt.xlabel("Total Arrivals")
plt.ylabel("US$ Earnings (Millions)")
plt.grid(True)
plt.show()

# -------------------------
# D) Correlation heatmap
# -------------------------
plt.figure(figsize=(6, 5))
corr_matrix = df[["Year", "Total_Arrivals", "USD_Earnings_Millions"]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 12})
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -------------------------
# E) Histogram + KDE: Earnings distribution
# -------------------------
plt.figure(figsize=(10, 5))
sns.histplot(df["USD_Earnings_Millions"], kde=True, bins=10)
plt.axvline(df["USD_Earnings_Millions"].mean(),   linestyle="--", color="red",
            label=f"Mean  ${df['USD_Earnings_Millions'].mean():.0f}M")
plt.axvline(df["USD_Earnings_Millions"].median(), linestyle=":",  color="green",
            label=f"Median ${df['USD_Earnings_Millions'].median():.0f}M")
plt.title("Distribution of Tourism Earnings (US$ Millions)")
plt.xlabel("US$ Earnings (Millions)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# F) Boxplots: Outlier detection
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].boxplot(df["Total_Arrivals"])
axes[0].set_title("Boxplot: Total Arrivals")
axes[0].set_ylabel("Arrivals")

axes[1].boxplot(df["USD_Earnings_Millions"])
axes[1].set_title("Boxplot: Earnings (US$ Millions)")
axes[1].set_ylabel("US$ (Millions)")
plt.suptitle("Outlier Detection")
plt.tight_layout()
plt.show()

# -------------------------
# G) Bar chart: Year-on-Year growth rate
# -------------------------
yoy = df.set_index("Year")["Total_Arrivals"].pct_change() * 100
yoy = yoy.dropna()
bar_colors = ["red" if v < 0 else "steelblue" for v in yoy]

plt.figure(figsize=(13, 5))
plt.bar(yoy.index, yoy.values, color=bar_colors, edgecolor="white", linewidth=0.4)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Year-on-Year Growth Rate in Tourist Arrivals (%)")
plt.xlabel("Year")
plt.ylabel("Growth (%)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# =========================================================
# EXTRA EDA 1: ARRIVALS BY NATIONALITY (Stacked area)
# =========================================================
abn = pd.read_excel(file_path, sheet_name="Arrivals by Nationality", skiprows=2)
abn["Year"] = pd.to_numeric(abn["Year"], errors="coerce")
abn["Third Country"] = pd.to_numeric(abn["Third Country"], errors="coerce")
abn["Indian"] = pd.to_numeric(abn["Indian"], errors="coerce")
abn = abn.dropna(subset=["Year", "Third Country", "Indian"]).sort_values("Year")

plt.figure(figsize=(12, 6))
plt.stackplot(abn["Year"], abn["Indian"], abn["Third Country"], labels=["Indian", "Third Country"], alpha=0.85)
plt.title("Tourist Composition Over Time (Indian vs Third Country)")
plt.xlabel("Year")
plt.ylabel("Number of Tourists")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# =========================================================
# EXTRA EDA 2: PURPOSE OF VISIT
# =========================================================
pov = pd.read_excel(file_path, sheet_name="Purpose of Visit", skiprows=2)
pov = pov.rename(columns={"Unnamed: 8": "Total_Reported"})
pov["Year"] = pd.to_numeric(pov["Year"], errors="coerce")

purpose_cols = ["Holiday/Pleasure", "Trekking/Mountaineering", "Business",
                "Pilgrimage", "Official/Conv", "Others", "Total_Reported"]

for c in purpose_cols:
    if c in pov.columns:
        pov[c] = pd.to_numeric(pov[c], errors="coerce")

pov = pov.dropna(subset=["Year", "Total_Reported"]).sort_values("Year")

for c in ["Holiday/Pleasure", "Trekking/Mountaineering", "Business", "Pilgrimage", "Official/Conv", "Others"]:
    if c in pov.columns:
        pov[c] = pov[c].fillna(0)

# Stacked bar — last 5 years
pov_last5 = pov.tail(5).set_index("Year")[["Holiday/Pleasure", "Trekking/Mountaineering", "Business", "Pilgrimage", "Others"]]

plt.figure(figsize=(12, 6))
pov_last5.plot(kind="bar", stacked=True, ax=plt.gca())
plt.title("Purpose of Visit (Last 5 Years)")
plt.xlabel("Year")
plt.ylabel("Number of Tourists")
plt.grid(True, axis="y")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Pie — latest year
latest_year_pov = int(pov["Year"].max())
latest_row = pov[pov["Year"] == latest_year_pov].iloc[0]

labels = ["Holiday/Pleasure", "Trekking/Mountaineering", "Business", "Pilgrimage", "Official/Conv", "Others"]
values = [latest_row[l] for l in labels]

plt.figure(figsize=(7, 7))
plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white"))
plt.title(f"Purpose of Visit Distribution ({latest_year_pov})")
plt.show()

# =========================================================
# EXTRA EDA 3: FOREIGN EXCHANGE EARNINGS
# =========================================================
fx = pd.read_excel(file_path, sheet_name="Foreign Exchange Earnings", skiprows=2)
fx = fx.dropna(subset=["Fiscal Year"])
fx["Fiscal Year"] = fx["Fiscal Year"].astype(str)

fx_years = []
for fy in fx["Fiscal Year"]:
    try:
        start_year = int(fy.split("/")[0])
        fx_years.append(start_year + 1)
    except:
        fx_years.append(np.nan)

fx["Year"] = fx_years
fx["US$ (Millions)"] = pd.to_numeric(fx["US$ (Millions)"], errors="coerce")
fx["Avg Exchange Rate"] = pd.to_numeric(fx["Avg Exchange Rate"], errors="coerce")
fx = fx.dropna(subset=["Year", "US$ (Millions)"]).sort_values("Year")

# Bar chart for earnings (more variety than another line)
plt.figure(figsize=(12, 5))
plt.bar(fx["Year"], fx["US$ (Millions)"], color="steelblue", edgecolor="white", linewidth=0.4)
plt.title("Foreign Exchange Earnings from Tourism (US$ Millions)")
plt.xlabel("Year (FY Ending Year)")
plt.ylabel("US$ (Millions)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# Exchange rate line
plt.figure(figsize=(12, 5))
plt.plot(fx["Year"], fx["Avg Exchange Rate"], marker="o", linewidth=2)
plt.title("Average Exchange Rate Over Time")
plt.xlabel("Year (FY Ending Year)")
plt.ylabel("Avg Exchange Rate")
plt.grid(True)
plt.show()

# =========================================================
# EXTRA EDA 4: MAJOR NATIONALITIES
# =========================================================
maj = pd.read_excel(file_path, sheet_name="Major Nationalities", skiprows=2)

maj_long = maj.melt(id_vars=["Nationality"], var_name="Year", value_name="Arrivals")
maj_long["Year"] = pd.to_numeric(maj_long["Year"], errors="coerce")
maj_long["Arrivals"] = pd.to_numeric(maj_long["Arrivals"], errors="coerce")
maj_long = maj_long.dropna(subset=["Year", "Arrivals"])

latest_year_maj = int(maj_long["Year"].max())
top10_latest = maj_long[maj_long["Year"] == latest_year_maj].sort_values("Arrivals", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top10_latest["Nationality"], top10_latest["Arrivals"])
plt.title(f"Top 10 Tourist Source Countries ({latest_year_maj})")
plt.xlabel("Arrivals")
plt.gca().invert_yaxis()
plt.grid(True, axis="x")
plt.show()

# Top 5 trend
top5_names = top10_latest["Nationality"].head(5).tolist()
maj_top5 = maj_long[maj_long["Nationality"].isin(top5_names)].copy()

plt.figure(figsize=(12, 5))
for name in top5_names:
    temp = maj_top5[maj_top5["Nationality"] == name].sort_values("Year")
    plt.plot(temp["Year"], temp["Arrivals"], marker="o", label=name)

plt.title("Trend of Top 5 Source Countries Over Time")
plt.xlabel("Year")
plt.ylabel("Arrivals")
plt.grid(True)
plt.legend()
plt.show()


# 6) MODEL 1: FULL DATA (TRAIN/TEST + METRICS)

X = df[["Total_Arrivals"]]
y = df["USD_Earnings_Millions"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X_train, y_train)

y_pred_test_1 = model1.predict(X_test)
residuals1 = y_test - y_pred_test_1

mae1 = mean_absolute_error(y_test, y_pred_test_1)
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred_test_1))
r21 = r2_score(y_test, y_pred_test_1)

print("\n===== MODEL 1 (FULL DATA) =====")
print("Intercept (β0):", model1.intercept_)
print("Slope (β1):", model1.coef_[0])
print("MAE:", mae1)
print("RMSE:", rmse1)
print("R^2:", r21)

# Regression line for plotting (fit on all data for visualization)
model1_all = LinearRegression()
model1_all.fit(X, y)

X_line = np.linspace(X["Total_Arrivals"].min(), X["Total_Arrivals"].max(), 100).reshape(-1, 1)
y_line = model1_all.predict(X_line)

# Train/Test visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, label="Train set", alpha=0.7)
plt.scatter(X_test, y_test, label="Test set", alpha=0.7)
plt.plot(X_line, y_line, linewidth=2, label="Regression line")
plt.title("Train/Test Split Visualization (Model 1)")
plt.xlabel("Total Arrivals")
plt.ylabel("US$ Earnings (Millions)")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test_1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle="--", color="gray", label="Perfect fit")
plt.xlabel("Actual Earnings")
plt.ylabel("Predicted Earnings")
plt.title("Actual vs Predicted (Model 1)")
plt.legend()
plt.grid(True)
plt.show()

# Residual distribution
plt.figure(figsize=(8, 5))
plt.hist(residuals1, bins=8)
plt.title("Residual Distribution (Model 1)")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 7) MODEL 2: EXCLUDE COVID YEARS (COMPARISON)

df2 = df[~df["Year"].isin([2020, 2021])].copy()

X2 = df2[["Total_Arrivals"]]
y2 = df2["USD_Earnings_Millions"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)

y2_pred_test = model2.predict(X2_test)
residuals2 = y2_test - y2_pred_test

mae2 = mean_absolute_error(y2_test, y2_pred_test)
rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred_test))
r22 = r2_score(y2_test, y2_pred_test)

print("\n===== MODEL 2 (EXCLUDE COVID 2020-2021) =====")
print("Intercept (β0):", model2.intercept_)
print("Slope (β1):", model2.coef_[0])
print("MAE:", mae2)
print("RMSE:", rmse2)
print("R^2:", r22)

# Comparison summary
print("\n===== COMPARISON =====")
print("Model 1 RMSE:", rmse1, "| R^2:", r21)
print("Model 2 RMSE:", rmse2, "| R^2:", r22)

best_model = model1
best_label = "Model 1 (Full Data)"
best_rmse = rmse1

if rmse2 < rmse1:
    best_model = model2
    best_label = "Model 2 (Exclude COVID)"
    best_rmse = rmse2

print("\nBest model by RMSE:", best_label, "RMSE:", best_rmse)

# Metric comparison bar chart
metrics = ["MAE", "RMSE", "R2"]
model1_scores = [mae1, rmse1, r21]
model2_scores = [mae2, rmse2, r22]

x = np.arange(len(metrics))
plt.figure(figsize=(8, 5))
plt.bar(x - 0.2, model1_scores, width=0.4, label="Model 1")
plt.bar(x + 0.2, model2_scores, width=0.4, label="Model 2")
plt.xticks(x, metrics)
plt.title("Model Comparison Metrics")
plt.legend()
plt.grid(True)
plt.show()

# Residual distribution model 2
plt.figure(figsize=(8, 5))
plt.hist(residuals2, bins=8)
plt.title("Residual Distribution (Model 2)")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 8) SIMPLE CLI PREDICTION

print("\n===== CLI PREDICTION =====")
print("Type arrivals number to predict earnings, or type 'exit' to quit.")

while True:
    user_input = input("Enter Total Arrivals (or 'exit'): ").strip().lower()
    if user_input == "exit":
        print("Exiting.")
        break

    try:
        val = float(user_input)
        input_df = pd.DataFrame([[val]], columns=["Total_Arrivals"])
        pred = best_model.predict(input_df)[0]
        print(f"Predicted Tourism Earnings: ${pred:,.2f} million\n")
    except:
        print("Invalid input. Please enter a numeric value.\n")