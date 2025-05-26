import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# Load CSV file
col_names = ["gender", "age", "height", "weight", "BMI"]
df = pd.read_csv(
    'c:/Users/Chaeun/.vscode/3-1 DataScience/bmi_data_lab2.csv',
    header=0,
    names=col_names
)

print("- feature names:", df.columns.tolist())
print("- data types:\n", df.dtypes)
print("- statistical data:\n", df.describe())

# Linear regression on full dataset: height ➝ weight
clean_df = df.dropna(subset=["height", "weight"])
model = LinearRegression()
model.fit(clean_df[["height"]], clean_df["weight"])

# Predicted weight w' and error e = w - w'
df_pred = clean_df.copy()
df_pred["w_pred"] = model.predict(df_pred[["height"]])
df_pred["e"] = df_pred["weight"] - df_pred["w_pred"]

# ze: Normalized error using mean and standard deviation (z-score)
e_mean = df_pred["e"].mean()
e_std = df_pred["e"].std()
df_pred["ze"] = (df_pred["e"] - e_mean) / e_std

# Histogram of ze
plt.figure(figsize=(8, 5))
plt.hist(df_pred["ze"], bins=10, color="mediumpurple", edgecolor="k")
plt.title("Histogram of Normalized Error (ze)")
plt.xlabel("ze")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Outlier-based BMI estimation using alpha
# If ze < -alpha → predict BMI = 0, if ze > alpha → predict BMI = 4
alpha = 1

df_pred["BMI_est"] = df_pred["BMI"]
df_pred.loc[df_pred["ze"] < -alpha, "BMI_est"] = 0
df_pred.loc[df_pred["ze"] > alpha, "BMI_est"] = 4

# Compare predicted vs actual BMI (Accuracy check)
compare = df_pred[["BMI", "BMI_est"]].value_counts().reset_index(name="count")
print("\n[Full Dataset] Comparison of Actual vs Predicted BMI:")
print(compare)

# Gender-wise analysis for Df, Dm groups
gender_results = []

for gender in df["gender"].unique():
    print(f"\n--- Gender: {gender} ---")

    df_gender = df[df["gender"] == gender].dropna(subset=["height", "weight"])

    if len(df_gender) < 2:
        print("Skipped due to insufficient data.")
        continue

    # Train linear regression for each gender group
    model = LinearRegression()
    model.fit(df_gender[["height"]], df_gender["weight"])

    df_gender["w_pred"] = model.predict(df_gender[["height"]])
    df_gender["e"] = df_gender["weight"] - df_gender["w_pred"]

    # Calculate ze for the gender group
    e_mean = df_gender["e"].mean()
    e_std = df_gender["e"].std()
    df_gender["ze"] = (df_gender["e"] - e_mean) / e_std

    # Histogram of ze (gender-specific)
    plt.figure(figsize=(6, 4))
    plt.hist(df_gender["ze"], bins=10, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of ze (Gender: {gender})")
    plt.xlabel("ze")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Generate predicted BMI
    df_gender["BMI_est"] = df_gender["BMI"]
    df_gender.loc[df_gender["ze"] < -alpha, "BMI_est"] = 0
    df_gender.loc[df_gender["ze"] > alpha, "BMI_est"] = 4

    # Print comparison of actual vs predicted BMI
    comparison = df_gender[["BMI", "BMI_est"]].value_counts().reset_index(name="count")
    print(comparison)

    gender_results.append((gender, comparison))