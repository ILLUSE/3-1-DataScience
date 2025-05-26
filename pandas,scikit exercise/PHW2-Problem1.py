import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# Load CSV file
col_names = ["gender", "age", "height", "weight", "BMI"]
df = pd.read_csv(
    'c:/Users/Chaeun/.vscode/3-1 DataScience/bmi_data_lab2.csv',
    header=0,  # skip header row
    names=col_names
)
# Print dataset statistical data, feature names & data types
print("- feature names:", df.columns.tolist())
print("- data types:\n", df.dtypes)
print("- statistical data:\n", df.describe())

# Visualize height/weight histograms by BMI (bins=10)
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 15))
fig.suptitle("Height & Weight Histograms by BMI (bins=10)", fontsize=16)

for i in range(5):
    bmi_group = df[df["BMI"] == i]
    axs[i, 0].hist(bmi_group["height"].dropna(), bins=10, color="skyblue")
    axs[i, 0].set_title(f"BMI {i} - Height")
    axs[i, 0].set_xlabel("Height (inches)")
    axs[i, 0].set_ylabel("Count")
    axs[i, 0].grid(True)

    axs[i, 1].hist(bmi_group["weight"].dropna(), bins=10, color="salmon")
    axs[i, 1].set_title(f"BMI {i} - Weight")
    axs[i, 1].set_xlabel("Weight (Pounds)")
    axs[i, 1].set_ylabel("Count")
    axs[i, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Visualize scaling result
scalers = {
    "StandardScaler": preprocessing.StandardScaler(),
    "MinMaxScaler": preprocessing.MinMaxScaler(),
    "RobustScaler": preprocessing.RobustScaler()
}
df_scaled = df.dropna(subset=["height", "weight"])[["height", "weight"]]

plt.figure(figsize=(18, 5))
for idx, (name, scaler) in enumerate(scalers.items()):
    scaled = scaler.fit_transform(df_scaled)
    scaled_df = pd.DataFrame(scaled, columns=["height", "weight"])

    plt.subplot(1, 3, idx + 1)
    plt.hist(scaled_df["height"], bins=10, color="skyblue", alpha=0.6, label="Height")
    plt.hist(scaled_df["weight"], bins=10, color="salmon", alpha=0.6, label="Weight")
    plt.title(name)
    plt.xlabel("Scaled Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

plt.suptitle("Scaling Results for Height and Weight", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# Outlier removal
df.loc[(df["height"] < 60) | (df["height"] > 80), "height"] = np.nan
# Acceptable height: 60–80 inches

df.loc[(df["weight"] < 90) | (df["weight"] > 150), "weight"] = np.nan # loc : 조건에 맞는 것 선택
# Acceptable weight: 90–150 pounds

# Print missing value statistics
print(f"\nNumber of rows containing NaN: {df.isna().any(axis=1).sum()}")
print("\nNumber of NaNs per column:")
print(df.isna().sum())

# Extract only rows without missing values
df_extracted = df.dropna()
print(f"\nNumber of complete rows (no NaN): {len(df_extracted)}")

# Imputation using mean, median, ffill, and bfill
df_mean_filled = df.copy()
df_mean_filled["height"] = df_mean_filled["height"].fillna(df["height"].mean())
df_mean_filled["weight"] = df_mean_filled["weight"].fillna(df["weight"].mean())

df_median_filled = df.copy()
df_median_filled["height"] = df_median_filled["height"].fillna(df["height"].median())
df_median_filled["weight"] = df_median_filled["weight"].fillna(df["weight"].median())

df_ffill = df.copy().ffill()
df_bfill = df.copy().bfill()

# Imputation using linear regression (on full dataset)
clean_df = df.dropna(subset=["height", "weight"]) # 하나라도 nan이 있는 행은 제거
dirty_df = df[df["height"].isna() | df["weight"].isna()] #만약 nan 존재시 여기에 저장됨

model_hw = LinearRegression() # 성형회귀 모델
model_hw.fit(clean_df[["height"]], clean_df["weight"]) # height 통해 weight 에측 
model_wh = LinearRegression()
model_wh.fit(clean_df[["weight"]], clean_df["height"])

df_reg_filled = df.copy()
has_height_nan_weight = df["height"].notna() & df["weight"].isna() # height는 존재 , weight는 nan인 열 찾기
df_reg_filled.loc[has_height_nan_weight, "weight"] = model_hw.predict(df.loc[has_height_nan_weight][["height"]]) # model_hw모델을 활용하여 값 채워넣기

has_weight_nan_height = df["weight"].notna() & df["height"].isna()
df_reg_filled.loc[has_weight_nan_height, "height"] = model_wh.predict(df.loc[has_weight_nan_height][["weight"]])

# Group-wise regression imputation (by gender and BMI)
df_gender_bmi_filled = df.copy()

for gender in df["gender"].unique():
    for bmi in df["BMI"].unique():
        sub_df = df[(df["gender"] == gender) & (df["BMI"] == bmi)]
        clean_sub_df = sub_df.dropna(subset=["height", "weight"])

        if len(clean_sub_df) < 2:
            continue

        model = LinearRegression()
        model.fit(clean_sub_df[["height"]], clean_sub_df["weight"])

        weight_missing = (df["gender"] == gender) & (df["BMI"] == bmi) & df["weight"].isna() & df["height"].notna()
        X_weight = df.loc[weight_missing][["height"]]
        if not X_weight.empty:
            df_gender_bmi_filled.loc[weight_missing, "weight"] = model.predict(X_weight)

        model_rev = LinearRegression()
        model_rev.fit(clean_sub_df[["weight"]], clean_sub_df["height"])

        height_missing = (df["gender"] == gender) & (df["BMI"] == bmi) & df["height"].isna() & df["weight"].notna()
        X_height = df.loc[height_missing][["weight"]]
        if not X_height.empty:
            df_gender_bmi_filled.loc[height_missing, "height"] = model_rev.predict(X_height)

# Visualization: Before vs After imputation (scatter plot)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
fig.suptitle("Before vs After Imputation (Height vs Weight)", fontsize=16)

axes[0].scatter(df["height"], df["weight"], alpha=0.5, color='gray')
axes[0].set_title("Before Imputation")
axes[0].set_xlabel("Height (inches)")
axes[0].set_ylabel("Weight (pounds)")
axes[0].grid(True)

axes[1].scatter(df_gender_bmi_filled["height"], df_gender_bmi_filled["weight"], alpha=0.5, color='blue')
axes[1].set_title("After Imputation")
axes[1].set_xlabel("Height (inches)")
axes[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Visualization: Highlight dirty data
plt.figure(figsize=(8, 6))
plt.scatter(df_gender_bmi_filled["height"], df_gender_bmi_filled["weight"], alpha=0.5, label="Cleaned Data")
plt.scatter(dirty_df["height"], dirty_df["weight"], color='red', label="Previously Dirty", edgecolor="k", s=80)
plt.title("Height vs Weight (Dirty Data Highlighted)", fontsize=14)
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.grid(True)
plt.legend()
plt.show()
