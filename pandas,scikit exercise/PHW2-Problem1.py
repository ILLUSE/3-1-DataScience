import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read csv
df = pd.read_csv('c:\\Users\\codns\\.vscode\\3-1-DataScience\\bmi_data_lab2.csv') #path can be change

#Print dataset statistical data, feature names & data types
print("-feature names:", df.columns.tolist())
print("-data type:\n", df.dtypes)
print("-statistical data:\n", df.describe())

#Plot height & weight histograms (bins=10) for each BMI value
height = df["Height (Inches)"]
weight = df["Weight (Pounds)"]

import matplotlib.pyplot as plt

# BMI 0
bmi0 = df[df["BMI"] == 0]
plt.hist(bmi0["height"].dropna(), bins=10)
plt.title("Histogram of BMI 0 Height")
plt.xlabel("Height (inches)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

plt.hist(bmi0["weight"].dropna(), bins=10)
plt.title("Histogram of BMI 0 Weight")
plt.xlabel("Weight (Pounds)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

# BMI 1
bmi1 = df[df["BMI"] == 1]
plt.hist(bmi1["height"].dropna(), bins=10)
plt.title("Histogram of BMI 1 Height")
plt.xlabel("Height (inches)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

plt.hist(bmi1["weight"].dropna(), bins=10)
plt.title("Histogram of BMI 1 Weight")
plt.xlabel("Weight (Pounds)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

# BMI 2
bmi2 = df[df["BMI"] == 2]
plt.hist(bmi2["height"].dropna(), bins=10)
plt.title("Histogram of BMI 2 Height")
plt.xlabel("Height (inches)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

plt.hist(bmi2["weight"].dropna(), bins=10)
plt.title("Histogram of BMI 2 Weight")
plt.xlabel("Weight (Pounds)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

# BMI 3
bmi3 = df[df["BMI"] == 3]
plt.hist(bmi3["height"].dropna(), bins=10)
plt.title("Histogram of BMI 3 Height")
plt.xlabel("Height (inches)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

plt.hist(bmi3["weight"].dropna(), bins=10)
plt.title("Histogram of BMI 3 Weight")
plt.xlabel("Weight (Pounds)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

# BMI 4
bmi4 = df[df["BMI"] == 4]
plt.hist(bmi4["height"].dropna(), bins=10)
plt.title("Histogram of BMI 4 Height")
plt.xlabel("Height (inches)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()

plt.hist(bmi4["weight"].dropna(), bins=10)
plt.title("Histogram of BMI 4 Weight")
plt.xlabel("Weight (Pounds)")
plt.ylabel("Number of People")
plt.grid(True)
plt.show()
