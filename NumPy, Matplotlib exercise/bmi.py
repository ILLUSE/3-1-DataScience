import matplotlib.pyplot as plt
import numpy as np
#Create a wt array and an ht array, each of size 100
#Fill the wt array with 100 random float numbers between 40.0 and 90.0.
wt = np.random.uniform(40.0, 90.0, 100)

# Fill the ht array with 100 random integers between  140 and 200(centimeters).
ht = np.random.randint(140, 200, 100)

# Compute the BMI for the 100 students, store them in a bmi array, and print the array
bmi = wt / ((ht / 100) ** 2 )
print(bmi)

# Draw the bar chart, histogram, pie chart,
# and scatter plot of the (height, weight) data in the NumPy exercise. (Use 4 categories for the BMI index)
#  BMI            Weight status
#  Below 18.5     Underweight
#  18.5–24.9      Healthy
#  25.0–29.9      Overweight
#  30.0 and above Obese
bmi_data = []
bmi_count = [0,0,0,0]
for value in bmi:
    if value < 18.5:
        bmi_data.append("Underweight")
        bmi_count[0] += 1  
    elif value < 25.0:
        bmi_data.append("Healthy")
        bmi_count[1] += 1  
    elif value < 30.0:
        bmi_data.append("Overweight")
        bmi_count[2] += 1 
    else:
        bmi_data.append("Obese")
        bmi_count[3] += 1  

bmi_data = np.array(bmi_data)  # change list to numpy array

#   Bar chart
#   Plot the student distribution for each bmi level (#bars = 4)
weight_status = ['Underweight','Healthy','Overweight','Obese']
plt.bar(weight_status,bmi_count)
plt.show()

#   Histogram
#   Plot the student distribution for each bmi level (#bins = 4)
plt.hist(bmi, bins=[0, 18.5, 25, 30,max(bmi)])
plt.xticks([0, 18.5, 25, 30,max(bmi)])
plt.show()

#   Pie chart
#   Plot the ratio of students for each bmi level
plt.pie(bmi_count,labels = weight_status,autopct = '%1.2f%%' )
plt.show()

#   Scatter plot
#   Plot (height, weight) points
plt.scatter(ht, wt, color='b')
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()