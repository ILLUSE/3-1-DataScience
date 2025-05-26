import pandas as pd
import numpy as np

# Define dataset
data = pd.DataFrame({
    'height': [158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170],
    'weight': [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68],
    'size':   ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']
})

# Normalize features (to avoid scale dominance)
def normalize(df):
    return (df - df.mean()) / df.std()

norm_features = normalize(data[['height', 'weight']])

# Add normalized values to original dataframe
data['norm_height'] = norm_features['height']
data['norm_weight'] = norm_features['weight']

# New sample to classify
new = pd.Series({'height': 161, 'weight': 61})
new_norm = (new - data[['height', 'weight']].mean()) / data[['height', 'weight']].std()

# Euclidean distance calculator
def euclidean_distance(row, new_point):
    return np.sqrt((row['norm_height'] - new_point['height'])**2 + (row['norm_weight'] - new_point['weight'])**2)

# K-NN prediction function
def knn_predict(data, new_norm, k):
    data['distance'] = data.apply(lambda row: euclidean_distance(row, new_norm), axis=1)
    nearest = data.nsmallest(k, 'distance')
    prediction = nearest['size'].mode()[0]
    return prediction, nearest[['height', 'weight', 'size', 'distance']]

# Predict for k=3
pred_k3, neighbors_k3 = knn_predict(data.copy(), new_norm, 3)
print("Prediction with k=3:", pred_k3)
print("Nearest Neighbors (k=3):")
print(neighbors_k3)

# Predict for k=5
pred_k5, neighbors_k5 = knn_predict(data.copy(), new_norm, 5)
print("\nPrediction with k=5:", pred_k5)
print("Nearest Neighbors (k=5):")
print(neighbors_k5)
