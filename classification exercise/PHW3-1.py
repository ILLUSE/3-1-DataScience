import pandas as pd
import numpy as np
from math import log2

# Dataset
data = pd.DataFrame({
    'district': ['Suburban', 'Suburban', 'Rural', 'Urban', 'Suburban', 'Suburban', 'Suburban',
                 'Rural', 'Rural', 'Rural', 'Urban', 'Urban', 'Urban', 'Urban'],
    'house': ['Detached', 'Semi-detached', 'Semi-detached', 'Detached', 'Detached', 'Semi-detached',
              'Semi-detached', 'Detached', 'Detached', 'Detached', 'Detached', 'Detached',
              'Detached', 'Detached'],
    'income': ['High', 'High', 'Low', 'Low', 'High', 'High', 'High', 'High', 'High', 'High',
               'High', 'High', 'High', 'High'],
    'prev_customer': ['No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
                      'Yes', 'Yes', 'No', 'No'],
    'outcome': ['Nothing', 'Respond', 'Respond', 'Nothing', 'Nothing', 'Respond', 'Respond',
                'Respond', 'Respond', 'Respond', 'Nothing', 'Nothing', 'Respond', 'Respond']
})

# Entropy calculator
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum([p * log2(p) for p in probs])

# Information gain calculator
def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals, counts = np.unique(df[attr], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / sum(counts)) * entropy(df[df[attr] == vals[i]][target])
        for i in range(len(vals))
    )
    return total_entropy - weighted_entropy

# Build Decision Tree recursively using ID3 algorithm
def build_tree(df, features, target):
    # Case 1: If all target values are the same, return that class label (leaf node)
    if len(np.unique(df[target])) == 1:
        return df[target].iloc[0]
    
    # Case 2: If there are no more features to split, return the most common class label
    if not features:
        return df[target].mode()[0]
    
    # Step 1: Calculate information gain for each feature
    gains = [info_gain(df, feat, target) for feat in features]
    
    # Step 2: Select the feature with the highest information gain
    best_feat = features[np.argmax(gains)]
    
    # Step 3: Initialize the tree structure as a nested list
    tree = [best_feat, {}]

    # Step 4: For each unique value of the selected feature
    for value in df[best_feat].unique():
        # Create a subset of the data where the feature equals that value
        sub_df = df[df[best_feat] == value]

        # Remove the used feature for the next recursive split
        sub_features = [f for f in features if f != best_feat]

        # Recursively build the subtree
        subtree = build_tree(sub_df, sub_features, target)

        # Add the subtree to the current tree node
        tree[1][value] = subtree
    
    # Step 5: Return the completed tree
    return tree


# Build tree
features = ['district', 'house', 'income', 'prev_customer']
target = 'outcome'
decision_tree = build_tree(data, features, target)

# Show the nested list structure
print("Decision Tree Structure:")
print(decision_tree)

# Prediction function
def predict(tree, sample):
    if isinstance(tree, str):
        return tree
    attr, branches = tree
    val = sample[attr]
    if val in branches:
        return predict(branches[val], sample)
    else:
        return "Unknown"  # fallback if unseen attribute value

# Test prediction
test_customer = {
    'district': 'Suburban',
    'house': 'Detached',
    'income': 'Low',
    'prev_customer': 'Yes'
}
prediction = predict(decision_tree, test_customer)
print("\nPrediction for new customer:", prediction)
