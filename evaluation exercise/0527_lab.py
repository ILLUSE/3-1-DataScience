# linear regression , knn ,decision tree with evaluation 
# 📦 라이브러리 불러오기

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

# ✅ 1. 데이터 불러오기 & 분할
data = load_breast_cancer()
X = data.data
y = data.target

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# -----------------------------------------------------
# ✅ 2. Linear Regression (회귀 모델)
print("\n🧠 Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 성능 평가
print("Test R^2 Score:", r2_score(y_test, y_pred_lr))
cv_scores_lr = cross_val_score(lr, X, y, cv=5)
print("CV R^2 Scores:", cv_scores_lr)
print("Mean CV R^2 Score:", np.mean(cv_scores_lr))

# -----------------------------------------------------
# ✅ 3. kNN Classifier + GridSearchCV
print("\n🧠 kNN Classifier (Grid Search 포함)")
knn = KNeighborsClassifier()

# 하이퍼파라미터 탐색 (k=1~24)
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(X, y)

# 최적 k값 및 점수
print("Best k:", knn_gscv.best_params_)
print("Best CV Accuracy:", knn_gscv.best_score_)

# 최적 k로 다시 학습 후 평가
best_k = knn_gscv.best_params_['n_neighbors']
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_knn = knn_final.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred_knn))
cv_scores_knn = cross_val_score(knn_final, X, y, cv=5)
print("CV Accuracy Scores:", cv_scores_knn)
print("Mean CV Accuracy:", np.mean(cv_scores_knn))

# -----------------------------------------------------
# ✅ 4. Decision Tree Classifier
print("\n🧠 Decision Tree Classifier")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred_dt))
cv_scores_dt = cross_val_score(dt, X, y, cv=5)
print("CV Accuracy Scores:", cv_scores_dt)
print("Mean CV Accuracy:", np.mean(cv_scores_dt))
