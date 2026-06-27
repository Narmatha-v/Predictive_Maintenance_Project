# =====================================
# Data Loading
# =====================================

import pandas as pd

data = pd.read_csv("ai4i2020.csv")

print(data.head())

# =====================================
# Dataset Overview
# =====================================

print("Dataset Shape:", data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nDataset Information:")
data.info()

# =====================================
# Data Preprocessing
# =====================================

data = data.drop(["UDI", "Product ID"], axis=1)

print("\nDataset After Removing Columns:")
print(data.head())

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data["Type"] = encoder.fit_transform(data["Type"])

print("\nEncoded Dataset Preview:")
print(data.head())

# =====================================
# Feature and Target Selection
# =====================================

X = data.drop("Machine failure", axis=1)

y = data["Machine failure"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# =====================================
# Missing Value Analysis
# =====================================

print("\nMissing Values:")
print(data.isnull().sum())

# =====================================
# Target Variable Distribution
# =====================================

import seaborn as sns
import matplotlib.pyplot as plt

print("\nMachine Failure Distribution:")
print(data["Machine failure"].value_counts())

sns.countplot(x="Machine failure", data=data)

plt.title("Machine Failure Distribution")

plt.savefig("outputs/machine_failure_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================
# Machine Failure Count
# =====================================

print("\nMachine Failure Distribution:")
print(data["Machine failure"].value_counts())

# =====================================
# Correlation Heatmap
# =====================================

plt.figure(figsize=(12, 8))

sns.heatmap(data.corr(), annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================
# Remove Failure Indicator Columns
# =====================================

data = data.drop(["TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)

print("\nDataset After Removing Failure Indicator Columns:")
print(data.head())

# =====================================
# Update Features and Target
# =====================================

X = data.drop("Machine failure", axis=1)

y = data["Machine failure"]

print("\nUpdated Features Shape:", X.shape)
print("Updated Target Shape:", y.shape)

# =====================================
# Train Test Split
# =====================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Shapes")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# =====================================
# Logistic Regression Model
# =====================================

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

# =====================================
# Logistic Regression Evaluation
# =====================================

from sklearn.metrics import accuracy_score, classification_report

log_accuracy = accuracy_score(y_test, y_pred_log)

print("\nLogistic Regression Accuracy:")
print(log_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# =====================================
# Decision Tree Model
# =====================================

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# =====================================
# Decision Tree Evaluation
# =====================================

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:")
print(dt_accuracy)

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# =====================================
# Decision Tree Cross Validation
# =====================================
from sklearn.model_selection import cross_val_score

cv_dt = cross_val_score(
    dt_model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\nDecision Tree CV Scores:")
print(cv_dt)

print("\nDecision Tree Average F1:")
print(cv_dt.mean())

# =====================================
# Random Forest Model
# =====================================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# =====================================
# Random Forest Evaluation
# =====================================

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# =====================================
# Random Forest Cross Validation
# =====================================

cv_rf = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\nRandom Forest CV Scores:")
print(cv_rf)

print("\nRandom Forest Average F1:")
print(cv_rf.mean())

# =====================================
# Support Vector Machine
# =====================================

from sklearn.svm import SVC

svm_model = SVC(
    probability=True
)

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

# =====================================
# SVM Evaluation
# =====================================

svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\nSVM Accuracy:")
print(svm_accuracy)

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# =====================================
# SVM Cross Validation
# =====================================

cv_svm = cross_val_score(
    svm_model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\nSVM CV Scores:")
print(cv_svm)

print("\nSVM Average F1:")
print(cv_svm.mean())

# =====================================
# Gradient Boosting Model
# =====================================

from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(
    random_state=42
)

gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)

# =====================================
# Gradient Boosting Evaluation
# =====================================

gb_accuracy = accuracy_score(y_test, y_pred_gb)

print("\nGradient Boosting Accuracy:")
print(gb_accuracy)

print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))

# =====================================
# Gradient Boosting Cross Validation
# =====================================

cv_gb = cross_val_score(
    gb_model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\nGradient Boosting CV Scores:")
print(cv_gb)

print("\nGradient Boosting Average F1:")
print(cv_gb.mean())

# =====================================
# Stratified K-Fold Validation
# =====================================

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    log_model,
    X,
    y,
    cv=skf,
    scoring="f1"
)

print("\nStratified K-Fold Scores:")
print(cv_scores)

print("\nAverage F1 Score:")
print(cv_scores.mean())

# =====================================
# Confusion Matrix
# =====================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)

print("\nConfusion Matrix:")
print(cm)

# =====================================
# Classification Report
# =====================================

report = classification_report(
    y_test,
    y_pred_rf
)

print("\nClassification Report:")
print(report)

# =====================================
# ROC Curve and AUC Score
# =====================================

from sklearn.metrics import roc_curve, roc_auc_score

y_prob = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label="Random Forest")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()
plt.savefig("outputs/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

auc_score = roc_auc_score(
    y_test,
    y_prob
)

print("\nAUC Score:")
print(auc_score)

# =====================================
# Model Comparison
# =====================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM"
    ],
    "Accuracy": [
        log_accuracy,
        dt_accuracy,
        rf_accuracy,
        gb_accuracy,
        svm_accuracy
    ]
})

print("\nModel Comparison:")
print(results)

# =====================================
# Feature Importance
# =====================================

importance = rf_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

# =====================================
# Feature Importance Visualization
# =====================================

plt.figure(figsize=(8, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")

plt.title(
    "Feature Importance - Random Forest"
)

plt.gca().invert_yaxis()
plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================
# Model Accuracy Comparison
# =====================================

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "SVM"
]

accuracy = [
    log_accuracy,
    dt_accuracy,
    rf_accuracy,
    gb_accuracy,
    svm_accuracy
]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracy)

plt.ylabel("Accuracy")

plt.title(
    "Model Accuracy Comparison"
)

plt.xticks(rotation=30)
plt.savefig("outputs/model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================
# Confusion Matrix Visualization
# =====================================

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    rf_model,
    X_test,
    y_test
)

plt.title("Random Forest Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================
# Training vs Testing Accuracy
# =====================================

train_acc = rf_model.score(
    X_train,
    y_train
)

test_acc = rf_model.score(
    X_test,
    y_test
)

print(
    "\nTraining Accuracy:",
    train_acc
)

print(
    "Testing Accuracy:",
    test_acc
)

# =====================================
# Hyperparameter Tuning
# =====================================

from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    params,
    cv=5
)

grid.fit(
    X_train,
    y_train
)

print(
    "\nBest Parameters:"
)

print(
    grid.best_params_
)

# =====================================
# Best Model
# =====================================

best_model = grid.best_estimator_

best_model.fit(
    X_train,
    y_train
)

# =====================================
# Save Model
# =====================================

import joblib

joblib.dump(
    best_model,
    "model.pkl"
)

print(
    "\nModel saved as model.pkl"
)