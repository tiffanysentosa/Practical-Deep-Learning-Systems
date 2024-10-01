from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np

plt.ioff() # interactive mode is off

import matplotlib
matplotlib.use('TkAgg') 

# Load the ARFF file of my Databse from OpenML 
file_path = '/Users/tsentosa/Desktop/Practical Deep Learning Systems/wine_quality.arff'  # Replace with your file path
print("Loading ARFF file...")
data, meta = arff.loadarff(file_path)

# Convert to DataFrame
print("Converting ARFF data to DataFrame...")
df = pd.DataFrame(data)

# Create binary classification target based on quality
print("Creating binary classification target based on quality...")
df['quality_binary'] = (df['quality'] > 5).astype(int)  # 1 for good wine, 0 for bad wine

# Separate features and target
print("Separating features and target variable...")
X = df.drop(['quality', 'quality_binary'], axis=1)  # Features
y = df['quality_binary']  # Target

# Split into training and testing datasets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
print("Standardizing the features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers with limited estimators for AdaBoost
print("Initializing AdaBoost and Logistic Regression classifiers...")
ada_clf = AdaBoostClassifier(random_state=42, n_estimators=50, algorithm='SAMME')  # Use 50 estimators to reduce time
lr_clf = LogisticRegression(random_state=42, max_iter=1000)

# Train classifiers
print("Training AdaBoost classifier...")
ada_clf.fit(X_train, y_train)
print("Training Logistic Regression classifier...")
lr_clf.fit(X_train, y_train)

# Predict probabilities
print("Predicting probabilities for test data...")
ada_probs = ada_clf.predict_proba(X_test)[:, 1]
lr_probs = lr_clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and Precision-Recall curve
print("Computing ROC and Precision-Recall curves for AdaBoost and Logistic Regression...")
fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_probs)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
precision_ada, recall_ada, _ = precision_recall_curve(y_test, ada_probs)
precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_probs)

# Print shapes and lengths to debug
print(f"fpr_ada shape: {fpr_ada.shape}, tpr_ada shape: {tpr_ada.shape}")
print(f"fpr_ada first 5 values: {fpr_ada[:5]}")
print(f"tpr_ada first 5 values: {tpr_ada[:5]}")
print(f"precision_ada first 5 values: {precision_ada[:5]}")
print(f"recall_ada first 5 values: {recall_ada[:5]}")

# Calculate metrics for an all-positive classifier
all_positive_precision = y_test.mean()  # Proportion of positive instances in the dataset
all_positive_recall = 1.0  # All positives are correctly identified
all_positive_fpr = 1.0  # All negatives are incorrectly classified as positives
all_positive_tpr = 1.0  # All positives are correctly classified

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict the class labels for the test data
ada_preds = ada_clf.predict(X_test)
lr_preds = lr_clf.predict(X_test)

# Calculate confusion matrices
confusion_matrix_ada = confusion_matrix(y_test, ada_preds)
confusion_matrix_lr = confusion_matrix(y_test, lr_preds)

# Plot and save the confusion matrix for AdaBoost
plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_ada, display_labels=['Bad Wine', 'Good Wine']).plot(cmap='Blues')
plt.title('Confusion Matrix for AdaBoost')
plt.savefig('confusion_matrix_adaboost.png')  # Save the confusion matrix as an image file
print("Saved Confusion Matrix for AdaBoost as 'confusion_matrix_adaboost.png'")
plt.close()  # Close the plot

# Plot and save the confusion matrix for Logistic Regression
plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_lr, display_labels=['Bad Wine', 'Good Wine']).plot(cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.savefig('confusion_matrix_logistic_regression.png')  # Save the confusion matrix as an image file
print("Saved Confusion Matrix for Logistic Regression as 'confusion_matrix_logistic_regression.png'")
plt.close()  # Close the plot

# Plotting ROC and PR curves with titles and comments

# ROC Curve for AdaBoost
print("Plotting ROC Curve for AdaBoost...")
plt.figure(figsize=(10, 5))
RocCurveDisplay(fpr=fpr_ada, tpr=tpr_ada, estimator_name='AdaBoost').plot()
plt.scatter(all_positive_fpr, all_positive_tpr, color='red', label='All Positive Classifier', zorder=5)
plt.legend()
plt.title('ROC Curve for AdaBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_curve_adaboost.png')  # Save to file instead of showing
print("Saved ROC Curve for AdaBoost as 'roc_curve_adaboost.png'")

# ROC Curve for Logistic Regression
print("Plotting ROC Curve for Logistic Regression...")
plt.figure(figsize=(10, 5))
RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr, estimator_name='Logistic Regression').plot()
plt.scatter(all_positive_fpr, all_positive_tpr, color='red', label='All Positive Classifier', zorder=5)
plt.legend()
plt.title('ROC Curve for Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_curve_logistic_regression.png')  # Save to file instead of showing
print("Saved ROC Curve for Logistic Regression as 'roc_curve_logistic_regression.png'")

# PR Curve for AdaBoost
print("Plotting Precision-Recall Curve for AdaBoost...")
plt.figure(figsize=(10, 5))
PrecisionRecallDisplay(precision=precision_ada, recall=recall_ada, estimator_name='AdaBoost').plot()
plt.scatter(all_positive_recall, all_positive_precision, color='red', label='All Positive Classifier', zorder=5)
plt.legend()
plt.title('Precision-Recall Curve for AdaBoost')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('pr_curve_adaboost.png')  # Save to file instead of showing
print("Saved Precision-Recall Curve for AdaBoost as 'pr_curve_adaboost.png'")

# PR Curve for Logistic Regression
print("Plotting Precision-Recall Curve for Logistic Regression...")
plt.figure(figsize=(10, 5))
PrecisionRecallDisplay(precision=precision_lr, recall=recall_lr, estimator_name='Logistic Regression').plot()
plt.scatter(all_positive_recall, all_positive_precision, color='red', label='All Positive Classifier', zorder=5)
plt.legend()
plt.title('Precision-Recall Curve for Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('pr_curve_logistic_regression.png')  # Save to file instead of showing
print("Saved Precision-Recall Curve for Logistic Regression as 'pr_curve_logistic_regression.png'")


# Combined ROC Curve for both classifiers
plt.figure(figsize=(10, 5))
RocCurveDisplay(fpr=fpr_ada, tpr=tpr_ada, estimator_name='AdaBoost').plot(ax=plt.gca())  # Use the same axis
RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr, estimator_name='Logistic Regression').plot(ax=plt.gca())  # Use the same axis
plt.scatter(all_positive_fpr, all_positive_tpr, color='red', label='All Positive Classifier', zorder=5)
plt.legend()
plt.title('Combined ROC Curve for AdaBoost and Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('combined_roc_curve.png')  # Save the ROC curve figure
print("Saved Combined ROC Curve as 'combined_roc_curve.png'")
#plt.show()  # Display the combined ROC curve

# Combined Precision-Recall Curve for both classifiers
plt.figure(figsize=(10, 5))
PrecisionRecallDisplay(precision=precision_ada, recall=recall_ada, estimator_name='AdaBoost').plot(ax=plt.gca())  # Use the same axis
PrecisionRecallDisplay(precision=precision_lr, recall=recall_lr, estimator_name='Logistic Regression').plot(ax=plt.gca())  # Use the same axis
plt.scatter(all_positive_recall, all_positive_precision, color='red', label='All Positive Classifier', zorder=5)
plt.legend()
plt.title('Combined Precision-Recall Curve for AdaBoost and Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('combined_pr_curve.png')  # Save the PR curve figure
print("Saved Combined Precision-Recall Curve as 'combined_pr_curve.png'")
#plt.show()  # Display the combined Precision-Recall curve

print("#\n#\n#\n#")

# Calculate AUROC for both classifiers
auroc_ada = roc_auc_score(y_test, ada_probs)
auroc_lr = roc_auc_score(y_test, lr_probs)

# Calculate AUPR for both classifiers
aupr_ada = auc(recall_ada, precision_ada)
aupr_lr = auc(recall_lr, precision_lr)

# Function to calculate PR Gain
## CHANGE
def calculate_pr_gain(recall, precision, positive_fraction):
    pi = positive_fraction
    precision_gain = (precision - pi) / ((1 - pi) * precision)
    recall_gain = (recall - pi) / ((1 - pi) * recall)
    # recall_gain = recall

    mask = (precision_gain >= 0) & (recall_gain >= 0)
    precision_gain = precision_gain[mask]
    recall_gain = recall_gain[mask]
    
    return precision_gain, recall_gain

# Calculate positive fraction (π) of the dataset
positive_fraction = y_test.mean()

# Calculate PR Gain for AdaBoost
precision_gain_ada, recall_gain_ada = calculate_pr_gain(recall_ada, precision_ada, positive_fraction)
auprg_ada = auc(recall_gain_ada, precision_gain_ada)

# Calculate PR Gain for Logistic Regression
precision_gain_lr, recall_gain_lr = calculate_pr_gain(recall_lr, precision_lr, positive_fraction)
auprg_lr = auc(recall_gain_lr, precision_gain_lr)

# Print all metrics for comparison
print("Metrics Comparison for AdaBoost and Logistic Regression:")
print(f"AdaBoost AUROC: {auroc_ada:.4f}")
print(f"Logistic Regression AUROC: {auroc_lr:.4f}")

print(f"AdaBoost AUPR: {aupr_ada:.4f}")
print(f"Logistic Regression AUPR: {aupr_lr:.4f}")

print(f"AdaBoost AUPRG: {auprg_ada:.4f}")
print(f"Logistic Regression AUPRG: {auprg_lr:.4f}")


    