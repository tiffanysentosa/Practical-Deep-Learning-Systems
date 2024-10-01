### Homework 1 - Deep Learning System (COMS 6998-015)

This repository contains the code and resources for **Homework 1** from the **Practical Deep Learning Systems** course (COMS 6998-015). The homework focuses on various aspects of deep learning, including bias-variance tradeoff, precision-recall metrics, learning rate scheduling, and CNN architectures.

### Problem Breakdown:

1. **Problem 1 - Bias-Variance Tradeoff (35 points)**  
   - Derive and analyze the bias-variance decomposition.
   - Implement estimators for a given function and analyze underfitting/overfitting.
   - Generate datasets, fit models, and display bias-variance tradeoff.

2. **Problem 2 - Precision, Recall, ROC (20 points)**  
   - Study the relationship between ROC and Precision-Recall (PR) curves.
   - Implement binary classifiers (Adaboost, Logistic Regression) and plot ROC and PR curves.
   - Calculate and compare AUROC, AUPR, and AUPRG metrics.

3. **Problem 3 - Learning Rate, Batch Size, FashionMNIST (15 points)**  
   - Work with cyclical learning rates and batch sizes on the FashionMNIST dataset.
   - Train small models, plot loss curves, and test the impact of batch size and learning rate adjustments.

4. **Problem 4 - Convolutional Neural Networks Architectures (20 points)**  
   - Analyze and compare CNN architectures (VGG, GoogLeNet).
   - Calculate memory and parameters for different layers.
   - Study Inception modules and compute operations for both naive and dimension-reduction versions.

5. **Problem 5 - Parameter-Server Based Asynchronous SGD Training (10 points)**  
   - Study gradient updates and staleness in an asynchronous SGD training system.

---

### Key Files:

- **Code for Problems 1, 2, and 3**: Contains Python implementations of the bias-variance analysis, ROC/PR curve plotting, and cyclical learning rate models. The code leverages libraries like `matplotlib`, `scikit-learn`, and `PyTorch`.
  
- **Images**: Plots and figures generated during the experiments (ROC/PR curves, loss vs. learning rate plots, etc.) are included in this repository.

### How to Use:

1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```

2. Navigate to the respective problem directories (e.g., `problem1`, `problem2`, etc.) for the code and data.

3. Execute the Python scripts for the respective problems.

---

### References:

All homework details and questions are based on **Homework 1** PDF, which contains the exact specifications and tasks for the course. You can refer to the PDF for detailed problem descriptions. Some code is given to us by various sources that can also be found on the **Homework 1** PDF. 
