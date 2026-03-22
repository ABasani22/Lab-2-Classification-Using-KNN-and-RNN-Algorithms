# MSCS 634 – Lab 2: KNN and Radius Neighbors Classification

**Course:** MSCS 634 – Advanced Data Mining and Predictive Analytics  
**Lab:** Lab 2 – K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) Classifiers

---

## Purpose

This lab explores the performance of two distance-based classification algorithms — **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** — applied to the Wine Dataset from scikit-learn. The dataset contains 178 wine samples across three classes, described by 13 chemical properties (e.g., alcohol content, flavanoids, color intensity).

The goals are to:
1. Understand how parameter choices (k for KNN, radius for RNN) affect classification accuracy.
2. Visualize accuracy trends across parameter values.
3. Compare the two classifiers and identify when each is preferable.

---

## Repository Contents

| File | Description |
|---|---|
| `MSCS_634_Lab_2.ipynb` | Jupyter Notebook with all code, plots, and analysis |
| `README.md` | This file — overview, insights, and reflections |

---

## Key Insights

### KNN
- **Tested k values:** 1, 5, 11, 15, 21
- **Observation:** k = 1 often achieves high accuracy on this dataset but risks overfitting. Mid-range values (k = 5, 11) typically offer the best generalization by averaging out noise from individual samples. Higher k values (15, 21) can slightly underfit as they include more distant, less-relevant neighbors.
- **Best result:** Generally observed at k = 5 or k = 11, depending on the random split.

### RNN
- **Tested radius values:** 350, 400, 450, 500, 550, 600
- **Observation:** Small radii (350) can leave test points without any qualifying neighbors, triggering the `outlier_label` fallback and reducing accuracy. Mid-range radii (450–500) offer a good balance. Larger radii (550–600) stabilize accuracy as most training samples fall within the neighborhood.
- **Note:** These radius values are calibrated for the **raw (unscaled)** feature space. After StandardScaler normalization, scaled distances are much smaller, so the radius values effectively include all training samples — demonstrating the importance of tuning radius on the same scale as the training data.

### KNN vs. RNN Comparison

| Scenario | Preferred Classifier |
|---|---|
| Uniform class density across feature space | KNN — simpler and more stable |
| Varying local densities between classes | RNN — radius adapts to local structure |
| Quick prototyping with easy parameter tuning | KNN — k is intuitive to tune |
| Datasets where outliers should be excluded | RNN — points outside radius are ignored |

**Overall:** For the Wine Dataset, KNN provides competitive and consistent accuracy across a range of k values, making it the more practical and interpretable choice. RNN can match KNN at optimal radii but is more sensitive to parameter selection.

---

## Challenges and Decisions

### 1. Feature Scaling
Distance-based classifiers are highly sensitive to feature scale. The Wine Dataset contains features with vastly different ranges (e.g., `proline` ranges 0–1680, while `nonflavanoid_phenols` ranges 0.1–0.7). **StandardScaler** was applied to normalize all features, which is critical for fair distance computation.

### 2. RNN Outlier Handling
Some test points had no neighbors within smaller radii. The `outlier_label='most_frequent'` parameter was used to assign the most common training class to these points, preventing errors and allowing fair comparison across radius values.

### 3. Radius Value Selection
The lab-specified radius values (350–600) are appropriate for the unscaled feature space. After scaling, these radii include all training samples, making RNN equivalent to a majority-class vote at those settings. A more realistic RNN experiment on scaled data would use radii in the range 1–10. This distinction is discussed in the notebook observations.

### 4. Train/Test Split Reproducibility
`random_state=42` and `stratify=y` were used in `train_test_split` to ensure reproducibility and maintain class proportions across training and test sets.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MSCS_634_Lab_2.git
   cd MSCS_634_Lab_2
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn notebook
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook MSCS_634_Lab_2.ipynb
   ```

4. Run all cells from top to bottom (`Kernel > Restart & Run All`).

---

## Requirements

- Python 3.8+
- scikit-learn
- numpy
- pandas
- matplotlib
- jupyter
