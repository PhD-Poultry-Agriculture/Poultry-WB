# Poultry-WB
A research repository for Ross's classification of WoodenBreast markers via ML, AI
# Poultry-WB Analysis

This project analyzes the dataset of chicken groups (`CONTROL` and `WIDE_BREAST`) to evaluate the significance of various features using statistical tests and False Discovery Rate (FDR) correction. The primary goal is to identify features that show significant differences between the two groups.

## Project Overview

The dataset contains measurements of multiple features for chickens in two groups: `CONTROL` and `WIDE_BREAST`. These measurements are taken across four timestamps, and the median values for each feature are calculated for each chicken. The key objectives are:

1. **Statistical Testing (T-test)**: To compare feature differences across timestamps between the two groups.
2. **FDR Correction**: To control for multiple comparisons and identify significantly different features.

## Workflow

### 1. **Data Preparation**
- The dataset is read from a CSV file (`Experiment-PolarData.csv`), containing feature values for both chicken groups.
- Median values for each feature are precomputed for each chicken across different timestamps.

### 2. **Distance Calculation**
- For each feature, the absolute differences in feature values between consecutive timestamps are calculated for each group:
  - `CONTROL`
  - `WIDE_BREAST`

### 3. **T-Test**
- A **t-test** is performed for each feature to compare the differences in feature values between the two groups across timestamps.
- The t-test outputs p-values indicating whether the feature differences are statistically significant.

### 4. **FDR Correction**
- Multiple features are tested, so **FDR correction** is applied to adjust the p-values and control for the false discovery rate.
- The **Benjamini-Hochberg** procedure is used to correct p-values and identify features that reject the null hypothesis after correction.

### 5. **Permutation Testing (Future Work)**
- **Permutation testing** involves generating random permutations of the dataset and performing t-tests and FDR correction on each permutation.
- This step helps verify the robustness of the results by comparing them across different random groupings of the dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Poultry-WB.git
   cd Poultry-WB
