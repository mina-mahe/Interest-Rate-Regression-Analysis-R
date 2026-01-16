# Predicting Loan Interest Rates - Multi-Model R Analysis

## Project Overview
This project focuses on building a predictive framework to automate **interest rate determination** for loan applicants. Using **R**, I performed end-to-end data science tasks including sophisticated **feature engineering**, statistical validation of linear assumptions, and a performance comparison between **Linear Regression**, **Decision Trees**, and **Random Forests**.

## Executive Summary & Key Insights
* **Primary Rate Drivers:** **FICO Scores** and **Loan Term** (36 vs 60 months) were identified as the most critical predictors of interest rates.
* **Risk Sensitivity:** Every unit increase in **Credit Inquiries** over the last 6 months significantly pushes the interest rate higher, reflecting increased borrower risk.
* **Geographic Tiers:** Through **Target Encoding**, I identified high-risk geographic clusters (e.g., VT, HI, AK, MS) where interest rates are structurally higher.
* **Model Champion:** The **Random Forest** model achieved the highest predictive accuracy with the lowest **RMSE**, effectively capturing non-linear interactions.
* **Operational Efficiency:** By automating the rate-setting process, the model reduces human bias and speeds up the loan approval lifecycle.


## Technical R Highlights
* **Feature Engineering:** Calculated **FICO midpoints** and transformed **Employment Length** into a continuous numeric scale for better model sensitivity.
* **Dimensionality Reduction:** Consolidated 14+ loan purposes and 50 states into **ranked tiers** based on mean interest rate impact.
* **Statistical Validation:** Utilized **Variance Inflation Factor (VIF)** to detect and remove multicollinearity, ensuring a stable linear model.
* **Residual Analysis:** Performed rigorous testing of assumptions, including **Homoscedasticity** and **Normality of Errors**, via `ggplot2`.
* **Ensemble Learning:** Implemented a **Random Forest** with 500 trees and evaluated **Variable Importance** to rank business impact.


## Predictive Workflow & Comparison
1. **Data Cleaning:** Handled messy strings, percentage signs, and casted data types using **dplyr**.
2. **Multicollinearity Check:** Used the **car** library to prune the feature set based on VIF scores.
3. **Linear Regression Fit:** Refined a multivariate model using **Stepwise Selection** (p < 0.05).
4. **Decision Tree Analysis:** Developed an interpretable tree using **rpart** to visualize borrower segmentation.
5. **Random Forest Training:** Built a robust ensemble model to handle complex interactions between the variables.
6. **Performance Benchmarking:** Compared **Test RMSE** across all models to select the final algorithm.
