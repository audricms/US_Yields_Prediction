# U.S. Yields Prediction

[![ENSAE](https://img.shields.io/badge/ENSAE%20Paris-2025--2026-blue)](https://www.ensae.fr/)
[![Course](https://img.shields.io/badge/Course-Machine%20Learning%20for%20Portfolio%20Management-green)]()

## 🎓 Academic Context

This project was conducted as part of the **Machine Learning for Portfolio Management** course taught by **Sylvain Champonnois** at **ENSAE Paris** during the first semester of the 2025-2026 academic year.

The objective was to compare linear models against complex non-linear models to determine if significant non-linearities in financial variables allow complex architectures to outperform simple linear baselines in forecasting the U.S. yield curve.

## 📌 Project Overview

This repository hosts a comprehensive empirical study on forecasting the **direction** of U.S. Treasury yields.

The project is designed as a **single, self-contained Jupyter Notebook**. It encompasses the entire research pipeline—from data ingestion via the FED API to model training, rigorous rolling-window evaluation, and statistical testing.

**Research Goal:** To predict whether weekly U.S. government bond yields will increase ($Y=1$) or decrease ($Y=0$) over a 20-year period, comparing the efficacy of classical econometric approaches versus modern machine learning techniques.

## 🚀 Key Findings

Our analysis of 171 models trained on 15-year rolling windows yielded the following critical insights:

* **Methodological Pivot:** Initial attempts at **daily regression** using penalized linear models (Ridge, Lasso, Elastic Net) failed to identify predictive relationships (OOS $R^2 \approx 0$). PCA did not improve results. Consequently, the project successfully pivoted to a **weekly binary classification framework**, which significantly improved the signal-to-noise ratio.
* **Short-End Predictability:** Predictive power is heavily concentrated at the **short end of the yield curve (< 1 year)**. Models predicting these maturities statistically outperformed the "majority class" benchmark (verified via McNemar's tests).
* **Best Model Performance:**
    * **Logistic Regression** was the most consistent performer, achieving up to **71% out-of-sample accuracy** for the 3-month yield.
    * **Random Forest** offered marginal improvements in some cases, capturing non-linearities, but at the cost of interpretability.
    * **Complexity Trap:** Complex models like **XGBoost** and **LSTM** suffered from massive overfitting, predicting mostly zeros and failing to generalize.
* **Feature Importance Insights:**
    * For most maturities, the best results came from datasets containing **macroeconomic and financial variables** (Mutual Information $> 0.035$).
    * **The 1-Month Exception:** The 1-month yield was best predicted (**65% accuracy**) using *only* functions of past yields (Mutual Information $> 0.05$). This highlights a strong, unique trend-following and mean-reverting behavior at the very short end of the curve.
* **Long-End Efficiency:** For long-term yields (> 2 years), results were disappointing. Models rarely outperformed the naive benchmark, suggesting these markets are more efficient and less dependent on the macroeconomic variables used.

## 🛠️ Methodology & Models

The notebook implements the following end-to-end workflow:

1.  **Data Source:** Automated fetching of macroeconomic and financial data via the **FRED API**.
2.  **Feature Engineering:**
    * Stationarity transformations (differencing/log-differencing).
    * Endogenous feature creation: Lags, rolling means, quantiles.
    * **Feature Selection:** We created multiple datasets based on **Mutual Information (MI)** thresholds ($>0.03, >0.035, >0.04, >0.05$) to test if models could handle high-dimensional noise versus pre-filtered data.
3.  **Models Evaluated:**
    * **Classifiers:** Logistic Regression (L2), Random Forest, XGBoost.
    * **Deep Learning:** Long Short-Term Memory (LSTM) Network.
4.  **Evaluation:**
    * Rigorous **15-year rolling-window** backtesting (predicting the subsequent 4 weeks).
    * Statistical significance verification using **McNemar's Test** against a "Majority Class Classifier" benchmark.

## 🔭 Future Extensions

Potential avenues for further research identified in this project include:
* **Ensemble Learning:** Aggregating the signals from our trained models to improve stability and accuracy.
* **Time-Varying Feature Importance:** Analyzing how feature importance evolves over the rolling periods to better understand regime changes.
* **Trading Strategy Implementation:** Utilizing the strong signal on the short end of the curve (maturity < 1 year) to backtest a concrete trading strategy using bond price data.

## 📦 Usage & Environment

**This project is self-contained.** You do not need to manually install a requirements file before running the notebook.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/audricms/US_Yields_Prediction.git](https://github.com/audricms/US_Yields_Prediction.git)
    ```
2.  **Open the notebook:**
    Launch Jupyter Lab or Jupyter Notebook and open `US_yield_prediction.ipynb`.
3.  **Run the first cell:**
    The first code cell contains an **environment setup script**. It will automatically check your active Python environment for required libraries (like `torch`, `xgboost`, `fredapi`) and install any that are missing.

### Dependencies Used
The analysis relies on the following core libraries:
* `numpy`, `pandas` (Data Manipulation)
* `matplotlib`, `seaborn` (Visualization)
* `scikit-learn` (Machine Learning)
* `xgboost` (Gradient Boosting)
* `torch` (Deep Learning)
* `statsmodels` (Econometrics)
* `fredapi` (Data Source)

## 📜 License

This project is available under the MIT License.
