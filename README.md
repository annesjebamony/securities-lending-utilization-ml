
# Securities Lending Utilization Analysis

This project analyzes short lending utilization patterns using the **FIS Astec Analytics Short Lending Data (SLD)** feed from Nasdaq Data Link. It aims to uncover trends, identify anomalies, and predict future utilization behavior to support Securities-Based Lending (SBL) firms.

---

The goal of this project is to analyze how the borrowing demand for securities fluctuates over time. I aim to predict utilization percentages and detect anomalies using data science and machine learning methods, helping SBL firms optimize lending strategies, assess market risk, and price loans more effectively.

---

## Dataset Description

- `FIS-U1.csv`: Utilization data (e.g., AAPL, ISIN, utilization percent, dates)
- `FIS-R1.csv`: Rate data (loan rates per symbol, no date)
- Source: [Nasdaq Data Link - SLD Feed](https://data.nasdaq.com/databases/SLD)

---

## Tools & Technologies

- **SQL (SQLite in Python)** for in-memory joins
- **Python (Pandas, NumPy, Seaborn, Matplotlib)** for preprocessing and visualization
- **Scikit-learn** for modeling (regression & ensemble)
- **Jupyter Notebook** for execution and documentation

---

## Exploratory Data Analysis

- Utilization Over Time (AAPL)
- Weekly Trends in Utilization
- Utilization Distribution (Histogram)
- 7-Day Rolling Average of Utilization
- Z-score Based Anomaly Detection
---

## Machine Learning Models

- Linear Regression
- Random Forest Regressor (Best: R² = 0.7157)
- Gradient Boosting
- K-Nearest Neighbors
- Voting Regressor (Ensemble)

---

## Results Summary

| Model              | MSE     | R² Score |
|-------------------|---------|----------|
| Linear Regression | 0.4947  | 0.0179   |
| Random Forest     | 0.1432  | 0.7157   |
| Gradient Boosting | 0.2942  | 0.4160   |
| KNN               | 0.4101  | 0.1860   |
| Ensemble          | 0.2908  | 0.4227   |

---

---

## Key Insight & Business Value for SBL Firms

One of the most impactful findings in this project was the use of **Z-score-based anomaly detection** to shows rare but critical utilization spikes. While most utilization values remained under 1%, the model identified high-utilization anomalies, particularly in 2014 and 2016, where utilization surged beyond 8%.

### How SBL Firms Benefit:
- **Proactive Risk Management**: These spikes signal speculative trading or short squeezes. SBL firms can use this information to flag risky periods and adjust collateral or inventory strategies.
- **Dynamic Loan Pricing**: Insights from the utilization distribution help firms create tiered pricing models — adjusting interest rates for low vs. high utilization scenarios.
- **Inventory Forecasting**: 7-day rolling averages help firms prepare for periods of sustained borrowing, ensuring asset availability and preventing liquidity gaps.
- **Model-Based Strategy**: With a Random Forest R² score of 0.71, SBL companies can trust the model to support strategic forecasting of lending demand.

These insights elevate SBL operations from reactive to proactive, using data science to stay ahead of market shifts.

---

