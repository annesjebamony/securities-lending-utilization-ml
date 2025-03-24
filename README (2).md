
# 📊 Securities Lending Utilization Analysis

This project analyzes short lending utilization patterns using the **FIS Astec Analytics Short Lending Data (SLD)** feed from Nasdaq Data Link. It aims to uncover trends, identify anomalies, and predict future utilization behavior to support Securities-Based Lending (SBL) firms.

---

## 🔍 Problem Statement

The goal of this project is to analyze how the borrowing demand for securities fluctuates over time. We aim to predict utilization percentages and detect anomalies using data science and machine learning methods, helping SBL firms optimize lending strategies, assess market risk, and price loans more effectively.

---

## 📦 Dataset Description

- `FIS-U1.csv`: Utilization data (e.g., AAPL, ISIN, utilization percent, dates)
- `FIS-R1.csv`: Rate data (loan rates per symbol, no date)
- Source: [Nasdaq Data Link - SLD Feed](https://data.nasdaq.com/databases/SLD)

---

## 🛠 Tools & Technologies

- **SQL (SQLite in Python)** for in-memory joins
- **Python (Pandas, NumPy, Seaborn, Matplotlib)** for preprocessing and visualization
- **Scikit-learn** for modeling (regression & ensemble)
- **Jupyter Notebook** for execution and documentation

---

## 📈 Exploratory Data Analysis

- Utilization Over Time (AAPL)
- Weekly Trends in Utilization
- Utilization Distribution (Histogram)
- 7-Day Rolling Average of Utilization
- Z-score Based Anomaly Detection

![Utilization Over Time](visuals/utilization_over_time.png)
![Weekly Boxplot](visuals/boxplot_by_weekday.png)
![Utilization Distribution](visuals/utilization_distribution.png)

---

## 🤖 Machine Learning Models

- Linear Regression
- Random Forest Regressor (Best: R² = 0.7157)
- Gradient Boosting
- K-Nearest Neighbors
- Voting Regressor (Ensemble)

![Voting Regressor](visuals/voting_regressor.png)

---

## 📊 Results Summary

| Model              | MSE     | R² Score |
|-------------------|---------|----------|
| Linear Regression | 0.4947  | 0.0179   |
| Random Forest     | 0.1432  | 0.7157   |
| Gradient Boosting | 0.2942  | 0.4160   |
| KNN               | 0.4101  | 0.1860   |
| Ensemble          | 0.2908  | 0.4227   |

---

## 🎓 Educational Impact

Developed as part of **DSC 478 - Programming Machine Learning Applications** at DePaul University, this project supports hands-on learning in data analysis, SQL integration, ML modeling, and financial risk analytics using real-world institutional datasets.

---

## 📄 Report & SQL

- [📘 Final Report](Final_Project_Report_Annes.pdf)
- [📄 SQL Preprocessing Logic](sql/join_and_feature_engineering.sql)

---

## 📌 Author

👩‍💻 Annes Jebasingh Jebamony  
Course: DSC 478 | Instructor: David Hubbard

---

