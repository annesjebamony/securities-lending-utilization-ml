import pandas as pd
import sqlite3

# Load CSVs from the specified file paths
u1 = pd.read_csv('/content/FIS-U1.csv')
r1 = pd.read_csv('/content/FIS_R1.csv')

# Check first few rows
print("Utilization Results (U1):")
print(u1.head())

print("\nRates (R1):")
print(r1.head())

"""Created SQL Tables In-Memory"""

# Connect to in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Convert to SQL tables
u1.to_sql('U1', conn, index=False, if_exists='replace')
r1.to_sql('R1', conn, index=False, if_exists='replace')

""" Preview Column Names (important for SQL joins)"""

print("U1 Columns:", u1.columns.tolist())
print("R1 Columns:", r1.columns.tolist())

query = '''
SELECT
    u.tradingsymbol,
    u.date,
    u.utilizationpercentunits,
    r.tradingsymbol AS r_symbol
FROM
    U1 u
JOIN
    R1 r
ON
    u.tradingsymbol = r.tradingsymbol
'''
joined_df = pd.read_sql_query(query, conn)
joined_df.head()

""" Exploratory Data Analysis (EDA)"""

# Basic info
joined_df.info()

# Summary statistics
joined_df.describe()

# Null values
print("Missing values:\n", joined_df.isnull().sum())

"""Summary Statistics"""

print(joined_df.describe())

"""Feature Engineering"""

df = joined_df.copy()

# Convert date to datetime if not already
df['date'] = pd.to_datetime(df['date'])

# Extract time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6

# Encode trading symbol
df['symbol_encoded'] = df['tradingsymbol'].astype('category').cat.codes

df.head()

"""Prepare Data for Modeling"""

from sklearn.model_selection import train_test_split

# Features and target
X = df[['year', 'month', 'day', 'dayofweek', 'symbol_encoded']]
y = df['utilizationpercentunits']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""4 Models + 1 Ensemble"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)
knn = KNeighborsRegressor()

# Fit models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Ensemble
ensemble = VotingRegressor(estimators=[
    ('lr', lr),
    ('rf', rf),
    ('gbr', gbr),
    ('knn', knn)
])
ensemble.fit(X_train, y_train)

"""Evaluate Models"""

models = {'Linear Regression': lr, 'Random Forest': rf, 'Gradient Boosting': gbr, 'KNN': knn, 'Ensemble': ensemble}

for name, model in models.items():
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MSE = {mse:.4f}, R² = {r2:.4f}")

"""Line Plot: Utilization Over Time (AAPL)"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='date', y='utilizationpercentunits')
plt.title('Utilization Over Time for AAPL', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Utilization (%)')
plt.grid(True)
plt.show()

"""Trend by Day of the Week"""

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='dayofweek', y='utilizationpercentunits')
plt.title('Utilization by Day of the Week', fontsize=14)
plt.xlabel('Day of Week (0=Monday)')
plt.ylabel('Utilization (%)')
plt.grid(True)
plt.show()

"""Distribution Plot"""

plt.figure(figsize=(10, 5))
sns.histplot(df['utilizationpercentunits'], kde=True, bins=30)
plt.title('Distribution of Utilization Percentages', fontsize=14)
plt.xlabel('Utilization (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

"""Rolling Average (7-Day Smooth)"""

df['7_day_avg'] = df['utilizationpercentunits'].rolling(window=7).mean()

plt.figure(figsize=(14, 6))
sns.lineplot(x='date', y='utilizationpercentunits', data=df, label='Actual', alpha=0.5)
sns.lineplot(x='date', y='7_day_avg', data=df, label='7-Day Rolling Avg', color='orange')
plt.title('Utilization & 7-Day Rolling Average', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Utilization (%)')
plt.legend()
plt.grid(True)
plt.show()

"""Anomaly Detection with Z-score

"""

from scipy.stats import zscore
df['zscore'] = zscore(df['utilizationpercentunits'])
anomalies = df[df['zscore'].abs() > 2]

plt.figure(figsize=(14,6))
sns.lineplot(x='date', y='utilizationpercentunits', data=df, label='Utilization')
plt.scatter(anomalies['date'], anomalies['utilizationpercentunits'], color='red', label='Anomalies')
plt.title('Utilization with Anomalies Highlighted (Z-score)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Utilization (%)')
plt.legend()
plt.grid(True)
plt.show()
