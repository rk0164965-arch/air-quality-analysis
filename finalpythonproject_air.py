import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\rk016\Downloads\pythonProjectDataset.csv")

print("Columns:", df.columns)
print(df.head())

# ---------------- CLEAN DATA ----------------
df = df.dropna()

df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')
df = df.dropna()

# ---------------- AUTO DETECT POLLUTANT COLUMN ----------------
pollutant_col = df.columns[7]

# ---------------- EDA ----------------

# 1. Distribution
plt.hist(df['pollutant_avg'])
plt.title("Pollutant Average Distribution")
plt.show()

# 2. Countplot
sns.countplot(x=pollutant_col, data=df)
plt.xticks(rotation=45)
plt.title("Pollutant Count")
plt.show()

# 3. Boxplot
sns.boxplot(x=pollutant_col, y='pollutant_avg', data=df)
plt.xticks(rotation=45)
plt.title("Pollutant vs Average")
plt.show()

# 4. Heatmap
sns.heatmap(df[['latitude','longitude','pollutant_avg']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ---------------- EXTRA ATTRACTIVE VISUALS ----------------

# Top 10 polluted cities
top_cities = df.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False).head(10)

top_cities.plot(kind='bar')
plt.title("Top 10 Polluted Cities")
plt.xticks(rotation=45)
plt.show()

# Pie chart - pollutant distribution
df[pollutant_col].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Pollutant Distribution")
plt.ylabel("")
plt.show()

# Trend graph
df['pollutant_avg'].head(50).plot()
plt.title("Pollution Trend")
plt.xlabel("Index")
plt.ylabel("Pollution Level")
plt.show()

# Extra info
print("Average Pollution Level:", df['pollutant_avg'].mean())
print("Maximum Pollution Level:", df['pollutant_avg'].max())

# ---------------- LINEAR REGRESSION ----------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[['latitude','longitude']]
y = df['pollutant_avg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

# Final Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
