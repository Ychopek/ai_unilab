import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Загрузка

df = pd.read_csv("train.csv")
print("First rows:")
print(df.head())

# 2. Проверка

print("\nData info:")
df.info()

print("\nNaN before cleaning:")
print(df.isnull().sum().sum())


# 3. Очистка данных

cat_cols = df.select_dtypes(include=["object", "string"]).columns
print("\nCategorical columns:")
print(cat_cols)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df = df.copy()
print("\nAfter encoding:")
print(df.head())

df = df.fillna(df.mean(numeric_only=True))
print("\nNaN after cleaning:")
print(df.isnull().sum().sum())

# 4. Регрессия

print("\n Регрессия")

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

model_reg = LinearRegression()
model_reg.fit(X_train, y_train)



y_pred = model_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

print("\nSample predictions:")
print(pd.DataFrame({
    "Real": y_test.values[:5],
    "Predicted": y_pred[:5]
}))

# 5. Классификация
print("\n--- CLASSIFICATION ---")

df = df.assign(
    PriceCategory=(df["SalePrice"] > df["SalePrice"].median()).astype(int)
)

print("Class distribution:")
print(df["PriceCategory"].value_counts())

X = df.drop(["SalePrice", "PriceCategory"], axis=1)
y = df["PriceCategory"]
X = X.iloc[:, :50]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_clf = LogisticRegression(max_iter=2000, solver="liblinear")
model_clf.fit(X_train, y_train)

y_pred = model_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nSample classification:")
print(pd.DataFrame({
    "Real": y_test.values[:5],
    "Predicted": y_pred[:5]
}))

