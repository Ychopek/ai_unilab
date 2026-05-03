import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("train.csv")

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)




print("РЕГРЕССИЯ")

# предсказание SalePrice
X_reg = df.drop("SalePrice", axis=1)
y_reg = df["SalePrice"]

# Разделение  обучение/тест
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

model_reg = LinearRegression()
model_reg.fit(X_train_r, y_train_r)

y_pred_r = model_reg.predict(X_test_r)

print(f"Среднеквадратичная ошибка: {mean_squared_error(y_test_r, y_pred_r):.2f}")
print(f"Коэффициент детерминации: {r2_score(y_test_r, y_pred_r):.4f}")

print("\nКЛАССИФИКАЦИЯ")

# 1 — дорогой дом, 0 — дешевый
median_price = df["SalePrice"].median()
df["IsExpensive"] = (df["SalePrice"] > median_price).astype(int)

X_clf = df.drop(["SalePrice", "IsExpensive"], axis=1)
y_clf = df["IsExpensive"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

model_clf = LogisticRegression(max_iter=1000)
model_clf.fit(X_train_c, y_train_c)

y_pred_c = model_clf.predict(X_test_c)

print(f"Точность: {accuracy_score(y_test_c, y_pred_c):.4f}")
print("\nотчет:")
print(classification_report(y_test_c, y_pred_c))