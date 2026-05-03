import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, classification_report,
                             roc_curve, auc)
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X_reg = df.drop(columns=['SalePrice'])
y_reg = df['SalePrice']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=5, random_state=42) # 5 ближе всего
model.fit(X_train_r, y_train_r)
y_pred = model.predict(X_test_r)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred)):,.0f}")
print(f"R²:   {r2_score(y_test_r, y_pred):.4f}")

y_cls = (df['SalePrice'] >= df['SalePrice'].median()).astype(int)
X_cls = df.drop(columns=['SalePrice'])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train_c, y_train_c)
y_proba_c = clf.predict_proba(X_test_c)

fpr, tpr, thresholds = roc_curve(y_test_c, y_proba_c[:, 1])
roc_auc = auc(fpr, tpr)
print(f"Accuracy: {accuracy_score(y_test_c, clf.predict(X_test_c)):.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")
print(classification_report(y_test_c, clf.predict(X_test_c)))


plt.plot(fpr, tpr, marker='o')
plt.ylim([0, 1.1]); plt.xlim([0, 1.1])
plt.ylabel('TPR'); plt.xlabel('FPR')
plt.title(f'ROC curve (AUC = {roc_auc:.3f})')
plt.show()