import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("train.csv")

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 1 = дорогой дом 0 = дешёвый
median_price = df['SalePrice'].median()
y = (df['SalePrice'] >= median_price).astype(int)
X = df.drop(columns=['SalePrice'])


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"  Всего домов          : {len(df)}")
print(f"  Признаков            : {X.shape[1]}")
print(f"  Медиана цены         : ${median_price:,.0f}")
print(f"  Дорогих домов (y=1)  : {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Дешёвых домов (y=0)  : {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)

rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print(f"точность: ",rf.oob_score_)
print(f"ошибка  : ",1 - rf.oob_score_)

print(f"RFC ",classification_report(y_test, rf_pred))
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X_train, y_train)
ada_pred  = ada.predict(X_test)
ada_proba = ada.predict_proba(X_test)[:, 1]

print(f"ABC ",classification_report(y_test, ada.predict(X_test)))
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred  = gb.predict(X_test)
gb_proba = gb.predict_proba(X_test)[:, 1]

print(f"GBC ", classification_report(y_test, gb.predict(X_test)))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ROC-данные для каждой модели
models_roc = [
    (y_test, rf_proba,  'Random Forest',     '#2a9d8f'),
    (y_test, ada_proba, 'AdaBoost',          '#e9c46a'),
    (y_test, gb_proba,  'Gradient Boosting', '#57a744'),
]

plt.figure(figsize=(8, 7))

for y_true, y_score, name, color in models_roc:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2.4,
             label=f'{name}  (AUC = {roc_auc:.3f})')

# Диагональ — случайная модель
plt.plot([0, 1], [0, 1], '--', color='gray', lw=1.2,
         label='Случайная модель')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.03])
plt.xlabel('FPR (False Positive Rate)', fontsize=11)
plt.ylabel('TPR (True Positive Rate)', fontsize=11)
plt.title('ROC-кривые', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('lab4.png', dpi=150, bbox_inches='tight')
plt.show()