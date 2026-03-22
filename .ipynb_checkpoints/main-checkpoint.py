import pandas as pd

df = pd.read_csv("train.csv")
print(df.head())
print(df.describe())
print(df.info())

df["LotFrontage"].fillna(df["LotFrontage"].mean(), inplace=True)

print(df.head())