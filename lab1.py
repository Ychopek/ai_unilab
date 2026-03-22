import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv")
print(df.head())
print(df.describe())
df.info()

pd.set_option("display.max_columns", None)

print("Before:")
print(df["LotFrontage"].isnull().sum())

df["LotFrontage"] = df["LotFrontage"].replace("NA", pd.NA)
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].mean())

print("After:")
print(df["LotFrontage"].isnull().sum())


print(df["LotArea"].describe())
scaler = MinMaxScaler()
df["LotArea"] = scaler.fit_transform(df[["LotArea"]])
print(df["LotArea"].describe())


df= pd.get_dummies(df, columns=['Neighborhood'],
drop_first=True)

df.filter(like="Neighborhood").head(10)
print(df.columns)

