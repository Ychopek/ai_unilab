import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv")
print(df.head())
print(df.describe())
df.info()

pd.set_option("display.max_columns", None)

print("Before:")
print(df["LotFrontage"].isnull().sum())



for column in df.select_dtypes(include=['float64', 'int64']).columns:
     df[column] = df[column].replace("NA", pd.NA)
     df[column] = df[column].fillna(df[column].mean())

print("After:")
print(df["LotFrontage"].isnull().sum())


print(df["LotArea"].describe())
scaler = MinMaxScaler()
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column] = scaler.fit_transform(df[[column]])
print(df["LotArea"].describe())

for column in df.select_dtypes(include=['object', 'string']).columns:
        df= pd.get_dummies(df, columns=[column],
drop_first=True)

df.filter(like="Neighborhood").head(10)
print(df.columns)

