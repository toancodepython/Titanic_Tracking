import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import mlflow

num_cols = ['Age','SibSp','Parch','Fare']
cate_cols = ['Embarked', 'Sex']

df = pd.read_csv('./data/data.csv')

# drop 2 row null trong cot 'Embarked'
df.dropna(subset=['Embarked'],inplace = True)

#fill gia tri trong cot Age
df.Age = df.Age.fillna(df.Age.median())

encoder = LabelEncoder()
for col in cate_cols:
    if df[col].dtype == 'object':
        df[col] = encoder.fit_transform(df[col])
        df[col] = df[col].astype(int)
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Drop nhung cot khong can thiet
df = df.drop(['Name', 'Ticket', 'Cabin', 'httpPassengerId'], axis=1)
# chuẩn hóa giá trị
scaler = MinMaxScaler()
df_scaled  = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns= df.columns)
df_scaled.to_csv('./data/processed_data.csv')