import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
df = pd.read_csv(url)

df = df.dropna(subset=["price"])

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])  # Уникаємо NaN після перетворення
df['price_category'] = pd.cut(df['price'], bins=[0, 40, 70, float('inf')], labels=["low", "medium", "high"])

le = LabelEncoder()
categorical_columns = ["origin", "destination", "train_type", "train_class", "fare"]
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

X = df[["origin", "destination", "train_type", "train_class", "fare"]]
y = df["price_category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
