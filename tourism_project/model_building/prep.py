# Cleans the raw data and writes the train/test splits to the working directory.
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# The first column is just the old index and CustomerID is a unique key,
# neither carries any signal for the model.
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Gender has a spelling variant ("Fe Male") that should be "Female"
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# strip stray whitespace in the text columns, then drop repeated rows
text_cols = df.select_dtypes(include="object").columns
df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())
before = len(df)
df = df.drop_duplicates()
print(f"Duplicate rows removed: {before - len(df)}")

# Missing values are filled inside the model pipeline (train.py), after the
# split, so the test set never influences the imputation values.
print("Missing values left:", int(df.isna().sum().sum()))

X = df.drop(columns=[TARGET])
y = df[TARGET]

# stratify because only ~19% of customers bought the package
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train:", Xtrain.shape, "| Test:", Xtest.shape)
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
