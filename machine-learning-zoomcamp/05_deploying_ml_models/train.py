import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

C = 1.0
n_splits = 5
output_file = f"model_C={C}.bin"

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data preparation
# clean column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# clean categorical column data as well
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)

for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")


# Clean numerical variables
df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == "yes").astype(int)

# Setting up the validation framework
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

# delete to avoid using it as a feature by mistake - to avoid data leakage and overfitting
del df_train["churn"]
del df_val["churn"]
del df_test["churn"]

numerical = ["tenure", "monthlycharges", "totalcharges"]
categorical = list(set(df.columns.to_list()) - set(numerical + ["churn", "customerid"]))

# One-hot encoding
# Used to encode categorical variables into numerical variables
dv = DictVectorizer(sparse=False)

train_dicts = df_train[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)


# Training logistic regression with scikit-learn
def train(df_train, y_train, C=1.0):
    # C: regularization parameter
    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


print(f"Doing validaton with C={C}")

scores = []

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

fold = 1

for train_idx, val_idx in tqdm(kfold.split(df_full_train), total=n_splits):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f"AUC on fold {fold} is {auc:.3f}")

    fold += 1

print("Validation results:")
print(f"C={C} -> {np.mean(scores):.3f} +- {np.std(scores):.3f}")

# Train on full dataset
print("Training the final model")
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print(f"AUC on test set is {auc:.3f}")

# Save the model
with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)  # save the model and the vectorizer

print(f"Model saved to {output_file}")
