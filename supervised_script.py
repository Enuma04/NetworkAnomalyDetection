import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ------------------------------------------------
# Helper functions
# ------------------------------------------------

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def prepare_supervised_data(df):
    df = df.copy()
    df["y"] = (df["Label"] != "BENIGN").astype(int)
    X = df.drop(columns=["Label", "y"])
    y = df["y"]
    return X, y


# helper for later explanation
def get_top_feature_contributions(pipeline, x_row, top_n=5):

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    x_df = x_row.to_frame().T
    x_scaled = scaler.transform(x_df)[0]

    contributions = x_scaled * clf.coef_[0]

    contrib_df = pd.DataFrame({
        "feature": x_df.columns,
        "contribution": contributions
    })

    contrib_df = contrib_df.sort_values("contribution", ascending=False)

    return contrib_df.head(top_n)


# ------------------------------------------------
# 1. Load multi-day training data
# ------------------------------------------------

train_files = [
    "ddos_csv/Monday-WorkingHours.pcap_ISCX.csv",
    "ddos_csv/Tuesday-WorkingHours.pcap_ISCX.csv",
    "ddos_csv/Wednesday-workingHours.pcap_ISCX.csv"
]

train_dfs = [load_and_clean(file) for file in train_files]

train_df = pd.concat(train_dfs, ignore_index=True)

print("Training label distribution:")
print(train_df["Label"].value_counts())


X_train, y_train = prepare_supervised_data(train_df)


# ------------------------------------------------
# 2. Train supervised model
# ------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
])

model.fit(X_train, y_train)


# ------------------------------------------------
# 3. Save trained model for deployment
# ------------------------------------------------

joblib.dump(model, "network_detector.pkl")

# also save feature order
joblib.dump(list(X_train.columns), "feature_columns.pkl")

print("\nModel saved as network_detector.pkl")
print("Feature column order saved as feature_columns.pkl")


# ------------------------------------------------
# 4. Evaluate on future traffic (drift testing)
# ------------------------------------------------

test_files = [
    "ddos_csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "ddos_csv/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "ddos_csv/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "ddos_csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "ddos_csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

for test_file in test_files:

    print("\n" + "=" * 70)
    print(f"Testing on: {test_file}")

    test_df = load_and_clean(test_file)

    print(test_df["Label"].value_counts())

    X_test, y_test = prepare_supervised_data(test_df)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    print("\nClassification report:\n",
          classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\nROC AUC:", roc_auc_score(y_test, y_prob))