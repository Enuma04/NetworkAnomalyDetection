import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def prepare_anomaly_test_data(df):
    df = df.copy()
    y = (df["Label"] != "BENIGN").astype(int)
    X = df.drop(columns=["Label"])
    return X, y


# ----------------------------
# 1) Load multi-day BENIGN-only training data
# ----------------------------
train_files = [
    "ddos_csv/Monday-WorkingHours.pcap_ISCX.csv",
    "ddos_csv/Tuesday-WorkingHours.pcap_ISCX.csv",
    "ddos_csv/Wednesday-workingHours.pcap_ISCX.csv"
]

train_dfs = [load_and_clean(file) for file in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)

train_benign = train_df[train_df["Label"] == "BENIGN"].copy()
X_train = train_benign.drop(columns=["Label"])

print("Benign-only training shape:", X_train.shape)

# ----------------------------
# 2) Train anomaly detector
# ----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("iso", IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train)

# ----------------------------
# 3) Test on future files
# ----------------------------
test_files = [
    "ddos_csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "ddos_csv/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "ddos_csv/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "ddos_csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "ddos_csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

scaler = model.named_steps["scaler"]
iso = model.named_steps["iso"]

for test_file in test_files:
    print("\n" + "=" * 70)
    print(f"Testing on: {test_file}")

    test_df = load_and_clean(test_file)
    print(test_df["Label"].value_counts())

    X_test, y_test = prepare_anomaly_test_data(test_df)

    X_test_scaled = scaler.transform(X_test)

    pred = iso.predict(X_test_scaled)          # 1 = normal, -1 = anomaly
    y_pred = (pred == -1).astype(int)          # convert to 1 = attack/anomaly
    scores = -iso.score_samples(X_test_scaled) # higher = more anomalous

    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("\nROC AUC:", roc_auc_score(y_test, scores))