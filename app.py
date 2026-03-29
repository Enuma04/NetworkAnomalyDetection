from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import requests

# ------------------------------------------------
# Load trained model and feature order
# ------------------------------------------------
model = joblib.load("network_detector.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ------------------------------------------------
# Ollama config
# ------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

app = FastAPI(title="On-Prem Network Detector API")


# ------------------------------------------------
# Request schema
# ------------------------------------------------
class FlowInput(BaseModel):
    features: dict


# ------------------------------------------------
# Helper: top feature contributions
# ------------------------------------------------
def get_top_feature_contributions(pipeline, x_df, top_n=5):
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    x_scaled = scaler.transform(x_df)[0]
    contributions = x_scaled * clf.coef_[0]

    contrib_df = pd.DataFrame({
        "feature": x_df.columns,
        "contribution": contributions
    })

    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).head(top_n)

    return [
        {
            "feature": row["feature"],
            "contribution": float(row["contribution"])
        }
        for _, row in contrib_df.iterrows()
    ]


# ------------------------------------------------
# Helper: local LLM explanation using Ollama
# ------------------------------------------------
def generate_llm_explanation(prediction, attack_probability, confidence, top_features):
    label_text = "attack-like / suspicious" if prediction == 1 else "benign"

    feature_lines = []
    for item in top_features:
        direction = "increases" if item["contribution"] > 0 else "decreases"
        feature_lines.append(
            f"- {item['feature']}: contribution {item['contribution']:.4f} ({direction} attack likelihood)"
        )

    prompt = f"""
    You are a cybersecurity incident assistant.

    Write a short analyst-facing incident summary in 2-4 sentences.
    Be precise and practical.
    Do not invent malware names or unsupported claims.
    If the prediction is benign, clearly say the flow appears normal and avoid language suggesting an attack occurred.
    If the prediction is suspicious, explain why and suggest a next step.

    Classification result:
    - Predicted class: {label_text}
    - Attack probability: {attack_probability:.6f}
    - Confidence: {confidence:.6f}

    Top contributing features:
    {chr(10).join(feature_lines)}

    Return plain text only.
    """.strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


# ------------------------------------------------
# Health endpoint
# ------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------
# Score endpoint
# ------------------------------------------------
@app.post("/score")
def score(flow: FlowInput):
    try:
        df = pd.DataFrame([flow.features])

        missing = [col for col in feature_columns if col not in df.columns]
        extra = [col for col in df.columns if col not in feature_columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required features",
                    "missing_count": len(missing),
                    "missing_features": missing[:10],
                    "extra_features": extra
                }
            )

        df = df[feature_columns]

        prediction = int(model.predict(df)[0])
        attack_probability = float(model.predict_proba(df)[0][1])
        confidence = attack_probability if prediction == 1 else 1 - attack_probability

        top_features = get_top_feature_contributions(model, df, top_n=5)

        explanation = generate_llm_explanation(
            prediction=prediction,
            attack_probability=attack_probability,
            confidence=confidence,
            top_features=top_features
        )

        return {
            "prediction": prediction,
            "attack_probability": attack_probability,
            "confidence": confidence,
            "top_features": top_features,
            "incident_summary": explanation
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Local LLM request failed: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------
# prints sample input for fastapi endpoints test
# ------------------------------------------------
 
# import pandas as pd
# import json

# df = pd.read_csv("ddos_csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# df.columns = df.columns.str.strip()
# df["Label"] = df["Label"].astype(str).str.strip()

# # remove label column because API expects only features
# sample = df.drop(columns=["Label"]).iloc[0].to_dict()

# print(json.dumps({"features": sample}, indent=2))


# {
#   "features": {
#     "Destination Port": 54865.0,
#     "Flow Duration": 3.0,
#     "Total Fwd Packets": 2.0,
#     "Total Backward Packets": 0.0,
#     "Total Length of Fwd Packets": 12.0,
#     "Total Length of Bwd Packets": 0.0,
#     "Fwd Packet Length Max": 6.0,
#     "Fwd Packet Length Min": 6.0,
#     "Fwd Packet Length Mean": 6.0,
#     "Fwd Packet Length Std": 0.0,
#     "Bwd Packet Length Max": 0.0,
#     "Bwd Packet Length Min": 0.0,
#     "Bwd Packet Length Mean": 0.0,
#     "Bwd Packet Length Std": 0.0,
#     "Flow Bytes/s": 4000000.0,
#     "Flow Packets/s": 666666.6667,
#     "Flow IAT Mean": 3.0,
#     "Flow IAT Std": 0.0,
#     "Flow IAT Max": 3.0,
#     "Flow IAT Min": 3.0,
#     "Fwd IAT Total": 3.0,
#     "Fwd IAT Mean": 3.0,
#     "Fwd IAT Std": 0.0,
#     "Fwd IAT Max": 3.0,
#     "Fwd IAT Min": 3.0,
#     "Bwd IAT Total": 0.0,
#     "Bwd IAT Mean": 0.0,
#     "Bwd IAT Std": 0.0,
#     "Bwd IAT Max": 0.0,
#     "Bwd IAT Min": 0.0,
#     "Fwd PSH Flags": 0.0,
#     "Bwd PSH Flags": 0.0,
#     "Fwd URG Flags": 0.0,
#     "Bwd URG Flags": 0.0,
#     "Fwd Header Length": 40.0,
#     "Bwd Header Length": 0.0,
#     "Fwd Packets/s": 666666.6667,
#     "Bwd Packets/s": 0.0,
#     "Min Packet Length": 6.0,
#     "Max Packet Length": 6.0,
#     "Packet Length Mean": 6.0,
#     "Packet Length Std": 0.0,
#     "Packet Length Variance": 0.0,
#     "FIN Flag Count": 0.0,
#     "SYN Flag Count": 0.0,
#     "RST Flag Count": 0.0,
#     "PSH Flag Count": 0.0,
#     "ACK Flag Count": 1.0,
#     "URG Flag Count": 0.0,
#     "CWE Flag Count": 0.0,
#     "ECE Flag Count": 0.0,
#     "Down/Up Ratio": 0.0,
#     "Average Packet Size": 9.0,
#     "Avg Fwd Segment Size": 6.0,
#     "Avg Bwd Segment Size": 0.0,
#     "Fwd Header Length.1": 40.0,
#     "Fwd Avg Bytes/Bulk": 0.0,
#     "Fwd Avg Packets/Bulk": 0.0,
#     "Fwd Avg Bulk Rate": 0.0,
#     "Bwd Avg Bytes/Bulk": 0.0,
#     "Bwd Avg Packets/Bulk": 0.0,
#     "Bwd Avg Bulk Rate": 0.0,
#     "Subflow Fwd Packets": 2.0,
#     "Subflow Fwd Bytes": 12.0,
#     "Subflow Bwd Packets": 0.0,
#     "Subflow Bwd Bytes": 0.0,
#     "Init_Win_bytes_forward": 33.0,
#     "Init_Win_bytes_backward": -1.0,
#     "act_data_pkt_fwd": 1.0,
#     "min_seg_size_forward": 20.0,
#     "Active Mean": 0.0,
#     "Active Std": 0.0,
#     "Active Max": 0.0,
#     "Active Min": 0.0,
#     "Idle Mean": 0.0,
#     "Idle Std": 0.0,
#     "Idle Max": 0.0,
#     "Idle Min": 0.0
#   }
# }