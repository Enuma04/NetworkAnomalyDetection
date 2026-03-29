# On-Prem Network Anomaly Detection with LLM Explanations

This project implements a machine learning-based network intrusion detection system using the CIC-IDS2017 dataset. It evaluates model performance under traffic drift and integrates a locally hosted Large Language Model (LLM) to generate human-readable incident explanations.

---

## Features

- Supervised ML model (Logistic Regression) for anomaly detection
- Evaluation under **traffic drift** (train on earlier days, test on later days)
- FastAPI backend for real-time inference
- Feature contribution analysis for explainability
- LLM-based incident summaries using Ollama

---

## Project Overview

The system is trained on historical network traffic (Monday–Wednesday) and evaluated on future traffic (Thursday–Friday) to simulate real-world deployment conditions.

It outputs:
- Prediction (benign or attack)
- Attack probability
- Confidence score
- Top contributing features
- LLM-generated incident summary

---

## Setup

### 1. Install dependencies
pip install -r requirements.txt
2. Run the API
uvicorn app:app --reload
3. Open in browser
http://127.0.0.1:8000/docs
Example API Request

Use the /score endpoint:


{
  "features": {
    "Destination Port": 54865.0,
    "Flow Duration": 3.0,
    "Total Fwd Packets": 2.0,
    "Total Backward Packets": 0.0,
    "Total Length of Fwd Packets": 12.0,
    "Total Length of Bwd Packets": 0.0,
    "Fwd Packet Length Max": 6.0,
    "Fwd Packet Length Min": 6.0,
    "Fwd Packet Length Mean": 6.0,
    "Fwd Packet Length Std": 0.0,
    "Bwd Packet Length Max": 0.0,
    "Bwd Packet Length Min": 0.0,
    "Bwd Packet Length Mean": 0.0,
    "Bwd Packet Length Std": 0.0,
    "Flow Bytes/s": 4000000.0,
    "Flow Packets/s": 666666.6667,
    "Flow IAT Mean": 3.0,
    "Flow IAT Std": 0.0,
    "Flow IAT Max": 3.0,
    "Flow IAT Min": 3.0,
    "Fwd IAT Total": 3.0,
    "Fwd IAT Mean": 3.0,
    "Fwd IAT Std": 0.0,
    "Fwd IAT Max": 3.0,
    "Fwd IAT Min": 3.0,
    "Bwd IAT Total": 0.0,
    "Bwd IAT Mean": 0.0,
    "Bwd IAT Std": 0.0,
    "Bwd IAT Max": 0.0,
    "Bwd IAT Min": 0.0,
    "Fwd PSH Flags": 0.0,
    "Bwd PSH Flags": 0.0,
    "Fwd URG Flags": 0.0,
    "Bwd URG Flags": 0.0,
    "Fwd Header Length": 40.0,
    "Bwd Header Length": 0.0,
    "Fwd Packets/s": 666666.6667,
    "Bwd Packets/s": 0.0,
    "Min Packet Length": 6.0,
    "Max Packet Length": 6.0,
    "Packet Length Mean": 6.0,
    "Packet Length Std": 0.0,
    "Packet Length Variance": 0.0,
    "FIN Flag Count": 0.0,
    "SYN Flag Count": 0.0,
    "RST Flag Count": 0.0,
    "PSH Flag Count": 0.0,
    "ACK Flag Count": 1.0,
    "URG Flag Count": 0.0,
    "CWE Flag Count": 0.0,
    "ECE Flag Count": 0.0,
    "Down/Up Ratio": 0.0,
    "Average Packet Size": 9.0,
    "Avg Fwd Segment Size": 6.0,
    "Avg Bwd Segment Size": 0.0,
    "Fwd Header Length.1": 40.0,
    "Fwd Avg Bytes/Bulk": 0.0,
    "Fwd Avg Packets/Bulk": 0.0,
    "Fwd Avg Bulk Rate": 0.0,
    "Bwd Avg Bytes/Bulk": 0.0,
    "Bwd Avg Packets/Bulk": 0.0,
    "Bwd Avg Bulk Rate": 0.0,
    "Subflow Fwd Packets": 2.0,
    "Subflow Fwd Bytes": 12.0,
    "Subflow Bwd Packets": 0.0,
    "Subflow Bwd Bytes": 0.0,
    "Init_Win_bytes_forward": 33.0,
    "Init_Win_bytes_backward": -1.0,
    "act_data_pkt_fwd": 1.0,
    "min_seg_size_forward": 20.0,
    "Active Mean": 0.0,
    "Active Std": 0.0,
    "Active Max": 0.0,
    "Active Min": 0.0,
    "Idle Mean": 0.0,
    "Idle Std": 0.0,
    "Idle Max": 0.0,
    "Idle Min": 0.0
  }
}

 Example Response
{
  "prediction": 0,
  "attack_probability": 8.010406079757295e-256,
  "confidence": 1,
  "top_features": [
    {
      "feature": "Destination Port",
      "contribution": -502.8741098445608
    },
    {
      "feature": "Packet Length Mean",
      "contribution": 15.841711488517651
    },
    {
      "feature": "Average Packet Size",
      "contribution": -13.377245930177862
    },
    {
      "feature": "Flow Duration",
      "contribution": -11.240259495183249
    },
    {
      "feature": "Flow IAT Max",
      "contribution": 10.031875531786657
    }
  ],
  "incident_summary": "The incident suggests an increase in the flow duration and flow IAT Max values, which are indicators of potential network probing or scanning activities. Further investigation is recommended to determine the actual cause of the observed behavior."
}

--
Evaluation

The model was evaluated under traffic drift conditions:
Trained on: Monday–Wednesday traffic
Tested on: Thursday–Friday traffic

Results showed:

Strong performance on DDoS traffic
Poor generalization on web attacks, bot traffic, and infiltration
Significant impact of distribution shift (traffic drift)

--

This project uses the CIC-IDS2017 dataset.

Due to size limitations, the dataset is not included in this repository.

Download from:
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

Place CSV files in:

ddos_csv/

--
Notes
The model is pre-trained and included (.pkl files)
LLM explanations require a local Ollama instance
LLM Setup (Optional)

To enable explanations:

ollama run gemma:2b (ollama should be installed from website)
--
Repository

Full implementation available here:
https://github.com/Enuma04/NetworkAnomalyDetection