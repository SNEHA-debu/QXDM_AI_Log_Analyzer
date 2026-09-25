# AI-Powered QXDM Technical Log Analyzer

An end-to-end Machine Learning, NLP, Generative AI, and FastAPI
prototype for analyzing synthetic QXDM-style LTE/5G modem logs.

## Project Overview

This project demonstrates an AI-assisted telecom log analysis pipeline.

The system can:

- Parse synthetic QXDM-style LTE/5G modem logs
- Extract structured telecom events
- Perform UE-level feature engineering
- Classify event categories using NLP
- Detect anomalous UE behavior
- Predict a root-cause category using Machine Learning
- Generate grounded technical explanations using Generative AI
- Expose the complete inference pipeline through FastAPI

NOTE: This project uses synthetic QXDM-style data.
It does not use proprietary Qualcomm QXDM logs.

## Architecture

Synthetic QXDM-style Logs
        |
        v
Log Parser
        |
        v
Data Preprocessing
        |
        v
Feature Engineering
        |
        v
TF-IDF + NLP Classification
        |
        v
Random Forest
        |
        v
Root-Cause Category Prediction
        |
        v
Qwen2.5 GenAI
        |
        v
Grounded Technical Explanation
        |
        v
FastAPI
        |
        v
JSON Response

## Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Random Forest
- Isolation Forest
- Logistic Regression

### NLP
- TF-IDF
- N-grams
- Event-category classification

### Generative AI
- Qwen2.5-1.5B-Instruct
- Prompt Engineering
- Grounded Technical Explanation
- GenAI Guardrails

### API
- FastAPI
- Uvicorn
- Pydantic

## Project Structure

QXDM_AI_Log_Analyzer/
|-- README.md
|-- requirements.txt
|-- data/
|-- models/
|   |-- tfidf_vectorizer.joblib
|   |-- nlp_classifier.joblib
|   |-- root_cause_model.joblib
|   `-- feature_metadata.json
|-- src/
|   |-- parser.py
|   |-- inference.py
|   |-- genai.py
|   `-- api.py
|-- notebooks/
`-- tests/

## Feature Engineering

UE-level features include:

- Total events
- Total failure events
- Failure rate
- Authentication failure count
- Handover failure count
- Total retransmissions
- Average retransmission value
- NLP authentication-event count
- NLP connection-event count
- NLP mobility-event count
- NLP registration-event count
- NLP retransmission-event count

## NLP Pipeline

Event text is transformed using TF-IDF.

A Logistic Regression classifier categorizes events into:

- Authentication
- Connection
- Mobility
- Registration
- Retransmission

The NLP category counts are aggregated per UE and combined
with telecom-domain numerical features.

## Anomaly Detection

Isolation Forest is used to identify unusual UE behavior.

## Root-Cause Category Prediction

A Random Forest classifier predicts categories such as:

- Normal
- Authentication Failure
- Handover Failure
- High Retransmission

The output is a model-predicted category and not a confirmed
production-network root cause.

## Generative AI Layer

Qwen2.5-1.5B-Instruct converts structured ML results into
an engineer-friendly technical explanation.

Guardrails are used to:

- Use only supplied measurements
- Avoid unsupported causal claims
- Clearly identify ML outputs as predictions
- Avoid introducing unobserved network conditions

The GenAI model explains the ML result.
It does not independently determine the root cause.

## FastAPI

The application exposes the following endpoint:

POST /analyze

Pipeline:

Raw Log -> Parser -> NLP -> Feature Engineering ->
Random Forest -> Qwen GenAI -> JSON Response

The end-to-end pipeline was successfully verified using
an HTTP POST request returning HTTP 200.

## Installation

Install dependencies:

pip install -r requirements.txt

Run the API:

uvicorn src.api:app --host 127.0.0.1 --port 8000

GPU execution is recommended for the Qwen GenAI component.

## Model Evaluation

The ML and NLP classifiers achieved perfect classification
scores on the held-out portion of the controlled synthetic dataset.

These results should not be interpreted as real-world telecom
diagnostic accuracy because the dataset and labels were
synthetically generated using domain-informed rules.

## Future Improvements

- Evaluation using appropriately authorized real modem-log data
- More diverse synthetic failure scenarios
- Cross-validation
- Ablation testing
- Feature correlation analysis
- Transformer-based event classification
- Model monitoring
- Docker deployment
- Cloud deployment
- Visualization dashboard

## Disclaimer

This is an educational and portfolio prototype built using
synthetic QXDM-style LTE/5G modem logs.

It does not contain proprietary Qualcomm QXDM data and should
not be treated as a validated production diagnostic system.