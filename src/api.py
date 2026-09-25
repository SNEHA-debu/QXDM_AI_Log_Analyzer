
from fastapi import FastAPI
from pydantic import BaseModel

from .inference import analyze_ue
from .genai import generate_ai_explanation


app = FastAPI(
    title="AI-Powered QXDM Log Analyzer",
    description=(
        "ML, NLP and GenAI analysis of "
        "synthetic QXDM-style LTE/5G modem logs"
    ),
    version="1.0.0"
)


class LogRequest(BaseModel):
    log_data: str


@app.get("/")
def home():
    return {
        "message":
        "AI-Powered QXDM Log Analyzer API is running"
    }


@app.post("/analyze")
def analyze_log(request: LogRequest):

    # ML + NLP inference
    analysis_results = analyze_ue(
        request.log_data
    )

    if not analysis_results:
        return {
            "status": "error",
            "message": "No valid log records found"
        }

    final_results = []

    # GenAI explanation for each UE
    for analysis in analysis_results:

        ai_explanation = generate_ai_explanation(
            analysis
        )

        final_results.append({
            "analysis": analysis,
            "ai_explanation": ai_explanation
        })

    return {
        "status": "success",
        "results": final_results
    }
