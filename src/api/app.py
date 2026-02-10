from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import datetime

app = FastAPI(title="Ticket Classifier API", version="1.0")

# Load model once when API starts
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "ticket_classifier.joblib"
log_path = project_root / "logs" / "api_requests.log"

model = joblib.load(model_path)


class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    category: str
    input_text: str


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Ticket Classifier API is running"}


@app.post("/predict", response_model=TicketResponse)
def predict_ticket(request: TicketRequest):
    prediction = model.predict([request.text])[0]

    # log request
    timestamp = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | INPUT={request.text} | PRED={prediction}\n")

    return {
        "category": prediction,
        "input_text": request.text
    }
