from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#source venv/bin/activate
#uvicorn main:app --reload
#ngrok http 8000
#.\ngrok.exe http 8000

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = AutoModelForSequenceClassification.from_pretrained("bert_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("bert_sentiment_model")

history = []

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    return ["Negative", "Notr", "Positive"][pred]

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "history": history})

@app.post("/predict")
def classify(request: Request, username: str = Form(...), text: str = Form(...)):
    label = predict(text)
    history.append({"user": username, "text": text, "prediction": label})
    return templates.TemplateResponse("index.html", {"request": request, "prediction": label, "history": history})