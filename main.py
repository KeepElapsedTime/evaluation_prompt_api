#Import Package
import uvicorn, datetime , openai , re , gcloud , json , os, secrets, torch
from fastapi import FastAPI, HTTPException, Depends, status , Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# ENV------------- #

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

@app.get("/")
async def service_status(request: Request):
    time = datetime.datetime.now()
    return {"Server_Time": time,
            "Status": "Healthy"}

# Function - Eval
@app.get("/eval")
async def eval(
    prompt: str,
):    
    
    return classifier(prompt)

