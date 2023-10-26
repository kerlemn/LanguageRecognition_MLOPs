from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
from pathlib import Path
from pydantic import BaseModel
import re

app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl","rb") as f:
    model = pickle.load(f)

class Data(BaseModel):
    text: str

def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    pred = model.predict([text])
    return classes[pred[0]]

@app.get("/")
def read_root():
    print("Got it")
    return {"Hello": "World"}


@app.post("/predict")
def predict(text: Data):
    print("Got it")
    return {"class":predict_pipeline(text.text)}