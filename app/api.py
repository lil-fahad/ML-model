from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

class PriceData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@app.get("/")
async def root():
    return {"app":"ML Trading API","status":"running"}

@app.get("/health")
async def health():
    return {"status":"ok","ts":datetime.now().isoformat()}

@app.post("/predict")
async def predict(data: List[PriceData]):
    return {"prediction":"BUY","confidence":0.75}

@app.post("/signals")
async def signals(data: List[PriceData]):
    p=data[-1].close
    return {"signal":"BUY","price":p,"sl":p-225,"tp":p+300}

@app.get("/models")
async def models():
    return {"models":["enhanced_model","hybrid_model"]}
