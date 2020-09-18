from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from main import predict
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Body(BaseModel):
    question: str
    img_base64: str


@app.get('/')
def root():
    return {'message':'hello world'}


@app.post("/vqa")
def encode_file(body: Body):
    ans = predict(body.question, body.img_base64)
    return {'answer': ans}
