import os.path

from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from typing import List
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from data_preprocessor import DataPreprocessor

app = FastAPI()
model = joblib.load(os.path.join("inferences", "model.pkl"))
data_preprocessor = DataPreprocessor.from_inferences()
templates = Jinja2Templates(directory="templates")


class InputData(BaseModel):
    features: List[float]


@app.post("/predict/json")
async def predict(data: InputData):
    """
    Predict based on JSON input.
    """
    try:
        features = pd.DataFrame([data.features])
        features_preprocessed = data_preprocessor.preprocess(features)

        prediction = model.predict(features_preprocessed)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """
    Page with form to upload file.
    """
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict/file", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Uploaded file handler.
    """
    try:
        features = pd.read_csv(file.file)
        features_preprocessed = data_preprocessor.preprocess(features)

        predictions = model.predict(features_preprocessed)

        return templates.TemplateResponse(
            "predictions_res.html",
            {"request": request, "predictions": predictions.tolist()}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "predictions_res.html",
            {"request": request, "error": str(e)}
        )


@app.get("/ping")
def ping():
    return "pong"
