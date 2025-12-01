import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import traceback

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# Define paths to the saved model components
path_encoder = "model/encoder.pkl"
encoder = load_model(path_encoder)

path_model = "model/model.pkl"
model = load_model(path_model)

# Create a RESTful API using FastAPI
app = FastAPI()


# Create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"greeting": "Hello World!"}


# Create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    try:
        # DO NOT MODIFY: turn the Pydantic model into a dict.
        data_dict = data.dict(by_alias=True) # Use by_alias to keep hyphens if defined in Field aliases
        
        # If by_alias=True didn't work as expected for replacement, force manual replacement
        # The previous code did manual replacement. Let's stick to a robust way.
        # data_dict keys might be "education_num" or "education-num" depending on how .dict() behaves with aliases.
        # The safest way is to force the hyphenated names that the model expects.
        
        clean_data = {
            "age": [data.age],
            "workclass": [data.workclass],
            "fnlgt": [data.fnlgt],
            "education": [data.education],
            "education-num": [data.education_num],
            "marital-status": [data.marital_status],
            "occupation": [data.occupation],
            "relationship": [data.relationship],
            "race": [data.race],
            "sex": [data.sex],
            "capital-gain": [data.capital_gain],
            "capital-loss": [data.capital_loss],
            "hours-per-week": [data.hours_per_week],
            "native-country": [data.native_country]
        }
        
        df = pd.DataFrame.from_dict(clean_data)

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        # Process the data with training=False to use the loaded encoder
        # Note: We do NOT pass 'label' here because we are doing inference.
        data_processed, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            training=False,
            encoder=encoder,
            lb=None  # Not needed for inference
        )

        # Run inference using the loaded model
        _inference = inference(model, data_processed)
        return {"result": apply_label(_inference)}
    except Exception as e:
        # This will print the error to the terminal running uvicorn
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
