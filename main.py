import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
        data_dict = data.dict(by_alias=True)
        # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
        # The data has names with hyphens and Python does not allow those as variable names.
        # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.

        # Fix for F841: We actually use data_dict now!
        # This replaces underscores with hyphens to match the training data format
        data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
        df = pd.DataFrame.from_dict(data)

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
        # Note: Not passing 'label' here because I am doing inference.
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
