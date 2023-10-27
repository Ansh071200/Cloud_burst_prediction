from typing import Union
# from pydantic import BaseModel
from fastapi import FastAPI
from model import rfc
from model import Lr
from api_calls import weather_data
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# Create a Pydantic model to define the expected data structure in the request body
# pp,pre,cloud_c,wind_speed,wind_direc,temperature

@app.get("/prediction")
def prediction():
    weather_d = weather_data()
    rfc_value= rfc(weather_d["hourly"]['precipitation'][0],weather_d["hourly"]['precipitation'][0],weather_d["hourly"]['cloudcover_mid'][0],weather_d["hourly"]['windspeed_180m'][0],weather_d["hourly"]['winddirection_180m'][0],weather_d["hourly"]['temperature_180m'][0])
    return {"data":rfc_value}


@app.get("/intensity")
def intensity():
    weather_d = weather_data()
    Lr_value = Lr(weather_d["hourly"]['precipitation'][0],weather_d["hourly"]['precipitation'][0],weather_d["hourly"]['cloudcover_mid'][0],weather_d["hourly"]['windspeed_180m'][0],weather_d["hourly"]['winddirection_180m'][0],weather_d["hourly"]['temperature_180m'][0])
    return {"intensity":Lr_value}

@app.get("/current_weather")
def current_weather():
    weather_d = weather_data()["current_weather"]
    return {"Current weather data" : weather_d}
