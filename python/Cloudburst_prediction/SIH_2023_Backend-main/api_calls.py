import requests
def weather_data():
    api_url = "https://api.open-meteo.com/v1/forecast?latitude=13.0878&longitude=80.2785&hourly=precipitation,cloudcover_mid,windspeed_180m,winddirection_180m,temperature_180m&current_weather=true"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            weather_data = response.json()
            return weather_data
        else:
            return None    
    except Exception as e:
        return None  
    

data=weather_data()
print(data)
