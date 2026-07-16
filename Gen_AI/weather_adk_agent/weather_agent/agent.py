import os
from dotenv import load_dotenv
import requests
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()



def get_weather(lat: float, lng: float) -> dict:
    """Get the weather information for a given latitude and longitude of a city.

    Returns the current weather information and hourly projection for next 5 hours for the specified city's latitude and longitude.
    Use this tool whenever the user asks for the weather information of a city. Provide the latitude and longitude of the city as input parameters.

    Args:
        lat (float): Latitude of the city.
        lng (float): Longitude of the city.

    Returns:
        dict: A dictionary containing the current weather information and hourly projection for next 5 hours.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lng}"
            f"&current_weather=true"
            f"&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,windspeed_10m,rain"
        )
        response = requests.get(url)
        data = response.json()
        current_weather = data.get("current_weather", {})
        hourly_data = data.get("hourly", {})

        if not current_weather:
            return {"error": "Weather data not available for the provided coordinates/city"}

        result = {
            "current_weather": current_weather,
            "next_5_hours": [
                {
                    "time": hourly_data["time"][i],
                    "temperature_2m": hourly_data["temperature_2m"][i],
                    "apparent_temperature": hourly_data["apparent_temperature"][i],
                    "relative_humidity_2m": hourly_data["relative_humidity_2m"][i],
                    "windspeed_10m": hourly_data["windspeed_10m"][i],
                    "rain": hourly_data["rain"][i]
                }
                for i in range(5)
            ]
        }

        return result
    except Exception as e:
        return {"message": "An error occurred while fetching weather data", "error": str(e)}

SYSTEM_PROMPT = """You are a helpful assistant that can answer questions about the weather in different cities.
If the user asks about the weather in a specific city, you should use relevant tools to get the information. 
Anything not related to weather information should be answered based on your own knowledge.
"""


root_agent = Agent(
    name="weather_agent",
    model=LiteLlm(model="groq/llama-3.1-8b-instant"),
    description="Answers the user query related to weather",
    instruction=SYSTEM_PROMPT,
    tools=[get_weather]
)
