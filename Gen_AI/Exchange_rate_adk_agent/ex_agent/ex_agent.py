
# Loading libraries
import os
from dotenv import load_dotenv
import requests
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    # Geting the current exchange rate between two currencies, e.g. USD to EUR.
    base_currency = base_currency.upper().strip()
    target_currency = target_currency.upper().strip()

    try:
        url = f"https://api.frankfurter.dev/v2/rate/{base_currency}/{target_currency}"
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200 or "rate" not in data:
            return {"error": data.get("message", "Could not fetch exchange rate")}

        return {
            "base_currency": base_currency,
            "target_currency": target_currency,
            "rate": data["rate"],
            "date": data.get("date")
        }
    except Exception as e:
        return {"error": str(e)}


def get_historical_exchange_rate(base_currency: str, target_currency: str, date: str) -> dict:
    # Getting the exchange rate between two currencies on a past date, format YYYY-MM-DD.
    base_currency = base_currency.upper().strip()
    target_currency = target_currency.upper().strip()

    try:
        url = f"https://api.frankfurter.dev/v2/rate/{base_currency}/{target_currency}?date={date}"
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200 or "rate" not in data:
            return {"error": data.get("message", "Could not fetch exchange rate")}

        return {
            "base_currency": base_currency,
            "target_currency": target_currency,
            "rate": data["rate"],
            "date": data.get("date")
        }
    except Exception as e:
        return {"error": str(e)}


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about forex/currency exchange rates.
Use get_exchange_rate for current rates, and get_historical_exchange_rate for a specific past date.
Answer anything unrelated to forex using your own knowledge.
"""

root_agent = Agent(
    name="forex_agent",
    model=LiteLlm(model="groq/llama-3.1-8b-instant"),
    description="Answers the user query related to forex/currency exchange rates",
    instruction=SYSTEM_PROMPT,
    tools=[get_exchange_rate, get_historical_exchange_rate]
)
