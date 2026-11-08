import os, sys, warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.utils.pydantic")

load_dotenv()

from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter

# * __ Model _________________________________________________
def make_model(slug="nvidia/nemotron-3.5-super-vl:free"):
  return ChatOpenRouter(model=slug, max_tokens=2048, temperature=0.3)


calls = {"n": 0}

@tool
def flaky_weather(city:str) -> str:
  """Get weather for a city. Fails the first 2 times."""
  calls["n"] += 1
  print(f"[TOOL] attempt #{calls['n']}")

  if calls["n"] < 3:
    raise ConnectionError("Weather API down")

  return f"{city}: 22C, Sunny"

@tool
def always_fails(city: str) -> str:
  """Get weather for a city. Always fails."""
  raise ValueError("boom")




