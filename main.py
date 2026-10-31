import os
import sys
import warnings
import requests
import uuid


# Suppress Pydantic v1 warning on Python 3.14+
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.utils.pydantic")

from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals (avoids UnicodeEncodeError with emojis)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Create agent provide middleware, tools, etc.:
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime


# Load ENV
load_dotenv()

@dataclass
class Context:
  user_id: str

@dataclass
class ResponseFormat:
  summary: str
  temperature_celsius: str
  temperature_fahrenheit: str
  weather_description: str
  humidity: str
  wind_speed: str
  feels_like: str




@tool("get_weather", description="Return weather information for a given city", return_direct=False)
def get_weather(city: str):
    """Fetch current weather information for a city."""
    print(f"\n [TOOL] Fetching weather for {city}")

    try:
        response = requests.get(
            f"https://wttr.in/{city}?format=j1",
            headers={"User-Agent": "curl/8.0"},
            timeout=15,
        )
        response.raise_for_status()

        print("[TOOL] Weather data received successfully")
        data = response.json()
        current = data.get("current_condition", [{}])[0]
        return {
            "city": city,
            "temperature_C": current.get("temp_C"),
            "temperature_F": current.get("temp_F"),
            "weather_desc": current.get("weatherDesc", [{}])[0].get("value"),
            "humidity": current.get("humidity"),
            "wind_speed_kmph": current.get("windspeedKmph"),
            "feels_like_C": current.get("FeelsLikeC"),
        }
    except requests.RequestException as error:
        print(f"[TOOL ERROR] Could not fetch weather: {error}")
        return f"Unable to fetch weather for {city}"



@tool("locate_user", description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
  match runtime.context.user_id:
    case "ABC123":
      return "Vienna"
    case "DEF456":
      return "London"
    case _: 
      return "Unknown"



# =====================================================================
# Model Selection: OpenRouter or Gemini
# =====================================================================
# Option 1: OpenRouter (Requires max_tokens to prevent credit limit error)
from langchain_openrouter import ChatOpenRouter

print("[INIT] Initializing OpenRouter model...")
# model = ChatOpenRouter(
#     model="reflection/beam-1.0-501b-a23b",
#     max_tokens=2048,  # Crucial: prevents OpenRouter requesting 131,072 max tokens
#     temperature=0.7,
# )

model = ChatOpenRouter(
    model="reflection/beam-1.0-501b-a23b",
    max_tokens=2048,  # Crucial: prevents OpenRouter requesting 131,072 max tokens
    temperature=0.3,
)

checkpointer = InMemorySaver()


# Option 2: Gemini (Uncomment below to use Gemini instead)
# from langchain_google_genai import ChatGoogleGenerativeAI
# print("[INIT] Initializing Gemini model...")
# model = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash",  # Note: valid name is gemini-3.5-flash (not gemini-3-flash)
#     temperature=0.7,
# )

print("[INIT] Creating weather agent...")
agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt=(
        "You are a helpful and humorous weather assistant. "
        "Always crack jokes while answering. "
        "Use the get_weather tool whenever the user asks about weather. "
        "Explain the weather clearly and mention the city. "
        "If the user doesn't name a city, call locate_user first, then get_weather. "
    ),
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)


def run_agent(user_question: str, thread_id: str, user_id: str = "ABC123"):
    print("\n" + "=" * 50)
    print("[START] Weather assistant started!")
    print("=" * 50)

    user_input = {
        "messages": [
            {
                "role": "user",
                "content": user_question,
            }
        ]
    }

    config = {"configurable": {"thread_id": thread_id}}

    final_answer = None

    try:
        for chunk in agent.stream(
            user_input,
            config,
            context=Context(user_id=user_id),
            stream_mode="updates",
        ):
            for stage, data in chunk.items():
                if stage == "model":
                    # LangChain state uses "messages" (plural)
                    messages = data.get("messages", [])

                    if messages:
                        last_message = messages[-1]

                        if getattr(last_message, "tool_calls", None):
                            print("\n[AGENT] Model decided to call a tool...")
                        elif getattr(last_message, "text", None):
                            print("\n[AGENT] Model generated a response.")
                            final_answer = last_message.text
                        elif last_message.content:
                            print("\n[AGENT] Model generated a response.")
                            if isinstance(last_message.content, list):
                                text_parts = [
                                    item.get("text", "") if isinstance(item, dict) else str(item)
                                    for item in last_message.content
                                ]
                                final_answer = "".join(text_parts)
                            else:
                                final_answer = str(last_message.content)
                elif stage == "tools":
                    print("\n[AGENT] Tool execution completed.")
                else:
                    print(f"\n[AGENT] Current stage: {stage}")

        # Final answer is outside the chunk streaming loop
        state = agent.get_state(config)
        result = state.values.get("structured_response")
        
        print("\n" + "=" * 50)
        print("[FINAL ANSWER]")
        print("=" * 50)
        
        if result:
            print(f"Summary:     {result.summary}")
            print(f"Temp:        {result.temperature_celsius}°C / {result.temperature_fahrenheit}°F")
            print(f"Condition:   {result.weather_description}")
            print(f"Humidity:    {result.humidity}")
            print(f"Wind:        {result.wind_speed}")
            print(f"Feels like:  {result.feels_like}")
        elif final_answer:
            print(final_answer)
        else:
            print("No final response was generated.")

        print("\n[DONE] Agent execution finished!")

    except Exception as error:
        print("\n" + "=" * 50)
        print("[ERROR] Agent execution failed!")
        print("=" * 50)
        print(error)


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    
    while True:
        question = input("\nAsk about the weather (or type 'exit' to quit): ")

        if question.lower().strip() in ["exit", "quit"]:
            print("\nGoodbye! Stay weather-ready!")
            break

        if not question.strip():
            print("[WARNING] Please enter a question.")
            continue

        run_agent(question, thread_id)





