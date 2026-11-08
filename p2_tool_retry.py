

from base import *
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

agents = create_agent(
  make_model(),
  tools=[flaky_weather],
  middleware=[ToolRetryMiddleware(max_retries=3, initial_delay=1.0, backoff_factor=2.0)]
)

res = agents.invoke({"messages": [{
  "role": "user", 
  "content": "Weather in Dhaka"
}]})

print(res["messages"][-1].content)