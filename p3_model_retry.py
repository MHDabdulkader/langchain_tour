from base import *
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

attempt = {"n": 0}

class FlakyModel(ChatOpenRouter):
  def _generate(self, *args, **kwargs):
    attempt["n"] += 1
    print(f"[MODEL] attempt #{attempt['n']}")

    if attempt['n'] < 3:
      raise ConnectionError("model down")
    return super()._generate(*args, **kwargs)

model = FlakyModel(
  model="nvidia/nemotron-3.5-super-vl:free",
  max_tokens = 1024,
  temperature=0.3
)



agent = create_agent(
  model,
  tools = [],
  middleware=[ModelRetryMiddleware(max_retries=3,  initial_delay=1.0, backoff_factor=2.0)]
)


res = agent.invoke({"messages": [{"role": "user", "content": "Say hi"}]})
print(res["messages"][-1].content)
