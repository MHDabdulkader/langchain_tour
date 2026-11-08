from base import *
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_error(request, handler):
  try:
    return handler(request)

  except Exception as e:
    print(f"[MW] tool error caught: {e}")

    return ToolMessage(
      content=f"Tool failed: {e}. Tell the user you couldn't fetch it.",
      tool_call_id=request.tool_call["id"]
    )

agents = create_agent(make_model(), tools=[always_fails], middleware=[handle_tool_error])
res = agents.invoke({"message": [{
  "role": "user",
  "content": "Weather in Dhaka"
}]})

print(res["message"][-1].content)
