import asyncio
from agent_framework import Agent, MCPStdioTool, InMemoryHistoryProvider
from agent_framework.openai import OpenAIChatClient
import os
from dotenv import load_dotenv
load_dotenv()
async def main():

    weather_tool = MCPStdioTool(
        name="weathermcptools",
        command="python",
        args=["weather_server_mcp.py"]
    )

    model_client = OpenAIChatClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    history = InMemoryHistoryProvider(load_messages=True)

    audit_store = InMemoryHistoryProvider(
        "audit",
        load_messages=False,
        store_context_messages=True,  # include context added by other providers
    )

    agent = Agent(
        name="WeatherAgentClient",
        instructions="""
        You are a helpful assistant that uses tools to get the current weather for a location.
        Use MCP tools for weather and forecast questions.
        """,
        client=model_client,
        tools= [weather_tool],
        context_providers= [history, audit_store]
    )

    while True:
        q = input("You: ")

        if q.lower() == "exit":
            break

        result = await agent.run(q)

        print("AI:", result)

if __name__ == "__main__":
    asyncio.run(main())