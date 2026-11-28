virtual env setup
---  python -m venv .mcp-venv
--- activate the virtual env

#step1 : install all the liberaries



# command to run client
python client_langgraph_agent.py


# things to remembers
MCP connections are statful means once we connect server to client then a session is created and the calls happen that way
MCP servers are 
Question: "How is MCP different from a standard API call in terms of connection management?"
Answer: In a standard API call, the connection between client and server closes immediately after a response is received. In
MCP, the connection remains open (persistent session) for the entire duration of the agent's execution, allowing the agent to
make multiple tool calls within a single session without re-establishing the connection each time.
-----
-----

# step 1 : you need to install all the ibraries 
python -m pip install "mcp[cli]>=1.24,<2" requests openai python-dotenv langgraph langchain-openai langchain-core langchain-mcp-adapters typing-extensions

# step 2 : Define your MCP tool using FastMCP 
# step 3 : Test your tool before you connect it with your Agent 
(. Using MCP inspector we test our tool)
npx -y @modelcontextprotocol/inspector@v1-latest python weather_server.py


# step 4 : Define your agent and run it with belo wquery
Example queries:

```text
What is the current weather in Mumbai, India?
Give me a 3-day weather forecast for Delhi, India.
Compare the current weather in Mumbai and Delhi.
```
