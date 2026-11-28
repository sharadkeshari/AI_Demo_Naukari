from crewai import Agent, Task, Crew
from crewai.tools import tool

@tool("Weather Tool")
def get_weather(city: str) -> str:
    return f"Weather in {city} is 25°C"

weather_agent = Agent(
    role="Weather Expert",
    goal="Provide weather information",
    backstory="Weather specialist",
    verbose=True
)

task = Task(
    description="What is the weather in Delhi?",
    agent=weather_agent,
    expected_output="Current weather"
)

crew = Crew(
    agents=[weather_agent],
    tasks=[task]
)

result = crew.kickoff()
print(result)