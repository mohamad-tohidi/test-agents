import os
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

load_dotenv()

model = InferenceClientModel(
    model_id="google/gemma-3-4b-it",
    token=os.getenv("HF_TOKEN"),
)

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)


def main():
    print("Hello from test-agents!")
    result = agent.run(
        "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
    )
    print(result)


if __name__ == "__main__":
    main()
