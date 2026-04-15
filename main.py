import os
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    FinalAnswerTool,
    UserInputTool,
    VisitWebpageTool,
)
import yaml

load_dotenv()

model = InferenceClientModel(
    model_id="google/gemma-4-31B-it",
    token=os.getenv("HF_TOKEN"),
)


with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

final_answer = FinalAnswerTool()


agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        UserInputTool(),
        VisitWebpageTool(),
        final_answer,
    ],
    # tools=[],
    model=model,
    prompt_templates=prompt_templates,
)


def main():
    print("Hello from test-agents!")
    # print(os.getenv("HF_TOKEN"))
    result = agent.run(
        "بلعم باعورا که بود؟ آیه دقیقی که در قرآن دربارش اومده رو هم بنویس"
    )
    print(result)


if __name__ == "__main__":
    main()
