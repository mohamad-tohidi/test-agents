import os
from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    FinalAnswerTool,
    LogLevel,
    ToolCallingAgent,
)
import yaml
import json


load_dotenv()


model = InferenceClientModel(
    model_id="google/gemma-4-26B-A4B-it",
    token=os.getenv("HF_TOKEN"),
)


# with open("prompts.yaml", "r") as stream:
#     prompt_templates = yaml.safe_load(stream)

# final_answer = FinalAnswerTool()


# agent = CodeAgent(
#     # tools=[store_tool, get_tool],
#     model=model,
#     add_base_tools=True,
#     prompt_templates=prompt_templates,
#     verbosity_level=LogLevel.INFO,
# )


agent = ToolCallingAgent(add_base_tools=True, tools=[], model=model)


def main():
    print("Hello from test-agents!")
    # print(os.getenv("HF_TOKEN"))
    result = agent.run(
        """
    who is the president of Iran during the war of hurmuz?    
    """,
        return_full_result=True,
    )
    print(result)
    print(json.dumps(agent.visualize()))


if __name__ == "__main__":
    main()
