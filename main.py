import argparse
import threading

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
    WebSearchTool,
    VisitWebpageTool,
)
from smolagents.monitoring import LogLevel


append_answer_lock = threading.Lock()

custom_role_conversions = {
    "tool-call": "assistant",
    "tool-response": "user",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        type=str,
        help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'",
    )

    return parser.parse_args()


def build_model():
    model_params = {
        "model_id": "openai/gemma4:e2b-it-q4_K_M",
        "custom_role_conversions": custom_role_conversions,
        "api_base": "http://0.0.0.0:11434/v1",
        "api_key": "Lanat bar omar",
        "max_completion_tokens": 8192,
    }

    return LiteLLMModel(**model_params)


def create_agent():
    model = build_model()

    web_tools = [WebSearchTool(), VisitWebpageTool()]

    deep_researcher = ToolCallingAgent(
        model=model,
        tools=web_tools,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="deep_researcher",
        description=(
            "A research specialist that can search the web and open webpages. "
            "Use this agent for any question that needs online research. "
            "Give it the full task in plain English, including timeframe and constraints."
        ),
        provide_run_summary=True,
    )

    # optional: make instructions more explicit
    deep_researcher.prompt_templates["managed_agent"]["task"] += """
You only have two tools:
1. WebSearchTool: use it to find relevant pages
2. VisitWebpageTool: use it to read pages you found

Work like a deep researcher:
- search first
- open the most relevant pages
- compare sources
- extract the answer carefully
- if the request is ambiguous, ask for clarification
"""

    manager_agent = CodeAgent(
        model=model,
        tools=[],
        max_steps=12,
        verbosity_level=LogLevel.DEBUG,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[deep_researcher],
    )

    return manager_agent


def main():
    args = parse_args()
    agent = create_agent()
    answer = agent.run(args.question)
    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
